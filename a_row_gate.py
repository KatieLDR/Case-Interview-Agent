#!/usr/bin/env python3
"""a_row_gate.py — Step 2 gate (REFACTOR_PLAN §4/§5, §5.1 table A).

Runs the SHARED resolver `backend.domain.matching.locate()` against the catalog A-rows
and checks each resolves as the catalog expects. Because all three arms now call the same
`locate()`, "identical KBMatch across arms" holds by construction; this harness validates
the substantive half of the gate — that `locate()` is CORRECT (and that the F-M1/F-M2
divergences converge).

LLM-backed, so it needs a real key. Run from the project root (where `backend/` lives):

    GEMINI_API_KEY=...  python3 a_row_gate.py

Determinism: classifier/matcher calls are pinned temperature=0 (llm.py). A row that flaps
across runs is flagged in §5 as inherently ambiguous (sharpen the classifier prompt).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Same as llm.py — pick up GEMINI_API_KEY from a .env at the project root, so running
# under Poetry needs no extra exports if the app's .env already has the key.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.getenv("GEMINI_API_KEY"):
    sys.exit("GEMINI_API_KEY not set — this gate calls the live classifier. "
             "Set it and re-run from the project root.")

from backend.domain import matching as m  # noqa: E402

WITHHELD = {"Financial Impact", "Risk & Governance"}

# (id, item the intent layer would pass to locate, expectation)
# expectation keys: level (exact), pillar (exact name or None), withheld (bool or None=any)
CASES = [
    ("A1", "change management",
     dict(level="none",    pillar=None,               withheld=False)),  # novel_not_in_kb
    ("A2", "IT budget",
     dict(level=("pillar", "concept"), pillar="Financial Impact", withheld=True)),  # revealed_withheld
    ("A3", "financial impact",
     dict(level=("pillar", "concept"), pillar="Financial Impact", withheld=True)),
    ("A4", "regulatory risk and the EU AI Act",
     dict(level=("pillar", "concept"), pillar="Risk & Governance", withheld=True)),
    ("A5", "data quality",
     dict(level="concept", pillar="Feasibility",      withheld=False)),  # SHOWN concept -> duplicate (F-M1)
    ("A6", "vendor lock-in",
     dict(level="none",    pillar=None,               withheld=False)),  # novel sub-point
    ("A10", "team skills to maintain this",
     dict(level="concept", pillar="Feasibility",     withheld=False)),  # existing concept -> duplicate
]


def ok(km, exp):
    lvl = exp["level"]
    if isinstance(lvl, tuple):
        if km.level not in lvl:
            return False
    elif km.level != lvl:
        return False
    if exp["pillar"] is not None and (km.pillar or "") != exp["pillar"]:
        return False
    if exp["withheld"] is not None and km.pillar_is_withheld != exp["withheld"]:
        return False
    return True


def main():
    # F-CASE gate-realism (2026-06-09): the intent layer passes the user's NATURAL casing
    # to locate(), not a curated lower-case string. So each A-row is now checked under
    # several casings (as-typed / lower / UPPER / Title) and only PASSES if they ALL resolve
    # to the SAME (level, pillar) AND meet the catalog expectation. This would have caught
    # "IT Budget" -> none while "IT budget" -> Financial Impact (the live smoke-test miss).
    print(f"{'row':4} {'item':34} {'level':8} {'pillar':18} {'withheld':8} {'match_type':26} result")
    print("-" * 110)
    passed = 0
    for rid, item, exp in CASES:
        variants = list(dict.fromkeys([item, item.lower(), item.upper(), item.title()]))
        results = [(v, m.locate(v)) for v in variants]
        good   = all(ok(km, exp) for _, km in results)
        agree  = (len({km.level for _, km in results}) == 1
                  and len({(km.pillar or "") for _, km in results}) == 1)
        row_ok = good and agree
        passed += row_ok
        km0 = results[0][1]
        note = ""
        if not row_ok:
            if not agree:
                note = "  CASE-SPLIT " + repr({v: (km.level, km.pillar) for v, km in results})
            else:
                note = "  expected " + repr(exp)
        print(f"{rid:4} {item[:33]:34} {km0.level:8} {str(km0.pillar)[:17]:18} "
              f"{str(km0.pillar_is_withheld):8} {str(km0.match_type)[:25]:26} "
              f"{'PASS' if row_ok else 'FAIL' + note}")
    print("-" * 110)
    print(f"{passed}/{len(CASES)} A-rows resolve as the catalog expects under ALL casings.")
    if passed != len(CASES):
        print("Gate NOT met — inspect the FAIL rows (CASE-SPLIT = locate() is case-sensitive; "
              "else prompt wording or threshold).")
        sys.exit(1)
    print("Step-2 matching gate MET (shared locate() correct + case-robust on A-rows). "
          "Identical-across-arms holds by construction once EXP/HITL are wired.")


if __name__ == "__main__":
    main()
