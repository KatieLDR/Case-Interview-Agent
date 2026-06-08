#!/usr/bin/env python3
"""step2_gate.py — Step-2 coverage BEYOND the A-rows (REFACTOR_PLAN §5 B/C/D).

The A-row gate (a_row_gate.py) tests locate() on adds. This tests the other Step-2
deliverables the A-rows never touch:
  - grounding.ground_pillar / ground_concept          (KB explanation lookup — §3.2)
  - matching.resolve_removal_target                   (deictic / positional / named — Fork A)
  - the filler / "else"-bug guard in locate()         (catalog D6 regression)

Most checks are DETERMINISTIC and make zero API calls (grounding, filler, deictic,
positional). Only the two NAMED-removal rows call the live classifier (via locate); they
are skipped if GEMINI_API_KEY is absent. Run from the project root:

    poetry run python step2_gate.py

This validates the SHARED resolver/lookup that all three arms will adopt; it is not a
test of the removal *machine* (challenge / confirm / nothing-to-remove guard) — that is
Step-4 handler behavior. Here we only assert that targets RESOLVE correctly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.domain import matching as m       # noqa: E402
from backend.domain import grounding as g       # noqa: E402

passed = total = 0


def check(label, cond, detail=""):
    global passed, total
    total += 1
    passed += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail and not cond else ""))


print("=== 1. GROUNDING (ground_pillar / ground_concept) — 0 API calls ===")
check("ground_pillar('Feasibility') non-empty (shown area)", bool(g.ground_pillar("Feasibility")))
check("ground_pillar('Financial Impact') non-empty (withheld, still groundable for reveal)",
      bool(g.ground_pillar("Financial Impact")))
check("ground_pillar('Risk & Governance') non-empty (withheld)", bool(g.ground_pillar("Risk & Governance")))
check("ground_concept('data_cleanliness_availability') non-empty", bool(g.ground_concept("data_cleanliness_availability")))
check("ground_pillar('Nonexistent Area') == '' (unknown -> empty)", g.ground_pillar("Nonexistent Area") == "")
check("ground_concept('not_a_real_id') == ''", g.ground_concept("not_a_real_id") == "")
swap_name = "Average number of steps walked per day by the IT team"
check("ground_pillar(swap concept) non-empty (swap fallback)", bool(g.ground_pillar(swap_name)))
check("no source-refs ([a]/[b]) leak into grounding output",
      not any(f"[{c}]" in g.ground_pillar("Feasibility") for c in "abcdefgh"))

print("\n=== 2. FILLER / 'else'-bug regression (catalog D6) — 0 API calls ===")
for t in ["what else?", "anything else?", "what else could we add?", "more",
          "something else", "and so on", "else"]:
    km = m.locate(t)
    check(f"locate({t!r}) -> none (never a pillar named 'else')", km.level == "none",
          detail=f"got level={km.level} pillar={km.pillar}")

print("\n=== 3. REMOVAL RESOLUTION — deictic / positional (Fork A) — 0 API calls ===")
focus = m.KBMatch(pillar="Feasibility", level="concept",
                  concept_id="data_cleanliness_availability",
                  match_type="surfaced_unreached_shown")
r = m.resolve_removal_target("remove this", last_discussed=focus, shown_bullets=[])
check("'remove this' + last_discussed=concept -> returns that focus (level=concept)",
      r.level == "concept" and r.concept_id == "data_cleanliness_availability",
      detail=f"got level={r.level} cid={r.concept_id}")

r = m.resolve_removal_target("remove it", last_discussed=None, shown_bullets=[])
check("'remove it' + no focus (BlackBox full-render) -> needs_disambiguation, not a silent guess",
      r.level == "none" and r.needs_disambiguation, detail=f"got level={r.level} disambig={r.needs_disambiguation}")

shown = ["GenAI vs. simpler automation", "Responsible AI principles", "Stakeholder buy-in"]
r = m.resolve_removal_target("remove the second point", last_discussed=None, shown_bullets=shown)
check("'remove the second point' + shown_bullets -> shown[1]",
      r.matched_text == "Responsible AI principles", detail=f"got matched_text={r.matched_text!r}")

r = m.resolve_removal_target("remove the third point", last_discussed=None, shown_bullets=shown)
check("'remove the third point' + shown_bullets -> shown[2]",
      r.matched_text == "Stakeholder buy-in", detail=f"got matched_text={r.matched_text!r}")

print("\n=== 4. REMOVAL RESOLUTION — named targets (uses locate -> LIVE classifier) ===")
if not os.getenv("GEMINI_API_KEY"):
    print("  [SKIP] GEMINI_API_KEY not set — named-removal rows need the live classifier.")
else:
    r = m.resolve_removal_target("Feasibility", last_discussed=None, shown_bullets=[])
    check("B1 'Feasibility' -> resolves to Feasibility, shown (handler then challenges)",
          (r.pillar == "Feasibility") and not r.pillar_is_withheld,
          detail=f"got pillar={r.pillar} withheld={r.pillar_is_withheld} level={r.level}")
    r = m.resolve_removal_target("Risk & Governance", last_discussed=None, shown_bullets=[])
    check("B5 'Risk & Governance' -> resolves, withheld (handler then -> nothing-to-remove)",
          (r.pillar == "Risk & Governance") and r.pillar_is_withheld,
          detail=f"got pillar={r.pillar} withheld={r.pillar_is_withheld} level={r.level}")

print("\n" + "-" * 70)
print(f"{passed}/{total} checks passed.")
sys.exit(0 if passed == total else 1)
