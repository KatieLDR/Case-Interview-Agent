#!/usr/bin/env python3
"""step2_delegation_gate.py — Step-2 PLUMBING gate for Explainable + HITL.

The other two gates test the SHARED layer directly:
  - a_row_gate.py  -> matching.locate() resolution CORRECTNESS  (needs a key)
  - step2_gate.py  -> grounding + resolve_removal_target + filler guard (mostly key-free)

Neither proves the thing this file proves: that the EXP and HITL agents actually
DELEGATE to that shared layer (rather than keeping a drifted private copy). This gate
instantiates both agents, replaces the single LLM seam (matching.classify_json) with a
canned stub, and asserts:

  1. every delegated matcher routes THROUGH matching.* (seam call-count > 0; patching
     matching.classify_json changes their output),
  2. return CONTRACTS are unchanged per arm:
       - EXP _match_key_question -> {"question": <VERBATIM, [a] refs KEPT>, "sources"}
       - HITL _match_key_question -> ref-STRIPPED str (== the shared function's output)
       - _match_pillar -> area name | None ; EXP _match_concept -> parent pillar | None
  3. EXP/HITL _concept_grounding is BYTE-IDENTICAL to grounding.ground_pillar
     (real pillar AND the planted swap concept),
  4. EXP _is_excluded_bullet is the shared predicate,
  5. negative paths (stub says matched=False) return None.

It makes ZERO live API calls, so it needs no GEMINI_API_KEY — but unlike step2_gate it
imports the agents, so it needs the same runtime deps the app imports (google.genai,
firebase_admin, dotenv). Run from the project root:

    poetry run python step2_delegation_gate.py

This is PLUMBING identity, not resolution accuracy. Accuracy parity across arms is the
live a_row_gate (run it in a keyed env once this passes).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import re  # noqa: E402

from backend.domain import matching as m       # noqa: E402
from backend.domain import grounding as g       # noqa: E402
from backend.knowledge import knowledge_base as kb        # noqa: E402
from backend.agents.explainable import ExplainableAgent  # noqa: E402
from backend.agents.hitl import HITLAgent                # noqa: E402

passed = total = 0


def check(label, cond, detail=""):
    global passed, total
    total += 1
    passed += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   -> {detail}" if detail and not cond else ""))


# ── fixtures from the REAL KB (no hardcoded study content) ──────────────────
_PILLAR = next(p for p in kb.get_all_pillars() if p.get("key_questions"))
PNAME = _PILLAR["name"]
KQS = _PILLAR["key_questions"]
_CONCEPT = next(c for c in kb.get_all_concepts() if not c.get("swap") and c.get("pillar_id"))
CPARENT = kb.get_pillar_by_id(_CONCEPT["pillar_id"])["name"]
SWAP_NAME = (kb.get_swap_concept() or {}).get("name", "")
_REF = re.compile(r"\[[a-z]\]")
# index of a key_question that actually carries an inline [a] ref (for the citation path)
_REF_IDX = next((i for i, q in enumerate(KQS) if _REF.search(q)), 0)
_REF_KQ_PRESENT = bool(_REF.search(KQS[_REF_IDX]))

# ── stub the ONE shared LLM seam; count calls to prove routing ──────────────
_seam = {"n": 0, "mode": "match"}


def _stub(prompt, *a, **k):
    _seam["n"] += 1
    if _seam["mode"] == "miss":
        return {"matched": False, "confidence": 0.0}
    low = prompt.lower()
    if "key questions" in low:                       # ADD_MATCH (key-question)
        return {"matched": True, "matched_index": _REF_IDX, "confidence": 0.99}
    if "analytical concepts" in low:                 # ADD_CONCEPT
        return {"matched": True, "matched_index": 0, "confidence": 0.99}
    return {"matched": True, "matched_pillar": PNAME, "confidence": 0.99}  # ADD_PILLAR/area


m.classify_json = _stub


def _bare(cls):
    """Instantiate without running __init__ (which needs a live case/session); set only
    the attributes the tested methods read."""
    a = cls.__new__(cls)
    a.excluded_sub_bullets = {}
    return a


exp, hit = _bare(ExplainableAgent), _bare(HITLAgent)

print(f"KB fixtures: pillar={PNAME!r} ({len(KQS)} kqs, ref-kq idx={_REF_IDX}, "
      f"ref present={_REF_KQ_PRESENT}) | concept parent={CPARENT!r} | swap={SWAP_NAME!r}\n")

print("=== 1. EXP delegates + contracts ===")
_seam["n"] = 0
check("EXP _match_pillar -> area name", exp._match_pillar("x") == PNAME)
check("EXP _match_concept -> PARENT pillar name", exp._match_concept("x") == CPARENT)
kq = exp._match_key_question("x", PNAME)
check("EXP _match_key_question -> dict", isinstance(kq, dict))
check("EXP key-question keeps VERBATIM text (refs intact)",
      isinstance(kq, dict) and kq.get("question") == KQS[_REF_IDX],
      detail=repr(kq.get("question") if isinstance(kq, dict) else kq)[:80])
if _REF_KQ_PRESENT:
    check("EXP citation path: [a] refs survive (this is why EXP keeps a dict)",
          isinstance(kq, dict) and bool(_REF.search(kq.get("question", ""))))
check("EXP key-question dict has 'sources' key", isinstance(kq, dict) and "sources" in kq)
check("EXP routed through matching.* (seam called)", _seam["n"] > 0, detail=f"seam={_seam['n']}")

print("\n=== 2. HITL delegates + contracts ===")
_seam["n"] = 0
check("HITL _match_pillar -> area name", hit._match_pillar("x") == PNAME)
hkq = hit._match_key_question("x", PNAME)
check("HITL _match_key_question -> str (ref-STRIPPED)", isinstance(hkq, str))
check("HITL key-question == shared matcher's stripped form",
      hkq == m._strip_source_refs(KQS[_REF_IDX]), detail=repr(hkq)[:80])
check("HITL key-question has NO dangling [a] refs",
      isinstance(hkq, str) and not _REF.search(hkq))
check("HITL routed through matching.* (seam called)", _seam["n"] > 0, detail=f"seam={_seam['n']}")

print("\n=== 3. Grounding identity (both arms == grounding.ground_pillar) — 0 seam ===")
_seam["n"] = 0
check("EXP _concept_grounding == ground_pillar (real pillar)",
      exp._concept_grounding(PNAME) == g.ground_pillar(PNAME))
check("HITL _concept_grounding == ground_pillar (real pillar)",
      hit._concept_grounding(PNAME) == g.ground_pillar(PNAME))
if SWAP_NAME:
    check("EXP _concept_grounding == ground_pillar (SWAP concept)",
          exp._concept_grounding(SWAP_NAME) == g.ground_pillar(SWAP_NAME))
    check("HITL _concept_grounding == ground_pillar (SWAP concept)",
          hit._concept_grounding(SWAP_NAME) == g.ground_pillar(SWAP_NAME))
check("grounding made NO LLM calls", _seam["n"] == 0, detail=f"seam={_seam['n']}")

print("\n=== 4. EXP _is_excluded_bullet via shared predicate ===")
exp.excluded_sub_bullets = {PNAME: [KQS[0]]}
check("excluded bullet -> True", exp._is_excluded_bullet(PNAME, KQS[0]))
check("non-excluded bullet -> False", not exp._is_excluded_bullet(PNAME, "some unrelated point"))
exp.excluded_sub_bullets = {}

print("\n=== 5. Negative paths (stub: matched=False) -> None ===")
_seam["mode"] = "miss"
check("EXP _match_pillar -> None", exp._match_pillar("x") is None)
check("EXP _match_concept -> None", exp._match_concept("x") is None)
check("EXP _match_key_question -> None", exp._match_key_question("x", PNAME) is None)
check("HITL _match_pillar -> None", hit._match_pillar("x") is None)
check("HITL _match_key_question -> None", hit._match_key_question("x", PNAME) is None)
_seam["mode"] = "match"

print(f"\n{'='*60}\nStep-2 delegation gate: {passed}/{total} passed.")
if passed == total:
    print("MET — EXP & HITL both delegate to the shared domain layer; contracts intact.\n"
          "Next: live a_row_gate.py in a keyed env to confirm RESOLUTION parity across arms.")
sys.exit(0 if passed == total else 1)
