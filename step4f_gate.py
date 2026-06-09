#!/usr/bin/env python3
"""step4f_gate.py — Step-4f gate for the steering/delegation guard in classify_intent
(the F-I1-HITL fix). Deterministic: we STUB classify_json to return the label the LLM
would assign (including the buggy `add, detail='guide it'` shape observed live), then
assert the post-classification guard reroutes whole-message steering phrases to
ask_agent_to_suggest with NO detail — while a real named concept (even one that contains
a steering word) passes through untouched. Shared router, so this covers all three arms.
Run: python3 step4f_gate.py
"""
import _bootstrap                                      # noqa: F401  (SDK stubs, no key)
from backend.interaction import intents as I

passed = total = 0
def check(label, cond, detail=""):
    global passed, total
    total += 1; passed += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail and not cond else ""))

# ── stub classify_json: simulate the LLM's raw label BEFORE the guard runs ──
# We make the stub MISLABEL steering phrases as the buggy add+detail shape (worst case),
# so a green gate proves the deterministic guard fixes it regardless of the LLM.
def make_stub(intent="add", detail_echo=True, parent=None):
    def _stub(prompt):
        # pull the user message out of the formatted prompt
        msg = prompt.split("USER MESSAGE")[-1]
        # crude: the message is on the line after the rule; just use a sentinel via closure
        return {"intent": intent,
                "detail": _stub.detail if detail_echo else None,
                "parent": parent, "confidence": 0.95}
    _stub.detail = None
    return _stub

def classify(text, raw_intent="add", raw_detail=None):
    """Run classify_intent with the stubbed LLM returning (raw_intent, raw_detail)."""
    stub = make_stub(intent=raw_intent, detail_echo=True)
    stub.detail = raw_detail if raw_detail is not None else text
    I.classify_json = stub
    return I.classify_intent(text, current_pillar="Feasibility",
                             walkthrough_pillars="Strategic Fit, Feasibility")

print("STEP 4f GATE — steering/delegation guard in classify_intent (F-I1-HITL)\n")

# ── the live bug: "guide it" was returned as add/detail='guide it' -> phantom pillar ──
r = classify("guide it", raw_intent="add", raw_detail="guide it")
check("'guide it' (LLM said add) -> rerouted to ask_agent_to_suggest, detail None",
      r.intent == "ask_agent_to_suggest" and r.detail is None and r.parent is None,
      f"{r.intent}/{r.detail!r}")

# ── other steering phrases, even if the LLM mislabels them as add ──
for phrase in ("you lead", "you decide", "lead it", "you pick", "your call",
               "up to you", "you take it from here", "you drive it", "you're the lead"):
    r = classify(phrase, raw_intent="add", raw_detail=phrase)
    check(f"steering '{phrase}' -> ask_agent_to_suggest, no detail",
          r.intent == "ask_agent_to_suggest" and r.detail is None, f"{r.intent}/{r.detail!r}")

# ── normalization: trailing punctuation / case must still match ──
r = classify("Guide it.", raw_intent="add", raw_detail="Guide it")
check("'Guide it.' (case + punctuation) -> still rerouted",
      r.intent == "ask_agent_to_suggest" and r.detail is None, f"{r.intent}/{r.detail!r}")

# ── AGENCY PRESERVED: a real named concept is NOT a steering phrase -> stays add ──
r = classify("data privacy", raw_intent="add", raw_detail="data privacy")
check("real concept 'data privacy' -> stays add, detail kept (agency preserved)",
      r.intent == "add" and r.detail == "data privacy", f"{r.intent}/{r.detail!r}")

# ── whole-message match only: a concept that CONTAINS a steering word is untouched ──
r = classify("user guide quality", raw_intent="add", raw_detail="user guide quality")
check("'user guide quality' (contains 'guide') -> NOT rerouted, stays add",
      r.intent == "add" and r.detail == "user guide quality", f"{r.intent}/{r.detail!r}")

r = classify("you decide whether to add data privacy", raw_intent="add", raw_detail="data privacy")
check("steering word in a longer concept sentence -> NOT rerouted (whole-message only)",
      r.intent == "add" and r.detail == "data privacy", f"{r.intent}/{r.detail!r}")

# ── REGRESSION: the existing filler guard still works (F-I1 'else'/'what else') ──
r = classify("what else", raw_intent="add", raw_detail="else")
check("regression: filler 'else' -> still rerouted to ask_agent_to_suggest, no detail",
      r.intent == "ask_agent_to_suggest" and r.detail is None, f"{r.intent}/{r.detail!r}")

# ── a genuine suggest request is unchanged ──
r = classify("what else should we look at?", raw_intent="ask_agent_to_suggest", raw_detail=None)
check("genuine suggest request -> ask_agent_to_suggest (unchanged)",
      r.intent == "ask_agent_to_suggest" and r.detail is None, f"{r.intent}/{r.detail!r}")

# ── a real remove is unaffected by the steering guard ──
r = classify("remove Feasibility", raw_intent="remove", raw_detail="Feasibility")
check("real remove -> stays remove, detail kept (steering guard does not touch it)",
      r.intent == "remove" and r.detail == "Feasibility", f"{r.intent}/{r.detail!r}")

# ── unit: _is_steering_message ──
check("_is_steering_message unit checks",
      I._is_steering_message("guide it") and I._is_steering_message("  Guide it!  ")
      and not I._is_steering_message("data privacy")
      and not I._is_steering_message("user guide quality")
      and not I._is_steering_message("go ahead"))   # 'go ahead' deliberately NOT steering

print(f"\n{passed}/{total} checks passed")
import sys; sys.exit(0 if passed == total else 1)
