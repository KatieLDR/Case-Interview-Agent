#!/usr/bin/env python3
"""step4d_gate.py — Step-4d headless gate for the HITL agent wired onto the shared
interaction/handlers PendingAction + resolve_pending machine.

HITL is BUTTON-driven, so its removal path bypasses the LLM (D-Q1): confirm/cancel call
resolve_pending(decision=...) directly, and the justification gate is the only place a
typed turn matters (D-H2). That makes this gate FULLY DETERMINISTIC — no key, no stubbed
classifier needed for the removal/justification/swap rows (the F-M4 nudge-vs-elicited rows
stub classify_intent only). We verify on EXPECTED BEHAVIOUR LABELS — which §3.6 events
fired (delete only at confirm; swap = detection with NO delete) + parked state — not raw
LLM text. Run: python3 step4d_gate.py

GATE-COVERAGE LESSON (carried from BUG-4c1, §S/line 150): the FIRST rows exercise the
sub-bullet removal path (the gap that hid the 4c NameError), not pillar/swap removal only.
"""
import _bootstrap                                      # noqa: F401  (SDK stubs, no key)
from backend import hitl_agent as HA
from backend.hitl_agent import HITLAgent
from backend.interaction import handlers as h
from backend.domain import matching as m

passed = total = 0
def check(label, cond, detail=""):
    global passed, total
    total += 1; passed += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail and not cond else ""))

# ── event recorder: patch the logger names imported INTO hitl_agent ──
EVENTS = []
def rec(name):
    def _f(*a, **k): EVENTS.append((name, a[1:] if len(a) > 1 else ()))
    return _f
for nm in ("log_user_message", "log_agent_response", "log_interruption",
           "log_memory_override", "update_answer", "log_concept_added",
           "stamp_started_at", "log_question", "log_add_pillar",
           "log_add_sub_bullet", "log_delete", "log_swap_questioned"):
    setattr(HA, nm, rec(nm))
def names(): return [e[0] for e in EVENTS]
def clear(): EVENTS.clear()

# ── deterministic confirmation (free-text yes/no/other) for resolve_pending ──
def fake_confirm(text):
    t = (text or "").strip().lower()
    if t in ("yes", "yep", "do it", "ok", "sure", "go ahead"): return "confirm"
    if t in ("no", "nope", "keep it", "never mind", "cancel"):  return "decline"
    return "other"
h._classify_confirmation = fake_confirm

SWAP = "Average number of steps walked per day by the IT team"

# ── fake ConceptSwap ──
class FakeSwap:
    def __init__(self):
        self.config = {"wrong_concept": SWAP, "wrong_framework": "fitness tracking",
                       "match_terms": ["steps", "walk"], "match_stems": []}
        self.injected = True; self.detected = False
    @property
    def is_injected(self): return self.injected
    @property
    def is_detected(self): return self.detected
    def matches(self, text):
        t = (text or "").lower(); return "step" in t or "walk" in t or SWAP.lower() in t
    def check_detection(self, text):
        if self.detected: return False
        t = (text or "").lower()
        if "doesn't belong" in t or "seems off" in t or "not sure about this" in t:
            self.detected = True; return True
        return False
    def force_detected(self): self.detected = True

# ── build a HITL instance without __init__ / LLM / firebase ──
def make_agent(*, concepts=None, index=0, justification=None, blocks=None,
               sub_points=None, swap_presented=False):
    a = HITLAgent.__new__(HITLAgent)
    a.session_id = "gate"; a.history = []; a._pending = False
    a.walkthrough_concepts = list(concepts if concepts is not None
                                  else ["Strategic Fit", SWAP, "Feasibility"])
    a.walkthrough_index = index
    a.walkthrough_active = True; a.walkthrough_done = False
    a.swap_position = a.walkthrough_concepts.index(SWAP) if SWAP in a.walkthrough_concepts else 1
    a.swap_presented = swap_presented
    a.excluded_concepts = []; a.approved_concepts = []
    a.excluded_sub_bullets = {}
    a.concept_blocks = dict(blocks or {})
    a.user_sub_points = {k: list(v) for k, v in (sub_points or {}).items()}
    a.justification_pillars = set(justification or [])
    a.pending = None; a.pending_suggestion = None
    a.last_discussed = None; a.shown_bullets = []
    a._last_surface = None; a._last_sub_add = None
    a.user_contributed_concepts = set(); a.navigated_pillars = set()
    a.awaiting_sub_point = False; a.awaiting_revisit_add = False
    a.awaiting_justification = False; a.awaiting_user_suggestion = False
    a.revisit_target = None; a.justification_for = None
    a.prompt_index = 0; a.ack_index = 0
    a.concept_swap = FakeSwap()
    a._last_agent_text = lambda: ""
    # persona streamers -> labelled recorders
    def streamer(tag):
        def _g(*a_, **k_):
            EVENTS.append((f"stream:{tag}", ())); yield f"[{tag}]"
        return _g
    for tag in ("_stream_concept", "_stream_concept_qa", "_stream_swap_caught",
                "_stream_summary", "_stream_freeform", "_stream_proactive_prompt",
                "_walkthrough_complete_message", "_stream_justification_prompt"):
        setattr(a, tag, streamer(tag))
    return a

def drain(gen):
    return "".join(gen)

print("STEP 4d GATE — HITL wired onto the shared confirmation machine\n")

# ════════════════════════════════════════════════════════════════════════════
#  SUB-BULLET REMOVAL FIRST (the BUG-4c1 gate-coverage lesson, §S/line 150)
# ════════════════════════════════════════════════════════════════════════════
BLOCK = {"Feasibility": "- Data quality and availability\n- Single-developer dependency"}

# 1a — ➖ remove a presented sub-bullet: park, NO delete at intent (F-R1)
a = make_agent(index=2, blocks=BLOCK)            # on Feasibility
clear(); drain(a.on_remove_point("Single-developer dependency"))
check("sub-bullet remove -> parked (remove_sub_bullet), NO delete at intent",
      a.pending is not None and a.pending.type == "remove_sub_bullet"
      and "log_delete" not in names(), str(names()))
check("sub-bullet park -> remove-point confirmation buttons show",
      a.should_show_remove_point_confirmation() is True)

# 1b — confirm: exactly ONE delete (F-R1), bullet excluded + stripped from the block
clear(); drain(a.on_confirm_remove_point())
check("sub-bullet confirm -> exactly one log_delete (F-R1)",
      names().count("log_delete") == 1, str(names()))
check("sub-bullet confirm -> recorded in excluded_sub_bullets + stripped from block",
      any("single-developer" in b.lower() for b in a.excluded_sub_bullets.get("Feasibility", []))
      and "single-developer" not in a.concept_blocks["Feasibility"].lower()
      and a.pending is None, str(a.excluded_sub_bullets))

# 1c — cancel a parked sub-bullet removal: NO delete, point kept
a = make_agent(index=2, blocks=BLOCK)
drain(a.on_remove_point("Data quality and availability"))
clear(); drain(a.on_cancel_remove_point())
check("sub-bullet cancel -> NO delete, pending cleared, point kept",
      "log_delete" not in names() and a.pending is None
      and "data quality" in a.concept_blocks["Feasibility"].lower(), str(names()))

# ════════════════════════════════════════════════════════════════════════════
#  PILLAR REJECT — delete at CONFIRM only (F-R1), the old intent-delete is gone (F-R4)
# ════════════════════════════════════════════════════════════════════════════
# 2a — ❌ skip a normal pillar: park, NO delete at intent (this is the F-R4 fix)
a = make_agent(index=0)                          # on Strategic Fit
clear(); drain(a.on_reject_concept())
check("pillar reject -> parked (remove_pillar), NO delete at intent (F-R4 fixed)",
      a.pending is not None and a.pending.type == "remove_pillar"
      and "log_delete" not in names(), str(names()))
check("pillar park (no justification) -> confirmation buttons show",
      a.should_show_confirmation_buttons() is True)
# 2b — confirm: exactly one delete, excluded, cursor advanced
clear(); drain(a.on_confirm_reject())
check("pillar confirm -> exactly one log_delete (F-R1), excluded, cursor advanced",
      names().count("log_delete") == 1 and "Strategic Fit" in a.excluded_concepts
      and a.walkthrough_index == 1 and a.pending is None, str(names()))

# 2c — cancel a pillar reject: NO delete, kept + approved, cursor advanced
a = make_agent(index=0)
drain(a.on_reject_concept())
clear(); drain(a.on_cancel_reject())
check("pillar cancel -> NO delete, kept + approved, cursor advanced",
      "log_delete" not in names() and "Strategic Fit" not in a.excluded_concepts
      and "Strategic Fit" in a.approved_concepts and a.walkthrough_index == 1, str(names()))

# ════════════════════════════════════════════════════════════════════════════
#  SWAP REJECT — DETECTION on confirm, NEVER a delete (§0 / F-S3)
# ════════════════════════════════════════════════════════════════════════════
a = make_agent(index=1, swap_presented=True)     # sitting ON the swap
clear(); drain(a.on_reject_concept())
check("swap reject -> parked is_swap, NO delete, NO detection yet",
      a.pending is not None and a.pending.is_swap
      and "log_delete" not in names() and not a.concept_swap.is_detected, str(names()))
clear(); drain(a.on_confirm_reject())
check("swap confirm -> DETECTED + memory_override, NO log_delete (§0), excluded",
      a.concept_swap.is_detected and "log_delete" not in names()
      and "log_memory_override" in names() and SWAP in a.excluded_concepts
      and a.pending is None, str(names()))

# ════════════════════════════════════════════════════════════════════════════
#  JUSTIFICATION GATE (D-H2) — B8 passes -> confirm; B9 fails -> re-ask, NO delete
# ════════════════════════════════════════════════════════════════════════════
# 4a — a justification pillar parks requires_justification; buttons HIDDEN (type a reason)
a = make_agent(index=0, justification=["Strategic Fit"])
clear(); drain(a.on_reject_concept())
check("justification pillar -> parked requires_justification, NO confirm buttons",
      a.pending is not None and a.pending.requires_justification
      and a.should_show_confirmation_buttons() is False, str(names()))

# 4b — B9: a weak reason -> needs_justification, stays parked, NO delete
clear(); out = drain(a._stream_main("asdf"))
check("B9 weak reason -> re-ask, stays parked, NO delete",
      "log_delete" not in names() and a.pending is not None
      and a.pending.requires_justification, str(names()))

# 4c — B8: a meaningful reason -> confirm, exactly one delete, excluded
clear(); drain(a._stream_main("it does not apply to our case at all"))
check("B8 meaningful reason -> confirm, exactly one log_delete, excluded",
      names().count("log_delete") == 1 and "Strategic Fit" in a.excluded_concepts
      and a.pending is None, str(names()))

# 4d — cancelling a justification removal mid-reason ('never mind') -> kept, NO delete
a = make_agent(index=0, justification=["Strategic Fit"])
drain(a.on_reject_concept())
clear(); drain(a._stream_main("never mind"))
check("justification removal cancelled by text -> NO delete, kept",
      "log_delete" not in names() and "Strategic Fit" not in a.excluded_concepts
      and a.pending is None, str(names()))

# ════════════════════════════════════════════════════════════════════════════
#  F-M4 — free-text add NUDGES (0 add_pillar); elicited add executes ONCE
# ════════════════════════════════════════════════════════════════════════════
import backend.interaction.intents as I
def fake_intent(intent, detail=None, parent=None):
    return I.IntentResult(intent=intent, detail=detail, parent=parent,
                          confidence=1.0, error=None)

# normal turn: free-text "add ..." -> nudge, NO add_pillar (F-M4 structural fix)
a = make_agent(index=0)
I.classify_intent = lambda text, **kw: fake_intent("add", detail="data privacy")
HA.intents.classify_intent = I.classify_intent
clear(); drain(a._stream_main("we should add data privacy"))
check("F-M4 free-text add on normal turn -> NUDGE, 0 add_pillar",
      "log_add_pillar" not in names(), str(names()))

# elicited turn: same add at the proactive prompt -> exactly one add_pillar
a = make_agent(index=0)
a.awaiting_user_suggestion = True
a._check_duplicate_proactive = lambda c: {"is_duplicate": False, "matched_concept": None}
a._match_pillar = lambda c: None
I.classify_intent = lambda text, **kw: fake_intent("add", detail="data privacy")
HA.intents.classify_intent = I.classify_intent
clear(); drain(a._stream_main("data privacy"))
check("F-M4 elicited add (proactive prompt) -> exactly one add_pillar",
      names().count("log_add_pillar") == 1, str(names()))

# ════════════════════════════════════════════════════════════════════════════
#  RE-RENDER on ➕ (watch item: re-render after a committed add)
# ════════════════════════════════════════════════════════════════════════════
a = make_agent(index=2, blocks={"Feasibility": "- Data quality and availability"})
def _ssp(pillar, item, modality="text"):
    a.user_sub_points.setdefault(pillar, []).append(item.strip()); return item.strip(), True
a._store_sub_point = _ssp
a.awaiting_sub_point = True
clear(); out = drain(a._stream_main("budget headroom for the rollout"))
check("➕ add-mode -> committed point re-renders the block",
      "budget headroom" in out.lower() and "Feasibility" in out, out[:80])

# ════════════════════════════════════════════════════════════════════════════
#  D-H3 adapter conformance (read surface + swap channel + justification scope)
# ════════════════════════════════════════════════════════════════════════════
a = make_agent(index=1, justification=["Strategic Fit"], swap_presented=True,
               blocks={"Strategic Fit": "- Alignment with strategy"})
check("adapter: presented_pillars read-only up to cursor, excludes excluded",
      a.presented_pillars() == ["Strategic Fit", SWAP])
check("adapter: current_pillar read-only (does not advance cursor)",
      a.current_pillar() == SWAP and a.walkthrough_index == 1)
check("adapter: presented_sub_bullets returns stripped bullets",
      "Alignment with strategy" in a.presented_sub_bullets().get("Strategic Fit", []))
check("adapter: swap_name live until detected; is_swap_target on current swap",
      a.swap_name() == SWAP and a.is_swap_target(m.KBMatch(level="none"), "the steps metric"))
check("adapter: requires_justification -> True for a justification pillar, False for swap",
      a.requires_justification(m.KBMatch(pillar="Strategic Fit", level="pillar")) is True
      and a.requires_justification(m.KBMatch(pillar=SWAP, level="pillar")) is False)
a.mark_swap_detected()
check("adapter: mark_swap_detected -> force_detected + excluded, swap_name now None",
      a.concept_swap.is_detected and SWAP in a.excluded_concepts and a.swap_name() is None)

print(f"\n{passed}/{total} checks passed")
import sys; sys.exit(0 if passed == total else 1)
