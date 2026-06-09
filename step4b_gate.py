#!/usr/bin/env python3
"""step4b_gate.py — Step-4b headless gate for the BlackBox -> shared-handlers wiring.

Proves the BlackBoxAgent HandlerSession adapter (D-H3) + the outcome renderer are CORRECT:
real handlers.dispatch / resolve_pending drive a real BlackBoxAgent (built via __new__ to
skip firebase/LLM __init__), with STUBBED resolution (locate / resolve_removal_target) and a
deterministic confirmation classifier. Streaming helpers are stubbed to recorders so we test
the EVENT TIMING + state, not the framework re-render. Live LLM accuracy stays with
a_row_gate / d_row_gate / step4_handlers_gate (already green).

Covers the sanctioned 4b deltas: F-R1 delete-at-confirm (challenge fires NO delete; confirm
does), Fork-A deictic disambiguation, F-M2 reveal, F-M1 duplicate, novel-area add, sub-bullet
Fork-B (logs stored text), swap removal = DETECTION not delete (§0), D7 accept-reveal (no DV),
B5 nothing-to-remove add-offer + B6 offered-add = user_spontaneous, W9 swap-question at gate.

    python3 step4b_gate.py
"""
import os, sys, types as _t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root (backend/ is the package)

# ── stub SDKs so backend.llm / concept_swap import headless ──
_g = _t.ModuleType("google"); _genai = _t.ModuleType("google.genai")
_gt = _t.ModuleType("google.genai.types")
class _Client:
    def __init__(self, *a, **k): self.models = _t.SimpleNamespace(generate_content=lambda **kk: None)
class _Cfg:
    def __init__(self, *a, **k): pass
_genai.Client = _Client; _gt.GenerateContentConfig = _Cfg; _genai.types = _gt; _g.genai = _genai
class _Part:
    def __init__(self, text=None, **k): self.text = text
class _Content:
    def __init__(self, role=None, parts=None, **k): self.role = role; self.parts = parts or []
_gt.Content = _Content; _gt.Part = _Part
sys.modules.update({"google": _g, "google.genai": _genai, "google.genai.types": _gt})
_dot = _t.ModuleType("dotenv"); _dot.load_dotenv = lambda *a, **k: None
sys.modules["dotenv"] = _dot

# ── recording stub for backend.logger (so logger.py / firebase never load) ──
EVENTS = []
def _rec(name):
    def f(*a, **k): EVENTS.append((name, a, k))
    return f
_log = _t.ModuleType("backend.logger")
for _n in ["create_session","end_session","stamp_started_at","log_user_message",
           "log_agent_response","log_interruption","log_memory_override","update_answer",
           "log_warmup_response","log_concept_added","log_add_pillar","log_add_sub_bullet",
           "log_delete","log_question","log_swap_questioned","_log_event"]:
    setattr(_log, _n, _rec(_n))
_log.create_session = lambda *a, **k: "test-sid"
sys.modules["backend.logger"] = _log

from backend.interaction import handlers as h        # noqa: E402
from backend.domain import matching as m             # noqa: E402
import backend.black_box_agent as bbm                # noqa: E402
BlackBoxAgent = bbm.BlackBoxAgent

passed = total = 0
def check(label, cond, detail=""):
    global passed, total
    total += 1; passed += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail and not cond else ""))

def events_of(name):
    return [e for e in EVENTS if e[0] == name]

# ── deterministic resolution stubs (we test the adapter+render, not locate) ──
WRONG = "Process Step Count"
_LOCATE = {
    "feasibility":       m.KBMatch(pillar="Feasibility", level="pillar"),
    "it budget":         m.KBMatch(pillar="Financial Impact", pillar_is_withheld=True, level="pillar"),
    "risk & governance": m.KBMatch(pillar="Risk & Governance", pillar_is_withheld=True, level="pillar"),
    "vendor lock-in":    m.KBMatch(level="none", matched_text="vendor lock-in"),
}
def fake_locate(text):
    return _LOCATE.get((text or "").strip().lower(), m.KBMatch(level="none", matched_text=text))
def fake_resolve(text, *, last_discussed=None, shown_bullets=None):
    t = (text or "").strip().lower().replace("remove ", "", 1)
    if t in ("this", "it", "that"):
        return last_discussed if (last_discussed and last_discussed.level != "none") \
               else m.KBMatch(level="none", needs_disambiguation=True)
    return fake_locate(t)
h.m.locate = fake_locate
h.m.resolve_removal_target = fake_resolve
h._classify_confirmation = lambda txt: {"yes":"confirm","no":"decline"}.get((txt or "").strip().lower(), "other")

# ── build a real BlackBoxAgent without running __init__ (no firebase/LLM) ──
def fresh_agent(swap_injected=True, swap_detected=False):
    a = BlackBoxAgent.__new__(BlackBoxAgent)
    a.session_id = "test-sid"; a.history = []; a._pending = False
    a.excluded_concepts = []; a.excluded_sub_bullets = {}
    a.user_added_pillars = []; a.user_sub_points = {}
    a.pending = None; a.pending_suggestion = None
    a.last_discussed = None; a.shown_bullets = []
    a._last_surface = None; a._last_sub_add = None
    a.kg_context = {"concepts": ["Strategic Fit", "Solution Design & Scope", "Feasibility"],
                    "framework": "(framework)"}
    a.concept_swap = _t.SimpleNamespace(
        is_injected=swap_injected, is_detected=swap_detected,
        config={"wrong_concept": WRONG},
        matches=lambda txt: WRONG.lower() in (txt or "").lower() or "step count" in (txt or "").lower(),
        force_detected=lambda: setattr(a.concept_swap, "is_detected", True),
        check_detection=lambda txt: False)
    # stub streaming helpers -> recorders (event timing, not re-render, is the point)
    def _rerender(pre=""):
        a.history.append(bbm.types.Content(role="model", parts=[bbm.types.Part(text="[RERENDER] " + pre)]))
        if False: yield
    def _ack():
        if False: yield
    def _qa(u):
        a.history.append(bbm.types.Content(role="model", parts=[bbm.types.Part(text="[QA] " + u)]))
        if False: yield
    def _cqa(u, c):
        a.history.append(bbm.types.Content(role="model", parts=[bbm.types.Part(text="[CONFIRM-QA] " + u)]))
        if False: yield
    a._yield_rerender = _rerender
    a._ack_no_reprint = _ack
    a._stream_qa = _qa
    a._stream_confirm_qa = _cqa
    a._classify_swap_question = lambda u: True   # default; override per-row
    a._reply_is_question = lambda u: True        # default; override per-row
    return a

def drive(agent, intent, detail=None, parent=None, user_text=None, source="user_spontaneous"):
    """Run one full turn exactly like _stream_main's tail: snapshot -> dispatch -> render."""
    res = _t.SimpleNamespace(intent=intent, detail=detail, parent=parent)
    ut = user_text if user_text is not None else (detail or "")
    was_pending = agent.pending is not None
    pa = agent.pending
    outcome = h.dispatch(res, agent, user_text=ut, source=source)
    list(agent._render_outcome(outcome, ut, was_pending=was_pending, pa=pa))
    return outcome

print("STEP 4b GATE — BlackBox shared-handlers wiring\n")

# ── 0. adapter satisfies the HandlerSession surface (D-H3) ──
print("[0] HandlerSession adapter surface")
a = fresh_agent()
for attr in ["pending","pending_suggestion","last_discussed","shown_bullets",
             "excluded_concepts","excluded_sub_bullets"]:
    check(f"attr {attr}", hasattr(a, attr))
for meth in ["presented_pillars","presented_sub_bullets","surfaced_pillar_names","current_pillar",
             "surface_pillar","add_sub_point","swap_name","is_swap_target","mark_swap_detected",
             "requires_justification"]:
    check(f"method {meth}", callable(getattr(a, meth, None)))
for dead in ["_begin_removal","_handle_add","_resolve_pending_excl","_classify_removal_target"]:
    check(f"retired {dead} gone", not hasattr(a, dead))

# ── 1. F-R1: challenge fires NO delete; confirm fires log_delete ──
print("\n[1] F-R1 delete-at-confirm (pillar)")
a = fresh_agent(); EVENTS.clear()
drive(a, "remove", detail="Feasibility", user_text="remove Feasibility")
check("challenge parks pending", a.pending is not None and a.pending.type == "remove_pillar")
check("challenge: NO log_delete", len(events_of("log_delete")) == 0)
drive(a, "none", user_text="yes")          # confirm
check("confirm clears pending", a.pending is None)
check("confirm: log_delete fires once", len(events_of("log_delete")) == 1,
      f"got {len(events_of('log_delete'))}")
check("confirm: target excluded", "Feasibility" in a.excluded_concepts)
check("confirm: memory_override fired", len(events_of("log_memory_override")) == 1)

# ── 2. abandon: 'no' -> no delete, pending cleared ──
print("\n[2] abandon")
a = fresh_agent(); EVENTS.clear()
drive(a, "remove", detail="Feasibility", user_text="remove Feasibility")
drive(a, "none", user_text="no")
check("abandon: pending cleared", a.pending is None)
check("abandon: NO log_delete", len(events_of("log_delete")) == 0)
check("abandon: not excluded", "Feasibility" not in a.excluded_concepts)

# ── 3. Fork-A: deictic with no focus -> needs_disambiguation (no park, no delete) ──
print("\n[3] Fork-A deictic disambiguation")
a = fresh_agent(); EVENTS.clear()
o = drive(a, "remove", detail="it", user_text="remove it")
check("deictic -> needs_disambiguation", isinstance(o, h.RemovalOutcome) and o.stage == "needs_disambiguation")
check("deictic: nothing parked", a.pending is None)
check("deictic: NO log_delete", len(events_of("log_delete")) == 0)

# ── 4. F-M2 reveal: add 'IT budget' reveals Financial Impact (logs add_pillar) ──
print("\n[4] F-M2 reveal")
a = fresh_agent(); EVENTS.clear()
o = drive(a, "add", detail="IT budget", user_text="add IT budget")
check("reveal action", isinstance(o, h.AddOutcome) and o.action == "revealed")
check("Financial Impact surfaced", "Financial Impact" in a.user_added_pillars)
check("reveal logs add_pillar", len(events_of("log_add_pillar")) == 1)
check("reveal logs concept_added", len(events_of("log_concept_added")) == 1)

# ── 5. F-M1 duplicate: add a shown pillar -> duplicate, NO add_pillar ──
print("\n[5] F-M1 duplicate")
a = fresh_agent(); EVENTS.clear()
o = drive(a, "add", detail="Feasibility", user_text="add Feasibility")
check("duplicate action", isinstance(o, h.AddOutcome) and o.action == "duplicate")
check("duplicate: NO add_pillar", len(events_of("log_add_pillar")) == 0)

# ── 6. novel area add -> new pillar, logs add_pillar ──
print("\n[6] novel area add")
a = fresh_agent(); EVENTS.clear()
o = drive(a, "add", detail="vendor lock-in", user_text="add vendor lock-in")
check("added_new pillar", isinstance(o, h.AddOutcome) and o.action == "added_new" and o.level == "pillar")
check("novel area surfaced", "vendor lock-in" in a.user_added_pillars)
check("novel area logs add_pillar", len(events_of("log_add_pillar")) == 1)

# ── 7. sub-bullet render (Fork-B: logs the STORED text), is_new gating ──
print("\n[7] sub-bullet add render (Fork-B)")
a = fresh_agent(); EVENTS.clear()
a._last_sub_add = {"pillar": "Feasibility", "stored": "Stored formatted text", "raw": "roi", "is_new": True}
list(a._render_add(h.AddOutcome(action="added_new", pillar="Feasibility", level="sub_bullet",
                                counted=True, text="roi")))
adds = events_of("log_add_sub_bullet")
check("sub-bullet logs add_sub_bullet", len(adds) == 1)
check("sub-bullet logs STORED text (Fork-B)", adds and adds[0][1][1] == "Stored formatted text",
      f"got {adds[0][1] if adds else None}")
check("sub-bullet logs concept_added(raw)", any(e[1][1] == "roi" for e in events_of("log_concept_added")))
# is_new False -> no log
a._last_sub_add = {"pillar": "Feasibility", "stored": "x", "raw": "x", "is_new": False}; EVENTS.clear()
list(a._render_add(h.AddOutcome(action="added_new", pillar="Feasibility", level="sub_bullet",
                                counted=True, text="x")))
check("duplicate sub-bullet: NO log", len(events_of("log_add_sub_bullet")) == 0)

# ── 8. swap removal = DETECTION, never a delete (§0) ──
print("\n[8] swap removal -> detection not delete")
a = fresh_agent(); EVENTS.clear()
o = drive(a, "remove", detail="the step count metric", user_text="remove the step count metric")
check("swap challenge parks swap pending", a.pending is not None and a.pending.is_swap)
check("swap challenge: NO log_delete", len(events_of("log_delete")) == 0)
drive(a, "none", user_text="yes")
check("swap confirm: detected", a.concept_swap.is_detected is True)
check("swap confirm: NO log_delete (§0)", len(events_of("log_delete")) == 0)
check("swap confirm: memory_override marker", len(events_of("log_memory_override")) == 1)

# ── 9. D7: accept an agent suggestion -> reveal, NO DV (no add_pillar log) ──
print("\n[9] D7 accept-suggestion (no DV)")
a = fresh_agent(); EVENTS.clear()
o1 = drive(a, "ask_agent_to_suggest", user_text="what else should I consider?")
check("suggest parks pending_suggestion", a.pending_suggestion is not None and
      a.pending_suggestion.get("origin") == "agent_suggest")
suggested = a.pending_suggestion["item"]
EVENTS.clear()
o2 = drive(a, "none", user_text="yes")     # accept
check("D7 accept -> SuggestOutcome revealed", isinstance(o2, h.SuggestOutcome) and o2.revealed)
check("D7: withheld pillar surfaced", suggested in a.user_added_pillars)
check("D7: NO add_pillar log (not a DV)", len(events_of("log_add_pillar")) == 0)
check("D7: NO concept_added log", len(events_of("log_concept_added")) == 0)

# ── 10. B5 nothing-to-remove add-offer + B6 offered-add = user_spontaneous ──
print("\n[10] B5 remove-offer + B6 accept")
a = fresh_agent(); EVENTS.clear()
o = drive(a, "remove", detail="Risk & Governance", user_text="remove Risk & Governance")
check("B5 nothing_to_remove + offer", isinstance(o, h.RemovalOutcome)
      and o.stage == "nothing_to_remove" and o.suggest_add_alternative == "Risk & Governance")
check("B5 parks remove_offer", a.pending_suggestion is not None
      and a.pending_suggestion.get("origin") == "remove_offer")
check("B5: NO log_delete", len(events_of("log_delete")) == 0)
EVENTS.clear()
o2 = drive(a, "none", user_text="yes")     # B6 accept -> user_spontaneous add
check("B6 accept -> AddOutcome", isinstance(o2, h.AddOutcome))
check("B6: Risk & Governance surfaced", "Risk & Governance" in a.user_added_pillars)
check("B6: logs add_pillar (user_spontaneous)", len(events_of("log_add_pillar")) == 1)

# ── 11. W9: question AT the confirm gate preserves log_swap_questioned ──
print("\n[11] W9 swap-question at gate")
a = fresh_agent(); EVENTS.clear()
drive(a, "remove", detail="the step count metric", user_text="remove the step count metric")  # swap challenge
a._reply_is_question = lambda u: True
EVENTS.clear()
drive(a, "question", user_text="why is that one in there?")   # 'other' at gate, is a question
check("gate stays challenged (pending kept)", a.pending is not None and a.pending.is_swap)
check("W9: log_question fired at gate", len(events_of("log_question")) == 1)
check("W9: log_swap_questioned fired at gate", len(events_of("log_swap_questioned")) == 1)
check("W9 gate: NO log_delete", len(events_of("log_delete")) == 0)

print(f"\n{'='*52}\n  RESULT: {passed}/{total} checks passed\n{'='*52}")
sys.exit(0 if passed == total else 1)
