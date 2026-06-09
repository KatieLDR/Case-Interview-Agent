#!/usr/bin/env python3
"""step4c_gate.py — Step-4c headless gate for the Explainable agent wired onto
interaction/handlers.dispatch / resolve_pending.

Mirrors step4_handlers_gate's posture: SDKs stubbed (no key/LLM), classify_intent /
m.locate / m.resolve_removal_target / handlers._classify_confirmation are deterministic,
and the persona STREAM helpers are recorders. We verify on EXPECTED BEHAVIOUR LABELS —
which render path fired + which §3.6 events fired (delete only at confirm, swap = detection
with NO delete) — not raw LLM text (non-determinism). Run: python3 step4c_gate.py
"""
import _bootstrap                                   # noqa: F401  (SDK/dotenv/firebase stubs)
import types as _t
from backend.interaction import intents, handlers as h
from backend.domain import matching as m
from backend.agents import explainable as EA
from backend.agents.explainable import ExplainableAgent

passed = total = 0
def check(label, cond, detail=""):
    global passed, total
    total += 1; passed += bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"   {detail}" if detail and not cond else ""))

# ── event recorder: patch the logger names imported INTO explainable_agent ──
EVENTS = []
def rec(name):
    def _f(*a, **k): EVENTS.append((name, a[1:] if len(a) > 1 else ()))
    return _f
for nm in ("log_user_message", "log_agent_response", "log_interruption",
           "log_memory_override", "update_answer", "log_concept_added",
           "log_question", "log_add_pillar", "log_add_sub_bullet", "log_delete",
           "log_swap_questioned"):
    setattr(EA, nm, rec(nm))

def names(): return [e[0] for e in EVENTS]
def clear(): EVENTS.clear()

# ── deterministic intent router (keyword -> IntentResult) ──
def fake_classify_intent(text, **kw):
    t = text.lower()
    if t.startswith("remove") or t.startswith("drop") or "take out" in t:
        return intents.IntentResult(intent="remove", detail=text, confidence=1.0,
                                    error=None, parent=None)
    if t.startswith("add") or "include" in t:
        # crude parent split: "add X under Y"
        parent = None
        if " under " in t:
            parent = text.split(" under ", 1)[1].strip()
            text2 = text.split(" under ", 1)[0]
        else:
            text2 = text.replace("add", "", 1).strip()
        return intents.IntentResult(intent="add", detail=text2.strip(), confidence=1.0,
                                    error=None, parent=parent)
    if t.startswith("go back") or t.startswith("revisit") or "back to" in t:
        d = text.split("to", 1)[1].strip() if "to" in t else text
        return intents.IntentResult(intent="revisit", detail=d, confidence=1.0,
                                    error=None, parent=None)
    if "suggest" in t or "what else" in t or "what am i missing" in t or "missing" in t:
        return intents.IntentResult(intent="ask_agent_to_suggest", detail=text,
                                    confidence=1.0, error=None, parent=None)
    if t.startswith("what") or t.startswith("why") or t.endswith("?"):
        return intents.IntentResult(intent="question", detail=text, confidence=1.0,
                                    error=None, parent=None)
    if t in ("next", "move on", "continue", "go on"):
        return intents.IntentResult(intent="advance", detail=None, confidence=1.0,
                                    error=None, parent=None)
    if "doesn't belong" in t or "not sure about this" in t or "seems off" in t or "doubt" in t:
        return intents.IntentResult(intent="doubt", detail=text, confidence=1.0,
                                    error=None, parent=None)
    if t in ("yes", "yep", "do it", "ok", "sure", "go ahead"):
        return intents.IntentResult(intent="none", detail=text, confidence=1.0,
                                    error=None, parent=None)
    if t in ("no", "nope", "keep it", "never mind"):
        return intents.IntentResult(intent="none", detail=text, confidence=1.0,
                                    error=None, parent=None)
    return intents.IntentResult(intent="none", detail=text, confidence=0.5,
                                error=None, parent=None)
intents.classify_intent = fake_classify_intent
EA.intents.classify_intent = fake_classify_intent

# ── deterministic KB resolution (mirrors step4_handlers_gate's _LOCATE) ──
SWAP = "Average number of steps walked per day by the IT team"
_LOCATE = {
    "strategic fit":     m.KBMatch(pillar="Strategic Fit", level="pillar"),
    "feasibility":       m.KBMatch(pillar="Feasibility", level="pillar"),
    "financial impact":  m.KBMatch(pillar="Financial Impact", pillar_is_withheld=True, level="pillar"),
    "it budget":         m.KBMatch(pillar="Financial Impact", pillar_is_withheld=True, level="pillar"),
    "risk & governance": m.KBMatch(pillar="Risk & Governance", pillar_is_withheld=True, level="pillar"),
    "risk governance":   m.KBMatch(pillar="Risk & Governance", pillar_is_withheld=True, level="pillar"),
    "data quality":      m.KBMatch(pillar="Feasibility", level="concept",
                                   concept_id="data_q", matched_text="data quality"),
    "vendor lock-in":    m.KBMatch(level="none", matched_text="vendor lock-in"),
}
def fake_locate(text):
    return _LOCATE.get((text or "").strip().lower(), m.KBMatch(level="none",
                       matched_text=(text or "").strip()))
PRESENTED_SUB = "Data quality and availability: is input data clean, complete, and consistently available?"
def fake_resolve(text, *, last_discussed=None, shown_bullets=None):
    t = (text or "").strip().lower().replace("remove ", "", 1).replace("drop ", "", 1)
    if "step" in t or "walk" in t:
        return m.KBMatch(pillar=SWAP, level="pillar", matched_text=SWAP)
    if "data quality" in t:                      # a PRESENTED Feasibility sub-bullet
        return m.KBMatch(pillar="Feasibility", level="concept", pillar_is_withheld=False,
                         matched_text=PRESENTED_SUB, concept_id="data_q")
    if "phantom" in t:                           # NAMED concept, parent shown, bullet NOT presented
        return m.KBMatch(pillar="Feasibility", level="concept", pillar_is_withheld=False,
                         matched_text="a concept never surfaced here", concept_id="phantom")
    return fake_locate(t)
m.locate = fake_locate; h.m.locate = fake_locate
m.resolve_removal_target = fake_resolve; h.m.resolve_removal_target = fake_resolve

# ── deterministic confirmation (free-text yes/no/other) ──
def fake_confirm(text):
    t = (text or "").strip().lower()
    if t in ("yes", "yep", "do it", "ok", "sure", "go ahead"): return "confirm"
    if t in ("no", "nope", "keep it", "never mind"):           return "decline"
    return "other"
h._classify_confirmation = fake_confirm

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
        t = (text or "").lower()
        return "step" in t or "walk" in t or SWAP.lower() in t
    def check_detection(self, text):
        if self.detected: return False
        t = (text or "").lower()
        if "doesn't belong" in t or "seems off" in t or "not sure about this" in t:
            self.detected = True; return True
        return False
    def force_detected(self): self.detected = True
    def get_system_prompt_block(self): return ""
    def maybe_inject(self, *a): pass
    def log_presented(self): pass

# ── build an EXP instance without __init__/LLM ──
def make_agent(*, index=2, done=False, swap_presented=False, swap_detected=False):
    a = ExplainableAgent.__new__(ExplainableAgent)
    a.session_id = "gate"
    a.history = []
    a._pending = False
    a.has_main_contribution = False
    a.kg_context = {"framework": "AI Implementation Framework",
                    "case_type": "process automation",
                    "concepts": ["Strategic Fit", "Solution Design & Scope", "Feasibility"]}
    # walkthrough: [Strategic Fit, Solution Design & Scope, <SWAP>, Feasibility]
    a.walkthrough_concepts = ["Strategic Fit", "Solution Design & Scope", SWAP, "Feasibility"]
    a.swap_position = 2
    a.walkthrough_index = index
    a.walkthrough_done = done
    a.swap_presented = swap_presented
    a.excluded_concepts = []
    a.excluded_sub_bullets = {}
    a.user_sub_points = {}
    a.user_added_pillars = []
    a.pending = None
    a.pending_suggestion = None
    a.last_discussed = None
    a.shown_bullets = []
    a._last_surface = None
    a._last_sub_add = None
    sw = FakeSwap(); sw.detected = swap_detected
    a.concept_swap = sw
    # recorder stubs for persona streams (yield a labelled sentinel)
    a._render_pillar_block = lambda name: f"[block:{name}]"
    a._render_pillar_block_no_sources = lambda name: f"[block:{name}]"
    a._match_key_question = lambda item, pillar: None
    a._format_sub_bullet = lambda item: item.strip()
    a._last_agent_message = lambda: ""
    a._reply_is_question = lambda text: text.strip().endswith("?")
    def streamer(tag):
        def _g(*a_, **k_):
            EVENTS.append((f"stream:{tag}", ())); yield f"[{tag}]"
        return _g
    for tag in ("_stream_concept", "_stream_concept_qa", "_stream_swap_caught",
                "_stream_summary", "_stream_freeform", "_stream_pushback",
                "_stream_sub_bullet_pushback"):
        setattr(a, tag, streamer(tag))
    # _stream_summary must also set walkthrough_done (real one does)
    def _summary(*a_, **k_):
        a.walkthrough_done = True; EVENTS.append(("stream:_stream_summary", ()))
        yield "[summary]"
    a._stream_summary = _summary
    return a

def run(a, text):
    clear()
    return "".join(a._stream_main(text)), names()

print("STEP 4c GATE — Explainable wired onto shared handlers\n")

# ── F-R1: pillar removal — challenge has NO delete; confirm fires exactly one delete ──
a = make_agent(index=3)            # on Feasibility
_, ev = run(a, "remove Feasibility")
check("F-R1 remove pillar -> challenged pushback, NO delete",
      "stream:_stream_pushback" in ev and "log_delete" not in ev and a.pending is not None, str(ev))
_, ev = run(a, "yes")
check("F-R1 confirm -> exactly one log_delete + advance, pillar excluded",
      ev.count("log_delete") == 1 and "Feasibility" in [x.lower().title() for x in a.excluded_concepts]
      and a.pending is None, str(ev))

# ── swap removal = DETECTION, never a delete (§0) ──
a = make_agent(index=2, swap_presented=True)   # sitting ON the swap
_, ev = run(a, "remove the steps walked metric")
check("swap remove -> challenged (is_swap), pushback, NO delete",
      "stream:_stream_pushback" in ev and "log_delete" not in ev
      and a.pending is not None and a.pending.is_swap, str(ev))
_, ev = run(a, "yes")
check("swap confirm -> detected, swap_caught, NO log_delete (§0)",
      a.concept_swap.is_detected and "stream:_stream_swap_caught" in ev
      and "log_delete" not in ev and SWAP in a.excluded_concepts, str(ev))

# ── swap semantic backstop (W2): non-steering rejection while on swap ──
a = make_agent(index=2, swap_presented=True)
_, ev = run(a, "this doesn't belong here")
check("swap backstop -> detection + swap_caught, NO delete",
      a.concept_swap.is_detected and "stream:_stream_swap_caught" in ev
      and "log_delete" not in ev, str(ev))

# ── F-M2: add a withheld area -> reveal -> log_add_pillar ──
a = make_agent(index=3)
_, ev = run(a, "add IT budget")
check("F-M2 add IT budget -> reveal Financial Impact, log_add_pillar fires",
      "log_add_pillar" in ev and "Financial Impact" in a.walkthrough_concepts, str(ev))

# ── F-M1: add an already-shown concept -> duplicate, NO add event ──
a = make_agent(index=3)
_, ev = run(a, "add data quality")
check("F-M1 add data quality (shown concept) -> duplicate, NO add log",
      "log_add_pillar" not in ev and "log_add_sub_bullet" not in ev, str(ev))

# ── novel sub-point under current concept -> add_sub_bullet ──
a = make_agent(index=3)
_, ev = run(a, "add vendor lock-in")
check("novel add -> sub_bullet under current, log_add_sub_bullet fires",
      "log_add_sub_bullet" in ev and "vendor lock-in" in a.user_sub_points.get("Feasibility", []), str(ev))

# ── W9: question on the swap -> log_question + log_swap_questioned ──
a = make_agent(index=2, swap_presented=True)
_, ev = run(a, "why is this metric here?")
check("W9 question on swap -> log_question + log_swap_questioned, NO detection",
      "log_question" in ev and "log_swap_questioned" in ev and not a.concept_swap.is_detected, str(ev))

# ── W9 + W5: doubt on the swap -> fallback Q&A, swap_questioned, NON-destructive ──
a = make_agent(index=2, swap_presented=True)
_, ev = run(a, "I doubt this one fits, can you explain?")  # doubt, must NOT trip detection
check("W5/W9 doubt on swap -> grounded Q&A, swap_questioned, NO delete/detection",
      "log_swap_questioned" in ev and "log_delete" not in ev
      and "stream:_stream_concept_qa" in ev, str(ev))

# ── abandon: decline a parked removal -> NO delete ──
a = make_agent(index=3)
run(a, "remove Feasibility")
_, ev = run(a, "no")
check("abandon removal -> NO delete, pending cleared, pillar kept",
      "log_delete" not in ev and a.pending is None
      and "Feasibility" not in a.excluded_concepts, str(ev))

# ── nothing-to-remove + B5 offer; then B6 user_spontaneous add ──
a = make_agent(index=3)
_, ev = run(a, "remove Risk & Governance")     # withheld, not presented
check("nothing_to_remove (withheld) -> B5 add offer parked, NO delete",
      "log_delete" not in ev and a.pending_suggestion is not None
      and a.pending_suggestion.get("origin") == "remove_offer", str(ev))
_, ev = run(a, "yes")
check("B6 offered-add accept -> add Risk & Governance (log_add_pillar)",
      "log_add_pillar" in ev and "Risk & Governance" in a.walkthrough_concepts, str(ev))

# ── revisit: jump the cursor back to a passed pillar ──
a = make_agent(index=3)        # on Feasibility; Strategic Fit is passed
_, ev = run(a, "go back to Strategic Fit")
check("revisit -> cursor jumps to passed pillar, re-stream concept",
      a.walkthrough_index == 0 and a.walkthrough_done is False
      and "stream:_stream_concept" in ev, str(ev))

# ── D7: ask agent to suggest, then accept -> reveal (no delete) ──
a = make_agent(index=3)
_, ev = run(a, "what else am I missing?")
check("suggest -> SuggestOutcome offered (a withheld area), NO log yet",
      "log_add_pillar" not in ev and a.pending_suggestion is not None
      and a.pending_suggestion.get("origin") == "agent_suggest", str(ev))
_, ev = run(a, "yes")
check("D7 accept suggestion -> reveal (surface), area now in walk",
      any(p in a.walkthrough_concepts for p in ("Financial Impact", "Risk & Governance")), str(ev))

# ── advance: passive move-on -> next concept, no DV ──
a = make_agent(index=0)
_, ev = run(a, "move on")
check("advance -> cursor +1, stream next, NO add/remove/question log",
      a.walkthrough_index == 1 and "stream:_stream_concept" in ev
      and not ({"log_add_pillar","log_delete","log_question"} & set(ev)), str(ev))

# ── sub-bullet removal of a PRESENTED point — exercises presented_sub_bullets()
#    (regression guard for the _strip_source_refs NameError) + legit F-R1 at concept level ──
a = make_agent(index=3)            # on Feasibility (its KB sub-bullets are presented)
_, ev = run(a, "remove data quality")
check("remove PRESENTED sub-bullet -> challenged, NO crash, NO delete yet",
      "stream:_stream_sub_bullet_pushback" in ev and "log_delete" not in ev
      and a.pending is not None and a.pending.type == "remove_sub_bullet", str(ev))
_, ev = run(a, "yes")
check("confirm sub-bullet removal -> exactly one log_delete (F-R1), bullet excluded",
      ev.count("log_delete") == 1
      and any("data quality" in b.lower() for b in a.excluded_sub_bullets.get("Feasibility", [])), str(ev))

# ── F-R2b FIXED (4e): a NAMED concept whose PARENT is presented but whose BULLET is NOT
#    presented must be nothing_to_remove (NOT parked) — no phantom delete at confirm. ──
a = make_agent(index=3)
_, ev = run(a, "remove the phantom point")
check("F-R2b [FIXED@4e] named non-presented bullet -> nothing_to_remove, no pending, no delete",
      a.pending is None and "log_delete" not in ev, str(ev))

# ── F-R5 OBSERVABLE-DROP: a confirmed sub-bullet removal must actually DISAPPEAR from the
#    re-rendered block AND the summary — not merely land in excluded_sub_bullets (the BUG-4c1 /
#    F-R5 gate-coverage lesson). Uses the REAL _render_bullets_and_sources (block) + the shared
#    is_excluded_bullet predicate against the RAW ref-bearing KB bullet (exactly what
#    _stream_summary evaluates per bullet). ──
a = make_agent(index=3)
run(a, "remove data quality"); run(a, "yes")          # challenge -> confirm (full bullet stored)
block_md, _src = a._render_bullets_and_sources("Feasibility")
RAW_DQ = "Data quality and availability: is input data clean, complete, and consistently available? [a][b]"
check("F-R5 confirmed removal DROPS bullet from re-rendered block (observable, not just state)",
      "data quality and availability" not in block_md.lower()
      and "single-developer" in block_md.lower(), block_md)
check("F-R5 same removal is filtered from the summary (is_excluded_bullet on RAW ref-bearing KB bullet)",
      m.is_excluded_bullet(a.excluded_sub_bullets, "Feasibility", RAW_DQ) is True,
      str(a.excluded_sub_bullets))

print(f"\n{passed}/{total} checks passed")
import sys; sys.exit(0 if passed == total else 1)
