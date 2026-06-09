"""step5_gate.py  —  Gate for REFACTOR_PLAN.md Step 5 (logging relocation, §3.6).

WHAT THIS PROVES (and what it cannot). Step 5's gate is "events match baseline in
CONTENT/COUNTERS, NOT empty diff" — the firing LOCATION moved from per-arm code into the one
shared `events.record`, so the source location changes by design. This harness asserts the
content/counter contract at the firing layer, with NO Firestore and NO Gemini (it drives
`events.record` with pre-built Outcomes against an in-memory FakeSink — `record` is a pure
mapping and never calls the LLM). Because the firing point is now arm-agnostic, cross-arm
identity (I-1) is proven BY CONSTRUCTION here: the same Outcome under three arm contexts must
yield identical event types + study fields.

WHAT STILL NEEDS A LIVE/MANUAL PASS (unchanged from §5): that each ARM actually calls
`events.record` at the right turn with the right Outcome + context (the rewire), the swap
classifier-channel events (swap_presented at render, swap_detected via ConceptSwap), and the
P3/P3b timer strings + I4 rendered summary. Those are boot + render facts this harness can't
see. Run: `python step5_gate.py` (logic), then a short live smoke for the arm wiring.

Run from the repo root:  python step5_gate.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields as dc_fields

from backend.logging import events as ev


# ════════════════════════════════════════════════════════════════════════════
#  FakeSink — records every write in order. No Firestore.
# ════════════════════════════════════════════════════════════════════════════
class FakeSink:
    def __init__(self):
        self.events: list[tuple[str, str, dict, str | None]] = []   # (sid, etype, fields, counter)
        self.flags: dict[str, dict] = {}
        self.counters: dict[str, int] = {}

    def write_event(self, session_id, etype, fields, counter=None):
        self.events.append((session_id, etype, dict(fields), counter))
        if counter:
            self.counters[counter] = self.counters.get(counter, 0) + 1

    def set_flag(self, session_id, field_name, value):
        self.flags.setdefault(session_id, {})[field_name] = value

    def stamp_started(self, session_id): self.set_flag(session_id, "started_at", "STAMP")
    def stamp_ended(self, session_id):   self.set_flag(session_id, "ended_at", "STAMP")

    # helpers
    def types(self): return [e[1] for e in self.events]
    def reset(self): self.__init__()


# ════════════════════════════════════════════════════════════════════════════
#  Outcomes: prefer the REAL handler dataclasses; fall back to field-mirrored
#  stand-ins so the gate runs in a deps-free sandbox. Assert parity when both exist.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class AddOutcome:
    action: str; pillar: str | None; level: str; counted: bool
    explanation: str | None = None; source: str = "user_spontaneous"
    matched_pillar_id: str | None = None; text: str | None = None

@dataclass
class RemovalOutcome:
    stage: str; target: str | None; level: str; pillar: str | None = None
    needs_justification: bool = False; justification: str | None = None
    consequence_facts: list = field(default_factory=list); post_delete_branch: bool = False
    is_swap: bool = False; suggest_add_alternative: str | None = None

@dataclass
class SuggestOutcome:
    level: str; suggested_item: str; grounding: str | None = None
    accepted: bool = False; revealed: bool = False

@dataclass
class QuestionOutcome:
    target_level: str | None; target: str | None
    grounding: str | None = None; is_about_swap: bool = False

@dataclass
class AdvanceOutcome:
    passive: bool = False

@dataclass
class FallbackOutcome:
    reason: str = "unclear"

STANDINS = {
    "AddOutcome": AddOutcome, "RemovalOutcome": RemovalOutcome,
    "SuggestOutcome": SuggestOutcome, "QuestionOutcome": QuestionOutcome,
    "AdvanceOutcome": AdvanceOutcome, "FallbackOutcome": FallbackOutcome,
}

_PARITY = []
try:
    from backend import handlers as H        # real dataclasses (needs genai/firebase importable)
    REAL = {n: getattr(H, n) for n in STANDINS}
    OUT = REAL
    for n, standin in STANDINS.items():
        real_names = {f.name for f in dc_fields(REAL[n])}
        stand_names = {f.name for f in dc_fields(standin)}
        missing = stand_names - real_names
        if missing:
            _PARITY.append(f"  ✗ {n}: gate stand-in has fields the real Outcome lacks: {missing}")
        # fields events.record READS must exist on the real outcome:
    SRC = "REAL handler outcomes"
except Exception as e:                        # sandbox / deps absent -> stand-ins
    OUT = STANDINS
    SRC = f"STAND-IN outcomes (real import failed: {type(e).__name__})"

def mk(name, **kw):
    return OUT[name](**kw)


# ════════════════════════════════════════════════════════════════════════════
#  Assertion plumbing
# ════════════════════════════════════════════════════════════════════════════
RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail and not cond else ""))

def ctx(arm):
    cfgs = {
        "bb":   ev.EventContext("S", agent_type="black_box",  modality="text"),
        "exp":  ev.EventContext("S", agent_type="explainable", modality="text"),
        "hitl": ev.EventContext("S", agent_type="hitl",        modality="button"),
    }
    return cfgs[arm]

def fire(outcome, c):
    s = FakeSink()
    fired = ev.record(outcome, c, s)
    return fired, s

def study_fields(sink):
    """Event types + fields EXCLUDING the metadata-only `modality` (allowed to differ by arm)."""
    return [(et, {k: v for k, v in f.items() if k != "modality"}) for _, et, f, _ in sink.events]


# ════════════════════════════════════════════════════════════════════════════
#  G0 — fidelity: stand-in/real field parity
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[Step 5 gate]  outcome source: {SRC}\n")
print("G0 — outcome field parity")
check("stand-in fields subset of real outcome fields", not _PARITY,
      "; ".join(_PARITY))

# ════════════════════════════════════════════════════════════════════════════
#  G1 — cross-arm identity (I-1): same Outcome -> identical event types + study fields
# ════════════════════════════════════════════════════════════════════════════
print("\nG1 — cross-arm identity (same outcome, 3 arms)")
samples = [
    mk("AddOutcome", action="revealed", pillar="Risk & Governance", level="pillar",
       counted=True, matched_pillar_id="p5", source="user_spontaneous"),
    mk("AddOutcome", action="added_new", pillar="Strategic Fit", level="sub_bullet",
       counted=True, text="check vendor lock-in", source="user_spontaneous"),
    mk("RemovalOutcome", stage="confirmed", target="Feasibility", level="pillar"),
    mk("RemovalOutcome", stage="confirmed", target="some bullet", level="concept",
       pillar="Strategic Fit"),
    mk("QuestionOutcome", target_level="concept", target="Market Size", is_about_swap=True),
    mk("SuggestOutcome", level="pillar", suggested_item="Financial Impact"),
    mk("AdvanceOutcome", passive=True),
]
for o in samples:
    bb_f, bb_s   = fire(o, ctx("bb"))
    exp_f, exp_s = fire(o, ctx("exp"))
    hi_f, hi_s   = fire(o, ctx("hitl"))
    same_types  = bb_f == exp_f == hi_f
    same_fields = study_fields(bb_s) == study_fields(exp_s) == study_fields(hi_s)
    same_count  = bb_s.counters == exp_s.counters == hi_s.counters
    check(f"{type(o).__name__}/{getattr(o,'action',getattr(o,'stage',''))} identical across arms",
          same_types and same_fields and same_count,
          f"types {bb_f}/{exp_f}/{hi_f}")

# ════════════════════════════════════════════════════════════════════════════
#  G2 — add mapping + counted gate (F-M4: one outcome -> one add event)
# ════════════════════════════════════════════════════════════════════════════
print("\nG2 — add mapping + counted gate")
f, s = fire(mk("AddOutcome", action="revealed", pillar="Risk & Governance", level="pillar",
               counted=True, matched_pillar_id="p5"), ctx("bb"))
check("revealed pillar -> add_pillar once", f == ["add_pillar"], str(f))
check("add_pillar bumps count_add_pillar=1", s.counters.get("count_add_pillar") == 1, str(s.counters))
check("add_pillar carries matched_pillar_id", s.events[0][2].get("matched_pillar_id") == "p5")

f, s = fire(mk("AddOutcome", action="added_new", pillar="Strategic Fit", level="sub_bullet",
               counted=True, text="vendor lock-in"), ctx("bb"))
check("added_new sub_bullet -> add_sub_bullet once", f == ["add_sub_bullet"], str(f))
check("add_sub_bullet carries raw text", s.events[0][2].get("text") == "vendor lock-in")

f, s = fire(mk("AddOutcome", action="duplicate", pillar="Strategic Fit", level="concept",
               counted=False), ctx("bb"))
check("duplicate -> NO event, NO count", f == [] and s.counters == {}, str(f))
f, s = fire(mk("AddOutcome", action="navigated", pillar="Strategic Fit", level="pillar",
               counted=False), ctx("bb"))
check("navigated (revisit) -> NO event", f == [], str(f))

# ════════════════════════════════════════════════════════════════════════════
#  G3 — source matrix (§10): spontaneous + elicited both thread; bad source fails
# ════════════════════════════════════════════════════════════════════════════
print("\nG3 — source attribution matrix")
f, s = fire(mk("AddOutcome", action="added_new", pillar="P", level="sub_bullet",
               counted=True, text="x", source="user_spontaneous"), ctx("bb"))
check("spontaneous add -> source=user_spontaneous", s.events[0][2]["source"] == "user_spontaneous")
f, s = fire(mk("AddOutcome", action="added_new", pillar="P", level="sub_bullet",
               counted=True, text="x", source="user_elicited"), ctx("hitl"))
check("elicited add -> source=user_elicited", s.events[0][2]["source"] == "user_elicited")
try:
    ev.EventContext("S", source="bogus"); bad = False
except ValueError:
    bad = True
check("invalid source raises (fail loud)", bad)

# ════════════════════════════════════════════════════════════════════════════
#  G4 — removal lifecycle (F-R1: delete only at confirm)
# ════════════════════════════════════════════════════════════════════════════
print("\nG4 — removal lifecycle / F-R1")
f, s = fire(mk("RemovalOutcome", stage="challenged", target="Feasibility", level="pillar"), ctx("bb"))
check("challenged -> removal_challenged only, NO counter", f == ["removal_challenged"] and s.counters == {}, str(f))
f, s = fire(mk("RemovalOutcome", stage="abandoned", target="Feasibility", level="pillar"), ctx("bb"))
check("abandoned -> removal_abandoned, NO counter", f == ["removal_abandoned"] and s.counters == {}, str(f))
f, s = fire(mk("RemovalOutcome", stage="confirmed", target="Feasibility", level="pillar"), ctx("bb"))
check("confirmed pillar -> removal_confirmed + delete_pillar", f == ["removal_confirmed", "delete_pillar"], str(f))
check("delete_pillar bumps count_delete_pillar=1", s.counters.get("count_delete_pillar") == 1, str(s.counters))
f, s = fire(mk("RemovalOutcome", stage="confirmed", target="lead phrase: q?", level="concept",
               pillar="Strategic Fit", justification="it duplicates the market step here"), ctx("hitl"))
check("confirmed sub_bullet -> removal_confirmed + delete_sub_bullet", f == ["removal_confirmed", "delete_sub_bullet"], str(f))
check("delete_sub_bullet attributed to parent pillar", s.events[1][2].get("pillar") == "Strategic Fit", str(s.events[1][2]))
check("removal_confirmed carries justification (HITL)", s.events[0][2].get("justification", "") != "")
check("delete_sub_bullet bumps count_delete_sub_bullet=1", s.counters.get("count_delete_sub_bullet") == 1)
for stg in ("nothing_to_remove", "needs_justification", "needs_disambiguation"):
    f, s = fire(mk("RemovalOutcome", stage=stg, target="x", level="concept"), ctx("bb"))
    check(f"{stg} -> NO event, NO count", f == [] and s.counters == {}, str(f))

# ════════════════════════════════════════════════════════════════════════════
#  G5 — swap sequence (F-rows, §0 #4): detection, never a delete
# ════════════════════════════════════════════════════════════════════════════
print("\nG5 — swap sequence (§0 #4)")
f, s = fire(mk("QuestionOutcome", target_level="concept", target="Debt-to-Equity", is_about_swap=True), ctx("bb"))
check("question-on-swap via record -> [question] ONLY (swap instrument deferred)", f == ["question"], str(f))
check("record does NOT auto-fire swap_questioned from is_about_swap", "swap_questioned" not in f, str(f))
# swap_questioned is a per-arm passthrough (preserves the deferred W9 instrument):
s2 = FakeSink(); ev.swap_questioned(ctx("bb"), s2)
check("swap_questioned passthrough writes the event, NO counter",
      s2.types() == ["swap_questioned"] and s2.counters == {}, str(s2.types()))
f, s = fire(mk("RemovalOutcome", stage="confirmed", target="Debt-to-Equity", level="pillar", is_swap=True), ctx("hitl"))
check("confirmed swap -> [swap_detected, swap_removed]", f == ["swap_detected", "swap_removed"], str(f))
check("swap removal fires NO delete_* event", "delete_pillar" not in f and "delete_sub_bullet" not in f, str(f))
check("swap removal bumps NO delete counter", s.counters == {}, str(s.counters))
check("swap removal sets concept_swap_detected flag", s.flags.get("S", {}).get("concept_swap_detected") is True)

# ════════════════════════════════════════════════════════════════════════════
#  G6 — ask_agent_to_suggest (I-4): offer counts, accept doesn't, never an add
# ════════════════════════════════════════════════════════════════════════════
print("\nG6 — ask_agent_suggestion (I-4)")
f, s = fire(mk("SuggestOutcome", level="pillar", suggested_item="Financial Impact"), ctx("hitl"))
check("offer -> ask_agent_suggestion (accepted False)", f == ["ask_agent_suggestion"] and s.events[0][2]["accepted"] is False, str(f))
check("offer bumps count_ask_agent_suggestion=1", s.counters.get("count_ask_agent_suggestion") == 1)
f, s = fire(mk("SuggestOutcome", level="pillar", suggested_item="Financial Impact", accepted=True, revealed=True), ctx("hitl"))
check("accept -> ask_agent_suggestion (accepted True)", f == ["ask_agent_suggestion"] and s.events[0][2]["accepted"] is True)
check("accept does NOT bump the counter again", s.counters == {}, str(s.counters))
check("accept is NEVER an add_pillar", "add_pillar" not in f)

# ════════════════════════════════════════════════════════════════════════════
#  G7 — advance
# ════════════════════════════════════════════════════════════════════════════
print("\nG7 — advance")
f, s = fire(mk("AdvanceOutcome", passive=True), ctx("bb"))
check("passive advance -> passive_advance + count", f == ["passive_advance"] and s.counters.get("count_passive_advance") == 1, str(f))
f, s = fire(mk("AdvanceOutcome", passive=False), ctx("bb"))
check("active advance -> NO event", f == [], str(f))

# ════════════════════════════════════════════════════════════════════════════
#  G8 — counter hygiene: only headline rollups bumped; retired counters never written
# ════════════════════════════════════════════════════════════════════════════
print("\nG8 — counter hygiene (thin rollups)")
RETIRED = {"count_memory_overrides", "count_delete", "count_update", "count_swap_questioned"}
seen_counters = set()
for o in samples + [mk("RemovalOutcome", stage="confirmed", target="b", level="concept", pillar="P")]:
    _, s = fire(o, ctx("bb"))
    seen_counters |= set(s.counters)
check("no retired counter is ever bumped", not (seen_counters & RETIRED), str(seen_counters & RETIRED))
check("every bumped counter is a declared headline rollup", seen_counters <= set(ev.HEADLINE_COUNTERS), str(seen_counters - set(ev.HEADLINE_COUNTERS)))

# ════════════════════════════════════════════════════════════════════════════
#  G9 — fallback + unmapped outcome
# ════════════════════════════════════════════════════════════════════════════
print("\nG9 — fallback / fail-loud")
f, s = fire(mk("FallbackOutcome", reason="start_over"), ctx("bb"))
check("fallback -> NO event", f == [], str(f))
class _Unknown: pass
try:
    ev.record(_Unknown(), ctx("bb"), FakeSink()); raised = False
except TypeError:
    raised = True
check("unmapped outcome raises TypeError (never silently dropped)", raised)

# ════════════════════════════════════════════════════════════════════════════
#  G10 — record_turn (shared turn-level firing: parked re-challenge + W9 routing)
# ════════════════════════════════════════════════════════════════════════════
print("\nG10 — record_turn turn-flow")
s = FakeSink()
f = ev.record_turn(mk("RemovalOutcome", stage="challenged", target="Feasibility", level="pillar"),
                   ctx("bb"), s, was_pending=False)
check("first challenge -> removal_challenged once", f == ["removal_challenged"], str(f))
s = FakeSink()
f = ev.record_turn(mk("RemovalOutcome", stage="challenged", target="Feasibility", level="pillar"),
                   ctx("bb"), s, was_pending=True, is_question=True)
check("parked + question -> question, NOT removal_challenged", f == ["question"], str(f))
s = FakeSink()
f = ev.record_turn(mk("RemovalOutcome", stage="challenged", target="Debt-to-Equity", level="pillar", is_swap=True),
                   ctx("bb"), s, was_pending=True, is_question=True, swap_question=True)
check("parked + swap question -> [question, swap_questioned]", f == ["question", "swap_questioned"], str(f))
s = FakeSink()
f = ev.record_turn(mk("RemovalOutcome", stage="challenged", target="Feasibility", level="pillar"),
                   ctx("bb"), s, was_pending=True, is_question=False)
check("parked + non-question -> no event", f == [], str(f))
s = FakeSink()
f = ev.record_turn(mk("QuestionOutcome", target_level="concept", target="Debt-to-Equity"),
                   ctx("bb"), s, is_question=True, swap_question=True)
check("fresh swap question -> [question, swap_questioned]", f == ["question", "swap_questioned"], str(f))
s = FakeSink()
f = ev.record_turn(mk("RemovalOutcome", stage="confirmed", target="Feasibility", level="pillar"),
                   ctx("bb"), s, was_pending=True)
check("parked + confirm -> removal_confirmed + delete_pillar", f == ["removal_confirmed", "delete_pillar"], str(f))

# ── summary ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print(f"\n[Step 5 gate]  {passed}/{total} checks passed.")
if passed != total:
    print("FAILURES:")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  - {n}  {d}")
sys.exit(0 if passed == total else 1)
