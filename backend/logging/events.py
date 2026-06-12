"""backend/logging/events.py  —  Step 5 of REFACTOR_PLAN.md (logging relocation).

THE ONE SHARED FIRING POINT (I-1). Every research event in the study originates here,
driven by the Step-4 Outcome objects — never from per-agent code. Before this module the
three agents each fired their own `log_*` calls scattered through their render methods
(BlackBox `_render_add`/`_render_removal`, EXP, HITL), which is exactly the cross-condition
confound the validity contract forbids: if an event can originate in one arm only, or fire
on a different code path per arm, the study DVs are not comparable. `record()` collapses all
of that to a single arm-agnostic mapping from an Outcome to its §3.6 event(s).

RECONCILIATION WITH §3.5 / handlers.D-H1 (recorded in REFACTOR_PLAN §S, Step 5):
  §3.5 says "handlers fire the §3.6 event"; handlers.py D-H1 keeps handlers PURE (no
  Firestore) and says "the log_* call moves into the handler in Step 5." Reconciled to honour
  BOTH handler purity AND I-1 ("logging fires from the shared layer, never per-agent"):
  the firing LOGIC (which event, which fields, which counter) lives in this ONE shared module,
  driven by the Outcome the handler returns. The agent's only remaining logging act is a single
  `events.record(outcome, ctx)` call — mechanical and identical across arms, so it is not a
  per-agent behaviour. The §3.6 schema is now known in exactly one place.

DATA MODEL (§3.6): events authoritative, counters thin rollups. A counter is bumped ONLY
alongside writing its event (the sink does the bump when `record` hands it a counter name).
No wide column grids; every cross-tab (`swap_outcome`, `removal_abandon_rate`,
`spontaneous_contributions`/`elicited_contributions`) is recomputed from the event stream in
analysis, never stored. Runtime signals (`match_type`, pillar-state, advance-readiness) are
NOT logged (I-5).

ATTRIBUTION (I-4): every contribution event carries `source` (`user_spontaneous |
user_elicited`). `record` records whatever `source` the Outcome / EventContext carries — it
does NOT decide it. `user_elicited` is set by the arm when the contribution arrives while a
proactive prompt is active (HITL `awaiting_user_suggestion`); this is HITL's treatment effect,
EXPECTED to differ by arm (high in HITL, ~0 elsewhere) and therefore NOT a confound. Agent-named
content is an `ask_agent_suggestion`, never an add.

SWAP (§0 #4 preservation wall): the swap MECHANISM (ConceptSwap Direction B/C detection,
injection) is untouched. Step 5 only unifies the swap LOGGING vocabulary onto the §3.6 names
(`swap_presented | swap_questioned | swap_detected | swap_removed`). A confirmed removal that
targets the swap is DETECTION (`swap_detected` + `swap_removed`), NEVER a `delete_*` event.

PURITY: this module imports NOTHING from `handlers`/`llm`/`genai` — it dispatches on the
Outcome's class NAME (so the gate can drive it with no API key and no Firestore) and writes
through an injected `EventSink`. The real sink is `backend.logging.sink.FirestoreSink`; the
Step-5 gate passes a `FakeSink`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ════════════════════════════════════════════════════════════════════════════
#  §3.6 EVENT VOCABULARY (the single source of truth for event names + counters)
# ════════════════════════════════════════════════════════════════════════════
# Contribution / interaction events
ADD_PILLAR            = "add_pillar"
ADD_SUB_BULLET        = "add_sub_bullet"
QUESTION              = "question"
ASK_AGENT_SUGGESTION  = "ask_agent_suggestion"
REMOVAL_CHALLENGED    = "removal_challenged"
REMOVAL_CONFIRMED     = "removal_confirmed"
REMOVAL_ABANDONED     = "removal_abandoned"
DELETE_PILLAR         = "delete_pillar"
DELETE_SUB_BULLET     = "delete_sub_bullet"
PASSIVE_ADVANCE       = "passive_advance"
INTERRUPTION          = "interruption"
# Swap sequence (mechanism preserved; names unified onto §3.6)
SWAP_PRESENTED        = "swap_presented"
SWAP_QUESTIONED       = "swap_questioned"
SWAP_DETECTED         = "swap_detected"
SWAP_REMOVED          = "swap_removed"
# Lifecycle
SESSION_STARTED       = "session_started"
SESSION_ENDED         = "session_ended"

# Event -> headline counter field (§3.6 "thin rollups"). Events not listed here
# carry NO counter and are derived from the event stream in analysis (e.g. the
# whole swap sequence, removal_challenged/confirmed/abandoned -> removal_abandon_rate).
COUNTER_FOR = {
    ADD_PILLAR:           "count_add_pillar",
    ADD_SUB_BULLET:       "count_add_sub_bullet",
    QUESTION:             "count_questions",
    ASK_AGENT_SUGGESTION: "count_ask_agent_suggestion",
    DELETE_PILLAR:        "count_delete_pillar",
    DELETE_SUB_BULLET:    "count_delete_sub_bullet",
    PASSIVE_ADVANCE:      "count_passive_advance",
    INTERRUPTION:         "count_interruptions",
    SWAP_QUESTIONED:      "count_swap_questioned",
}

# The full set of session-document rollups Step 5 maintains (sink.create_session seeds
# these to 0). RETIRED in Step 5 (see §S): count_memory_overrides, count_delete (now split
# pillar/sub_bullet), count_update (no `update` event in the I-2 taxonomy), count_swap_questioned
# (swap is fully event-derived). Transcript/preserved counters (count_user_messages,
# count_agent_responses, count_answer_updates) live on in the sink, untouched.
HEADLINE_COUNTERS = tuple(sorted(set(COUNTER_FOR.values())))

VALID_SOURCES = ("user_spontaneous", "user_elicited")


# ════════════════════════════════════════════════════════════════════════════
#  SINK CONTRACT — events.py writes through this; FirestoreSink / FakeSink impl.
# ════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class EventSink(Protocol):
    def write_event(self, session_id: str, etype: str, fields: dict,
                    counter: str | None = None) -> None: ...
    def set_flag(self, session_id: str, field_name: str, value) -> None: ...
    def stamp_started(self, session_id: str) -> None: ...
    def stamp_ended(self, session_id: str) -> None: ...


# ════════════════════════════════════════════════════════════════════════════
#  EVENT CONTEXT — the per-turn envelope the arm supplies. Carries the attribution
#  (`source`), the interaction modality (button vs text — HITL's affordance), and the
#  session id. NOT part of the §3.6 field schema beyond `source`; `modality` is retained
#  optional metadata (it reflects HITL's button manipulation, not a confound) and never
#  drives a counter or the cross-arm event-type identity the gate asserts.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class EventContext:
    session_id: str
    source: str = "user_spontaneous"          # I-4; arm decides, events records
    modality: str = "text"                    # "text" | "button" (HITL); metadata only
    agent_type: str = "unknown"               # lifecycle / swap_presented payload
    wrong_concept: str | None = None          # swap_presented payload

    def __post_init__(self):
        if self.source not in VALID_SOURCES:
            # fail loud — a bad source value silently corrupts the attribution DV.
            raise ValueError(f"[events] invalid source={self.source!r}; "
                             f"expected one of {VALID_SOURCES}")


# ════════════════════════════════════════════════════════════════════════════
#  record() — the single outcome -> §3.6 event(s) mapper. Returns the list of event
#  types fired (in order) so the gate can assert cross-arm identity without reaching
#  into the sink. Dispatch is by Outcome CLASS NAME (no handlers import).
# ════════════════════════════════════════════════════════════════════════════
def record(outcome, ctx: EventContext, sink: EventSink) -> list[str]:
    kind = type(outcome).__name__
    if kind == "AddOutcome":
        return _record_add(outcome, ctx, sink)
    if kind == "RemovalOutcome":
        return _record_removal(outcome, ctx, sink)
    if kind == "QuestionOutcome":
        return _record_question(outcome, ctx, sink)
    if kind == "SuggestOutcome":
        return _record_suggest(outcome, ctx, sink)
    if kind == "AdvanceOutcome":
        return _record_advance(outcome, ctx, sink)
    if kind == "FallbackOutcome":
        return []                              # no study event for a fallback
    raise TypeError(f"[events.record] unmapped outcome type {kind!r} — a new Outcome "
                    f"must be given an explicit §3.6 mapping (fail loud, never drop).")


def record_turn(outcome, ctx: EventContext, sink: EventSink, *,
                was_pending: bool = False, is_question: bool = False,
                swap_question: bool = False) -> list[str]:
    """The SHARED turn-level firing entry point the arms call (I-1). Wraps `record` with the
    two pieces of turn-flow context the raw Outcome can't carry, both identical across arms:

      * PARKED-REMOVAL RE-CHALLENGE.  `resolve_pending` returns RemovalOutcome(stage=
        "challenged") on EVERY turn a removal stays parked (the "other" branch), not just the
        first. `removal_challenged` must fire ONCE, on the transition INTO parked (was_pending
        False). A later parked turn that is a question is a `question` event (preserving the
        arms' old inline log_question on that path) — never a second removal_challenged.

      * SWAP_QUESTIONED (W9).  Fired here, not inside `record`, because WHICH signal marks a
        question as swap-targeting is the deferred F-S instrument decision; the arm computes
        its own `swap_question` bool (BlackBox: is_injected & !detected & _classify_swap_
        question; the parked path historically used outcome.is_swap) and we only route the
        write. See _record_question.
    """
    kind = type(outcome).__name__
    if kind == "RemovalOutcome" and getattr(outcome, "stage", None) == "challenged" and was_pending:
        fired: list[str] = []
        if is_question:
            fired.append(_fire(sink, ctx, QUESTION,
                               {"target_level": None, "target": getattr(outcome, "target", None)}))
            if swap_question:
                sink.write_event(ctx.session_id, SWAP_QUESTIONED, {},
                                 counter=COUNTER_FOR[SWAP_QUESTIONED])
                fired.append(SWAP_QUESTIONED)
        return fired                            # NOT a second removal_challenged
    fired = record(outcome, ctx, sink)
    if kind == "QuestionOutcome" and swap_question:
        sink.write_event(ctx.session_id, SWAP_QUESTIONED, {},
                         counter=COUNTER_FOR[SWAP_QUESTIONED])
        fired.append(SWAP_QUESTIONED)
    return fired


def _fire(sink: EventSink, ctx: EventContext, etype: str, fields: dict) -> str:
    sink.write_event(ctx.session_id, etype, fields, counter=COUNTER_FOR.get(etype))
    return etype


# ── add / revisit ────────────────────────────────────────────────────────────
def _record_add(o, ctx, sink) -> list[str]:
    """add_pillar | add_sub_bullet, fired IFF the handler counted it (o.counted is the
    authoritative DV flag — duplicate/navigated => counted False => no event, no count).
    `source` rides from the Outcome (the handler threaded it from dispatch); EventContext
    carries it too for arms that prefer to set it there — Outcome wins if present."""
    if not getattr(o, "counted", False):
        return []                              # duplicate / navigated / not-a-contribution
    source = getattr(o, "source", None) or ctx.source
    if source not in VALID_SOURCES:
        raise ValueError(f"[events] invalid add source={source!r}")
    if o.level == "pillar":
        return [_fire(sink, ctx, ADD_PILLAR, {
            "pillar": o.pillar,
            "matched_pillar_id": getattr(o, "matched_pillar_id", None),
            "source": source,
            "modality": ctx.modality,
        })]
    if o.level == "sub_bullet":
        return [_fire(sink, ctx, ADD_SUB_BULLET, {
            "pillar": o.pillar,
            "text": getattr(o, "text", None),
            "source": source,
            "modality": ctx.modality,
        })]
    # counted but neither pillar nor sub_bullet — should not happen; fail loud.
    raise ValueError(f"[events] counted add with unexpected level={o.level!r}")


# ── removal (challenge / confirm / abandon) + delete + swap detection ─────────
def _record_removal(o, ctx, sink) -> list[str]:
    """The removal lifecycle. NOTE the firing is gated on the handler STAGE, not on intent —
    this is the F-R1 fix made permanent: a delete event exists only at `confirmed`, never at
    the first (challenged) turn. Swap target => DETECTION (swap_detected + swap_removed),
    never a delete (§0 #4)."""
    stage = o.stage
    if stage == "challenged":
        return [_fire(sink, ctx, REMOVAL_CHALLENGED,
                      {"target": o.target, "level": o.level})]
    if stage == "abandoned":
        return [_fire(sink, ctx, REMOVAL_ABANDONED,
                      {"target": o.target, "level": o.level})]
    if stage == "confirmed":
        if getattr(o, "is_swap", False):
            # swap caught and removed: detection, never a delete. `swap_detected` may have
            # already fired via the classifier channel (ConceptSwap) — the sink/analysis is
            # idempotent on the swap flags; we still emit both so the sequence is complete
            # for arms whose swap removal is button-driven (HITL force_detected at confirm).
            fired = [_fire(sink, ctx, SWAP_DETECTED, {}),
                     _fire(sink, ctx, SWAP_REMOVED, {})]
            sink.set_flag(ctx.session_id, "concept_swap_detected", True)
            return fired
        fired = [_fire(sink, ctx, REMOVAL_CONFIRMED, {
            "target": o.target, "level": o.level,
            "justification": getattr(o, "justification", None),   # HITL only; None elsewhere
        })]
        if o.level == "pillar":
            fired.append(_fire(sink, ctx, DELETE_PILLAR, {"pillar": o.target}))
        else:                                  # concept / sub_bullet
            fired.append(_fire(sink, ctx, DELETE_SUB_BULLET,
                               {"pillar": _parent_of(o), "text": o.target}))
        return fired
    # nothing_to_remove | needs_justification | needs_disambiguation -> NO event
    # (no challenge, no delete, no count — matches the existence guard contract).
    return []


def _parent_of(o):
    """Parent pillar for a sub_bullet delete. RemovalOutcome doesn't carry the parent
    directly (the PendingAction does); the arm passes it via ctx-free Outcome only when it
    resolved a sub_bullet. We read an optional `pillar` attr if the arm set one, else None
    (analysis can still attribute via the preceding removal_challenged)."""
    return getattr(o, "pillar", None)


# ── question ──────────────────────────────────────────────────────────────────
def _record_question(o, ctx, sink) -> list[str]:
    """Fires `question` for every question turn (preserves the arms' previous
    unconditional log_question).

    NOTE — swap_questioned is deliberately NOT fired here. Which signal draws the
    questioned-vs-detected line (BlackBox's W9 LLM `_classify_swap_question` vs the
    deterministic `is_swap_target`) is an OPEN instrument decision (F-S family),
    explicitly outside the Step-5 migration. To relocate logging WITHOUT changing the
    instrument, each arm keeps computing its own swap-question signal at the firing site
    and writes it through the shared `swap_questioned()` passthrough below — so only the
    event vocabulary unifies, not the detection behaviour. `o.is_about_swap` (set by the
    handler from is_swap_target) is intentionally ignored here; unifying onto it is a
    future instrument change, not a Step-5 relocation."""
    return [_fire(sink, ctx, QUESTION,
                  {"target_level": o.target_level, "target": o.target})]


# ── ask_agent_to_suggest (offer + accept), never an add (I-4) ─────────────────
def _record_suggest(o, ctx, sink) -> list[str]:
    """One `ask_agent_suggestion` event. The OFFER (accepted False) bumps
    count_ask_agent_suggestion; a later ACCEPT (accepted True) writes the acceptance fact with
    NO second bump (the suggestion was offered once). Accepting an agent suggestion NEVER
    produces an add_pillar (I-4: agent-named content is a suggestion, not a contribution)."""
    accepted = getattr(o, "accepted", False)
    counter = None if accepted else COUNTER_FOR[ASK_AGENT_SUGGESTION]
    sink.write_event(ctx.session_id, ASK_AGENT_SUGGESTION, {
        "level": o.level, "suggested_item": o.suggested_item, "accepted": accepted,
    }, counter=counter)
    return [ASK_AGENT_SUGGESTION]


# ── advance ───────────────────────────────────────────────────────────────────
def _record_advance(o, ctx, sink) -> list[str]:
    if getattr(o, "passive", False):
        return [_fire(sink, ctx, PASSIVE_ADVANCE, {})]
    return []


# ════════════════════════════════════════════════════════════════════════════
#  NON-OUTCOME PASSTHROUGHS — §3.6 events that fire at render/lifecycle moments,
#  not from a turn Outcome. Kept here so the §3.6 schema lives in ONE module.
# ════════════════════════════════════════════════════════════════════════════
def session_started(ctx: EventContext, sink: EventSink) -> None:
    sink.stamp_started(ctx.session_id)
    sink.write_event(ctx.session_id, SESSION_STARTED, {"agent_type": ctx.agent_type})


def session_ended(ctx: EventContext, sink: EventSink) -> None:
    sink.stamp_ended(ctx.session_id)
    sink.write_event(ctx.session_id, SESSION_ENDED, {"agent_type": ctx.agent_type})


def swap_presented(ctx: EventContext, sink: EventSink) -> None:
    sink.set_flag(ctx.session_id, "concept_swap_presented", True)
    sink.write_event(ctx.session_id, SWAP_PRESENTED,
                     {"wrong_concept": ctx.wrong_concept, "agent_type": ctx.agent_type})


def swap_detected(ctx: EventContext, sink: EventSink) -> None:
    """Classifier-channel detection (ConceptSwap Direction B/C / force_detected). The
    removal-confirm channel emits its own swap_detected+swap_removed; both are fine — the
    flag write is idempotent and analysis derives the ordinal from first-occurrence."""
    sink.set_flag(ctx.session_id, "concept_swap_detected", True)
    sink.write_event(ctx.session_id, SWAP_DETECTED, {})


def swap_questioned(ctx: EventContext, sink: EventSink) -> None:
    """W9 — a question that lands on the swap. Drawn by the ONE shared classifier
    (C4: base _classify_swap_question, no per-arm override), routed here to unify the
    §3.6 event vocabulary. C4 (Decision 6a): increments count_swap_questioned so
    questioned_kept is a headline rollup, consistent with the §3.6 counter rule."""
    sink.write_event(ctx.session_id, SWAP_QUESTIONED, {},
                     counter=COUNTER_FOR[SWAP_QUESTIONED])


def question(ctx: EventContext, sink: EventSink, *,
             target_level: str | None = None, target: str | None = None) -> None:
    """A `question` event fired from a render-coupled, arm-specific path that has no
    corresponding turn Outcome — e.g. Explainable's revisit-as-Q&A (AddOutcome navigated
    falls through to grounded Q&A) and its doubt/consequence walkthrough branches. These
    firings are legitimately per-arm (they exist only where the arm has a walkthrough), so
    they can't ride the outcome mapping; routing them through this passthrough still unifies
    the §3.6 vocabulary + the count_questions rollup. NOT used for the normal QuestionOutcome
    path, which fires via record/record_turn."""
    sink.write_event(ctx.session_id, QUESTION,
                     {"target_level": target_level, "target": target},
                     counter=COUNTER_FOR[QUESTION])


def interruption(ctx: EventContext, sink: EventSink, *, where: str = "") -> None:
    sink.write_event(ctx.session_id, INTERRUPTION, {"where": where},
                     counter=COUNTER_FOR[INTERRUPTION])
