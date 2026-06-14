from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

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

# Swap sequence
SWAP_PRESENTED        = "swap_presented"
SWAP_QUESTIONED       = "swap_questioned"
SWAP_DETECTED         = "swap_detected"
SWAP_REMOVED          = "swap_removed"

# Lifecycle
SESSION_STARTED       = "session_started"
SESSION_ENDED         = "session_ended"

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
HEADLINE_COUNTERS = tuple(sorted(set(COUNTER_FOR.values())))
VALID_SOURCES = ("user_spontaneous", "user_elicited")

@runtime_checkable
class EventSink(Protocol):
    def write_event(self, session_id: str, etype: str, fields: dict,
                    counter: str | None = None) -> None: ...
    def set_flag(self, session_id: str, field_name: str, value) -> None: ...
    def stamp_started(self, session_id: str) -> None: ...
    def stamp_ended(self, session_id: str) -> None: ...

@dataclass
class EventContext:
    session_id: str
    source: str = "user_spontaneous"
    modality: str = "text"
    agent_type: str = "unknown"
    wrong_concept: str | None = None


    def __post_init__(self):
        if self.source not in VALID_SOURCES:
            raise ValueError(f"[events] invalid source={self.source!r}; "
                             f"expected one of {VALID_SOURCES}")


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
        return []
    raise TypeError(f"[events.record] unmapped outcome type {kind!r} — a new Outcome "
                    f"must be given an explicit §3.6 mapping (fail loud, never drop).")


def record_turn(outcome, ctx: EventContext, sink: EventSink, *,
                was_pending: bool = False, is_question: bool = False,
                swap_question: bool = False) -> list[str]:
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
        return fired
    fired = record(outcome, ctx, sink)
    if kind == "QuestionOutcome" and swap_question:
        sink.write_event(ctx.session_id, SWAP_QUESTIONED, {},
                         counter=COUNTER_FOR[SWAP_QUESTIONED])
        fired.append(SWAP_QUESTIONED)
    return fired


def _fire(sink: EventSink, ctx: EventContext, etype: str, fields: dict) -> str:
    sink.write_event(ctx.session_id, etype, fields, counter=COUNTER_FOR.get(etype))
    return etype


def _record_add(o, ctx, sink) -> list[str]:
    if not getattr(o, "counted", False):
        return []
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
    raise ValueError(f"[events] counted add with unexpected level={o.level!r}")


def _record_removal(o, ctx, sink) -> list[str]:
    stage = o.stage
    if stage == "challenged":
        return [_fire(sink, ctx, REMOVAL_CHALLENGED,
                      {"target": o.target, "level": o.level})]
    if stage == "abandoned":
        return [_fire(sink, ctx, REMOVAL_ABANDONED,
                      {"target": o.target, "level": o.level})]
    if stage == "confirmed":
        if getattr(o, "is_swap", False):
            fired = [_fire(sink, ctx, SWAP_DETECTED, {}),
                     _fire(sink, ctx, SWAP_REMOVED, {})]
            sink.set_flag(ctx.session_id, "concept_swap_detected", True)
            return fired
        fired = [_fire(sink, ctx, REMOVAL_CONFIRMED, {
            "target": o.target, "level": o.level,
            "justification": getattr(o, "justification", None),
        })]
        if o.level == "pillar":
            fired.append(_fire(sink, ctx, DELETE_PILLAR, {"pillar": o.target}))
        else:                                  # concept / sub_bullet
            fired.append(_fire(sink, ctx, DELETE_SUB_BULLET,
                               {"pillar": _parent_of(o), "text": o.target}))
        return fired
    return []


def _parent_of(o):
    return getattr(o, "pillar", None)


def _record_question(o, ctx, sink) -> list[str]:
    return [_fire(sink, ctx, QUESTION,
                  {"target_level": o.target_level, "target": o.target})]


def _record_suggest(o, ctx, sink) -> list[str]:
    accepted = getattr(o, "accepted", False)
    counter = None if accepted else COUNTER_FOR[ASK_AGENT_SUGGESTION]
    sink.write_event(ctx.session_id, ASK_AGENT_SUGGESTION, {
        "level": o.level, "suggested_item": o.suggested_item, "accepted": accepted,
    }, counter=counter)
    return [ASK_AGENT_SUGGESTION]


def _record_advance(o, ctx, sink) -> list[str]:
    if getattr(o, "elicited", False):
        return [_fire(sink, ctx, ASK_AGENT_SUGGESTION,
                      {"level": "pillar", "suggested_item": None, "accepted": False})]
    if getattr(o, "passive", False):
        return [_fire(sink, ctx, PASSIVE_ADVANCE, {})]
    return []


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
    sink.set_flag(ctx.session_id, "concept_swap_detected", True)
    sink.write_event(ctx.session_id, SWAP_DETECTED, {})


def swap_questioned(ctx: EventContext, sink: EventSink) -> None:
    sink.write_event(ctx.session_id, SWAP_QUESTIONED, {},
                     counter=COUNTER_FOR[SWAP_QUESTIONED])


def question(ctx: EventContext, sink: EventSink, *,
             target_level: str | None = None, target: str | None = None) -> None:
    sink.write_event(ctx.session_id, QUESTION,
                     {"target_level": target_level, "target": target},
                     counter=COUNTER_FOR[QUESTION])


def interruption(ctx: EventContext, sink: EventSink, *, where: str = "") -> None:
    sink.write_event(ctx.session_id, INTERRUPTION, {"where": where},
                     counter=COUNTER_FOR[INTERRUPTION])
