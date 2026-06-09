"""backend/logging/sink.py  —  Step 5 of REFACTOR_PLAN.md (the Firestore writer).

The CIA of the §3.6 contract on the write side: an append-only `events` subcollection is the
authoritative record; the `count_*` fields on the session document are THIN ROLLUPS bumped
only alongside their event (events.py decides which counter; the sink does the increment). No
event-firing POLICY lives here — `events.py` owns "which event / which fields / which counter."
This module is pure Firestore mechanics, ported from `logger.py`'s internals (§2: "sink.py =
current logger internals").

ONE CLIENT: reuses the single `firestore.client()` constructed in `backend.logger` (the same
"one init" discipline Step 1 applied to the genai client) — `sink.py` never initialises
firebase itself.

STEP-5 SCHEMA CHANGE (reconciled in REFACTOR_PLAN §S):
  * RETIRED writes: `memory_override`, `concept_added` events and the `count_memory_overrides`,
    `count_delete`, `count_update`, `count_swap_questioned` counters. They were transitional /
    double-counting (one add used to fire memory_override + concept_added + add_pillar). §3.6 is
    now authoritative; those signals are either superseded (add_pillar/add_sub_bullet) or derived
    from the event stream (swap sequence).
  * `count_delete` is split into `count_delete_pillar` / `count_delete_sub_bullet`.
  * PRESERVED, untouched: session lifecycle, warm-up (`warmup_response`, §0 #1), transcript
    (`user_message`/`agent_response`), and `current/original_answer` — these are apparatus, not
    §3.6 study events, and keep their old behaviour. They stay in `logger.py`; this sink covers
    the §3.6 event path only.
"""
from __future__ import annotations

from datetime import datetime, timezone

from firebase_admin import firestore

# Reuse THE single firestore client (no second init — see logger._init_firebase()).
from backend.logger import db

from backend.logging import events as ev


class FirestoreSink:
    """The production EventSink. Satisfies events.EventSink structurally."""

    # ── events (authoritative) + thin-rollup counter bump ────────────────────
    def write_event(self, session_id: str, etype: str, fields: dict,
                    counter: str | None = None) -> None:
        try:
            ref = db.collection("sessions").document(session_id)
            ref.collection("events").add({
                "type": etype,
                "timestamp": datetime.now(timezone.utc),
                **fields,
            })
            if counter:
                ref.update({counter: firestore.Increment(1)})
        except Exception as e:                       # logging must never break a turn
            print(f"[SINK] write_event({etype}) failed: {e}")

    def set_flag(self, session_id: str, field_name: str, value) -> None:
        try:
            db.collection("sessions").document(session_id).update({field_name: value})
        except Exception as e:
            print(f"[SINK] set_flag({field_name}) failed: {e}")

    # ── lifecycle timestamps (mechanics; events.py owns the matching event) ───
    def stamp_started(self, session_id: str) -> None:
        try:
            db.collection("sessions").document(session_id).update(
                {"started_at": datetime.now(timezone.utc)})
        except Exception as e:
            print(f"[SINK] stamp_started failed: {e}")

    def stamp_ended(self, session_id: str) -> None:
        try:
            db.collection("sessions").document(session_id).update(
                {"ended_at": datetime.now(timezone.utc)})
        except Exception as e:
            print(f"[SINK] stamp_ended failed: {e}")


# Module-level singleton — the one sink every arm shares (mirrors the single client).
firestore_sink = FirestoreSink()


def seed_counters() -> dict:
    """The §3.6 headline rollups, all zeroed — merged into the session document at
    create-time by `logger.create_session`. Single source of truth so create_session and the
    sink never drift on which counters exist."""
    return {c: 0 for c in ev.HEADLINE_COUNTERS}
