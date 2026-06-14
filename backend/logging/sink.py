from __future__ import annotations

from datetime import datetime, timezone
from firebase_admin import firestore

from backend.logger import db
from backend.logging import events as ev

class FirestoreSink:
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
        except Exception as e:
            print(f"[SINK] write_event({etype}) failed: {e}")


    def set_flag(self, session_id: str, field_name: str, value) -> None:
        try:
            db.collection("sessions").document(session_id).update({field_name: value})
        except Exception as e:
            print(f"[SINK] set_flag({field_name}) failed: {e}")


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


firestore_sink = FirestoreSink()


def seed_counters() -> dict:
    return {c: 0 for c in ev.HEADLINE_COUNTERS}
