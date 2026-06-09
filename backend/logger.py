import os
import uuid
from datetime import datetime, timezone
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

# ── Firebase init (runs once) ──────────────────────────────────────────────
def _init_firebase():
    if not firebase_admin._apps:
        key_path = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

_init_firebase()
db = firestore.client()

# §3.6 headline counters live in ONE place (events.py). logger -> events is cycle-free
# (events imports nothing heavy; sink imports logger for `db`, never the reverse).
from backend.logging import events as _ev
_HEADLINE_COUNTERS = _ev.HEADLINE_COUNTERS


# ── Session ────────────────────────────────────────────────────────────────
def create_session(user_id: str = "anonymous", agent_type: str = "unknown") -> str:
    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set({
        "user_id": user_id,
        "agent_type": agent_type,
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "ended_at": None,
        "current_answer": None,
        "original_answer": None,
        # ── warm-up ────────────────────────────────────────────────────
        # Change log: 2026-05-01 — added for warm-up phase.
        # Stores raw participant response for optional post-hoc analysis.
        "warmup_response": None,
        # ── transcript / apparatus counters (PRESERVED — not §3.6 study events) ──
        "count_user_messages": 0,
        "count_agent_responses": 0,
        "count_answer_updates": 0,
        # ── §3.6 headline rollups (Step 5). Single source of truth =
        #    backend.logging.events.HEADLINE_COUNTERS. Events are authoritative;
        #    these are thin caches recomputed from events in analysis. RETIRED in
        #    Step 5: count_memory_overrides, count_delete, count_update,
        #    count_swap_questioned (superseded by §3.6 / fully event-derived). ──
        **{c: 0 for c in _HEADLINE_COUNTERS},
        # ── concept swap (preserved apparatus flags) ──
        "concept_swap_presented": False,
        "concept_swap_detected": False,
        # ── HITL-specific (preserved) ──
        "concepts_approved": [],
        "concepts_rejected": [],
    })
    return session_id


def save_original_case(session_id: str, case_text: str) -> None:
    db.collection("sessions").document(session_id).update({
        "original_case": case_text,
    })


def get_original_case(session_id: str) -> str | None:
    try:
        doc = db.collection("sessions").document(session_id).get()
        if doc.exists:
            return doc.to_dict().get("original_case")
    except Exception as e:
        print(f"[CASE] failed to fetch original case: {e}")
    return None


def end_session(session_id: str) -> None:
    try:
        db.collection("sessions").document(session_id).update({
            "ended_at": datetime.now(timezone.utc),
        })
        print(f"[SESSION] ended_at stamped for session: {session_id}")
    except Exception as e:
        print(f"[SESSION] failed to stamp ended_at: {e}")

def stamp_started_at(session_id: str) -> None:
    try:
        db.collection("sessions").document(session_id).update({
            "started_at": datetime.now(timezone.utc),
        })
        print(f"[SESSION] started_at stamped for session: {session_id}")
    except Exception as e:
        print(f"[SESSION] failed to stamp started_at: {e}")

# ── Counter map ────────────────────────────────────────────────────────────
# Step 5: trimmed to the PRESERVED transcript/apparatus counters only. All §3.6
# study-event counters (add_pillar/add_sub_bullet/question/delete_*/ask_agent_
# suggestion/passive_advance/interruption) are now owned by the shared sink
# (backend.logging.sink), bumped via events.COUNTER_FOR. interruption stays here
# because log_interruption() is a preserved apparatus helper (§3.9 phase machine).
_COUNTER_MAP = {
    "user_message":    "count_user_messages",
    "agent_response":  "count_agent_responses",
    "interruption":    "count_interruptions",
}


# ── Event helpers ──────────────────────────────────────────────────────────
def _log_event(session_id: str, event_type: str, payload: dict) -> None:
    session_ref = db.collection("sessions").document(session_id)
    session_ref.collection("events").add({
        "type": event_type,
        "timestamp": datetime.now(timezone.utc),
        **payload,
    })
    counter_field = _COUNTER_MAP.get(event_type)
    if counter_field:
        session_ref.update({
            counter_field: firestore.Increment(1)
        })


def update_answer(session_id: str, answer: str) -> None:
    session_ref = db.collection("sessions").document(session_id)
    try:
        doc = session_ref.get()
        is_first_write = doc.to_dict().get("original_answer") is None
        update_payload = {
            "current_answer": answer,
            "count_answer_updates": firestore.Increment(1),
        }
        if is_first_write:
            update_payload["original_answer"] = answer
        session_ref.update(update_payload)
        print(f"[ANSWER] stored, first_write={is_first_write}")
    except Exception as e:
        print(f"[ANSWER] failed: {e}")


# ── Public logging methods ─────────────────────────────────────────────────
def log_warmup_response(session_id: str, response: str) -> None:
    """
    Log raw warm-up response to Firestore.
    Stored as a session field for optional post-hoc analysis.
    Change log: 2026-05-01 — added for warm-up phase.
    """
    try:
        db.collection("sessions").document(session_id).update({
            "warmup_response": response,
        })
        _log_event(session_id, "warmup_response", {
            "response": response,
        })
        print(f"[WARMUP] response logged for session={session_id}")
    except Exception as e:
        print(f"[WARMUP] failed to log response: {e}")


# ── Swap / framework / concept-add logging: RETIRED in Step 5 ────────────────
# These moved to the shared §3.6 layer:
#   concept_swap_presented  -> backend.logging.events.swap_presented()
#   concept_swap_detected   -> backend.logging.events.swap_detected() (+ swap_removed
#                              on a confirmed swap removal, via events.record)
#   concept_added/memory_override -> superseded by add_pillar / add_sub_bullet
#   log_framework_switched  -> dead since Step 0.5 (multi-framework subsystem removed)
# Kept out of logger.py so the §3.6 schema lives in exactly one module (I-1).


def log_user_message(session_id: str, message: str) -> None:
    _log_event(session_id, "user_message", {"message": message})


def log_agent_response(session_id: str, response: str) -> None:
    _log_event(session_id, "agent_response", {"response": response})


def log_interruption(session_id: str, context: str = "") -> None:
    _log_event(session_id, "interruption", {"context": context})

# ── Agency interaction events: RETIRED in Step 5 ─────────────────────────────
# log_memory_override, log_interaction_event and the thin helpers
# (log_question / log_add_pillar / log_add_sub_bullet / log_delete /
# log_swap_questioned) are gone. Their job — turning a user agency act into a
# §3.6 event + thin-rollup counter — now belongs to the ONE shared firing point
# backend.logging.events.record(outcome, ctx, sink), driven by the Step-4
# handler Outcome. This is the I-1 relocation: an event can no longer originate
# in per-agent code, so it can no longer differ by arm.
