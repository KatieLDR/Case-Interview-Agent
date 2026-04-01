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


# ── Session ────────────────────────────────────────────────────────────────
def create_session(user_id: str = "anonymous", agent_type: str = "unknown") -> str:
    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set({
        "user_id": user_id,
        "agent_type": agent_type,
        "started_at": datetime.now(timezone.utc),
        "ended_at": None,
        "current_answer": None,
        "original_answer": None,
        # ── counters ──
        "count_user_messages": 0,
        "count_agent_responses": 0,
        "count_interruptions": 0,
        "count_memory_overrides": 0,
        "count_answer_updates": 0,
        # ── concept swap ──
        "concept_swap_presented": False,
        "concept_swap_detected": False,
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


# ── Counter map ────────────────────────────────────────────────────────────
_COUNTER_MAP = {
    "user_message":    "count_user_messages",
    "agent_response":  "count_agent_responses",
    "interruption":    "count_interruptions",
    "memory_override": "count_memory_overrides",
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
def log_concept_swap_presented(session_id: str) -> None:
    db.collection("sessions").document(session_id).update({
        "concept_swap_presented": True,
    })
    _log_event(session_id, "concept_swap_presented", {})


def log_concept_swap_detected(session_id: str) -> None:
    db.collection("sessions").document(session_id).update({
        "concept_swap_detected": True,
    })
    _log_event(session_id, "concept_swap_detected", {})


def log_framework_switched(
    session_id: str,
    from_framework: str,
    to_framework: str,
    switch_index: int,
) -> None:
    try:
        db.collection("sessions").document(session_id).update({
            "framework_switched": True,
        })
        _log_event(session_id, "framework_switched", {
            "from_framework": from_framework,
            "to_framework":   to_framework,
            "at_index":       switch_index,
        })
        print(f"[FRAMEWORK SWITCH] logged: {from_framework} → {to_framework} "
              f"at index={switch_index}")
    except Exception as e:
        print(f"[FRAMEWORK SWITCH] failed to log: {e}")


# Change log: 2026-04-01 — added log_concept_added().
# Logs a user-initiated concept addition as both a research event and
# an override — increments count_memory_overrides since adding a concept
# is a user override of the original framework structure.
def log_concept_added(session_id: str, concept_name: str) -> None:
    """
    Log a user-initiated concept addition to Firestore.

    Increments count_memory_overrides (via _COUNTER_MAP on memory_override
    event type) since concept addition is an override of the original
    framework structure — consistent with concept_excluded and framework_switch.

    Also writes a dedicated concept_added event for granular research analysis.
    """
    try:
        # Log as memory_override to increment count_memory_overrides
        _log_event(session_id, "memory_override", {
            "old_context": "framework without user concept",
            "new_context": f"user added: {concept_name}",
        })
        # Log dedicated event for granular analysis
        _log_event(session_id, "concept_added", {
            "concept_name": concept_name,
        })
        print(f"[CONCEPT ADDED] logged: '{concept_name}' for session={session_id}")
    except Exception as e:
        print(f"[CONCEPT ADDED] failed to log: {e}")


def log_user_message(session_id: str, message: str) -> None:
    _log_event(session_id, "user_message", {"message": message})


def log_agent_response(session_id: str, response: str) -> None:
    _log_event(session_id, "agent_response", {"response": response})


def log_interruption(session_id: str, context: str = "") -> None:
    _log_event(session_id, "interruption", {"context": context})


def log_memory_override(session_id: str, old_context: str, new_context: str) -> None:
    _log_event(session_id, "memory_override", {
        "old_context": old_context,
        "new_context": new_context,
    })