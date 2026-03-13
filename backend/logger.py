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
    """Create a new session document and return the session_id."""
    session_id = str(uuid.uuid4())
    db.collection("sessions").document(session_id).set({
        "user_id": user_id,
        "agent_type": agent_type,
        "started_at": datetime.now(timezone.utc),
        "ended_at": None,
        "original_case": None,
        # ── counters ──
        "count_user_messages": 0,
        "count_agent_responses": 0,
        "count_interruptions": 0,
        "count_memory_overrides": 0,
    })
    return session_id


def save_original_case(session_id: str, case_text: str) -> None:
    """Store the original case problem on the session document."""
    db.collection("sessions").document(session_id).update({
        "original_case": case_text,
    })


def end_session(session_id: str) -> None:
    """Stamp the session with an end time."""
    db.collection("sessions").document(session_id).update({
        "ended_at": datetime.now(timezone.utc),
    })


# ── Counter map ────────────────────────────────────────────────────────────
_COUNTER_MAP = {
    "user_message":    "count_user_messages",
    "agent_response":  "count_agent_responses",
    "interruption":    "count_interruptions",
    "memory_override": "count_memory_overrides",
}


# ── Event helpers ──────────────────────────────────────────────────────────
def _log_event(session_id: str, event_type: str, payload: dict) -> None:
    """Generic event logger — writes event and increments session counter."""
    session_ref = db.collection("sessions").document(session_id)

    # Write event to subcollection
    session_ref.collection("events").add({
        "type": event_type,
        "timestamp": datetime.now(timezone.utc),
        **payload,
    })

    # Increment counter on session document if applicable
    counter_field = _COUNTER_MAP.get(event_type)
    if counter_field:
        session_ref.update({
            counter_field: firestore.Increment(1)
        })


# ── Public logging methods ─────────────────────────────────────────────────
def log_user_message(session_id: str, message: str) -> None:
    _log_event(session_id, "user_message", {"message": message})


def log_agent_response(session_id: str, response: str) -> None:
    _log_event(session_id, "agent_response", {"response": response})


def log_interruption(session_id: str, context: str = "") -> None:
    """Call this when user sends a new message before agent finishes."""
    _log_event(session_id, "interruption", {"context": context})


def log_memory_override(session_id: str, old_context: str, new_context: str) -> None:
    """Call this when user resets or corrects the conversation context."""
    _log_event(session_id, "memory_override", {
        "old_context": old_context,
        "new_context": new_context,
    })