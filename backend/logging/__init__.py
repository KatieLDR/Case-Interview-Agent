"""backend/logging — Step 5: the shared logging layer (events authoritative, counters thin
rollups). `events` owns the §3.6 schema and the one firing point (I-1); `sink` is the
Firestore writer.

NOTE: this package __init__ re-exports ONLY the dependency-light `events` surface. `sink`
(which pulls in firebase_admin) is imported EXPLICITLY where the real writer is needed
(`from backend.logging.sink import firestore_sink`). This keeps `import backend.logging.events`
firebase-free so the Step-5 gate runs with no Firestore credentials.
"""
from backend.logging.events import (
    record, record_turn, EventContext, EventSink,
    session_started, session_ended,
    swap_presented, swap_detected, swap_questioned, question, interruption,
    COUNTER_FOR, HEADLINE_COUNTERS, VALID_SOURCES,
)
