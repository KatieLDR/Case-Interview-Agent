#!/usr/bin/env python3
"""1_pull_snapshot.py — one-time, READ-ONLY pull of session docs from Firestore.

Step 1 of the analysis-scoring pipeline (see ba-agent/docs/analysis_scoring_plan.md).

Two modes:
  --probe       Inspect collections WITHOUT writing anything. Prints doc counts,
                non-null participant_id counts, and one sample doc's field names for
                sessions_g1, sessions_g0, sessions. Use this to confirm empirically
                which collection holds the official cohort (expected: sessions_g1,
                pids look like Qualtrics R_...). Settles finding 1 with data, not assumption.
  (default)     Pull every doc from --collection into an immutable, date-stamped
                analysis/data/raw_snapshot_<date>.json. Never overwrites an existing
                snapshot. Timestamps ISO-stringified.

This script is standalone by design: it does NOT import backend.logger, whose
module-level init defaults SESSIONS_COLLECTION to "sessions" — exactly the
wrong-collection trap this pipeline exists to avoid. Reads only: .stream()/.get().
Never .set()/.update()/.delete(). No LLM. Zero changes to backend/.
"""

import argparse
import datetime as _dt
import json
import os
import sys

# Standalone Firestore init from firebase_key.json (mirrors backend/logger.py:11-18,
# but with no SESSIONS_COLLECTION default and no side effects on import).
import firebase_admin
from firebase_admin import credentials, firestore

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)  # ba-agent/
DATA_DIR = os.path.join(HERE, "data")

# Collections to inspect in --probe mode. Official cohort expected in sessions_g1.
PROBE_COLLECTIONS = ["sessions_g1", "sessions_g0", "sessions"]


def _init_db():
    """Read-only Firestore client from firebase_key.json in the repo root."""
    if not firebase_admin._apps:
        key_path = os.getenv("FIREBASE_KEY_PATH", os.path.join(REPO_ROOT, "firebase_key.json"))
        if not os.path.exists(key_path):
            sys.exit(f"ERROR: firebase key not found at {key_path} "
                     f"(set FIREBASE_KEY_PATH to override).")
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


def _jsonable(v):
    """Make a Firestore value JSON-serialisable. Timestamps -> ISO strings."""
    # Firestore returns datetimes (with tz) for timestamp fields.
    if isinstance(v, _dt.datetime):
        return v.isoformat()
    if isinstance(v, _dt.date):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    # firestore GeoPoint / DocumentReference / bytes -> repr fallback (rare here).
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def probe(db):
    """Print counts + a sample field list per collection. Writes nothing."""
    print("=" * 70)
    print("PROBE — read-only. No files written. No data modified.")
    print("=" * 70)
    for coll in PROBE_COLLECTIONS:
        try:
            docs = list(db.collection(coll).stream())
        except Exception as e:  # noqa: BLE001 — surface any access error per-collection
            print(f"\n[{coll}]  ERROR reading collection: {e}")
            continue
        n = len(docs)
        non_null_pid = 0
        sample_fields = None
        sample_pid = None
        for d in docs:
            data = d.to_dict() or {}
            pid = data.get("participant_id")
            if pid not in (None, ""):
                non_null_pid += 1
                if sample_pid is None:
                    sample_pid = pid
            if sample_fields is None:
                sample_fields = sorted(data.keys())
        print(f"\n[{coll}]")
        print(f"  docs                : {n}")
        print(f"  non-null pid        : {non_null_pid}")
        print(f"  sample pid          : {sample_pid!r}  "
              f"(Qualtrics ResponseId looks like 'R_...')")
        if sample_fields:
            print(f"  sample doc fields   : {', '.join(sample_fields)}")
        else:
            print("  sample doc fields   : (no docs)")
    print("\n" + "=" * 70)
    print("Confirm the OFFICIAL cohort is the collection whose pids look like 'R_...'")
    print("(expected: sessions_g1). Then run without --probe:")
    print("    python analysis/1_pull_snapshot.py --collection sessions_g1")
    print("=" * 70)


def pull(db, collection, out_path):
    """Pull every doc from `collection` into out_path. Never overwrites."""
    if os.path.exists(out_path):
        sys.exit(f"ERROR: snapshot already exists at {out_path}. "
                 f"Snapshots are immutable — remove it manually to re-pull, "
                 f"or wait for tomorrow's date stamp.")
    docs = list(db.collection(collection).stream())
    records = []
    non_null_pid = 0
    for d in docs:
        data = d.to_dict() or {}
        rec = {"doc_id": d.id}
        rec.update({k: _jsonable(v) for k, v in data.items()})
        records.append(rec)
        if data.get("participant_id") not in (None, ""):
            non_null_pid += 1

    snapshot = {
        "_meta": {
            "collection": collection,
            "pulled_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "n_docs": len(records),
            "n_non_null_pid": non_null_pid,
        },
        "docs": records,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} docs ({non_null_pid} with non-null pid) "
          f"from '{collection}' -> {out_path}")
    print("Snapshot is immutable. All downstream steps read this file.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="Inspect collections read-only; write nothing.")
    ap.add_argument("--collection", default="sessions_g1",
                    help="Collection to pull (default: sessions_g1). Ignored with --probe.")
    ap.add_argument("--out", default=None,
                    help="Output path (default: analysis/data/raw_snapshot_<YYYY-MM-DD>.json).")
    args = ap.parse_args()

    db = _init_db()

    if args.probe:
        probe(db)
        return

    # Filename encodes the collection so a pilot (sessions_g0) snapshot can never be
    # confused with the official (sessions_g1) one — two live collections are in play.
    out_path = args.out or os.path.join(
        DATA_DIR, f"raw_snapshot_{args.collection}_{_dt.date.today().isoformat()}.json")
    pull(db, args.collection, out_path)


if __name__ == "__main__":
    main()
