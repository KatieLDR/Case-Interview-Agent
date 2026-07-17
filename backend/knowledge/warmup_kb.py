"""Loader for the warm-up practice knowledge base.

A deliberately tiny mirror of ``knowledge_base.py``: the warm-up is a miniature of
the main-phase interaction contract (only add/remove, KB-grounded wording, withheld
pillars that reveal on mention), so it has its own small, immutable KB rather than
reusing the case-interview one. Same cached-JSON-load pattern.
"""

import json
import logging
from pathlib import Path

_WKB_PATH = Path(__file__).parent / "warmup_kb.json"

_wkb_data: dict | None = None


def _load() -> dict:
    """Load and cache the warm-up KB JSON. First access reads the file; later calls
    return the cached dict."""
    global _wkb_data
    if _wkb_data is None:
        with open(_WKB_PATH, "r", encoding="utf-8") as f:
            _wkb_data = json.load(f)
        logging.info(f"[WARMUP KB] loaded from {_WKB_PATH}")
    return _wkb_data


def get_all_pillars() -> list[dict]:
    """Return all warm-up pillars in file order (shown and withheld)."""
    return _load()["pillars"]


def get_shown_pillars() -> list[dict]:
    """Return only the pillars presented up front (shown=True), in file order."""
    return [p for p in _load()["pillars"] if p.get("shown", False)]


def get_pillar_by_name(name: str) -> dict | None:
    """Return a pillar dict by case-insensitive name match, or None."""
    if not name:
        return None
    target = name.strip().lower()
    for p in _load()["pillars"]:
        if p["name"].lower() == target:
            return p
    return None
