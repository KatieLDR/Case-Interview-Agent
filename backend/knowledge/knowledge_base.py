import json
import logging
import os
from pathlib import Path

# ── JSON file path ─────────────────────────────────────────────────────────
_KB_PATH = Path(__file__).parent / "knowledge_base.json"

_kb_data: dict | None = None


def _load() -> dict:
    """
    Load and cache the knowledge base JSON.
    Called once on first access; subsequent calls return cached data.
    """
    global _kb_data
    if _kb_data is None:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            _kb_data = json.load(f)
        logging.info(f"[KB] loaded from {_KB_PATH}")
    return _kb_data


# ══════════════════════════════════════════════════════════════════════════
# Pillar accessors
# ══════════════════════════════════════════════════════════════════════════

def get_shown_pillars() -> list[dict]:
    """Return pillars where shown=True, in order."""
    kb = _load()
    return [p for p in kb["pillars"] if p.get("shown", False)]


def get_all_pillars() -> list[dict]:
    """Return all pillars in order."""
    kb = _load()
    return kb["pillars"]


def get_pillar_by_id(pillar_id: str) -> dict | None:
    """Return a single pillar dict by id, or None if not found."""
    kb = _load()
    for p in kb["pillars"]:
        if p["id"] == pillar_id:
            return p
    return None


# ══════════════════════════════════════════════════════════════════════════
# Concept accessors
# ══════════════════════════════════════════════════════════════════════════

def get_shown_concepts() -> list[dict]:
    """
    Return concepts where shown=True and swap=False, ordered by pillar then number.
    These are the concepts the agent presents in the walkthrough.
    """
    kb = _load()
    shown_pillar_ids = {p["id"] for p in kb["pillars"] if p.get("shown", False)}
    return [
        c for c in kb["concepts"]
        if c.get("shown", False)
        and not c.get("swap", False)
        and c.get("pillar_id") in shown_pillar_ids
    ]


def get_all_concepts() -> list[dict]:
    """Return all concepts including withheld and swap."""
    kb = _load()
    return kb["concepts"]


def get_concept_by_id(concept_id: str) -> dict | None:
    """Return a concept dict by id, or None if not found."""
    kb = _load()
    for c in kb["concepts"]:
        if c["id"] == concept_id:
            return c
    return None


def get_concepts_for_pillar(pillar_id: str) -> list[dict]:
    """Return all non-swap concepts belonging to a pillar."""
    kb = _load()
    return [
        c for c in kb["concepts"]
        if c.get("pillar_id") == pillar_id and not c.get("swap", False)
    ]


def get_swap_concept() -> dict | None:
    """Return the swap concept dict, or None if not found."""
    kb = _load()
    for c in kb["concepts"]:
        if c.get("swap", False):
            return c
    return None


# ══════════════════════════════════════════════════════════════════════════
# Framework accessor
# ══════════════════════════════════════════════════════════════════════════

def get_framework_name() -> str:
    """Return the framework name string."""
    kb = _load()
    return kb["framework"]["name"]


def get_framework_sources() -> list[dict]:
    """Return the top-level framework sources list."""
    kb = _load()
    return kb["framework"].get("sources", [])


# ══════════════════════════════════════════════════════════════════════════
# Citation helpers
# ══════════════════════════════════════════════════════════════════════════

def get_pillar_for_concept_name(concept_name: str) -> dict | None:
    """
    Given a concept name (exact or close match), return its parent pillar dict.
    Used by rag_explainer to look up pillar description and sources.
    Returns None if concept not found.
    """
    kb = _load()
    concept_name_lower = concept_name.lower().strip()
    for c in kb["concepts"]:
        if c["name"].lower().strip() == concept_name_lower:
            if c.get("pillar_id"):
                return get_pillar_by_id(c["pillar_id"])
    return None


def get_sources_for_pillar(pillar_id: str) -> list[dict]:
    """
    Return sources for a pillar, aggregated from all its concepts.
    Deduplicates by URL. Skips sources with url=None.
    """
    kb = _load()
    seen_urls = set()
    sources = []
    for c in kb["concepts"]:
        if c.get("pillar_id") == pillar_id and not c.get("swap", False):
            for s in c.get("sources", []):
                url = s.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append(s)
    return sources


def render_sources_as_html(sources: list[dict]) -> str:
    """
    Render a list of source dicts as HTML anchor tags separated by ' · '.
    Sources with url=None are rendered as plain text.
    Sources with verification_status=PLACEHOLDER are skipped entirely.
    """
    parts = []
    for s in sources:
        title = s.get("title", "Source")
        url = s.get("url")
        status = s.get("verification_status", "")

        if url:
            parts.append(f'<a href="{url}" target="_blank">{title}</a>')
        else:
            parts.append(title)

    return " · ".join(parts) if parts else ""


# ══════════════════════════════════════════════════════════════════════════
# Sub-bullet and key question accessors
# ══════════════════════════════════════════════════════════════════════════

def get_sub_bullets(pillar_id: str) -> list[str]:
    """
    Return the 2 static sub-bullets for a pillar.
    Used by all agents for static framework presentation.
    """
    pillar = get_pillar_by_id(pillar_id)
    if pillar is None:
        return []
    return pillar.get("sub_bullets", [])


def get_sub_bullets_sources(pillar_id: str) -> str:
    """
    Return the formatted sources line for sub-bullets.
    Used by ExplainableAgent only.
    """
    pillar = get_pillar_by_id(pillar_id)
    if pillar is None:
        return ""
    return pillar.get("sub_bullets_sources", "")


def get_key_questions(pillar_id: str) -> list[str]:
    """
    Return all key questions for a pillar.
    Used for LLM matching when user adds a new concept.
    """
    pillar = get_pillar_by_id(pillar_id)
    if pillar is None:
        return []
    return pillar.get("key_questions", [])


def get_key_questions_sources(pillar_id: str) -> str:
    """
    Return the formatted sources line for key questions.
    Used by ExplainableAgent only when user adds a matched concept.
    """
    pillar = get_pillar_by_id(pillar_id)
    if pillar is None:
        return ""
    return pillar.get("key_questions_sources", "")


def get_all_key_questions_flat() -> list[dict]:
    """
    Return all key questions across all pillars as a flat list.
    Each item includes pillar_id, pillar_name, and question text (without source refs).
    Used by LLM matcher to find closest match when user adds a concept.
    """
    result = []
    for pillar in get_all_pillars():
        for q in pillar.get("key_questions", []):
            # Strip inline source refs like [a][b] for cleaner LLM matching
            clean_q = q.split(" [")[0].strip()
            result.append({
                "pillar_id":   pillar["id"],
                "pillar_name": pillar["name"],
                "question":    clean_q,
                "full":        q,
            })
    return result


def build_static_framework_text(include_sources: bool = False) -> str:
    """
    Build the full static framework text for all shown pillars.
    include_sources=True for ExplainableAgent (adds description + sources).
    include_sources=False for BlackBox and HITL (sub-bullets only).

    Change log: 2026-05-28 — added for static framework presentation
    """
    shown = get_shown_pillars()
    lines = ["Here is how I would structure the analysis:\n"]

    for pillar in shown:
        lines.append(f"**{pillar['name']}**")
        if include_sources:
            lines.append(pillar.get("description", ""))
            lines.append("")
        for bullet in pillar.get("sub_bullets", []):
            lines.append(f"- {bullet}")
        if include_sources:
            sources = pillar.get("sub_bullets_sources", "")
            if sources:
                lines.append(f"\n{sources}")
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Ordered concept names — same interface as knowledge_graph.get_ordered_concepts
# ══════════════════════════════════════════════════════════════════════════

def get_ordered_concept_names() -> list[str]:
    """
    Return shown non-swap concept names in pillar order.
    Keeps same behaviour as kg.get_ordered_concepts() for agent compatibility.
    """
    shown = get_shown_concepts()
    return [c["name"] for c in shown]