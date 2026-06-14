from __future__ import annotations

import re

from backend.knowledge import knowledge_base as kb

_REF_RE = re.compile(r"\s*\[[a-z]\]")


def _strip_source_refs(text: str) -> str:
    return _REF_RE.sub("", text or "").strip()


def ground_pillar(name: str) -> str:
    pillar = next(
        (p for p in kb.get_all_pillars() if p["name"].lower() == name.lower()),
        None,
    )
    if pillar:
        parts: list[str] = []
        if pillar.get("description"):
            parts.append(pillar["description"])
        kqs = [_strip_source_refs(q) for q in pillar.get("key_questions", [])]
        if kqs:
            parts.append("\n".join(f"- {q}" for q in kqs))
        return "\n\n".join(parts).strip()

    swap = kb.get_swap_concept()
    if swap and name.lower() == (swap.get("name") or "").lower():
        return "\n".join(f"- {_strip_source_refs(b)}" for b in swap.get("sub_bullets", []))
    return ""


def ground_concept(concept_id: str) -> str:
    c = kb.get_concept_by_id(concept_id)
    if not c:
        return ""
    return _strip_source_refs(c.get("explanation") or "")
