"""backend/domain/grounding.py  —  Step 2 of REFACTOR_PLAN.md (domain extraction).

Shared, agent-agnostic KB explanation lookup. Pure: reads the KB only, no `self`,
no agent reference. Dependency direction:  agent -> grounding -> knowledge_base.

Returns the PLAIN explanation text. Explainable layers citations + counterfactual on
top via `rag_explainer.py` (which stays Explainable-only); BlackBox/HITL use the plain
explanation directly (§3.2).

Unifies EXP/HITL `_concept_grounding` (they had drifted only in source-ref stripping;
the swap-fallback also referenced agent config — both reconciled here). BlackBox had no
grounding lookup of its own; it adopts this shared one when wired.

Glossary note (REFACTOR_PLAN header): the code's legacy `_concept_grounding(concept)`
actually took a *pillar* name (the walkthrough unit), not one of the 30 KB criteria.
Named accordingly here: `ground_pillar` for the walkthrough unit, `ground_concept` for
an individual criterion.
"""
from __future__ import annotations

import re

from backend.knowledge import knowledge_base as kb

# Leaf utility (mirrors matching._strip_source_refs intentionally; both read only their
# input — kept self-contained so grounding does not import matching's private helpers).
_REF_RE = re.compile(r"\s*\[[a-z]\]")


def _strip_source_refs(text: str) -> str:
    return _REF_RE.sub("", text or "").strip()


def ground_pillar(name: str) -> str:
    """Plain grounding for a pillar / walkthrough unit: its description followed by its
    key-questions, source refs stripped, NO sources block (sources are an Explainable-only
    render concern). Was EXP/HITL `_concept_grounding`. Falls back to the planted swap
    concept's sub-bullets when `name` is the swap (matched against the KB swap concept's
    own name, so no agent config is needed)."""
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
    """Plain grounding for one of the 30 KB criteria: its explanation text (refs stripped).
    The concept-level companion to ground_pillar (§3.2 'pillar/concept')."""
    c = kb.get_concept_by_id(concept_id)
    if not c:
        return ""
    return _strip_source_refs(c.get("explanation") or "")
