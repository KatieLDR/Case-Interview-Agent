import os
import json
import logging
from google import genai
from dotenv import load_dotenv
from backend import knowledge_graph as kg

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

# ── Faithfulness threshold ─────────────────────────────────────────────────
# Kept at 0.85 — classifier must be highly confident before flagging unfaithful.
# The safe fallback (default faithful=True when confidence < threshold) protects
# against false positives on legitimate consulting elaboration.
# Change log: 2026-03-31 — reverted 0.70 experiment; root cause is prompt
#   over-sensitivity, not threshold. Prompt tightened instead.
FAITHFULNESS_THRESHOLD = 0.85

# ══════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════

_FAITHFULNESS_PROMPT = """
You are a grounding classifier for a knowledge-graph-backed case interview system.

Your job is to check whether an agent's concept block stays within the analytical
scope defined by the Knowledge Graph data below.

─── KNOWLEDGE GRAPH DATA ──────────────────────────────────────────────────
{kg_data_block}
───────────────────────────────────────────────────────────────────────────

─── AGENT CONCEPT BLOCK ───────────────────────────────────────────────────
{concept_block}
───────────────────────────────────────────────────────────────────────────

Mark as NOT faithful if the concept block:
- Introduces concepts from a completely different framework
  (e.g. NPS or market sizing in a cost analysis block)
- Makes claims that clearly contradict the concept's Description or scope
- Introduces analytical lenses not mentioned in the concept's Description

Mark as FAITHFUL if:
- The block stays within the analytical scope described in the Description field
- The block uses case-specific elaboration consistent with the concept's scope
- The block references sibling or parent concepts in passing — this is normal
  consulting practice and does NOT indicate unfaithfulness
- The block uses consulting framing language without introducing new domains

When in doubt, mark as faithful.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"faithful": true or false, "confidence": float between 0.0 and 1.0}}
"""

_CITATION_PROMPT = """
You are a knowledge graph citation assistant for a case interview coaching system.

Write a concise 2-3 sentence natural language justification explaining:
1. Why this concept belongs in this framework (one sentence)
2. Where it sits in the analytical sequence — what comes before and after it (one sentence)
3. Why it matters specifically for this case (one sentence)

─── KNOWLEDGE GRAPH DATA ──────────────────────────────────────────────────
{kg_data_block}
───────────────────────────────────────────────────────────────────────────

Rules:
- Plain language only — no jargon, no technical terms
- Do NOT mention "Knowledge Graph", "KG", "database", or any technical system
- Do NOT start with "This concept" — vary your opening
- Stay grounded in the KG data provided — do not invent context
- Maximum 3 sentences total

Respond with ONLY the justification text — no labels, no markdown, no preamble.
"""


# ══════════════════════════════════════════════════════════════════════════
# KG data builder
# ══════════════════════════════════════════════════════════════════════════

def build_kg_data(concept_name: str, framework: str) -> dict:
    """
    Pull the KG node data needed for faithfulness check and citation.

    Returns a dict with:
      concept     : str
      framework   : str
      description : str   — concept scope and meaning (from KG description field)
      parent      : str | None
      siblings    : list[str] — other children of the same parent
      children    : list[str]
      is_branch   : bool
      all_concepts: list[str]

    Change log: 2026-03-31 — initial implementation
    Change log: 2026-03-31 — replaced predecessor/successor with parent/children
    Change log: 2026-03-31 — added description and siblings for richer
      faithfulness classifier grounding. Without descriptions, classifier had
      insufficient data to distinguish legitimate elaboration from out-of-scope
      claims, leading to false positives on clean concept blocks.
    """
    try:
        ordered = kg.get_ordered_concepts(framework)
    except Exception as e:
        logging.warning(f"[RAG] build_kg_data failed to fetch ordered concepts: {e}")
        ordered = []

    parent      = None
    siblings    = []
    children    = []
    is_branch   = False
    description = ""

    if concept_name in ordered or _concept_exists_in_kg(concept_name, framework):
        try:
            # Single batched KG query — replaces 5 sequential calls
            # Change log: 2026-03-31 — batched to reduce round-trips
            full = kg.get_concept_full_data(concept_name, framework)
            parent      = full["parent"]
            siblings    = full["siblings"]
            children    = full["children"]
            is_branch   = full["is_branch"]
            description = full["description"]
        except Exception as e:
            logging.warning(f"[RAG] tree data fetch failed for '{concept_name}': {e}")
    else:
        logging.info(f"[RAG] concept '{concept_name}' not found in KG for framework '{framework}'")

    return {
        "concept":      concept_name,
        "framework":    framework,
        "description":  description,
        "parent":       parent,
        "siblings":     siblings,
        "children":     children,
        "is_branch":    is_branch,
        "all_concepts": ordered,
    }


def _concept_exists_in_kg(concept_name: str, framework: str) -> bool:
    """Check if a concept exists in the KG for a given framework (including branch nodes)."""
    try:
        return kg.concept_belongs_to_framework(concept_name, framework)
    except Exception:
        return False


def _format_kg_block(kg_data: dict) -> str:
    """
    Format KG data dict into a readable block for prompt injection.

    Change log: 2026-03-31 — replaced predecessor/successor with parent/children
    Change log: 2026-03-31 — added description and siblings for richer
      faithfulness classifier grounding.
    """
    parent    = kg_data["parent"]    or f"{kg_data['framework']} (top-level bucket)"
    siblings  = ", ".join(kg_data.get("siblings", [])) or "None (no siblings)"
    children  = ", ".join(kg_data["children"])          or "None (leaf concept)"
    all_c     = ", ".join(kg_data["all_concepts"])       or "N/A"
    node_type = "Branch node (groups sub-concepts)" if kg_data["is_branch"] else "Leaf concept (analysed directly)"
    desc      = kg_data.get("description") or "No description available."

    return (
        f"Concept     : {kg_data['concept']}\n"
        f"Description : {desc}\n"
        f"Framework   : {kg_data['framework']}\n"
        f"Node type   : {node_type}\n"
        f"Parent      : {parent}\n"
        f"Siblings    : {siblings}\n"
        f"Children    : {children}\n"
        f"All concepts: {all_c}"
    )


# ══════════════════════════════════════════════════════════════════════════
# Faithfulness check
# ══════════════════════════════════════════════════════════════════════════

def check_faithfulness(concept_name: str, concept_block: str, kg_data: dict) -> dict:
    """
    Check whether the streamed concept block stays within KG-supported bounds.

    Returns:
      {"faithful": bool, "confidence": float}

    On classifier error, defaults to faithful=True to avoid false warnings.

    Change log: 2026-03-31 — initial implementation
    """
    kg_block = _format_kg_block(kg_data)
    prompt   = _FAITHFULNESS_PROMPT.format(
        kg_data_block=kg_block,
        concept_block=concept_block[:1000],  # cap to avoid token bloat
    )

    try:
        response = client.models.generate_content(
            model=CLASSIFIER_MODEL,
            contents=prompt,
        )
        raw    = _strip_fences(response.text)
        parsed = json.loads(raw)

        faithful   = parsed.get("faithful", True)
        confidence = parsed.get("confidence", 0.0)

        result = faithful if confidence >= FAITHFULNESS_THRESHOLD else True
        logging.info(
            f"[FAITHFULNESS] concept='{concept_name}', "
            f"faithful={faithful}, confidence={confidence:.2f}, result={result}"
        )
        return {"faithful": result, "confidence": confidence}

    except Exception as e:
        logging.warning(f"[FAITHFULNESS] classifier error for '{concept_name}': {e}")
        return {"faithful": True, "confidence": 0.0}


# ══════════════════════════════════════════════════════════════════════════
# Citation generator
# ══════════════════════════════════════════════════════════════════════════

def generate_citation(concept_name: str, kg_data: dict, faithful: bool) -> str:
    """
    Build the full citation block to stream after a concept block.

    Format:
      ---
      📚 **Source:** `KG::Concept::{name}` — {framework}
      {NL justification — 2-3 sentences}
      [⚠️ warning line — only if not faithful]

    Returns a plain string ready to yield to the Chainlit stream.

    Change log: 2026-03-31 — initial implementation
    """
    kg_block     = _format_kg_block(kg_data)
    entity_id    = f"KG::Concept::{concept_name}"
    framework    = kg_data["framework"]

    # Generate NL justification
    justification = _generate_justification(kg_block, concept_name)

    # Assemble citation block
    lines = [
        "\n\n---",
        f"📚 **Source:** `{entity_id}` — {framework}",
        justification,
    ]

    if not faithful:
        lines.append(
            "\n⚠️ *I may have added context beyond what the framework specifies — please verify.*"
        )

    return "\n".join(lines)


def _generate_justification(kg_block: str, concept_name: str) -> str:
    """Call Gemini to generate the 2-3 sentence NL justification."""
    prompt = _CITATION_PROMPT.format(kg_data_block=kg_block)

    try:
        response = client.models.generate_content(
            model=CLASSIFIER_MODEL,
            contents=prompt,
        )
        justification = response.text.strip()
        logging.info(f"[CITATION] justification generated for '{concept_name}'")
        return justification

    except Exception as e:
        logging.warning(f"[CITATION] justification generation failed for '{concept_name}': {e}")
        # Graceful fallback — still show citation without NL justification
        return f"This concept is part of the structured framework for this analysis."


# ══════════════════════════════════════════════════════════════════════════
# Public functions — called from explainable_agent.py
# ══════════════════════════════════════════════════════════════════════════

def build_citation_header(concept_name: str, framework: str) -> tuple[str, dict]:
    """
    Build the citation header and NL justification — called BEFORE streaming
    the concept block so the source appears first in the UI.

    Returns:
      (citation_header: str, kg_data: dict)

    The kg_data is returned so the caller can pass it directly to
    check_and_append_warning() after streaming, avoiding a second KG query.

    Change log: 2026-03-31 — added to support source-first presentation order.
    """
    kg_data       = build_kg_data(concept_name, framework)
    entity_id     = f"KG::Concept::{concept_name}"
    fw_name       = kg_data["framework"]
    justification = _generate_justification(_format_kg_block(kg_data), concept_name)
    header = (
        f"📚 **Source:** `{entity_id}` — {fw_name}\n"
        f"{justification}\n\n"
    )
    return header, kg_data


def check_and_append_warning(concept_name: str, concept_block: str, kg_data: dict) -> str | None:
    """
    Run faithfulness check on the streamed concept block.
    Returns the warning string if unfaithful, None if faithful.

    Called AFTER the concept block has finished streaming.

    Change log: 2026-03-31 — added to support source-first presentation order.
    """
    result = check_faithfulness(concept_name, concept_block, kg_data)
    if not result["faithful"]:
        return "\n⚠️ *I may have added context beyond what the framework specifies — please verify.*"
    return None


def build_and_check_citation(concept_name: str, framework: str, concept_block: str) -> str:
    """
    Legacy single-call pipeline — kept for backward compatibility with tests.
    Returns full citation block (header + justification + optional warning).

    For production use in _stream_concept(), prefer the two-step approach:
      1. build_citation_header() — before streaming concept
      2. check_and_append_warning() — after streaming concept

    Change log: 2026-03-31 — refactored internals to delegate to new functions.
    """
    kg_data      = build_kg_data(concept_name, framework)
    faith_result = check_faithfulness(concept_name, concept_block, kg_data)
    citation     = generate_citation(concept_name, kg_data, faith_result["faithful"])
    return citation


# ══════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text
