import os
import json
import logging
from google import genai
from dotenv import load_dotenv
from backend import knowledge_graph as kg
# Change log: 2026-04-02 — get_framework_for_concept used for cross-framework citation
from backend.knowledge_graph import get_framework_for_concept

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

# ── Faithfulness threshold ─────────────────────────────────────────────────
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
    Pull KG node data for faithfulness check and citation.

    Change log: 2026-04-01 — added 'in_kg' boolean flag to returned dict.
    Callers use in_kg to decide whether to show citation or unverified note.
    Previously, build_citation_header() would fabricate a citation for
    user-added concepts not in KG because there was no way to detect the miss.
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
    in_kg            = False   # True if concept in current framework
    other_frameworks = []      # Non-empty if concept exists in other frameworks
    # Change log: 2026-04-01 — in_kg flag
    # Change log: 2026-04-02 — other_frameworks for cross-framework citation

    if concept_name in ordered or _concept_exists_in_kg(concept_name, framework):
        in_kg = True
        try:
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
        # Check if concept exists in any other framework
        try:
            other_frameworks = kg.get_framework_for_concept(concept_name)
        except Exception as e:
            logging.warning(f"[RAG] cross-framework lookup failed for '{concept_name}': {e}")

    return {
        "concept":           concept_name,
        "framework":         framework,
        "description":       description,
        "parent":            parent,
        "siblings":          siblings,
        "children":          children,
        "is_branch":         is_branch,
        "all_concepts":      ordered,
        "in_kg":             in_kg,
        "other_frameworks":  other_frameworks,
    }


def _concept_exists_in_kg(concept_name: str, framework: str) -> bool:
    try:
        return kg.concept_belongs_to_framework(concept_name, framework)
    except Exception:
        return False


def _format_kg_block(kg_data: dict) -> str:
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
    Returns {"faithful": bool, "confidence": float}
    Defaults to faithful=True on error.
    """
    kg_block = _format_kg_block(kg_data)
    prompt   = _FAITHFULNESS_PROMPT.format(
        kg_data_block=kg_block,
        concept_block=concept_block[:1000],
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
        result     = faithful if confidence >= FAITHFULNESS_THRESHOLD else True

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
    kg_block     = _format_kg_block(kg_data)
    entity_id    = f"KG::Concept::{concept_name}"
    framework    = kg_data["framework"]
    justification = _generate_justification(kg_block, concept_name)

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
        return "This concept is part of the structured framework for this analysis."


# ══════════════════════════════════════════════════════════════════════════
# Public functions
# ══════════════════════════════════════════════════════════════════════════

def build_citation_header(concept_name: str, framework: str) -> tuple[str | None, dict]:
    """
    Build citation header + NL justification — called BEFORE streaming concept.

    Three states — Change log: 2026-04-02:
      State 1 — concept in current framework:
        Returns (header, kg_data) with full KG citation + NL justification.
      State 2 — concept in a different framework:
        Returns (header, kg_data) with cross-framework citation + NL justification
        sourced from the correct framework. Source line includes (not current framework).
      State 3 — concept not in KG anywhere:
        Returns (None, kg_data). Caller shows unverified note + LLM answer.
    """
    kg_data = build_kg_data(concept_name, framework)

    # ── State 1: Concept in current framework — full citation ──────────────
    if kg_data["in_kg"]:
        entity_id     = "KG::Concept::" + concept_name
        fw_name       = kg_data["framework"]
        justification = _generate_justification(_format_kg_block(kg_data), concept_name)
        header = (
            "\U0001F4DA **Source:** `" + entity_id + "` \u2014 " + fw_name + "\n"
            + justification + "\n\n"
        )
        logging.info("[CITATION] full citation for '" + concept_name + "' in '" + framework + "'")
        return header, kg_data

    # ── State 2: Concept in a different framework — cross-framework citation
    other = kg_data.get("other_frameworks", [])
    if other:
        other_fw   = other[0]["framework"]
        # Fetch full KG data from the framework it actually belongs to
        cross_data = build_kg_data(concept_name, other_fw)
        entity_id  = "KG::Concept::" + concept_name
        justification = _generate_justification(_format_kg_block(cross_data), concept_name)
        header = (
            "\U0001F4DA **Source:** `" + entity_id + "` \u2014 "
            + other_fw + " (not current framework)\n"
            + justification + "\n\n"
        )
        logging.info(
            "[CITATION] cross-framework citation for '" + concept_name
            + "' \u2014 belongs to '" + other_fw + "', not '" + framework + "'"
        )
        return header, cross_data

    # ── State 3: Concept not in KG anywhere — caller shows unverified note ──
    logging.info("[CITATION] '" + concept_name + "' not found in any framework")
    return None, kg_data


def check_and_append_warning(concept_name: str, concept_block: str, kg_data: dict) -> str | None:
    """
    Run faithfulness check after concept block streams.
    Returns warning string if unfaithful, None if faithful.
    """
    result = check_faithfulness(concept_name, concept_block, kg_data)
    if not result["faithful"]:
        return "\n⚠️ *I may have added context beyond what the framework specifies — please verify.*"
    return None


def build_and_check_citation(concept_name: str, framework: str, concept_block: str) -> str:
    """Legacy single-call pipeline — kept for backward compatibility with tests."""
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