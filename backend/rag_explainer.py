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
# Consistent with other classifiers in the system.
FAITHFULNESS_THRESHOLD = 0.85

# ══════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════

_FAITHFULNESS_PROMPT = """
You are a grounding classifier for a knowledge-graph-backed case interview system.

Your job is to check whether an agent's concept block stays within the bounds
of the provided Knowledge Graph data — or whether it introduces claims, sub-buckets,
or rationale that go beyond what the KG data supports.

─── KNOWLEDGE GRAPH DATA ──────────────────────────────────────────────────
{kg_data_block}
───────────────────────────────────────────────────────────────────────────

─── AGENT CONCEPT BLOCK ───────────────────────────────────────────────────
{concept_block}
───────────────────────────────────────────────────────────────────────────

Determine:
- Is every claim in the concept block directly supported by or reasonably
  inferable from the KG data above?
- Are there sub-buckets or rationale sentences that introduce new concepts
  not present in the KG data?

Rules:
- Minor elaboration that stays consistent with the KG data is ACCEPTABLE
- Sub-buckets that name concepts from a completely different framework are NOT acceptable
- Generic consulting language ("based on best practices") is ACCEPTABLE as framing
- New quantitative claims or named metrics not in KG data are NOT acceptable

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
      parent      : str | None  — parent concept or framework name in HAS_CHILD tree
      children    : list[str]   — direct children (empty for leaf concepts)
      is_branch   : bool        — True if intermediate branch node
      all_concepts: list[str]   — full leaf concept list for this framework

    Change log: 2026-03-31 — initial implementation
    Change log: 2026-03-31 — replaced predecessor/successor with parent/children
      to reflect real consulting tree structure. predecessor/successor implied
      linear sequence across parallel branches which is analytically incorrect.
    """
    try:
        ordered = kg.get_ordered_concepts(framework)
    except Exception as e:
        logging.warning(f"[RAG] build_kg_data failed to fetch ordered concepts: {e}")
        ordered = []

    # Tree-aware fields — only populated for concepts in the KG
    parent    = None
    children  = []
    is_branch = False

    if concept_name in ordered or _concept_exists_in_kg(concept_name, framework):
        try:
            parent    = kg.get_concept_parent(concept_name, framework)
            children  = kg.get_concept_children(concept_name, framework)
            is_branch = kg.is_branch_node(concept_name, framework)
        except Exception as e:
            logging.warning(f"[RAG] tree data fetch failed for '{concept_name}': {e}")
    else:
        # Wrong concept (swap) — not in KG
        logging.info(f"[RAG] concept '{concept_name}' not found in KG for framework '{framework}'")

    return {
        "concept":      concept_name,
        "framework":    framework,
        "parent":       parent,
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
    to reflect tree structure. Citation NL justification now says
    "sub-bucket of X" not "follows X", which is analytically accurate.
    """
    parent   = kg_data["parent"]   or f"{kg_data['framework']} (top-level bucket)"
    children = ", ".join(kg_data["children"]) if kg_data["children"] else "None (leaf concept)"
    all_c    = ", ".join(kg_data["all_concepts"]) if kg_data["all_concepts"] else "N/A"
    node_type = "Branch node (groups sub-concepts)" if kg_data["is_branch"] else "Leaf concept (analysed directly)"

    return (
        f"Concept     : {kg_data['concept']}\n"
        f"Framework   : {kg_data['framework']}\n"
        f"Node type   : {node_type}\n"
        f"Parent      : {parent}\n"
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
# Public convenience function — called from explainable_agent.py
# ══════════════════════════════════════════════════════════════════════════

def build_and_check_citation(concept_name: str, framework: str, concept_block: str) -> str:
    """
    Full pipeline: build KG data → check faithfulness → generate citation block.

    This is the single entry point called from ExplainableAgent._stream_concept().

    Args:
        concept_name  : name of the concept just streamed
        framework     : current KG framework name
        concept_block : the full text of the concept block that was streamed

    Returns:
        Citation block string ready to stream to UI.

    Change log: 2026-03-31 — initial implementation
    """
    kg_data     = build_kg_data(concept_name, framework)
    faith_result = check_faithfulness(concept_name, concept_block, kg_data)
    citation    = generate_citation(concept_name, kg_data, faith_result["faithful"])
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