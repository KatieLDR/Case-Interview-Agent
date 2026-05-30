import json
import logging
import os
from google import genai
from dotenv import load_dotenv
from backend import knowledge_base as kb

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

# ── Faithfulness threshold ─────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.85


# ══════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════

_PILLAR_MATCH_PROMPT = """
You are a classifier for a case interview coaching system.

The agent is presenting a concept to the user. Your job is to match the concept
name to one of the pillars in the knowledge base below.

─── KNOWLEDGE BASE PILLARS ───────────────────────────────────────────────
{pillars_block}
──────────────────────────────────────────────────────────────────────────

─── CONCEPT NAME ─────────────────────────────────────────────────────────
{concept_name}
──────────────────────────────────────────────────────────────────────────

Rules:
- Match if the concept name is clearly a sub-concept of one of the pillars above
- Match if the concept name is a close paraphrase or synonym of a sub-concept
- Match the pillar the concept most naturally belongs to
- If the concept does not match any pillar, return null

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched_pillar_id": "pillar_id string or null", "confidence": float between 0.0 and 1.0}}
"""

_FAITHFULNESS_PROMPT = """
You are a grounding classifier for a knowledge-base-backed case interview system.

Your job is to check whether an agent's concept block stays within the analytical
scope defined by the knowledge base data below.

─── KNOWLEDGE BASE DATA ──────────────────────────────────────────────────
Pillar      : {pillar_name}
Description : {pillar_description}
Sub-concepts: {sub_concepts}
──────────────────────────────────────────────────────────────────────────

─── AGENT CONCEPT BLOCK ───────────────────────────────────────────────────
{concept_block}
───────────────────────────────────────────────────────────────────────────

Mark as NOT faithful if the concept block:
- Introduces concepts from a completely different domain
- Makes claims that clearly contradict the pillar description or scope
- Introduces analytical lenses not mentioned in the pillar description

Mark as FAITHFUL if:
- The block stays within the analytical scope described in the pillar description
- The block uses case-specific elaboration consistent with the pillar scope
- The block uses consulting framing language without introducing new domains

When in doubt, mark as faithful.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"faithful": true or false, "confidence": float between 0.0 and 1.0}}
"""

PILLAR_MATCH_THRESHOLD = 0.80


# ══════════════════════════════════════════════════════════════════════════
# Pillar matcher — LLM fuzzy match
# ══════════════════════════════════════════════════════════════════════════

def _match_pillar(concept_name: str) -> dict | None:
    """
    LLM fuzzy-match a concept name to a pillar in the knowledge base.
    Returns the matched pillar dict, or None if no confident match.
    """
    all_pillars = kb.get_all_pillars()

    # Build pillars block for the prompt
    lines = []
    for p in all_pillars:
        sub_concepts = [c["name"] for c in kb.get_concepts_for_pillar(p["id"])]
        sub_str = ", ".join(sub_concepts) if sub_concepts else "none"
        lines.append(
            f"- id: {p['id']}\n"
            f"  name: {p['name']}\n"
            f"  sub-concepts: {sub_str}"
        )
    pillars_block = "\n".join(lines)

    prompt = _PILLAR_MATCH_PROMPT.format(
        pillars_block=pillars_block,
        concept_name=concept_name,
    )

    try:
        response = client.models.generate_content(
            model=CLASSIFIER_MODEL,
            contents=prompt,
        )
        raw    = _strip_fences(response.text)
        parsed = json.loads(raw)

        pillar_id  = parsed.get("matched_pillar_id")
        confidence = parsed.get("confidence", 0.0)

        if pillar_id and confidence >= PILLAR_MATCH_THRESHOLD:
            pillar = kb.get_pillar_by_id(pillar_id)
            logging.info(
                f"[PILLAR MATCH] concept='{concept_name}' "
                f"matched to pillar='{pillar_id}' confidence={confidence:.2f}"
            )
            return pillar

        logging.info(
            f"[PILLAR MATCH] concept='{concept_name}' "
            f"no confident match (id={pillar_id}, confidence={confidence:.2f})"
        )
        return None

    except Exception as e:
        logging.warning(f"[PILLAR MATCH] error for '{concept_name}': {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# Citation builder
# ══════════════════════════════════════════════════════════════════════════

def build_citation_header(concept_name: str, framework: str) -> tuple[str | None, dict]:
    """
    Build citation header for a concept.

    Two states:
      State 1 — concept matches a pillar in JSON:
        Returns (header, kb_data) with pillar description + HTML source links.
      State 2 — concept not matched:
        Returns (None, kb_data). Caller shows unverified note.

    kb_data dict always returned for faithfulness check downstream.

    Change log: 2026-05-25 — replaced KG lookup with JSON pillar match
    """
    pillar = _match_pillar(concept_name)

    kb_data = {
        "concept":     concept_name,
        "framework":   framework,
        "pillar":      pillar,
        "in_kb":       pillar is not None,
    }

    if pillar is None:
        logging.info(f"[CITATION] '{concept_name}' not matched to any pillar")
        return None, kb_data

    # Build source links
    sources      = kb.get_sources_for_pillar(pillar["id"])
    sources_html = kb.render_sources_as_html(sources)

    description = pillar.get("description", "")

    # Build header block
    # Format:
    # **Concept Name**
    # [pillar description]
    #
    # 📚 **Sources:** [link1] · [link2]
    #
    sources_line = (
        f"\n\n📚 **Sources:** {sources_html}" if sources_html else ""
    )

    header = (
        f"{description}"
        f"{sources_line}"
        f"\n\n"
    )

    logging.info(
        f"[CITATION] built for '{concept_name}' "
        f"pillar='{pillar['id']}' sources={len(sources)}"
    )
    return header, kb_data


# ══════════════════════════════════════════════════════════════════════════
# Faithfulness check
# ══════════════════════════════════════════════════════════════════════════

def check_and_append_warning(concept_name: str, concept_block: str, kb_data: dict) -> str | None:
    """
    Run faithfulness check after concept block streams.
    Returns warning string if unfaithful, None if faithful.

    Change log: 2026-05-25 — grounded in JSON pillar description instead of KG block
    """
    pillar = kb_data.get("pillar")

    if pillar is None:
        # No pillar matched — skip faithfulness check
        return None

    sub_concepts = [c["name"] for c in kb.get_concepts_for_pillar(pillar["id"])]
    sub_str      = ", ".join(sub_concepts) if sub_concepts else "none"

    prompt = _FAITHFULNESS_PROMPT.format(
        pillar_name        = pillar["name"],
        pillar_description = pillar.get("description", ""),
        sub_concepts       = sub_str,
        concept_block      = concept_block[:1000],
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

        if not result:
            return "\n\n⚠️ *I may have added context beyond what the framework specifies — please verify.*"
        return None

    except Exception as e:
        logging.warning(f"[FAITHFULNESS] classifier error for '{concept_name}': {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# Legacy compatibility — kept for any remaining callers
# ══════════════════════════════════════════════════════════════════════════

def build_and_check_citation(concept_name: str, framework: str, concept_block: str) -> str:
    """Legacy single-call pipeline — kept for backward compatibility with tests."""
    header, kb_data = build_citation_header(concept_name, framework)
    warning         = check_and_append_warning(concept_name, concept_block, kb_data)
    result          = header or ""
    if warning:
        result += warning
    return result


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