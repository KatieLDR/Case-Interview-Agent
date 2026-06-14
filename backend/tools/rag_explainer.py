import logging

from dotenv import load_dotenv

from backend.knowledge import knowledge_base as kb
from backend.llm import (
    classify_json, FAITHFULNESS_THRESHOLD, PILLAR_MATCH_THRESHOLD
)
from backend.tools.prompts.rag_explainer import (
    _PILLAR_MATCH_PROMPT, _FAITHFULNESS_PROMPT
)

load_dotenv()


# Pillar matcher — LLM fuzzy match
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
        parsed = classify_json(prompt)

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


# Citation builder
def build_citation_header(concept_name: str, framework: str) -> tuple[str | None, dict]:
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


# Faithfulness check
def check_and_append_warning(concept_name: str, concept_block: str, kb_data: dict) -> str | None:
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
        parsed = classify_json(prompt)

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


# Legacy compatibility — kept for any remaining callers
def build_and_check_citation(concept_name: str, framework: str, concept_block: str) -> str:
    header, kb_data = build_citation_header(concept_name, framework)
    warning         = check_and_append_warning(concept_name, concept_block, kb_data)
    result          = header or ""
    if warning:
        result += warning
    return result