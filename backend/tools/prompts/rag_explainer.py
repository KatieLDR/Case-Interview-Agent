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