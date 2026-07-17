WARMUP_PC_CLASSIFIER = """\
A user is adding "{item}" to a moving-to-a-new-city plan. They were asked whether it is
a new top-level PILLAR (a major section like Housing or Admin) or a BULLET (a detail)
under an existing pillar.

Classify their reply into exactly ONE branch:
- "pillar": a new top-level pillar / section ("pillar", "new pillar", "its own pillar",
  "a section", "make it top-level").
- "bullet": a detail under a pillar — this INCLUDES a bare "bullet" / "point" / "a detail"
  even when they DON'T name a pillar. If they name a pillar to file it under, put that
  name in "parent".

When in doubt, prefer "bullet".

Respond ONLY with valid JSON, no markdown:
{{"branch": "pillar|bullet", "parent": "pillar name or null"}}

Reply: "{reply}"
"""

WARMUP_LOCATE_PROMPT = """\
You match a user's phrase to an item in a small moving-to-a-new-city plan.

Here are the areas of the plan and their points:
{pillars_block}

The user said: "{user_text}"

Decide whether the user's phrase refers to one of the AREAS above, or to one of the
POINTS listed under an area. Match on meaning, not exact words. If the phrase is a new
idea that does not clearly correspond to any listed area or point, return nulls.

Respond ONLY with valid JSON, no markdown:
{{"pillar": "area name or null", "bullet": "point text or null", "confidence": float between 0.0 and 1.0}}
"""
