ADD_PILLAR_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.

The user wants to add a new area: "{item}".
Each area below is given as "Name: description". Check whether the user's new
area is essentially the SAME AREA of analysis as one of them — same topic or
clearly the same scope, possibly different wording.

─── AREAS ───────────────────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────
Match only at the level of the whole AREA. If the user's text is just one
specific point that would sit *inside* an area (rather than naming the area
itself), set matched=false.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched": true or false, "matched_pillar": "pillar name or null", "confidence": float}}
"""

ADD_CONCEPT_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.
The user wrote: "{item}".

Below is a numbered list of analytical concepts, each as "Name: explanation".
Decide whether the user's text refers to the SAME underlying concern as ONE of
these concepts — the same analytical point, possibly worded differently.

─── CONCEPTS ────────────────────────────────────────────────────────────────
{concepts}
────────────────────────────────────────────────────────────────────────────
Judge by MEANING, not shared words. Two phrases that share only a generic term
(e.g. both contain "data") are NOT a match unless they address the same concern.
For example "data privacy / protection" is about confidentiality and PII — it is
NOT the same concern as data QUALITY or AVAILABILITY, even though both say "data".
Match only if the user's text is essentially that concept; reward a confident,
meaning-level match with high confidence and a loose/topical one with low. If it
is only loosely or topically related, set matched=false.

Respond ONLY with valid JSON, no markdown:
{{"matched": true or false, "matched_index": integer or null, "confidence": float}}
"""

ADD_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.

The user added "{item}" under the pillar "{pillar}".
Check whether it matches one of the pillar's existing key questions below.

─── KEY QUESTIONS FOR {pillar} ──────────────────────────────────────────────
{key_questions}
────────────────────────────────────────────────────────────────────────────

A match means the user's addition is essentially the same point as one of the
key questions above — same topic, possibly different wording.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched": true or false, "matched_index": integer or null, "confidence": float}}
"""

LOCATE_AREA_PROMPT = """
The user wrote: "{item}".

Below are the framework's AREAS. Each lists its description and the specific concerns it
covers (criteria, key questions, and points).

{areas}

Decide whether the user's text clearly corresponds to exactly ONE area -- i.e. it names
that area, or it is one of the concerns that area covers, or the funding / resource /
specific aspect of one of those concerns. Match ONLY if it clearly belongs to a SINGLE
area's listed concerns. If it is unrelated, too vague, or could belong to several areas
equally well, set matched=false.

Respond ONLY with valid JSON, no markdown:
{{"matched": true or false, "matched_pillar": "area name or null", "confidence": float}}
"""