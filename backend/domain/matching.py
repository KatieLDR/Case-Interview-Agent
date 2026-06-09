"""backend/domain/matching.py  —  Step 2 of REFACTOR_PLAN.md (domain extraction).

Pure KB matching/resolution. NO agent state, NO `self`. State the matcher needs is
passed by value. Dependency direction is one-way:  agent -> matching -> knowledge_base.
This module must never import from any agent.

Unifies the per-agent matchers that had drifted (F-M1..F-M6):
  _match_pillar / _match_concept / _match_key_question / _classify_removal_target
  + _norm / _is_excluded_bullet.
HITL previously lacked the concept matcher and the removal-target classifier entirely;
routing every arm through locate() gives all three identical resolution by construction.

Behaviour-preserving for Step 2 (A-rows must reproduce baseline). The only deliberate
change: a concept-level hit now keeps `concept_id` + `level="concept"` instead of
collapsing to the parent pillar name, so a named criterion ("data quality") resolves as
a duplicate identically across arms (F-M1). Handlers (Step 4) consume KBMatch.level.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.knowledge import knowledge_base as kb
from backend.llm import classify_json, ADD_MATCH_THRESHOLD, CONCEPT_MATCH_THRESHOLD

# ── ported verbatim from the agents (source-ref stripping) ──────────────────
_REF_RE = re.compile(r"\s*\[[a-z]\]")

def _strip_source_refs(text: str) -> str:
    return _REF_RE.sub("", text or "").strip()

def _norm(text: str) -> str:
    """Whitespace/ref-insensitive, lower-cased — for set-membership comparisons."""
    return re.sub(r"\s+", " ", _REF_RE.sub("", text or "")).strip().lower()

# Deictic references resolve against what was LAST DISCUSSED (passed in), never re-parsed
# from free text — keeps removal deterministic (I-7) and identical across arms.
_DEICTIC = {
    "this", "it", "that", "this one", "that one", "the last one",
    "the one above", "the last point", "the above", "remove this",
}
# Filler words are NEVER a valid concept/area name — guards the "else" pillar bug (F-I1/F-I2).
_FILLER = {"else", "more", "other", "others", "anything", "something", "etc", "and so on"}
# Minimal function-word set, so an all-filler PHRASE ("anything else", "what else")
# is caught, not just a bare token. Deliberately tiny — never strip a real topic word.
_GLUE = {"the", "a", "an", "to", "we", "i", "you", "is", "are", "any", "of", "for",
         "please", "can", "could", "would", "do", "what", "should", "and", "or",
         "add", "remove", "delete", "include", "consider", "want", "like", "think",
         "also", "so", "on", "about", "let", "us"}

# Ordinal references to the currently-shown bullets ("the second point").
_ORDINALS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1,
}

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
Decide whether the user's text clearly refers to ONE of these concepts — i.e.
it is essentially that concept, or a specific instance of it, possibly worded
differently.

─── CONCEPTS ────────────────────────────────────────────────────────────────
{concepts}
────────────────────────────────────────────────────────────────────────────
Match ONLY if you are confident it is essentially that concept. If the user's
text is only loosely or topically related, set matched=false.

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


@dataclass
class KBMatch:
    """Result of resolving user text against the KB hierarchy (§3.2).

    `match_type`, pillar-state and score are RUNTIME signals (I-5) — used for control,
    not persisted. `needs_disambiguation` is set only by resolve_removal_target when a
    deictic ("this") arrives with no recorded focus (BlackBox full-render case)."""
    pillar: str | None = None          # ALWAYS the resolved PARENT pillar name
    pillar_is_withheld: bool = False
    level: str = "none"                # "pillar" | "concept" | "none"
    concept_id: str | None = None
    matched_text: str | None = None    # exact KB text for a key_question hit (verbatim)
    match_type: str | None = None      # revealed_withheld | surfaced_unreached_shown | novel_not_in_kb
    score: float = 0.0
    needs_disambiguation: bool = False


# ── small pure predicate (was BlackBox/EXP `_is_excluded_bullet`) ───────────
def is_excluded_bullet(excluded_sub_bullets: dict, pillar_name: str, bullet: str) -> bool:
    """True if `bullet` was already removed from `pillar_name`. The excluded-map is
    SESSION state, so it is passed in explicitly (no `self`)."""
    removed = {_norm(b) for b in (excluded_sub_bullets or {}).get(pillar_name, [])}
    return _norm(bullet) in removed


def _pillar_is_withheld(pillar_name: str | None) -> bool:
    if not pillar_name:
        return False
    p = next((p for p in kb.get_all_pillars()
              if p["name"].lower() == pillar_name.lower()), None)
    return bool(p) and not p.get("shown", True)


# ── the three matching passes (one shared copy; was triplicated) ────────────
def match_pillar(item: str) -> tuple[str | None, float]:
    """Does the text name one of the framework AREAS (shown or withheld)? Description-
    enriched so semantically-adjacent terms ("IT budget" -> "Financial Impact") resolve."""
    pillars = kb.get_all_pillars()
    if not pillars:
        return None, 0.0
    block = "\n".join(f"- {p['name']}: {(p.get('description') or '').strip()}" for p in pillars)
    try:
        parsed = classify_json(ADD_PILLAR_MATCH_PROMPT.format(item=item, pillars=block))
        conf = parsed.get("confidence", 0.0)
        if parsed.get("matched") and conf >= ADD_MATCH_THRESHOLD:
            return parsed.get("matched_pillar"), conf
    except Exception as e:
        print(f"[matching._match_pillar] {e}")
    return None, 0.0


def match_concept(item: str) -> tuple[dict | None, float]:
    """Search every analytical concept (name + explanation) across all pillars, SWAP
    EXCLUDED. Returns the matched concept dict (keeps id + pillar_id) — not just the
    parent name — so locate() can mark level=concept (the F-M1 fix)."""
    concepts = [c for c in kb.get_all_concepts()
                if not c.get("swap", False) and c.get("pillar_id")]
    if not concepts:
        return None, 0.0
    block = "\n".join(f"{i}. {c['name']}: {(c.get('explanation') or '').strip()}"
                      for i, c in enumerate(concepts))
    try:
        parsed = classify_json(ADD_CONCEPT_MATCH_PROMPT.format(item=item, concepts=block))
        conf = parsed.get("confidence", 0.0)
        if parsed.get("matched") and conf >= CONCEPT_MATCH_THRESHOLD:
            idx = parsed.get("matched_index")
            if idx is not None and 0 <= idx < len(concepts):
                return concepts[idx], conf
    except Exception as e:
        print(f"[matching._match_concept] {e}")
    return None, 0.0


def match_key_question(item: str, pillar_name: str) -> tuple[str | None, float]:
    """Does the addition duplicate a key_question already inside `pillar_name`?
    Returns the matched KB question text VERBATIM (source-refs stripped)."""
    pillar = next((p for p in kb.get_all_pillars()
                   if p["name"].lower() == pillar_name.lower()), None)
    if pillar is None:
        return None, 0.0
    kqs = pillar.get("key_questions", [])
    if not kqs:
        return None, 0.0
    kq_block = "\n".join(f"{i}. {q}" for i, q in enumerate(kqs))
    try:
        parsed = classify_json(ADD_MATCH_PROMPT.format(item=item, pillar=pillar_name, key_questions=kq_block))
        conf = parsed.get("confidence", 0.0)
        if parsed.get("matched") and conf >= ADD_MATCH_THRESHOLD:
            idx = parsed.get("matched_index")
            if idx is not None and 0 <= idx < len(kqs):
                return _strip_source_refs(kqs[idx]), conf
    except Exception as e:
        print(f"[matching._match_key_question] {e}")
    return None, 0.0


# ── public entry points ─────────────────────────────────────────────────────
# Area matcher for locate() PASS 2. Unlike BlackBox's match_pillar (terse name+desc,
# strict "reject sub-points" guard), this presents each area ENRICHED with its concrete
# concerns — concept names + key_questions + sub_bullets — so a user term that is the
# funding/resource/aspect of one of those concerns (e.g. "IT budget" -> Financial Impact's
# cost concerns) resolves to the area. This is the "match the whole hierarchy" surface the
# plan (§3.2) specifies; the old locate() only saw concept names + bare pillar descriptions.
# locate-private (BlackBox keeps its own match_pillar untouched). JSON example doubled-braced.
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


def _match_area(item: str) -> tuple[str | None, float]:
    """PASS 2 of locate(): match against each area ENRICHED with its concerns
    (concept names + key_questions + sub_bullets). Returns (pillar_name, confidence)."""
    pillars = kb.get_all_pillars()
    concepts = kb.get_all_concepts()
    blocks = []
    for p in pillars:
        parts = [f"## {p['name']}"]
        if p.get("description"):
            parts.append(p["description"].strip())
        concerns = [c["name"] for c in concepts
                    if c.get("pillar_id") == p["id"] and not c.get("swap")]
        concerns += [_strip_source_refs(q) for q in p.get("key_questions", [])]
        concerns += [_strip_source_refs(b) for b in p.get("sub_bullets", [])]
        if concerns:
            parts.append("Covers: " + "; ".join(dict.fromkeys(concerns)))
        blocks.append("\n".join(parts))
    areas = "\n\n".join(blocks)
    try:
        parsed = classify_json(LOCATE_AREA_PROMPT.format(item=item, areas=areas))
        conf = parsed.get("confidence", 0.0)
        if parsed.get("matched") and conf >= ADD_MATCH_THRESHOLD:
            return parsed.get("matched_pillar"), conf
    except Exception as e:
        print(f"[matching._match_area] {e}")
    return None, 0.0


def locate(user_text: str) -> KBMatch:
    """Resolve user text against the WHOLE KB hierarchy, most-specific-first (§3.2):
        PASS 1  CONCEPT  (name + explanation)                 -> level=concept (+id)
        PASS 2  AREA     (description + key_questions +
                          sub_bullets + concept names)         -> level=pillar
        PASS 3  nothing                                        -> level=none (novel)
    Concept-before-area keeps a named criterion ("data quality") resolving as the
    duplicate it is (F-M1). PASS 2 is enriched with each area's CONCRETE concerns (the
    key_questions/sub_bullets the old locate() ignored), so a term that is the funding /
    resource / aspect of an area's concern ("IT budget" -> Financial Impact's cost
    concerns) resolves to that area instead of falling through to novel. Matching is
    against real KB text with strict framing, so genuinely novel terms ("change
    management", "vendor lock-in") still fall to none. Filler short-circuits to none."""
    text = (user_text or "").strip()
    toks = [t for t in re.findall(r"[a-z0-9]+", _norm(text)) if t not in _GLUE]
    if not text or not toks or all(t in _FILLER for t in toks):
        return KBMatch(level="none", match_type="novel_not_in_kb")

    # F-CASE (Step-2 amendment 2026-06-09): case-fold the probe sent to the LLM matchers.
    # locate() already lower-cases for the filler/token gate above; the matcher calls must
    # use the SAME normalised form, or a user's natural casing ("IT Budget") misses a match
    # the curated lower-case gate input ("IT budget") makes. matched_text / concept_id come
    # from the KB, so this only changes what the matcher SEES, never the KBMatch fields.
    probe = text.lower()

    # PASS 1 — specific criterion
    concept, cscore = match_concept(probe)
    if concept:
        parent = kb.get_pillar_by_id(concept["pillar_id"])
        pname = parent["name"] if parent else None
        withheld = bool(parent) and not parent.get("shown", True)
        return KBMatch(
            pillar=pname, pillar_is_withheld=withheld, level="concept",
            concept_id=concept["id"], matched_text=concept.get("name"),
            match_type="revealed_withheld" if withheld else "surfaced_unreached_shown",
            score=cscore,
        )

    # PASS 2 — area, enriched with its concrete concerns (key_questions + sub_bullets)
    pillar_name, ascore = _match_area(probe)
    if pillar_name:
        withheld = _pillar_is_withheld(pillar_name)
        return KBMatch(
            pillar=pillar_name, pillar_is_withheld=withheld, level="pillar",
            match_type="revealed_withheld" if withheld else "surfaced_unreached_shown",
            score=ascore,
        )

    # PASS 3 — novel
    return KBMatch(level="none", match_type="novel_not_in_kb")

def resolve_removal_target(
    user_text: str,
    *,
    last_discussed: KBMatch | None = None,
    shown_bullets: list[str] | None = None,
) -> KBMatch:
    """Resolve WHAT a removal targets. Deictic/positional refs resolve deterministically
    against passed-in context (NO LLM, identical across arms — I-7); named targets fall
    through to locate(). `last_discussed` is the structured focus the agent recorded on
    its previous turn (walkthrough arms: the current pillar/concept; BlackBox after a full
    re-render: typically None -> needs_disambiguation rather than a silent pillar guess)."""
    text = (user_text or "").strip()
    norm = _norm(text)
    shown_bullets = shown_bullets or []

    # 1 — deictic "remove this / it / that": resolve to what was last discussed.
    if norm in _DEICTIC or any(norm == d or norm.endswith(" " + d) for d in _DEICTIC):
        if last_discussed is not None and last_discussed.level != "none":
            return last_discussed
        return KBMatch(level="none", needs_disambiguation=True)

    # 2 — positional "the second point" against the currently-shown bullets.
    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", norm) and shown_bullets:
            try:
                bullet = shown_bullets[idx]
            except IndexError:
                return KBMatch(level="none", needs_disambiguation=True)
            return KBMatch(level="concept", matched_text=_strip_source_refs(bullet),
                           match_type="surfaced_unreached_shown")

    # 3 — named target: same KB resolution as an add.
    return locate(text)


@dataclass
class SubPointSlot:
    pillar: str
    position: str = "end"   # only "end" today; KB keeps insertion order


def place_sub_point(item: str, pillar: str) -> SubPointSlot:
    """Slot for a NEW sub-point (`item`) inside an already-chosen `pillar` (distinct from
    'find existing'). Trivial today — append; `item` is unused for now but kept in the
    signature per §3.2. Watch-item: if this ever needs the same LLM pass as locate(),
    merge them rather than keep two prompts."""
    return SubPointSlot(pillar=pillar, position="end")
