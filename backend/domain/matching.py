from __future__ import annotations

import re
from dataclasses import dataclass

from backend.domain.grounding import _strip_source_refs
from backend.knowledge import knowledge_base as kb
from backend.llm import (
    classify_json, ADD_MATCH_THRESHOLD, CONCEPT_MATCH_THRESHOLD
)
from backend.domain.prompts.matching import (
    ADD_PILLAR_MATCH_PROMPT, ADD_CONCEPT_MATCH_PROMPT,
    ADD_MATCH_PROMPT, LOCATE_AREA_PROMPT
)

CONCEPT_GENERIC_FLOOR = 0.92

_GENERIC_CACHE = None
_REF_RE = re.compile(r"\s*\[[a-z]\]")
_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b")

def _drop_articles(s: str) -> str:
    """Lowercased text with articles (a/an/the) removed and whitespace collapsed —
    for comparing a concept name to a key-question/sub-bullet lead robustly."""
    return re.sub(r"\s+", " ", _ARTICLE_RE.sub(" ", s)).strip()
_DEICTIC = {
    "this", "it", "that", "this one", "that one", "the last one",
    "the one above", "the last point", "the above", "remove this",
}
_FILLER = {"else", "more", "other", "others", "anything", "something", "etc", "and so on"}
_GLUE = {"the", "a", "an", "to", "we", "i", "you", "is", "are", "any", "of", "for",
         "please", "can", "could", "would", "do", "what", "should", "and", "or",
         "add", "remove", "delete", "include", "consider", "want", "like", "think",
         "also", "so", "on", "about", "let", "us"}
_ORDINALS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1,
}

@dataclass
class KBMatch:
    pillar: str | None = None
    pillar_is_withheld: bool = False
    level: str = "none"
    concept_id: str | None = None
    matched_text: str | None = None
    match_type: str | None = None
    score: float = 0.0
    needs_disambiguation: bool = False

@dataclass
class SubPointSlot:
    pillar: str
    position: str = "end"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", _REF_RE.sub("", text or "")).strip().lower()


def is_excluded_bullet(excluded_sub_bullets: dict, pillar_name: str, bullet: str) -> bool:
    removed = {_norm(b) for b in (excluded_sub_bullets or {}).get(pillar_name, [])}
    return _norm(bullet) in removed


def _pillar_is_withheld(pillar_name: str | None) -> bool:
    if not pillar_name:
        return False
    p = next((p for p in kb.get_all_pillars()
              if p["name"].lower() == pillar_name.lower()), None)
    return bool(p) and not p.get("shown", True)


def _pillar_id_for_name(pillar_name: str | None) -> str | None:
    if not pillar_name:
        return None
    p = next((p for p in kb.get_all_pillars()
              if p["name"].lower() == pillar_name.lower()), None)
    return p["id"] if p else None


def match_pillar(item: str) -> tuple[str | None, float]:
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


def _content_tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", _norm(text or "")) if t not in _GLUE}


def _generic_tokens() -> set:
    global _GENERIC_CACHE
    if _GENERIC_CACHE is None:
        counts: dict = {}
        for c in kb.get_all_concepts():
            if c.get("swap"):
                continue
            for t in _content_tokens(c.get("name", "")):
                counts[t] = counts.get(t, 0) + 1
        _GENERIC_CACHE = {t for t, n in counts.items() if n >= 2}
    return _GENERIC_CACHE


def _only_generic_overlap(item: str, concept: dict) -> bool:
    item_toks = _content_tokens(item)
    cand_toks = (_content_tokens(concept.get("name", ""))
                 | _content_tokens(concept.get("explanation", "")))
    return len((item_toks & cand_toks) - _generic_tokens()) == 0


def match_concept(item: str, *, pillar_id: str | None = None,
                  apply_generic_floor: bool = True) -> tuple[dict | None, float]:
    concepts = [c for c in kb.get_all_concepts()
                if not c.get("swap", False) and c.get("pillar_id")
                and (pillar_id is None or c.get("pillar_id") == pillar_id)]
    if not concepts:
        return None, 0.0
    block = "\n".join(f"{i}. {c['name']}: {(c.get('explanation') or '').strip()}"
                      for i, c in enumerate(concepts))
    try:
        parsed = classify_json(ADD_CONCEPT_MATCH_PROMPT.format(item=item, concepts=block))
        conf = parsed.get("confidence", 0.0)
        if parsed.get("matched"):
            idx = parsed.get("matched_index")
            if idx is not None and 0 <= idx < len(concepts):
                cand = concepts[idx]
                floor = CONCEPT_MATCH_THRESHOLD
                if apply_generic_floor and _only_generic_overlap(item, cand):
                    floor = CONCEPT_GENERIC_FLOOR
                if conf >= floor:
                    return cand, conf
    except Exception as e:
        print(f"[matching._match_concept] {e}")
    return None, 0.0


def match_key_question(item: str, pillar_name: str) -> tuple[str | None, float]:
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


def _match_area(item: str) -> tuple[str | None, float]:
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


def pillar_gist(name: str) -> str:
    p = next((p for p in kb.get_all_pillars()
              if p["name"].lower() == (name or "").lower()), None)
    if not p or not p.get("description"):
        return ""
    first = p["description"].split(". ")[0].strip().rstrip(".")
    return first + "." if first else ""


def normalize_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def concept_bullet(concept_id: str, *, refs: bool = True) -> str | None:
    c = kb.get_concept_by_id(concept_id)
    if not c:
        return None
    name = (c.get("name") or "").strip()
    pillar = kb.get_pillar_by_id(c.get("pillar_id"))
    bullet = None
    if pillar and name:
        # Compare leads with articles (a/an/the) dropped, so authored drift between
        # a concept's name and its key-question lead (e.g. "by third-party provider"
        # vs "by a third-party provider") doesn't defeat the match and silently lose
        # the citation in the fallback below.
        ncanon = _drop_articles(name.lower())

        def _lead_match(text: str) -> bool:
            lead = _drop_articles(_strip_source_refs(text).split(":", 1)[0].strip().lower())
            return bool(lead) and (lead == ncanon or ncanon.startswith(lead) or lead.startswith(ncanon))

        # Prefer a displayed sub-bullet; otherwise fall back to the concept's key
        # question, which carries its source ref ([x]). Without this fallback a
        # concept that isn't one of the shown sub-bullets resolves to the bare
        # name and loses its citation when added.
        bullet = next((b for b in pillar.get("sub_bullets", []) if _lead_match(b)), None)
        if bullet is None:
            bullet = next((q for q in pillar.get("key_questions", []) if _lead_match(q)), None)
    if bullet is None:
        bullet = name or None
    if bullet is None:
        return None
    return bullet if refs else _strip_source_refs(bullet)


def canonical_add_bullet(text: str, *, refs: bool = True) -> str | None:
    km = locate(text)
    if km.level == "concept" and km.concept_id:
        return concept_bullet(km.concept_id, refs=refs)
    return None


def locate(user_text: str) -> KBMatch:
    text = (user_text or "").strip()
    toks = [t for t in re.findall(r"[a-z0-9]+", _norm(text)) if t not in _GLUE]
    if not text or not toks or all(t in _FILLER for t in toks):
        return KBMatch(level="none", match_type="novel_not_in_kb")

    probe = text.lower()

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

    pillar_name, ascore = _match_area(probe)
    if pillar_name:
        pidf = _pillar_id_for_name(pillar_name)
        if pidf:
            sub, sscore = match_concept(probe, pillar_id=pidf, apply_generic_floor=False)
            if sub:
                cwithheld = _pillar_is_withheld(pillar_name)
                return KBMatch(
                    pillar=pillar_name, pillar_is_withheld=cwithheld, level="concept",
                    concept_id=sub["id"], matched_text=sub.get("name"),
                    match_type="revealed_withheld" if cwithheld else "surfaced_unreached_shown",
                    score=sscore,
                )
        withheld = _pillar_is_withheld(pillar_name)
        return KBMatch(
            pillar=pillar_name, pillar_is_withheld=withheld, level="pillar",
            match_type="revealed_withheld" if withheld else "surfaced_unreached_shown",
            score=ascore,
        )

    return KBMatch(level="none", match_type="novel_not_in_kb")


def resolve_removal_target(
    user_text: str,
    *,
    last_discussed: KBMatch | None = None,
    shown_bullets: list[str] | None = None,
) -> KBMatch:
    text = (user_text or "").strip()
    norm = _norm(text)
    shown_bullets = shown_bullets or []

    if norm in _DEICTIC or any(norm == d or norm.endswith(" " + d) for d in _DEICTIC):
        if last_discussed is not None and last_discussed.level != "none":
            return last_discussed
        return KBMatch(level="none", needs_disambiguation=True)

    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", norm) and shown_bullets:
            try:
                bullet = shown_bullets[idx]
            except IndexError:
                return KBMatch(level="none", needs_disambiguation=True)
            return KBMatch(level="concept", matched_text=_strip_source_refs(bullet),
                           match_type="surfaced_unreached_shown")

    return locate(text)


def place_sub_point(item: str, pillar: str) -> SubPointSlot:
    return SubPointSlot(pillar=pillar, position="end")
