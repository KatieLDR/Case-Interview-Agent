from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from backend.domain import grounding as g
from backend.domain import matching as m
from backend.knowledge import knowledge_base as kb
from backend.llm import classify_json

@dataclass
class AddOutcome:
    action: str
    pillar: str | None
    level: str
    counted: bool
    explanation: str | None = None
    source: str = "user_spontaneous"
    matched_pillar_id: str | None = None
    text: str | None = None
    also_covered: str | None = None
    matched_text: str | None = None
    navigate_bullet: str | None = None
    navigate_concept_id: str | None = None

@dataclass
class RemovalOutcome:
    stage: str
    target: str | None
    level: str
    pillar: str | None = None
    needs_justification: bool = False
    justification: str | None = None
    consequence_facts: list = field(default_factory=list)
    post_delete_branch: bool = False
    is_swap: bool = False
    suggest_add_alternative: str | None = None

@dataclass
class SuggestOutcome:
    level: str
    suggested_item: str
    grounding: str | None = None
    accepted: bool = False
    revealed: bool = False

@dataclass
class QuestionOutcome:
    target_level: str | None
    target: str | None
    grounding: str | None = None
    is_about_swap: bool = False

@dataclass
class AdvanceOutcome:
    passive: bool = False
    elicited: bool = False

@dataclass
class FallbackOutcome:
    reason: str = "unclear"

Outcome = (
    AddOutcome | RemovalOutcome | SuggestOutcome
    | QuestionOutcome | AdvanceOutcome | FallbackOutcome
)

@dataclass
class PendingAction:
    type: str
    target: str
    level: str
    pillar: str | None = None
    justification: str | None = None
    requires_justification: bool = False
    is_swap: bool = False
    consequence_facts: list = field(default_factory=list)


def is_meaningful_justification(text: str) -> bool:
    t = (text or "").strip()
    words = [w for w in t.split() if any(ch.isalpha() for ch in w)]
    return len(words) >= 3 and len(t) >= 12


class HandlerSession(Protocol):
    pending: PendingAction | None
    pending_suggestion: dict | None
    last_discussed: m.KBMatch | None
    shown_bullets: list[str]
    excluded_concepts: list[str]
    excluded_sub_bullets: dict[str, list[str]]

    def presented_pillars(self) -> list[str]: ...
    def presented_sub_bullets(self) -> dict[str, list[str]]: ...
    def surfaced_pillar_names(self) -> set[str]: ...
    def current_pillar(self) -> str | None: ...

    def surface_pillar(self, name: str) -> None: ...
    def add_sub_point(self, pillar: str, text: str) -> None: ...

    def swap_name(self) -> str | None: ...
    def is_swap_target(self, km: m.KBMatch, user_text: str) -> bool: ...
    def mark_swap_detected(self) -> None: ...

    def requires_justification(self, km: m.KBMatch) -> bool: ...


_CONFIRM_FIRST = {"yes", "yep", "yeah", "yup", "sure", "confirm", "confirmed", "ok",
                  "okay", "absolutely", "definitely", "go", "proceed", "remove", "delete"}
_DECLINE_FIRST = {"no", "nope", "nah", "keep", "cancel", "stop", "leave", "dont", "never"}
_DECLINE_PHRASES = ("never mind", "actually keep", "do not", "keep it", "dont remove")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip("?.!, ")


def _classify_confirmation(text: str) -> str:
    words = re.findall(r"[a-z']+", _norm(text))
    joined = " ".join(words)
    if words:
        if words[0] in _DECLINE_FIRST or any(joined.startswith(p) for p in _DECLINE_PHRASES):
            return "decline"
        if words[0] in _CONFIRM_FIRST:
            return "confirm"
    try:
        parsed = classify_json(
            "A user was asked to confirm removing part of a framework "
            "(reply 'yes' to remove, 'no' to keep).\n"
            "Classify their reply as one of: confirm | decline | other (a question or "
            "something unrelated).\n"
            'Respond ONLY with JSON, no markdown: {"decision": "confirm|decline|other"}\n\n'
            f'Reply: "{text}"'
        )
        d = parsed.get("decision", "other")
        return d if d in ("confirm", "decline", "other") else "other"
    except Exception as e:                       # transport/parse error -> safe: re-ask
        logging.warning(f"[handlers._classify_confirmation] {e} -> other")
        return "other"


def _affirms(text: str) -> bool:
    return _classify_confirmation(text) == "confirm"


def _accepts_offer(text: str) -> bool:
    if _affirms(text):
        return True
    w = re.findall(r"[a-z']+", _norm(text))
    if w and w[0] in ("add", "include", "keep"):
        rest = w[1:]
        return (not rest) or rest[0] in ("it", "that", "this", "them", "one", "please")
    return False


def _add_accepts_suggestion(ps: dict, intent: str, detail, parent, user_text: str) -> bool:
    if intent != "add":
        return False
    item = _norm(ps.get("item", ""))
    if not item:
        return False
    cands = [_norm(c) for c in (detail, parent) if c]
    if not cands:                                # contentless 'add it' / 'include it' -> accept
        return True
    for c in cands:
        if c == item or (len(c) >= 4 and (c in item or item in c)):
            return True
    return False


_OWN_AREA = ("own area", "separate area", "new area", "its own", "on its own", "own pillar",
             "separate pillar", "new pillar", "new section", "standalone", "by itself",
             "separate", "new topic", "different area", "own thing")
_HERE = ("keep it here", "under this", "this area", "this pillar", "right here", "under it",
         "where we are", "current area", "under current", "keep here", "here", "current")

_PILLAR_NAME_STOP = {"area", "areas", "pillar", "pillars", "section", "sections",
                     "topic", "topics", "part", "the", "a", "an", "and", "or", "of"}

def _names_pillar(text: str, pillar_name: str | None) -> bool:
    """True when the user's text is essentially the pillar's NAME (a request to
    navigate there), vs a content phrase that merely matched the pillar. Used to
    decide whether surfacing the area should count as pillar agency."""
    if not pillar_name:
        return False
    def _toks(s: str) -> set:
        return {t for t in re.findall(r"[a-z0-9]+", _norm(s)) if t not in _PILLAR_NAME_STOP}
    t, p = _toks(text), _toks(pillar_name)
    return bool(t) and t <= p


def _concept_already_in_pillar(session, km) -> bool:
    """True when concept `km` is already a displayed point under its pillar — a static
    KB sub-bullet or one the user already added. Lets a point that an on-screen pillar
    doesn't yet contain still count as a new sub-bullet (vs re-raising a shown one)."""
    if not km.pillar:
        return False
    kbp = next((p for p in kb.get_all_pillars()
                if p["name"].lower() == km.pillar.lower()), None)
    bullet = (m.concept_bullet(km.concept_id, refs=False) if km.concept_id else None) or km.matched_text
    if not bullet:
        return False
    key = g._strip_source_refs(bullet).strip().lower()
    statics = [g._strip_source_refs(b).strip().lower()
               for b in (kbp.get("sub_bullets", []) if kbp else [])]
    user = [g._strip_source_refs(b).strip().lower()
            for b in getattr(session, "user_sub_points", {}).get(km.pillar, [])]
    return key in statics or key in user


def _placement_choice(text: str):
    n = _norm(text)
    if any(p in n for p in _OWN_AREA):
        return ("own", None)
    mobj = re.search(r"\b(?:under|in|to|put it in|add it to)\s+(.+)$", n)
    if mobj:
        km = m.locate(mobj.group(1).strip())
        if km.level in ("pillar", "concept") and km.pillar:
            return ("named", km.pillar)
    if any(p in n for p in _HERE):
        return ("current", None)
    return ("unclear", None)


def _wants_navigate(text: str) -> bool:
    if _affirms(text):
        return True
    n = _norm(text)
    return n.startswith(("go ", "go", "navigate", "take me", "lets go", "let us go", "jump"))


def _parent_is_explicit(parent: str | None, user_text: str) -> bool:
    if not parent:
        return False
    t = _norm(user_text)
    if any(d in t for d in ("here", "this pillar", "this area", "this section", "this one")):
        return True
    ptoks = [w for w in re.findall(r"[a-z]+", _norm(parent)) if len(w) >= 4]
    return any(w in t for w in ptoks)


def resolve_placement(session: HandlerSession, user_text: str) -> Outcome | None:
    pp = getattr(session, "pending_placement", None) or {}
    session.pending_placement = None
    kind = pp.get("kind")
    item = pp.get("item", "")

    if kind == "navigate":
        if _wants_navigate(user_text):
            if pp.get("reveal_on_accept") and pp.get("target"):
                # Surfacing a WITHHELD pillar via a sub-point: the user contributed a
                # sub-bullet, not a pillar. Reveal the pillar (so it renders) but count
                # the contribution at sub_bullet level — never as a pillar add — to
                # mirror HITL._store_sub_point. The bullet itself is stored/rendered by
                # render_add's `navigated` branch via navigate_bullet.
                session.surface_pillar(pp.get("target"))
                return AddOutcome(action="navigated", pillar=pp.get("target"),
                                  level="sub_bullet", counted=True,
                                  text=pp.get("navigate_bullet") or item,
                                  source="user_spontaneous",
                                  matched_pillar_id=pp.get("matched_pillar_id"),
                                  navigate_bullet=pp.get("navigate_bullet"),
                                  navigate_concept_id=pp.get("concept_id"))
            return AddOutcome(action="navigated", pillar=pp.get("target"), level="pillar",
                              counted=False, source="user_spontaneous",
                              navigate_bullet=pp.get("navigate_bullet"),
                              navigate_concept_id=pp.get("concept_id"))
        if (pp.get("level") == "concept" and session.current_pillar()
                and any(k in _norm(user_text) for k in ("keep", "here", "stay", "leave it"))):
            dest = session.current_pillar()
            slot = m.place_sub_point(item, dest)
            session.add_sub_point(slot.pillar, item)
            return AddOutcome(action="added_new", pillar=slot.pillar, level="sub_bullet",
                              counted=not pp.get("counted_here", False), text=item,
                              source="user_spontaneous")
        if pp.get("level") == "concept" and (pp.get("navigate_bullet") or pp.get("reveal_on_accept")):
            session._carried_navigate = {"target": pp.get("target"),
                                         "navigate_bullet": pp.get("navigate_bullet"),
                                         "concept_id": pp.get("concept_id"),
                                         "reveal_on_accept": pp.get("reveal_on_accept", False),
                                         "matched_pillar_id": pp.get("matched_pillar_id")}
        return None

    if kind == "novel":
        choice, named = _placement_choice(user_text)
        if choice == "own":
            name = m.normalize_name(item)
            session.surface_pillar(name)
            return AddOutcome(action="added_new", pillar=name, level="pillar", counted=True,
                              matched_pillar_id=None, text=name, source="user_spontaneous")
        if choice == "current" and session.current_pillar():
            dest = session.current_pillar()
            slot = m.place_sub_point(item, dest)
            session.add_sub_point(slot.pillar, item)
            return AddOutcome(action="added_new", pillar=slot.pillar, level="sub_bullet",
                              counted=True, text=item, source="user_spontaneous")
        if choice == "named" and named:
            presented = {n.lower() for n in session.presented_pillars()}
            dest = named if named.lower() in presented else m.normalize_name(named)
            if dest.lower() not in presented:
                session.surface_pillar(dest)
            slot = m.place_sub_point(item, dest)
            session.add_sub_point(slot.pillar, item)
            return AddOutcome(action="added_new", pillar=slot.pillar, level="sub_bullet",
                              counted=True, text=item, source="user_spontaneous")
        return None
    return None


def _pillar_id(name: str | None) -> str | None:
    if not name:
        return None
    p = next((p for p in kb.get_all_pillars() if p["name"].lower() == name.lower()), None)
    return p["id"] if p else None


def add_handler(intent: str, km: m.KBMatch, source: str, session: HandlerSession,
                *, text: str, parent: str | None = None) -> AddOutcome:
    if intent == "revisit":
        carry = getattr(session, "_carried_navigate", None)
        session._carried_navigate = None
        if carry and km.pillar and (carry.get("target") or "").lower() == km.pillar.lower():
            if carry.get("reveal_on_accept"):
                session.surface_pillar(km.pillar)
                return AddOutcome(action="navigated", pillar=km.pillar, level="pillar",
                                  counted=True, source=source,
                                  matched_pillar_id=carry.get("matched_pillar_id"),
                                  navigate_bullet=carry.get("navigate_bullet"),
                                  navigate_concept_id=carry.get("concept_id"))
            return AddOutcome(action="navigated", pillar=km.pillar, level="pillar",
                              counted=False, source=source,
                              navigate_bullet=carry.get("navigate_bullet"),
                              navigate_concept_id=carry.get("concept_id"))
        return AddOutcome(action="navigated", pillar=km.pillar, level="pillar",
                          counted=False, source=source)

    presented = {n.lower() for n in session.presented_pillars()}

    if parent:
        if (km.level == "concept" and km.pillar
                and km.pillar.lower() == parent.lower()):
            return AddOutcome(action="duplicate", pillar=parent, level="concept",
                              counted=False, text=text, matched_text=km.matched_text,
                              source=source)
        if parent.lower() not in presented:
            matched, _ = m.match_pillar(parent)
            parent = matched or m.normalize_name(parent)
            if parent.lower() not in presented:
                session.surface_pillar(parent)
        slot = m.place_sub_point(text, parent)
        session.add_sub_point(slot.pillar, text)
        # Count only when the point is genuinely new to the pillar (the agent's
        # add_sub_point dedups + records is_new). A repeat "add X under Y" must not
        # re-count.
        is_new = (getattr(session, "_last_sub_add", None) or {}).get("is_new", True)
        also = (km.pillar if (km.level == "concept" and km.pillar
                              and km.pillar.lower() != slot.pillar.lower()
                              and km.pillar.lower() in presented) else None)
        return AddOutcome(action="added_new", pillar=slot.pillar, level="sub_bullet",
                          counted=bool(is_new), text=text, also_covered=also,
                          explanation=(m.pillar_gist(also) if also else None), source=source)

    if km.level == "concept":
        presented = {n.lower() for n in session.presented_pillars()}
        unreached = bool(km.pillar) and km.pillar.lower() not in presented
        key = (km.concept_id or km.matched_text or text or "").lower()
        seen = getattr(session, "_agency_concepts", None)
        if seen is None:
            seen = set()
            session._agency_concepts = seen
        # Count a sub-bullet contribution when the concept is genuinely new to its
        # pillar: the pillar isn't on screen yet (unreached), OR it's on screen but
        # doesn't yet contain this point. Withheld pillars defer the count to the
        # reveal-on-accept step (EXP) / render (BB). Once per concept via `seen`.
        counted = bool(key and key not in seen and not km.pillar_is_withheld
                       and (unreached or not _concept_already_in_pillar(session, km)))
        if counted:
            seen.add(key)
        expl = ((g.ground_concept(km.concept_id) if km.concept_id else None)
                or m.pillar_gist(km.pillar) or None)
        nav_bullet = m.concept_bullet(km.concept_id, refs=False) if km.concept_id else None
        reveal_on_accept = bool(km.pillar_is_withheld and unreached)
        session.pending_placement = {"kind": "navigate", "target": km.pillar,
                                     "level": "concept", "item": text,
                                     "counted_here": counted,
                                     "concept_id": km.concept_id,
                                     "navigate_bullet": nav_bullet,
                                     "reveal_on_accept": reveal_on_accept,
                                     "matched_pillar_id": _pillar_id(km.pillar) if reveal_on_accept else None}
        return AddOutcome(action="navigate_offer", pillar=km.pillar, level="sub_bullet",
                          counted=counted, matched_text=km.matched_text,
                          explanation=expl, text=text, source="user_spontaneous",
                          navigate_bullet=nav_bullet, navigate_concept_id=km.concept_id)

    if km.level == "pillar":
        if km.pillar_is_withheld and not (km.pillar and km.pillar.lower() in presented):
            session.surface_pillar(km.pillar)
            return AddOutcome(action="revealed", pillar=km.pillar, level="pillar",
                              counted=True, matched_pillar_id=_pillar_id(km.pillar),
                              explanation=g.ground_pillar(km.pillar) or None, source=source)
        # A phrase that matched a pillar by CONTENT (not by naming it) and whose
        # pillar isn't on screen yet: the user surfaced that area early. Credit it
        # once as a pillar engagement (add_pillar), mirroring HITL. Literally naming
        # an in-framework pillar to navigate stays uncounted (counted=False).
        unreached = bool(km.pillar) and km.pillar.lower() not in presented
        counted = False
        if unreached and not _names_pillar(text, km.pillar):
            seen_p = getattr(session, "_agency_pillars", None)
            if seen_p is None:
                seen_p = set()
                session._agency_pillars = seen_p
            if km.pillar.lower() not in seen_p:
                seen_p.add(km.pillar.lower())
                counted = True
        session.pending_placement = {"kind": "navigate", "target": km.pillar,
                                     "level": "pillar", "item": text}
        return AddOutcome(action="navigate_offer", pillar=km.pillar, level="pillar",
                          counted=counted, matched_pillar_id=_pillar_id(km.pillar),
                          explanation=m.pillar_gist(km.pillar) or None,
                          source=source)

    session.pending_placement = {"kind": "novel", "item": text}
    return AddOutcome(action="ask_placement", pillar=None, level="none",
                      counted=False, text=text, source=source)


def _resolve_presented_bullet(needle: str, shown: list[str]) -> str | None:
    if not needle:
        return None
    n = m._norm(needle)
    for b in shown:
        if m._norm(b) == n:
            return b
    for b in shown:
        if m._norm(b.split(":", 1)[0]) == n:
            return b
    if len(n) >= 6:
        for b in shown:
            if n in m._norm(b):
                return b
    return None


def removal_handler(km: m.KBMatch, session: HandlerSession, *, user_text: str) -> RemovalOutcome:
    if km.needs_disambiguation:
        return RemovalOutcome(stage="needs_disambiguation", target=None, level="none")

    swap = session.swap_name()
    if swap and session.is_swap_target(km, user_text):
        req = session.requires_justification(km)
        session.pending = PendingAction(type="remove_pillar", target=swap, level="pillar",
                                        is_swap=True, requires_justification=req)
        return RemovalOutcome(stage="challenged", target=swap, level="pillar",
                              is_swap=True, needs_justification=req)

    if km.level == "none":
        return RemovalOutcome(stage="nothing_to_remove",
                              target=km.matched_text or user_text, level="none")

    if km.level == "pillar":
        presented = {n.lower() for n in session.presented_pillars()}
        if not km.pillar or km.pillar.lower() not in presented:
            offer = km.pillar if (km.pillar and km.pillar_is_withheld) else None
            if offer:
                session.pending_suggestion = {"level": "pillar", "item": offer,
                                              "origin": "remove_offer"}
            return RemovalOutcome(stage="nothing_to_remove", target=km.pillar or user_text,
                                  level="pillar", suggest_add_alternative=offer)
        req = session.requires_justification(km)
        session.pending = PendingAction(type="remove_pillar", target=km.pillar, level="pillar",
                                        requires_justification=req,
                                        consequence_facts=_consequence_facts(km))
        return RemovalOutcome(stage="challenged", target=km.pillar, level="pillar",
                              needs_justification=req,
                              consequence_facts=_consequence_facts(km))

    target = km.matched_text or user_text
    parent = km.pillar
    present_subs = session.presented_sub_bullets()
    by_construction = km is getattr(session, "last_discussed", None)
    presented_pillars = {n.lower() for n in session.presented_pillars()}

    resolved_parent = resolved_bullet = None
    if parent is not None:
        if parent.lower() in presented_pillars and not km.pillar_is_withheld:
            shown = next((v for k, v in present_subs.items() if k.lower() == parent.lower()), [])
            hit = _resolve_presented_bullet(target, shown)
            if hit is not None:
                resolved_parent, resolved_bullet = parent, hit
    else:
        for pn, bullets in present_subs.items():
            hit = _resolve_presented_bullet(target, bullets)
            if hit is not None:
                resolved_parent, resolved_bullet = pn, hit
                break

    if resolved_bullet is not None:
        parent, target = resolved_parent, _resolve_presented_bullet
    elif by_construction and parent is not None:
        pass
    else:
        return RemovalOutcome(stage="nothing_to_remove", target=target, level="concept")  # F-R2b
    bullet = target
    if m.is_excluded_bullet(session.excluded_sub_bullets, parent, bullet):
        return RemovalOutcome(stage="nothing_to_remove", target=bullet, level="concept")
    req = session.requires_justification(km)
    session.pending = PendingAction(type="remove_sub_bullet", target=bullet, level="concept",
                                    pillar=parent, requires_justification=req)
    return RemovalOutcome(stage="challenged", target=bullet, level="concept",
                          needs_justification=req)


def resolve_pending(session: HandlerSession, reply_text: str, *,
                    decision: str | None = None,
                    justification: str | None = None) -> RemovalOutcome:
    pa = session.pending
    if pa is None:
        return RemovalOutcome(stage="abandoned", target=None, level="none")

    if decision is None:
        decision = _classify_confirmation(reply_text)

    if decision == "decline":
        session.pending = None
        return RemovalOutcome(stage="abandoned", target=pa.target, level=pa.level,
                              is_swap=pa.is_swap)

    if decision == "confirm":
        if pa.requires_justification:
            reason = justification if justification is not None else reply_text
            if not is_meaningful_justification(reason):
                return RemovalOutcome(stage="needs_justification", target=pa.target,
                                      level=pa.level, needs_justification=True,
                                      is_swap=pa.is_swap)
            pa.justification = reason
        return _confirm_removal(session, pa)

    return RemovalOutcome(stage="challenged", target=pa.target, level=pa.level,
                          needs_justification=pa.requires_justification, is_swap=pa.is_swap)


def _confirm_removal(session: HandlerSession, pa: PendingAction) -> RemovalOutcome:
    session.pending = None
    if pa.is_swap:
        session.mark_swap_detected()
        return RemovalOutcome(stage="confirmed", target=pa.target, level=pa.level,
                              is_swap=True, justification=pa.justification,
                              post_delete_branch=True)
    if pa.type == "remove_sub_bullet" and pa.pillar:
        session.excluded_sub_bullets.setdefault(pa.pillar, [])
        if not m.is_excluded_bullet(session.excluded_sub_bullets, pa.pillar, pa.target):
            session.excluded_sub_bullets[pa.pillar].append(pa.target)
    else:
        if pa.target.lower() not in [e.lower() for e in session.excluded_concepts]:
            session.excluded_concepts.append(pa.target)
    return RemovalOutcome(stage="confirmed", target=pa.target, level=pa.level,
                          pillar=pa.pillar,
                          justification=pa.justification, post_delete_branch=True,
                          consequence_facts=pa.consequence_facts)


def question_handler(km: m.KBMatch, session: HandlerSession, *, user_text: str) -> QuestionOutcome:
    if km.level == "concept" and km.concept_id:
        grounding = g.ground_concept(km.concept_id) or None
        return QuestionOutcome(target_level="concept", target=km.matched_text,
                               grounding=grounding,
                               is_about_swap=session.is_swap_target(km, user_text))
    if km.level == "pillar":
        return QuestionOutcome(target_level="pillar", target=km.pillar,
                               grounding=g.ground_pillar(km.pillar) or None,
                               is_about_swap=session.is_swap_target(km, user_text))
    return QuestionOutcome(target_level=None, target=None, grounding=None,
                           is_about_swap=session.is_swap_target(km, user_text))


def suggest_handler(session: HandlerSession):
    if session.current_pillar() is not None:
        return advance_handler(session, passive=False, elicited=True)
    return SuggestOutcome(level="pillar", suggested_item=None)


def advance_handler(session: HandlerSession, *, passive: bool = False,
                    elicited: bool = False) -> AdvanceOutcome:
    return AdvanceOutcome(passive=passive, elicited=elicited)


def fallback_handler(reason: str = "unclear") -> FallbackOutcome:
    return FallbackOutcome(reason=reason)


def _consequence_facts(km: m.KBMatch) -> list:
    if not km.pillar:
        return []
    p = next((p for p in kb.get_all_pillars() if p["name"].lower() == km.pillar.lower()), None)
    if not p:
        return []
    return [g._strip_source_refs(q) for q in p.get("key_questions", [])][:3]


def dispatch(intent_result, session: HandlerSession, *, user_text: str,
             source: str = "user_spontaneous") -> Outcome:
    if session.pending is not None:
        return resolve_pending(session, user_text)

    session._carried_navigate = None

    intent = intent_result.intent
    detail = intent_result.detail
    parent = intent_result.parent

    if getattr(session, "pending_placement", None) is not None:
        placed = resolve_placement(session, user_text)
        if placed is not None:
            return placed

    if session.pending_suggestion is not None:
        ps = session.pending_suggestion
        accepting = (_accepts_offer(user_text)
                     or _add_accepts_suggestion(ps, intent, detail, parent, user_text))
        session.pending_suggestion = None
        if accepting:
            if ps.get("origin") == "agent_suggest":
                session.surface_pillar(ps["item"])
                return SuggestOutcome(level=ps["level"], suggested_item=ps["item"],
                                      grounding=g.ground_pillar(ps["item"]) or None,
                                      accepted=True, revealed=True)
            km = m.locate(ps["item"])
            return add_handler("add", km, "user_spontaneous", session, text=ps["item"])

    if intent in ("add", "revisit"):
        if intent == "add" and parent and not _parent_is_explicit(parent, user_text):
            parent = None
        probe = (detail or user_text) if intent == "add" else (parent or detail or user_text)
        km = m.locate(probe)
        return add_handler(intent, km, source, session, text=detail or user_text, parent=parent)

    if intent == "remove":
        target_text = detail or user_text
        km = m.resolve_removal_target(target_text,
                                      last_discussed=session.last_discussed,
                                      shown_bullets=session.shown_bullets)
        return removal_handler(km, session, user_text=target_text)

    if intent == "question":
        km = m.locate(detail or user_text)
        return question_handler(km, session, user_text=detail or user_text)

    if intent == "ask_agent_to_suggest":
        return suggest_handler(session)

    if intent == "advance":
        return advance_handler(session, passive=True)

    if intent == "doubt":
        return fallback_handler(reason="unclear")

    return fallback_handler(reason="start_over" if intent == "none" else "unclear")
