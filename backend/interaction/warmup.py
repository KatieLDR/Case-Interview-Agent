"""Warm-up interaction engine — a miniature of the main-phase contract.

The warm-up practice ("moving to a new city") trains participants on the exact
limitations they meet in the measured main phase, so it mirrors that contract instead
of the old free-form LLM merge:

  • only **add** and **remove** mutate the plan; every other request gets a fixed
    "you can only add or remove" reply (rephrasing is unavailable),
  • all shown content is grounded in an immutable mini KB (``warmup_kb``) — a withheld
    area reveals with its *canonical* KB wording, never the user's phrasing,
  • a novel point triggers a placement choice (own area vs. under an existing one),
  • a removal is challenged (yes/no) before it applies,
  • the plan is re-rendered deterministically from KB − exclusions + verbatim user adds.

This module is self-contained on purpose: it does NOT import the main-phase
``handlers`` / ``domain`` / ``knowledge_base`` (those are wired to the case KB). It
reuses only the KB-independent intent router (``classify_intent``) and the shared
``classify_json`` funnel. State lives in ``WarmupState`` on the agent instance.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.knowledge import warmup_kb as wkb
from backend.llm import classify_json
from backend.interaction.intents import classify_intent
from backend.interaction.prompts.warmup import WARMUP_LOCATE_PROMPT, WARMUP_PC_CLASSIFIER

_USER_PILLAR_EMOJI = "📌"
_LLM_LOCATE_FLOOR = 0.6

# ── Participant-facing copy (draft — researcher reviews exact wording) ──────────
_LIMITATION = (
    "In this practice you can do two things: **add** a point "
    "(e.g. *add: check school districts*) or **remove** one "
    "(e.g. *remove the bank account point*). Rephrasing or other changes aren't "
    "available — just like in the main task. What would you like to add or remove?"
)
_REVEAL = ("Good thinking — that's worth its own pillar. Here's what it could cover:")
# Mirror of the main phase's ASK_PC (agents/prompts/base.py): asked once per add — is
# this a top-level pillar, or a bullet under an existing pillar?
_ASK_PC = (
    "Is **{item}** a new **pillar** (a top-level area of the plan), or a **bullet** "
    "under an existing pillar?\n\n"
    "*Reply **pillar**, name the pillar it belongs under, or say **bullet** if you're "
    "not sure which pillar yet.*"
)
# Mirror of the main phase's ASK_WORDING (agents/prompts/base.py): the framework's
# standard points are literature-anchored, but the user's OWN contribution may keep
# their wording — yes -> canonical, no -> theirs.
_ASK_WORDING = (
    "Your point **{user}** matches a standard framework bullet, usually phrased as "
    "**{kb}**. Reply **yes** to use the standard wording, or **no** to keep your own."
)
_ASK_WORDING_PILLAR = (
    "Your pillar **{user}** matches a standard part of the framework, usually called "
    "**{kb}**. Reply **yes** to use the standard name, or **no** to keep your own."
)
_WORDING_STANDARD = "Got it — using the standard wording. ✍️"
_WORDING_KEPT = "Got it — keeping your wording. ✍️"
_WORDING_REASK = (
    "Sorry, I didn't catch that — **yes** to use the standard wording "
    "**{kb}**, or **no** to keep **{user}**."
)
_READDED = "Okay, I've put that back in."
_PLACEMENT_Q = (
    "Which pillar should **{item}** go under — {names}? "
    "*(reply e.g. \"under {first}\", or \"its own pillar\" to make it a new pillar)*"
)
_ADDED = "Added. 👍"
_ALREADY = "Good news — that's already in the plan (see **{pillar}**)."
_REMOVE_CHALLENGE = "Just to confirm — should I remove **{target}** from the plan? *(yes / no)*"
_REMOVE_RECONFIRM = "Sorry, I didn't catch that — should I remove **{target}**? *(yes / no)*"
_REMOVED = "Removed. ✂️"
_KEPT = "Okay, keeping it."
_REMOVE_MISS = (
    "I couldn't find that in the current plan — you can only remove points that are "
    "shown above."
)
_ONE_AT_A_TIME = "Let's take one at a time — starting with **{item}**.\n\n"


@dataclass
class WarmupState:
    revealed: list[str] = field(default_factory=list)           # withheld pillars surfaced, in order
    user_pillars: list[str] = field(default_factory=list)       # novel pillars, verbatim, in order
    user_points: dict[str, list[str]] = field(default_factory=dict)   # pillar name -> verbatim bullets
    excluded_bullets: dict[str, list[str]] = field(default_factory=dict)
    excluded_pillars: list[str] = field(default_factory=list)
    pending_removal: dict | None = None    # {"kind": "pillar"|"bullet", "pillar": str, "bullet": str|None}
    pending_pc: dict | None = None         # {"item": str, "prefix": str} — awaiting area/point answer
    pending_placement: str | None = None   # verbatim novel item awaiting placement
    pending_wording: dict | None = None    # {"kind": "pillar"|"bullet", "pillar": str, ...}
    wording_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    # pillar -> {norm(canonical bullet): user's chosen wording} (ASK_WORDING 'no')
    interacted: bool = False               # True once any add/remove/reveal succeeded
    last_reply: str = ""                   # previous engine reply (intent-router context)


# ── Text normalisation ─────────────────────────────────────────────────────────
_STOP = {
    "the", "a", "an", "to", "in", "of", "for", "and", "or", "we", "i", "you",
    "should", "do", "does", "is", "are", "how", "what", "our", "us", "it", "that",
    "this", "add", "remove", "delete", "drop", "please", "can", "could", "would",
    "also", "new", "area", "areas", "pillar", "pillars", "section", "sections",
    "point", "points", "bullet", "about", "consider", "include", "need", "want",
    "think", "let", "lets", "on", "put", "under", "into",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def _tokens(text: str) -> set[str]:
    return {w for w in _norm(text).split() if w and w not in _STOP}


def _bullet_matches(user_text: str, bullet: str) -> bool:
    un, bn = _norm(user_text), _norm(bullet)
    if not un or not bn:
        return False
    if un in bn or bn in un:
        return True
    ut, bt = _tokens(user_text), _tokens(bullet)
    if not bt or not ut:
        return False
    overlap = len(ut & bt)
    return overlap >= max(1, (len(bt) + 1) // 2)


def _pillar_name_matches(user_text: str, pillar_name: str) -> bool:
    ut, pt = _tokens(user_text), _tokens(pillar_name)
    if not ut or not pt:
        return False
    return ut <= pt or pt <= ut


# ── Plan rendering (deterministic — no LLM) ─────────────────────────────────────
def _pillar_visible_bullets(state: WarmupState, name: str, kb_sub_bullets: list[str]) -> list[str]:
    excl = {_norm(b) for b in state.excluded_bullets.get(name, [])}
    ov = state.wording_overrides.get(name, {})
    out = [ov.get(_norm(b), b) for b in kb_sub_bullets if _norm(b) not in excl]
    out += [b for b in state.user_points.get(name, []) if _norm(b) not in excl]
    return out


def _visible_pillars(state: WarmupState) -> list[dict]:
    """Ordered visible areas: shown KB pillars → revealed withheld → user pillars,
    each minus current exclusions. Shared by rendering and matching."""
    out: list[dict] = []
    for p in wkb.get_shown_pillars():
        if p["name"] in state.excluded_pillars:
            continue
        out.append({"name": p["name"], "emoji": p["emoji"],
                    "bullets": _pillar_visible_bullets(state, p["name"], p["sub_bullets"])})
    for name in state.revealed:
        if name in state.excluded_pillars:
            continue
        p = wkb.get_pillar_by_name(name)
        if not p:
            continue
        out.append({"name": p["name"], "emoji": p["emoji"],
                    "bullets": _pillar_visible_bullets(state, p["name"], p["sub_bullets"])})
    for name in state.user_pillars:
        if name in state.excluded_pillars:
            continue
        out.append({"name": name, "emoji": _USER_PILLAR_EMOJI,
                    "bullets": _pillar_visible_bullets(state, name, [])})
    return out


def render_plan(state: WarmupState) -> str:
    lines = ["**Your plan so far:**\n"]
    for p in _visible_pillars(state):
        lines.append(f"{p['emoji']} **{p['name']}**")
        for b in p["bullets"]:
            lines.append(f"- {b}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _with_plan(msg: str, state: WarmupState) -> str:
    return f"{msg}\n\n---\n\n{render_plan(state)}"


# ── Visibility + matching ───────────────────────────────────────────────────────
def _pillar_currently_visible(state: WarmupState, name: str) -> bool:
    if name in state.excluded_pillars:
        return False
    p = wkb.get_pillar_by_name(name)
    if p:
        return bool(p.get("shown", False)) or name in state.revealed
    return name in state.user_pillars


def _locate(state: WarmupState, text: str, *, visible_only: bool) -> dict | None:
    """Deterministic match of `text` to a warm-up item. Returns a dict with
    level/pillar/bullet/withheld/excluded, or None. `visible_only` restricts to
    currently displayed items (used for removal)."""
    # KB bullets first (most specific)
    for p in wkb.get_all_pillars():
        name = p["name"]
        if visible_only and not _pillar_currently_visible(state, name):
            continue
        excl_set = {_norm(x) for x in state.excluded_bullets.get(name, [])}
        ov = state.wording_overrides.get(name, {})
        for b in p["sub_bullets"]:
            excluded = _norm(b) in excl_set
            if visible_only and excluded:
                continue
            override = ov.get(_norm(b))
            if _bullet_matches(text, b) or (override and _bullet_matches(text, override)):
                return {"level": "bullet", "pillar": name, "bullet": b, "source": "kb",
                        "display": override or b,
                        "withheld": not p.get("shown", False), "excluded": excluded}
    # user-added bullets
    for pname, bl in state.user_points.items():
        if visible_only and not _pillar_currently_visible(state, pname):
            continue
        excl_set = {_norm(x) for x in state.excluded_bullets.get(pname, [])}
        for b in bl:
            excluded = _norm(b) in excl_set
            if visible_only and excluded:
                continue
            if _bullet_matches(text, b):
                return {"level": "bullet", "pillar": pname, "bullet": b, "source": "user",
                        "withheld": False, "excluded": excluded}
    # pillar names (KB)
    for p in wkb.get_all_pillars():
        name = p["name"]
        if visible_only and not _pillar_currently_visible(state, name):
            continue
        if _pillar_name_matches(text, name):
            return {"level": "pillar", "pillar": name, "bullet": None, "source": "kb",
                    "withheld": not p.get("shown", False),
                    "excluded": name in state.excluded_pillars}
    # user pillar names
    for name in state.user_pillars:
        if visible_only and not _pillar_currently_visible(state, name):
            continue
        if _pillar_name_matches(text, name):
            return {"level": "pillar", "pillar": name, "bullet": None, "source": "user",
                    "withheld": False, "excluded": name in state.excluded_pillars}
    return None


def _locate_llm(state: WarmupState, text: str) -> dict | None:
    """LLM fallback used only for `add` when the deterministic match finds nothing —
    catches semantic matches (e.g. 'meeting new people' → Social). Maps a confident
    hit back onto a KB pillar/bullet; on any error returns None (treated as novel)."""
    block = "\n".join(
        f"- {p['name']}: {', '.join(p['sub_bullets'])}" for p in wkb.get_all_pillars()
    )
    try:
        parsed = classify_json(WARMUP_LOCATE_PROMPT.format(pillars_block=block, user_text=text))
    except Exception as e:
        logging.warning(f"[WARMUP locate] classifier error: {e} — treating as novel")
        return None
    try:
        conf = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf < _LLM_LOCATE_FLOOR:
        return None
    pillar_name = parsed.get("pillar")
    if not isinstance(pillar_name, str) or not pillar_name.strip():
        return None
    p = wkb.get_pillar_by_name(pillar_name.strip())
    if not p:
        return None
    bullet = parsed.get("bullet")
    bullet = bullet.strip() if isinstance(bullet, str) and bullet.strip() else None
    # Snap the LLM's bullet back to a canonical KB bullet when possible.
    if bullet:
        bullet = next((b for b in p["sub_bullets"] if _bullet_matches(bullet, b)), bullet)
    name = p["name"]
    excl = _norm(bullet) in {_norm(x) for x in state.excluded_bullets.get(name, [])} if bullet else \
        name in state.excluded_pillars
    return {"level": "bullet" if bullet else "pillar", "pillar": name, "bullet": bullet,
            "source": "kb", "withheld": not p.get("shown", False), "excluded": excl}


# ── Mutation helpers ────────────────────────────────────────────────────────────
def _titlecase_first(s: str) -> str:
    s = (s or "").strip()
    return s[:1].upper() + s[1:] if s else s


def _add_user_point(state: WarmupState, pillar: str, text: str) -> None:
    bl = state.user_points.setdefault(pillar, [])
    if _norm(text) not in {_norm(b) for b in bl}:
        bl.append(text)


def _add_user_pillar(state: WarmupState, name: str) -> None:
    name = _titlecase_first(name)
    if not any(_norm(name) == _norm(p["name"]) for p in _visible_pillars(state)):
        state.user_pillars.append(name)


def _reveal_pillar(state: WarmupState, name: str) -> None:
    """Surface a withheld KB pillar (with its canonical bullets), clearing any prior
    exclusion. No-op if already visible."""
    p = wkb.get_pillar_by_name(name)
    if not p:
        return
    if name in state.excluded_pillars:
        state.excluded_pillars.remove(name)
    if not p.get("shown", False) and name not in state.revealed:
        state.revealed.append(name)


def _bullet_match_in_pillar(pillar: str, text: str) -> str | None:
    """If `text` matches one of `pillar`'s canonical KB bullets, return that bullet."""
    p = wkb.get_pillar_by_name(pillar)
    if not p:
        return None
    return next((b for b in p["sub_bullets"] if _bullet_matches(text, b)), None)


def _resolve_pillar_for_add(state: WarmupState, name: str) -> str | None:
    """Resolve a parent-area name for an add, revealing a withheld KB pillar if that's
    what the user named. Returns the canonical pillar name, or None if unknown."""
    p = wkb.get_pillar_by_name(name)
    if not p:
        loc = _locate(state, name, visible_only=False)
        if loc and loc["level"] == "pillar":
            p = wkb.get_pillar_by_name(loc["pillar"])
            if not p:
                return loc["pillar"]  # a user pillar
        else:
            for un in state.user_pillars:
                if _norm(un) == _norm(name):
                    return un
            return None
    if not p.get("shown", False) and p["name"] not in state.revealed:
        state.revealed.append(p["name"])
    if p["name"] in state.excluded_pillars:
        state.excluded_pillars.remove(p["name"])
    return p["name"]


# ── Confirmation classification (mirrors handlers._classify_confirmation) ────────
_CONFIRM_FIRST = {"yes", "yep", "yeah", "yup", "sure", "confirm", "confirmed", "ok",
                  "okay", "absolutely", "definitely", "go", "proceed", "remove", "delete"}
_DECLINE_FIRST = {"no", "nope", "nah", "keep", "cancel", "stop", "leave", "dont", "never"}
_DECLINE_PHRASES = ("never mind", "actually keep", "do not", "keep it", "dont remove")


def _classify_confirmation(text: str) -> str:
    n = re.sub(r"\s+", " ", (text or "").strip().lower()).strip("?.!, ")
    words = re.findall(r"[a-z']+", n)
    joined = " ".join(words)
    if words:
        if words[0] in _DECLINE_FIRST or any(joined.startswith(p) for p in _DECLINE_PHRASES):
            return "decline"
        if words[0] in _CONFIRM_FIRST:
            return "confirm"
    try:
        parsed = classify_json(
            "A user was asked to confirm removing part of a plan "
            "(reply 'yes' to remove, 'no' to keep).\n"
            "Classify their reply as one of: confirm | decline | other (a question or "
            "something unrelated).\n"
            'Respond ONLY with JSON, no markdown: {"decision": "confirm|decline|other"}\n\n'
            f'Reply: "{text}"'
        )
        d = parsed.get("decision", "other")
        return d if d in ("confirm", "decline", "other") else "other"
    except Exception as e:
        logging.warning(f"[WARMUP confirm] {e} -> other")
        return "other"


# ── Placement parsing (mirrors handlers._placement_choice) ──────────────────────
_OWN_AREA = ("own area", "separate area", "new area", "its own", "on its own", "own pillar",
             "separate pillar", "new pillar", "new section", "standalone", "by itself",
             "separate", "new topic", "different area", "own thing", "own")


def _placement_choice(state: WarmupState, text: str) -> tuple[str, str | None]:
    n = _norm(text)
    if any(p in n for p in _OWN_AREA):
        return ("own", None)
    mobj = re.search(r"\b(?:under|in|to|put it in|add it to)\s+(.+)$", n)
    if mobj:
        pillar = _resolve_visible_pillar(state, mobj.group(1).strip())
        if pillar:
            return ("named", pillar)
    # A bare area name in reply to the placement question.
    direct = _resolve_visible_pillar(state, text)
    if direct:
        return ("named", direct)
    return ("unclear", None)


def _resolve_visible_pillar(state: WarmupState, name: str) -> str | None:
    loc = _locate(state, name, visible_only=True)
    if loc and loc["level"] == "pillar":
        return loc["pillar"]
    return None


# ── Pillar-or-point classification (mirrors agents/base.py _classify_pc) ─────────
# Exact one/two-word replies the ASK_PC prompt tells users to type (bare "area" /
# "point" aren't substrings of the multi-word phrases, so match them explicitly).
_PC_PILLAR_EXACT = {"area", "pillar", "section", "own area", "new area", "a new area",
                    "own", "new one", "top level", "top-level"}
_PC_POINT_EXACT = {"point", "bullet", "detail", "sub-point", "a point", "a bullet"}
_PC_PILLAR_WORDS = ("new area", "own area", "its own", "an area", "as an area", "a section",
                    "new section", "separate area", "top level", "top-level", "pillar",
                    "category", "theme")
_PC_POINT_WORDS = ("point", "bullet", "sub-point", "subpoint", "sub point", "detail",
                   "under ", "below ")


def _classify_pc(state: WarmupState, reply: str, item: str) -> tuple[str, str | None]:
    """Map an ASK_PC reply to ('pillar', None) or ('bullet', parent_or_None). Deterministic
    fast-paths first; LLM only when ambiguous. Never 'unclear' — the safe default is
    'bullet' (which converges at the placement ask that itself offers 'its own area')."""
    low = (reply or "").strip().lower().strip("?.! ")
    if low in _PC_PILLAR_EXACT:
        return ("pillar", None)
    if low in _PC_POINT_EXACT:
        return ("bullet", None)
    # A named currently-visible area (not phrased as a NEW one) -> a point under it.
    known = _resolve_visible_pillar(state, reply)
    if known and not any(w in low for w in ("new area", "own area", "its own", "separate")):
        return ("bullet", known)
    if (any(w in low for w in _PC_PILLAR_WORDS)
            and not any(w in low for w in ("point", "bullet", "under "))):
        return ("pillar", None)
    if any(w in low for w in _PC_POINT_WORDS):
        parent = None
        mobj = re.search(r"\bunder\s+(.+)$", low)
        if mobj:
            parent = _resolve_visible_pillar(state, mobj.group(1).strip())
        return ("bullet", parent)
    try:
        p = classify_json(WARMUP_PC_CLASSIFIER.format(item=item, reply=reply))
        if p.get("branch") == "pillar":
            return ("pillar", None)
        par = p.get("parent")
        par = _resolve_visible_pillar(state, par) if isinstance(par, str) and par.strip() else None
        return ("bullet", par)
    except Exception as e:
        logging.warning(f"[WARMUP pc] classifier error: {e} -> bullet")
        return ("bullet", None)


# ── Add flow — mirrors the main phase: ASK_PC → wording → render ─────────────────
def _handle_add(state: WarmupState, item: str, parent: str | None, prefix: str) -> str:
    """Entry for an `add`. An explicit "under Y" skips ASK_PC straight to the point
    branch (like the main phase's default_parent); otherwise ask area-or-point first."""
    item = (item or "").strip()
    if not item:
        return _LIMITATION

    if parent:
        dest = _resolve_pillar_for_add(state, parent)
        if dest:
            return _commit_point_under(state, dest, item, prefix)

    state.pending_pc = {"item": item, "prefix": prefix}
    return prefix + _ASK_PC.format(item=item)


def _resolve_pc(state: WarmupState, reply_text: str) -> str:
    """Resolve the ASK_PC reply into the pillar branch or the point branch."""
    pc = state.pending_pc
    item, prefix = pc["item"], pc.get("prefix", "")
    state.pending_pc = None
    branch, parent = _classify_pc(state, reply_text, item)
    if branch == "pillar":
        return _handle_add_pillar(state, item, prefix)
    return _handle_add_point(state, item, parent, prefix)


def _handle_add_pillar(state: WarmupState, item: str, prefix: str) -> str:
    """User says `item` is a top-level area. If it maps to a KB pillar, offer the
    wording choice (standard name vs their own) BEFORE revealing — mirrors
    _af_pillar_branch. Otherwise create a verbatim user area."""
    loc = _locate(state, item, visible_only=False) or _locate_llm(state, item)
    kb_pillar = loc["pillar"] if (loc and wkb.get_pillar_by_name(loc["pillar"])) else None

    if kb_pillar:
        if _pillar_currently_visible(state, kb_pillar):
            return _with_plan(prefix + _ALREADY.format(pillar=kb_pillar), state)
        if _norm(item) != _norm(kb_pillar):
            state.pending_wording = {"kind": "pillar", "pillar": kb_pillar, "user_text": item}
            return prefix + _ASK_WORDING_PILLAR.format(user=item, kb=kb_pillar)
        _reveal_pillar(state, kb_pillar)
        state.interacted = True
        return _with_plan(prefix + _REVEAL, state)

    _add_user_pillar(state, item)
    state.interacted = True
    return _with_plan(prefix + _ADDED, state)


def _handle_add_point(state: WarmupState, item: str, parent: str | None, prefix: str) -> str:
    """User says `item` is a point. With a named area → commit there; otherwise locate
    it in the KB (reveal + wording ask on a withheld match, re-add an excluded one, note
    a duplicate, or fall to the placement ask for a genuinely novel point)."""
    if parent:
        return _commit_point_under(state, parent, item, prefix)

    loc = _locate(state, item, visible_only=False) or _locate_llm(state, item)

    if loc is None:
        state.pending_placement = item
        return prefix + _placement_prompt(state, item)

    pillar = loc["pillar"]

    if loc["withheld"] and pillar not in state.revealed and pillar not in state.excluded_pillars:
        # Ask before revealing/rendering when there's a rephrase question to ask;
        # _resolve_wording does the reveal once the user answers (reveal_pillar=True).
        if loc["level"] == "bullet" and loc["bullet"] and _norm(item) != _norm(loc["bullet"]):
            state.pending_wording = {"kind": "bullet", "pillar": pillar,
                                     "kb_bullet": loc["bullet"], "user_text": item,
                                     "reveal_pillar": True}
            return prefix + _ASK_WORDING.format(user=item, kb=loc["bullet"])
        _reveal_pillar(state, pillar)
        state.interacted = True
        return _with_plan(prefix + _REVEAL, state)

    if loc["excluded"]:
        if loc["level"] == "bullet" and loc["bullet"] is not None:
            excl = state.excluded_bullets.get(pillar, [])
            state.excluded_bullets[pillar] = [b for b in excl if _norm(b) != _norm(loc["bullet"])]
        else:
            if pillar in state.excluded_pillars:
                state.excluded_pillars.remove(pillar)
        state.interacted = True
        return _with_plan(prefix + _READDED, state)

    return _with_plan(prefix + _ALREADY.format(pillar=pillar), state)


def _commit_point_under(state: WarmupState, dest: str, item: str, prefix: str) -> str:
    """Add `item` under a resolved area `dest`. If it matches one of dest's canonical KB
    bullets with different wording, offer the wording choice; a same-wording match is a
    duplicate; otherwise store the user's point verbatim."""
    match = _bullet_match_in_pillar(dest, item)
    if match:
        if _norm(item) == _norm(match):
            return _with_plan(prefix + _ALREADY.format(pillar=dest), state)
        state.pending_wording = {"kind": "bullet", "pillar": dest,
                                 "kb_bullet": match, "user_text": item}
        return prefix + _ASK_WORDING.format(user=item, kb=match)
    _add_user_point(state, dest, item)
    state.interacted = True
    return _with_plan(prefix + _ADDED, state)


def _placement_prompt(state: WarmupState, item: str) -> str:
    names = ", ".join(p["name"] for p in _visible_pillars(state))
    first = _visible_pillars(state)[0]["name"] if _visible_pillars(state) else "Housing"
    return _PLACEMENT_Q.format(item=item, names=names, first=first)


def _handle_remove(state: WarmupState, target: str) -> str:
    target = (target or "").strip()
    loc = _locate(state, target, visible_only=True)
    if loc is None:
        llm = _locate_llm(state, target)
        if llm and _pillar_currently_visible(state, llm["pillar"]) and not llm["excluded"]:
            loc = llm
    if loc is None:
        return _with_plan(_REMOVE_MISS, state)

    display = (loc.get("display") or loc["bullet"]) if loc["level"] == "bullet" else loc["pillar"]
    state.pending_removal = {"kind": loc["level"], "pillar": loc["pillar"],
                             "bullet": loc["bullet"], "display": display}
    return _REMOVE_CHALLENGE.format(target=display)


def _resolve_removal(state: WarmupState, reply_text: str) -> str:
    pa = state.pending_removal
    display = pa.get("display") or (pa["bullet"] if pa["kind"] == "bullet" else pa["pillar"])
    decision = _classify_confirmation(reply_text)

    if decision == "decline":
        state.pending_removal = None
        return _with_plan(_KEPT, state)

    if decision == "confirm":
        if pa["kind"] == "bullet" and pa["bullet"] is not None:
            bl = state.excluded_bullets.setdefault(pa["pillar"], [])
            if _norm(pa["bullet"]) not in {_norm(b) for b in bl}:
                bl.append(pa["bullet"])
        else:
            if pa["pillar"] not in state.excluded_pillars:
                state.excluded_pillars.append(pa["pillar"])
        state.pending_removal = None
        state.interacted = True
        return _with_plan(_REMOVED, state)

    return _REMOVE_RECONFIRM.format(target=display)  # keep pending, re-ask


def _resolve_wording(state: WarmupState, reply_text: str) -> str:
    """ASK_WORDING resolution — mirrors _af_pillar_wording_reply in agents/base.py.
    Pillar kind: confirm -> reveal the canonical area; decline -> create the user's
    own area. Bullet kind: confirm -> canonical stays; decline -> the user's wording
    takes that bullet's slot (via wording_overrides). Anything else re-asks."""
    pw = state.pending_wording
    kind = pw.get("kind", "bullet")
    decision = _classify_confirmation(reply_text)
    kb_label = pw["pillar"] if kind == "pillar" else pw["kb_bullet"]

    if decision not in ("confirm", "decline"):
        return _WORDING_REASK.format(kb=kb_label, user=pw["user_text"])

    state.pending_wording = None
    state.interacted = True

    if kind == "pillar":
        if decision == "confirm":
            _reveal_pillar(state, pw["pillar"])
            return _with_plan(_WORDING_STANDARD, state)
        _add_user_pillar(state, pw["user_text"])
        return _with_plan(_WORDING_KEPT, state)

    # bullet kind
    if pw.get("reveal_pillar"):
        _reveal_pillar(state, pw["pillar"])
    if decision == "confirm":
        return _with_plan(_WORDING_STANDARD, state)
    state.wording_overrides.setdefault(pw["pillar"], {})[_norm(pw["kb_bullet"])] = pw["user_text"]
    return _with_plan(_WORDING_KEPT, state)


def _handle_placement(state: WarmupState, reply_text: str) -> str:
    item = state.pending_placement
    choice, named = _placement_choice(state, reply_text)

    if choice == "own":
        _add_user_pillar(state, item)
        state.pending_placement = None
        state.interacted = True
        return _with_plan(_ADDED, state)

    if choice == "named" and named:
        state.pending_placement = None
        return _commit_point_under(state, named, item, "")

    # Unclear — keep pending and re-ask.
    return _placement_prompt(state, item)


# ── Entry point ─────────────────────────────────────────────────────────────────
def warmup_turn(state: WarmupState, user_text: str) -> str:
    """Handle one warm-up message and return the reply markdown (the frontend appends
    the '✅ Done' prompt). Ordering mirrors handlers.dispatch(): resolve a pending
    removal, then a pending area/point (ASK_PC) answer, then a pending wording choice,
    then a pending placement, then route the fresh intent."""
    reply = _turn_inner(state, user_text)
    state.last_reply = reply
    return reply


def _turn_inner(state: WarmupState, user_text: str) -> str:
    if state.pending_removal is not None:
        return _resolve_removal(state, user_text)

    if state.pending_pc is not None:
        return _resolve_pc(state, user_text)

    if state.pending_wording is not None:
        return _resolve_wording(state, user_text)

    if state.pending_placement is not None:
        return _handle_placement(state, user_text)

    visible_names = ", ".join(p["name"] for p in _visible_pillars(state)) or "(none)"
    visible_bullets = "\n".join(
        f"- {b}" for p in _visible_pillars(state) for b in p["bullets"]
    ) or "(none)"
    ir = classify_intent(
        user_text,
        current_pillar="(the practice plan)",
        current_bullets=visible_bullets,
        walkthrough_pillars=visible_names,
        last_agent=state.last_reply or "(nothing yet)",
    )

    if ir.error:
        return _LIMITATION

    if ir.intent in ("add", "revisit"):
        prefix = ""
        item = ir.detail or user_text
        if ir.multi and len(ir.items) >= 2:
            item = ir.items[0]
            prefix = _ONE_AT_A_TIME.format(item=item)
        # revisit names an existing area → straight to the area branch (no ASK_PC).
        if ir.intent == "revisit":
            return _handle_add_pillar(state, item, prefix)
        return _handle_add(state, item, ir.parent, prefix)

    if ir.intent == "remove":
        return _handle_remove(state, ir.detail or user_text)

    # question / ask_agent_to_suggest / advance / doubt / none — no rephrase, no suggest.
    return _LIMITATION
