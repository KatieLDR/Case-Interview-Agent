from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.llm import classify_json
from backend.interaction.prompts.intents import INTENT_ROUTER_PROMPT

INTENTS = frozenset(
    {"add", "remove", "question", "ask_agent_to_suggest",
     "revisit", "doubt", "advance", "none"}
)
BUTTON_INTENT = {
    "approve":  "advance",
    "advance":  "advance",
    "next":     "advance",
    "reject":   "remove",
    "remove":   "remove",
    "revisit":  "revisit",
    "add":      "add",
}

_FALLBACK_INTENT = "question"
_FILLER_WORDS = {
    "else", "more", "other", "others", "another", "anything",
    "something", "etc", "stuff", "things", "anything else",
    "something else", "what else", "anything more",
}
_STEERING_PHRASES = {
    "guide it", "guide me", "guide us", "you guide", "you guide it",
    "please guide", "please guide me", "please lead",
    "you lead", "you lead it", "you're lead", "you're the lead",
    "youre the lead", "you are the lead", "lead it", "lead the way",
    "take the lead", "you take the lead",
    "you decide", "you decide it", "you decide what's next",
    "you decide whats next", "you choose", "you pick",
    "you drive", "you drive it", "drive it",
    "your call", "up to you", "it's up to you", "its up to you",
    "you take it from here", "take it from here",
    "whatever you think", "whatever you suggest",
    "you suggest", "you tell me",
}

@dataclass
class IntentResult:
    intent: str
    detail: str | None
    confidence: float
    error: bool = False
    parent: str | None = None
    # Multi-point safeguard (only meaningful for an "add" with ≥2 separable items)
    multi: bool = False
    items: list[str] = field(default_factory=list)
    pillar_name: str | None = None


def intent_for_button(button_name: str) -> str | None:
    return BUTTON_INTENT.get((button_name or "").strip().lower())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip("?.!, ")


def _is_filler_detail(detail: str | None) -> bool:
    if not detail:
        return True
    return _norm(detail) in _FILLER_WORDS


def _is_steering_message(text: str) -> bool:
    return _norm(text) in _STEERING_PHRASES


# Deterministic structural floor for multi-point detection: the LLM occasionally
# misses an obvious list, so explicit list formats (Header: a, b / bullets / multi
# short lines) are caught here regardless. The LLM still handles inline prose
# ("culture fit and team morale"); this only guarantees explicit lists fire.
_BULLET_RE = re.compile(r"^\s*(?:[-*•·–—]|\d+[.)])\s+")
_INLINE_SPLIT_RE = re.compile(r"\s*(?:,|;|/|\band\b|&)\s*", re.I)


def _split_inline(s: str) -> list[str]:
    return [p.strip() for p in _INLINE_SPLIT_RE.split(s or "") if p.strip()]


def _looks_like_point(line: str) -> bool:
    t = (line or "").strip()
    return bool(t) and not t.endswith("?") and len(t.split()) <= 10


def _structural_list(text: str) -> tuple[str | None, list[str]]:
    """(header, items) when the message is an explicit list, else (None, []).
    Header is a short colon-label or a short non-bullet line above bullets."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return None, []
    # Case A: "Header: a, b" (colon label on the first line) + any following lines.
    m = re.match(r"^\s*([^:\n]{1,60}?):\s*(.*)$", lines[0])
    if m and len(m.group(1).split()) <= 6:
        items = _split_inline(m.group(2))
        items += [_BULLET_RE.sub("", l).strip() for l in lines[1:]]
        items = [i for i in items if i]
        if len(items) >= 2:
            return m.group(1).strip(), items
    # Case B/C: ≥2 bullet lines, optional short header line above them.
    bullets = [l for l in lines if _BULLET_RE.match(l)]
    if len(bullets) >= 2:
        header = None
        first_b = next(i for i, l in enumerate(lines) if _BULLET_RE.match(l))
        if first_b == 1 and not _BULLET_RE.match(lines[0]) and len(lines[0].split()) <= 6:
            header = lines[0].rstrip(":").strip()
        return header, [_BULLET_RE.sub("", b).strip() for b in bullets]
    # Case D: ≥2 short, point-like plain lines (no header identifiable).
    if len(lines) >= 2 and all(_looks_like_point(l) for l in lines):
        return None, lines
    return None, []


def classify_intent(
    user_text: str,
    *,
    current_pillar: str = "(none)",
    current_bullets: str = "(none)",
    walkthrough_pillars: str = "(none)",
    last_agent: str = "(nothing yet)",
) -> IntentResult:
    prompt = INTENT_ROUTER_PROMPT.format(
        current_pillar=current_pillar or "(none)",
        current_bullets=current_bullets or "(none)",
        walkthrough_pillars=walkthrough_pillars or "(none)",
        last_agent=last_agent or "(nothing yet)",
        user_msg=user_text,
    )

    try:
        parsed = classify_json(prompt)
    except Exception as e:
        logging.warning(f"[INTENT] classifier error: {e} — defaulting to {_FALLBACK_INTENT}")
        # The LLM is down, but an explicit list must still fire the safeguard.
        s_header, s_items = _structural_list(user_text)
        if len(s_items) >= 2:
            return IntentResult(intent="add", detail=s_items[0], confidence=0.0, error=True,
                                multi=True, items=s_items, pillar_name=s_header)
        return IntentResult(intent=_FALLBACK_INTENT, detail=None, confidence=0.0, error=True)

    intent = parsed.get("intent", _FALLBACK_INTENT)
    if intent not in INTENTS:
        intent = _FALLBACK_INTENT

    detail = parsed.get("detail")
    if not isinstance(detail, str):
        detail = None
    else:
        detail = detail.strip() or None
    parent = parsed.get("parent")
    if not isinstance(parent, str):
        parent = None
    else:
        parent = parent.strip() or None

    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    # Multi-point safeguard fields
    multi_raw = bool(parsed.get("multi", False))
    items = parsed.get("items")
    if not isinstance(items, list):
        items = []
    items = [i.strip() for i in items if isinstance(i, str) and i.strip()]
    pillar_name = parsed.get("pillar")
    pillar_name = pillar_name.strip() if isinstance(pillar_name, str) and pillar_name.strip() else None
    # Seed detail from the first item so the add/filler reroute below doesn't misfire
    # on a multi-add whose single detail the model left null.
    if multi_raw and items and not detail:
        detail = items[0]

    if _is_steering_message(user_text):
        if intent != "ask_agent_to_suggest":
            logging.info(f"[INTENT] '{user_text[:60]}' -> steering/delegation; "
                         f"rerouting {intent} -> ask_agent_to_suggest")
        intent, detail, parent = "ask_agent_to_suggest", None, None

    if _is_filler_detail(detail):
        detail = None
    if intent == "add" and detail is None:
        logging.info(f"[INTENT] '{user_text[:60]}' -> add with filler/empty detail; "
                     f"rerouting to ask_agent_to_suggest")
        intent = "ask_agent_to_suggest"
    elif intent in {"remove", "revisit"} and detail is None:
        logging.info(f"[INTENT] '{user_text[:60]}' -> {intent} with no target; falling to none")
        intent = "none"

    if intent in {"advance", "doubt", "none", "ask_agent_to_suggest"}:
        detail = None
    if intent != "add" or _is_filler_detail(parent):
        parent = None

    # Multi is only meaningful for an add with ≥2 separable items; otherwise drop it
    # (and its pillar hint) so downstream never treats a single point as a block.
    multi = multi_raw and intent == "add" and len(items) >= 2
    if not multi:
        items = []
        pillar_name = None

    # Structural floor: an explicit list always fires multi, even if the LLM missed
    # it. Skip when the LLM read a removal (don't turn "remove: a, b" into an add).
    if intent != "remove":
        s_header, s_items = _structural_list(user_text)
        if len(s_items) >= 2:
            intent, multi = "add", True
            if len(items) < 2:
                items = s_items
            if not pillar_name and s_header:
                pillar_name = s_header

    logging.info(f"[INTENT] '{user_text[:60]}' -> {intent} "
                 f"(detail={detail!r}, parent={parent!r}, conf={confidence:.2f}, "
                 f"multi={multi}, items={items}, pillar={pillar_name!r})")
    return IntentResult(intent=intent, detail=detail, parent=parent, confidence=confidence,
                        error=False, multi=multi, items=items, pillar_name=pillar_name)
