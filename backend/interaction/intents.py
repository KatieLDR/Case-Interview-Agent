from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from backend.llm import classify_json

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

INTENT_ROUTER_PROMPT = """\
You route a user's message during a framework problem-solving session.
Return the SINGLE best intent. Resolve only WHAT the user wants — do NOT try to
identify WHICH framework element they mean (a separate step does that).

Intents:
- "add"      : wants to ADD a new point or area — including proposals phrased as
               "we should consider X", "we need to think about X", "what about X",
               "how about X", "can we also look at X", "add X", "include X".
               detail = the NEW thing to add (a noun phrase). A leading "no" does
               NOT cancel an add ("no, add X" is still add).
- "remove"   : a CLEAR command to remove a whole area OR one specific point
               ("remove X", "drop this", "take out the part about Y", "delete that").
               detail = what to remove.
- "question" : asking to UNDERSTAND something already shown ("what is X?", "why is
               this here?", "how does this apply?", "can you explain?").
- "ask_agent_to_suggest" : asking the AGENT to propose what to add/consider next —
               the user is NOT naming their own idea, they want the agent's.
               ("what else should we consider?", "any suggestions from you?",
               "what would you add?", "am I missing anything?", "what should come
               next?" used as 'what do you suggest', "anything else?").
               detail = null ALWAYS (the user named no concept of their own).
- "revisit"  : wants to RETURN to / GO BACK to / LOOK AGAIN at an area already
               covered earlier — pure NAVIGATION, no new content. Triggers include
               "let's go back to X", "go back to X", "return to X", "can we revisit X",
               "let's look at X again", "take me back to X". This IS a valid action:
               NEVER classify a clear "go back to X" as none or advance.
               detail = the named area, ALWAYS extract it ("go back to Strategic Fit"
               -> detail "Strategic Fit"). If the user ALSO names a NEW point to put
               there, that is "add" (detail = the new point), not revisit.
- "doubt"    : vaguely doubts the current point belongs, WITHOUT a clear remove
               command ("I'm not sure this fits", "this seems off", "is this
               necessary?"). detail = null.
- "advance"  : ready to move on to the next area, with no other request
               ("yes", "ok", "next", "move on", "continue", "makes sense",
               "got it", "sounds good"). detail = null.
- "none"     : none of the above / unclear / a request to RESTART or WIPE the
               session ("start over", "redo this", "reset", "regenerate everything")
               — restarting is not allowed; classify those as none. detail = null.

KEY DISAMBIGUATION:
- ask_agent_to_suggest vs add: if the user names THEIR OWN concept to include -> add
  (detail = that concept). If the user asks the agent to come up with it -> 
  ask_agent_to_suggest (detail = null). "what about regulatory risk?" names a concept
  -> add. "what else should we look at?" names none -> ask_agent_to_suggest.
- ask_agent_to_suggest vs advance: "what should come next?" / "what's next?" asks the
  agent to SUGGEST -> ask_agent_to_suggest. A bare "next" / "move on" is advance.
- "else", "more", "other", "anything", "something" are NEVER a concept. If the whole
  message is just "anything else?" / "what else?" -> ask_agent_to_suggest, detail null.
  NEVER put "else"/"more"/etc into detail.
- delegation/steering vs add: if the user HANDS THE NEXT MOVE to you and names no
  concept of their own ("you decide", "you lead", "guide it", "your call", "you pick",
  "up to you", "you take it from here") -> ask_agent_to_suggest, detail null. These are
  NEVER an add, and "guide"/"lead"/"drive"/"decide" are NEVER concept names. (The agent
  often invites this — "shall I take the lead?", "would you like me to suggest one?" —
  so the reply is the user accepting the agent's lead, not naming an area.)
- add vs question: proposing to INCLUDE something -> add ("add X", "we should consider X",
  "what about X"). But an INTERROGATIVE asking to UNDERSTAND how/why a concept relates to,
  fits, applies to, connects with, or belongs under an area is "question", even when it
  names BOTH a concept and an area ("how is X related to Y", "how does X fit (into) Y",
  "can you explain how X fits Y", "why does X matter for Y", "where does X belong"). The
  verbs "fit"/"fits"/"fit into", "relate(d) to", "apply", "connect", "belong" inside a
  "how/why/can you explain …?" frame signal a question about an existing relationship, NOT
  a request to place X under Y — classify as question, parent = null. Only an imperative or
  proposal to INCLUDE X is add.
- Naming an existing area AS THE PLACE for a new point is "add", not revisit — the area
  is just the location, the new point is the thing being added. detail = the new point.
  ("Let's revisit Strategic Fit, we need to understand the opportunity frequency"
   -> add, detail "opportunity frequency".)
- "parent": ONLY for add — the EXISTING area the user names as the destination via
  "under" / "as part of" / "within" / "in" ("add data privacy under Feasibility"
  -> detail "data privacy", parent "Feasibility"). null when no destination is named;
  null for every non-add intent.
- revisit vs none/advance: a request to RETURN to a NAMED area already covered is
  ALWAYS revisit (detail = that area) — never none, never advance. A bare "next" /
  "move on" with no named target is advance; "go back to Strategic Fit" is revisit.
- doubt vs remove: doubt is hesitation with NO command; remove is a clear instruction.

─── CURRENT PILLAR ───────────────────────────────────────────────────────────
{current_pillar}
Its points:
{current_bullets}
────────────────────────────────────────────────────────────────────────────
─── PILLARS COVERED SO FAR (for revisit) ─────────────────────────────────────
{walkthrough_pillars}
────────────────────────────────────────────────────────────────────────────
─── WHAT THE AGENT LAST SAID ─────────────────────────────────────────────────
{last_agent}
────────────────────────────────────────────────────────────────────────────
─── USER MESSAGE ─────────────────────────────────────────────────────────────
{user_msg}
────────────────────────────────────────────────────────────────────────────

Respond ONLY with valid JSON, no markdown:
{{"intent": "add|remove|question|ask_agent_to_suggest|revisit|doubt|advance|none", "detail": "string or null", "parent": "string or null", "confidence": float}}
"""


@dataclass
class IntentResult:
    intent: str
    detail: str | None
    confidence: float
    error: bool = False
    parent: str | None = None


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

    logging.info(f"[INTENT] '{user_text[:60]}' -> {intent} (detail={detail!r}, parent={parent!r}, conf={confidence:.2f})")
    return IntentResult(intent=intent, detail=detail, parent=parent, confidence=confidence, error=False)
