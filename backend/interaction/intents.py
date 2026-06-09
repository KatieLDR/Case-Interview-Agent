"""backend/interaction/intents.py  —  Step 3 of REFACTOR_PLAN.md (unified intent taxonomy).

ONE classifier, ONE taxonomy (I-2), fired from the shared layer for every arm (I-1).
Pure module: NO agent state, NO `self`. The turn context the router needs is passed
by value. Dependency direction is one-way:  agent -> intents -> llm.
This module must never import from any agent, and never resolves a KB target.

WHY THIS EXISTS — the Step-0 audit found three incompatible classifiers, not one:
  * BlackBox : a cascade (_detect_override `redo|concept_excluded|concept_added|none`
               -> _classify_question y/n). No advance/doubt/revisit/suggest. "anything
               else" -> a pillar literally named "else" (D6 catastrophe, F-I1/F-I2).
  * Explainable : one context-enriched single-pass router (advance|add|remove|question|
               doubt|redo|none). revisit folded into add; no ask_agent_to_suggest.
               -> THIS is the base we generalise from.
  * HITL : buttons + state-flags, plus one narrow free-text classifier
               (suggestion|guidance|sub_point) gated behind the proactive prompt — AND
               it still calls the inherited _detect_override on normal turns, so one add
               reaches matching via TWO entry points and double-counts add_pillar (F-M4).

This module replaces all three with a single taxonomy:

    add | remove | question | ask_agent_to_suggest | revisit | doubt | advance | none

DESIGN DECISIONS (locked 2026-06-08, see REFACTOR_PLAN §S Step-3 block):
  * D-Q1  Buttons bypass the LLM and map straight to the enum (BUTTON_INTENT). Free-text
          goes through `classify_intent`. The HITL "free-text an action -> nudge to the
          button instead of executing" behaviour is PERSONA RENDERING of the shared intent
          (a Step-4 render seam), NOT a second classifier — so classification stays shared.
  * D-Q2  INTENT ONLY. The router says *what* the user wants and extracts a raw `detail`
          phrase; it does NOT resolve which KB element that is. Resolution is the dispatch
          step calling `domain.matching.locate()` (I-3). `detail` stays an unresolved
          string, exactly as all three arms produce today.

  * `redo` is REMOVED (§0): users may not wipe a session. "start over"/"restart"/"reset"
    classify as `none` -> the agent renders its fallback.
  * Filler-word guard (F-I1/F-I2): "else / more / other / another / anything / something"
    can never be a concept name. Belt-and-braces with `matching.locate()`'s own guard.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from backend.llm import classify_json

# ── Taxonomy (I-2) ──────────────────────────────────────────────────────────
INTENTS = frozenset(
    {"add", "remove", "question", "ask_agent_to_suggest",
     "revisit", "doubt", "advance", "none"}
)

# Default on classifier error: `question` is the safe fallback (never silently
# advances or removes). The result is flagged `error=True` so Step-5 logging can
# keep LLM-error fallbacks OUT of the question count (F-I3).
_FALLBACK_INTENT = "question"


@dataclass
class IntentResult:
    intent: str            # one of INTENTS
    detail: str | None     # raw noun phrase to add/remove/revisit — UNRESOLVED (D-Q2)
    confidence: float
    error: bool = False    # True iff this is an LLM-error fallback (F-I3; do not log as a real question)
    parent: str | None = None  # explicit "add X under Y" destination — raw, UNRESOLVED (D-Q2)


# ── D-Q1: deterministic button -> intent (no LLM) ───────────────────────────
# HITL exposes these as buttons; app.py maps a clicked button to one of these
# names. Other arms have no buttons and never hit this map.
BUTTON_INTENT = {
    "approve":  "advance",   # ✅ approve concept / move on
    "advance":  "advance",
    "next":     "advance",
    "reject":   "remove",    # ❌ reject concept
    "remove":   "remove",    # ➖ remove point
    "revisit":  "revisit",   # ↩️ revisit a past pillar
    "add":      "add",       # ➕ add a point
}


def intent_for_button(button_name: str) -> str | None:
    """Map a deterministic UI button to a shared intent (D-Q1). None if unknown
    (e.g. confirm/cancel buttons, which are handled by the confirmation machine,
    not the intent taxonomy)."""
    return BUTTON_INTENT.get((button_name or "").strip().lower())


# ── Filler-word guard (F-I1 / F-I2) ─────────────────────────────────────────
# A bare filler is NEVER a concept. These can appear as the whole message
# ("anything else?") or as an extracted detail ("else"); both are caught.
_FILLER_WORDS = {
    "else", "more", "other", "others", "another", "anything",
    "something", "etc", "stuff", "things", "anything else",
    "something else", "what else", "anything more",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip("?.!, ")


def _is_filler_detail(detail: str | None) -> bool:
    """True if `detail` is empty or a pure filler token (no real concept in it)."""
    if not detail:
        return True
    return _norm(detail) in _FILLER_WORDS


# ── The one router prompt (derived from Explainable's INTENT_ROUTER_PROMPT) ──
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
- add vs question: proposing something to include -> add; asking to understand what is
  already shown -> question.
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


def classify_intent(
    user_text: str,
    *,
    current_pillar: str = "(none)",
    current_bullets: str = "(none)",
    walkthrough_pillars: str = "(none)",
    last_agent: str = "(nothing yet)",
) -> IntentResult:
    """Classify ONE free-text turn into the unified taxonomy (D-Q2: intent only).

    All context is passed by value (pure function). The caller builds the context
    strings from its own state — exactly the kwargs Explainable already assembles.

    Returns an IntentResult whose `detail` is a RAW phrase, never a resolved KB
    target. The dispatch step then calls `domain.matching.locate(detail or user_text)`.

    Buttons do NOT call this — see `intent_for_button` (D-Q1).
    """
    prompt = INTENT_ROUTER_PROMPT.format(
        current_pillar=current_pillar or "(none)",
        current_bullets=current_bullets or "(none)",
        walkthrough_pillars=walkthrough_pillars or "(none)",
        last_agent=last_agent or "(nothing yet)",
        user_msg=user_text,
    )

    try:
        parsed = classify_json(prompt)
    except Exception as e:  # transport/parse error — classify_json raises by policy
        logging.warning(f"[INTENT] classifier error: {e} — defaulting to {_FALLBACK_INTENT}")
        return IntentResult(intent=_FALLBACK_INTENT, detail=None, confidence=0.0, error=True)

    intent = parsed.get("intent", _FALLBACK_INTENT)
    if intent not in INTENTS:
        intent = _FALLBACK_INTENT

    # GUARD: a non-string detail (model occasionally returns a list/dict) must not
    # reach the filler check, which calls .strip() — coerce anything non-str to None.
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

    # GUARD: confidence parse is outside the classify_json try/except; a non-numeric
    # value ("high") would crash the turn here. Default to 0.0 on any parse failure.
    try:
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    # ── Filler-word guard (F-I1 / F-I2): defence in depth with matching.locate() ──
    # 1. A filler can never be a concept -> drop it from detail.
    if _is_filler_detail(detail):
        detail = None
    # 2. An ADD with no real concept left is the "else"-bug shape (BlackBox once made a
    #    pillar called "else"): "what else could we add" == asking the agent to suggest.
    if intent == "add" and detail is None:
        logging.info(f"[INTENT] '{user_text[:60]}' -> add with filler/empty detail; "
                     f"rerouting to ask_agent_to_suggest")
        intent = "ask_agent_to_suggest"
    # 3. A REMOVE or REVISIT with no target cannot act (a targetless "delete"/"go back").
    #    Rerouting these to *suggest* would be a category error, so fall to none ->
    #    the agent renders its fallback (non-destructive). Deictic words ("this"/"it"/
    #    "that") are NOT filler, so "remove this" keeps its detail and reaches Fork-A.
    elif intent in {"remove", "revisit"} and detail is None:
        logging.info(f"[INTENT] '{user_text[:60]}' -> {intent} with no target; falling to none")
        intent = "none"

    # advance/doubt/none/ask_agent_to_suggest never carry a target concept.
    if intent in {"advance", "doubt", "none", "ask_agent_to_suggest"}:
        detail = None
    # parent is meaningful ONLY for an explicit "add X under Y" — never elsewhere,
    # never a filler token.
    if intent != "add" or _is_filler_detail(parent):
        parent = None

    logging.info(f"[INTENT] '{user_text[:60]}' -> {intent} (detail={detail!r}, parent={parent!r}, conf={confidence:.2f})")
    return IntentResult(intent=intent, detail=detail, parent=parent, confidence=confidence, error=False)
