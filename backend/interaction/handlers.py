"""backend/interaction/handlers.py  —  Step 4 of REFACTOR_PLAN.md (handlers + PendingAction).

ONE shared "what happens" layer for every arm (I-1). The router (Step 3) says *what* the
user wants; `domain.matching.locate()` says *which* KB element (I-3); these handlers do the
INVARIANT WORK and return a STRUCTURED OUTCOME — never rendered text. Personas render the
outcome (§3.5). Today removal/add logic is triplicated and divergent (F-R1: EXP logged 3
deletes for 1 deletion, HITL 2 for 0); routing all three through one `removal_handler`
makes the study DVs converge by construction.

PURITY (same sense as matching.py / intents.py): NO `self`, NO agent object, NO Chainlit,
NO logging/Firestore. State is read/written through a neutral `HandlerSession` (D-H3); the
LLM is reached only through the shared `classify_json` (a classifier call, like locate()).
Dependency direction: agent -> handlers -> {matching, grounding, llm, knowledge_base}.

DESIGN DECISIONS (this build; recorded in REFACTOR_PLAN §S):
  * D-H1  Handlers are pure; they DO NOT log. They return an Outcome whose `stage`/`action`
          tells the agent what to log. This is what fixes F-R1 (a `delete` is logged only
          when the agent sees stage="confirmed", never at intent) WITHOUT doing Step 5's
          logging relocation early. §3.5 says "handlers fire the event"; reconciled: in
          Step 4 the handler produces the outcome that DRIVES the firing — the `log_*` call
          moves into the handler in Step 5.
  * D-H2  Justification is an effort-only GATE before confirm (shared
          `is_meaningful_justification`, promoted from HITL `_is_substantive_justification`).
          The GATE MECHANISM lands here (B8/B9). The SCOPE — which decisions require it
          (today HITL's random 2-of-3) and the accept/advance-side justification — stays the
          §3.7 reconciliation parked at Step 6. The handler exposes `requires_justification`;
          wiring decides when it is True.
  * D-H3  Handlers read/write a neutral `HandlerSession`, never arm-specific attrs
          (BB `user_added_pillars` vs HITL `walkthrough_concepts`). Agents adapt to it in
          the wiring steps; BaseAgent owns it in Step 6.

PRESERVATION WALL (§0 #4): the concept-swap MECHANISM is unchanged. A removal that targets
the swap is DETECTION, never a delete event — the per-arm swap-match channel (BB text-match
vs HITL button-on-current-concept) stays in the session adapter (`is_swap_target`); we only
route a uniform `is_swap` signal through the machine. F-S1/F-S3/F-S4 instrument decisions
are out of this step.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from backend.domain import matching as m
from backend.domain import grounding as g
from backend.knowledge import knowledge_base as kb
from backend.llm import classify_json


# ════════════════════════════════════════════════════════════════════════════
#  STRUCTURED OUTCOMES (§3.5) — handlers return these; personas render them.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class AddOutcome:
    action: str                       # revealed | added_new | duplicate | navigated
    pillar: str | None
    level: str                        # pillar | sub_bullet | concept | none
    counted: bool                     # whether this is a logged contribution (DV)
    explanation: str | None = None    # plain KB grounding for a reveal (EXP cites on top)
    source: str = "user_spontaneous"  # I-4 attribution (user_spontaneous | user_elicited)
    matched_pillar_id: str | None = None   # §3.6 add_pillar field (null for a novel area)
    text: str | None = None           # raw user text for add_sub_bullet (F-M6: NOT model output)
    also_covered: str | None = None   # B1.1: a presented pillar where this also lives (note only)
    matched_text: str | None = None   # exact KB text of the matched concept (duplicate messages)
    navigate_bullet: str | None = None   # §2a: canonical bullet (refs stripped) of a recognised
                                         # concept, carried through navigate so a COUNTED concept
                                         # that is not itself a shown sub_bullet still renders on
                                         # arrival (else the agency act leaves no artifact).
    navigate_concept_id: str | None = None   # §2a: the concept id, so an arm can re-derive the
                                             # bullet at its own ref level (EXP keeps refs).


@dataclass
class RemovalOutcome:
    # stage: challenged | confirmed | abandoned | nothing_to_remove |
    #        needs_justification | needs_disambiguation
    stage: str
    target: str | None
    level: str                        # pillar | concept | none
    pillar: str | None = None         # parent pillar of a sub_bullet target — Step 5 §3.6
                                      # attribution for delete_sub_bullet (additive, behaviour-
                                      # neutral; set by _confirm_removal from pa.pillar)
    needs_justification: bool = False     # HITL gate is open (reason still required)
    justification: str | None = None      # the reason that passed the gate (HITL)
    consequence_facts: list = field(default_factory=list)   # EXP counterfactual material
    post_delete_branch: bool = False      # on confirm -> "you suggest / I lead" prompt
    is_swap: bool = False                 # swap target -> detection, NOT a delete event (§0)
    suggest_add_alternative: str | None = None   # B5: "did you mean *add* X?" (offer only)


@dataclass
class SuggestOutcome:
    level: str                        # pillar | concept
    suggested_item: str               # drawn from WITHHELD KB
    grounding: str | None = None      # plain KB grounding (EXP layers a citation on top)
    accepted: bool = False            # D7: user took the agent's suggestion
    revealed: bool = False            # the withheld item is now surfaced (persona renders it)


@dataclass
class QuestionOutcome:
    target_level: str | None          # pillar | concept | None
    target: str | None
    grounding: str | None = None      # plain KB explanation (EXP cites; rag_explainer on top)
    is_about_swap: bool = False       # W9 — preserves log_swap_questioned firing


@dataclass
class AdvanceOutcome:
    passive: bool = False             # passive_advance (no add/remove/question/suggest)
    elicited: bool = False            # #5: advanced because the user asked for a suggestion


@dataclass
class FallbackOutcome:
    reason: str = "unclear"           # unclear | start_over | nothing_resolved


Outcome = (
    AddOutcome | RemovalOutcome | SuggestOutcome
    | QuestionOutcome | AdvanceOutcome | FallbackOutcome
)


# ════════════════════════════════════════════════════════════════════════════
#  PendingAction — ONE shared confirmation machine (§3.5), replaces per-agent
#  pending_excl / pending_sub_excl / pending_add / pending_clarify.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PendingAction:
    type: str                         # remove_pillar | remove_sub_bullet
    target: str                       # pillar name OR sub-bullet text
    level: str                        # pillar | concept
    pillar: str | None = None         # parent pillar for a sub_bullet removal
    justification: str | None = None
    requires_justification: bool = False   # HITL gate (D-H2 mechanism; scope wired later)
    is_swap: bool = False             # swap target -> detection on confirm, never a delete
    consequence_facts: list = field(default_factory=list)


# ── shared effort-only justification gate (D-H2; was HITL._is_substantive_justification) ──
def is_meaningful_justification(text: str) -> bool:
    """Effort-only gate: did the user give a real reason, not 'asdf'/''/'no'? No grading of
    QUALITY — just substance (§3.5; B8 passes, B9 fails). Byte-equivalent to HITL's old
    `_is_substantive_justification` so the existing HITL behavior is preserved, now shared."""
    t = (text or "").strip()
    words = [w for w in t.split() if any(ch.isalpha() for ch in w)]
    return len(words) >= 3 and len(t) >= 12


# ════════════════════════════════════════════════════════════════════════════
#  HandlerSession — the neutral state surface (D-H3). Any object exposing these
#  satisfies it; each arm adapts its own state in the wiring steps, BaseAgent
#  owns it in Step 6. Handlers touch NOTHING outside this contract.
# ════════════════════════════════════════════════════════════════════════════
class HandlerSession(Protocol):
    # --- parked flow state (shared; replaces per-arm pending_*) ---
    pending: PendingAction | None
    pending_suggestion: dict | None            # {level, item, origin} — D7 / B6 memory
    # pending_placement: {kind, item, target?, level?} — the #4 ask-flow / #1#3 navigate
    #   offer. Read via getattr(...) so it needs no __init__ slot in BaseAgent; set/cleared
    #   by add_handler/resolve_placement. Walkthrough arms (EXP) only; BB renders + clears.
    last_discussed: m.KBMatch | None           # Fork-A deictic target (current focus)
    shown_bullets: list[str]                   # currently-shown bullets (positional removal)
    # --- shared contribution state (both arms have these) ---
    excluded_concepts: list[str]
    excluded_sub_bullets: dict[str, list[str]]

    # --- presentation queries (per-arm: BB shown+user_added-excluded+swap; HITL surfaced) ---
    def presented_pillars(self) -> list[str]: ...
    def presented_sub_bullets(self) -> dict[str, list[str]]: ...
    def surfaced_pillar_names(self) -> set[str]: ...     # everything already shown/added/excluded
    def current_pillar(self) -> str | None: ...          # walkthrough focus; None for BlackBox

    # --- neutral mutators for arm-specific storage (adds) ---
    def surface_pillar(self, name: str) -> None: ...     # reveal a withheld/unreached pillar
    def add_sub_point(self, pillar: str, text: str) -> None: ...

    # --- swap channel (PRESERVED per-arm, §0 #4) ---
    def swap_name(self) -> str | None: ...               # wrong_concept iff injected & not detected
    def is_swap_target(self, km: m.KBMatch, user_text: str) -> bool: ...
    def mark_swap_detected(self) -> None: ...

    # --- HITL justification policy (D-H2; scope decided at wiring/Step 6) ---
    def requires_justification(self, km: m.KBMatch) -> bool: ...


# ════════════════════════════════════════════════════════════════════════════
#  Shared confirmation classifier (was BlackBox._classify_confirmation). Used by
#  the free-text arms to resolve a parked removal; HITL passes a button decision
#  in directly. A deterministic keyword fast-path keeps obvious yes/no off the
#  LLM (I-7); ambiguous replies fall through to the shared classifier.
# ════════════════════════════════════════════════════════════════════════════
_CONFIRM_FIRST = {"yes", "yep", "yeah", "yup", "sure", "confirm", "confirmed", "ok",
                  "okay", "absolutely", "definitely", "go", "proceed", "remove", "delete"}
_DECLINE_FIRST = {"no", "nope", "nah", "keep", "cancel", "stop", "leave", "dont", "never"}
_DECLINE_PHRASES = ("never mind", "actually keep", "do not", "keep it", "dont remove")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).strip("?.!, ")


def _classify_confirmation(text: str) -> str:
    """Return 'confirm' | 'decline' | 'other' for a reply to a removal challenge. A
    punctuation-insensitive keyword check resolves the common yes/no cases deterministically
    (I-7); decline is tested first so 'no, keep it' is not read as a confirm; only ambiguous
    replies reach the shared classifier."""
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
    """Is the reply a bare affirmation ('yes'/'ok'/'sure')? — used to accept a pending
    suggestion (D7) or an offered add (B6) without a fresh concept being named."""
    return _classify_confirmation(text) == "confirm"


def _accepts_offer(text: str) -> bool:
    """Does this reply ACCEPT a parked offer (agent suggestion D7 / removal add-offer B6)?
    A genuine affirmation, or a bare 'add it'/'include it' with no NEW concept named — but
    NOT a skip ('move on', 'next') and NOT 'add <something else>' (which is a fresh add that
    must fall through to add_handler)."""
    if _affirms(text):
        return True
    w = re.findall(r"[a-z']+", _norm(text))
    if w and w[0] in ("add", "include", "keep"):
        rest = w[1:]
        return (not rest) or rest[0] in ("it", "that", "this", "them", "one", "please")
    return False


def _add_accepts_suggestion(ps: dict, intent: str, detail, parent, user_text: str) -> bool:
    """Fork B (F-A1): an `add` turn that NAMES the offered item (or is a contentless 'add it')
    right after a parked suggestion IS an acceptance of it — even when phrasing like
    "let's include it" defeats `_accepts_offer`'s first-word parse (the classifier has already
    resolved the deictic to `add(detail=<item>)`). Matched ONLY against the parked item, so an
    `add <something-else>` still falls through to add_handler as a genuine fresh add."""
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


# ── #4 placement-choice detection (D2: deterministic phrasing first, locate last) ──────
_OWN_AREA = ("own area", "separate area", "new area", "its own", "on its own", "own pillar",
             "separate pillar", "new pillar", "new section", "standalone", "by itself",
             "separate", "new topic", "different area", "own thing")
_HERE = ("keep it here", "under this", "this area", "this pillar", "right here", "under it",
         "where we are", "current area", "under current", "keep here", "here", "current")


def _placement_choice(text: str):
    """Resolve a reply to the #4 ask ("own area / under current / under a named area").
    Deterministic phrasing first (D2); a named destination resolves via locate(); anything
    unrecognised returns ('unclear', None) so dispatch clears the offer and routes normally.
    Returns ('own',None) | ('current',None) | ('named',<pillar>) | ('unclear',None)."""
    n = _norm(text)
    if any(p in n for p in _OWN_AREA):
        return ("own", None)
    # explicit "under/in/to <X>" where X resolves to a real area -> named (checked before the
    # generic 'here' words so "under feasibility" is a named area, not the current one).
    mobj = re.search(r"\b(?:under|in|to|put it in|add it to)\s+(.+)$", n)
    if mobj:
        km = m.locate(mobj.group(1).strip())
        if km.level in ("pillar", "concept") and km.pillar:
            return ("named", km.pillar)
    if any(p in n for p in _HERE):
        return ("current", None)
    # NOTE (bug-A fix): no bare-name fallback. A reply must use explicit placement language
    # (own-area phrasing, "under <X>", or a current-pillar deictic) to count as an answer. A
    # fresh command like "consider data quality" must NOT be read as "put the parked item
    # under data-quality's pillar" — it returns 'unclear' so dispatch clears the offer and
    # routes the new turn normally.
    return ("unclear", None)


def _wants_navigate(text: str) -> bool:
    """A reply that ACCEPTS a navigate offer ("yes" / "go there" / "let's go")."""
    if _affirms(text):
        return True
    n = _norm(text)
    return n.startswith(("go ", "go", "navigate", "take me", "lets go", "let us go", "jump"))


def _parent_is_explicit(parent: str | None, user_text: str) -> bool:
    """Point-1 guard: did the USER actually place the add under `parent`, or did the
    classifier infer the current pillar? True only when the user named the destination
    (a content word of the pillar appears in the raw text) or used a current-pillar deictic
    ("here" / "this pillar"). Prevents "what about data quality" (parent inferred as the
    current pillar) from being dumped there instead of routed through recognition."""
    if not parent:
        return False
    t = _norm(user_text)
    if any(d in t for d in ("here", "this pillar", "this area", "this section", "this one")):
        return True
    ptoks = [w for w in re.findall(r"[a-z]+", _norm(parent)) if len(w) >= 4]
    return any(w in t for w in ptoks)


def resolve_placement(session: HandlerSession, user_text: str) -> Outcome | None:
    """Resolve a parked pending_placement. Returns an Outcome the persona renders, or None
    when the reply does not answer the offer (dispatch then clears it and routes the turn
    normally — "anything else -> dispatch normally"). The pending_placement is cleared here
    in every branch; only the resolved PLACEMENT counts (D-count), never the ask itself."""
    pp = getattr(session, "pending_placement", None) or {}
    session.pending_placement = None
    kind = pp.get("kind")
    item = pp.get("item", "")

    if kind == "navigate":
        if _wants_navigate(user_text):
            # Withheld-reveal accept: the user confirmed discussing a hidden pillar they
            # surfaced via one of its concepts (e.g. ROI -> Financial Impact). Reveal it now
            # — surface_pillar puts it in the walk so the navigate actually lands, and this
            # is the user's agency act, counted once (mirrors pillar-level 3a). Shared across
            # all three arms because surface_pillar is each arm's own walk/list append.
            if pp.get("reveal_on_accept") and pp.get("target"):
                session.surface_pillar(pp.get("target"))
                return AddOutcome(action="navigated", pillar=pp.get("target"), level="pillar",
                                  counted=True, source="user_spontaneous",
                                  matched_pillar_id=pp.get("matched_pillar_id"),
                                  navigate_bullet=pp.get("navigate_bullet"),
                                  navigate_concept_id=pp.get("concept_id"))  # §2a
            return AddOutcome(action="navigated", pillar=pp.get("target"), level="pillar",
                              counted=False, source="user_spontaneous",
                              navigate_bullet=pp.get("navigate_bullet"),
                              navigate_concept_id=pp.get("concept_id"))  # §2a
        # declined: for a CONCEPT offer, an explicit "keep it here" records it under the
        # current area. If the surfacing was already credited at the offer (D-a, unreached
        # concept), the placement does NOT count again — counted_here guards the double.
        if (pp.get("level") == "concept" and session.current_pillar()
                and any(k in _norm(user_text) for k in ("keep", "here", "stay", "leave it"))):
            dest = session.current_pillar()
            slot = m.place_sub_point(item, dest)
            session.add_sub_point(slot.pillar, item)
            return AddOutcome(action="added_new", pillar=slot.pillar, level="sub_bullet",
                              counted=not pp.get("counted_here", False), text=item,
                              source="user_spontaneous")
        # §2a (fix-2): the reply did not resolve as an explicit navigate/keep here, but the
        # intent router may still route it to this same target as `revisit`/`advance`
        # ("yeah go there" misses _wants_navigate yet classifies as revisit). The parked
        # offer (incl. its carried canonical bullet) is about to be cleared; stash it so the
        # follow-on navigate to the SAME pillar carries navigate_bullet/_concept_id too —
        # one shared carry across both the navigate-accept and the revisit/advance paths.
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
            name = m.normalize_name(item)            # D6 style: "culture" -> "Culture"
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
        return None                                  # unclear -> clear + dispatch normally
    return None


# ── small KB helper ──────────────────────────────────────────────────────────
def _pillar_id(name: str | None) -> str | None:
    if not name:
        return None
    p = next((p for p in kb.get_all_pillars() if p["name"].lower() == name.lower()), None)
    return p["id"] if p else None


# ════════════════════════════════════════════════════════════════════════════
#  HANDLERS — each does its intent's invariant work and returns an Outcome.
# ════════════════════════════════════════════════════════════════════════════
def add_handler(intent: str, km: m.KBMatch, source: str, session: HandlerSession,
                *, text: str, parent: str | None = None) -> AddOutcome:
    """add / revisit. Consumes locate()'s KBMatch (the F-M1/F-M2 convergence point).
    With an explicit parent ("add X under Y") X is placed under Y (B1.1). With NO parent:
        level==concept  -> known criterion living under Z -> navigate_offer (#1/#3): offer
                           to go to Z or keep it here; park pending_placement.
        level==pillar   -> withheld  -> REVEAL (add_pillar; the user's agency signal, #5)
                           shown      -> navigate_offer: presented = go back & edit, unreached
                                          = go there now; park pending_placement (no count).
        level==none     -> NOVEL -> ask_placement (#4): never auto-placed; ask own-area /
                           under-current / under-named and resolve the choice next turn.
    `revisit` is pure navigation to an existing pillar (D5: keyed off the router's revisit
    intent, not a literal word; the router never lets revisit carry new content — that is an
    `add` with `parent`), so it returns action='navigated'."""
    if intent == "revisit":
        # Navigate to the named (passed) area + re-render; no contribution counted.
        # §2a (fix-2): if a concept-navigate offer to THIS pillar was parked and not
        # resolved on the navigate-accept path (e.g. "yeah go there" -> revisit), carry its
        # canonical bullet through so the render seam surfaces the recognised concept on
        # arrival. Same fields, same carry as the navigate-accept branch — one shared rule.
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

    # 1 — EXPLICIT PLACEMENT wins (agency, B1.1). "add X under Y" puts X under Y as the
    #     user's own point, even when X also matches a known concept somewhere else. It is
    #     a genuine duplicate ONLY when X already lives under the very pillar the user named.
    if parent:
        if (km.level == "concept" and km.pillar
                and km.pillar.lower() == parent.lower()):
            return AddOutcome(action="duplicate", pillar=parent, level="concept",
                              counted=False, text=text, matched_text=km.matched_text,
                              source=source)
        if parent.lower() not in presented:
            parent = m.normalize_name(parent)   # D6 style: "culture" -> "Culture"
            session.surface_pillar(parent)   # register a NOVEL destination so it renders (F-A fix)
        slot = m.place_sub_point(text, parent)
        session.add_sub_point(slot.pillar, text)
        # Tell the user if X is ALSO covered elsewhere — but only name a pillar that is
        # already on screen. Never leak a withheld area here; that is the reveal-on-match step.
        also = (km.pillar if (km.level == "concept" and km.pillar
                              and km.pillar.lower() != slot.pillar.lower()
                              and km.pillar.lower() in presented) else None)
        return AddOutcome(action="added_new", pillar=slot.pillar, level="sub_bullet",
                          counted=True, text=text, also_covered=also,
                          explanation=(m.pillar_gist(also) if also else None), source=source)

    # 2 — named existing CRITERION (no placement). The user raised a known concept that
    #     lives under pillar Z -> offer to navigate there (#1/#3), parking the offer. When Z
    #     is NOT yet presented, surfacing it is an agency act -> count add_sub_bullet ONCE
    #     (D-a). The offer is grounded with the concept's KB explanation (D-c).
    if km.level == "concept":
        presented = {n.lower() for n in session.presented_pillars()}
        unreached = bool(km.pillar) and km.pillar.lower() not in presented
        key = (km.concept_id or km.matched_text or text or "").lower()
        seen = getattr(session, "_agency_concepts", None)
        if seen is None:
            seen = set()
            session._agency_concepts = seen
        # A withheld-pillar concept defers its agency credit to the reveal on accept (below),
        # so it must NOT also count at the offer. A SHOWN unreached concept keeps D-a (count
        # the surfacing once, here).
        counted = bool(unreached and key and key not in seen
                       and not km.pillar_is_withheld)   # D-a: once, unreached, shown only
        if counted:
            seen.add(key)
        expl = ((g.ground_concept(km.concept_id) if km.concept_id else None)
                or m.pillar_gist(km.pillar) or None)            # D-c grounding
        # §2a: capture the concept's canonical KB bullet so a later "go there" can RENDER it.
        # refs=False here is the arm-neutral store; EXP re-derives with refs on its own render
        # path. None when the concept has no bullet form — the navigate then just re-renders
        # the pillar as before (no regression).
        nav_bullet = m.concept_bullet(km.concept_id, refs=False) if km.concept_id else None
        # A user-named concept under a WITHHELD, not-yet-presented pillar (e.g. "ROI" ->
        # Payback period -> Financial Impact) is the USER's reveal act (#5 sibling), not the
        # agent volunteering a hidden pillar. The OFFER only names the pillar and asks ("want
        # to discuss there?"); the actual reveal+surface is deferred to the ACCEPT turn in
        # resolve_placement, mirroring the pillar-level 3a contract. The agent-suggest path
        # (suggest_handler) still never names a withheld pillar. counted_here stays False so
        # the agency credit lands on the reveal at accept, not on the mere offer.
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

    # 3 — resolved to a whole AREA.
    if km.level == "pillar":
        # 3a — WITHHELD pillar named by the user -> reveal-on-match (the agency signal,
        #      #5 sibling). The agent never offers these; surfacing one is the user's act.
        if km.pillar_is_withheld and not (km.pillar and km.pillar.lower() in presented):
            session.surface_pillar(km.pillar)
            return AddOutcome(action="revealed", pillar=km.pillar, level="pillar",
                              counted=True, matched_pillar_id=_pillar_id(km.pillar),
                              explanation=g.ground_pillar(km.pillar) or None, source=source)
        # 3b — a SHOWN pillar (presented or not-yet-reached) -> offer to navigate to it
        #      (presented: "go back and edit"; unreached: "go there now"). Naming a planned
        #      shown pillar is navigation, never a counted contribution. Park the offer.
        session.pending_placement = {"kind": "navigate", "target": km.pillar,
                                     "level": "pillar", "item": text}
        return AddOutcome(action="navigate_offer", pillar=km.pillar, level="pillar",
                          counted=False, explanation=m.pillar_gist(km.pillar) or None,
                          source=source)

    # 4 — NOVEL (level==none, no explicit parent). D7: never auto-place. ASK the user where
    #     it goes (own area / under the current area / under a named area); park the question
    #     and resolve their choice next turn in resolve_placement (count follows the choice).
    session.pending_placement = {"kind": "novel", "item": text}
    return AddOutcome(action="ask_placement", pillar=None, level="none",
                      counted=False, text=text, source=source)



def _resolve_presented_bullet(needle: str, shown: list[str]) -> str | None:
    """Map a removal target to the EXACT presented bullet text, or None if it is not in the
    block. The target may arrive as a full bullet, or (via locate()->_match_concept) as a
    concept NAME which is the lead phrase before ':' in its sub_bullet
    ("Single-developer dependency" -> "Single-developer dependency: is the tool ...? [c]").
    Resolving to the full presented text serves BOTH guards: a None return is the F-R2b
    existence guard, and a hit gives the FULL bullet so the parked/stored exclusion key
    matches what the render + summary filter on (F-R5). Comparison is ref-insensitive (_norm)."""
    if not needle:
        return None
    n = m._norm(needle)
    for b in shown:                              # 1 — exact (ref-insensitive) full-bullet match
        if m._norm(b) == n:
            return b
    for b in shown:                              # 2 — concept name == bullet lead phrase
        if m._norm(b.split(":", 1)[0]) == n:
            return b
    if len(n) >= 6:                              # 3 — guarded containment (LLM phrasing drift)
        for b in shown:
            if n in m._norm(b):
                return b
    return None


def removal_handler(km: m.KBMatch, session: HandlerSession, *, user_text: str) -> RemovalOutcome:
    """First removal turn: validate the target is PRESENT (the existence guard, F-R2), then
    PARK a PendingAction and return stage='challenged'. NOTHING is deleted here (F-R1: delete
    fires only at confirm). The swap is handled as DETECTION, not a delete (§0)."""
    # 0 — deictic with no recorded focus (BlackBox full-render) -> ask, don't guess (Fork A).
    if km.needs_disambiguation:
        return RemovalOutcome(stage="needs_disambiguation", target=None, level="none")

    # 1 — SWAP target (preserved per-arm channel). Challenge -> confirm -> detection.
    swap = session.swap_name()
    if swap and session.is_swap_target(km, user_text):
        req = session.requires_justification(km)
        session.pending = PendingAction(type="remove_pillar", target=swap, level="pillar",
                                        is_swap=True, requires_justification=req)
        return RemovalOutcome(stage="challenged", target=swap, level="pillar",
                              is_swap=True, needs_justification=req)

    # 2 — nothing resolved -> nothing to remove (B7).
    if km.level == "none":
        return RemovalOutcome(stage="nothing_to_remove",
                              target=km.matched_text or user_text, level="none")

    # 3 — whole PILLAR.
    if km.level == "pillar":
        presented = {n.lower() for n in session.presented_pillars()}
        if not km.pillar or km.pillar.lower() not in presented:
            # not present -> nothing to remove; if it's a real (withheld) KB area, offer add (B5).
            offer = km.pillar if (km.pillar and km.pillar_is_withheld) else None
            if offer:
                # Park the offer so a following 'yes' is the USER's own add (B6, user_spontaneous),
                # NOT an agent suggestion (origin distinguishes it from suggest_handler's D7).
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

    # 4 — a specific point (concept / sub_bullet level). The target may arrive as a concept
    #     NAME / lead phrase ("Single-developer dependency"), a full bullet, or a deictic focus.
    #     Resolve it to the exact PRESENTED bullet so (a) the stored exclusion key matches what
    #     render+summary filter on (F-R5) and (b) the existence guard is real (F-R2b). A deictic
    #     "remove this" target IS the recorded last-discussed focus -> present by construction,
    #     so it is challenged even if it is not a sub_bullet in the current block.
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
        parent, target = resolved_parent, resolved_bullet   # F-R5: full presented bullet text
    elif by_construction and parent is not None:
        pass                                                # deictic focus: present by construction
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
    """Second removal turn: resolve the parked PendingAction. `decision` is supplied for a
    deterministic button (HITL confirm/cancel); free-text arms leave it None and it is
    classified. The HITL justification gate (requires_justification) blocks `confirmed`
    until a meaningful reason arrives (B8 passes -> confirm; B9 fails -> needs_justification,
    stays challenged, NO delete)."""
    pa = session.pending
    if pa is None:                               # defensive: nothing parked
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
                # gate fails -> re-ask, stay challenged, nothing deleted (B9)
                return RemovalOutcome(stage="needs_justification", target=pa.target,
                                      level=pa.level, needs_justification=True,
                                      is_swap=pa.is_swap)
            pa.justification = reason            # passes (B8) -> proceed to confirm
        return _confirm_removal(session, pa)

    # 'other' (a question / unrelated) -> stay challenged, persona re-asks.
    return RemovalOutcome(stage="challenged", target=pa.target, level=pa.level,
                          needs_justification=pa.requires_justification, is_swap=pa.is_swap)


def _confirm_removal(session: HandlerSession, pa: PendingAction) -> RemovalOutcome:
    """Commit the parked removal. The ONLY place an exclusion is applied and (in the agent,
    driven by this stage) a delete is logged — F-R1. Swap = detection, never a delete (§0)."""
    session.pending = None
    if pa.is_swap:
        session.mark_swap_detected()             # swap DETECTED — no delete event
        return RemovalOutcome(stage="confirmed", target=pa.target, level=pa.level,
                              is_swap=True, justification=pa.justification,
                              post_delete_branch=True)
    if pa.type == "remove_sub_bullet" and pa.pillar:
        session.excluded_sub_bullets.setdefault(pa.pillar, [])
        if not m.is_excluded_bullet(session.excluded_sub_bullets, pa.pillar, pa.target):
            session.excluded_sub_bullets[pa.pillar].append(pa.target)
    else:                                        # remove_pillar
        if pa.target.lower() not in [e.lower() for e in session.excluded_concepts]:
            session.excluded_concepts.append(pa.target)
    return RemovalOutcome(stage="confirmed", target=pa.target, level=pa.level,
                          pillar=pa.pillar,
                          justification=pa.justification, post_delete_branch=True,
                          consequence_facts=pa.consequence_facts)


def question_handler(km: m.KBMatch, session: HandlerSession, *, user_text: str) -> QuestionOutcome:
    """A question about shown content. Grounds from the KB (plain; EXP/rag_explainer layer
    citations on top). W9: a question that lands on the swap concept still flags
    is_about_swap so the agent can preserve log_swap_questioned."""
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
    """ask_agent_to_suggest under reveal-on-match (#5): the agent NEVER names a withheld
    pillar. Mid-walk -> advance to the next SHOWN pillar, logged as agent-suggest (the user
    asked for guidance). Exhausted / no walkthrough cursor (BlackBox) -> a SuggestOutcome
    with no item, which the persona renders as a throw-back inviting the user (no withheld)."""
    if session.current_pillar() is not None:
        return advance_handler(session, passive=False, elicited=True)
    return SuggestOutcome(level="pillar", suggested_item=None)


def advance_handler(session: HandlerSession, *, passive: bool = False,
                    elicited: bool = False) -> AdvanceOutcome:
    """Move on. `passive` marks a passive_advance; `elicited` marks an advance triggered by
    ask_agent_to_suggest (#5) so it logs as agent-suggest instead of passive (§3.6)."""
    return AdvanceOutcome(passive=passive, elicited=elicited)


def fallback_handler(reason: str = "unclear") -> FallbackOutcome:
    """none / unclear / 'start over' (redo removed, §0). Persona renders its fallback."""
    return FallbackOutcome(reason=reason)


def _consequence_facts(km: m.KBMatch) -> list:
    """Material for Explainable's counterfactual on a removal challenge — the KB key-questions
    that would be lost. Plain facts (no citation); EXP renders them, other arms ignore."""
    if not km.pillar:
        return []
    p = next((p for p in kb.get_all_pillars() if p["name"].lower() == km.pillar.lower()), None)
    if not p:
        return []
    return [g._strip_source_refs(q) for q in p.get("key_questions", [])][:3]


# ════════════════════════════════════════════════════════════════════════════
#  DISPATCH — the seam between router and handler (D-Q2 carry-in). Calls
#  locate()/resolve_removal_target HERE so the handler receives a resolved
#  KBMatch (I-3). A parked PendingAction is resolved FIRST (the two-turn loop).
# ════════════════════════════════════════════════════════════════════════════
def dispatch(intent_result, session: HandlerSession, *, user_text: str,
             source: str = "user_spontaneous") -> Outcome:
    """Route ONE classified free-text turn to its handler, resolving the KB target in
    between. `intent_result` is intents.IntentResult (intent + raw detail/parent). Buttons
    bypass this (D-Q1): they map straight to an intent enum, and confirm/cancel call
    `resolve_pending` directly."""
    # 0 — a parked removal owns this turn: resolve the confirmation machine first.
    if session.pending is not None:
        return resolve_pending(session, user_text)

    # §2a (fix-2): the navigate carry is strictly one-turn — set in resolve_placement (0a,
    # below) and consumed in add_handler's revisit branch within THIS same call. Clear any
    # residual from a prior turn here so a set-but-not-consumed carry can never leak forward.
    session._carried_navigate = None

    intent = intent_result.intent
    detail = intent_result.detail
    parent = intent_result.parent

    # 0a — a parked placement offer (#4 ask-flow / #1#3 navigate) owns this turn IF the
    #      reply answers it; a non-answer clears it (inside resolve_placement) and the turn
    #      routes normally below.
    if getattr(session, "pending_placement", None) is not None:
        placed = resolve_placement(session, user_text)
        if placed is not None:
            return placed

    # 0b — a pending agent suggestion (D7) or removal-offer (B6) owns an affirmation.
    if session.pending_suggestion is not None:
        ps = session.pending_suggestion
        accepting = (_accepts_offer(user_text)
                     or _add_accepts_suggestion(ps, intent, detail, parent, user_text))  # F-A1 (Fork B)
        session.pending_suggestion = None
        if accepting:
            if ps.get("origin") == "agent_suggest":          # D7 — agent-originated reveal
                session.surface_pillar(ps["item"])
                return SuggestOutcome(level=ps["level"], suggested_item=ps["item"],
                                      grounding=g.ground_pillar(ps["item"]) or None,
                                      accepted=True, revealed=True)
            # B6 — the idea was the USER's (they tried to remove it) -> spontaneous add
            km = m.locate(ps["item"])
            return add_handler("add", km, "user_spontaneous", session, text=ps["item"])
        # not accepting -> fall through and dispatch this turn normally.

    if intent in ("add", "revisit"):
        # B3: for an ADD, resolve WHAT is being added (detail), not the destination
        # (parent). "add X under Y" must look up X, not Y — else it false-matches the
        # Y pillar and blocks the add. revisit carries no content -> resolve its target.
        # Point-1 guard: the classifier sometimes INFERS parent=current_pillar even when the
        # user named no destination ("what about data quality" -> parent="Strategic Fit").
        # Honor parent ONLY when the user actually placed it (named it, or a current-pillar
        # deictic); otherwise drop it so the add goes through concept recognition / the ask-
        # flow instead of being dumped under the current pillar.
        if intent == "add" and parent and not _parent_is_explicit(parent, user_text):
            parent = None
        probe = (detail or user_text) if intent == "add" else (parent or detail or user_text)
        km = m.locate(probe)
        return add_handler(intent, km, source, session, text=detail or user_text, parent=parent)

    if intent == "remove":
        # B4: thread the EXTRACTED target through to swap-detection. "I think we
        # should not consider this" -> detail names the swap though the raw text does
        # not; the swap is excluded from locate() so km never carries it.
        target_text = detail or user_text
        km = m.resolve_removal_target(target_text,
                                      last_discussed=session.last_discussed,
                                      shown_bullets=session.shown_bullets)
        return removal_handler(km, session, user_text=target_text)

    if intent == "question":
        km = m.locate(detail or user_text)
        # swap recognition keys on the LLM-EXTRACTED target too, not just the raw
        # sentence: "why walked per day relevant?" -> detail names the swap though the
        # raw text lacks a "step" token (mirror of the B4 removal fix).
        return question_handler(km, session, user_text=detail or user_text)

    if intent == "ask_agent_to_suggest":
        return suggest_handler(session)

    if intent == "advance":
        return advance_handler(session, passive=True)

    if intent == "doubt":
        # vague doubt, no clear command -> non-destructive; persona invites a clear remove.
        return fallback_handler(reason="unclear")

    return fallback_handler(reason="start_over" if intent == "none" else "unclear")
