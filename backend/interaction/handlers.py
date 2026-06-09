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
from backend import knowledge_base as kb
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
    """add / revisit. Consumes locate()'s KBMatch (the F-M1/F-M2 convergence point):
        level==concept  -> the user named an existing criterion -> DUPLICATE (no count)
        level==pillar   -> a whole area:
                             already presented      -> DUPLICATE
                             not yet presented       -> REVEAL (add_pillar + matched id)
        level==none     -> NOVEL: under a destination pillar -> sub_bullet; else new area
    `revisit` is pure navigation to a PASSED pillar (the router never lets revisit carry
    new content — that is an `add` with `parent`), so it returns action='navigated'."""
    if intent == "revisit":
        # Navigate to the named (passed) area + re-render; no contribution counted.
        return AddOutcome(action="navigated", pillar=km.pillar, level="pillar",
                          counted=False, source=source)

    # 1 — named existing criterion -> duplicate (F-M1). No count, no artifact.
    if km.level == "concept":
        return AddOutcome(action="duplicate", pillar=km.pillar, level="concept",
                          counted=False, source=source)

    # 2 — resolved to a whole AREA.
    if km.level == "pillar":
        presented = {n.lower() for n in session.presented_pillars()}
        if km.pillar and km.pillar.lower() in presented:
            return AddOutcome(action="duplicate", pillar=km.pillar, level="pillar",
                              counted=False, source=source)
        # not yet presented -> reveal it (withheld reveal OR shown-but-unreached) [F-M2]
        session.surface_pillar(km.pillar)
        return AddOutcome(action="revealed", pillar=km.pillar, level="pillar",
                          counted=True, matched_pillar_id=_pillar_id(km.pillar),
                          explanation=g.ground_pillar(km.pillar) or None, source=source)

    # 3 — NOVEL (level==none). Under a destination pillar -> sub-point; else a new area.
    dest = parent or session.current_pillar()
    if dest:
        slot = m.place_sub_point(text, dest)
        session.add_sub_point(slot.pillar, text)
        return AddOutcome(action="added_new", pillar=slot.pillar, level="sub_bullet",
                          counted=True, text=text, source=source)
    session.surface_pillar(text)                 # BlackBox: no walkthrough focus -> new area
    return AddOutcome(action="added_new", pillar=text, level="pillar",
                      counted=True, matched_pillar_id=None, text=text, source=source)


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


def suggest_handler(session: HandlerSession) -> SuggestOutcome | None:
    """ask_agent_to_suggest: deterministically pick the first WITHHELD pillar not yet
    surfaced, grounded from KB text (never free-form -> cannot confabulate; D1/D6). Parks a
    pending_suggestion so a following 'yes' is recognised as accepting THE AGENT's idea (D7),
    not a user add. Suggesting is NOT adding — no artifact, no add_pillar."""
    surfaced = {n.lower() for n in session.surfaced_pillar_names()}
    for p in kb.get_all_pillars():
        if not p.get("shown", True) and p["name"].lower() not in surfaced:
            session.pending_suggestion = {"level": "pillar", "item": p["name"],
                                          "origin": "agent_suggest"}
            return SuggestOutcome(level="pillar", suggested_item=p["name"],
                                  grounding=g.ground_pillar(p["name"]) or None)
    return None                                  # nothing left to suggest


def advance_handler(session: HandlerSession, *, passive: bool = False) -> AdvanceOutcome:
    """Move on. `passive` marks a passive_advance (advanced with no add/remove/question/
    suggest this turn) so the agent can log it distinctly (§3.6)."""
    return AdvanceOutcome(passive=passive)


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

    intent = intent_result.intent
    detail = intent_result.detail
    parent = intent_result.parent

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
        km = m.locate(parent or detail or user_text)
        return add_handler(intent, km, source, session, text=detail or user_text, parent=parent)

    if intent == "remove":
        km = m.resolve_removal_target(detail or user_text,
                                      last_discussed=session.last_discussed,
                                      shown_bullets=session.shown_bullets)
        return removal_handler(km, session, user_text=user_text)

    if intent == "question":
        km = m.locate(detail or user_text)
        return question_handler(km, session, user_text=user_text)

    if intent == "ask_agent_to_suggest":
        return suggest_handler(session)

    if intent == "advance":
        return advance_handler(session, passive=True)

    if intent == "doubt":
        # vague doubt, no clear command -> non-destructive; persona invites a clear remove.
        return fallback_handler(reason="unclear")

    return fallback_handler(reason="start_over" if intent == "none" else "unclear")
