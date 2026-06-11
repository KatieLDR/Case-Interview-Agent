import json
import logging
import random
import re
from google.genai import types
from backend.agents.base import BaseAgent           # Step 6b: sibling of BaseAgent (F-ARCH2)
from backend.llm import (
    CLASSIFIER_MODEL, MAIN_MODEL, client, classify_json, ANSWER_THRESHOLD,
)
from backend.knowledge.cases import get_case, get_clarification_facts
from backend.tools.concept_swap import ConceptSwap
from backend.logger import (
    create_session, log_user_message, log_agent_response,
    log_interruption, update_answer, stamp_started_at,
)
from backend.logging import events as ev
from backend.logging.sink import firestore_sink as _sink
from backend.knowledge import knowledge_base as kb           # JSON KB — static presentation + matching
from backend.domain import matching, grounding      # Step 2: shared KB matchers + grounding
from backend.interaction import intents              # Step 3: unified intent taxonomy (I-2)
from backend.interaction import handlers as h        # Step 4d: shared PendingAction + resolve_pending (I-1)

# D-Q1: the shared button<->intent map is canonical. DERIVE the set of free-text
# intents that HITL also exposes as a button from intent_for_button, so HITL and the
# shared map can never drift. A free-text action in one of these is NUDGED to its
# button instead of executed; question / ask_agent_to_suggest are answered directly.
_BUTTON_ACTION_INTENTS = frozenset(
    intents.intent_for_button(b) for b in intents.BUTTON_INTENT
) - {None}
_HITL_BUTTON_LABELS = {
    "advance": "**✅ Include** or **❌ Skip** (to move past this pillar)",
    "remove":  "**❌ Skip** (whole pillar) or **➖ Remove a point**",
    "add":     "**➕ Add point to consider in this pillar**",
    "revisit": "**↩️ Add point to a past pillar**",
}

# Strip inline source refs like " [a]" / "[b]" — HITL suppresses sources, so these
# markers must never reach the user (no Sources line resolves them). Change log: 2026-05-29



# Deictic references to "the concept we're on" — never a pillar name. The intent
# classifier emits these RAW in the `parent` slot (D-Q2; intents.py preserves
# deictics for downstream resolution, see its line-316 note). The elicited add
# path resolves them to the CURRENT concept at dispatch — see _add_sub_point.
# Change log: 2026-06-09 (Step-5 gap #1: elicited add_sub_bullet attribution).
_DEICTIC_PARENTS = frozenset({
    "this", "it", "here", "this one", "this concept", "this area",
    "this pillar", "this section", "current", "current concept",
    "the current concept", "the current pillar",
})


def _is_deictic_parent(name) -> bool:
    """True if `name` is a deictic reference to the current concept rather than a
    named pillar. Whole-string match on the normalized text (mirrors intents.py's
    filler/steering guards) — never a substring, so a real pillar that merely
    contains such a word is unaffected."""
    if not name:
        return False
    return re.sub(r"\s+", " ", str(name).strip().lower()).strip("?.!, ") in _DEICTIC_PARENTS

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "AI Implementation"

# ── Proactive prompts — rotating fixed list ────────────────────────────────
PROACTIVE_PROMPTS = [
    # User-first
    "What's your instinct for the next area to explore, or would you like me to suggest one?",
    "Any thoughts on what to tackle next, or would you prefer my guidance?",
    "What angle would you take next, or shall I continue building this out?",
    "What's your next move on this, or would you like me to step in?",
    # Guidance-first
    "Would you like me to guide the next step, or is there an area you'd want to drive?",
    "Shall I take the lead here, or do you have a direction in mind?",
    "Would you prefer my guidance on this, or do you have thoughts on where to go next?",
    "Before I continue, is there an area you'd want to prioritise, or shall I proceed?",
]

# ── Justification acknowledgements — hardcoded, no LLM ────────────────────
JUSTIFICATION_ACKS = [
    "Noted — let's continue.",
    "Thanks for sharing that.",
    "Got it.",
    "Understood.",
]

# ── New-area / sub-point matching — moved to domain/ (Step 2) ───────────────
# The local ADD_PILLAR_MATCH_PROMPT / ADD_MATCH_PROMPT / ADD_MATCH_THRESHOLD copies
# were retired: _match_pillar / _match_key_question below now delegate to the shared
# backend.domain.matching functions, so all three arms resolve identically by
# construction (I-1/I-3). NOTE this is a deliberate convergence for HITL — its old
# pillar matcher used a bare-name, un-guarded, "hidden areas only" prompt; the shared
# matcher is description-enriched and carries the area-vs-point guard (the canonical
# Explainable wording the plan selected — the "HITL dropped the guard"/F-M1 family).

# Reformat an unmatched user sub-point into the terse framework sub-bullet style.
# No new content, no source — just style normalization. Change log: 2026-05-29
SUB_BULLET_FORMAT_PROMPT = """
You reformat a user's note into the terse style of a case-framework sub-bullet.

Rules:
- Keep the user's MEANING exactly — add no new content, examples, or sources.
- Output ONE short phrase (roughly 4–12 words), no leading dash, no trailing period.
- Drop filler ("I think", "we should also", "maybe", "consider").
- Match this style:
    "Payback period until cumulative benefits exceed total costs"
    "Single-developer dependency and key-person risk"
    "GDPR compliance for personal or confidential data"

User note: "{item}"

Output ONLY the reformatted phrase, nothing else.
"""

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

HITL_CLARIFICATION_SYSTEM_PROMPT = """
You are a strategic thinking partner facilitating a case interview session.

─── OPEN CLARIFICATION ───────────────────────────────────────────────────────
The candidate may ask clarifying questions about the case. Answer ONLY from
the CASE INFORMATION SHEET below. If a question is outside the sheet, say:
"I'm afraid I don't have that information for this case."

─── RULES ───────────────────────────────────────────────────────────────────
- Do NOT present the case or any framework concepts during this phase
- Do NOT coach or evaluate the candidate
- Keep responses concise — one to three sentences
- Never reveal what framework will be used
─────────────────────────────────────────────────────────────────────────────
"""

# Change log: 2026-05-29 — de-coupled from the coffee-shop / profitability case;
# presentation is now static (JSON), so this prompt only governs Q&A and the
# 2-bullet generation for user-added (non-KB) areas.
HITL_MAIN_SYSTEM_PROMPT = """
You are a strategic thinking partner facilitating a structured framework
walkthrough. You propose concepts one at a time — the candidate decides
whether to include each one. You facilitate, you do not direct.

─── RHETORICAL CONTEXT ──────────────────────────────────────────────────────
Audience : A candidate building a structured plan for the case above
Genre    : Concept-by-concept facilitated walkthrough with explicit approval
Purpose  : Surface each concept clearly and let the candidate decide
Subject  : The business problem described in the case above
Writer   : Strategic thinking partner — facilitator, not expert authority
─────────────────────────────────────────────────────────────────────────────

─── WHEN CANDIDATE ASKS A QUESTION ──────────────────────────────────────────
Answer naturally in 2–3 sentences. Stay grounded in the case above.
Plain language only — no jargon, no technical terms.
After answering, stop — do not re-present the concept block.
─────────────────────────────────────────────────────────────────────────────

─── RULES ───────────────────────────────────────────────────────────────────
- Never mention a knowledge graph, database, or technical system
- Never evaluate, score, or tell the candidate they are right or wrong
- Never suggest what the candidate should approve or reject
- One concept at a time — never present two concepts in one response
- Maximum 2 sub-bullets per concept
- Do NOT include sources or citations
- Facilitate, do not direct
─────────────────────────────────────────────────────────────────────────────
"""


class HITLAgent(BaseAgent):
    """
    Human-in-the-Loop agent — concept-by-concept walkthrough with explicit
    Include / Skip / Add buttons per concept, proactive prompts between concepts,
    and a deterministic 2-of-3 justification step.

    Change log: 2026-04-09 — initial build
    Change log: 2026-04-20 — proactive prompts, justification steps
    Change log: 2026-05-05 — removed Q1/Q2; warmup phase added
    Change log: 2026-05-12 — begin_analysis() replaces start_main_phase()
    Change log: 2026-05-16 — swap detection order fix; concept_added double-log fix
    Change log: 2026-05-29 — MIGRATED to GenAI JSON case (parity with Explainable):
                             static JSON presentation (no description, no sources),
                             withheld-pillar matching for new areas, deterministic
                             summary, deterministic 2-of-3 justification + min-substance
                             gate, ➕ Add sub-point flow (match key question, no source).
    Change log: 2026-06-07 — restored ➖ remove-point + ↩️ revisit-pillar handlers
                             (app.py expects them); pending_sub_excl / awaiting_revisit_add
                             state added. Remove-point logs delete at CONFIRM.
    """

    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self._init_flow_state()
        self.session_id    = create_session(user_id, agent_type="hitl")
        self.original_case = get_case("hitl")

        # ── Phase sequence: warmup → clarification → main ──────────────────
        self.clarification_facts = get_clarification_facts("hitl")

        # ── Concept Swap ───────────────────────────────────────────────────
        self.concept_swap = ConceptSwap(
            agent_type="hitl",
            session_id=self.session_id
        )

        # ── KG context (JSON-backed via inherited _fetch_kg_context) ───────
        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        logging.info(
            f"[KG INIT] case_type={CASE_TYPE}, "
            f"framework={self.kg_context['framework']}, "
            f"concepts={self.kg_context['concepts']}"
        )

        # ── Walkthrough state ──────────────────────────────────────────────
        self.walkthrough_concepts = []
        self.walkthrough_index    = 0
        self.walkthrough_active   = False
        self.walkthrough_done     = False
        self.approved_concepts    = []
        self.swap_presented       = False
        self.swap_position        = 0

        # ── Presented bullets — concept_name → bullet-lines string ─────────
        # Stored at present-time (JSON for KB pillars/swap, LLM for non-KB).
        # The deterministic summary renders from this. Change log: 2026-05-29
        self.concept_blocks = {}

        # ── User-added sub-points — concept_name → [point, ...] ────────────
        # Source-free (HITL suppresses sources). Rendered inline + in summary.
        # Change log: 2026-05-29

        # ── Deterministic 2-of-3 justification ─────────────────────────────
        # The shown pillars whose decision requires justification this session.
        # Populated in _build_walkthrough_concepts. Change log: 2026-05-29
        self.justification_pillars = set()
        self._justified_concepts   = set()   # 6f D3: concepts justified (accept or reject)

        # ── Pending confirmation state ─────────────────────────────────────
        # Step 4d: the per-arm pending_excl / pending_sub_excl (a concept string /
        # a (concept, bullet) tuple) are REPLACED by the shared PendingAction the
        # removal buttons park and interaction/handlers.resolve_pending resolves.
        # The shared confirmation machine fires the delete ONLY at stage="confirmed"
        # (F-R1), so on_reject_concept no longer logs a delete at intent (F-R4).

        # ── Interaction-mode flags ─────────────────────────────────────────
        self.awaiting_user_suggestion  = False
        self.awaiting_justification    = False
        self.justification_for         = None
        self.awaiting_sub_point        = False   # ➕ Add mode. Change log: 2026-05-29
        self.awaiting_revisit_add      = False   # ↩️ revisit add-mode
        self.revisit_target            = None    # pillar being revisited
        self.prompt_index              = 0
        self.ack_index                 = 0
        self.user_contributed_concepts = set()
        self.navigated_pillars         = set()   # dedupe add_pillar on navigate-to-planned (task 1)


        # ── Conversation history ───────────────────────────────────────────
        self.history = [
            types.Content(
                role="user",
                parts=[types.Part(text=self.original_case)]
            ),
            types.Content(
                role="model",
                parts=[types.Part(text=(
                    "I have received the case. We are now in the clarification round. "
                    "Feel free to ask any questions about the case before we begin."
                ))]
            ),
        ]

    # ══════════════════════════════════════════════════════════════════════
    # Opening message
    # ══════════════════════════════════════════════════════════════════════

    def get_opening_message(self) -> str:
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
            f"---\n"
            f"Take your time to read it. Feel free to ask any clarifying questions "
            f"about the case before you begin.\n\n"
            f"When you're ready, I'll walk you through a framework one concept at a "
            f"time. For each one, ask me a question before deciding or "
            f"you can **include** it, **skip** it, **add** your own points.\n\n"
        )

    def get_pre_analysis_instruction(self) -> str:
        return (
            "📖 *After you click the button below, I'll walk you through each concept "
            "one at a time. Use **Include**, **Skip**, or **➕ Add** for each concept, "
            "or type a question first.*"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Phase transition
    # ══════════════════════════════════════════════════════════════════════

    def begin_analysis(self):
        """Generator — called when user clicks 'Got it, show me the full analysis'."""
        self._start_main_phase_setup()

        yield (
            "⚠️ Your goal is to build a structured plan for this case. "
            "Review each factor below, share your thoughts, and you **should not only "
            "read it** but also add or remove anything you think is missing."
        )

        yield "⏱️ Your 20-minute session has started. The timer is shown on the left."

        self.walkthrough_concepts = self._build_walkthrough_concepts()
        self.walkthrough_active   = True
        self.walkthrough_index    = 0
        self.swap_presented       = False

        yield (
            f"✅ **Let's begin.**\n\n"
            f"I'll walk you through the framework one concept at a time. "
            f"For each one, use the buttons to **include** or **skip** it, "
            f"**➕ add** your own points, or just type a question first.\n\n"
            f"---\n\n"
        )
        yield from self._stream_concept(is_first=True)

    # ══════════════════════════════════════════════════════════════════════
    # Walkthrough state helpers
    # ══════════════════════════════════════════════════════════════════════

    def _build_walkthrough_concepts(self) -> list:
        base     = list(self.kg_context["concepts"])
        wrong    = self.concept_swap.config["wrong_concept"]
        position = len(base) // 2
        base.insert(position, wrong)
        self.swap_position = position

        # Deterministic 2-of-3: pick which shown pillars require a justification
        # this session. Arms from the first concept (no proactive-prompt dependency).
        # Swap + user-added concepts never require justification. Change log: 2026-05-29
        shown = [p["name"] for p in kb.get_shown_pillars()]
        self.justification_pillars = set(shown)   # 6f D2: every shown pillar requires a
        #                                           reason (random 2-of-3 retired; swap excluded
        #                                           by requires_justification/before_advance)

        logging.info(
            f"[WALKTHROUGH] built={base}, swap_position={position}, "
            f"justification_pillars={sorted(self.justification_pillars)}, "
            f"framework={self.kg_context['framework']}"
        )
        return base

    def _current_concept(self) -> str | None:
        excluded_lower = [e.lower() for e in self.excluded_concepts]
        while self.walkthrough_index < len(self.walkthrough_concepts):
            concept = self.walkthrough_concepts[self.walkthrough_index]
            if concept.lower() not in excluded_lower:
                return concept
            logging.debug(f"[WALKTHROUGH] skipping excluded: {concept}")
            self.walkthrough_index += 1
        return None

    def _is_wrong_concept(self, concept: str) -> bool:
        return concept.lower() == self.concept_swap.config["wrong_concept"].lower()

    def _walkthrough_complete_message(self):
        self.walkthrough_done = True
        yield (
            "✅ We've covered all the concepts. "
            "Click **‼️End Session** to see your final framework. "
            "**Note: this cannot be undone**.\n\n"
        )

    def _get_proactive_prompt(self) -> str:
        prompt = PROACTIVE_PROMPTS[self.prompt_index % len(PROACTIVE_PROMPTS)]
        self.prompt_index += 1
        return prompt

    def _locate_concept(self, name: str) -> int | None:
        """Index of a concept in the walkthrough (case-insensitive), or None."""
        for i, c in enumerate(self.walkthrough_concepts):
            if c.lower() == name.lower():
                return i
        return None

    def _normalize_pillar(self, name: str) -> str:
        """Resolve a loosely-cased pillar/concept name to its canonical form."""
        for c in self.walkthrough_concepts:
            if c.lower() == name.lower():
                return c
        for p in kb.get_all_pillars():
            if p["name"].lower() == name.lower():
                return p["name"]
        return name

    @staticmethod
    def _is_substantive_justification(text: str) -> bool:
        """Min-substance gate: real words, not 'asdf' / punctuation. No grading."""
        t = text.strip()
        words = [w for w in t.split() if any(ch.isalpha() for ch in w)]
        return len(words) >= 3 and len(t) >= 12

    # ── LLM classifiers ────────────────────────────────────────────────────

    # _classify_intent retired (Step 3, W3): the proactive branch now uses the ONE
    # shared backend.interaction.intents.classify_intent — both HITL entry points
    # (this + the inherited _detect_override) are replaced by the single router, so
    # one input can no longer reach the add path twice (F-M4).

    def _check_duplicate_proactive(self, concept: str) -> dict:
        """Check if a user-suggested concept matches one already in the walkthrough."""
        all_concepts = list(self.walkthrough_concepts)
        try:
            result = classify_json(
                    f"You are checking if a user-suggested concept matches any concept "
                    f"in an existing list.\n\n"
                    f"User suggested: \"{concept}\"\n\n"
                    f"Existing concepts: {all_concepts}\n\n"
                    f"Reply with JSON only, no markdown, no explanation:\n"
                    f"{{\"is_duplicate\": true or false, "
                    f"\"matched_concept\": \"exact string from list or null\"}}\n\n"
                    f"is_duplicate=true ONLY if the user suggestion is clearly the same "
                    f"concept as one in the list — same topic, possibly different wording.\n"
                    f"matched_concept must be the exact string from the list.\n"
                    f"WHEN IN DOUBT: is_duplicate=false."
            )
            logging.info(f"[DUPLICATE] check result: {result}")
            return result
        except Exception as e:
            logging.warning(f"[DUPLICATE] check failed: {e} — defaulting to not duplicate")
            return {"is_duplicate": False, "matched_concept": None}

    # _match_* wrappers collapsed (Step 6c): call domain.matching.* directly (I-3).

    def _format_sub_bullet(self, item: str) -> str:
        """
        Reformat an unmatched user sub-point into terse framework-bullet style.
        Style only — no new content, no source. Falls back to the raw input on
        any error. Change log: 2026-05-29
        """
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=SUB_BULLET_FORMAT_PROMPT.format(item=item),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            out = grounding._strip_source_refs(self._strip_fences(response.text)).strip().strip("-• ").rstrip(".")
            if out:
                logging.info(f"[SUB-POINT FORMAT] '{item}' → '{out}'")
                return out
        except Exception as e:
            logging.warning(f"[SUB-POINT FORMAT] error: {e} — keeping raw")
        return item.strip()

    def _store_sub_point(self, pillar: str, item: str, modality: str = "text",
                         source: str = "user_spontaneous") -> tuple[str, bool]:
        """
        Match item to a key question (cleaned, source-free) or keep raw; store under
        pillar in user_sub_points. Returns (stored_text, is_new) — is_new is False if
        an identical point is already recorded (e.g. two phrasings matching the same
        key question), so callers can avoid duplicates. Change log: 2026-05-29
        """
        pillar  = self._normalize_pillar(pillar)
        matched, _ = matching.match_key_question(item, pillar)
        # Matched → canonical key-question wording (source-free). No match →
        # reformat the user's words into framework-bullet style (no new content).
        stored  = matched if matched else self._format_sub_bullet(item)
        # Already a presented bullet for this concept (KB, swap, OR LLM-generated)
        # → already shown, don't duplicate. Source = concept_blocks, not KB, so this
        # also catches duplicates of generated bullets for user-added concepts.
        # Change log: 2026-06-02
        block = self.concept_blocks.get(pillar, "")
        shown_lines = [l.strip().lstrip("-• ").strip().lower()
                       for l in block.splitlines() if l.strip()]
        if stored.lower() in shown_lines:
            return stored, False
        existing = self.user_sub_points.setdefault(pillar, [])
        if any(s.lower() == stored.lower() for s in existing):
            logging.info(f"[SUB-POINT] duplicate skipped: '{stored}' under '{pillar}'")
            return stored, False
        existing.append(stored)
        # §3.6 add_sub_bullet via the ONE shared mapping. Only reached when genuinely new
        # (duplicates returned above), so counted=True. source/modality carry HITL's
        # elicitation + button affordance (I-4).
        ev.record(h.AddOutcome(action="added_new", pillar=pillar, level="sub_bullet",
                               counted=True, text=stored, source=source),
                  self._evctx(source=source, modality=modality), _sink)
        logging.info(f"[SUB-POINT] '{item}' → '{stored}' under '{pillar}'")
        return stored, True

    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 0. A parked removal owns this turn (Step 4d). The shared confirmation
        #      machine resolves it. For a justification-gated removal this free-text
        #      turn IS the reason (D-H2: a meaningful reason -> confirm [B8]; a weak
        #      one -> re-ask, stays parked, NO delete [B9]). A plain yes/no on a
        #      non-gated parked removal is classified. Buttons take the same machine
        #      via resolve_pending(decision=...) (D-Q1), so the firing is identical. ─
        if self.pending is not None:
            log_user_message(self.session_id, user_input)
            pa = self.pending
            if pa.requires_justification and pa.justification is None:
                # the reason turn — allow an explicit cancel, else treat as the reason
                if h._classify_confirmation(user_input) == "decline":
                    outcome = h.resolve_pending(self, user_input, decision="decline")
                else:
                    outcome = h.resolve_pending(self, user_input, decision="confirm",
                                                justification=user_input)
            else:
                outcome = h.resolve_pending(self, user_input)      # free-text yes/no
            yield from self.render_removal(outcome, pa=pa)
            return

        # ── 0a. ➕ Add mode — collect sub-points, re-render, stay open ──────
        if self.awaiting_sub_point:
            concept = self._current_concept() or "this concept"
            log_user_message(self.session_id, f"[SUB-POINT ADD] {user_input}")
            stored, is_new = self._store_sub_point(concept, user_input, modality="button")

            block  = self.concept_blocks.get(concept, "").strip()
            points = self.user_sub_points.get(concept, [])
            rerender = f"**{concept}**"
            if block:
                rerender += f"\n{block}"
            if points:
                rerender += "\n" + "\n".join(f"- {p}" for p in points)

            lead = (
                f"Added under **{concept}**. Here's how it looks now:"
                if is_new else
                f"You've already got that one under **{concept}**. Here's how it looks now:"
            )
            yield (
                f"{lead}\n\n"
                f"{rerender}\n\n"
                f"Anything else to add under **{concept}**? "
                f"Click **✅ Done adding** when you're finished."
            )
            return

        # ── 0b. ↩️ Revisit-add mode — add a point to a PAST pillar, stay open ─
        if self.awaiting_revisit_add:
            target = self.revisit_target or self._current_concept() or "this concept"
            log_user_message(self.session_id, f"[REVISIT ADD] {user_input}")
            stored, is_new = self._store_sub_point(target, user_input, modality="button")
            lead = (f"Added to **{target}**." if is_new
                    else f"You've already got that one under **{target}**.")
            yield (
                f"{lead}\n\n"
                f"Anything else to add to **{target}**? "
                f"Click **✅ Done adding** when you're finished."
            )
            return

        # ── 1. Justification collection (min-substance gate) ───────────────
        if self.awaiting_justification:
            if not h.is_meaningful_justification(user_input):   # 6f: shared effort gate
                log_user_message(self.session_id, f"[JUSTIFICATION:retry] {user_input}")
                yield "Could you say a bit more about your reasoning? A sentence is plenty.\n\n"
                return  # stay in awaiting_justification

            log_user_message(self.session_id, f"[JUSTIFICATION:{self.justification_for}] {user_input}")
            logging.info(f"[JUSTIFICATION] collected for={self.justification_for}: '{user_input}'")
            self.awaiting_justification = False
            self.justification_for      = None
            # 6f accept-flow (item B): record the held concept as justified, then
            # release the advance on_approve_concept / cancel-keep parked pending this
            # reason. Index still points at the held concept (neither path advanced).
            cur = self._current_concept()
            if cur is not None:
                self._justified_concepts.add(cur)
            self.walkthrough_index += 1
            yield from self._stream_justification_ack()
            return

        # ── 2. Proactive suggestion handling ───────────────────────────────
        if self.awaiting_user_suggestion:
            self.awaiting_user_suggestion = False
            log_user_message(self.session_id, f"[PROACTIVE RESPONSE] {user_input}")

            # Step 3 (W3): the proactive response goes through the ONE shared router
            # (I-1/I-2), NOT a second classifier. State-dependent persona rendering:
            # in this ELICITED-suggestion state an `add` is EXECUTED (the legitimate
            # user_elicited contribution) — the normal turn nudges instead (F-M4).
            pres = intents.classify_intent(
                user_input,
                current_pillar=self._current_concept() or "(none)",
                current_bullets=self._ctx_bullets(),
                walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
                last_agent=self._last_agent_text() or "(nothing yet)",
            )

            # add + explicit parent ("X under Y") = a sub-point on an existing pillar.
            if pres.intent == "add" and pres.detail and pres.parent:
                logging.info(f"[PROACTIVE] sub-point: '{pres.detail}' → '{pres.parent}'")
                yield from self._add_sub_point(pres.parent, pres.detail)
                yield from self._stream_concept_qa()  # stay on current concept
                return

            # Anything that is NOT the user naming their own concept = 'guide me'
            # (advance / ask_agent_to_suggest / question / doubt / none): proceed with
            # the planned next concept, exactly as the old `guidance` branch did.
            if not (pres.intent == "add" and pres.detail):
                logging.info(f"[PROACTIVE] guidance/continue (intent={pres.intent})")
                yield from self._stream_concept(is_first=False)
                return

            # suggestion — the user named their OWN concept to explore. Resolve to a
            # target (existing / withheld pillar / genuinely new), never a duplicate.
            concept = pres.detail
            dup     = self._check_duplicate_proactive(concept)

            if dup["is_duplicate"] and dup["matched_concept"]:
                target           = dup["matched_concept"]
                matched_withheld = None
            else:
                matched_withheld, _ = matching.match_pillar(concept)
                target           = matched_withheld or concept

            idx = self._locate_concept(target)
            if idx is not None:
                # Already in the walkthrough — navigate, never re-insert
                if idx >= self.walkthrough_index:
                    if idx != self.walkthrough_index:
                        self.walkthrough_concepts.pop(idx)
                        self.walkthrough_concepts.insert(self.walkthrough_index, target)
                    # Navigate-to-planned = agency: participant can't see what was
                    # planned, so proactively naming it is an add_pillar act.
                    # Log once per concept. Change log: 2026-06-02
                    if target.lower() not in self.navigated_pillars:
                        self.navigated_pillars.add(target.lower())
                        ev.record(h.AddOutcome(action="added_new", pillar=target,
                                               level="pillar", counted=True,
                                               source="user_elicited"),
                                  self._evctx(source="user_elicited", modality="text"), _sink)
                    logging.info(f"[PROACTIVE] navigate to '{target}' (was idx={idx})")
                    yield f"Sure — let's look at **{target}** now.\n\n"
                    yield from self._stream_concept(is_first=False)
                else:
                    logging.info(f"[PROACTIVE] '{target}' already covered (idx={idx})")
                    yield (
                        f"We've already covered **{target}** — it's in your "
                        f"framework.\n\n"
                    )
                    yield from self._stream_proactive_prompt()
            else:
                # Genuinely new — insert at current position and present WITH buttons
                if matched_withheld:
                    logging.info(f"[PROACTIVE] new area → withheld pillar '{matched_withheld}'")
                    yield (
                        f"Good call — **{matched_withheld}** is an important area. "
                        f"Let's add it.\n\n"
                    )
                else:
                    logging.info(f"[PROACTIVE] new user concept: '{target}'")
                    yield f"Great suggestion — let's explore **{target}** now.\n\n"
                yield from self._stream_user_contributed_concept(target)
            return

        # ── 3. Unified intent (Step 3) — ONE shared router (I-1/I-2) replaces BOTH
        #      _detect_override (this normal turn) AND _classify_intent (proactive).
        #      F-M4 fixed STRUCTURALLY: a FREE-TEXT add no longer executes here (it
        #      nudges to the ➕ button, D-Q1), so add_pillar cannot fire via two paths
        #      for one input. W8: no confidence floor.
        just_added_concept = None
        res = intents.classify_intent(
            user_input,
            current_pillar=self._current_concept() or "(none)",
            current_bullets=self._ctx_bullets(),
            walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
            last_agent=self._last_agent_text() or "(nothing yet)",
        )
        intent = res.intent

        # ── 3b. Swap removal is the SAME button path as any pillar (Katie's instrument
        #      decision 2026-06-09; resolves F-S3 for HITL). A free-text "remove [swap]"
        #      is NOT specially force-detected on a steering `remove` intent (that would
        #      contradict W2, which runs swap-check on NON-steering intents only). Like
        #      any free-text remove it is NUDGED to ❌ (D-Q1, below); detection happens
        #      when the user clicks ❌ -> on_confirm_reject -> force_detected (swap
        #      DETECTED, never a delete). swap_questioned (question on the swap) + the
        #      semantic backstop (below) are the remaining text channels.
        cs_detected = False
        swap_live   = self.swap_presented and not self.concept_swap.is_detected

        # ── 4. Swap detection — semantic backstop, NON-STEERING only
        #      (W2: intent not in (add, remove) ≡ the old `not override` gate).
        if swap_live and intent not in ("add", "remove"):
            cs_detected = self.concept_swap.check_detection(user_input)
            if cs_detected:
                wrong = self.concept_swap.config["wrong_concept"]
                # check_detection() already fired §3.6 swap_detected via ConceptSwap._log_detected.
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)
                self.walkthrough_index += 1
                logging.info(f"[SWAP] caught via text — index→{self.walkthrough_index}")

        # ── 4b. Invariant check ────────────────────────────────────────────
        if (self.walkthrough_active
                and self.walkthrough_index > self.swap_position
                and not self.swap_presented):
            logging.error(f"[INVARIANT] rewinding to swap_position={self.swap_position}")
            self.walkthrough_index = self.swap_position

        # ── 5. Log and append user message ─────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 6. Routing log ─────────────────────────────────────────────────
        logging.info(
            f"[ROUTE] active={self.walkthrough_active}, done={self.walkthrough_done}, "
            f"swap_presented={self.swap_presented}, index={self.walkthrough_index}, "
            f"intent={intent}, cs_detected={cs_detected}"
        )

        # ── 7. Route ───────────────────────────────────────────────────────
        if not self.walkthrough_active:
            self.walkthrough_concepts     = self._build_walkthrough_concepts()
            self.walkthrough_active       = True
            self.walkthrough_index        = 0
            self.swap_presented           = False
            yield from self._stream_concept(is_first=True)

        elif self.walkthrough_done:
            yield from self._stream_freeform()

        elif cs_detected:
            yield from self._stream_swap_caught()
            yield "\n\n"
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

        elif intent in _BUTTON_ACTION_INTENTS:
            # D-Q1: HITL exposes add/remove/revisit/advance as buttons, so a FREE-TEXT
            # action is NUDGED to its button instead of executed (intended HITL change,
            # W5 convergence; the F-M4 fix). Full nudge render is a Step-4 seam.
            yield from self._nudge_to_button(intent)

        elif intent == "ask_agent_to_suggest":
            # Answered directly (D-Q1): STATE a grounded withheld-KB suggestion, never
            # free-form (catalog D1). Suggesting is not adding.
            yield from self._handle_suggest(user_input)

        else:
            # question / doubt / none — answered directly via KB-grounded Q&A.
            current  = self._current_concept()
            _on_swap = (swap_live and current is not None
                        and self._is_wrong_concept(current))
            # Parity with BlackBox/Explainable: a typed turn landing in Q&A IS a
            # question — §3.6 question, with swap_questioned the on-swap subset (W9).
            ev.question(self._evctx(modality="text"), _sink)
            if _on_swap:
                ev.swap_questioned(self._evctx(modality="text"), _sink)
            yield from self._stream_concept_qa(just_added=just_added_concept)

    # ══════════════════════════════════════════════════════════════════════
    # Proactive prompt + justification
    # ══════════════════════════════════════════════════════════════════════

    def _ctx_bullets(self) -> str:
        """Current concept's visible points (KB block + user sub-points) — router context."""
        cur   = self._current_concept() or ""
        block = (self.concept_blocks.get(cur, "") or "").strip()
        pts   = self.user_sub_points.get(cur, [])
        lines = ([block] if block else []) + [f"- {p}" for p in pts]
        return "\n".join(lines) or "(none)"

    def _handle_suggest(self, user_input: str):
        """ask_agent_to_suggest (normal turn) — STATE a suggestion drawn
        DETERMINISTICALLY from the WITHHELD KB (catalog D1), grounded, NEVER free-form
        -> cannot confabulate. Suggesting is NOT adding (no artifact, no add_pillar).
        HITL override of the inherited BlackBox method: HITL tracks surfaced pillars
        differently (walkthrough_concepts / user_contributed_concepts / excluded_
        concepts, NOT user_added_pillars). The shared SuggestOutcome handler unifies
        all three arms in Step 4 (§S)."""
        surfaced = ({c.lower() for c in self.walkthrough_concepts}
                    | {c.lower() for c in self.user_contributed_concepts}
                    | {e.lower() for e in self.excluded_concepts})
        withheld = [p for p in kb.get_all_pillars()
                    if not p.get("shown", False) and p["name"].lower() not in surfaced]
        if withheld:
            target = withheld[0]
            why = grounding.ground_pillar(target["name"]).split("\n")[0].strip()
            # §3.6 ask_agent_suggestion OFFER (accepted=False) — the agent NAMED a withheld
            # area on user request. Suggesting is not adding (I-4); accepting it later is a
            # separate ask_agent_suggestion(accepted=True), never an add_pillar.
            ev.record(h.SuggestOutcome(level="pillar", suggested_item=target["name"],
                                       accepted=False, revealed=False),
                      self._evctx(modality="text"), _sink)
            msg = (f"One area we haven't covered yet is **{target['name']}**"
                   + (f" — {why}" if why else "")
                   + "\n\nIf it fits your case, you can add it with the **➕** button.")
        else:
            msg = ("You've surfaced the main areas I'd flag — use the buttons to add, "
                   "skip, or revisit any part of the framework.")
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _nudge_to_button(self, intent: str):
        """D-Q1 persona render: HITL surfaces add/remove/revisit/advance as buttons, so
        a FREE-TEXT action is nudged to its button rather than executed. This is the
        intended HITL behaviour change behind the F-M4 fix (a free-text add no longer
        logs add_pillar a second time). The richer nudge render is a Step-4 seam."""
        label = _HITL_BUTTON_LABELS.get(intent, "the buttons below")
        msg = ("In this mode each change is made with a button, so it stays explicit. "
               f"You can do that with {label} below — go ahead and click when you're ready.")
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _stream_proactive_prompt(self):
        self.awaiting_user_suggestion = True
        prompt = self._get_proactive_prompt()
        logging.info(f"[PROACTIVE] prompt_index={self.prompt_index - 1}")
        yield prompt

    def _stream_justification_prompt(self, for_decision: str, concept: str = None):
        self.awaiting_justification = True
        self.justification_for      = for_decision
        concept_name = concept or self._current_concept() or "this concept"

        if for_decision == "accept":
            yield (
                f"Before we move on — what makes **{concept_name}** essential for this case? "
                f"What would we risk missing if we excluded it?"
            )
        else:
            yield (
                f"You've chosen to leave out **{concept_name}**. "
                f"What's your reasoning for excluding it?"
            )

    def _stream_justification_ack(self):
        ack = JUSTIFICATION_ACKS[self.ack_index % len(JUSTIFICATION_ACKS)]
        self.ack_index += 1
        yield ack + "\n\n"
        next_concept = self._current_concept()
        if next_concept is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_proactive_prompt()

    def _stream_user_contributed_concept(self, concept: str):
        """
        Insert a user-suggested area at the CURRENT position and present it WITH
        Include / Skip / ➕ Add buttons — NOT auto-approved. This lets the user add
        sub-points to the new pillar and decide to include or skip it, exactly like
        any other concept. Change log: 2026-05-29 (revised — was auto-approve)
        """
        self.walkthrough_concepts.insert(self.walkthrough_index, concept)
        if self.walkthrough_index <= self.swap_position:
            self.swap_position += 1
            logging.info(f"[USER CONCEPT] swap_position shifted to {self.swap_position}")
        # User named their OWN area in response to the proactive prompt -> elicited add_pillar
        # (a user contribution, NOT an agent suggestion; I-4).
        ev.record(h.AddOutcome(action="added_new", pillar=concept, level="pillar",
                               counted=True, source="user_elicited"),
                  self._evctx(source="user_elicited", modality="text"), _sink)
        logging.info(f"[USER CONCEPT] inserted at index={self.walkthrough_index}: '{concept}'")

        # Present as the current concept; Include/Skip/Add buttons handle the rest.
        # (Not added to user_contributed_concepts, so should_show_buttons() is True.)
        yield from self._stream_concept(is_first=False)

    # ══════════════════════════════════════════════════════════════════════
    # Streaming sub-methods
    # ══════════════════════════════════════════════════════════════════════

    def _add_sub_point(self, matched_concept: str, sub_point: str):
        """
        Store a sub-point under a concept (match key question, source-free) and
        yield a confirmation. Used by the proactive sub_point + override paths.
        Change log: 2026-05-29 — writes user_sub_points; dedupe-aware
        """
        # Deictic parent ("this concept"/"this"/"it"/"here") names no pillar — the
        # classifier emits it raw (D-Q2). Resolve to the CURRENT concept HERE, before
        # _normalize_pillar (which only canonicalises real names): otherwise an elicited
        # "add under this concept: X" stores X under a phantom pillar named "this
        # concept", corrupting the elicited-contribution DV attribution.
        # Change log: 2026-06-09 (Step-5 gap #1).
        if _is_deictic_parent(matched_concept):
            current = self._current_concept()
            if current:
                logging.info(f"[PROACTIVE] deictic parent {matched_concept!r} -> {current!r}")
                matched_concept = current
        pillar = self._normalize_pillar(matched_concept)
        # Only caller is the proactive (awaiting_user_suggestion) path -> elicited (I-4).
        stored, is_new = self._store_sub_point(pillar, sub_point, source="user_elicited")
        if is_new:
            yield f"Got it — adding that under **{pillar}**:\n- {stored}\n\n"
        else:
            yield f"You've already got that one under **{pillar}**.\n\n"

    def _stream_concept(self, is_first: bool):
        concept = self._current_concept()
        if concept is None:
            yield from self._walkthrough_complete_message()
            return

        is_wrong = self._is_wrong_concept(concept)

        # Resolve to a KB pillar by name (walkthrough uses pillar names)
        pillar = None
        if not is_wrong:
            pillar = next(
                (p for p in kb.get_all_pillars()
                 if p["name"].lower() == concept.lower()),
                None
            )

        # ── Static presentation: KB pillar or swap — no LLM, no sources ───
        if is_wrong or pillar is not None:
            if is_wrong:
                swap    = kb.get_swap_concept()
                bullets = swap.get("sub_bullets", []) if swap else []
            else:
                bullets = pillar.get("sub_bullets", [])

            # Strip inline [a][b] refs — HITL suppresses sources. Stripping here
            # means both the live block AND the summary (which reads concept_blocks)
            # come out clean. Change log: 2026-05-29
            bullets = [grounding._strip_source_refs(b) for b in bullets]
            bullet_lines = "\n".join(f"- {b}" for b in bullets)
            # Store bullets ONLY (no heading) for the deterministic summary
            self.concept_blocks[concept] = bullet_lines

            # Fold in any sub-points added earlier while on another concept
            display = bullet_lines
            added   = self.user_sub_points.get(concept, [])
            if added:
                display += "\n" + "\n".join(f"- {p}" for p in added)

            prefix = f"**{concept}**\n{display}"
            if is_first:
                prefix = "💡 When you're finished, click ‼️End Session to close your session. Note: this cannot be undone. \n\n Here is how I would structure this analysis:\n\n" + prefix

            self.history.append(
                types.Content(role="user",
                              parts=[types.Part(text=f"[Present concept: {concept}]")])
            )
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=prefix)])
            )
            log_agent_response(self.session_id, prefix)

            if is_wrong:
                self.swap_presented = True
                if not self.concept_swap.is_injected:
                    self.concept_swap.maybe_inject(prefix)
                    self.concept_swap.log_presented()
                    logging.info(f"[SWAP] presented at position={self.swap_position}")

            logging.info(f"[CONCEPT BLOCK] static stored for '{concept}'")
            yield prefix
            return

        # ── LLM presentation: non-KB user-added concept — 2 bullets, no sources ──
        heading = "Here is how I would structure this analysis:\n\n" if is_first else ""
        heading += f"**{concept}**\n"
        yield heading

        instruction = (
            f"{self._build_system_prompt()}\n\n"
            f"─── CONCEPT TO PRESENT NOW ───────────────────────────────────\n"
            f"Concept: **{concept}**\n"
            f"This is a user-added area, not part of the standard framework.\n"
            f"Output ONLY 2 sub-bullets for this concept, specific to this case.\n"
            f"Do NOT repeat the concept name.\n"
            f"Do NOT add a closing question — buttons handle advancement.\n"
            f"Do NOT include any sources or citations.\n"
            f"Maximum 2 sub-bullets.\n"
            f"─────────────────────────────────────────────────────────────\n"
        )
        task_injection = (
            f"[Output only 2 sub-bullets for **{concept}**. "
            f"No concept name, no closing question, no sources.]"
        )

        full_reply = []
        for token in self._stream_with_instruction(
            instruction    = instruction,
            prefix         = "",
            task_injection = task_injection,
            track_swap     = False,
            store_answer   = False,
        ):
            full_reply.append(token)
            yield token

        self.concept_blocks[concept] = "".join(full_reply).strip()
        logging.info(f"[CONCEPT BLOCK] LLM-generated stored for '{concept}'")

    def _stream_concept_qa(self, just_added: str | None = None):
        current = self._current_concept()
        concept = current or "the current concept"
        on_swap = (
            self.swap_presented
            and not self.concept_swap.is_detected
            and current is not None
            and self._is_wrong_concept(current)
        )

        added_note = ""
        grounding = self._concept_grounding(concept)
        grounding_block = ""
        if grounding:
            grounding_block = (
                f"─── KNOWN POINTS FOR THIS CONCEPT (ground your answer here) ───\n"
                f"{grounding}\n"
                f"─── GROUNDING RULE ───────────────────────────────────────────\n"
                f"Base your answer on the KNOWN POINTS above and the case. Do NOT\n"
                f"introduce regulations, statistics, or framework concepts that are\n"
                f"not among the known points. Do NOT output bracketed letter markers.\n"
                f"─────────────────────────────────────────────────────────────\n\n"
            )
        if just_added:
            added_note = (
                f"Good idea — I'll add **{just_added}** after we finish "
                f"**{concept}**.\n\n"
            )

        instruction = (
            f"{self._build_system_prompt()}\n\n"
            f"─── CURRENT CONCEPT ──────────────────────────────────────────\n"
            f"Concept: **{concept}**\n"
            f"On swap concept: {on_swap}\n"
            f"Framework: {self.kg_context['framework']} | "
            f"Case: {self.kg_context['case_type']}\n"
            f"Framework concepts (in order): "
            f"{', '.join(c for c in self.walkthrough_concepts if just_added is None or c.lower() != just_added.lower())}\n"
            f"─────────────────────────────────────────────────────────────\n\n"
            f"─────────────────────────────────────────────────────────────\n\n"
            f"{grounding_block}"
            f"─── RULES ────────────────────────────────────────────────────\n"
            f"Answer in 2–3 sentences. Plain language only.\n"
            f"Do NOT end with a question — buttons handle advancement.\n"
            f"Do NOT re-present the concept block.\n"
            f"Do NOT present or describe any other concept.\n"
            f"Do NOT claim to add, remove, keep, or skip anything.\n"
            f"Do NOT suggest whether the candidate should approve or reject.\n"
            f"─────────────────────────────────────────────────────────────\n"
        )

        yield from self._stream_with_instruction(
            instruction = instruction,
            prefix      = added_note,
        )
        # After a question, channel the decision back to the buttons — stops users
        # from typing "skip this" (which only logs a phantom override and doesn't
        # act). Gated on should_show_buttons so it never points at hidden buttons.
        # Change log: 2026-06-02
        if self.should_show_buttons():
            yield (
                "\n\n*Would you like to **include** or **skip** this, "
                "or **➕ add** a point? Just use the buttons below.*"
            )
    def _concept_grounding(self, concept: str) -> str:
        """Q&A grounding → shared grounding.ground_pillar (Step 2): description +
        key-questions (refs stripped), plus the planted-swap fallback. Behaviour-
        identical to the old local copy — HITL already stripped refs, and the shared
        swap fallback keys off the KB swap concept's own name (== this arm's
        wrong_concept)."""
        return grounding.ground_pillar(concept)
    def _stream_swap_caught(self):
        wrong       = self.concept_swap.config["wrong_concept"]
        wrong_fw    = self.concept_swap.config["wrong_framework"]
        active_fw   = self.kg_context["framework"]
        active_case = self.kg_context["case_type"]

        instruction = (
            f"{self._build_system_prompt()}\n\n"
            f"─── CONTEXT ──────────────────────────────────────────────────\n"
            f"Wrong concept flagged: **{wrong}**\n"
            f"It belongs to: {wrong_fw} (a different type of analysis)\n"
            f"This case is: {active_case} — {active_fw}\n"
            f"─────────────────────────────────────────────────────────────\n\n"
            f"─── RULES ────────────────────────────────────────────────────\n"
            f"Respond in 2–3 sentences:\n"
            f"1. Acknowledge the catch warmly\n"
            f"2. Explain briefly why it belongs to a different analysis\n"
            f"3. One closing sentence confirming you are moving on\n"
            f"- Do NOT end with a question\n"
            f"- Do NOT preview the next concept\n"
            f"─────────────────────────────────────────────────────────────\n"
        )

        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_freeform(self):
        concepts_str = " → ".join(self.kg_context["concepts"])
        instruction  = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────\n\n"
            f"You are a strategic thinking partner. The user has seen the full "
            f"framework. Answer their question concisely in plain language. "
            f"Ask ONE follow-up question after answering."
        )
        yield from self._stream_with_instruction(instruction=instruction)

    # ══════════════════════════════════════════════════════════════════════
    # System prompts
    # ══════════════════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> str:
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"

        kg_block = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────\n"
            f"Case Type : {self.kg_context['case_type']}\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"(These are the correct ordered concepts. Ground your answer here.)\n"
            f"──────────────────────────────────────────────────────────────\n\n"
        )

        swap_block = self.concept_swap.get_system_prompt_block()
        return kg_block + swap_block + HITL_MAIN_SYSTEM_PROMPT

    def _build_clarification_system_prompt(self) -> str:
        if self.clarification_facts:
            facts_lines = "\n".join(
                f"- {topic.upper()}: {answer}"
                for topic, answer in self.clarification_facts.items()
            )
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────\n"
                f"{facts_lines}\n"
                f"──────────────────────────────────────────────────────────────\n\n"
            )
        else:
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────\n"
                f"No additional facts are available for this case.\n"
                f"Deflect all clarification questions with: "
                f"\"I'm afraid I don't have that information for this case.\"\n"
                f"──────────────────────────────────────────────────────────────\n\n"
            )
        return facts_block + HITL_CLARIFICATION_SYSTEM_PROMPT

    # ══════════════════════════════════════════════════════════════════════
    # Button handlers
    # ══════════════════════════════════════════════════════════════════════

    def on_approve_concept(self):
        concept = self._current_concept()
        if concept is None:
            return

        if concept not in self.approved_concepts:
            self.approved_concepts.append(concept)

        logging.info(f"[APPROVE] concept='{concept}', index={self.walkthrough_index}")

        # 6f accept-flow (item B): HOLD the advance until the concept is justified
        # (mirror of the reject side). before_advance() consults _justified_concepts —
        # incl. the swap (item A) and every shown pillar (D2). Not yet justified -> ask
        # and do NOT advance; the typed reason populates _justified_concepts and releases
        # the advance (success tail in _stream_main).
        if not self.before_advance(self):
            yield from self._stream_justification_prompt("accept", concept=concept)
            return

        self.walkthrough_index += 1
        next_concept = self._current_concept()
        if next_concept is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_proactive_prompt()

    def on_reject_concept(self):
        """❌ Skip (first removal turn) — PARK a PendingAction; NOTHING is deleted here
        (F-R1: delete fires only at confirm; F-R4: the old intent-stage delete is gone).
        The swap is parked as is_swap -> detection on confirm, never a delete (§0/F-S3).
        A justification-pillar parks requires_justification -> the next typed turn is the
        reason gate (D-H2); other pillars confirm/cancel via the buttons (D-Q1)."""
        concept = self._current_concept()
        if concept is None:
            return
        is_swap = self._is_wrong_concept(concept)
        # 6f decision: every concept asks for a reason on skip, INCLUDING the swap, so the
        # swap is not the lone concept skipped without justification (removes a UI tell;
        # the reason also separates genuine detection from incidental skipping). The swap
        # still resolves as DETECTION on confirm (is_swap), never a delete.
        req = (concept in self.justification_pillars) or is_swap
        self.pending = h.PendingAction(
            type="remove_pillar", target=concept, level="pillar",
            is_swap=is_swap, requires_justification=req)
        logging.info(f"[REJECT] parked pending concept='{concept}' (swap={is_swap}, req_just={req})")
        if req:
            yield (
                f"Before we skip **{concept}** — what's your reasoning for leaving it "
                f"out? A sentence is plenty."
            )
        else:
            yield (
                f"Are you sure you want to skip **{concept}**?\n\n"
                f"*Use the buttons below to confirm.*"
            )

    def on_confirm_reject(self):
        """❌ confirm button (D-Q1, bypasses the LLM): resolve the parked removal via the
        SHARED machine. Delete fires only at stage='confirmed' (F-R1); swap -> detection,
        NO delete (§0). A justification-gated pillar is resolved by the typed reason in
        _stream_main, not this button — clicking confirm with no reason re-asks (B9)."""
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="confirm")
        yield from self.render_removal(outcome, pa=pa)

    def on_cancel_reject(self):
        """❌ cancel button (D-Q1): abandon the parked removal -> keep the concept. NO
        delete. Keeping = accept side -> the accept-side justification reflection still
        fires for a justification pillar (that gate is the Step-6 reconciliation, left
        as baseline here)."""
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="decline")
        yield from self.render_removal(outcome, pa=pa)

    # ── shared-outcome renderer for the removal buttons + the typed-reason turn ──
    def render_removal(self, o, user_input=None, *, was_pending=False, pa=None):
        """Render a RemovalOutcome from the shared machine and drive HITL navigation.
        Step 5: §3.6 events fire ONCE here via the shared record (the stage drives the
        firing — F-R1: delete only at confirmed; swap = detection, never delete). pa is the
        PendingAction snapshot (resolve_pending may have cleared self.pending)."""
        ev.record(o, self._evctx(modality="button"), _sink)
        stage = o.stage

        if stage == "needs_justification":
            # gate failed (B9) — re-ask, stay parked, NOTHING deleted.
            yield "Could you say a bit more about your reasoning? A sentence is plenty.\n\n"
            return

        if stage == "confirmed":
            if pa.is_swap:
                # §0 — swap DETECTED on confirm (swap_detected+swap_removed fired by record at
                # top; mark_swap_detected ran in _confirm_removal). NEVER a delete.
                logging.info(f"[SWAP] detected via Reject button — concept='{pa.target}'")
                self.walkthrough_index += 1
                yield f"Got it — removing **{pa.target}** from the framework.\n\n"
                yield from self._after_removal_continue(pa.target, was_reject=True)
                return
            if pa.type == "remove_sub_bullet":
                # _confirm_removal recorded the exclusion; mirror it into HITL's render
                # sources so the block re-renders without it. delete_sub_bullet fired by record.
                self._apply_sub_bullet_removal(pa.pillar, pa.target)
                logging.info(f"[REMOVE POINT CONFIRMED] '{pa.target}' from '{pa.pillar}'")
                yield f"Done — removed that point from **{pa.pillar}**.\n\n"
                return
            # whole pillar — _confirm_removal already excluded it; the cursor auto-skips.
            # delete_pillar fired by record at top (F-R1: at CONFIRM).
            logging.info(f"[REJECT CONFIRMED] concept='{pa.target}'")
            self.walkthrough_index += 1
            yield f"Got it — removing **{pa.target}** from the framework.\n\n"
            yield from self._after_removal_continue(pa.target, was_reject=True)
            return

        if stage == "abandoned":
            if pa.type == "remove_sub_bullet":
                logging.info(f"[REMOVE POINT CANCELLED] keeping point under '{pa.pillar}'")
                yield f"No problem — keeping that point in **{pa.pillar}**.\n\n"
                return
            # pillar kept = accept side
            if pa.target and pa.target not in self.approved_concepts:
                self.approved_concepts.append(pa.target)
            logging.info(f"[REJECT CANCELLED] concept='{pa.target}' kept in framework")
            # 6f accept-flow (item B): keeping = accept side -> hold the advance
            # until justified, exactly like on_approve_concept (before_advance is the
            # gate; index still points at the kept concept here).
            if not self.before_advance(self):
                yield from self._stream_justification_prompt("accept", concept=pa.target)
                return
            self.walkthrough_index += 1
            yield f"Keeping **{pa.target}** — let's continue.\n\n"
            yield from self._after_removal_continue(pa.target, was_reject=False)
            return

        # defensive (challenged/other shouldn't surface via the button/reason paths)
        yield "Reply **yes** to remove, or **no** to keep it.\n\n"

    def _after_removal_continue(self, concept, *, was_reject):
        """Post-removal navigation only. 6f accept-flow (item B): the accept/keep-side
        justification gate now lives UPSTREAM in before_advance (consulted by
        on_approve_concept and the abandoned/pillar-kept branch). The old
        `in justification_pillars` reflection trigger was REMOVED here — leaving it would
        re-fire the prompt for an already-justified concept. Reject-side justification is
        the PRE-confirm gate; there is no post-reject reflection prompt."""
        next_concept = self._current_concept()
        if next_concept is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_proactive_prompt()

    def _apply_sub_bullet_removal(self, concept: str, bullet: str):
        """Strip a confirmed-removed bullet from HITL's render sources (the shared
        excluded_sub_bullets record was already written by _confirm_removal)."""
        target = bullet.strip().lstrip("-• ").strip().lower()
        block = self.concept_blocks.get(concept, "")
        if block:
            self.concept_blocks[concept] = "\n".join(
                l for l in block.splitlines()
                if l.strip().lstrip("-• ").strip().lower() != target
            )
        pts = self.user_sub_points.get(concept, [])
        self.user_sub_points[concept] = [p for p in pts if p.strip().lower() != target]

    # ══════════════════════════════════════════════════════════════════════
    # 6h-3 — public render seams (BaseAgent contract, §3.7).
    # HITL is the button arm: it renders via its button/stream flow and does NOT
    # use the shared `_render_outcome` router. render_removal (above) is the one
    # real outcome-renderer. render_summary/render_framework are contract-symmetry
    # delegators. render_add/question/next_steps/fallback are loud-guard contract
    # seams (decision (i)): present so load-time enforcement holds and no silent
    # 4th behavior can appear; if ever reached off-path they fail loudly.
    # ══════════════════════════════════════════════════════════════════════
    def render_summary(self):
        # contract seam — HITL's terminal I-6 summary (shared 6g walked-subset).
        yield from self._stream_summary()

    def render_framework(self, preamble=""):
        # contract seam (7-seam symmetry; not invoked by the base router for the
        # button arm). Faithful 'show current state': current concept, else summary.
        if preamble:
            yield preamble
        if self._current_concept() is None:
            yield from self._stream_summary()
        else:
            yield from self._stream_concept(is_first=False)

    def render_add(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders adds via its button/stream flow (➕ Add → _store_sub_point / "
            "re-render in _stream_main), not the outcome router (6h-3, decision (i)).")

    def render_question(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders questions via _stream_concept_qa in its button/stream flow, "
            "not the outcome router (6h-3, decision (i)).")

    def render_next_steps(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders suggestions via _handle_suggest in its button/stream flow, "
            "not the outcome router (6h-3, decision (i)).")

    def render_fallback(self, *args, **kwargs):
        raise NotImplementedError(
            "HITL renders advance/affordance via _stream_proactive_prompt / "
            "_nudge_to_button in its button/stream flow, not the outcome router "
            "(6h-3, decision (i)).")

    def on_add_to_concept(self):
        """➕ Add — open add-mode for the current concept. Change log: 2026-05-29"""
        concept = self._current_concept()
        if concept is None:
            return
        self.awaiting_sub_point = True
        logging.info(f"[ADD] add-mode opened for concept='{concept}'")
        yield (
            f"What point would you like to add under **{concept}**? "
            f"You can add as many as you like — just type each one, and "
            f"click **✅ Done adding** when you're finished."
        )

    def on_done_adding(self):
        """✅ Done adding — close add-mode (sub-point OR revisit)."""
        self.awaiting_sub_point   = False
        self.awaiting_revisit_add = False
        self.revisit_target       = None
        concept = self._current_concept() or "this concept"
        logging.info(f"[ADD] add-mode closed for concept='{concept}'")
        yield f"Got it. Back to **{concept}** — include it, skip it, or add more.\n\n"

    # ── ➖ Remove a point: picker → confirm → commit (shared machine, Step 4d) ──
    def on_remove_point(self, bullet: str):
        """➖ Remove a point — PARK a sub-bullet PendingAction (challenge); nothing is
        deleted here (F-R1). Sub-bullet removals carry no justification gate at baseline
        (only the 2-of-3 pillar decisions do), so requires_justification stays False."""
        concept = self._current_concept()
        if concept is None or not bullet:
            return
        self.pending = h.PendingAction(
            type="remove_sub_bullet", target=bullet, level="concept",
            pillar=concept, requires_justification=False)
        logging.info(f"[REMOVE POINT] parked: '{bullet}' under '{concept}'")
        yield (
            f"Remove this point from **{concept}**?\n\n- {bullet}\n\n"
            f"*Use the buttons below to confirm.*"
        )

    def on_confirm_remove_point(self):
        """➖ confirm button (D-Q1): resolve via the shared machine -> delete at CONFIRM."""
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="confirm")
        yield from self.render_removal(outcome, pa=pa)

    def on_cancel_remove_point(self):
        """➖ cancel button (D-Q1): abandon -> keep the point. NO delete."""
        pa = self.pending
        if pa is None:
            return
        outcome = h.resolve_pending(self, "", decision="decline")
        yield from self.render_removal(outcome, pa=pa)

    # ── ↩️ Revisit a past pillar ───────────────────────────────────────────
    def on_revisit_pillar(self, pillar: str):
        if not pillar:
            return
        self.revisit_target       = self._normalize_pillar(pillar)
        self.awaiting_revisit_add = True
        logging.info(f"[REVISIT] add-mode opened for past pillar='{self.revisit_target}'")
        yield (
            f"Sure — what point would you like to add to **{self.revisit_target}**? "
            f"Type it, and click **✅ Done adding** when you're finished."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Step 4d — HandlerSession adapter (D-H3). HITL's runtime removal path is
    # BUTTON-driven (on_*_reject / on_*_remove_point -> resolve_pending), so it
    # drives the shared confirmation machine, the swap channel, and the
    # justification gate. The read queries + add mutators complete the neutral
    # surface so BaseAgent can own it at Step 6. HITL's free-text adds NUDGE to a
    # button (D-Q1) and suggestions use _handle_suggest, so those don't route
    # through dispatch here.
    # ══════════════════════════════════════════════════════════════════════

    # ── swap channel (PRESERVED per-arm, §0 #4) ──


    def _extra_swap_signal(self, km, user_text: str) -> bool:
        """6e: walkthrough arms also fire when the CURRENT concept is the swap."""
        cur = self.current_pillar()
        return bool(cur and self._is_wrong_concept(cur))


    def mark_swap_detected(self) -> None:
        """Swap DETECTED on confirm — force_detected + ensure it is excluded from the
        walk. NEVER a delete event (§0); the renderer logs no delete."""
        self.concept_swap.force_detected()
        wrong = self.concept_swap.config["wrong_concept"]
        if wrong not in self.excluded_concepts:
            self.excluded_concepts.append(wrong)

    def before_advance(self, session) -> bool:
        """6f: advance allowed unless the current concept still needs a reason. Every
        shown pillar AND the swap gate on justification (uniform). _justified_concepts
        records concepts justified on accept/reject. INERT until on_approve_concept
        consults this in the accept-flow build (its swap gate is verified there)."""
        cur = self._current_concept()
        if cur is None:
            return True
        if self._is_wrong_concept(cur):
            return cur in self._justified_concepts
        if cur not in self.justification_pillars:
            return True
        return cur in self._justified_concepts

    def requires_justification(self, km) -> bool:
        """6f: every shown pillar requires a reason (D2), and so does the swap — uniform
        treatment so the swap is not singled out by the UI. The swap still resolves as
        DETECTION on confirm, never a delete."""
        name = getattr(km, "pillar", None)
        if not name:
            return False
        if self._is_wrong_concept(name):
            return True
        return name in self.justification_pillars

    # ── read queries ──
    def current_pillar(self):
        """Read-only current concept (skips excluded WITHOUT advancing the cursor)."""
        excluded = [e.lower() for e in self.excluded_concepts]
        idx = self.walkthrough_index
        while idx < len(self.walkthrough_concepts):
            c = self.walkthrough_concepts[idx]
            if c.lower() not in excluded:
                return c
            idx += 1
        return None

    def presented_pillars(self) -> list:
        excluded = [e.lower() for e in self.excluded_concepts]
        return [c for c in self.walkthrough_concepts[:self.walkthrough_index + 1]
                if c.lower() not in excluded]

    def presented_sub_bullets(self) -> dict:
        """{concept -> [non-excluded bullet texts]} from HITL's render sources
        (concept_blocks + user_sub_points), source refs stripped."""
        out = {}
        for name in self.presented_pillars():
            bl = []
            for line in self.concept_blocks.get(name, "").splitlines():
                t = grounding._strip_source_refs(line.strip().lstrip("-• ").strip())
                if t and not matching.is_excluded_bullet(self.excluded_sub_bullets, name, t):
                    bl.append(t)
            for sp in self.user_sub_points.get(name, []):
                t = grounding._strip_source_refs(sp).strip()
                if t and not matching.is_excluded_bullet(self.excluded_sub_bullets, name, t):
                    bl.append(t)
            out[name] = bl
        return out

    def surfaced_pillar_names(self) -> set:
        names = {p["name"].lower() for p in kb.get_shown_pillars()}
        names |= {c.lower() for c in self.walkthrough_concepts}
        names |= {c.lower() for c in self.user_contributed_concepts}
        names |= {e.lower() for e in self.excluded_concepts}
        return names

    # ── add mutators (D-H3 conformance; HITL's runtime adds use the buttons + the
    #    proactive elicited-add path, not these — kept for the shared/BaseAgent surface) ──
    def surface_pillar(self, name: str) -> None:
        if name.lower() in [c.lower() for c in self.walkthrough_concepts]:
            self._last_surface = {"name": name, "is_new": False}
            return
        self.walkthrough_concepts.insert(self.walkthrough_index, name)
        self.user_contributed_concepts.add(name)
        self._last_surface = {"name": name, "is_new": True}

    def add_sub_point(self, pillar: str, text: str) -> None:
        stored, is_new = self._store_sub_point(pillar, text)
        self._last_sub_add = {"pillar": self._normalize_pillar(pillar),
                              "stored": stored, "raw": text, "is_new": is_new}

    # ══════════════════════════════════════════════════════════════════════
    # UI state queries
    # ══════════════════════════════════════════════════════════════════════

    def should_show_buttons(self) -> bool:
        return (
            self.phase == "main"
            and self.walkthrough_active
            and not self.walkthrough_done
            and self._current_concept() is not None
            and self.pending is None
            and not self.awaiting_user_suggestion
            and not self.awaiting_justification
            and not self.awaiting_sub_point
            and not self.awaiting_revisit_add
            and (self._current_concept() not in self.user_contributed_concepts)
        )

    def should_show_confirmation_buttons(self) -> bool:
        # ❌ confirm/cancel buttons for a parked PILLAR removal. A justification-gated
        # pillar awaiting its reason shows NO buttons (the user types the reason); the
        # buttons return once a reason has passed or for a non-gated pillar (D-Q1/D-H2).
        p = self.pending
        return (p is not None and p.type == "remove_pillar"
                and not (p.requires_justification and p.justification is None))

    def should_show_remove_point_confirmation(self) -> bool:
        p = self.pending
        return p is not None and p.type == "remove_sub_bullet"

    def removable_bullets(self) -> list[str]:
        concept = self._current_concept()
        if concept is None:
            return []
        out, seen = [], set()
        for src in (self.concept_blocks.get(concept, "").splitlines(),
                    [f"- {p}" for p in self.user_sub_points.get(concept, [])]):
            for l in src:
                t = l.strip().lstrip("-• ").strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    out.append(t)
        return out

    def past_pillars(self) -> list[str]:
        wrong = self.concept_swap.config["wrong_concept"].lower()
        excluded = [e.lower() for e in self.excluded_concepts]
        current = (self._current_concept() or "").lower()
        out, seen = [], set()
        for c in self.walkthrough_concepts[:self.walkthrough_index]:
            cl = c.lower()
            if cl in excluded or cl == wrong or cl == current or cl in seen:
                continue
            seen.add(cl)
            out.append(c)
        return out

    # ══════════════════════════════════════════════════════════════════════
    # Summary + session
    # ══════════════════════════════════════════════════════════════════════

    def end_session(self) -> None:
        from backend.logger import end_session as _end_session

        if self.approved_concepts:
            final_framework = f"Approved concepts: {', '.join(self.approved_concepts)}"
        else:
            final_framework = "No concepts explicitly approved."

        logging.info(
            f"[END SESSION] approved={self.approved_concepts}, "
            f"rejected={self.excluded_concepts}, "
            f"swap_detected={self.concept_swap.is_detected}"
        )

        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection("sessions").document(self.session_id).update({
                "final_framework":       final_framework,
                "concept_swap_detected": self.concept_swap.is_detected,
                "swap_detected_at_end":  self.concept_swap.is_detected,
                "concepts_approved":     self.approved_concepts,
                "concepts_rejected":     self.excluded_concepts,
            })
            logging.info(f"[END SESSION] Firestore stamped for session={self.session_id}")
        except Exception as e:
            logging.warning(f"[END SESSION] Firestore stamp failed: {e}")

        _end_session(self.session_id)
