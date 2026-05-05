import json
import logging
import random
from google.genai import types
from backend.black_box_agent import (
    BlackBoxAgent, CLASSIFIER_MODEL, MAIN_MODEL, client,
    ANSWER_THRESHOLD, OVERRIDE_THRESHOLD,
)
from backend.cases import get_case, get_clarification_facts
from backend.concept_swap import ConceptSwap
from backend.logger import (
    create_session, log_user_message, log_agent_response,
    log_interruption, log_memory_override, update_answer,
    log_framework_switched, log_concept_added, stamp_started_at,
)
from backend import knowledge_graph as kg

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "M&A"

# ── Proactive prompts — rotating fixed list ────────────────────────────────
# 8 variants split across user-first (1-4) and guidance-first (5-8).
# Rotation via prompt_index % 8 — no LLM generation needed.
# Change log: 2026-04-20 — HITL redesign
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
# Neutral, consistent, non-evaluative. Rotates via ack_index % 4.
# No LLM call — avoids hallucination risk and history contamination.
# Change log: 2026-04-22
JUSTIFICATION_ACKS = [
    "Noted — let's continue.",
    "Thanks for sharing that.",
    "Got it.",
    "Understood.",
]

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

HITL_MAIN_SYSTEM_PROMPT = """
You are a strategic thinking partner facilitating a structured M&A framework
walkthrough. You propose concepts one at a time — the candidate decides
whether to include each one. You facilitate, you do not direct.

─── RHETORICAL CONTEXT ──────────────────────────────────────────────────────
Audience : Case interview candidate building an M&A acquisition framework
Genre    : Concept-by-concept facilitated walkthrough with explicit approval
Purpose  : Surface each concept clearly and let the candidate decide
Subject  : M&A acquisition — US foods company acquiring British confectionery firm
Writer   : Strategic thinking partner — facilitator, not expert authority
─────────────────────────────────────────────────────────────────────────────

─── CONCEPT PRESENTATION FORMAT ─────────────────────────────────────────────
Present each concept using this exact format:

**[Concept Name]**
- [sub-bucket, 5–7 words, specific to this case]
- [sub-bucket, 5–7 words, specific to this case]
- [sub-bucket, 5–7 words, specific to this case]

Present concepts clearly and concisely. Let the candidate lead.
Answer questions when asked, but do not volunteer analysis unprompted.
Do NOT preview upcoming concepts.
Do NOT end with a question — buttons handle advancement.
─────────────────────────────────────────────────────────────────────────────

─── WHEN CANDIDATE ASKS A QUESTION ──────────────────────────────────────────
Answer naturally in 2–3 sentences. Stay grounded in the KG context
and case above. Plain language only.
After answering, stop — do not re-present the concept block.
─────────────────────────────────────────────────────────────────────────────

─── WHEN CANDIDATE ADDS A CONCEPT ───────────────────────────────────────────
Acknowledge in one sentence — confirm it will be included in the walkthrough.
Do not present it now — it will appear in sequence.
─────────────────────────────────────────────────────────────────────────────

─── RULES ───────────────────────────────────────────────────────────────────
- Never mention a knowledge graph, database, or technical system
- Never evaluate, score, or tell the candidate they are right or wrong
- Never suggest what the candidate should approve or reject
- One concept at a time — never present two concepts in one response
- Facilitate, do not direct
─────────────────────────────────────────────────────────────────────────────
"""

HITL_SUMMARY_PROMPT = """
You are a strategic consultant presenting a final framework summary.

The FRAMEWORK WALKTHROUGH below contains the exact concepts and sub-bullets
that were presented to the candidate. Use them directly.

FORMAT:
**Final Framework Summary**

**[Concept Name]**
- [sub-bullet from walkthrough]
- [sub-bullet from walkthrough]

(continue for all concepts in CONCEPTS TO INCLUDE)

─── RULES ─────────────────────────────────────────────────────────────────
- Include ONLY concepts listed in CONCEPTS TO INCLUDE
- Copy sub-bullets exactly from FRAMEWORK WALKTHROUGH — do not rewrite them
- No rationale sentences — summary only
- After the summary, ask ONE short follow-up question
- Never mention a knowledge graph, database, or technical system
─────────────────────────────────────────────────────────────────────────────
"""


class HITLAgent(BlackBoxAgent):
    """
    Human-in-the-Loop agent — concept-by-concept walkthrough with
    explicit Approve/Reject buttons per concept.

    Inherits from BlackBoxAgent:
      - _stream_clarification(), _build_clarification_system_prompt()
      - _detect_override(), _is_answer(), _strip_fences()
      - _strip_concept_swap_from_history()
      - send_message()
      - KG infrastructure (_fetch_kg_context, _update_kg_if_framework_mentioned)
      - _stream_with_instruction()
      - _start_main_phase_setup() — shared side effects, not a generator

    Overrides:
      - __init__                         : HITL case + walkthrough state
      - get_opening_message              : mechanic explanation
      - start_main_phase                 : generator — transition + first concept
      - _stream_main                     : justification + suggestion + walkthrough router
      - _stream_concept                  : present concept, store in concept_blocks
      - _stream_concept_qa               : answer question, buttons reattach via app.py
      - _stream_swap_caught              : acknowledge wrong concept, advance
      - _stream_summary                  : uses concept_blocks for reliable summary
      - _build_system_prompt             : KG context + HITL rules
      - _build_clarification_system_prompt: HITL clarification prompt
      - on_approve_concept               : button handler — approve + justification/advance
      - on_reject_concept                : button handler — set pending + pushback
      - on_confirm_reject                : button handler — commit exclusion + justification/advance
      - on_cancel_reject                 : button handler — auto-approve + justification/advance
      - should_show_buttons              : UI state query for app.py
      - should_show_confirmation_buttons : UI state query for app.py
      - get_summary                      : public wrapper for app.py summary streaming
      - end_session                      : stamp HITL-specific Firestore fields

    Architecture notes:
      TEXT ROUTING: User text during main phase routes via inherited
        stream_message() → BlackBoxAgent.stream_message() → self._stream_main().
        Python MRO resolves correctly. Do NOT override stream_message().

      CONCEPT BLOCKS: Sub-bullets stored in self.concept_blocks as streamed.
        Used directly in _stream_summary() — no history reconstruction.
        Python owns state principle — reliable, no LLM hallucination risk.

      CANCEL = AUTO-APPROVE: on_cancel_reject() auto-approves and advances.
        Pushback already served as deliberation. No second approval needed.

      PROACTIVE PROMPT: Shown before every concept from second onwards.
        User can suggest own concept or ask for guidance.
        Justification required at ~50% of steps, randomly assigned.

    Change log: 2026-04-09 — initial build
    Change log: 2026-04-10 — concept_blocks for summary, cancel auto-approve
    Change log: 2026-04-20 — proactive prompts, user-contributed concepts,
                             justification steps, fuzzy duplicate detection
    Change log: 2026-05-05 — removed Q1/Q2; replaced super() with
                             _start_main_phase_setup(); warmup phase added;
                             personalisation rules removed from system prompt
    """

    def __init__(self, user_id: str = "anonymous"):
        # ── Core identity ──────────────────────────────────────────────────
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="hitl")
        self.original_case = get_case("hitl")
        self._pending      = False
        self.turn_count    = 0

        # ── Phase sequence: warmup → clarification → main ──────────────────
        # Change log: 2026-05-01 — warmup phase added before clarification.
        self.phase               = "warmup"
        self.clarification_facts = get_clarification_facts("hitl")

        # ── Concept Swap ───────────────────────────────────────────────────
        self.concept_swap = ConceptSwap(
            agent_type="hitl",
            session_id=self.session_id
        )

        # ── KG context ────────────────────────────────────────────────────
        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        logging.info(
            f"[KG INIT] case_type={CASE_TYPE}, "
            f"framework={self.kg_context['framework']}, "
            f"concepts={self.kg_context['concepts']}"
        )

        self._kg_framework_keywords = {
            "Economic Feasibility":    ["economic feasibility", "market entry", "market potential"],
            "Expanded Profit Formula": ["profit formula", "profitability", "revenue", "cost tree"],
            "Four-Pronged Strategy":   ["four-pronged", "pricing strategy", "price elasticity"],
            "Formulaic Breakdown":     ["formulaic breakdown", "guesstimate", "market sizing"],
            "Customized Issue Trees":  ["issue tree", "unconventional", "internal external"],
            "MA Fit Framework":        ["m&a", "acquisition", "merger", "fit framework"],
        }

        # ── Walkthrough state ──────────────────────────────────────────────
        self.walkthrough_concepts = []
        self.walkthrough_index    = 0
        self.walkthrough_active   = False
        self.walkthrough_done     = False
        self.excluded_concepts    = []
        self.approved_concepts    = []
        self.swap_presented       = False
        self.swap_position        = 0

        # ── Concept blocks — Python owns state ────────────────────────────
        self.concept_blocks = {}  # concept_name → sub-bullet text

        # ── Pending confirmation state ─────────────────────────────────────
        self.pending_excl = None

        # ── Proactive prompt state — HITL redesign 2026-04-20 ─────────────
        self.awaiting_user_suggestion  = False
        self.awaiting_justification    = False
        self.justification_for         = None   # "accept" | "reject"
        self.justification_required    = False
        self.prompt_index              = 0
        self.ack_index                 = 0
        self.user_contributed_concepts = set()

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
            f"time. For each concept, you can **include** it, **skip** it, or ask "
            f"me a question before deciding.\n\n"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Phase transition
    # ══════════════════════════════════════════════════════════════════════

    def start_main_phase(self):
        """
        Override — generator.
        Calls _start_main_phase_setup() for side effects: phase transition,
        history injection, stamp_started_at. Yields transition message then
        streams first concept directly.
        Change log: 2026-05-05 — removed Q1/Q2; replaced super() with
                                  _start_main_phase_setup(); walkthrough starts immediately
        """
        self._start_main_phase_setup()

        self.walkthrough_concepts = self._build_walkthrough_concepts()
        self.walkthrough_active   = True
        self.walkthrough_index    = 0
        self.swap_presented       = False

        yield (
            f"✅ **Clarification round closed — let's begin.**\n\n"
            f"I'll walk you through the framework one concept at a time. "
            f"For each one, use the buttons to **include** or **skip** it, "
            f"or just type a question first.\n\n"
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
        logging.info(
            f"[WALKTHROUGH] built={base}, swap_position={position}, "
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
            "Click **📊 Get Summary & End Session** to see your final framework.\n\n"
        )

    def _get_proactive_prompt(self) -> str:
        prompt = PROACTIVE_PROMPTS[self.prompt_index % len(PROACTIVE_PROMPTS)]
        self.prompt_index += 1
        return prompt

    def _should_require_justification(self) -> bool:
        return random.random() < 0.5

    def _classify_intent(self, user_input: str) -> dict:
        """
        Classify user response to proactive prompt as suggestion or guidance.
        Change log: 2026-04-21
        """
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=(
                    f"You are classifying a user response in a case interview session.\n\n"
                    f"The user was asked: 'What's your suggestion for the next area to "
                    f"explore, or would you like me to guide you?'\n\n"
                    f"User response: \"{user_input}\"\n\n"
                    f"Reply with JSON only, no markdown, no explanation:\n"
                    f"{{\"type\": \"suggestion\" or \"guidance\", "
                    f"\"concept\": \"extracted noun phrase or null\"}}\n\n"
                    f"type=guidance ALWAYS for:\n"
                    f"- Questions: 'anything else?', 'what's next?'\n"
                    f"- Deferrals: 'you decide', 'take a lead', 'take the lead', "
                    f"'continue', 'proceed', 'next step', 'move on', 'go on'\n"
                    f"- Uncertainty: 'I don't know', 'not sure', 'no ideas', "
                    f"'guide me', 'help me', 'no guidance'\n"
                    f"- Affirmations: 'ok', 'sure', 'yes', 'fine'\n"
                    f"- Verb phrases without a clear business noun\n\n"
                    f"type=suggestion ONLY if ALL true:\n"
                    f"- Names a specific business/analytical area\n"
                    f"- Is a noun or noun phrase\n"
                    f"- Could appear in a business case framework\n"
                    f"- Is NOT conversational or a verb phrase\n\n"
                    f"WHEN IN DOUBT: use guidance."
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            logging.info(f"[INTENT] classified: {result}")
            return result
        except Exception as e:
            logging.warning(f"[INTENT] classifier failed: {e} — defaulting to guidance")
            return {"type": "guidance", "concept": None}

    def _check_duplicate(self, concept: str) -> dict:
        """
        Check if user-suggested concept matches anything already in walkthrough.
        Change log: 2026-04-21
        """
        all_concepts = list(self.walkthrough_concepts)
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=(
                    f"You are checking if a user-suggested concept matches any concept "
                    f"in an existing list.\n\n"
                    f"User suggested: \"{concept}\"\n\n"
                    f"Existing concepts: {all_concepts}\n\n"
                    f"Reply with JSON only, no markdown, no explanation:\n"
                    f"{{\"is_duplicate\": true or false, "
                    f"\"matched_concept\": \"exact string from list or null\"}}\n\n"
                    f"is_duplicate=true ONLY if the user suggestion is clearly the same "
                    f"concept as one in the list — same topic, possibly different wording.\n"
                    f"Examples: 'revenue' matches 'Revenue Analysis', "
                    f"'cost' matches 'Cost Structure'.\n"
                    f"Do NOT match concepts that are merely related.\n"
                    f"matched_concept must be the exact string from the list.\n"
                    f"WHEN IN DOUBT: is_duplicate=false."
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            logging.info(f"[DUPLICATE] check result: {result}")
            return result
        except Exception as e:
            logging.warning(f"[DUPLICATE] check failed: {e} — defaulting to not duplicate")
            return {"is_duplicate": False, "matched_concept": None}

    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        """
        Main phase router.
        1. Justification collection (if awaiting_justification)
        2. User suggestion handling (if awaiting_user_suggestion)
        3. Swap detection
        4. Override detection + walkthrough routing

        Change log: 2026-04-09
        Change log: 2026-04-20 — proactive prompt + justification routing
        Change log: 2026-05-05 — removed Q1/Q2 state machine
        """
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 1. Justification collection ────────────────────────────────────
        if self.awaiting_justification:
            log_user_message(self.session_id, f"[JUSTIFICATION:{self.justification_for}] {user_input}")
            logging.info(f"[JUSTIFICATION] collected for={self.justification_for}: '{user_input}'")
            self.awaiting_justification = False
            self.justification_for      = None
            yield from self._stream_justification_ack()
            return

        # ── 2. User suggestion handling ────────────────────────────────────
        if self.awaiting_user_suggestion:
            self.awaiting_user_suggestion = False
            log_user_message(self.session_id, f"[PROACTIVE RESPONSE] {user_input}")

            intent = self._classify_intent(user_input)

            if intent["type"] == "guidance":
                logging.info(f"[PROACTIVE] user chose guidance")
                yield from self._stream_concept(is_first=False)
                return

            concept = intent.get("concept") or user_input.strip()
            dup     = self._check_duplicate(concept)

            if dup["is_duplicate"] and dup["matched_concept"]:
                matched = dup["matched_concept"]
                logging.info(f"[PROACTIVE] duplicate: '{concept}' matches '{matched}'")
                remaining_idx = None
                for i in range(self.walkthrough_index, len(self.walkthrough_concepts)):
                    if self.walkthrough_concepts[i].lower() == matched.lower():
                        remaining_idx = i
                        break
                if remaining_idx is not None:
                    self.walkthrough_concepts.pop(remaining_idx)
                    if remaining_idx < self.swap_position:
                        self.swap_position -= 1
                        logging.info(f"[PROACTIVE] swap_position shifted back to {self.swap_position}")
                yield (
                    f"Good thinking — **{matched}** is already part of the framework. "
                    f"Let's look at it now.\n\n"
                )
                yield from self._stream_user_contributed_concept(matched)
            else:
                logging.info(f"[PROACTIVE] new user concept: '{concept}'")
                yield f"Great suggestion — let's explore **{concept}** now.\n\n"
                yield from self._stream_user_contributed_concept(concept)
            return

        # ── 3. Swap detection ──────────────────────────────────────────────
        cs_detected = False
        if self.swap_presented:
            cs_detected = self.concept_swap.check_detection(user_input)
            if cs_detected:
                wrong = self.concept_swap.config["wrong_concept"]
                log_memory_override(
                    self.session_id,
                    old_context=f"included: {wrong}",
                    new_context=f"user rejected via text: {wrong}",
                )
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)
                self.walkthrough_index += 1
                logging.info(f"[SWAP] caught via text — index→{self.walkthrough_index}")

        # ── 3b. Invariant check ────────────────────────────────────────────
        if (self.walkthrough_active
                and self.walkthrough_index > self.swap_position
                and not self.swap_presented):
            logging.error(f"[INVARIANT] rewinding to swap_position={self.swap_position}")
            self.walkthrough_index = self.swap_position

        # ── 4. Override detection ──────────────────────────────────────────
        just_added_concept = None
        override = None if cs_detected else self._detect_override(user_input)
        if override:
            log_memory_override(
                self.session_id,
                old_context=f"override_type: {override['type']}",
                new_context=f"detail: {override['detail'] or 'n/a'}",
            )
            logging.info(f"[OVERRIDE] {override['type']} — {override['detail']}")

            if override["type"] == "redo":
                self.walkthrough_active        = False
                self.walkthrough_done          = False
                self.walkthrough_index         = 0
                self.walkthrough_concepts      = []
                self.excluded_concepts         = []
                self.approved_concepts         = []
                self.concept_blocks            = {}
                self.swap_presented            = False
                self.swap_position             = 0
                self.pending_excl              = None
                self.awaiting_user_suggestion  = False
                self.awaiting_justification    = False
                self.justification_for         = None
                self.prompt_index              = 0
                self.ack_index                 = 0
                self.user_contributed_concepts = set()
                if self.concept_swap.is_detected:
                    self.history = self._strip_concept_swap_from_history()
                yield "Noted — let me start the walkthrough fresh.\n\n"
                log_user_message(self.session_id, "[REDO TRIGGERED]")

            elif override["type"] == "framework_switch":
                if override.get("detail"):
                    self._update_kg_if_framework_mentioned(override["detail"])
                    log_memory_override(
                        self.session_id,
                        old_context=f"framework: {self.kg_context['framework']}",
                        new_context=f"switch requested: {override['detail']}",
                    )
                yield "Switching framework — let me restart the walkthrough.\n\n"
                log_user_message(self.session_id, "[FRAMEWORK SWITCH TRIGGERED]")

            elif override["type"] == "concept_added" and override.get("detail"):
                new_concept = override["detail"]
                insert_at   = self.walkthrough_index + 1
                self.walkthrough_concepts.insert(insert_at, new_concept)
                log_concept_added(self.session_id, new_concept)
                logging.info(f"[CONCEPT ADDED] '{new_concept}' at index={insert_at}")
                just_added_concept = new_concept

        # ── 5. Log and append user message ────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 6. Routing log ─────────────────────────────────────────────────
        logging.info(
            f"[ROUTE] active={self.walkthrough_active}, done={self.walkthrough_done}, "
            f"swap_presented={self.swap_presented}, index={self.walkthrough_index}, "
            f"cs_detected={cs_detected}"
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

        else:
            yield from self._stream_concept_qa(just_added=just_added_concept)

    # ══════════════════════════════════════════════════════════════════════
    # Proactive prompt + justification streaming
    # ══════════════════════════════════════════════════════════════════════

    def _stream_proactive_prompt(self):
        """
        Show rotating proactive prompt and set awaiting_user_suggestion.
        Called before every concept from second onwards.
        Change log: 2026-04-20
        """
        self.awaiting_user_suggestion = True
        self.justification_required   = self._should_require_justification()
        prompt = self._get_proactive_prompt()
        logging.info(
            f"[PROACTIVE] prompt_index={self.prompt_index - 1}, "
            f"justification_required={self.justification_required}"
        )
        yield prompt

    def _stream_justification_prompt(self, for_decision: str, concept: str = None):
        """
        Show justification prompt after Include/Skip or auto-approve.
        for_decision: "accept" or "reject"
        Change log: 2026-04-20
        """
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
                f"The agent suggested **{concept_name}** because it is part of the standard "
                f"framework for this type of analysis. Given this, what is your rationale "
                f"for excluding it?"
            )

    def _stream_justification_ack(self):
        """
        Brief hardcoded acknowledgement after justification text received.
        No LLM call — avoids hallucination risk.
        Change log: 2026-04-22
        """
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
        Handle a user-contributed concept.
        Inserts into walkthrough_concepts, marks as user-contributed,
        delegates to _stream_concept(), auto-approves after streaming.
        Change log: 2026-04-20
        """
        self.walkthrough_concepts.insert(self.walkthrough_index, concept)
        self.user_contributed_concepts.add(concept)
        if self.walkthrough_index <= self.swap_position:
            self.swap_position += 1
            logging.info(f"[USER CONCEPT] swap_position shifted to {self.swap_position}")
        log_concept_added(self.session_id, concept)
        logging.info(f"[USER CONCEPT] inserted at index={self.walkthrough_index}: '{concept}'")

        yield from self._stream_concept(is_first=False)

        if concept not in self.approved_concepts:
            self.approved_concepts.append(concept)
        logging.info(f"[USER CONCEPT] auto-approved: '{concept}'")

        self.walkthrough_index += 1

        yield "\n\n"
        if self.justification_required:
            self.justification_required = False
            yield from self._stream_justification_prompt("accept", concept=concept)
        else:
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

    # ══════════════════════════════════════════════════════════════════════
    # Streaming sub-methods
    # ══════════════════════════════════════════════════════════════════════

    def _stream_concept(self, is_first: bool):
        concept = self._current_concept()
        if concept is None:
            yield from self._walkthrough_complete_message()
            return

        is_wrong   = self._is_wrong_concept(concept)
        swap_block = self.concept_swap.get_system_prompt_block() if is_wrong else ""

        if is_first:
            prefix = "Here is how I would structure this acquisition analysis:\n\n"
        else:
            prefix = ""
        prefix += f"**{concept}**\n"

        instruction = (
            f"{swap_block}"
            f"{self._build_system_prompt()}\n\n"
            f"─── CONCEPT TO PRESENT NOW ───────────────────────────────────\n"
            f"Concept: **{concept}**\n"
            f"Output ONLY the sub-bullets for this concept.\n"
            f"Do NOT repeat the concept name.\n"
            f"Do NOT add a closing question — buttons handle advancement.\n"
            f"─────────────────────────────────────────────────────────────\n"
        )

        task_injection = (
            f"[Output only the sub-bullets for **{concept}**. "
            f"Do not repeat the concept name. Do not ask a question at the end.]"
        )

        already_logged = self.concept_swap.is_injected if is_wrong else False

        full_concept_reply = []
        for token in self._stream_with_instruction(
            instruction    = instruction,
            prefix         = prefix,
            task_injection = task_injection,
            track_swap     = is_wrong,
            store_answer   = False,
        ):
            full_concept_reply.append(token)
            yield token

        self.concept_blocks[concept] = "".join(full_concept_reply)
        logging.info(f"[CONCEPT BLOCK] stored for '{concept}'")

        if is_wrong:
            self.swap_presented = True
            if not already_logged:
                self.concept_swap.log_presented()
                logging.info(f"[SWAP] concept presented at position={self.swap_position}")

    def _stream_concept_qa(self, just_added: str | None = None):
        concept = self._current_concept() or "the current concept"
        on_swap = (
            self.walkthrough_index == self.swap_position
            and self.swap_presented
            and not self.concept_swap.is_detected
        )

        added_note = ""
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
            f"─── RULES ────────────────────────────────────────────────────\n"
            f"Answer in 2–3 sentences. Plain language only.\n"
            f"Do NOT end with a question — buttons handle advancement.\n"
            f"Do NOT re-present the concept block.\n"
            f"Do NOT suggest whether the candidate should approve or reject.\n"
            f"─────────────────────────────────────────────────────────────\n"
        )

        yield from self._stream_with_instruction(
            instruction = instruction,
            prefix      = added_note,
        )

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

    def _stream_summary(self):
        self.walkthrough_done = True

        wrong = self.concept_swap.config["wrong_concept"].lower()

        if self.concept_swap.is_detected:
            excluded_lower = [e.lower() for e in self.excluded_concepts] + [wrong]
        else:
            excluded_lower = [e.lower() for e in self.excluded_concepts]

        active_concepts = [
            c for c in self.walkthrough_concepts
            if c.lower() not in excluded_lower
        ]

        for c in self.approved_concepts:
            if c not in active_concepts:
                active_concepts.append(c)

        logging.info(f"[SUMMARY] active_concepts={active_concepts}")

        concepts_with_blocks = "\n\n".join(
            f"**{c}**\n{self.concept_blocks.get(c, '').strip()}"
            for c in active_concepts
        )

        instruction = (
            f"{HITL_SUMMARY_PROMPT}\n\n"
            f"─── FRAMEWORK WALKTHROUGH ────────────────────────────────────\n"
            f"{concepts_with_blocks}\n"
            f"─────────────────────────────────────────────────────────────\n\n"
            f"─── CONCEPTS TO INCLUDE (in order) ───────────────────────────\n"
            f"{', '.join(active_concepts)}\n"
            f"Framework: {self.kg_context['framework']} | "
            f"Case: {self.kg_context['case_type']}\n"
            f"─────────────────────────────────────────────────────────────\n"
        )

        yield from self._stream_with_instruction(
            instruction    = instruction,
            task_injection = "Please produce the final framework summary now.",
            store_answer   = True,
        )

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
        """
        KG context + HITL rules.
        Change log: 2026-05-05 — removed candidate context block (Q1/Q2 removed)
        """
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"

        kg_block = (
            f"─── KNOWLEDGE GRAPH CONTEXT ──────────────────────────────────\n"
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
        """
        Include button clicked.
        Captures concept name before incrementing index.
        Change log: 2026-04-20 — justification step added
        """
        concept = self._current_concept()
        if concept is None:
            return

        if concept not in self.approved_concepts:
            self.approved_concepts.append(concept)

        logging.info(f"[APPROVE] concept='{concept}', index={self.walkthrough_index}")
        self.walkthrough_index += 1

        if self.justification_required:
            self.justification_required = False
            yield from self._stream_justification_prompt("accept", concept=concept)
        else:
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

    def on_reject_concept(self):
        """
        Skip button clicked. Sets pending_excl, streams pushback.
        """
        concept = self._current_concept()
        if concept is None:
            return

        self.pending_excl = concept
        logging.info(f"[REJECT] pending set for concept='{concept}'")

        try:
            description = kg.get_concept_description(
                concept_name = concept,
                framework    = self.kg_context["framework"]
            )
            description_line = f" It covers {description}." if description else ""
        except Exception as e:
            logging.warning(f"[REJECT] KG description fetch failed: {e}")
            description_line = ""

        yield (
            f"Are you sure you want to skip **{concept}**?{description_line}\n\n"
            f"*Use the buttons below to confirm.*"
        )

    def on_confirm_reject(self):
        """
        Yes, skip it button clicked. Commits exclusion + justification/advance.
        Change log: 2026-04-20 — justification step added
        """
        concept = self.pending_excl
        if concept is None:
            return

        if self._is_wrong_concept(concept):
            self.concept_swap.force_detected()
            logging.info(f"[SWAP] detected via Reject button — concept='{concept}'")

        if concept not in self.excluded_concepts:
            self.excluded_concepts.append(concept)

        log_memory_override(
            self.session_id,
            old_context=f"concept in framework: {concept}",
            new_context=f"user confirmed rejection via button: {concept}",
        )

        self.pending_excl = None
        logging.info(f"[REJECT CONFIRMED] concept='{concept}'")
        self.walkthrough_index += 1

        yield f"Got it — removing **{concept}** from the framework.\n\n"

        if self.justification_required:
            self.justification_required = False
            yield from self._stream_justification_prompt("reject", concept=concept)
        else:
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

    def on_cancel_reject(self):
        """
        Keep it button clicked. Auto-approves + justification/advance.
        Change log: 2026-04-20 — justification step added
        """
        concept = self.pending_excl
        self.pending_excl = None
        logging.info(f"[REJECT CANCELLED] concept='{concept}' kept in framework")

        if concept and concept not in self.approved_concepts:
            self.approved_concepts.append(concept)

        yield f"Keeping **{concept}** — let's continue.\n\n"
        self.walkthrough_index += 1

        if self.justification_required:
            self.justification_required = False
            yield from self._stream_justification_prompt("accept", concept=concept)
        else:
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

    # ══════════════════════════════════════════════════════════════════════
    # UI state queries
    # ══════════════════════════════════════════════════════════════════════

    def should_show_buttons(self) -> bool:
        """
        True if Include/Skip buttons should show.
        Change log: 2026-04-20 — added proactive/justification/user-contributed guards
        Change log: 2026-05-05 — removed clarification_step check (Q1/Q2 removed)
        """
        return (
            self.phase == "main"
            and self.walkthrough_active
            and not self.walkthrough_done
            and self._current_concept() is not None
            and self.pending_excl is None
            and not self.awaiting_user_suggestion
            and not self.awaiting_justification
            and (self._current_concept() not in self.user_contributed_concepts)
        )

    def should_show_confirmation_buttons(self) -> bool:
        return self.pending_excl is not None

    # ══════════════════════════════════════════════════════════════════════
    # Summary + session
    # ══════════════════════════════════════════════════════════════════════

    def get_summary(self):
        yield from self._stream_summary()

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