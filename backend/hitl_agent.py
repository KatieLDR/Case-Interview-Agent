import json
import logging
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
    stamp_hitl_context,
)
from backend import knowledge_graph as kg

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "M&A"

# ── Clarification step states ──────────────────────────────────────────────
# String states preferred over integers for readability and safe extensibility.
# Change log: 2026-04-09 — senior SE/architect recommendation.
CLARIFICATION_Q1_PENDING = "q1_pending"  # agent has not yet asked Q1
CLARIFICATION_Q2_PENDING = "q2_pending"  # Q1 answered, Q2 not yet asked
CLARIFICATION_OPEN       = "open"        # both answered, open Q&A active

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

HITL_CLARIFICATION_SYSTEM_PROMPT = """
You are a strategic thinking partner facilitating a case interview session.
Before presenting the case, you need to understand the candidate briefly.

─── PHASE A — TWO ORIENTATION QUESTIONS ─────────────────────────────────────
Ask these questions ONE AT A TIME. Wait for each answer before proceeding.

Step 1 — Ask Q1 (Context):
  "Before we dive in — have you worked with M&A frameworks before,
   or is this your first time approaching this type of case?"

Step 2 — Acknowledge Q1 in one sentence. Then ask Q2 (Exigence):
  "Good to know. What's your main goal for today — building out a
   complete acquisition framework, or getting a structured overview
   of the key areas?"

Step 3 — Acknowledge Q2 in one sentence. Summarise both answers
  briefly. Then say the case is ready whenever they are.

─── PHASE B — OPEN CLARIFICATION ────────────────────────────────────────────
After both questions are answered, the candidate may ask clarifying
questions about the case. Answer ONLY from the CASE INFORMATION SHEET
below. If a question is outside the sheet, say:
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

─── PERSONALISATION RULES ───────────────────────────────────────────────────
Read CANDIDATE CONTEXT below and adjust your presentation accordingly.

Based on M&A familiarity:
- First time / no experience → use plain language, add brief context in
  sub-bullets to help the candidate understand what each area covers
- Some experience → standard presentation
- Experienced / familiar → lean, concise sub-bullets — no hand-holding

Based on session goal:
- Complete framework / thorough → 3 sub-bullets per concept
- Quick overview / structured overview → 2 sub-bullets per concept

Apply these rules silently — do not mention them to the candidate.
─────────────────────────────────────────────────────────────────────────────

─── CONCEPT PRESENTATION FORMAT ─────────────────────────────────────────────
Present each concept using this exact format:

**[Concept Name]**
- [sub-bucket, 5–7 words, specific to this case]
- [sub-bucket, 5–7 words, specific to this case]
(- [sub-bucket, 5–7 words] — only if 3 sub-bullets indicated by session goal)

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
      - _stream_with_instruction() — moved to BlackBoxAgent 2026-04-09

    Overrides:
      - __init__                         : HITL case + walkthrough + clarification step state
      - get_opening_message              : mechanic explanation
      - start_main_phase                 : generator — asks Q1, initialises walkthrough
      - _stream_main                     : Q1/Q2 state machine + walkthrough router
      - _stream_concept                  : present concept, store in concept_blocks
      - _stream_concept_qa               : answer question, buttons reattach via app.py
      - _stream_swap_caught              : acknowledge wrong concept, advance
      - _stream_summary                  : uses concept_blocks for reliable summary
      - _build_system_prompt             : KG + candidate context + HITL rules
      - _build_clarification_system_prompt: HITL clarification prompt
      - on_approve_concept               : button handler — approve + advance
      - on_reject_concept                : button handler — set pending + pushback
      - on_confirm_reject                : button handler — commit exclusion + advance
      - on_cancel_reject                 : button handler — auto-approve + advance
      - should_show_buttons              : UI state query for app.py
      - should_show_confirmation_buttons : UI state query for app.py
      - get_summary                      : public wrapper for app.py summary streaming
      - end_session                      : stamp HITL-specific Firestore fields

    Architecture notes:
      TEXT ROUTING: User text during main phase routes via inherited
        stream_message() → BlackBoxAgent.stream_message() → self._stream_main().
        Python MRO resolves correctly. Do NOT override stream_message().

      Q1/Q2 NOT IN HISTORY: Q1/Q2 answers are config, not conversation.
        Stored in self.hitl_context/hitl_exigence, injected into system
        prompt at runtime. NOT appended to self.history.

      CONCEPT BLOCKS: Sub-bullets stored in self.concept_blocks as streamed.
        Used directly in _stream_summary() — no history reconstruction.
        Python owns state principle — reliable, no LLM hallucination risk.

      CANCEL = AUTO-APPROVE: on_cancel_reject() auto-approves and advances.
        Pushback already served as deliberation. No second approval needed.

    Change log: 2026-04-09 — initial build
    Change log: 2026-04-10 — concept_blocks for summary, cancel auto-approve,
                             fixed duplicate on_confirm_reject, added on_reject_concept,
                             removed all auto-summary calls from button handlers
    """

    def __init__(self, user_id: str = "anonymous"):
        # ── Core identity ──────────────────────────────────────────────────
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="hitl")
        self.original_case = get_case("hitl")
        self._pending      = False

        # ── Clarification phase ────────────────────────────────────────────
        self.phase               = "clarification"
        self.clarification_facts = get_clarification_facts("hitl")

        # ── Proactive clarification step state ─────────────────────────────
        # "q1_pending" → "q2_pending" → "open"
        self.clarification_step = CLARIFICATION_Q1_PENDING

        # ── Candidate context ──────────────────────────────────────────────
        # Config only — NOT stored in history.
        self.hitl_context  = None  # Q1 answer — M&A familiarity
        self.hitl_exigence = None  # Q2 answer — session goal

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
        self.approved_concepts    = []  # HITL-specific — explicit approvals
        self.swap_presented       = False
        self.swap_position        = 0

        # ── Concept blocks — Python owns state ────────────────────────────
        # Stores sub-bullets exactly as streamed per concept.
        # Used by _stream_summary() — no LLM history reconstruction.
        # Change log: 2026-04-10
        self.concept_blocks = {}  # concept_name → sub-bullet text

        # ── Pending confirmation state ─────────────────────────────────────
        self.pending_excl = None

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
                    "I will ask you two quick questions before we begin."
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
            f"When you're ready to start, click **\"I'm Ready — Let's Start\"** below."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Phase transition
    # ══════════════════════════════════════════════════════════════════════

    def start_main_phase(self):
        """
        Override — generator.
        Calls super() for side effects: phase, history injection, stamp_started_at.
        Yields Q1. Q2 and walkthrough handled in _stream_main().
        Change log: 2026-04-09
        """
        super().start_main_phase()
        self.clarification_step = CLARIFICATION_Q1_PENDING
        yield (
            f"✅ **Clarification round closed.**\n\n"
            f"Before we begin, I have two quick questions to help me tailor "
            f"this session for you.\n\n"
            f"---\n\n"
            f"**Have you worked with M&A frameworks before, or is this your "
            f"first time approaching this type of case?**"
        )

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
        """Shared helper — yields walkthrough complete message."""
        self.walkthrough_done = True
        yield (
            "✅ We've covered all the concepts. "
            "Click **📊 Get Summary & End Session** to see your final framework.\n\n"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        """
        Main phase router.
        1. Q1/Q2 proactive clarification
        2. Swap detection
        3. Override detection + walkthrough routing

        Change log: 2026-04-09
        Change log: 2026-04-10 — Q1/Q2 not appended to history
        """
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 0. Proactive clarification ─────────────────────────────────────
        if self.clarification_step == CLARIFICATION_Q1_PENDING:
            self.hitl_context = user_input.strip()
            log_user_message(self.session_id, f"[Q1 CONTEXT] {user_input}")
            logging.info(f"[HITL] Q1 answer stored: '{self.hitl_context}'")
            self.clarification_step = CLARIFICATION_Q2_PENDING
            yield (
                f"Got it — good to know.\n\n"
                f"**What's your main goal for today — building a complete "
                f"acquisition framework, or getting a quick structured overview?**"
            )
            return

        if self.clarification_step == CLARIFICATION_Q2_PENDING:
            self.hitl_exigence = user_input.strip()
            log_user_message(self.session_id, f"[Q2 EXIGENCE] {user_input}")
            logging.info(f"[HITL] Q2 answer stored: '{self.hitl_exigence}'")
            self.clarification_step = CLARIFICATION_OPEN
            stamp_hitl_context(
                session_id    = self.session_id,
                hitl_context  = self.hitl_context,
                hitl_exigence = self.hitl_exigence,
            )
            yield (
                f"Perfect — let's get started.\n\n"
                f"Here is how I would structure the analysis. "
                f"For each concept, use the buttons to include or skip it, "
                f"or just ask me a question first.\n\n"
                f"---\n\n"
            )
            self.walkthrough_concepts = self._build_walkthrough_concepts()
            self.walkthrough_active   = True
            self.walkthrough_index    = 0
            self.swap_presented       = False
            yield from self._stream_concept(is_first=True)
            return

        # ── 1. Swap detection ──────────────────────────────────────────────
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

        # ── 1b. Invariant check ────────────────────────────────────────────
        if (self.walkthrough_active
                and self.walkthrough_index > self.swap_position
                and not self.swap_presented):
            logging.error(f"[INVARIANT] rewinding to swap_position={self.swap_position}")
            self.walkthrough_index = self.swap_position

        # ── 2. Override detection ──────────────────────────────────────────
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
                self.walkthrough_active   = False
                self.walkthrough_done     = False
                self.walkthrough_index    = 0
                self.walkthrough_concepts = []
                self.excluded_concepts    = []
                self.approved_concepts    = []
                self.concept_blocks       = {}
                self.swap_presented       = False
                self.swap_position        = 0
                self.pending_excl         = None
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

        # ── 3. Log and append user message ────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 4. Routing log ─────────────────────────────────────────────────
        logging.info(
            f"[ROUTE] active={self.walkthrough_active}, done={self.walkthrough_done}, "
            f"swap_presented={self.swap_presented}, index={self.walkthrough_index}, "
            f"cs_detected={cs_detected}"
        )

        # ── 5. Route ───────────────────────────────────────────────────────
        if not self.walkthrough_active:
            self.walkthrough_concepts = self._build_walkthrough_concepts()
            self.walkthrough_active   = True
            self.walkthrough_index    = 0
            self.swap_presented       = False
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
                yield from self._stream_concept(is_first=False)

        else:
            yield from self._stream_concept_qa(just_added=just_added_concept)

    # ══════════════════════════════════════════════════════════════════════
    # Streaming sub-methods
    # ══════════════════════════════════════════════════════════════════════

    def _stream_concept(self, is_first: bool):
        """
        Present concept at current walkthrough_index.
        Stores streamed sub-bullets in concept_blocks for summary use.
        Buttons attached by app.py via should_show_buttons().

        Change log: 2026-04-09
        Change log: 2026-04-10 — collect tokens into concept_blocks.
                                  removed auto-summary, uses _walkthrough_complete_message.
        """
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

        # ── Stream and collect tokens ──────────────────────────────────────
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

       # ── Store sub-bullets for summary — Python owns state ─────────────
        # Store for all concepts including wrong concept.
        # If wrong concept is detected, _stream_summary() excludes it via active_concepts.
        # If not detected (passive approval), block is needed for summary.
        self.concept_blocks[concept] = "".join(full_concept_reply)
        logging.info(f"[CONCEPT BLOCK] stored for '{concept}'")

        # ── Swap tracking ──────────────────────────────────────────────────
        if is_wrong:
            self.swap_presented = True
            if not already_logged:
                self.concept_swap.log_presented()
                logging.info(f"[SWAP] concept presented at position={self.swap_position}")

    def _stream_concept_qa(self, just_added: str | None = None):
        """
        Answer a question about the current concept.
        Buttons reattach via app.py should_show_buttons() after streaming.
        Change log: 2026-04-09
        """
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
            logging.info(f"[QA] concept_added acknowledgement for '{just_added}'")

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
        """
        Acknowledge user caught wrong concept via text.
        Caller handles advancement.
        Change log: 2026-04-09
        """
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
        """
        Generate final framework summary using concept_blocks.
        concept_blocks contains exactly what was streamed per concept —
        no history reconstruction, no LLM hallucination risk.
        Only called via get_summary() — never auto-called.

        Change log: 2026-04-09
        Change log: 2026-04-10 — uses concept_blocks, standalone HITL_SUMMARY_PROMPT
        """
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

        logging.info(f"[SUMMARY] active_concepts={active_concepts}")

        # Build framework directly from stored blocks — Python owns state
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
        """
        Post-walkthrough freeform.
        Change log: 2026-04-09
        """
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
        KG context + candidate context + HITL rules.
        Graceful fallback if Q1/Q2 not yet collected.
        Change log: 2026-04-09
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

        context_block = ""
        if self.hitl_context or self.hitl_exigence:
            context_block = (
                f"─── CANDIDATE CONTEXT ────────────────────────────────────────\n"
                f"M&A familiarity : {self.hitl_context or 'not specified'}\n"
                f"Session goal    : {self.hitl_exigence or 'not specified'}\n"
                f"──────────────────────────────────────────────────────────────\n\n"
            )

        return kg_block + swap_block + context_block + HITL_MAIN_SYSTEM_PROMPT

    def _build_clarification_system_prompt(self) -> str:
        """
        HITL clarification prompt with facts sheet.
        Change log: 2026-04-09
        """
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
        Logs approval, advances, streams next concept.
        Does NOT auto-stream summary.
        Change log: 2026-04-09 / 2026-04-10
        """
        concept = self._current_concept()
        if concept is None:
            return

        if concept not in self.approved_concepts:
            self.approved_concepts.append(concept)

        logging.info(f"[APPROVE] concept='{concept}', index={self.walkthrough_index}")

        self.walkthrough_index += 1
        next_concept = self._current_concept()

        if next_concept is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_concept(is_first=False)

    def on_reject_concept(self):
        """
        Skip button clicked.
        Sets pending_excl, streams KG-grounded pushback.
        No exclusion yet — confirmation required.
        Change log: 2026-04-09
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
        Yes, skip it button clicked.
        Commits exclusion, advances. Does NOT auto-stream summary.
        Change log: 2026-04-09 / 2026-04-10
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
        next_concept = self._current_concept()

        yield f"Got it — removing **{concept}** from the framework.\n\n"

        if next_concept is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_concept(is_first=False)

    def on_cancel_reject(self):
        """
        Keep it button clicked.
        Auto-approves concept and advances — no second approval click needed.
        Pushback already served as the deliberation moment.
        Change log: 2026-04-09
        Change log: 2026-04-10 — auto-approve and advance instead of
                                  returning to decision point.
        """
        concept = self.pending_excl
        self.pending_excl = None
        logging.info(f"[REJECT CANCELLED] concept='{concept}' kept in framework")

        # Auto-approve — user chose to keep after deliberation
        if concept and concept not in self.approved_concepts:
            self.approved_concepts.append(concept)

        yield f"Keeping **{concept}** — let's continue.\n\n"

        self.walkthrough_index += 1
        next_concept = self._current_concept()

        if next_concept is None:
            yield from self._walkthrough_complete_message()
        else:
            yield from self._stream_concept(is_first=False)

    # ══════════════════════════════════════════════════════════════════════
    # UI state queries
    # ══════════════════════════════════════════════════════════════════════

    def should_show_buttons(self) -> bool:
        """
        True if Include/Skip buttons should show.
        False when pending (confirmation buttons show instead) or done.
        Change log: 2026-04-09
        """
        return (
            self.phase == "main"
            and self.walkthrough_active
            and not self.walkthrough_done
            and self._current_concept() is not None
            and self.clarification_step == CLARIFICATION_OPEN
            and self.pending_excl is None
        )

    def should_show_confirmation_buttons(self) -> bool:
        """
        True if Keep it / Yes skip it buttons should show.
        Change log: 2026-04-09
        """
        return self.pending_excl is not None

    # ══════════════════════════════════════════════════════════════════════
    # Summary + session
    # ══════════════════════════════════════════════════════════════════════

    def get_summary(self):
        """
        Public wrapper for app.py summary button.
        Uses _stream_summary() with concept_blocks — not send_message().
        Change log: 2026-04-09
        """
        yield from self._stream_summary()

    def end_session(self) -> None:
        """
        Stamp HITL-specific Firestore fields.
        Final framework from approved_concepts — not history scan.
        Change log: 2026-04-09
        """
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