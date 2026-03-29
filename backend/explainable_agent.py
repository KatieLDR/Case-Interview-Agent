import json
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
)
from backend import knowledge_graph as kg

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "Profitability"

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

SINGLE_CONCEPT_PROMPT = """
You are a strategic consultant walking a user through a framework ONE concept at a time.

YOUR ONLY JOB RIGHT NOW: Present the single concept named in CONCEPT TO PRESENT NOW.
Nothing else. No other concepts. No full framework. No numbered lists.

EXACT OUTPUT FORMAT — copy this structure precisely:

**[Concept Name]**
Based on consulting best practices, [one sentence, specific to Mining Co. and the
Silica Sand & Bentonite decision — not a generic statement].
- [sub-bucket, 5–7 words, specific to this case]
- [sub-bucket, 5–7 words, specific to this case]
- [sub-bucket, 5–7 words, specific to this case]

*Shall we move on to the next concept?*

─── EXAMPLE (for Volume) ──────────────────────────────────────────────────
**Volume**
Based on consulting best practices, understanding how much ore Mining Co. can
realistically extract and sell determines whether this venture is worth pursuing.
- Annual extraction capacity for Silica Sand
- Estimated market demand from key buyers
- Production constraints from remote site logistics

*Shall we move on to the next concept?*
─────────────────────────────────────────────────────────────────────────────

─── STRICT RULES ────────────────────────────────────────────────────────────
- Output ONLY the concept block above — nothing before, nothing after
- Do NOT write an intro paragraph
- Do NOT list other concepts or buckets
- Do NOT say "here is the framework" or "I recommend"
- Do NOT add numbered lists
- Maximum 3 sub-bullets
- Never mention a knowledge graph, database, or technical system
- Always end with: *Shall we move on to the next concept?*
"""

CONCEPT_QA_PROMPT = """
You are a strategic consultant who just introduced one concept from a framework.
The user has a question about it.

Answer in 2–3 sentences. Stay grounded in the Mining Co. case context.
Plain language only — no jargon, no technical terms.

After answering, end with exactly:
*Shall we move on to the next concept?*
"""

SUMMARY_PROMPT = """
You are a strategic consultant presenting a final framework summary.

FORMAT:
**Full Framework Summary**

**[Bucket 1]**
- sub-bucket (5–7 words)
- sub-bucket (5–7 words)

**[Bucket 2]**
- sub-bucket (5–7 words)
- sub-bucket (5–7 words)

(continue for all buckets)

─── RULES ─────────────────────────────────────────────────────────────────
- Include ONLY concepts listed in CONCEPTS TO INCLUDE below
- No rationale sentences — summary only
- After the summary, ask ONE short follow-up question to invite exploration
"""

SWAP_CAUGHT_PROMPT = """
The user has flagged that the concept you introduced does not belong here.

Respond by:
1. Acknowledging their catch warmly (one sentence)
2. Explaining in plain language why that concept belongs to a different type
   of analysis — no jargon, no technical terms
3. Confirming you are skipping it and moving forward
4. End with: *Shall we move on to the next concept?*

Do NOT re-introduce the wrong concept anywhere after this point.
"""

ADVANCE_CLASSIFIER_PROMPT = """
You are a classifier for a case interview walkthrough tool.

The agent just introduced one concept and asked "Shall we move on to the
next concept?" Determine whether the user is ready to advance OR still has
a question or concern.

Ready to advance:
- Explicit yes ("yes", "sure", "ok", "next", "move on", "let's go")
- Implicit acceptance ("got it", "makes sense", "clear", "understood")
- Short affirmations with no follow-up question

NOT ready to advance:
- Any question about the current concept
- Expressing confusion or disagreement
- Asking to remove or change the current concept

Respond ONLY with valid JSON, no explanation, no markdown:
{"advance": true or false, "confidence": float between 0.0 and 1.0}
"""

ADVANCE_THRESHOLD = 0.80


class ExplainableAgent(BlackBoxAgent):
    """
    Explainable agent — stateful concept-by-concept walkthrough.

    Inherits from BlackBoxAgent:
      - Clarification phase (_stream_clarification, start_main_phase)
      - _detect_override(), _is_answer(), _strip_fences()
      - _strip_concept_swap_from_history()
      - send_message(), end_session()
      - KG infrastructure (_fetch_kg_context, _update_kg_if_framework_mentioned)

    Overrides:
      - __init__            : explainable case + walkthrough state variables
      - get_opening_message : tailored intro
      - _build_system_prompt: minimal fallback (used by inherited send_message)
      - _stream_main        : stateful walkthrough logic
    """

    def __init__(self, user_id: str = "anonymous"):
        # Replicate BlackBoxAgent.__init__ with explainable config
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="explainable")
        self.original_case = get_case("explainable")
        self._pending      = False

        # ── Clarification phase ────────────────────────────────────────────
        self.phase               = "clarification"
        self.clarification_facts = get_clarification_facts("explainable")

        # ── Concept Swap ───────────────────────────────────────────────────
        self.concept_swap = ConceptSwap(
            agent_type="explainable",
            session_id=self.session_id
        )

        # ── KG context ────────────────────────────────────────────────────
        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        print(f"[KG INIT] case_type={CASE_TYPE}, "
              f"framework={self.kg_context['framework']}, "
              f"concepts={self.kg_context['concepts']}")

        self._kg_framework_keywords = {
            "Economic Feasibility":    ["economic feasibility", "market entry", "market potential"],
            "Expanded Profit Formula": ["profit formula", "profitability", "revenue", "cost tree"],
            "Four-Pronged Strategy":   ["four-pronged", "pricing strategy", "price elasticity"],
            "Formulaic Breakdown":     ["formulaic breakdown", "guesstimate", "market sizing"],
            "Customized Issue Trees":  ["issue tree", "unconventional", "internal external"],
        }

        # ── Walkthrough state ──────────────────────────────────────────────
        self.walkthrough_concepts = []    # ordered list built on first framework request
        self.walkthrough_index    = 0     # pointer to current concept
        self.walkthrough_active   = False # True after first concept shown
        self.walkthrough_done     = False # True after summary shown
        self.excluded_concepts    = []    # removed by user override or swap detection

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
                    "Please feel free to ask any questions about the case before you begin."
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
            f"before you begin. When you're ready, I'll walk you through the framework "
            f"one concept at a time — you can ask questions at each step.\n\n"
            f"When you're ready to start, click **\"I'm Ready — Let's Start\"** below."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Walkthrough state helpers
    # ══════════════════════════════════════════════════════════════════════

    def _build_walkthrough_concepts(self) -> list:
        """
        Build ordered concept list for this session.
        Inserts wrong concept at midpoint (mid-way injection).
        """
        base  = list(self.kg_context["concepts"])
        wrong = self.concept_swap.config["wrong_concept"]
        mid   = len(base) // 2
        base.insert(mid, wrong)
        print(f"[WALKTHROUGH] built={base}, swap_at={mid}")
        return base

    def _current_concept(self) -> str | None:
        """
        Return concept at current index, skipping excluded ones.
        Advances index past excluded concepts automatically.
        Returns None when all concepts are done.
        """
        excluded_lower = [e.lower() for e in self.excluded_concepts]
        while self.walkthrough_index < len(self.walkthrough_concepts):
            concept = self.walkthrough_concepts[self.walkthrough_index]
            if concept.lower() not in excluded_lower:
                return concept
            print(f"[WALKTHROUGH] skipping excluded: {concept}")
            self.walkthrough_index += 1
        return None

    def _is_wrong_concept(self, concept: str) -> bool:
        return concept.lower() == self.concept_swap.config["wrong_concept"].lower()

    def _is_ready_to_advance(self, user_input: str) -> bool:
        """Classify whether user is satisfied and ready for next concept."""
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{ADVANCE_CLASSIFIER_PROMPT}\n\nUser reply: \"{user_input}\"",
            )
            parsed = json.loads(self._strip_fences(response.text))
            result = (
                parsed.get("advance", False) and
                parsed.get("confidence", 0.0) >= ADVANCE_THRESHOLD
            )
            print(f"[ADVANCE] advance={parsed.get('advance')}, "
                  f"confidence={parsed.get('confidence'):.2f}, proceed={result}")
            return result
        except Exception as e:
            print(f"[ADVANCE] error: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 1. Swap detection ──────────────────────────────────────────────
        cs_detected = self.concept_swap.check_detection(user_input)
        if cs_detected:
            wrong = self.concept_swap.config["wrong_concept"]
            log_memory_override(
                self.session_id,
                old_context=f"included: {wrong}",
                new_context=f"user rejected: {wrong}",
            )
            if wrong not in self.excluded_concepts:
                self.excluded_concepts.append(wrong)
            # Advance index past the wrong concept
            self.walkthrough_index += 1
            print(f"[SWAP] caught — excluded, index→{self.walkthrough_index}")

        # ── 2. Override detection ──────────────────────────────────────────
        override = self._detect_override(user_input)
        if override:
            log_memory_override(
                self.session_id,
                old_context=f"override_type: {override['type']}",
                new_context=f"detail: {override['detail'] or 'n/a'}",
            )
            print(f"[OVERRIDE] {override['type']} — {override['detail']}")

            if override["type"] == "redo":
                # Full reset
                self.walkthrough_active   = False
                self.walkthrough_done     = False
                self.walkthrough_index    = 0
                self.walkthrough_concepts = []
                self.excluded_concepts    = []
                if self.concept_swap.is_detected:
                    self.history = self._strip_concept_swap_from_history()
                yield "Noted! Let me start the walkthrough fresh...\n\n"
                log_user_message(self.session_id, "[REDO TRIGGERED]")

            elif override["type"] == "concept_excluded" and override.get("detail"):
                excl = override["detail"]
                if excl not in self.excluded_concepts:
                    self.excluded_concepts.append(excl)
                print(f"[OVERRIDE] concept excluded: {excl}")

        # ── 3. Log and append user message ────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 4. Route ───────────────────────────────────────────────────────
        if not self.walkthrough_active:
            # First framework request
            self.walkthrough_concepts = self._build_walkthrough_concepts()
            self.walkthrough_active   = True
            self.walkthrough_index    = 0
            yield from self._stream_concept(is_first=True)

        elif self.walkthrough_done:
            # Post-summary free Q&A
            yield from self._stream_freeform(cs_detected)

        elif cs_detected:
            # Swap caught mid-walkthrough — acknowledge and re-ask
            yield from self._stream_swap_caught()

        elif self._is_ready_to_advance(user_input):
            # Advance to next concept
            self.walkthrough_index += 1
            concept = self._current_concept()
            if concept is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)

        else:
            # User has a question about current concept
            yield from self._stream_concept_qa()

    # ══════════════════════════════════════════════════════════════════════
    # Streaming sub-methods
    # ══════════════════════════════════════════════════════════════════════

    def _stream_concept(self, is_first: bool):
        """Present the concept at current walkthrough_index."""
        concept = self._current_concept()
        if concept is None:
            yield from self._stream_summary()
            return

        is_wrong   = self._is_wrong_concept(concept)
        swap_block = self.concept_swap.get_system_prompt_block() if is_wrong else ""

        instruction = (
            f"{swap_block}"
            f"{SINGLE_CONCEPT_PROMPT}\n\n"
            f"─── CONCEPT TO PRESENT NOW ───────────────────────────────────────────\n"
            f"Present ONLY this concept: **{concept}**\n"
            f"Do not write an intro paragraph. Start directly with **{concept}**.\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )

        # Explicit task injection — overrides the model tendency to summarize
        task_injection = (
            f"[Present only the **{concept}** concept now using the exact format. "
            f"No intro. No other concepts. Start directly with **{concept}**.]"
        )

        prefix = "Here is how I would structure the analysis:\n\n" if is_first else ""

        yield from self._stream_with_instruction(
            instruction=instruction,
            prefix=prefix,
            task_injection=task_injection,
            track_swap=is_wrong,
            store_answer=False,
        )

    def _stream_concept_qa(self):
        """Answer a question about current concept, re-ask to advance."""
        concept = self._current_concept() or "the current concept"
        instruction = (
            f"{CONCEPT_QA_PROMPT}\n\n"
            f"─── CONTEXT ──────────────────────────────────────────────────────────\n"
            f"Current concept: **{concept}**\n"
            f"Case: Mining Co. — Silica Sand & Bentonite profitability analysis\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_swap_caught(self):
        """Acknowledge swap catch, explain, skip, re-ask to advance."""
        wrong = self.concept_swap.config["wrong_concept"]
        instruction = (
            f"{SWAP_CAUGHT_PROMPT}\n\n"
            f"─── CONTEXT ──────────────────────────────────────────────────────────\n"
            f"Wrong concept flagged: **{wrong}**\n"
            f"It belongs to: Four-Pronged Strategy (Pricing analysis)\n"
            f"This case is: Profitability analysis — Expanded Profit Formula\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_summary(self):
        """Present the full framework summary of non-excluded concepts."""
        self.walkthrough_done  = True
        excluded_lower         = [e.lower() for e in self.excluded_concepts]
        active_concepts        = [
            c for c in self.walkthrough_concepts
            if c.lower() not in excluded_lower
        ]
        instruction = (
            f"{SUMMARY_PROMPT}\n\n"
            f"─── CONCEPTS TO INCLUDE (in order) ──────────────────────────────────\n"
            f"{', '.join(active_concepts)}\n"
            f"Case: Mining Co. — Silica Sand & Bentonite profitability analysis\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction, store_answer=True)

    def _stream_freeform(self, cs_detected: bool):
        """Post-summary free Q&A."""
        concepts_str = " → ".join(self.kg_context["concepts"])
        instruction  = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
            f"You are a strategic consultant. The user has seen the full framework. "
            f"Answer their question concisely in plain language — no jargon, no "
            f"mention of technical systems. Ask ONE follow-up question after answering."
        )
        if cs_detected:
            instruction += f"\n\n{SWAP_CAUGHT_PROMPT}"
        yield from self._stream_with_instruction(instruction=instruction)

    # ══════════════════════════════════════════════════════════════════════
    # Core streaming utility
    # ══════════════════════════════════════════════════════════════════════

    def _stream_with_instruction(
        self,
        instruction: str,
        prefix: str = "",
        task_injection: str = "",
        track_swap: bool = False,
        store_answer: bool = False,
    ):
        """
        Stream a Gemini response using the given system instruction.

        task_injection: if provided, appended as a user-turn into a temporary
        copy of history before streaming. This forces the model to follow the
        exact task without relying on the system prompt alone. The injection
        is NOT saved to self.history — it is ephemeral.

        Handles history append, logging, swap tracking, and answer storage.
        """
        self._pending = True
        full_reply    = []

        if prefix:
            yield prefix

        # Build contents — inject task turn if provided (ephemeral, not saved)
        contents = self.history
        if task_injection:
            contents = self.history + [
                types.Content(
                    role="user",
                    parts=[types.Part(text=task_injection)]
                )
            ]

        try:
            for chunk in client.models.generate_content_stream(
                model=MAIN_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)

            # Track swap injection if this is the wrong concept
            if track_swap:
                was_injected = self.concept_swap.is_injected
                self.concept_swap.maybe_inject(reply)
                if self.concept_swap.is_injected and not was_injected:
                    self.concept_swap.log_presented()

            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )

            if store_answer and self._is_answer(reply):
                update_answer(self.session_id, reply)

            log_agent_response(self.session_id, reply)

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

    # ══════════════════════════════════════════════════════════════════════
    # _build_system_prompt — minimal fallback used by inherited send_message
    # ══════════════════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> str:
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"
        return (
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n\n"
            f"You are a strategic consultant. Answer concisely in plain language. "
            f"No jargon. No mention of databases or technical systems."
        )