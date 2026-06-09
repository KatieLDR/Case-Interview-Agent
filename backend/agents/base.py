"""backend/base.py  â  Step 6a of REFACTOR_PLAN.md (F-ARCH2: BaseAgent extraction).

Resolves F-ARCH2: BlackBox was both a concrete agent AND the de-facto base that
ExplainableAgent / HITLAgent inherited from. BaseAgent is the shared turn engine;
BlackBoxAgent, ExplainableAgent and HITLAgent become SIBLINGS of it (6a wires
BlackBox; 6b reparents EXP/HITL). Method partition computed from the live override
+ call matrix (REFACTOR_PLAN Â§S Step-6 block), not guessed.

CONCRETE here  = the shared engine (turn entry, streaming, history, warm-up, shared
                 classifiers) â inherited unchanged by every arm.
SEAM (abstract)= persona / matching / grounding / swap hooks each arm implements.
                 A pure BaseAgent is never instantiated; the stubs fail loud.
CONCRETE-DEFAULT = _is_excluded_bullet / _build_clarification_system_prompt /
                 _swap_question_signal: a base method calls them, so base carries
                 BlackBox's body as the default; the overriding arm (EXP or HITL)
                 still wins polymorphically.
Constants (client/models/thresholds/classify_json/strip_fences) come from llm.py
(Step 1); BaseAgent imports them, never redefines them.
"""

import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from backend.logger import (
    create_session, end_session, stamp_started_at,
    log_user_message, log_agent_response,
    log_interruption,
    update_answer, log_warmup_response,
)
from backend.logging import events as ev
from backend.logging.sink import firestore_sink as _sink
from backend.knowledge.cases import get_case, get_clarification_facts
from backend.tools.concept_swap import ConceptSwap
from backend.knowledge import knowledge_base as kb

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
from backend.llm import (
    client, MAIN_MODEL, CLASSIFIER_MODEL, classify_json, strip_fences,
    ANSWER_THRESHOLD, ADD_MATCH_THRESHOLD, CONCEPT_MATCH_THRESHOLD,
)
from backend.domain import matching   # Step 2: shared KB matchers (locate / passes)
from backend.interaction import intents  # Step 3: unified intent taxonomy (I-2)
from backend.interaction import handlers  # Step 4: shared handlers + PendingAction (I-1)
from backend.domain import grounding     # Step 2: shared KB grounding (suggest render)

# ââ Shared module-level prompts/constants (moved from black_box_agent.py, Step 6a) ââ
CLARIFICATION_SYSTEM_PROMPT = """
You are a BCG case interviewer conducting the clarification round before the
candidate begins their structured analysis.

You have a fixed information sheet for this case. Your job is to answer the
candidate's questions based strictly on that sheet.

─── RULES ─────────────────────────────────────────────────────────────────
- Answer ONLY from the facts provided in the CASE INFORMATION SHEET below
- If the question is closely related to a fact on the sheet, infer naturally
  from it — but do not introduce new information that is not on the sheet
- If the question is outside the scope of the sheet, respond with:
  "I'm afraid I don't have that information for this case."
- Keep answers concise and professional — one to three sentences per answer
- Never reveal the framework or hint at the structure the candidate should use
- Never evaluate or coach the candidate during this phase
- Do not ask questions back to the candidate
"""

ANSWER_CLASSIFIER_PROMPT = """
You are a classifier for a case interview tool.

Determine whether the agent response contains a structured framework answer
with clear primary buckets and sub-buckets.

Short replies, clarifications, questions, or discussion do NOT qualify.

Respond ONLY with valid JSON, no explanation, no markdown:
{"is_answer": true or false, "confidence": float between 0.0 and 1.0}
"""

WARMUP_PROMPT = (
    "✏️ **Here is the practice task exercise:**\n\n"
    "You are moving to a new city for a new job opportunity. "
    "Before you go, you want to make sure you have a complete plan. "
    "Let's plan this together.\n\n"
    "*This should take about 2–3 minutes, there are no right or wrong answers.*\n\n"
    "**Here's my suggestion:**\n\n"
    "🏠 **Housing**\n"
    "- Should we find temporary accommodation?\n"
    "- How are the neighbourhoods?\n\n"
    "📋 **Admin**\n"
    "- Should we register at the new city hall?\n"
    "- Do we need a local bank account?\n\n"
    "*What else would you add, or is there anything you'd remove or change?*"
)

WARMUP_MERGE_PROMPT = """You are helping a user build a moving-to-a-new-city plan.

The starting plan is:
🏠 Housing
- Should we find temporary accommodation?
- How are the neighbourhoods?

📋 Admin
- Should we register at the new city hall?
- Do we need a local bank account?

The user has added the following ideas:
{additions}

Your task: produce an updated plan that incorporates the user's ideas.
- Keep the same emoji + bold header format
- Add new bullet points under the most relevant existing section, or create a new section if needed
- If the user pushed back on something, remove or reframe it
- If the user adds a new section (e.g. "Location"), generate 2 relevant sub-bullet questions for it
- If the user's intent is ambiguous, make a reasonable interpretation and proceed
- Keep it concise — bullet points only, no extra explanation
- Do NOT add any intro or closing sentence — return the plan only"""

MAX_TURNS_PER_SESSION = 50


class BaseAgent:
    """Shared turn engine for all three study arms. See module docstring."""

    # ââ SEAM declarations (each arm implements; base never instantiated) ââ
    def _build_system_prompt(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_system_prompt (BaseAgent seam)")

    def _stream_main(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _stream_main (BaseAgent seam)")

    def _stream_summary(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _stream_summary (BaseAgent seam)")

    def _format_sub_bullet(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _format_sub_bullet (BaseAgent seam)")

    def _match_key_question(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _match_key_question (BaseAgent seam)")

    def _match_pillar(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _match_pillar (BaseAgent seam)")

    def add_sub_point(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement add_sub_point (BaseAgent seam)")

    def begin_analysis(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement begin_analysis (BaseAgent seam)")

    def current_pillar(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement current_pillar (BaseAgent seam)")

    def end_session(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement end_session (BaseAgent seam)")

    def get_opening_message(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_opening_message (BaseAgent seam)")

    def get_pre_analysis_instruction(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_pre_analysis_instruction (BaseAgent seam)")

    def get_summary(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_summary (BaseAgent seam)")

    def is_swap_target(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement is_swap_target (BaseAgent seam)")

    def mark_swap_detected(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement mark_swap_detected (BaseAgent seam)")

    def presented_pillars(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement presented_pillars (BaseAgent seam)")

    def presented_sub_bullets(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement presented_sub_bullets (BaseAgent seam)")

    def requires_justification(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement requires_justification (BaseAgent seam)")

    def surface_pillar(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement surface_pillar (BaseAgent seam)")

    def surfaced_pillar_names(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement surfaced_pillar_names (BaseAgent seam)")

    def swap_name(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement swap_name (BaseAgent seam)")

    # ââ CONCRETE shared engine (+ 3 concrete-default seams) ââ
    def _fetch_kg_context(self, case_type: str) -> dict:
        """
        Load framework context from JSON knowledge base.
        Change log: 2026-05-28 — migrated from KG to JSON knowledge base.
        """
        from backend.knowledge import knowledge_base as kb
        framework = kb.get_framework_name()
        concepts  = [p["name"] for p in kb.get_shown_pillars()]
        return {"case_type": case_type, "framework": framework, "concepts": concepts}

    def _build_clarification_system_prompt(self) -> str:
        if self.clarification_facts:
            facts_lines = "\n".join(
                f"- {topic.upper()}: {answer}"
                for topic, answer in self.clarification_facts.items()
            )
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────────────\n"
                f"{facts_lines}\n"
                f"──────────────────────────────────────────────────────────────────────\n\n"
            )
        else:
            facts_block = (
                f"─── CASE INFORMATION SHEET ───────────────────────────────────────────\n"
                f"No additional facts are available for this case.\n"
                f"Deflect all clarification questions with: "
                f"\"I'm afraid I don't have that information for this case.\"\n"
                f"──────────────────────────────────────────────────────────────────────\n\n"
            )
        return facts_block + CLARIFICATION_SYSTEM_PROMPT

    def _build_tree_overview(self) -> str:
        from backend.knowledge import knowledge_base as kb
        shown = kb.get_shown_pillars()
        lines = ["**Framework Overview**\n"]
        for pillar in shown:
            lines.append(f"- {pillar['name']}")
        return "\n".join(lines)

    def show_tree(self) -> str:
        return self._build_tree_overview()

    def get_warmup_message(self) -> str:
        return WARMUP_PROMPT

    def merge_warmup_additions(self, additions: list[str]) -> str:
        """
        LLM call: merges user additions into the pre-built warmup plan.
        Fallback: original plan + user additions listed separately.
        Change log: 2026-05-22 — added for warmup redesign
        """
        if not additions:
            return WARMUP_PROMPT

        additions_text = "\n".join(f"- {a}" for a in additions)
        prompt = WARMUP_MERGE_PROMPT.format(additions=additions_text)

        try:
            response = client.models.generate_content(
                model=MAIN_MODEL,
                contents=prompt,
            )
            merged = response.text.strip()
            return (
                "**Here's your updated plan:**\n\n"
                + merged
            )
        except Exception as e:
            print(f"[WARMUP MERGE] LLM merge failed: {e}")
            additions_block = "\n".join(f"- {a}" for a in additions)
            return (
                "**Here's your updated plan:**\n\n"
                "🏠 **Housing**\n"
                "- Should we find temporary accommodation?\n"
                "- How are the neighbourhoods?\n\n"
                "📋 **Admin**\n"
                "- Should we register at the new city hall?\n"
                "- Do we need a local bank account?\n\n"
                "**Your additions:**\n"
                + additions_block
            )

    def _start_main_phase_setup(self):
        if self.phase == "main":
            return

        self.phase = "main"
        stamp_started_at(self.session_id)

        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=(
                    "[SYSTEM: The clarification round has ended. "
                    "The candidate is now ready to begin their structured analysis. "
                    "Switch to reference consultant mode and wait for the candidate "
                    "to present their framework or ask for one.]"
                ))]
            )
        )
        self.history.append(
            types.Content(
                role="model",
                parts=[types.Part(text=(
                    "Understood. The clarification round is now closed. "
                    "I'm ready for your structured analysis."
                ))]
            )
        )

        print(f"[PHASE] clarification → main for session={self.session_id}")

    def stream_message(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        if self.phase == "clarification":
            yield from self._stream_clarification(user_input)
        else:
            self.turn_count += 1
            if self.turn_count > MAX_TURNS_PER_SESSION:
                return
            yield from self._stream_main(user_input)

    def _stream_clarification(self, user_input: str):
        log_user_message(self.session_id, f"[CLARIFICATION] {user_input}")
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        self._pending = True
        full_reply = []
        try:
            for chunk in client.models.generate_content_stream(
                model=MAIN_MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_clarification_system_prompt(),
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )
            log_agent_response(self.session_id, f"[CLARIFICATION] {reply}")

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

    def _stream_framework_presentation(self):
        """Static first render of the full framework — no LLM. Change log: 2026-06-01"""
        reply = self._render_full_framework(is_first=True)
        self.history.append(types.Content(role="user",
            parts=[types.Part(text="Please present the full structured framework for this case.")]))
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        self.concept_swap.maybe_inject(reply)
        self.concept_swap.log_presented()
        update_answer(self.session_id, reply)
        log_agent_response(self.session_id, reply)
        yield reply

    def _last_agent_text(self) -> str:
        """Last model message (router context). BlackBox has no walkthrough cursor."""
        for c in reversed(self.history):
            if c.role == "model" and c.parts:
                return (c.parts[0].text or "")[:500]
        return ""

    def _render_full_framework(self, is_first: bool = False, closing: bool = True) -> str:
        from backend.knowledge import knowledge_base as kb
        excluded  = [e.lower() for e in self.excluded_concepts]
        shown     = [p for p in kb.get_shown_pillars() if p["name"].lower() not in excluded]
        swap      = kb.get_swap_concept()
        wrong     = self.concept_swap.config["wrong_concept"]
        swap_bul  = swap.get("sub_bullets", []) if swap else []
        show_swap = not self.concept_swap.is_detected
        position  = len(shown) // 2

        lines = ["💡 When you're finished, click ‼️End Session to close your session. Note: this cannot be undone. \n\n Here is how I would structure the analysis:\n"] if is_first else []

        def emit(name, kb_bullets):
            lines.append(f"**{name}**")
            for b in kb_bullets:
                if not self._is_excluded_bullet(name, b):
                    lines.append(f"- {_strip_source_refs(b)}")
            for sp in self.user_sub_points.get(name, []):
                if not self._is_excluded_bullet(name, sp):
                    lines.append(f"- {sp}")
            lines.append("")

        for i, p in enumerate(shown):
            if show_swap and i == position:
                lines.append(f"**{wrong}**")
                for b in swap_bul:
                    if not self._is_excluded_bullet(wrong, b):
                        lines.append(f"- {_strip_source_refs(b)}")
                lines.append("")
            emit(p["name"], p.get("sub_bullets", []))
        if show_swap and position >= len(shown):
            lines.append(f"**{wrong}**")
            for b in swap_bul:
                if not self._is_excluded_bullet(wrong, b):
                    lines.append(f"- {_strip_source_refs(b)}")
            lines.append("")
        for name in self.user_added_pillars:
            if name.lower() in excluded:
                continue
            kbp = next((p for p in kb.get_all_pillars() if p["name"].lower() == name.lower()), None)
            emit(name, kbp.get("sub_bullets", []) if kbp else [])

        if closing:
            lines.append("*Feel free to add, remove, or question any part of this — let's build this together.*")
        return "\n".join(lines).rstrip() + "\n"

    def _yield_rerender(self, preamble: str = ""):
        reply = preamble + self._render_full_framework(is_first=False)
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        update_answer(self.session_id, reply)
        log_agent_response(self.session_id, reply)
        yield reply

    def _is_excluded_bullet(self, pillar_name: str, bullet: str) -> bool:
        """→ shared pure predicate (Step 2); session excluded-map passed by value."""
        return matching.is_excluded_bullet(self.excluded_sub_bullets, pillar_name, bullet)

    def _stream_qa(self, user_input: str):
        framework    = self.kg_context["framework"]
        concepts_str = ", ".join(self.kg_context["concepts"]) or \
                       "Strategic Fit, Solution Design & Scope, Feasibility"
        instruction = (
            "You are a strategic consultant answering a question about a framework "
            "you have already presented.\n\n"
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Framework : {framework}\nPillars   : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
            f"{self.concept_swap.get_system_prompt_block()}"
            "─── ANSWER RULES ─────────────────────────────────────────────────────\n"
            "Answer the question in 2–3 sentences, plain language, grounded in the case.\n"
            "Do NOT reprint, restate, or regenerate the framework or any pillar block.\n"
            "Do NOT claim to add, remove, or change anything — answer only.\n"
            "Ask at most one short follow-up question.\n"
            "──────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_confirm_qa(self, user_input: str, concept: str):
        """Answer a question raised AT the removal-confirmation prompt, then re-offer the
        pending decision in ONE streamed message. The shared PendingAction (self.pending)
        stays parked, so the user's next yes/no still resolves the removal via
        handlers.resolve_pending. Change log: 2026-06-02; Step 4b: shared pending machine."""
        framework    = self.kg_context["framework"]
        concepts_str = ", ".join(self.kg_context["concepts"]) or \
                    "Strategic Fit, Solution Design & Scope, Feasibility"
        instruction = (
            "You are a strategic consultant. The user was asked to confirm removing "
            f"**{concept}** from the framework, and instead of answering yes or no they "
            "asked a question or made a comment.\n\n"
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Framework : {framework}\nPillars   : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
            f"{self.concept_swap.get_system_prompt_block()}"
            "─── ANSWER RULES ─────────────────────────────────────────────────────\n"
            "Answer their question in 2–3 sentences, plain language, grounded in the case.\n"
            "Do NOT reprint, restate, or regenerate the framework or any pillar block.\n"
            "Do NOT claim to add, remove, or change anything — nothing has changed yet.\n"
            "Do NOT ask any other follow-up question.\n"
            "End your reply with EXACTLY this sentence on a new line:\n"
            f"Still want to remove **{concept}**? Reply **yes** to remove it, or **no** to keep it.\n"
            "──────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _ack_no_reprint(self):
        acks = ["Sure — I'm here if you'd like to revisit anything.",
                "Understood — let me know if anything comes to mind.",
                "Noted — happy to adjust if you think of something."]
        ack = acks[self._ack_index % len(acks)]
        self._ack_index += 1
        self.history.append(types.Content(role="model", parts=[types.Part(text=ack)]))
        log_agent_response(self.session_id, ack)
        yield ack

    def _reply_is_question(self, text: str) -> bool:
        """At a removal-confirm gate, an 'other' reply may be a question. Baseline got
        is_question from the LLM confirmation classifier; the shared machine returns confirm/
        decline/other only, so the persona re-derives it here for the answer + W9."""
        prompt = (
            "A user was asked to confirm removing part of a framework (yes/no). Instead "
            "they replied with something else. Is their reply a QUESTION or request for "
            "information (vs a hedge like 'hmm' / 'not sure')?\n"
            'Respond ONLY with JSON, no markdown: {"is_question": true or false}\n\n'
            f'Reply: "{text}"'
        )
        try:
            return bool(classify_json(prompt).get("is_question", False))
        except Exception as e:
            print(f"[GATE QUESTION] error: {e}")
            return False

    def _emit(self, msg: str):
        """Append a plain (non-rerender) agent message + log it."""
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)

    def _evctx(self, *, source="user_spontaneous", modality="text"):
        return ev.EventContext(self.session_id, source=source, modality=modality,
                               agent_type=self.concept_swap.agent_type)

    def _swap_question_signal(self, outcome, user_input: str) -> bool:
        """BlackBox's W9 question-about-swap signal — the deferred F-S instrument,
        preserved exactly (is_injected & not detected & LLM _classify_swap_question).
        `outcome` is accepted so EXP/HITL can override using outcome.is_about_swap."""
        return (self.concept_swap.is_injected and not self.concept_swap.is_detected
                and self._classify_swap_question(user_input))

    def _fire_turn(self, outcome, user_input, was_pending):
        """The ONE firing call per turn (I-1). Computes the two turn-flow booleans BlackBox
        owns and hands them to the shared record_turn; all event/field/counter logic lives
        in backend.logging.events."""
        kind = type(outcome).__name__
        is_q = swap_q = False
        if kind == "QuestionOutcome":
            is_q = True
            swap_q = self._swap_question_signal(outcome, user_input)
        elif kind == "RemovalOutcome" and outcome.stage == "challenged" and was_pending:
            is_q = self._reply_is_question(user_input)
            swap_q = is_q and getattr(outcome, "is_swap", False)   # parked path used o.is_swap
        ev.record_turn(outcome, self._evctx(), _sink,
                       was_pending=was_pending, is_question=is_q, swap_question=swap_q)

    def send_message(self, user_input: str) -> str:
        log_user_message(self.session_id, user_input)

        summary_prompt = (
            f"Based on our conversation, provide a summary in this exact format:\n\n"
            f"**Final Framework: [Framework Name]**\n\n"
            f"**The Framework:**\n"
            f"For each primary pillar, list its analytical questions as bullet points.\n"
            f"Copy the EXACT analytical questions from our conversation — do not paraphrase or omit them.\n"
            f"If the user added new concepts or sub-bullets, include those too.\n\n"
            f"Then in 2-3 sentences: note any concepts the user removed and any "
            f"concepts they added during the session."
            f"Do NOT add a follow-up question at the end — summary only."
        )

        self.history.append(
            types.Content(role="user", parts=[types.Part(text=summary_prompt)])
        )
        try:
            response = client.models.generate_content(
                model=MAIN_MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_prompt(),
                ),
            )
            reply = response.text
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )
        except Exception as e:
            reply = f"Sorry, I encountered an error: {str(e)}"
        log_agent_response(self.session_id, reply)
        return reply

    def _is_answer(self, text: str) -> bool:
        try:
            parsed = classify_json(
                f"{ANSWER_CLASSIFIER_PROMPT}\n\nAgent response: \"{text[:800]}\""
            )
            result = (
                parsed.get("is_answer", False) and
                parsed.get("confidence", 0.0) >= ANSWER_THRESHOLD
            )
            print(f"[ANSWER] is_answer={parsed.get('is_answer')}, "
                  f"confidence={parsed.get('confidence')}, stored={result}")
            return result
        except Exception as e:
            print(f"[ANSWER] error: {e}")
            return False

    def _classify_swap_question(self, user_input: str) -> bool:
        """Swap-question check (W9): is this question specifically ABOUT the swap
        concept? The `is_about_swap` half of the retired `_classify_question` — the
        intent router now owns is_question. Called only on `question` turns while the
        swap is active. Returns False on any error (never over-logs swap_questioned).
        The §3.6 single shared questioned-vs-detected prompt is Step 5."""
        swap_concept = self.concept_swap.config["wrong_concept"]
        prompt = (
            "You are a classifier for a case interview tool.\n"
            "Determine whether the user's message is a QUESTION or request for explanation "
            "specifically ABOUT this concept:\n"
            f'"{swap_concept}"\n\n'
            'Respond ONLY with valid JSON, no markdown:\n'
            '{"is_about_swap": true or false}\n\n'
            f'User message: "{user_input}"'
        )
        try:
            return bool(classify_json(prompt).get("is_about_swap", False))
        except Exception as e:
            print(f"[SWAP-Q] classifier error: {e}")
            return False

    def _check_duplicate(self, concept: str, existing_concepts: list) -> dict:
        """
        Three-layer duplicate guard for concept_added path.
        Layer 2: exact string match (Python, no LLM)
        Layer 3: fuzzy LLM check via classify_json (shared CLASSIFIER_MODEL)
        Change log: 2026-05-12
        Change log: 2026-06-07 — routed via backend.llm.classify_json (§S Step 1); dropped stray fuzzy-duplicate model
        """
        # Layer 2 — exact string match
        for existing in existing_concepts:
            if concept.strip().lower() == existing.strip().lower():
                print(f"[DUPLICATE] exact match: '{concept}' == '{existing}'")
                return {"is_duplicate": True, "matched_concept": existing}

        # Layer 3 — LLM fuzzy check
        try:
            parsed = classify_json(
                    f"You are checking if a user-suggested concept is essentially "
                    f"the same as an existing concept.\n\n"
                    f"User suggested: \"{concept}\"\n\n"
                    f"Existing concepts: {existing_concepts}\n\n"
                    f"Reply with JSON only, no markdown:\n"
                    f"{{\"is_duplicate\": true or false, "
                    f"\"matched_concept\": \"exact string from list or null\"}}\n\n"
                    f"is_duplicate=true ONLY if clearly the same concept — "
                    f"same topic, possibly different wording.\n"
                    f"WHEN IN DOUBT: is_duplicate=false."
            )
            print(f"[DUPLICATE] fuzzy check: '{concept}' → {parsed}")
            return parsed
        except Exception as e:
            print(f"[DUPLICATE] fuzzy check failed: {e} — defaulting to duplicate (safe)")
            return {"is_duplicate": True, "matched_concept": None}

    def _strip_concept_swap_from_history(self) -> list:
        note_marker = "---\n💡"
        wrong       = self.concept_swap.config["wrong_concept"].lower()
        cleaned     = []

        for msg in self.history:
            if msg.role == "model" and msg.parts:
                text = msg.parts[0].text
                if note_marker in text:
                    text = text[:text.index(note_marker)].rstrip()
                lines = [l for l in text.split("\n") if wrong not in l.lower()]
                text  = "\n".join(lines)
                msg   = types.Content(
                    role="model",
                    parts=[types.Part(text=text)]
                )
            cleaned.append(msg)

        print(f"[STRIP] concept swap removed from history ({len(cleaned)} messages)")
        return cleaned

    def _summarize_history(self) -> str:
        lines = []
        for msg in self.history:
            role = msg.role.upper()
            text = msg.parts[0].text if msg.parts else ""
            lines.append(f"[{role}]: {text[:100]}")
        return "\n".join(lines)

    def _stream_with_instruction(
        self,
        instruction: str,
        prefix: str = "",
        task_injection: str = "",
        track_swap: bool = False,
        store_answer: bool = False,
    ):
        self._pending = True
        full_reply    = []

        if prefix:
            yield prefix

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

            if track_swap:
                self.concept_swap.maybe_inject(reply)

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

    @staticmethod
    def _strip_fences(text: str) -> str:
        # Single definition lives in backend.llm; thin wrapper kept so inherited
        # self._strip_fences call sites (explainable/hitl) still resolve. (§S Step 1)
        return strip_fences(text)
