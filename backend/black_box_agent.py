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
from backend.cases import get_case, get_clarification_facts
from backend.concept_swap import ConceptSwap
from backend import knowledge_base as kb

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

# ── Model config ───────────────────────────────────────────────────────────
# MAIN_MODEL / CLASSIFIER_MODEL imported from backend.llm (REFACTOR_PLAN §S Step 1)

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "AI Implementation"

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

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

SYSTEM_PROMPT = """
You are a strategic consultant specializing in structured frameworks. Your goal
is to provide a concise, high-level logical breakdown of business problems.

STRICT OUTPUT FORMAT — follow this exactly:

**Core Question**
One single question the framework aims to solve.

**The Framework**

**[Pillar]**
- [analytical question, 5-7 words, specific to this case]
- [analytical question, 5-7 words, specific to this case]

**[Pillar]**
- [analytical question, 5-7 words, specific to this case]
- [analytical question, 5-7 words, specific to this case]

(continue for all pillars — do NOT add any not in FRAMEWORK CONTEXT above,
UNLESS the user explicitly requests it)

─── STRUCTURAL EXAMPLE (format only — do not copy these questions) ──────
**Strategic Fit**
- Is GenAI the right tool, or would simpler rules-based automation achieve the same result?
- Is the use case consistent with responsible AI principles — transparency, human oversight, and accountability?

**Solution Design & Scope**
- Is the prototype scoped tightly enough to be reviewable and governable?
- What is the input data classification level — this determines the compliance path?

**Feasibility**
- Is the input data GDPR-compliant and sufficient in quality and volume?
- Is there a single-developer dependency with no designated long-term owner?
─────────────────────────────────────────────────────────────────────────

**Key Considerations** *(only if relevant)*
- Critical dependency 1
- Critical dependency 2

─── INTERACTION STYLE ────────────────────────────────────────────────────
You are a reference tool, not an interviewer. After presenting or updating
a framework, ask ONE short natural follow-up question to invite exploration.

If the user wants to ADD a new concept or bucket:
- Add it immediately, no pushback
- If it fits within Strategic Fit, Solution Design & Scope, or Feasibility → add as a sub-bullet under the right pillar
- If it is a new top-level area (e.g. Risks, Financial Impact) → add as a new primary pillar
- Never refuse user additions
- When a user explicitly asks to add a sub-bullet, always honour it — no limit applies

If the user wants to REMOVE or CHANGE an existing concept:
- Briefly explain your reasoning in one sentence
- Ask if they still want to proceed
- If they confirm, honour it immediately

─── RULES ────────────────────────────────────────────────────────────────
- Always use the exact format above — bold pillar headers, bullet questions
- Never use numbered lists for framework pillars
- Be direct and concise
- Never evaluate or score the user
- Ask only ONE follow-up question per response
- Always present exactly 2 analytical questions per pillar
"""

# ══════════════════════════════════════════════════════════════════════════
# Classifier prompts
# ══════════════════════════════════════════════════════════════════════════

ANSWER_CLASSIFIER_PROMPT = """
You are a classifier for a case interview tool.

Determine whether the agent response contains a structured framework answer
with clear primary buckets and sub-buckets.

Short replies, clarifications, questions, or discussion do NOT qualify.

Respond ONLY with valid JSON, no explanation, no markdown:
{"is_answer": true or false, "confidence": float between 0.0 and 1.0}
"""

# OVERRIDE_CLASSIFIER_PROMPT retired (Step 3 dead-code pass): _detect_override is
# gone; the unified router (interaction/intents.py) owns steering classification.

# ══════════════════════════════════════════════════════════════════════════
# Warm-up content
# Change log: 2026-05-05 — redesigned warm-up
# Change log: 2026-05-06 — added "Let's go" cue
# Change log: 2026-05-22 — pre-built plan, LLM merge
# ══════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════
# Thresholds
# ══════════════════════════════════════════════════════════════════════════
# ANSWER_THRESHOLD imported from backend.llm (§S Step 1); OVERRIDE_THRESHOLD retired (Step 3)
MAX_TURNS_PER_SESSION = 50

# Light cancel-escape phrases for the "which part to remove?" prompt (no LLM).
_CANCEL_PHRASES = {
    "never mind", "nevermind", "cancel", "stop", "forget it", "forget about it",
    "no", "nope", "no thanks", "nothing", "none", "leave it", "leave it alone",
    "keep it", "keep everything", "actually no", "skip", "skip it", "dont", "don't",
}

# ── Deterministic add-flow matchers (source-free) — Change log: 2026-06-01 ──
_REF_RE = re.compile(r"\s*\[[a-z]\]")
def _strip_source_refs(text: str) -> str:
    return _REF_RE.sub("", text or "").strip()

# ADD_MATCH_THRESHOLD imported from backend.llm (§S Step 1)

# Whole-KB concept search runs over ~34 non-swap concepts (much broader than the
# 5-item pillar-name search), and a false hit REVEALS a withheld pillar — i.e.
# stimulus contamination. So this path uses a deliberately higher bar than the
# pillar matcher. Change log: 2026-06-02
# CONCEPT_MATCH_THRESHOLD imported from backend.llm (§S Step 1)

# ADD_PILLAR_MATCH_PROMPT / ADD_MATCH_PROMPT / ADD_CONCEPT_MATCH_PROMPT moved to
# backend.domain.matching (Step 2 — one shared copy). The matcher methods below delegate;
# prompts + thresholds are unchanged (canonical text = EXP/BB, which were identical here).

REMOVAL_TARGET_PROMPT = """
You classify WHAT a user wants to remove from a framework.

Decide whether they mean:
- "pillar"     : a whole top-level area/pillar
- "sub_bullet" : ONE specific point/bullet inside a pillar

─── FRAMEWORK (all visible pillars and their current points) ──────────────────
{framework_bullets}
────────────────────────────────────────────────────────────────────────────
─── WHAT THE AGENT LAST SAID ─────────────────────────────────────────────────
{last_agent}
────────────────────────────────────────────────────────────────────────────
─── USER MESSAGE ─────────────────────────────────────────────────────────────
{user_msg}
────────────────────────────────────────────────────────────────────────────

Rules:
- Names/refers to a whole pillar or area → level="pillar", pillar=that name, bullet=null.
- Refers to one specific point → level="sub_bullet", pillar=its pillar name,
  bullet = the EXACT matching point text copied verbatim from the framework above.
- Vague "remove it / this / that": if the agent's last message discussed ONE
  specific point, treat as that sub_bullet; otherwise treat as the whole pillar.
- "bullet" MUST be copied verbatim from the framework above, or null for a pillar.

Respond ONLY with valid JSON, no markdown:
{{"level": "pillar" | "sub_bullet", "pillar": string or null, "bullet": string or null}}
"""

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

class BlackBoxAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self._pending      = False
        self.turn_count    = 0
        self.has_main_contribution = False   # gates End Session button (app.py reads this)
        # ── Deterministic framework state — Change log: 2026-06-01 ─────────
        self.user_sub_points    = {}    # pillar name -> [sub-bullet, ...]
        self.user_added_pillars = []    # user-added pillar names (order preserved)
        self.excluded_concepts  = []    # confirmed-removed pillar names
        self.excluded_sub_bullets = {}  # pillar name -> [removed bullet texts]
        # ── Step 4b: shared HandlerSession flow state (replaces pending_excl /
        #    pending_sub_excl / awaiting_removal_target). interaction/handlers.py
        #    owns the two-turn removal loop (PendingAction) now. ─────────────
        self.pending = None             # PendingAction parked by removal_handler
        self.pending_suggestion = None  # {level,item,origin} — D7 suggest / B6 remove-offer
        self.last_discussed = None      # Fork-A focus; BlackBox full-render -> stays None
        self.shown_bullets = []         # positional-removal context (unused in BB full-render)
        self._last_surface = None       # stash: surface_pillar result (render reads is_new)
        self._last_sub_add = None       # stash: add_sub_point result (render reads stored/is_new)
        self._ack_index         = 0     # rotates the no-reprint acknowledgements
        self.phase = "warmup"
        self.clarification_facts = get_clarification_facts("black_box")

        self.concept_swap = ConceptSwap(
            agent_type="black_box",
            session_id=self.session_id
        )

        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        print(f"[KB INIT] case_type={CASE_TYPE}, "
              f"framework={self.kg_context['framework']}, "
              f"concepts={self.kg_context['concepts']}")

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
    # KG helpers
    # ══════════════════════════════════════════════════════════════════════

    def _fetch_kg_context(self, case_type: str) -> dict:
        """
        Load framework context from JSON knowledge base.
        Change log: 2026-05-28 — migrated from KG to JSON knowledge base.
        """
        from backend import knowledge_base as kb
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

    def _build_system_prompt(self) -> str:
        framework    = self.kg_context["framework"]
        concepts_str = ", ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else \
                       "Strategic Fit, Solution Design & Scope, Feasibility"

        framework_block = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Case Type : {self.kg_context['case_type']}\n"
            f"Framework : {framework}\n"
            f"Pillars   : {concepts_str}\n"
            f"CRITICAL: Use pillars as PRIMARY BUCKET HEADERS.\n"
            f"Generate exactly 2 analytical questions directly under each pillar — no sub-headers.\n"
            f"Do NOT add any pillar not listed here UNLESS the user explicitly requests it.\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
        )

        swap_block = self.concept_swap.get_system_prompt_block()
        return framework_block + swap_block + SYSTEM_PROMPT

    def _build_tree_overview(self) -> str:
        from backend import knowledge_base as kb
        shown = kb.get_shown_pillars()
        lines = ["**Framework Overview**\n"]
        for pillar in shown:
            lines.append(f"- {pillar['name']}")
        return "\n".join(lines)

    def show_tree(self) -> str:
        return self._build_tree_overview()

    # ══════════════════════════════════════════════════════════════════════
    # Warm-up messages
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Opening message
    # ══════════════════════════════════════════════════════════════════════

    def get_opening_message(self) -> str:
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
        )

    def get_pre_analysis_instruction(self) -> str:
        return (
            "📖 *After you click the button below, read each concept carefully — "
            "add any ideas or questions that come to mind as we go.*"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Phase transition — shared setup (non-generator)
    # Change log: 2026-05-05
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Phase transition — tree/button flow
    # Change log: 2026-05-12 — replaces start_main_phase()
    # ══════════════════════════════════════════════════════════════════════

    def begin_analysis(self):
        """
        Generator — called when user clicks 'Got it, show me the full analysis'.
        Change log: 2026-05-12
        """
        self._start_main_phase_setup()

        yield (
            "⚠️ Your goal is to build a structured plan for this case. "
            "Review each factor below, share your thoughts, and you **should not only read it** but also add or remove anything you think is missing."
        )

        yield "⏱️ Your 20-minute session has started. The timer is shown on the left."

        yield from self._stream_framework_presentation()

        yield (
            "\n\n---\n\n"
            "📖 *Now it's your turn — do you have any questions about the framework "
            "presented? Add, remove, or update anything as you like. Once you've shared "
            "your first thoughts, an **‼️End Session** button will appear so you can "
            "finish whenever you're ready.*"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Main message handler
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Clarification phase streaming
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Framework presentation
    # Change log: 2026-05-01
    # Change log: 2026-05-12 — removed debug print
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Main phase streaming
    # Change log: 2026-05-12 — override first, swap gated on is_injected AND not override
    # Change log: 2026-05-16 — skip log_memory_override for concept_added
    # ══════════════════════════════════════════════════════════════════════

    def _last_agent_text(self) -> str:
        """Last model message (router context). BlackBox has no walkthrough cursor."""
        for c in reversed(self.history):
            if c.role == "model" and c.parts:
                return (c.parts[0].text or "")[:500]
        return ""

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # End-Session gate flips True on the first main-phase message (every path).
        self.has_main_contribution = True
        log_user_message(self.session_id, user_input)
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        # ── 1. Unified intent (Step 3). BlackBox renders the whole framework, so
        #      there is NO single 'current pillar'; the router gets the shown-pillar
        #      list + last agent message.
        res = intents.classify_intent(
            user_input,
            current_pillar="(none)",
            current_bullets="(none)",
            walkthrough_pillars=", ".join(self.kg_context["concepts"])
                or "Strategic Fit, Solution Design & Scope, Feasibility",
            last_agent=self._last_agent_text() or "(nothing yet)",
        )
        intent = res.intent

        # ── 1a. Swap semantic backstop — W2: only on a FRESH non-steering turn (no
        #      parked removal / suggestion), the faithful equivalent of the old
        #      `not override` gate. A NAMED swap removal is intent==remove and is
        #      detected inside removal_handler, so it is correctly excluded here.
        if (self.pending is None and self.pending_suggestion is None
                and self.concept_swap.is_injected and intent not in ("add", "remove")):
            if self.concept_swap.check_detection(user_input):
                # check_detection() already fired §3.6 swap_detected via ConceptSwap._log_detected.
                yield from self._yield_rerender("Understood — I've taken that out.\n\n")
                return

        # ── 2. Route through the shared handler layer (Step 4b). A parked removal /
        #      suggestion is resolved inside dispatch; snapshot it first so the render
        #      can recover pillar/type after the PendingAction machine clears it.
        was_pending = self.pending is not None
        pa_snapshot = self.pending
        outcome = handlers.dispatch(res, self, user_text=user_input)
        yield from self._render_outcome(outcome, user_input,
                                        was_pending=was_pending, pa=pa_snapshot)

    # ── Deterministic render ───────────────────────────────────────────────
    def _render_full_framework(self, is_first: bool = False, closing: bool = True) -> str:
        from backend import knowledge_base as kb
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

    def _last_agent_message(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts and c.parts[0].text:
                return c.parts[0].text[:600]
        return ""


    def _store_sub_point(self, pillar: str, item: str, modality: str = "text"):
        pillar  = self._normalize_pillar(pillar)
        matched = self._match_key_question(item, pillar)
        stored  = matched if matched else self._format_sub_bullet(item)
        # Already a shown static bullet on this pillar (incl. KB-backed user pillars)
        # → already in the framework, don't duplicate. Change log: 2026-06-02
        kbp = next((p for p in kb.get_all_pillars() if p["name"].lower() == pillar.lower()), None)
        if any(stored.lower() == _strip_source_refs(b).lower()
               for b in (kbp.get("sub_bullets", []) if kbp else [])):
            return stored, False
        # Re-placement: if this exact point already sits under another pillar, move it.
        for other, pts in self.user_sub_points.items():
            if other.lower() != pillar.lower():
                for s in list(pts):
                    if s.lower() == stored.lower():
                        pts.remove(s)
                        print(f"[MOVE] '{stored}' {other} -> {pillar}")
        existing = self.user_sub_points.setdefault(pillar, [])
        if any(s.lower() == stored.lower() for s in existing):
            return stored, False
        existing.append(stored)
        # Step 4b: add_sub_bullet logging is outcome-driven now (D-H1); the render
        #   fires log_concept_added + log_add_sub_bullet from the AddOutcome. Storage only.
        return stored, True

    def _normalize_pillar(self, name: str) -> str:
        from backend import knowledge_base as kb
        for p in kb.get_all_pillars():
            if p["name"].lower() == name.lower():
                return p["name"]
        for p in self.user_added_pillars:
            if p.lower() == name.lower():
                return p
        return name

    def _match_pillar(self, item: str):
        """Stage-1 matcher → shared domain matcher (Step 2). Returns the matched AREA
        name or None (contract unchanged)."""
        name, _score = matching.match_pillar(item)
        return name

    def _match_concept(self, item: str):
        """Stage-2 matcher → shared domain matcher (Step 2). Returns the matched concept's
        PARENT pillar name (contract unchanged; concept_id is exposed by locate() and
        adopted by the Step-4 handlers — see F-M1)."""
        concept, _score = matching.match_concept(item)
        if not concept:
            return None
        pillar = kb.get_pillar_by_id(concept["pillar_id"])
        return pillar["name"] if pillar else None

    def _match_key_question(self, item: str, pillar_name: str):
        """→ shared domain matcher (Step 2). Returns the matched key-question text
        (source refs stripped) or None (contract unchanged)."""
        text, _score = matching.match_key_question(item, pillar_name)
        return text

    def _format_sub_bullet(self, item: str) -> str:
        try:
            r = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=SUB_BULLET_FORMAT_PROMPT.format(item=item),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            out = _strip_source_refs(self._strip_fences(r.text)).strip().strip("-• ").rstrip(".")
            if out:
                return out
        except Exception as e:
            print(f"[SUB-POINT FORMAT] error: {e}")
        return item.strip()

    # ── Redo / Q&A / ack / summary ─────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════
    # Step 4b — HandlerSession adapter (D-H3) + outcome renderer.
    #   The shared layer (interaction/handlers.py) does the invariant work and
    #   returns a structured Outcome; BlackBox renders it (terse persona) and
    #   fires the §3.6 events DRIVEN BY the outcome (D-H1). The delete event
    #   therefore fires only at stage="confirmed" — the F-R1 fix.
    # ══════════════════════════════════════════════════════════════════════

    # ── HandlerSession queries ─────────────────────────────────────────────
    def presented_pillars(self) -> list:
        """Pillars currently rendered (mirrors _render_full_framework): shown KB
        pillars (minus excluded) + the swap (while active) + user-added."""
        excluded = [e.lower() for e in self.excluded_concepts]
        names = [p["name"] for p in kb.get_shown_pillars()
                 if p["name"].lower() not in excluded]
        if self.concept_swap.is_injected and not self.concept_swap.is_detected:
            names.append(self.concept_swap.config["wrong_concept"])
        for name in self.user_added_pillars:
            if name.lower() not in excluded and name not in names:
                names.append(name)
        return names

    def presented_sub_bullets(self) -> dict:
        """{pillar -> [non-excluded bullet texts]} for every presented pillar (KB
        sub-bullets refs-stripped + user sub-points). Drives the removal existence guard."""
        out = {}
        excluded = [e.lower() for e in self.excluded_concepts]
        def collect(name, kb_bullets):
            bl = [_strip_source_refs(b) for b in kb_bullets
                  if not self._is_excluded_bullet(name, b)]
            bl += [sp for sp in self.user_sub_points.get(name, [])
                   if not self._is_excluded_bullet(name, sp)]
            out.setdefault(name, [])
            out[name] += bl
        for p in kb.get_shown_pillars():
            if p["name"].lower() in excluded:
                continue
            collect(p["name"], p.get("sub_bullets", []))
        swap = kb.get_swap_concept()
        if self.concept_swap.is_injected and not self.concept_swap.is_detected and swap:
            collect(self.concept_swap.config["wrong_concept"], swap.get("sub_bullets", []))
        for name in self.user_added_pillars:
            if name.lower() in excluded:
                continue
            kbp = next((p for p in kb.get_all_pillars()
                        if p["name"].lower() == name.lower()), None)
            collect(name, kbp.get("sub_bullets", []) if kbp else [])
        return out

    def surfaced_pillar_names(self) -> set:
        """Everything already surfaced (shown / user-added / excluded). suggest_handler
        offers the first WITHHELD pillar NOT in this set."""
        names = {p["name"].lower() for p in kb.get_shown_pillars()}
        names |= {n.lower() for n in self.user_added_pillars}
        names |= {e.lower() for e in self.excluded_concepts}
        return names

    def current_pillar(self):
        return None   # BlackBox renders the whole framework — no walkthrough cursor.

    # ── HandlerSession mutators (pure state; logging is render-driven, D-H1) ──
    def surface_pillar(self, name: str) -> None:
        """Reveal a withheld/unreached pillar OR create a novel area — both are a
        user_added_pillars append for BlackBox. Stash is_new so the render logs
        add_pillar once (and not on a re-add)."""
        shown = name.lower() in [p["name"].lower() for p in kb.get_shown_pillars()]
        already = name.lower() in [p.lower() for p in self.user_added_pillars]
        if shown or already:
            self._last_surface = {"name": name, "is_new": False}
            return
        self.user_added_pillars.append(name)
        self._last_surface = {"name": name, "is_new": True}

    def add_sub_point(self, pillar: str, text: str) -> None:
        """Store a new sub-point (key-question canonicalisation + formatting + dedup +
        cross-pillar move live in _store_sub_point). Stash stored text + is_new so the
        render logs add_sub_bullet from the outcome (Fork-B: logs the stored text)."""
        stored, is_new = self._store_sub_point(pillar, text, "text")
        self._last_sub_add = {"pillar": pillar, "stored": stored, "raw": text, "is_new": is_new}

    # ── swap channel (PRESERVED per-arm, §0 #4) ─────────────────────────────
    def swap_name(self):
        if self.concept_swap.is_injected and not self.concept_swap.is_detected:
            return self.concept_swap.config["wrong_concept"]
        return None

    def is_swap_target(self, km, user_text: str) -> bool:
        """Does this turn target the swap concept? Deterministic name/term/stem match
        (concept_swap.matches) — BlackBox's text-match channel. (W9 question-about-swap
        is handled in the question render via _classify_swap_question.)"""
        if not self.swap_name():
            return False
        if self.concept_swap.matches(user_text):
            return True
        for cand in (getattr(km, "matched_text", None), getattr(km, "pillar", None)):
            if cand and self.concept_swap.matches(cand):
                return True
        return False

    def mark_swap_detected(self) -> None:
        self.concept_swap.force_detected()   # swap DETECTED on confirm — never a delete (§0)

    def requires_justification(self, km) -> bool:
        return False   # D-H2: BlackBox has no justification gate (scope is HITL, Step 6).

    # ── gate-reply question check (W9 / _stream_confirm_qa render) ──────────
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

    # ── §3.6 event firing (Step 5) ──────────────────────────────────────────
    # BlackBox never elicits contributions (no proactive suggestion gate), so its
    # source is always user_spontaneous; modality is always text. agent_type is read
    # from concept_swap so the inherited helper resolves correctly in EXP/HITL too.
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

    # ── outcome renderer ────────────────────────────────────────────────────
    def _render_outcome(self, outcome, user_input, *, was_pending=False, pa=None):
        # suggest_handler returns None when nothing is left to suggest.
        if outcome is None:
            msg = ("You've surfaced the main areas I'd flag — feel free to add, "
                   "remove, or question any part of what's here.")
            self._emit(msg); yield msg; return
        self._fire_turn(outcome, user_input, was_pending)   # §3.6 events (Step 5, I-1)
        if isinstance(outcome, handlers.AddOutcome):
            yield from self._render_add(outcome); return
        if isinstance(outcome, handlers.RemovalOutcome):
            yield from self._render_removal(outcome, user_input,
                                            was_pending=was_pending, pa=pa); return
        if isinstance(outcome, handlers.QuestionOutcome):
            yield from self._stream_qa(user_input); return
        if isinstance(outcome, handlers.SuggestOutcome):
            yield from self._render_suggest(outcome); return
        # AdvanceOutcome / FallbackOutcome -> terse ack, no reprint (W5).
        yield from self._ack_no_reprint()

    def _render_add(self, o):
        if o.action == "duplicate":
            if o.level == "pillar" and o.pillar:
                pre = f"**{o.pillar}** is already in the framework.\n\n"
            elif o.pillar:
                pre = f"That's already covered under **{o.pillar}**.\n\n"
            else:
                pre = "That's already in the framework.\n\n"
            yield from self._yield_rerender(pre); return
        if o.action == "navigated":
            yield from self._ack_no_reprint(); return   # revisit is not a BB behaviour
        if o.action == "revealed" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                yield from self._yield_rerender(f"Added **{o.pillar}** as a new area.\n\n")
            else:
                yield from self._yield_rerender(f"**{o.pillar}** is already in the framework.\n\n")
            return
        if o.action == "added_new" and o.level == "sub_bullet":
            st = self._last_sub_add or {}
            if st.get("is_new"):
                yield from self._yield_rerender(f"Added under **{o.pillar}**.\n\n")
            else:
                yield from self._yield_rerender(f"That's already under **{o.pillar}**.\n\n")
            return
        if o.action == "added_new" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                yield from self._yield_rerender(f"Added **{o.pillar}** as a new area.\n\n")
            else:
                yield from self._yield_rerender(f"**{o.pillar}** is already in the framework.\n\n")
            return
        yield from self._ack_no_reprint()   # defensive

    def _render_removal(self, o, user_input, *, was_pending=False, pa=None):
        stage = o.stage
        if stage == "confirmed":
            if o.is_swap:
                wrong = pa.target if pa else o.target
                yield from self._yield_rerender(f"Done — I've removed **{wrong}**.\n\n")
                return
            if pa and pa.type == "remove_sub_bullet":
                yield from self._yield_rerender(
                    f"Done — I've removed that point from **{pa.pillar}**.\n\n")
                return
            yield from self._yield_rerender(f"Done — I've removed **{o.target}**.\n\n")
            return
        if stage == "abandoned":
            if pa and pa.type == "remove_sub_bullet":
                msg = f"No problem — I'll keep that point in **{pa.pillar}**."
            else:
                msg = f"No problem — I'll keep **{o.target}**."
            self._emit(msg); yield msg; return
        if stage == "nothing_to_remove":
            if o.suggest_add_alternative:
                msg = (f"**{o.suggest_add_alternative}** isn't in the current framework. "
                       f"Did you mean to *add* it? Reply **yes** to add it.")
            else:
                msg = (f"**{o.target or 'That'}** isn't part of the current framework, "
                       f"so there's nothing to remove there.")
            self._emit(msg); yield msg; return
        if stage == "needs_disambiguation":
            options = self.presented_pillars()
            opt = ("\n\nCurrently in the framework: "
                   + ", ".join(f"**{n}**" for n in options) + ".") if options else ""
            msg = ("Which part would you like to remove? You can name the pillar or the point."
                   + opt + "\n\n*(Or say **never mind** to keep everything as is.)*")
            self._emit(msg); yield msg; return
        if stage == "challenged":
            if was_pending:
                if self._reply_is_question(user_input):
                    # §3.6 question (+ swap_questioned W9) already fired by _fire_turn.
                    yield from self._stream_confirm_qa(user_input, o.target); return
                msg = f"No rush — reply **yes** to remove **{o.target}**, or **no** to keep it."
                self._emit(msg); yield msg; return
            if self.pending and self.pending.type == "remove_sub_bullet":
                msg = (f"Are you sure you want to remove this point from "
                       f"**{self.pending.pillar}**?\n\n*\"{o.target}\"*\n\n"
                       f"*Reply **yes** to confirm, or **no** to keep it.*")
            else:
                msg = (f"Are you sure you want to remove **{o.target}**?\n\n"
                       f"*Reply **yes** to confirm, or **no** to keep it.*")
            self._emit(msg); yield msg; return
        # needs_justification (N/A for BlackBox) / defensive
        msg = f"No rush — reply **yes** to remove **{o.target}**, or **no** to keep it."
        self._emit(msg); yield msg

    def _render_suggest(self, o):
        if getattr(o, "revealed", False):
            # D7 accept: the withheld pillar is now surfaced -> re-render. No DV
            # (the ask_agent_suggestion event is Step 5); suggesting is not adding.
            yield from self._yield_rerender(f"Good point — I've included **{o.suggested_item}**.\n\n")
            return
        if not getattr(o, "suggested_item", None):
            msg = ("You've surfaced the main areas I'd flag — feel free to add, "
                   "remove, or question any part of what's here.")
            self._emit(msg); yield msg; return
        why = (o.grounding or "").split("\n")[0].strip()
        msg = (f"One area we haven't covered yet is **{o.suggested_item}**"
               + (f" — {why}" if why else "")
               + "\n\nIt's worth considering whether it applies to your case.")
        self._emit(msg); yield msg

    def _stream_summary(self):
        summary = "**Final Framework Summary**\n\n" + \
                  self._render_full_framework(is_first=False, closing=False)
        self.history.append(types.Content(role="model", parts=[types.Part(text=summary)]))
        update_answer(self.session_id, summary)
        log_agent_response(self.session_id, summary)
        yield summary

    def get_summary(self):
        yield from self._stream_summary()

    # ══════════════════════════════════════════════════════════════════════
    # Non-streaming fallback (summary)
    # Change log: 2026-05-12 — updated prompt to include sub-bullets
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Session control
    # Change log: 2026-05-12 — removed unreliable end_session swap detection
    # ══════════════════════════════════════════════════════════════════════

    def end_session(self) -> None:
        final_framework = ""
        fallback        = ""

        for msg in reversed(self.history):
            if msg.role != "model" or not msg.parts:
                continue
            text = msg.parts[0].text
            if not fallback:
                fallback = text
            if self._is_answer(text):
                final_framework = text
                print(f"[END SESSION] found final framework ({len(text)} chars)")
                break

        if not final_framework:
            final_framework = fallback
            print("[END SESSION] no structured framework — using last model message")

        detected_at_end = self.concept_swap.is_detected
        if self.concept_swap.is_detected:
            print("[END SESSION] concept swap already detected during chat")
        else:
            print("[END SESSION] concept swap not detected during session")

        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection("sessions").document(self.session_id).update({
                "final_framework":       final_framework[:1000],
                "concept_swap_detected": self.concept_swap.is_detected,
                "swap_detected_at_end":  detected_at_end,
            })
            print(f"[END SESSION] Firestore stamped for session={self.session_id}")
        except Exception as e:
            print(f"[END SESSION] Firestore stamp failed: {e}")

        end_session(self.session_id)

    # ══════════════════════════════════════════════════════════════════════
    # Classifiers
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # History helpers
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Core streaming utility — used by ExplainableAgent and HITLAgent
    # Change log: 2026-04-09 — moved up from ExplainableAgent.
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _strip_fences(text: str) -> str:
        # Single definition lives in backend.llm; thin wrapper kept so inherited
        # self._strip_fences call sites (explainable/hitl) still resolve. (§S Step 1)
        return strip_fences(text)
