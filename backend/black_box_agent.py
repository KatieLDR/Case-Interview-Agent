import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from backend.logger import (
    create_session, end_session, stamp_started_at,
    log_user_message, log_agent_response,
    log_interruption, log_memory_override,
    update_answer, log_warmup_response,
    log_concept_added, log_add_pillar, log_add_sub_bullet, log_delete, log_question, log_swap_questioned
)
from backend.cases import get_case, get_clarification_facts
from backend.concept_swap import ConceptSwap
from backend import knowledge_base as kb

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
from backend.llm import (
    client, MAIN_MODEL, CLASSIFIER_MODEL, classify_json, strip_fences,
    ANSWER_THRESHOLD, OVERRIDE_THRESHOLD, ADD_MATCH_THRESHOLD, CONCEPT_MATCH_THRESHOLD,
)

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

OVERRIDE_CLASSIFIER_PROMPT = """
You are a classifier for a case interview research tool.

Determine whether the user's message is attempting to steer or change the
agent's output — i.e. the content, structure, or direction of the framework.

If yes, classify the type:
- "redo"             : wants the ENTIRE framework regenerated from scratch / a fresh start
                       ("start over", "redo this", "start again", "regenerate everything").
                       Removing ONE concept is NEVER redo — that is concept_excluded.
- "concept_excluded" : wants to remove a specific concept or bucket
- "concept_added"    : wants to add a new concept or bucket — includes soft proposals
                       ("we should consider X", "we need to think about / understand X",
                       "what about X", "can we also look at X"; detail = the NEW thing)
- "none"             : not steering the output

This does NOT include:
- Asking for a framework for the first time
- Asking follow-up questions ("can you elaborate?", "why is X here?")
- General questions about the case
- Single word responses ("yes", "no", "ok", "sure")
- Questions asking where a concept is ("where's X", "where is X")
- Conversational rejections ("no thanks", "not needed", "no")
- Asking what happened to a concept ("what happened to X")

A leading "no" or "not" does NOT make a message non-steering. If a negation is
followed by an instruction to add, remove, or change a concept, classify by the
INSTRUCTION, not the leading word. Only a standalone negation with no instruction
("no", "no thanks", "not needed") is non-steering.


- "parent": the existing concept named after "under" / "as part of" / "within" — null if no parent explicitly named

Respond ONLY with valid JSON, no explanation, no markdown:
{"override": true or false, "type": "redo"|"concept_excluded"|"concept_added"|"none", "detail": string or null, "parent": string or null, "confidence": float}

Examples:
- "redo this" → {"override": true, "type": "redo", "detail": null, "parent": null, "confidence": 0.99}
- "try a completely different approach" → {"override": true, "type": "redo", "detail": null, "parent": null, "confidence": 0.97}
- "remove feasibility" → {"override": true, "type": "concept_excluded", "detail": "Feasibility", "parent": null, "confidence": 0.97}
- "add financial impact as a bucket" → {"override": true, "type": "concept_added", "detail": "Financial Impact", "parent": null, "confidence": 0.97}
- "what about regulatory risks?" → {"override": true, "type": "concept_added", "detail": "Regulatory risks", "parent": null, "confidence": 0.95}
- "give me a framework" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "why is feasibility here?" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.95}
- "yes" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "no" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "no thanks" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "ok" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "sure" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "where's strategic fit" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "what happened to feasibility" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "add data quality under Feasibility" → {"override": true, "type": "concept_added", "detail": "Data quality", "parent": "Feasibility", "confidence": 0.97}
- "add cost analysis as part of Financial Impact" → {"override": true, "type": "concept_added", "detail": "Cost analysis", "parent": "Financial Impact", "confidence": 0.96}
- "no, remove feasibility" → {"override": true, "type": "concept_excluded", "detail": "Feasibility", "parent": null, "confidence": 0.96}
- "no I think we should exclude it" → {"override": true, "type": "concept_excluded", "detail": "it", "parent": null, "confidence": 0.93}
- "remove it" → {"override": true, "type": "concept_excluded", "detail": "it", "parent": null, "confidence": 0.94}
- "remove this" → {"override": true, "type": "concept_excluded", "detail": "this", "parent": null, "confidence": 0.93}
- "start over" → {"override": true, "type": "redo", "detail": null, "parent": null, "confidence": 0.98}
- "we should consider team strategy alignment" → {"override": true, "type": "concept_added", "detail": "team strategy alignment", "parent": null, "confidence": 0.95}
- "we need to understand the frequency of the problem" → {"override": true, "type": "concept_added", "detail": "frequency of the problem", "parent": null, "confidence": 0.95}
- "consider if the use case meets the team goal" → {"override": true, "type": "concept_added", "detail": "use case meets the team goal", "parent": null, "confidence": 0.94}
"""

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
# ANSWER_THRESHOLD / OVERRIDE_THRESHOLD imported from backend.llm (§S Step 1)
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

ADD_PILLAR_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.
The user wants to add a new area: "{item}".
Each area below is given as "Name: description". Check whether the user's new
area is essentially the SAME AREA of analysis as one of them — same topic or
clearly the same scope, possibly different wording.

─── AREAS ───────────────────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────
Match only at the level of the whole AREA. If the user's text is just one
specific point that would sit *inside* an area (rather than naming the area
itself), set matched=false.

Respond ONLY with valid JSON, no markdown:
{{"matched": true or false, "matched_pillar": "pillar name or null", "confidence": float}}
"""

ADD_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.
The user added "{item}" under the pillar "{pillar}".
Check whether it matches one of the pillar's existing key questions below.

─── KEY QUESTIONS FOR {pillar} ──────────────────────────────────────────────
{key_questions}
────────────────────────────────────────────────────────────────────────────
A match means the addition is essentially the same point as one of the key
questions above — same topic, possibly different wording.

Respond ONLY with valid JSON, no markdown:
{{"matched": true or false, "matched_index": integer or null, "confidence": float}}
"""

# Whole-KB concept matcher — searches every analytical concept (name + explanation)
# across ALL pillars, shown or withheld, so a user naming a sub-concept of a withheld
# pillar ("breakeven point") can be resolved to its parent ("Financial Impact").
# The swap concept is excluded by the caller (swap:true). Change log: 2026-06-02
ADD_CONCEPT_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.
The user wrote: "{item}".

Below is a numbered list of analytical concepts, each as "Name: explanation".
Decide whether the user's text clearly refers to ONE of these concepts — i.e.
it is essentially that concept, or a specific instance of it, possibly worded
differently.

─── CONCEPTS ────────────────────────────────────────────────────────────────
{concepts}
────────────────────────────────────────────────────────────────────────────
Match ONLY if you are confident it is essentially that concept. If the user's
text is only loosely or topically related, set matched=false.

Respond ONLY with valid JSON, no markdown:
{{"matched": true or false, "matched_index": integer or null, "confidence": float}}
"""

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
        self.pending_excl       = None  # pillar awaiting "Are you sure?" confirmation
        self.pending_sub_excl   = None  # {"pillar": str, "bullet": str} — sub-bullet pending removal
        self.awaiting_removal_target = False  # True after an unresolved "remove X" — next reply names the target
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

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # End-Session gate flips True on the first main-phase message (every path).
        self.has_main_contribution = True
        log_user_message(self.session_id, user_input)
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        # ── 0. Pending removal confirmation ────────────────────────────────
        if self.pending_excl is not None:
            yield from self._resolve_pending_excl(user_input)
            return
        # ── 0a. Pending sub-bullet removal confirmation ─────────────────────
        if self.pending_sub_excl is not None:
            yield from self._resolve_pending_sub_excl(user_input)
            return
        # ── 0b. Awaiting removal target (unresolved referent) ──────────────
        if self.awaiting_removal_target:
            self.awaiting_removal_target = False
            if self._is_cancel(user_input):
                yield from self._cancel_removal()
                return
            yield from self._begin_removal(user_input, user_input)
            return
        # ── 1. Override detection ──────────────────────────────────────────
        override = self._detect_override(user_input)

        # ── 1a. Swap detection — only if presented AND no override ─────────
        cs_detected = False
        if self.concept_swap.is_injected and not override:
            cs_detected = self.concept_swap.check_detection(user_input)
            if cs_detected:
                wrong = self.concept_swap.config["wrong_concept"]
                log_memory_override(self.session_id,
                    old_context=f"included: {wrong}",
                    new_context=f"user rejected: {wrong}")   # stays detection (#3), no log_delete

        # ── 2. Override handling — deterministic, no LLM framework regen ───
        if override:
            if override["type"] == "redo":
                yield from self._handle_redo(); return
            if override["type"] == "concept_excluded" and override.get("detail"):
                yield from self._begin_removal(override["detail"], user_input); return
            if override["type"] == "concept_added" and override.get("detail"):
                yield from self._handle_add(override["detail"], override.get("parent")); return
            # any non-handled override type → no-op ack (framework is fixed for the study)
            yield from self._ack_no_reprint(); return

        # ── 3. Swap just caught — neutral ack + re-render without it ────────
        if cs_detected:
            yield from self._yield_rerender("Understood — I've taken that out.\n\n")
            return

        # ── 4. Question vs contextless reply ───────────────────────────────
        swap_active = self.concept_swap.is_injected and not self.concept_swap.is_detected
        q = self._classify_question(user_input,
                                    self.concept_swap.config["wrong_concept"], swap_active)
        if q["is_question"]:
            log_question(self.session_id, "text", detail=user_input[:200])
            if swap_active and q["is_about_swap"]:
                log_swap_questioned(self.session_id, "text", detail=user_input[:200])
            yield from self._stream_qa(user_input)
        else:
            yield from self._ack_no_reprint()

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

    # ── Removal (with confirmation) ────────────────────────────────────────
    def _begin_removal(self, detail: str, user_input: str = ""):
        wrong   = self.concept_swap.config["wrong_concept"]
        is_swap = self.concept_swap.matches(user_input) or self.concept_swap.matches(detail)
        if is_swap and self.concept_swap.is_injected and not self.concept_swap.is_detected:
            self.pending_excl = wrong                      # detection logged on confirm
            msg = (f"Are you sure you want to remove **{self.pending_excl}**?\n\n"
                   f"*Reply **yes** to confirm, or **no** to keep it.*")
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg
            return

        # Classify: whole pillar or one sub-bullet?
        rt = self._classify_removal_target(user_input or detail)
        if rt["level"] == "sub_bullet" and rt.get("bullet"):
            pillar = self._normalize_pillar(rt["pillar"]) if rt.get("pillar") else detail
            self.pending_sub_excl = {"pillar": pillar, "bullet": rt["bullet"]}
            log_delete(self.session_id, rt["bullet"], "text")
            msg = (f"Are you sure you want to remove this point from **{pillar}**?\n\n"
                   f"*\"{rt['bullet']}\"*\n\n"
                   f"*Reply **yes** to confirm, or **no** to keep it.*")
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg
            return

        target = self._normalize_pillar(rt.get("pillar") or detail)
        if not self._is_known_pillar(target):
            # Unresolved referent — confirm what they mean, don't guess/redo.
            # Stay in removal; the next reply re-enters here via step 0b.
            self.awaiting_removal_target = True
            options = self._removable_pillar_names()
            options_line = (
                "\n\nCurrently in the framework: "
                + ", ".join(f"**{o}**" for o in options) + "."
            ) if options else ""
            msg = ("Which part would you like to remove? "
                   "You can name the pillar or the point."
                   + options_line
                   + "\n\n*(Or say **never mind** to keep everything as is.)*")
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg
            return

        self.pending_excl = target
        log_delete(self.session_id, target, "text")    # removal intent (#1)
        msg = (f"Are you sure you want to remove **{self.pending_excl}**?\n\n"
               f"*Reply **yes** to confirm, or **no** to keep it.*")
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _is_known_pillar(self, name: str) -> bool:
        from backend import knowledge_base as kb
        n = (name or "").lower().strip()
        if not n:
            return False
        if n in [p["name"].lower() for p in kb.get_shown_pillars()]:
            return True
        return n in [p.lower() for p in self.user_added_pillars]
    
    def _removable_pillar_names(self) -> list[str]:
        """Pillars currently visible to the user (mirrors the rendered framework)."""
        from backend import knowledge_base as kb
        excluded = [e.lower() for e in self.excluded_concepts]
        names = [p["name"] for p in kb.get_shown_pillars()
                 if p["name"].lower() not in excluded]
        if self.concept_swap.is_injected and not self.concept_swap.is_detected:
            names.append(self.concept_swap.config["wrong_concept"])
        for name in self.user_added_pillars:
            if name.lower() not in excluded and name not in names:
                names.append(name)
        return names

    def _is_cancel(self, user_input: str) -> bool:
        """LLM intent check: is the user backing out of the removal prompt
        (keep everything / never mind) rather than naming a part to remove?"""
        prompt = (
            "A user was asked which part of a framework they want to REMOVE.\n"
            "Decide whether their reply is BACKING OUT / cancelling the removal "
            "(wants to keep everything, never mind, stop) — as opposed to naming "
            "a part to remove or saying anything else.\n"
            'Respond ONLY with JSON, no markdown: {"cancel": true or false}\n\n'
            f'Reply: "{user_input}"'
        )
        try:
            return bool(classify_json(prompt).get("cancel", False))
        except Exception as e:
            print(f"[CANCEL] error: {e}")
            return False   # safe default — re-ask rather than abandon the removal

    def _cancel_removal(self):
        msg = "No problem — I'll keep everything as is. Let me know if there's anything else."
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    # ── Sub-bullet removal helpers ─────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", _REF_RE.sub("", text or "")).strip().lower()

    def _is_excluded_bullet(self, pillar_name: str, bullet: str) -> bool:
        removed = {self._norm(b) for b in self.excluded_sub_bullets.get(pillar_name, [])}
        return self._norm(bullet) in removed

    def _last_agent_message(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts and c.parts[0].text:
                return c.parts[0].text[:600]
        return ""

    def _classify_removal_target(self, user_input: str) -> dict:
        """Decide whether a removal targets a whole pillar or one sub-bullet.
        Falls back to pillar-level on any error."""
        from backend import knowledge_base as kb
        excluded = [e.lower() for e in self.excluded_concepts]
        shown    = [p for p in kb.get_shown_pillars() if p["name"].lower() not in excluded]
        swap     = kb.get_swap_concept()
        wrong    = self.concept_swap.config["wrong_concept"]

        lines = []
        for p in shown:
            lines.append(f"**{p['name']}**")
            for b in p.get("sub_bullets", []):
                if not self._is_excluded_bullet(p["name"], b):
                    lines.append(f"  - {_strip_source_refs(b)}")
            for sp in self.user_sub_points.get(p["name"], []):
                if not self._is_excluded_bullet(p["name"], sp):
                    lines.append(f"  - {sp}")
        if self.concept_swap.is_injected and not self.concept_swap.is_detected and swap:
            lines.append(f"**{wrong}**")
            for b in swap.get("sub_bullets", []):
                if not self._is_excluded_bullet(wrong, b):
                    lines.append(f"  - {_strip_source_refs(b)}")
        for name in self.user_added_pillars:
            if name.lower() in excluded:
                continue
            lines.append(f"**{name}**")
            kbp = next((p for p in kb.get_all_pillars() if p["name"].lower() == name.lower()), None)
            for b in (kbp.get("sub_bullets", []) if kbp else []):
                if not self._is_excluded_bullet(name, b):
                    lines.append(f"  - {_strip_source_refs(b)}")
            for sp in self.user_sub_points.get(name, []):
                if not self._is_excluded_bullet(name, sp):
                    lines.append(f"  - {sp}")

        framework_block = "\n".join(lines) or "(none)"
        prompt = REMOVAL_TARGET_PROMPT.format(
            framework_bullets=framework_block,
            last_agent=self._last_agent_message() or "(nothing yet)",
            user_msg=user_input,
        )
        try:
            parsed = classify_json(prompt)
            level = parsed.get("level", "pillar")
            return {
                "level":  level if level in ("pillar", "sub_bullet") else "pillar",
                "pillar": parsed.get("pillar"),
                "bullet": parsed.get("bullet"),
            }
        except Exception as e:
            print(f"[REMOVAL TARGET] error: {e} — defaulting to pillar")
            return {"level": "pillar", "pillar": None, "bullet": None}

    def _resolve_pending_sub_excl(self, user_input: str):
        """3-way gate for a pending sub-bullet removal: confirm | decline | other.
        BlackBox: no reasoned pushback — re-offer the plain yes/no on 'other'."""
        pillar = self.pending_sub_excl["pillar"]
        bullet = self.pending_sub_excl["bullet"]
        cls    = self._classify_confirmation(user_input)

        if cls["decision"] == "confirm":
            self.excluded_sub_bullets.setdefault(pillar, [])
            if not self._is_excluded_bullet(pillar, bullet):
                self.excluded_sub_bullets[pillar].append(bullet)
            self.pending_sub_excl = None
            log_memory_override(self.session_id,
                old_context=f"sub-bullet in {pillar}: {bullet}",
                new_context=f"user confirmed removal")
            yield from self._yield_rerender(f"Done — I've removed that point from **{pillar}**.\n\n")

        elif cls["decision"] == "decline":
            self.pending_sub_excl = None
            msg = f"No problem — I'll keep that point in **{pillar}**."
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg

        else:
            if cls.get("is_question"):
                log_question(self.session_id, "text", detail=user_input[:200])
            msg = (f"No rush — reply **yes** to remove that point from **{pillar}**, "
                   f"or **no** to keep it.")
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg

    def _resolve_pending_excl(self, user_input: str):
        concept = self.pending_excl
        cls = self._classify_confirmation(user_input)

        # ── Confirm → remove ───────────────────────────────────────────────
        if cls["decision"] == "confirm":
            self.pending_excl = None
            wrong = self.concept_swap.config["wrong_concept"]
            if concept.lower() == wrong.lower():
                self.concept_swap.force_detected()             # swap removal = detection (#3)
            elif concept.lower() not in [e.lower() for e in self.excluded_concepts]:
                self.excluded_concepts.append(concept)
            log_memory_override(self.session_id,
                old_context=f"concept in framework: {concept}",
                new_context=f"user confirmed removal: {concept}")
            yield from self._yield_rerender(f"Done — I've removed **{concept}**.\n\n")
            return

        # ── Decline → keep ─────────────────────────────────────────────────
        if cls["decision"] == "decline":
            self.pending_excl = None
            msg = f"No problem — I'll keep **{concept}**."
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg
            return

        # ── Other → answer if it's a question, then re-offer; KEEP pending ──
        if cls["is_question"]:
            log_question(self.session_id, "text", detail=user_input[:200])
            wrong = self.concept_swap.config["wrong_concept"]
            swap_active = self.concept_swap.is_injected and not self.concept_swap.is_detected
            # At the gate the question is about the PENDING concept → state check, not LLM.
            if swap_active and concept.lower() == wrong.lower():
                log_swap_questioned(self.session_id, "text", detail=user_input[:200])
            yield from self._stream_confirm_qa(user_input, concept)
        else:
            msg = f"No rush — reply **yes** to remove **{concept}**, or **no** to keep it."
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg

    def _classify_confirmation(self, user_input: str) -> dict:
        """confirm = clear yes; decline = clear no; other = question / hedge / not-a-clear-yes-no.
        Whether an 'other' question is about the swap is decided by the CALLER from state
        (the pending concept) — the LLM can't resolve "that"/"this" here. Change log: 2026-06-03"""
        prompt = (
            "A user was asked to confirm removing a concept from a framework "
            "(e.g. 'Are you sure you want to remove it?').\n"
            "Classify their reply:\n"
            '- "confirm": a clear yes — go ahead and remove it\n'
            '- "decline": a clear no — keep it / leave it as is\n'
            '- "other": a question, an uncertain/hedging reply, or anything that is NOT a clear yes or no\n\n'
            "Vague 'proceed'-type words on their own ('move on', 'next', 'whatever', 'sure I guess') "
            'are "other", NOT confirm, unless the user clearly says to remove it.\n\n'
            'Respond ONLY with valid JSON, no markdown:\n'
            '{"decision": "confirm" | "decline" | "other", "is_question": true or false}\n\n'
            f'Reply: "{user_input}"'
        )
        try:
            p = classify_json(prompt)
            decision = p.get("decision", "other")
            if decision not in ("confirm", "decline", "other"):
                decision = "other"
            return {"decision": decision, "is_question": bool(p.get("is_question", False))}
        except Exception as e:
            print(f"[CONFIRM] error: {e}")
            return {"decision": "other", "is_question": False}

    # ── Add (silent placement + matching) ──────────────────────────────────
    # Two-stage match — Change log: 2026-06-02
    #   Stage 1: pillar-level match (user named the AREA). Reveal pillar, log add_pillar.
    #   Stage 2: whole-KB concept search (user named a POINT inside an area). Reveal the
    #            parent pillar silently and log add_sub_bullet ONLY — naming a sub-concept
    #            is a sub-bullet act, not a pillar act, so the add_pillar count must not
    #            rise. (Reveal is presentation, not a second contribution.)
    #   Stage 3: genuinely new free-text area → new pillar, log add_pillar.
    def _handle_add(self, detail: str, parent: str | None):
        from backend import knowledge_base as kb
        if parent:
            target = self._normalize_pillar(parent)
            if not self._is_known_pillar(target):
                matched = self._match_pillar(parent)      # semantic resolve (e.g. "use case")
                if matched:
                    target = self._normalize_pillar(matched)
            self._ensure_pillar_visible(target)
            _, is_new = self._store_sub_point(target, detail, "text")
            pre = (f"Added under **{target}**.\n\n" if is_new else
                   f"That's already under **{target}**.\n\n")
            yield from self._yield_rerender(pre); return

        # ── Stage 1: pillar-level match (naming the area) ──────────────────
        matched = self._match_pillar(detail)
        if matched:
            shown_names = [p["name"].lower() for p in kb.get_shown_pillars()]
            if matched.lower() in shown_names:
                # Named an already-shown area → fold the note in as a sub-point.
                _, is_new = self._store_sub_point(matched, detail, "text")
                pre = (f"Added under **{matched}**.\n\n" if is_new else
                       f"That's already under **{matched}**.\n\n")
                yield from self._yield_rerender(pre); return
            # Named a WITHHELD area → reveal it as a pillar (add_pillar).
            if matched.lower() not in [p.lower() for p in self.user_added_pillars]:
                self.user_added_pillars.append(matched)
                log_concept_added(self.session_id, matched)
                log_add_pillar(self.session_id, matched, "text")
                pre = f"Added **{matched}** as a new area.\n\n"
            else:
                pre = f"**{matched}** is already in the framework.\n\n"
            yield from self._yield_rerender(pre); return

        # ── Stage 2: whole-KB concept search (naming a point inside an area) ─
        concept_parent = self._match_concept(detail)
        if concept_parent:
            # Reveal the parent pillar WITHOUT logging add_pillar (granularity rule);
            # _store_sub_point logs the user's contribution as add_sub_bullet.
            self._reveal_pillar_silent(concept_parent)
            _, is_new = self._store_sub_point(concept_parent, detail, "text")
            pre = (f"Added under **{concept_parent}**.\n\n" if is_new else
                   f"That's already under **{concept_parent}**.\n\n")
            yield from self._yield_rerender(pre); return

        # ── Stage 3: genuinely new area → free-text pillar (add_pillar) ────
        new_name = detail.strip()
        if new_name.lower() not in [p.lower() for p in self.user_added_pillars]:
            self.user_added_pillars.append(new_name)
            log_concept_added(self.session_id, new_name)
            log_add_pillar(self.session_id, new_name, "text")
            pre = f"Added **{new_name}** as a new area.\n\n"
        else:
            pre = f"**{new_name}** is already in the framework.\n\n"
        yield from self._yield_rerender(pre)

    def _ensure_pillar_visible(self, name: str):
        from backend import knowledge_base as kb
        if name.lower() in [p["name"].lower() for p in kb.get_shown_pillars()]:
            return
        if name.lower() in [p.lower() for p in self.user_added_pillars]:
            return
        self.user_added_pillars.append(name)
        log_add_pillar(self.session_id, name, "text")

    def _reveal_pillar_silent(self, name: str):
        """Reveal a matched (typically withheld) pillar so its KB sub-bullets render,
        WITHOUT logging add_pillar. Used when the user named a sub-concept rather than
        the area itself — the contribution is a sub-bullet act, logged separately by
        _store_sub_point as add_sub_bullet. No-op if already shown or already revealed.
        Change log: 2026-06-02"""
        from backend import knowledge_base as kb
        if name.lower() in [p["name"].lower() for p in kb.get_shown_pillars()]:
            return                                  # already on screen — no reveal needed
        if name.lower() in [p.lower() for p in self.user_added_pillars]:
            return                                  # already revealed
        self.user_added_pillars.append(name)        # deliberately NO log_add_pillar

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
        log_concept_added(self.session_id, item)
        log_add_sub_bullet(self.session_id, stored, modality)
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
        """Stage-1 matcher: does the user's text name one of the framework's AREAS?
        Candidate block now includes each pillar's description so semantically-adjacent
        terms ("IT Budget" → "Financial Impact") resolve. Change log: 2026-06-02"""
        from backend import knowledge_base as kb
        pillars = kb.get_all_pillars()
        if not pillars:
            return None
        block = "\n".join(
            f"- {p['name']}: {(p.get('description') or '').strip()}" for p in pillars
        )
        try:
            parsed = classify_json(ADD_PILLAR_MATCH_PROMPT.format(item=item, pillars=block))
            if parsed.get("matched") and parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD:
                return parsed.get("matched_pillar")
        except Exception as e:
            print(f"[PILLAR MATCH] error: {e}")
        return None

    def _match_concept(self, item: str):
        """Stage-2 matcher: search EVERY analytical concept (name + explanation) across
        all pillars (swap excluded). On a confident hit, return the parent pillar's name
        via pillar_id. Lets a sub-concept of a withheld pillar ("breakeven point") resolve
        to its parent ("Financial Impact"). Change log: 2026-06-02"""
        from backend import knowledge_base as kb
        concepts = [c for c in kb.get_all_concepts()
                    if not c.get("swap", False) and c.get("pillar_id")]
        if not concepts:
            return None
        block = "\n".join(
            f"{i}. {c['name']}: {(c.get('explanation') or '').strip()}"
            for i, c in enumerate(concepts)
        )
        try:
            parsed = classify_json(ADD_CONCEPT_MATCH_PROMPT.format(item=item, concepts=block))
            if parsed.get("matched") and parsed.get("confidence", 0.0) >= CONCEPT_MATCH_THRESHOLD:
                idx = parsed.get("matched_index")
                if idx is not None and 0 <= idx < len(concepts):
                    pillar = kb.get_pillar_by_id(concepts[idx]["pillar_id"])
                    if pillar:
                        return pillar["name"]
        except Exception as e:
            print(f"[CONCEPT MATCH] error: {e}")
        return None

    def _match_key_question(self, item: str, pillar_name: str):
        from backend import knowledge_base as kb
        pillar = next((p for p in kb.get_all_pillars()
                       if p["name"].lower() == pillar_name.lower()), None)
        if pillar is None:
            return None
        kqs = pillar.get("key_questions", [])
        if not kqs:
            return None
        kq_block = "\n".join(f"{i}. {q}" for i, q in enumerate(kqs))
        try:
            parsed = classify_json(ADD_MATCH_PROMPT.format(item=item, pillar=pillar_name, key_questions=kq_block))
            if parsed.get("matched") and parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD:
                idx = parsed.get("matched_index")
                if idx is not None and 0 <= idx < len(kqs):
                    return _strip_source_refs(kqs[idx])
        except Exception as e:
            print(f"[ADD MATCH] error: {e}")
        return None

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
    def _handle_redo(self):
        self.user_sub_points      = {}
        self.user_added_pillars   = []
        self.excluded_concepts    = []
        self.excluded_sub_bullets = {}
        self.pending_excl         = None
        self.pending_sub_excl     = None
        self.has_main_contribution = False
        self.awaiting_removal_target = False
        log_user_message(self.session_id, "[REDO TRIGGERED]")
        reply = "Noted — here's a fresh framework.\n\n" + self._render_full_framework(is_first=True)
        self.history.append(types.Content(role="model", parts=[types.Part(text=reply)]))
        update_answer(self.session_id, reply)
        log_agent_response(self.session_id, reply)
        yield reply

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
        pending decision in ONE streamed message. Caller leaves pending_excl set, so the
        user's next yes/no still resolves the removal. Change log: 2026-06-02"""
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

    def _detect_override(self, user_input: str) -> dict | None:
        """Single classifier for all override types. Used for research logging only."""
        try:
            parsed = classify_json(
                f"{OVERRIDE_CLASSIFIER_PROMPT}\n\nUser message: \"{user_input}\""
            )
            if (parsed.get("override", False) and
                    parsed.get("confidence", 0.0) >= OVERRIDE_THRESHOLD and
                    parsed.get("type", "none") != "none"):
                return {
                    "type":       parsed["type"],
                    "detail":     parsed.get("detail"),
                    "parent":     parsed.get("parent"),
                    "confidence": parsed["confidence"],
                }
        except Exception as e:
            print(f"[OVERRIDE] error: {e}")
        return None
    
    def _classify_question(self, user_input: str, swap_concept: str, swap_active: bool) -> dict:
        """
        Lite classifier: is this a question / explanation request (vs an affirmation
        or comment), and is it specifically about the swap concept? Runs only on
        non-steering, non-swap-rejection turns. Change log: 2026-05-30
        """
        swap_clause = (
            f'Then decide whether the question is specifically ABOUT this concept:\n"{swap_concept}"\n'
            if swap_active else 'Set is_about_swap to false.\n'
        )
        prompt = (
            "You are a classifier for a case interview tool.\n"
            "Determine whether the user's message is a QUESTION or request for explanation "
            "about the framework/case — as opposed to an affirmation, acknowledgement, "
            "comment, or steering command.\n\n"
            "Questions: 'why is X here?', 'can you explain...', 'what does ... mean?', "
            "'how does this apply?', 'is this relevant?'\n"
            "NOT questions: 'ok', 'thanks', 'sounds good', 'yes', 'no', plain statements.\n\n"
            f"{swap_clause}\n"
            'Respond ONLY with valid JSON, no markdown:\n'
            '{"is_question": true or false, "is_about_swap": true or false}\n\n'
            f'User message: "{user_input}"'
        )
        try:
            parsed = classify_json(prompt)
            return {"is_question": bool(parsed.get("is_question", False)),
                    "is_about_swap": bool(parsed.get("is_about_swap", False))}
        except Exception as e:
            print(f"[QUESTION] classifier error: {e}")
            return {"is_question": False, "is_about_swap": False}
        
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