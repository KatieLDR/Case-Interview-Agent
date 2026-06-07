import json
import logging
import random
import re
from google.genai import types
from backend.black_box_agent import (
    BlackBoxAgent, CLASSIFIER_MODEL, MAIN_MODEL, client, classify_json,
    ANSWER_THRESHOLD, OVERRIDE_THRESHOLD,
)
from backend.cases import get_case, get_clarification_facts
from backend.concept_swap import ConceptSwap
from backend.logger import (
    create_session, log_user_message, log_agent_response,
    log_interruption, log_memory_override, update_answer,
    log_concept_added, stamp_started_at,log_question, log_add_pillar, log_add_sub_bullet, log_delete, log_swap_questioned,
)
from backend import knowledge_base as kb           # JSON KB — static presentation + matching

# Strip inline source refs like " [a]" / "[b]" — HITL suppresses sources, so these
# markers must never reach the user (no Sources line resolves them). Change log: 2026-05-29
_REF_RE = re.compile(r"\s*\[[a-z]\]")


def _strip_source_refs(text: str) -> str:
    return _REF_RE.sub("", text).strip()

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

# ── New-area / sub-point matching — Change log: 2026-05-29 ──────────────────
# Mirror Explainable's add-flow matchers. Defined here to avoid an agent-to-agent
# import; fold into a shared module post-study.
ADD_PILLAR_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.

The user wants to add a new area: "{item}".
Check whether it matches one of the framework's other (currently hidden) areas below.

─── OTHER AREAS ─────────────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────

A match means the user's new area is essentially the same area of analysis
as one of the areas above — same topic, possibly different wording.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched": true or false, "matched_pillar": "pillar name or null", "confidence": float}}
"""

ADD_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.

The user added "{item}" under the pillar "{pillar}".
Check whether it matches one of the pillar's existing key questions below.

─── KEY QUESTIONS FOR {pillar} ──────────────────────────────────────────────
{key_questions}
────────────────────────────────────────────────────────────────────────────

A match means the user's addition is essentially the same point as one of the
key questions above — same topic, possibly different wording.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched": true or false, "matched_index": integer or null, "confidence": float}}
"""

ADD_MATCH_THRESHOLD = 0.75

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


class HITLAgent(BlackBoxAgent):
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
        self.session_id    = create_session(user_id, agent_type="hitl")
        self.original_case = get_case("hitl")
        self._pending      = False
        self.turn_count    = 0

        # ── Phase sequence: warmup → clarification → main ──────────────────
        self.phase               = "warmup"
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
        self.excluded_concepts    = []
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
        self.user_sub_points = {}

        # ── Deterministic 2-of-3 justification ─────────────────────────────
        # The shown pillars whose decision requires justification this session.
        # Populated in _build_walkthrough_concepts. Change log: 2026-05-29
        self.justification_pillars = set()

        # ── Pending confirmation state ─────────────────────────────────────
        self.pending_excl     = None
        self.pending_sub_excl = None             # (concept, bullet) staged for removal

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
        k = min(2, len(shown))
        self.justification_pillars = set(random.sample(shown, k)) if shown else set()

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

    def _classify_intent(self, user_input: str) -> dict:
        """suggestion | guidance | sub_point (with parent)."""
        try:
            result = classify_json(
                    f"You are classifying a user response in a case interview session.\n\n"
                    f"The user was asked: 'What's your suggestion for the next area to "
                    f"explore, or would you like me to guide you?'\n\n"
                    f"User response: \"{user_input}\"\n\n"
                    f"Reply with JSON only, no markdown, no explanation:\n"
                    f"{{\"type\": \"suggestion\" or \"guidance\" or \"sub_point\", "
                    f"\"concept\": \"extracted noun phrase or null\", "
                    f"\"parent\": \"parent concept or null\"}}\n\n"
                    f"type=sub_point when user names BOTH a new thing AND an existing parent:\n"
                    f"- 'add data privacy under Feasibility' → concept='data privacy', parent='Feasibility'\n"
                    f"- 'X should go under Y' → concept=X, parent=Y\n"
                    f"CRITICAL for sub_point: concept = the NEW thing, parent = the EXISTING concept after 'under'/'as part of'/'within'\n\n"
                    f"type=suggestion when user names a concept with NO parent:\n"
                    f"- 'how about budget' → concept='budget', parent=null\n"
                    f"- Names a specific business/analytical area as a noun phrase\n\n"
                    f"type=guidance ALWAYS for:\n"
                    f"- Questions: 'anything else?', 'what's next?'\n"
                    f"- Deferrals: 'you decide', 'take a lead', 'continue', 'proceed'\n"
                    f"- Uncertainty: 'I don't know', 'not sure', 'guide me'\n"
                    f"- Affirmations: 'ok', 'sure', 'yes', 'fine'\n"
                    f"- Verb phrases without a clear business noun\n\n"
                    f"WHEN IN DOUBT: use guidance."
            )
            logging.info(f"[INTENT] classified: {result}")
            return result
        except Exception as e:
            logging.warning(f"[INTENT] classifier failed: {e} — defaulting to guidance")
            return {"type": "guidance", "concept": None, "parent": None}

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

    def _match_pillar(self, item: str) -> str | None:
        """LLM: does a suggested area match any framework pillar (shown OR withheld)?
        Returns the matched pillar name, or None. Change log: 2026-05-31
        (was _match_withheld_pillar — widened to all pillars so suggestions overlapping
        an already-shown pillar resolve consistently, not only withheld ones.)"""
        pillars = kb.get_all_pillars()
        if not pillars:
            return None
        pillars_block = "\n".join(f"- {p['name']}" for p in pillars)
        prompt = ADD_PILLAR_MATCH_PROMPT.format(item=item, pillars=pillars_block)
        try:
            parsed = classify_json(prompt)
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD):
                return parsed.get("matched_pillar")
        except Exception as e:
            logging.warning(f"[PILLAR MATCH] error: {e}")
        return None

    def _match_key_question(self, item: str, pillar_name: str) -> str | None:
        """
        LLM: does item match a key question in this pillar?
        Returns the matched key-question text with inline [a][b] refs STRIPPED
        (HITL suppresses sources), or None. Change log: 2026-05-29
        """
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == pillar_name.lower()),
            None
        )
        if pillar is None:
            return None
        key_questions = pillar.get("key_questions", [])
        if not key_questions:
            return None
        kq_block = "\n".join(f"{i}. {q}" for i, q in enumerate(key_questions))
        prompt   = ADD_MATCH_PROMPT.format(
            item=item, pillar=pillar_name, key_questions=kq_block
        )
        try:
            parsed = classify_json(prompt)
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD):
                idx = parsed.get("matched_index")
                if idx is not None and 0 <= idx < len(key_questions):
                    # Strip inline source refs — no dangling markers
                    return _strip_source_refs(key_questions[idx])
        except Exception as e:
            logging.warning(f"[ADD MATCH] error: {e}")
        return None

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
            out = _strip_source_refs(self._strip_fences(response.text)).strip().strip("-• ").rstrip(".")
            if out:
                logging.info(f"[SUB-POINT FORMAT] '{item}' → '{out}'")
                return out
        except Exception as e:
            logging.warning(f"[SUB-POINT FORMAT] error: {e} — keeping raw")
        return item.strip()

    def _store_sub_point(self, pillar: str, item: str, modality: str = "text") -> tuple[str, bool]:
        """
        Match item to a key question (cleaned, source-free) or keep raw; store under
        pillar in user_sub_points. Returns (stored_text, is_new) — is_new is False if
        an identical point is already recorded (e.g. two phrasings matching the same
        key question), so callers can avoid duplicates. Change log: 2026-05-29
        """
        pillar  = self._normalize_pillar(pillar)
        matched = self._match_key_question(item, pillar)
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
        log_concept_added(self.session_id, item)
        log_add_sub_bullet(self.session_id, stored, modality)
        logging.info(f"[SUB-POINT] '{item}' → '{stored}' under '{pillar}'")
        return stored, True

    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 0. ➕ Add mode — collect sub-points, re-render, stay open ──────
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
            if not self._is_substantive_justification(user_input):
                log_user_message(self.session_id, f"[JUSTIFICATION:retry] {user_input}")
                yield "Could you say a bit more about your reasoning? A sentence is plenty.\n\n"
                return  # stay in awaiting_justification

            log_user_message(self.session_id, f"[JUSTIFICATION:{self.justification_for}] {user_input}")
            logging.info(f"[JUSTIFICATION] collected for={self.justification_for}: '{user_input}'")
            self.awaiting_justification = False
            self.justification_for      = None
            yield from self._stream_justification_ack()
            return

        # ── 2. Proactive suggestion handling ───────────────────────────────
        if self.awaiting_user_suggestion:
            self.awaiting_user_suggestion = False
            log_user_message(self.session_id, f"[PROACTIVE RESPONSE] {user_input}")

            intent = self._classify_intent(user_input)

            if intent["type"] == "guidance":
                logging.info(f"[PROACTIVE] user chose guidance")
                yield from self._stream_concept(is_first=False)
                return

            if intent["type"] == "sub_point" and intent.get("concept") and intent.get("parent"):
                concept = intent["concept"]
                parent  = intent["parent"]
                logging.info(f"[PROACTIVE] sub-point: '{concept}' → '{parent}'")
                yield from self._add_sub_point(parent, concept)
                yield from self._stream_concept_qa()  # stay on current concept
                return

            # suggestion — resolve to a target concept (existing / withheld pillar /
            # genuinely new) and NEVER insert a duplicate. Change log: 2026-05-29
            concept = intent.get("concept") or user_input.strip()
            dup     = self._check_duplicate_proactive(concept)

            if dup["is_duplicate"] and dup["matched_concept"]:
                target           = dup["matched_concept"]
                matched_withheld = None
            else:
                matched_withheld = self._match_pillar(concept)
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
                        log_add_pillar(self.session_id, target, "text")
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

        # ── 3. Override detection ──────────────────────────────────────────
        just_added_concept = None
        override = self._detect_override(user_input)

        # ── 3b. Explicit removal of the wrong concept ──────────────────────
        if (override and
                override["type"] == "concept_excluded" and
                override.get("detail") and
                not self.concept_swap.is_detected and
                self.swap_presented):
            wrong = self.concept_swap.config["wrong_concept"]
            if (self.concept_swap.matches(override["detail"]) or
                    self.concept_swap.matches(user_input)):
                self.concept_swap.force_detected()
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)
                self.walkthrough_index += 1
                log_memory_override(
                    self.session_id,
                    old_context=f"included: {wrong}",
                    new_context=f"user removed wrong concept explicitly: {wrong}",
                )
                logging.info(f"[CONCEPT SWAP] detected via explicit removal")

        # ── 4. Swap detection — only if presented AND no override ──────────
        cs_detected = False
        if self.swap_presented and not override:
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

        # ── 4b. Invariant check ────────────────────────────────────────────
        if (self.walkthrough_active
                and self.walkthrough_index > self.swap_position
                and not self.swap_presented):
            logging.error(f"[INVARIANT] rewinding to swap_position={self.swap_position}")
            self.walkthrough_index = self.swap_position

        # ── 5. Override handling ───────────────────────────────────────────
        if override:
            if override["type"] != "concept_added":
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
                self.user_sub_points           = {}
                self.justification_pillars     = set()
                self.swap_presented            = False
                self.swap_position             = 0
                self.pending_excl              = None
                self.pending_sub_excl          = None
                self.awaiting_user_suggestion  = False
                self.awaiting_justification    = False
                self.justification_for         = None
                self.awaiting_sub_point        = False
                self.awaiting_revisit_add      = False
                self.revisit_target            = None
                self.prompt_index              = 0
                self.ack_index                 = 0
                self.user_contributed_concepts = set()
                self.navigated_pillars         = set()
                if self.concept_swap.is_detected:
                    self.history = self._strip_concept_swap_from_history()
                yield "Noted — let me start the walkthrough fresh.\n\n"
                log_user_message(self.session_id, "[REDO TRIGGERED]")

            elif override["type"] == "concept_added" and override.get("detail"):
                new_concept = override["detail"]
                parent      = override.get("parent")
                if parent:
                    logging.info(f"[CONCEPT ADDED] sub-point: '{new_concept}' → '{parent}'")
                    yield from self._add_sub_point(parent, new_concept)
                else:
                    dup = self._check_duplicate(new_concept, self.walkthrough_concepts)
                    if dup["is_duplicate"]:
                        logging.info(
                            f"[CONCEPT ADDED] sub-point: '{new_concept}' → '{dup['matched_concept']}'"
                        )
                        yield from self._add_sub_point(dup["matched_concept"], new_concept)
                    else:
                        # Match alternate wording to a withheld pillar so it renders
                        # real KB content when reached — consistent with the proactive
                        # path. Change log: 2026-05-29
                        matched_withheld = self._match_pillar(new_concept)
                        resolved  = matched_withheld or new_concept
                        insert_at = self.walkthrough_index + 1
                        self.walkthrough_concepts.insert(insert_at, resolved)
                        log_concept_added(self.session_id, resolved)
                        log_add_pillar(self.session_id, resolved, "text")
                        logging.info(
                            f"[CONCEPT ADDED] '{new_concept}' inserted as "
                            f"'{resolved}' at index={insert_at}"
                        )
                        just_added_concept = resolved

        # ── 6. Log and append user message ────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 7. Routing log ─────────────────────────────────────────────────
        logging.info(
            f"[ROUTE] active={self.walkthrough_active}, done={self.walkthrough_done}, "
            f"swap_presented={self.swap_presented}, index={self.walkthrough_index}, "
            f"cs_detected={cs_detected}"
        )

        # ── 8. Route ───────────────────────────────────────────────────────
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
            if override is None:   # pure question/comment, not a steering action
                current  = self._current_concept()
                _on_swap = (self.swap_presented
                            and not self.concept_swap.is_detected
                            and current is not None
                            and self._is_wrong_concept(current))
                # Parity with BlackBox/Explainable: a typed turn landing in Q&A IS a
                # question — log it unconditionally. swap_questioned is a subset
                # logged on top (Finding A). Change log: 2026-06-02
                log_question(self.session_id, "text", detail=user_input[:200])
                if _on_swap:
                    log_swap_questioned(self.session_id, "text", detail=user_input[:200])
            yield from self._stream_concept_qa(just_added=just_added_concept)

    # ══════════════════════════════════════════════════════════════════════
    # Proactive prompt + justification
    # ══════════════════════════════════════════════════════════════════════

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
        log_concept_added(self.session_id, concept)
        log_add_pillar(self.session_id, concept, "text")
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
        pillar = self._normalize_pillar(matched_concept)
        stored, is_new = self._store_sub_point(pillar, sub_point)
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
            bullets = [_strip_source_refs(b) for b in bullets]
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
        """Q&A grounding: description + key-questions, source refs stripped
        (HITL suppresses sources). Change log: 2026-05-31"""
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == concept.lower()),
            None
        )
        if pillar:
            pts = [_strip_source_refs(q) for q in pillar.get("key_questions", [])]
            parts = []
            if pillar.get("description"):
                parts.append(pillar["description"])
            if pts:
                parts.append("\n".join(f"- {p}" for p in pts))
            return "\n\n".join(parts).strip()
        swap = kb.get_swap_concept()
        if swap and concept.lower() == self.concept_swap.config["wrong_concept"].lower():
            return "\n".join(f"- {_strip_source_refs(b)}" for b in swap.get("sub_bullets", []))
        return ""
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
        """
        Deterministic summary (no LLM). Renders from explicit session state:
        concepts the user advanced PAST, minus excluded (and the swap if detected),
        each shown with its stored bullets + user-added sub-points. No sources.
        Change log: 2026-05-29 — deterministic; renders concept_blocks + user_sub_points
        """
        self.walkthrough_done = True

        wrong = self.concept_swap.config["wrong_concept"].lower()
        if self.concept_swap.is_detected:
            excluded_lower = [e.lower() for e in self.excluded_concepts] + [wrong]
        else:
            excluded_lower = [e.lower() for e in self.excluded_concepts]

        # Only concepts the user advanced PAST (deliberately decided), de-duplicated
        included = []
        seen = set()
        for c in self.walkthrough_concepts[:self.walkthrough_index]:
            cl = c.lower()
            if cl in excluded_lower or cl in seen:
                continue
            seen.add(cl)
            included.append(c)
        logging.info(f"[SUMMARY] included={included}")

        lines = ["**Final Framework Summary**", ""]
        for c in included:
            lines.append(f"**{c}**")
            block = self.concept_blocks.get(c, "").strip()
            if block:
                lines.append(block)
            for sp in self.user_sub_points.get(c, []):
                lines.append(f"- {sp}")
            lines.append("")

        summary_text = "\n".join(lines).rstrip()

        self.history.append(
            types.Content(role="model", parts=[types.Part(text=summary_text)])
        )
        update_answer(self.session_id, summary_text)
        log_agent_response(self.session_id, summary_text)

        yield summary_text

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
        self.walkthrough_index += 1

        if concept in self.justification_pillars:
            yield from self._stream_justification_prompt("accept", concept=concept)
        else:
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

    def on_reject_concept(self):
        concept = self._current_concept()
        if concept is None:
            return

        self.pending_excl = concept

        logging.info(f"[REJECT] pending set for concept='{concept}'")
        if not self._is_wrong_concept(concept):
            log_delete(self.session_id, concept, "button")   # intent (#1); swap stays detection (#3)
        yield (
            f"Are you sure you want to skip **{concept}**?\n\n"
            f"*Use the buttons below to confirm.*"
        )

    def on_confirm_reject(self):
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

        if concept in self.justification_pillars:
            yield from self._stream_justification_prompt("reject", concept=concept)
        else:
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

    def on_cancel_reject(self):
        concept = self.pending_excl
        self.pending_excl = None
        logging.info(f"[REJECT CANCELLED] concept='{concept}' kept in framework")

        if concept and concept not in self.approved_concepts:
            self.approved_concepts.append(concept)

        yield f"Keeping **{concept}** — let's continue.\n\n"
        self.walkthrough_index += 1

        if concept in self.justification_pillars:
            yield from self._stream_justification_prompt("accept", concept=concept)
        else:
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._walkthrough_complete_message()
            else:
                yield from self._stream_proactive_prompt()

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

    # ── ➖ Remove a point: picker → confirm → commit ───────────────────────
    def on_remove_point(self, bullet: str):
        concept = self._current_concept()
        if concept is None or not bullet:
            return
        self.pending_sub_excl = (concept, bullet)
        logging.info(f"[REMOVE POINT] pending: '{bullet}' under '{concept}'")
        yield (
            f"Remove this point from **{concept}**?\n\n- {bullet}\n\n"
            f"*Use the buttons below to confirm.*"
        )

    def on_confirm_remove_point(self):
        if not self.pending_sub_excl:
            return
        concept, bullet = self.pending_sub_excl
        self.pending_sub_excl = None
        target = bullet.strip().lstrip("-• ").strip().lower()

        block = self.concept_blocks.get(concept, "")
        if block:
            self.concept_blocks[concept] = "\n".join(
                l for l in block.splitlines()
                if l.strip().lstrip("-• ").strip().lower() != target
            )
        pts = self.user_sub_points.get(concept, [])
        self.user_sub_points[concept] = [p for p in pts
                                         if p.strip().lower() != target]

        log_delete(self.session_id, bullet, "button")   # logged at CONFIRM
        logging.info(f"[REMOVE POINT CONFIRMED] '{bullet}' from '{concept}'")
        yield f"Done — removed that point from **{concept}**.\n\n"

    def on_cancel_remove_point(self):
        concept = self.pending_sub_excl[0] if self.pending_sub_excl else "this concept"
        self.pending_sub_excl = None
        logging.info(f"[REMOVE POINT CANCELLED] keeping point under '{concept}'")
        yield f"No problem — keeping that point in **{concept}**.\n\n"

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
    # UI state queries
    # ══════════════════════════════════════════════════════════════════════

    def should_show_buttons(self) -> bool:
        return (
            self.phase == "main"
            and self.walkthrough_active
            and not self.walkthrough_done
            and self._current_concept() is not None
            and self.pending_excl is None
            and self.pending_sub_excl is None
            and not self.awaiting_user_suggestion
            and not self.awaiting_justification
            and not self.awaiting_sub_point
            and not self.awaiting_revisit_add
            and (self._current_concept() not in self.user_contributed_concepts)
        )

    def should_show_confirmation_buttons(self) -> bool:
        return self.pending_excl is not None

    def should_show_remove_point_confirmation(self) -> bool:
        return self.pending_sub_excl is not None

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