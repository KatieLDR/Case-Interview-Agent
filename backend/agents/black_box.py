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
from backend.agents.base import BaseAgent          # Step 6a: shared turn engine (F-ARCH2)

# ── Model config ───────────────────────────────────────────────────────────
# MAIN_MODEL / CLASSIFIER_MODEL imported from backend.llm (REFACTOR_PLAN §S Step 1)

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "AI Implementation"

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════


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


# OVERRIDE_CLASSIFIER_PROMPT retired (Step 3 dead-code pass): _detect_override is
# gone; the unified router (interaction/intents.py) owns steering classification.

# ══════════════════════════════════════════════════════════════════════════
# Warm-up content
# Change log: 2026-05-05 — redesigned warm-up
# Change log: 2026-05-06 — added "Let's go" cue
# Change log: 2026-05-22 — pre-built plan, LLM merge
# ══════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════
# Thresholds
# ══════════════════════════════════════════════════════════════════════════
# ANSWER_THRESHOLD imported from backend.llm (§S Step 1); OVERRIDE_THRESHOLD retired (Step 3)

# Light cancel-escape phrases for the "which part to remove?" prompt (no LLM).
_CANCEL_PHRASES = {
    "never mind", "nevermind", "cancel", "stop", "forget it", "forget about it",
    "no", "nope", "no thanks", "nothing", "none", "leave it", "leave it alone",
    "keep it", "keep everything", "actually no", "skip", "skip it", "dont", "don't",
}

# ── Deterministic add-flow matchers (source-free) — Change log: 2026-06-01 ──

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

class BlackBoxAgent(BaseAgent):
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self._init_flow_state()
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self.has_main_contribution = False   # gates End Session button (app.py reads this)
        # ── Deterministic framework state — Change log: 2026-06-01 ─────────
        self.user_added_pillars = []    # user-added pillar names (order preserved)
        # ── Step 4b: shared HandlerSession flow state (replaces pending_excl /
        #    pending_sub_excl / awaiting_removal_target). interaction/handlers.py
        #    owns the two-turn removal loop (PendingAction) now. ─────────────
        self._ack_index         = 0     # rotates the no-reprint acknowledgements
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



    # ══════════════════════════════════════════════════════════════════════
    # Warm-up messages
    # ══════════════════════════════════════════════════════════════════════



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


    # ══════════════════════════════════════════════════════════════════════
    # Clarification phase streaming
    # ══════════════════════════════════════════════════════════════════════


    # ══════════════════════════════════════════════════════════════════════
    # Framework presentation
    # Change log: 2026-05-01
    # Change log: 2026-05-12 — removed debug print
    # ══════════════════════════════════════════════════════════════════════


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
                and self.concept_swap.is_injected and intent not in ("add", "remove", "question")):
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



    def _last_agent_message(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts and c.parts[0].text:
                return c.parts[0].text[:600]
        return ""


    def _store_sub_point(self, pillar: str, item: str, modality: str = "text"):
        pillar  = self._normalize_pillar(pillar)
        matched, _ = matching.match_key_question(item, pillar)
        stored  = matched if matched else self._format_sub_bullet(item)
        # Already a shown static bullet on this pillar (incl. KB-backed user pillars)
        # → already in the framework, don't duplicate. Change log: 2026-06-02
        kbp = next((p for p in kb.get_all_pillars() if p["name"].lower() == pillar.lower()), None)
        if any(stored.lower() == grounding._strip_source_refs(b).lower()
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
        from backend.knowledge import knowledge_base as kb
        for p in kb.get_all_pillars():
            if p["name"].lower() == name.lower():
                return p["name"]
        for p in self.user_added_pillars:
            if p.lower() == name.lower():
                return p
        return name

    # _match_* wrappers collapsed (Step 6c): call domain.matching.* directly (I-3).

    def _format_sub_bullet(self, item: str) -> str:
        try:
            r = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=SUB_BULLET_FORMAT_PROMPT.format(item=item),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            out = grounding._strip_source_refs(self._strip_fences(r.text)).strip().strip("-• ").rstrip(".")
            if out:
                return out
        except Exception as e:
            print(f"[SUB-POINT FORMAT] error: {e}")
        return item.strip()

    # ── Redo / Q&A / ack / summary ─────────────────────────────────────────



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
            bl = [grounding._strip_source_refs(b) for b in kb_bullets
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




    # ── gate-reply question check (W9 / _stream_confirm_qa render) ──────────


    # ── §3.6 event firing (Step 5) ──────────────────────────────────────────
    # BlackBox never elicits contributions (no proactive suggestion gate), so its
    # source is always user_spontaneous; modality is always text. agent_type is read
    # from concept_swap so the inherited helper resolves correctly in EXP/HITL too.



    # ── outcome renderer ────────────────────────────────────────────────────
    def render_question(self, user_input):
        yield from self._stream_qa(user_input)

    def render_framework(self, preamble=""):
        yield from self._yield_rerender(preamble)

    def render_summary(self):
        yield from self._stream_summary()

    def render_fallback(self, outcome=None):
        """6h: None (nothing left to suggest) -> the terse 'surfaced main areas' line;
        AdvanceOutcome / FallbackOutcome -> ack with no reprint (W5). Persona-preserving."""
        if outcome is None:
            msg = ("You've surfaced the main areas I'd flag — feel free to add, "
                   "remove, or question any part of what's here.")
            self._emit(msg); yield msg; return
        yield from self._ack_no_reprint()

    def render_add(self, o):
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

    def render_removal(self, o, user_input, *, was_pending=False, pa=None):
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

    def render_next_steps(self, o):
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



        

    # ══════════════════════════════════════════════════════════════════════
    # History helpers
    # ══════════════════════════════════════════════════════════════════════



    # ══════════════════════════════════════════════════════════════════════
    # Core streaming utility — used by ExplainableAgent and HITLAgent
    # Change log: 2026-04-09 — moved up from ExplainableAgent.
    # ══════════════════════════════════════════════════════════════════════


    # ══════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════

