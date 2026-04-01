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
    log_framework_switched,
)
from backend import knowledge_graph as kg
from backend.rag_explainer import build_citation_header, check_and_append_warning, build_and_check_citation

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "Profitability"

# ── Swap injection position ───────────────────────────────────────────────
# Computed dynamically as len(base_concepts) // 2 (true midpoint) so the
# swap always appears in the middle of whatever framework is active.
# This keeps injection timing consistent even when the user switches framework.
# Change log: 2026-03-29 — replaced SwapState enum with SWAP_POSITION constant.
# Change log: 2026-03-29 — replaced fixed SWAP_POSITION=3 with dynamic
#   then to len // 2 (true midpoint) — cleaner placement.

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

SINGLE_CONCEPT_PROMPT = """
You are a strategic consultant walking a user through a framework ONE concept at a time.

The concept name and rationale have already been presented to the user above.
YOUR ONLY JOB: Output the sub-bullets and closing question — nothing else.

EXACT OUTPUT FORMAT — copy this structure precisely:

- [sub-bucket, 5–7 words, specific to this case]
- [sub-bucket, 5–7 words, specific to this case]
- [sub-bucket, 5–7 words, specific to this case]

*Shall we move on to the next concept?*

─── EXAMPLE (for Price per Unit) ────────────────────────────────────────────
- Current market rates for industrial Silica Sand
- Global Bentonite pricing trends and forecasts
- Negotiating power with potential buyers

*Shall we move on to the next concept?*
─────────────────────────────────────────────────────────────────────────────

─── STRICT RULES ────────────────────────────────────────────────────────────
- Do NOT repeat the concept name as a header — it is already shown above
- Do NOT write a rationale sentence — it is already shown above
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

Answer in 2–3 sentences. Stay grounded in the current case context.
Plain language only — no jargon, no technical terms.

After answering, use the closing specified in CLOSING INSTRUCTION below.
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
3. One short closing sentence confirming you are moving on — then STOP

STRICT RULES:
- Do NOT end with a question
- Do NOT say "shall we move on"
- Do NOT introduce, name, or preview the next concept
- Do NOT present any new framework bucket in this response
- Your response ends after the closing sentence — next concept follows separately
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

ADVANCE_THRESHOLD = 0.75

# ── Framework resolver prompt ──────────────────────────────────────────────
FRAMEWORK_RESOLVER_PROMPT = """
You are a classifier for a case interview tool.

The user mentioned they want to use a specific framework. Your job is to match
their mention to the available frameworks listed below.

Return a JSON list of framework names that match. Rules:
- Include a framework if the user's mention clearly refers to it
- Include a framework if the user's mention is a reasonable synonym or abbreviation
- If two frameworks are plausibly intended, include both
- If nothing matches, return an empty list []
- Do NOT invent framework names — only return names from the list below

Respond ONLY with a valid JSON array of strings, no explanation, no markdown:
["Framework Name"] or ["Name 1", "Name 2"] or []

Examples (given frameworks: Economic Feasibility, Expanded Profit Formula,
Four-Pronged Strategy, Formulaic Breakdown, Customized Issue Trees):
- "pricing" → ["Four-Pronged Strategy"]
- "profitability" → ["Expanded Profit Formula"]
- "market entry" → ["Economic Feasibility"]
- "cost framework" → ["Expanded Profit Formula"]
- "guesstimate" → ["Formulaic Breakdown"]
- "issue tree" → ["Customized Issue Trees"]
- "profit and pricing" → ["Expanded Profit Formula", "Four-Pronged Strategy"]
- "blockchain framework" → []
- "supply chain" → []
"""

# ── Walkthrough-aware override classifier prompt ───────────────────────────
# Change log: 2026-04-01 — Option C fix applied.
# Added {concepts} placeholder injected at call site with self.walkthrough_concepts.
# Added explicit rule: concept names in discussion context → "none".
# Added 4 new negative examples covering "let's discuss X", "can we talk about X" etc.
# Root cause fixed: classifier had no awareness of which names were concepts vs
# frameworks, so "let's discuss Variable Cost" was firing framework_switch.
WALKTHROUGH_OVERRIDE_PROMPT = """
You are a classifier for a case interview walkthrough tool.

The agent is currently walking the user through a framework one concept at a time.
Determine whether the user's message is attempting to steer or change the agent's output.

If yes, classify the type:
- "redo"               : user wants to RESTART the entire walkthrough from scratch
                         ("start over", "restart", "begin again", "let's start fresh",
                          "redo this", "try again from the beginning")
- "concept_excluded"   : user wants to remove a specific concept
                         ("remove X", "exclude X", "don't include X", "skip X",
                          "I don't want X", "drop X")
- "framework_switch"   : user wants to use a specific different FRAMEWORK
                         ("use Market Entry framework", "switch to profitability")
- "none"               : not steering the output

─── ACTIVE WALKTHROUGH CONCEPTS ──────────────────────────────────────────────
{concepts}
──────────────────────────────────────────────────────────────────────────────
CRITICAL — if the user mentions any name from the ACTIVE WALKTHROUGH CONCEPTS
list above in a discussion context ("let's discuss X", "can we talk about X",
"what about X", "tell me more about X") → classify as "none".
Only classify as "framework_switch" if the user explicitly names a FRAMEWORK,
not a concept from the list above.

CRITICAL — these are NOT redo during an active walkthrough:
- "move on" → none (means advance to next concept)
- "next" → none
- "continue" → none
- "yes" / "ok" / "sure" → none
- "let's move on" → none
- "proceed" / "got it" → none

Respond ONLY with valid JSON, no explanation, no markdown:
{{"override": true or false, "type": "redo"|"concept_excluded"|"framework_switch"|"none", "detail": string or null, "confidence": float}}

Examples (assuming concepts include: Volume, Price per Unit, Variable Cost per Unit, Fixed Cost):
- "start over" → {{"override": true, "type": "redo", "detail": null, "confidence": 0.99}}
- "restart from scratch" → {{"override": true, "type": "redo", "detail": null, "confidence": 0.98}}
- "move on" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "let's move on to the next one" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "next concept please" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "remove Variable Cost per Unit" → {{"override": true, "type": "concept_excluded", "detail": "Variable Cost per Unit", "confidence": 0.97}}
- "use Market Entry framework" → {{"override": true, "type": "framework_switch", "detail": "Market Entry", "confidence": 0.96}}
- "yes" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "makes sense, continue" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "let's discuss Variable Cost per Unit" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "can we talk about Price per Unit?" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "what about Volume?" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "tell me more about Fixed Cost" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
"""


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
        logging.info(f"[KG INIT] case_type={CASE_TYPE}, "
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
        # swap_presented: True once the wrong concept block has been streamed to user.
        # Detection only activates after this is True.
        self.swap_presented       = False
        # swap_position: computed as len(concepts) // 2 (true midpoint) when walkthrough is built.
        # Recomputed on framework switch. Stored for invariant check and semantic logs.
        self.swap_position        = 0

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

    def start_main_phase(self) -> str:
        """
        Override to append a summary hint to the phase transition message.
        Change log: 2026-03-31 — added summary hint.
        """
        base_message = super().start_main_phase()
        return (
            f"{base_message}\n\n"
            f"💡 *You can click **📊 Get Summary & End Session** at any time "
            f"to see your full framework and close the session.*"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Walkthrough state helpers
    # ══════════════════════════════════════════════════════════════════════

    def _build_walkthrough_concepts(self) -> list:
        """
        Build ordered concept list for current framework.
        Inserts wrong concept at len(base) // 2 (true midpoint).
        """
        base     = list(self.kg_context["concepts"])
        wrong    = self.concept_swap.config["wrong_concept"]
        position = len(base) // 2
        base.insert(position, wrong)
        self.swap_position = position
        logging.info(f"[WALKTHROUGH] built={base}, swap_position={position}, "
                     f"framework={self.kg_context['framework']}")
        return base

    def _rebuild_walkthrough_on_framework_switch(self, from_framework: str = "") -> None:
        to_framework = self.kg_context["framework"]

        self.walkthrough_concepts = self._build_walkthrough_concepts()
        self.walkthrough_index    = 0
        self.excluded_concepts    = []

        log_framework_switched(
            session_id     = self.session_id,
            from_framework = from_framework,
            to_framework   = to_framework,
            switch_index   = self.walkthrough_index,
        )

        if self.concept_swap.is_detected:
            wrong = self.concept_swap.config["wrong_concept"]
            self.excluded_concepts.append(wrong)
            self.swap_presented = True
            logging.info(f"[FRAMEWORK SWITCH] swap already caught — "
                         f"excluded from new walkthrough, framework={to_framework}")
        else:
            self.swap_presented = False
            logging.info(f"[FRAMEWORK SWITCH] swap not yet caught — "
                         f"re-injecting at position={self.swap_position}, "
                         f"framework={to_framework}")

    def _current_concept(self) -> str | None:
        excluded_lower = [e.lower() for e in self.excluded_concepts]
        while self.walkthrough_index < len(self.walkthrough_concepts):
            concept = self.walkthrough_concepts[self.walkthrough_index]
            if concept.lower() not in excluded_lower:
                return concept
            logging.debug(f"[WALKTHROUGH] skipping excluded: {concept}, index={self.walkthrough_index}")
            self.walkthrough_index += 1
        return None

    def _is_wrong_concept(self, concept: str) -> bool:
        return concept.lower() == self.concept_swap.config["wrong_concept"].lower()

    def _is_ready_to_advance(self, user_input: str) -> bool:
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
            logging.info(f"[ADVANCE] advance={parsed.get('advance')}, confidence={parsed.get('confidence'):.2f}, proceed={result}")
            return result
        except Exception as e:
            logging.warning(f"[ADVANCE] classifier error: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    # Framework resolver — LLM-based, KG-grounded
    # ══════════════════════════════════════════════════════════════════════

    def _fetch_kg_context_by_framework(self, framework_name: str) -> dict | None:
        try:
            concepts = kg.get_ordered_concepts(framework_name)
            if not concepts:
                return None
            all_fw = kg.get_all_frameworks()
            case_type = next(
                (f["case_type"] for f in all_fw if f["framework"] == framework_name),
                "Unknown"
            )
            return {"case_type": case_type, "framework": framework_name, "concepts": concepts}
        except Exception as e:
            logging.warning(f"[KG] _fetch_kg_context_by_framework error: {e}")
            return None

    def _resolve_framework(self, user_mention: str) -> list[str]:
        try:
            if not hasattr(self, "_kg_all_frameworks") or not self._kg_all_frameworks:
                self._kg_all_frameworks = kg.get_all_frameworks()
                logging.debug(f"[KG CACHE] loaded {len(self._kg_all_frameworks)} frameworks")
            frameworks = self._kg_all_frameworks
            if not frameworks:
                logging.warning("[RESOLVER] no frameworks returned from KG")
                return []

            fw_list = "\n".join(
                f"- {f['framework']} (case: {f['case_type']}): {f['description']}"
                for f in frameworks
            )

            prompt = (
                f"{FRAMEWORK_RESOLVER_PROMPT}\n\n"
                f"─── AVAILABLE FRAMEWORKS ─────────────────────────────────────────────\n"
                f"{fw_list}\n"
                f"──────────────────────────────────────────────────────────────────────\n\n"
                f"User mentioned: \"{user_mention}\""
            )

            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=prompt,
            )
            matches = json.loads(self._strip_fences(response.text))
            if not isinstance(matches, list):
                matches = []

            valid_names = {f["framework"] for f in frameworks}
            matches = [m for m in matches if m in valid_names]

            logging.info(f"[RESOLVER] user_mention='{user_mention}' → matches={matches}")
            return matches

        except Exception as e:
            logging.warning(f"[RESOLVER] error: {e}")
            return []

    def _format_framework_list(self) -> str:
        try:
            frameworks = kg.get_all_frameworks()
            return ", ".join(f["framework"] for f in frameworks)
        except Exception:
            return "the available frameworks"

    # ══════════════════════════════════════════════════════════════════════
    # Override detection — walkthrough-aware
    # Change log: 2026-04-01 — Option C fix:
    #   inject self.walkthrough_concepts into WALKTHROUGH_OVERRIDE_PROMPT
    #   via .format(concepts=concepts_str) so classifier can distinguish
    #   concept names from framework names.
    # ══════════════════════════════════════════════════════════════════════

    def _detect_override(self, user_input: str) -> dict | None:
        import json as _json
        from backend.black_box_agent import OVERRIDE_THRESHOLD as _THRESH

        if not self.walkthrough_active or self.walkthrough_done:
            return super()._detect_override(user_input)

        # Inject active concept list so classifier distinguishes
        # concept names from framework names — Option C fix
        concepts_str = ", ".join(self.walkthrough_concepts) \
                       if self.walkthrough_concepts else "none"
        prompt = WALKTHROUGH_OVERRIDE_PROMPT.format(concepts=concepts_str)

        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{prompt}\n\nUser message: \"{user_input}\"",
            )
            parsed = _json.loads(self._strip_fences(response.text))
            if (parsed.get("override", False) and
                    parsed.get("confidence", 0.0) >= _THRESH and
                    parsed.get("type", "none") != "none"):
                return {
                    "type":       parsed["type"],
                    "detail":     parsed.get("detail"),
                    "confidence": parsed["confidence"],
                }
        except Exception as e:
            logging.warning(f"[OVERRIDE] walkthrough classifier error: {e}")
        return None

    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 1. Swap detection ──────────────────────────────────────────────
        cs_detected = False
        if self.swap_presented:
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
                self.walkthrough_index += 1
                logging.info(f"[SWAP] caught — index→{self.walkthrough_index}, "
                             f"swap_position={self.swap_position}")
        elif self.walkthrough_active:
            logging.debug(f"[SWAP] detection skipped — swap not yet presented, "
                          f"index={self.walkthrough_index}, swap_position={self.swap_position}")

        # ── 1b. Invariant check ────────────────────────────────────────────
        if (self.walkthrough_active
                and self.walkthrough_index > self.swap_position
                and not self.swap_presented):
            logging.error(f"[INVARIANT VIOLATION] index={self.walkthrough_index} "
                          f"past swap_position={self.swap_position} but swap_presented=False "
                          f"— rewinding to swap_position to force presentation")
            self.walkthrough_index = self.swap_position

        # ── 2. Override detection ──────────────────────────────────────────
        override = None if cs_detected else self._detect_override(user_input)
        if override:
            log_memory_override(
                self.session_id,
                old_context=f"override_type: {override['type']}",
                new_context=f"detail: {override['detail'] or 'n/a'}",
            )
            logging.info(f"[OVERRIDE] {override['type']} — {override['detail']}, "
                         f"index={self.walkthrough_index}, swap_presented={self.swap_presented}")

            if override["type"] == "redo":
                self.walkthrough_active   = False
                self.walkthrough_done     = False
                self.walkthrough_index    = 0
                self.walkthrough_concepts = []
                self.excluded_concepts    = []
                self.swap_presented       = False
                self.swap_position        = 0
                if self.concept_swap.is_detected:
                    self.history = self._strip_concept_swap_from_history()
                yield "Noted! Let me start the walkthrough fresh...\n\n"
                log_user_message(self.session_id, "[REDO TRIGGERED]")

            elif override["type"] == "framework_switch":
                if not override.get("detail"):
                    override = {"type": "framework_unspecified"}
                    logging.info("[FRAMEWORK SWITCH] no framework specified — asking user")
                else:
                    matches = self._resolve_framework(override["detail"])
                    if len(matches) == 1:
                        new_framework = matches[0]
                        new_context   = self._fetch_kg_context_by_framework(new_framework)
                        if new_context:
                            from_fw = self.kg_context["framework"]
                            self.kg_context = new_context
                            if self.walkthrough_active:
                                self._rebuild_walkthrough_on_framework_switch(from_framework=from_fw)
                                logging.info(f"[FRAMEWORK SWITCH] mid-walkthrough → {new_framework}")
                            else:
                                log_framework_switched(
                                    session_id     = self.session_id,
                                    from_framework = from_fw,
                                    to_framework   = new_framework,
                                    switch_index   = 0,
                                )
                                logging.info(f"[FRAMEWORK SWITCH] pre-walkthrough → {new_framework}")
                    elif len(matches) >= 2:
                        override = {"type": "framework_clarification", "matches": matches,
                                    "detail": override["detail"]}
                        logging.info(f"[FRAMEWORK SWITCH] ambiguous — asking user: {matches}")
                    else:
                        override = {"type": "framework_not_found", "detail": override["detail"]}
                        logging.info(f"[FRAMEWORK SWITCH] no match for '{override['detail']}'")

            elif override["type"] == "concept_excluded" and override.get("detail"):
                excl = override["detail"]
                if excl not in self.excluded_concepts:
                    self.excluded_concepts.append(excl)
                logging.info(f"[OVERRIDE] concept excluded: {excl}")

        # ── 3. Log and append user message ────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 4. Semantic routing log ────────────────────────────────────────
        logging.info(f"[ROUTE] active={self.walkthrough_active}, "
                     f"done={self.walkthrough_done}, "
                     f"swap_presented={self.swap_presented}, "
                     f"index={self.walkthrough_index}, "
                     f"swap_position={self.swap_position}, "
                     f"cs_detected={cs_detected}")

        # ── 5. Route ───────────────────────────────────────────────────────
        if not self.walkthrough_active:
            self.walkthrough_concepts = self._build_walkthrough_concepts()
            self.walkthrough_active   = True
            self.walkthrough_index    = 0
            self.swap_presented       = False
            yield from self._stream_concept(is_first=True)

        elif self.walkthrough_done:
            yield from self._stream_freeform(cs_detected)

        elif override and override["type"] == "framework_switch":
            yield from self._stream_concept(is_first=True)

        elif override and override["type"] == "framework_clarification":
            matches   = override["matches"]
            match_str = " or ".join(f"**{m}**" for m in matches)
            yield from self._stream_with_instruction(
                instruction=(
                    f"The user mentioned switching frameworks. Based on what they said, "
                    f"it could mean {match_str}. Ask them which one they meant in one "
                    f"short, friendly sentence. Do not present any concept yet."
                )
            )

        elif override and override["type"] == "framework_not_found":
            available = self._format_framework_list()
            detail    = override.get("detail") or "that framework"
            yield from self._stream_with_instruction(
                instruction=(
                    f"The user mentioned '{detail}' as a framework. "
                    f"Respond in 2 short sentences: "
                    f"(1) Say you don't recognise '{detail}' in your knowledge base "
                    f"— keep it brief and neutral, no apology. "
                    f"(2) Ask which of these they'd like to use instead: {available}. "
                    f"Do not present any concept yet. Do not offer to work with '{detail}'."
                )
            )

        elif override and override["type"] == "framework_unspecified":
            available = self._format_framework_list()
            yield from self._stream_with_instruction(
                instruction=(
                    f"The user wants to switch frameworks but hasn't named one. "
                    f"Ask in one friendly sentence: do they have a specific framework "
                    f"in mind? If not, suggest they try one of these: {available}. "
                    f"Do not present any concept yet."
                )
            )

        elif cs_detected:
            yield from self._stream_swap_caught()
            yield "\n\n"
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)

        elif self._is_ready_to_advance(user_input):
            self.walkthrough_index += 1
            concept = self._current_concept()
            if concept is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)

        else:
            yield from self._stream_concept_qa()

    # ══════════════════════════════════════════════════════════════════════
    # Streaming sub-methods
    # ══════════════════════════════════════════════════════════════════════

    def _stream_concept(self, is_first: bool):
        """
        Present the concept at current walkthrough_index.
        Change log: 2026-03-31 — citation integrated, source block before concept text.
        """
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
            f"Concept: **{concept}**\n"
            f"Output ONLY the sub-bullets and closing question for this concept.\n"
            f"Do NOT repeat the concept name. Do NOT write a rationale sentence.\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )

        task_injection = (
            f"[Output only the sub-bullets and closing question for **{concept}**. "
            f"Do not repeat the concept name or rationale — those are already shown above.]"
        )

        already_logged = self.concept_swap.is_injected if is_wrong else False

        # ── Step 1: Citation header + concept name + justification as prefix ─
        kg_data = None
        if not is_wrong:
            try:
                citation_header, kg_data = build_citation_header(
                    concept_name = concept,
                    framework    = self.kg_context["framework"],
                )
                lines         = citation_header.strip().split("\n")
                source_line   = lines[0] if len(lines) > 0 else ""
                justification = lines[1] if len(lines) > 1 else ""
                prefix = (
                    f"{source_line}\n\n"
                    f"**{concept}**\n"
                    f"{justification}\n"
                )
                if is_first:
                    prefix = f"Here is how I would structure the analysis:\n\n{prefix}"
            except Exception as e:
                logging.warning(f"[CITATION] header failed for concept='{concept}': {e}")
                prefix = f"**{concept}**\n"
                if is_first:
                    prefix = f"Here is how I would structure the analysis:\n\n{prefix}"
        else:
            prefix = f"**{concept}**\n"
            if is_first:
                prefix = f"Here is how I would structure the analysis:\n\n{prefix}"

        # ── Step 2: Stream concept block ───────────────────────────────────
        full_concept_reply = []
        for token in self._stream_with_instruction(
            instruction=instruction,
            prefix=prefix,
            task_injection=task_injection,
            track_swap=is_wrong,
            store_answer=False,
        ):
            full_concept_reply.append(token)
            yield token

        # ── Step 3: Swap tracking ──────────────────────────────────────────
        if is_wrong:
            self.swap_presented = True
            if not already_logged:
                self.concept_swap.log_presented()
                logging.info(f"[SWAP] concept presented at position={self.swap_position} — "
                             f"detection active from next turn")
            else:
                logging.info(f"[SWAP] concept re-presented after framework switch — "
                             f"log_presented() skipped (already logged), "
                             f"position={self.swap_position}")
            return

        # ── Step 4: Faithfulness check → warning if needed ────────────────
        if kg_data is not None:
            try:
                concept_block = "".join(full_concept_reply)
                warning = check_and_append_warning(concept, concept_block, kg_data)
                if warning:
                    yield warning
            except Exception as e:
                logging.warning(f"[FAITHFULNESS] warning check failed for '{concept}': {e}")

    def _stream_concept_qa(self):
        concept    = self._current_concept() or "the current concept"
        on_swap    = (self.walkthrough_index == self.swap_position
                      and self.swap_presented
                      and not self.concept_swap.is_detected)

        closing = (
            "End with exactly:\n"
            "*I can see why you'd question this — shall we include it or move on without it?*"
            if on_swap else
            "End with exactly:\n"
            "*Shall we move on to the next concept?*"
        )

        instruction = (
            f"{CONCEPT_QA_PROMPT}\n\n"
            f"─── CLOSING INSTRUCTION ──────────────────────────────────────────────\n"
            f"{closing}\n"
            f"─── CONTEXT ──────────────────────────────────────────────────────────\n"
            f"Current concept: **{concept}**\n"
            f"On swap concept: {on_swap}\n"
            f"Framework: {self.kg_context['framework']} | Case: {self.kg_context['case_type']}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )

        if on_swap:
            logging.info(f"[SWAP QA] user questioning swap concept at "
                         f"index={self.walkthrough_index} — using explicit keep/remove prompt")

        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_swap_caught(self):
        """
        Change log: 2026-03-31 — replaced hardcoded framework strings with
        dynamic context from self.kg_context and concept_swap.config.
        """
        wrong       = self.concept_swap.config["wrong_concept"]
        wrong_fw    = self.concept_swap.config["wrong_framework"]
        active_fw   = self.kg_context["framework"]
        active_case = self.kg_context["case_type"]
        instruction = (
            f"{SWAP_CAUGHT_PROMPT}\n\n"
            f"─── CONTEXT ──────────────────────────────────────────────────────────\n"
            f"Wrong concept flagged: **{wrong}**\n"
            f"It belongs to: {wrong_fw} (a different type of analysis)\n"
            f"This case is: {active_case} analysis — {active_fw}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_summary(self):
        """
        Change log: 2026-03-31 — corrected swap inclusion logic.
        If swap caught: exclude from summary.
        If swap not caught: include (experimental outcome — do not remove silently).
        """
        self.walkthrough_done = True
        if self.swap_presented and not self.concept_swap.is_detected:
            logging.info(f"[SWAP] summary reached — swap was presented but not caught")

        wrong          = self.concept_swap.config["wrong_concept"].lower()
        excluded_lower = [e.lower() for e in self.excluded_concepts] + [wrong] \
                         if self.concept_swap.is_detected else \
                         [e.lower() for e in self.excluded_concepts]

        active_concepts = [
            c for c in self.walkthrough_concepts
            if c.lower() not in excluded_lower
        ]

        logging.info(f"[SUMMARY] active_concepts={active_concepts}")

        instruction = (
            f"{SUMMARY_PROMPT}\n\n"
            f"─── CONCEPTS TO INCLUDE (in order) ──────────────────────────────────\n"
            f"{', '.join(active_concepts)}\n"
            f"Framework: {self.kg_context['framework']} | Case: {self.kg_context['case_type']}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction, store_answer=True)

    def _stream_freeform(self, cs_detected: bool):
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
    # Fallback + session
    # ══════════════════════════════════════════════════════════════════════

    def end_session(self) -> None:
        from backend.logger import end_session as _end_session
        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection('sessions').document(self.session_id).update({
                'concept_swap_detected': self.concept_swap.is_detected,
                'swap_detected_at_end':  self.concept_swap.is_detected,
            })
            logging.info(f'[END SESSION] stamped session={self.session_id}, '
                         f'swap_detected={self.concept_swap.is_detected}')
        except Exception as e:
            logging.warning(f'[END SESSION] Firestore stamp failed: {e}')
        _end_session(self.session_id)

    def _build_system_prompt(self) -> str:
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"
        return (
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n\n"
            f"You are a strategic consultant. Answer concisely in plain language. "
            f"No jargon. No mention of databases or technical systems."
        )