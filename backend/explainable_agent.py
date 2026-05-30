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
    log_framework_switched, log_concept_added,
)
from backend import knowledge_graph as kg
from backend.rag_explainer import build_citation_header, check_and_append_warning

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "AI Implementation"

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

SINGLE_CONCEPT_PROMPT = """
You are a strategic consultant walking a user through a framework ONE concept at a time.

The concept name, explanation, and sources have already been presented to the user above.
YOUR ONLY JOB: Output the sub-bullets and closing question — nothing else.

EXACT OUTPUT FORMAT — copy this structure precisely:

- [analytical question, 5-7 words, specific to this case]
- [analytical question, 5-7 words, specific to this case]

*Shall we move on to the next concept?*

─── EXAMPLE (for Strategic Fit) ────────────────────────────────────────────
- Is the workflow problem frequent enough to justify governance overhead?
- Would a simpler rules-based solution achieve the same result?

*Shall we move on to the next concept?*
─────────────────────────────────────────────────────────────────────────────

─── STRICT RULES ────────────────────────────────────────────────────────────
- Do NOT repeat the concept name as a header — it is already shown above
- Do NOT write a rationale sentence — it is already shown above
- Do NOT write an intro paragraph
- Do NOT list other concepts or buckets
- Do NOT say "here is the framework" or "I recommend"
- Do NOT add numbered lists
- Maximum 2 sub-bullets
- Never mention a knowledge graph, database, or technical system
- Always end with: *Shall we move on to the next concept?*
"""

CONCEPT_QA_PROMPT = """
You are a strategic consultant who just introduced one concept from a framework.
The user has a question about it.

Answer in 2–3 sentences. Stay grounded in the current case context.
Plain language only — no jargon, no technical terms.

─── STRICT RULE ─────────────────────────────────────────────────────────────
Answer ONLY about the concept named in CURRENT CONCEPT below.
If the user mentions a different concept by name:
- Check the FRAMEWORK CONTEXT below for already-planned concepts.
  Only say a concept is already planned if you are highly confident it
  matches one from the list — exact name or a clear, unambiguous synonym.
  If uncertain, do not mention it — just answer about the CURRENT CONCEPT.
- If it is clearly not in the list, briefly acknowledge it in one clause
  ("we can look at that next") — then answer about the CURRENT CONCEPT only

If the user wants to remove a specific sub-point or sub-bullet:
- Acknowledge it clearly in one sentence: "Understood — I'll leave that out
  of the final summary."
- Do NOT remove the parent concept
- Do NOT ask for confirmation — honour it immediately
- Then end with the normal closing question
─────────────────────────────────────────────────────────────────────────────

─── CONDITIONAL FORMAT ───────────────────────────────────────────────────────
ON SWAP CONCEPT: {on_swap}

IF on_swap is False AND the user is asking WHY this concept matters, WHY it
is relevant, or WHY it belongs here — structure your answer using this format:
  "If we don't consider [concept], then [specific consequence for this case].
   Since [one sentence grounding it in the case], this concept is essential."

IF on_swap is True OR the user is asking anything other than why — answer
naturally in 2–3 sentences without the if-then format.
─────────────────────────────────────────────────────────────────────────────

After answering, use the closing specified in CLOSING INSTRUCTION below.
"""

SUMMARY_PROMPT = """
You are a strategic consultant presenting a final framework summary.

FORMAT:
**Full Framework Summary**

**[Bucket 1]**
- sub-bullet (5–7 words)
- sub-bullet (5–7 words)

**[Bucket 2]**
- sub-bullet (5–7 words)
- sub-bullet (5–7 words)

(continue for all buckets)

─── RULES ─────────────────────────────────────────────────────────────────
- Include ONLY concepts listed in CONCEPTS TO INCLUDE below
- No rationale sentences — summary only
- Do NOT add a follow-up question — end after the last concept

─── SUB-BULLET EXCLUSIONS ────────────────────────────────────────────────
Review the full conversation history. If the user explicitly asked to remove
a specific sub-point or sub-bullet during the walkthrough, exclude it from
the sub-bullets you generate for that concept. Do not mention the exclusion
— simply omit it.
─────────────────────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════
# Addition flow prompts — Change log: 2026-05-28
# ══════════════════════════════════════════════════════════════════════════

ADD_CLASSIFY_PROMPT = """
You are a classifier for a case interview framework tool.

The user wants to add something to the framework. Determine whether they are adding:
- "sub_bullet" : a specific point/question that belongs UNDER one of the existing pillars
- "pillar"     : a whole new top-level area of analysis

Then identify the best-guess target pillar from the list below (for sub_bullet),
or null (for a new pillar).

─── EXISTING PILLARS ─────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────

─── USER WANTS TO ADD ──────────────────────────────────────────────────────
{item}
────────────────────────────────────────────────────────────────────────────

Respond ONLY with valid JSON, no explanation, no markdown:
{{"kind": "sub_bullet" or "pillar", "target": "pillar name or null", "confidence": float}}
"""

ADD_RESOLVE_PROMPT = """
You are a classifier for a case interview framework tool.

The agent asked the user where they want to add "{item}".
Options were: add it under an existing pillar, or add it as a new separate area.

─── EXISTING PILLARS ─────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────

─── USER REPLY ──────────────────────────────────────────────────────────────
{reply}
────────────────────────────────────────────────────────────────────────────

Interpret the user's reply:
- If they confirm adding under a specific pillar, return that pillar name as target
- If they want it as a new separate area/pillar, set as_new_pillar=true
- If they cancel or change their mind (no, never mind, forget it), set confirmed=false

Respond ONLY with valid JSON, no explanation, no markdown:
{{"confirmed": true or false, "target": "pillar name or null", "as_new_pillar": true or false}}
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

ADD_PILLAR_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.

The user wants to add a new pillar: "{item}".
Check whether it matches one of the framework's other (currently hidden) pillars below.

─── OTHER PILLARS ───────────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────

A match means the user's new pillar is essentially the same area of analysis
as one of the pillars above — same topic, possibly different wording.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched": true or false, "matched_pillar": "pillar name or null", "confidence": float}}
"""

ADD_MATCH_THRESHOLD = 0.75

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
- "concept_added"      : user wants to ADD a new concept to the framework
                         ("what about X", "can we also consider X", "add X",
                          "include X", "how about X", "what if we add X")
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

CRITICAL — for "concept_added": the "detail" field must contain
the NEW thing the user wants to add, NOT any existing concept
they are referencing.

Example: "I think we should add financial impact as a pillar"
→ detail: "Financial impact" (the NEW thing)

Example: "what about regulatory risks under Feasibility"
→ detail: "Regulatory risks" (the NEW thing)
→ NOT detail: "Feasibility" (an existing concept)

This does NOT include:
- Questions asking why a concept is included ("why is X here?", "why are we considering X?")
- Questions asking if something belongs to a concept ("is X part of Y?")
- Single word responses ("no", "yes", "ok")

Respond ONLY with valid JSON, no explanation, no markdown:
{{"override": true or false, "type": "redo"|"concept_excluded"|"concept_added"|"framework_switch"|"none", "detail": string or null, "confidence": float}}

Examples (assuming concepts include: Strategic Fit, Use Case and Solution, Feasibility):
- "start over" → {{"override": true, "type": "redo", "detail": null, "confidence": 0.99}}
- "restart from scratch" → {{"override": true, "type": "redo", "detail": null, "confidence": 0.98}}
- "move on" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "let's move on to the next one" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "next concept please" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "remove Feasibility" → {{"override": true, "type": "concept_excluded", "detail": "Feasibility", "confidence": 0.97}}
- "what about Financial Impact?" → {{"override": true, "type": "concept_added", "detail": "Financial Impact", "confidence": 0.96}}
- "can we also consider Risks?" → {{"override": true, "type": "concept_added", "detail": "Risks", "confidence": 0.95}}
- "how about adding regulatory compliance?" → {{"override": true, "type": "concept_added", "detail": "Regulatory compliance", "confidence": 0.96}}
- "use a different framework" → {{"override": true, "type": "framework_switch", "detail": null, "confidence": 0.96}}
- "yes" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "no" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "no thanks" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "makes sense, continue" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "let's discuss Strategic Fit" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "can we talk about Feasibility?" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "what about Use Case and Solution?" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "why are we considering this?" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "why is this here?" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
- "is it part of Feasibility?" → {{"override": false, "type": "none", "detail": null, "confidence": 0.99}}
"""


class ExplainableAgent(BlackBoxAgent):
    """
    Explainable agent — stateful concept-by-concept walkthrough.

    Inherits from BlackBoxAgent:
      - Clarification phase (_stream_clarification)
      - _detect_override(), _is_answer(), _strip_fences()
      - _strip_concept_swap_from_history()
      - send_message(), end_session()
      - KG infrastructure (_fetch_kg_context, _update_kg_if_framework_mentioned)
      - show_tree(), _build_tree_overview()
      - _check_duplicate()

    Overrides:
      - __init__                  : explainable case + walkthrough state variables
      - get_opening_message       : tailored intro
      - get_pre_analysis_instruction : agent-specific instruction
      - begin_analysis            : tree/button flow — streams first concept
      - _build_system_prompt      : minimal fallback (used by inherited send_message)
      - _stream_main              : stateful walkthrough logic
      - _detect_override          : walkthrough-aware override classifier

    Change log: 2026-05-12 — begin_analysis() replaces start_main_phase()
    Change log: 2026-05-16 — concept_added double-log fix; _resolve_pending
                             log_memory_override removed; WALKTHROUGH_OVERRIDE_PROMPT
                             negative examples added
    Change log: 2026-05-17 — timer sentinel; UX note moved before first concept
    Change log: 2026-05-25 — citation block updated for JSON knowledge base;
                             CASE_TYPE updated to AI Implementation;
                             WALKTHROUGH_OVERRIDE_PROMPT examples updated for Allianz case
    """

    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="explainable")
        self.original_case = get_case("explainable")
        self._pending      = False
        self.turn_count    = 0

        self.phase = "warmup"
        self.clarification_facts = get_clarification_facts("explainable")

        self.concept_swap = ConceptSwap(
            agent_type="explainable",
            session_id=self.session_id
        )

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
        self.walkthrough_concepts = []
        self.walkthrough_index    = 0
        self.walkthrough_active   = False
        self.walkthrough_done     = False
        self.excluded_concepts    = []
        self.swap_presented       = False
        self.swap_position        = 0

        # ── Pending confirmation states ────────────────────────────────────
        self.pending_excl = None
        self.pending_fw   = None
        self.pending_add  = None   # {"item": str, "kind": str, "target": str}

        # ── User sub-points — populated by duplicate guard path ───────────
        # Change log: 2026-05-12
        self.user_sub_points = {}

        # ── User-added pillars — explicit tracking for summary ─────────────
        # Change log: 2026-05-28 — separate from walkthrough_concepts so summary
        # can distinguish "user added" from "not yet reached"
        self.user_added_pillars = []

        # ── User-added pillars — tracked separately so summary always
        #    includes them regardless of walkthrough position.
        # Change log: 2026-05-28
        self.user_added_pillars = []

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
        )

    def get_pre_analysis_instruction(self) -> str:
        return (
            "📖 *After you click the button below, I'll walk you through "
            "each concept one at a time — you can ask questions or suggest "
            "changes at any step.*"
        )

    def begin_analysis(self):
        """
        Generator — called when user clicks 'Got it, show me the full analysis'.
        Replaces start_main_phase().
        Change log: 2026-05-12
        Change log: 2026-05-17 — timer sentinel as first yield (split in app.py);
                                  UX note moved before first concept
        """
        self._start_main_phase_setup()

        yield (
            "⚠️ Your goal is to build a structured plan for this case. "
            "Review each factor below, share your thoughts, and you **should not only read it** but also add or remove anything you think is missing."
        )

        yield "⏱️ Your 20-minute session has started. The timer is shown on the left."
        self.walkthrough_concepts = self._build_walkthrough_concepts()
        self.walkthrough_active   = True
        self.walkthrough_index    = 0
        self.swap_presented       = False

        yield (
            "💡 *When you're finished, click **‼️End Session** "
            "to close your session. **Note: this cannot be undone.***\n\n---\n\n"
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
        else:
            self.swap_presented = False

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
    # Framework resolver
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
            frameworks = self._kg_all_frameworks
            if not frameworks:
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
    # Addition flow helpers — Change log: 2026-05-28
    # ══════════════════════════════════════════════════════════════════════

    def _classify_addition(self, item: str) -> dict:
        """LLM: is this a sub-bullet or a new pillar? Best-guess target pillar."""
        from backend import knowledge_base as kb
        pillars = ", ".join(p["name"] for p in kb.get_shown_pillars())
        prompt = ADD_CLASSIFY_PROMPT.format(pillars=pillars, item=item)
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL, contents=prompt,
            )
            parsed = json.loads(self._strip_fences(response.text))
            return {
                "kind":   parsed.get("kind", "sub_bullet"),
                "target": parsed.get("target"),
            }
        except Exception as e:
            logging.warning(f"[ADD CLASSIFY] error: {e}")
            return {"kind": "sub_bullet", "target": self._current_concept()}

    def _resolve_addition(self, reply: str) -> dict:
        """LLM: interpret user's confirmation reply for placement."""
        from backend import knowledge_base as kb
        pillars = ", ".join(p["name"] for p in kb.get_all_pillars())
        item    = self.pending_add["item"] if self.pending_add else ""
        prompt  = ADD_RESOLVE_PROMPT.format(pillars=pillars, item=item, reply=reply)
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL, contents=prompt,
            )
            parsed = json.loads(self._strip_fences(response.text))
            return {
                "confirmed":     parsed.get("confirmed", False),
                "target":        parsed.get("target"),
                "as_new_pillar": parsed.get("as_new_pillar", False),
            }
        except Exception as e:
            logging.warning(f"[ADD RESOLVE] error: {e}")
            return {"confirmed": False, "target": None, "as_new_pillar": False}

    def _match_key_question(self, item: str, pillar_name: str) -> dict | None:
        """LLM: does item match a key_question in this pillar? Return matched text+sources."""
        from backend import knowledge_base as kb
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
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL, contents=prompt,
            )
            parsed = json.loads(self._strip_fences(response.text))
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD):
                idx = parsed.get("matched_index")
                if idx is not None and 0 <= idx < len(key_questions):
                    return {
                        "question": key_questions[idx],
                        "sources":  pillar.get("key_questions_sources", ""),
                    }
        except Exception as e:
            logging.warning(f"[ADD MATCH] error: {e}")
        return None

    def _match_withheld_pillar(self, item: str) -> str | None:
        """LLM: does new pillar match a withheld pillar (Financial Impact / Risks)?"""
        from backend import knowledge_base as kb
        shown_ids = {p["id"] for p in kb.get_shown_pillars()}
        withheld  = [p for p in kb.get_all_pillars() if p["id"] not in shown_ids]
        if not withheld:
            return None

        pillars_block = "\n".join(f"- {p['name']}" for p in withheld)
        prompt = ADD_PILLAR_MATCH_PROMPT.format(item=item, pillars=pillars_block)
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL, contents=prompt,
            )
            parsed = json.loads(self._strip_fences(response.text))
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD):
                return parsed.get("matched_pillar")
        except Exception as e:
            logging.warning(f"[ADD PILLAR MATCH] error: {e}")
        return None

    # ══════════════════════════════════════════════════════════════════════
    # Override detection — walkthrough-aware
    # ══════════════════════════════════════════════════════════════════════

    def _detect_override(self, user_input: str) -> dict | None:
        import json as _json
        from backend.black_box_agent import OVERRIDE_THRESHOLD as _THRESH

        if not self.walkthrough_active or self.walkthrough_done:
            return super()._detect_override(user_input)

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
    # Change log: 2026-05-12 — override first, swap gated
    # Change log: 2026-05-16 — skip log_memory_override for concept_added
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 0. Resolve pending state ───────────────────────────────────────
        if self.pending_add is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(
                types.Content(role="user", parts=[types.Part(text=user_input)])
            )
            yield from self._resolve_pending_add(user_input)
            return

        if self.pending_excl is not None or self.pending_fw is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(
                types.Content(role="user", parts=[types.Part(text=user_input)])
            )
            yield from self._resolve_pending(user_input)
            return

        # ── 1. Override detection first ────────────────────────────────────
        just_added_concept = None
        override = self._detect_override(user_input)

        # ── 1b. Check if user explicitly removed the wrong concept ─────────
        cs_detected = False
        if (override and
                override["type"] == "concept_excluded" and
                not self.concept_swap.is_detected and
                self.swap_presented):
            wrong        = self.concept_swap.config["wrong_concept"]
            detail_lower = (override.get("detail") or "").lower().strip()
            wrong_lower  = wrong.lower().strip()
            current      = self._current_concept()
            names_swap = (
                detail_lower == wrong_lower or
                (len(detail_lower) >= 5 and detail_lower in wrong_lower) or
                (len(wrong_lower) >= 5 and wrong_lower in detail_lower)
            )
            on_swap_now = current is not None and current.lower() == wrong_lower
            if names_swap or on_swap_now:
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
                cs_detected = True
                override = None

        # ── 2. Swap detection — only if presented AND no override ──────────
        if not cs_detected and self.swap_presented and not override:
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
                logging.info(f"[SWAP] caught — index→{self.walkthrough_index}")
        elif self.walkthrough_active and not override and not cs_detected:
            logging.debug(f"[SWAP] detection skipped — swap not yet presented")

        # ── 3. Override logging and handling ──────────────────────────────
        if override:
            if override["type"] != "concept_added":
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
                self.pending_excl         = None
                self.pending_fw           = None
                self.pending_add          = None
                self.user_added_pillars   = []
                if self.concept_swap.is_detected:
                    self.history = self._strip_concept_swap_from_history()
                yield "Noted! Let me start the walkthrough fresh...\n\n"
                log_user_message(self.session_id, "[REDO TRIGGERED]")

            elif override["type"] == "framework_switch":
                if not override.get("detail"):
                    override = {"type": "framework_unspecified"}
                else:
                    matches = self._resolve_framework(override["detail"])
                    if len(matches) == 1:
                        self.pending_fw = matches[0]
                        override = {"type": "pending_fw_set"}
                    elif len(matches) >= 2:
                        override = {"type": "framework_clarification", "matches": matches,
                                    "detail": override["detail"]}
                    else:
                        override = {"type": "framework_not_found", "detail": override["detail"]}

            elif override["type"] == "concept_excluded":
                detail = (override.get("detail") or "").strip().lower()
                # Vague references ("it", "this", "that") → current concept
                vague = {"it", "this", "that", "this one", "that one", "this concept", "that concept"}
                if detail in vague or not detail:
                    excl = self._current_concept()
                else:
                    excl = override.get("detail")
                if excl:
                    self.pending_excl = excl
                    override = {"type": "pending_excl_set"}
                    logging.info("[OVERRIDE] concept exclusion pending: " + excl)

            elif override["type"] == "concept_added" and override.get("detail"):
                new_concept = override["detail"]
                classification = self._classify_addition(new_concept)
                self.pending_add = {
                    "item":   new_concept,
                    "kind":   classification["kind"],
                    "target": classification["target"],
                }
                override = {"type": "pending_add_set"}
                logging.info(f"[ADD] pending: '{new_concept}' "
                             f"kind={self.pending_add['kind']} "
                             f"target={self.pending_add['target']}")

        # ── 4. Log and append user message ────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 5. Routing log ─────────────────────────────────────────────────
        logging.info(f"[ROUTE] active={self.walkthrough_active}, "
                     f"done={self.walkthrough_done}, "
                     f"swap_presented={self.swap_presented}, "
                     f"index={self.walkthrough_index}, "
                     f"swap_position={self.swap_position}, "
                     f"cs_detected={cs_detected}")

        # ── 6. Route ───────────────────────────────────────────────────────
        if not self.walkthrough_active:
            self.walkthrough_concepts = self._build_walkthrough_concepts()
            self.walkthrough_active   = True
            self.walkthrough_index    = 0
            self.swap_presented       = False
            yield from self._stream_concept(is_first=True)

        elif self.walkthrough_done:
            yield from self._stream_freeform(cs_detected)

        elif override and override["type"] == "pending_excl_set":
            yield from self._stream_pushback("concept", self.pending_excl)

        elif override and override["type"] == "pending_fw_set":
            yield from self._stream_pushback("framework", self.pending_fw)

        elif override and override["type"] == "pending_add_set":
            yield from self._ask_add_placement()

        elif override and override["type"] == "framework_switch":
            yield from self._stream_concept(is_first=True)

        elif override and override["type"] == "framework_clarification":
            matches   = override["matches"]
            match_str = " or ".join(f"**{m}**" for m in matches)
            yield from self._stream_with_instruction(
                instruction=(
                    f"The user mentioned switching frameworks. It could mean {match_str}. "
                    f"Ask them which one they meant in one short, friendly sentence. "
                    f"Do not present any concept yet."
                )
            )

        elif override and override["type"] == "framework_not_found":
            available = self._format_framework_list()
            detail    = override.get("detail") or "that framework"
            yield from self._stream_with_instruction(
                instruction=(
                    f"The user mentioned '{detail}' as a framework. "
                    f"Respond in 2 short sentences: "
                    f"(1) Say you don't recognise '{detail}' — brief and neutral. "
                    f"(2) Ask which of these they'd like instead: {available}. "
                    f"Do not present any concept yet."
                )
            )

        elif override and override["type"] == "framework_unspecified":
            available = self._format_framework_list()
            yield from self._stream_with_instruction(
                instruction=(
                    f"The user wants to switch frameworks but hasn't named one. "
                    f"Ask in one friendly sentence: do they have a specific framework "
                    f"in mind? If not, suggest: {available}. "
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
            yield from self._stream_concept_qa(just_added=just_added_concept)

    # ══════════════════════════════════════════════════════════════════════
    # Pending resolution + pushback
    # Change log: 2026-05-16 — removed log_memory_override from _resolve_pending
    # ══════════════════════════════════════════════════════════════════════

    def _resolve_to_concept_name(self, name: str) -> str:
        for c in self.walkthrough_concepts:
            if c.lower() == name.lower():
                return c
        fallback = self._current_concept() or name
        logging.info(f"[RESOLVE] '{name}' not in walkthrough_concepts — fallback: '{fallback}'")
        return fallback

    def _resolve_pending(self, user_input: str):
        confirmed = self._is_ready_to_advance(user_input)

        if self.pending_excl is not None:
            excl = self.pending_excl
            if confirmed:
                excl = self._resolve_to_concept_name(excl)
                if excl not in self.excluded_concepts:
                    self.excluded_concepts.append(excl)
                self.pending_excl = None
                logging.info("[PENDING] exclusion confirmed: " + excl)
                self.walkthrough_index += 1
                yield "Understood — removing **" + excl + "** from the framework. Let's continue.\n\n"
                concept = self._current_concept()
                if concept is None:
                    yield from self._stream_summary()
                else:
                    yield from self._stream_concept(is_first=False)
            else:
                logging.info("[PENDING] exclusion not confirmed — continuing: " + excl)
                yield from self._stream_pushback("concept", excl)

        elif self.pending_fw is not None:
            fw = self.pending_fw
            if confirmed:
                new_context = self._fetch_kg_context_by_framework(fw)
                if new_context:
                    from_fw = self.kg_context["framework"]
                    self.kg_context = new_context
                    self._rebuild_walkthrough_on_framework_switch(from_framework=from_fw)
                self.pending_fw = None
                logging.info("[PENDING] framework switch confirmed: " + fw)
                yield "Switching to **" + fw + "**. Let's start from the beginning.\n\n"
                yield from self._stream_concept(is_first=True)
            else:
                logging.info("[PENDING] framework switch not confirmed — continuing: " + fw)
                yield from self._stream_pushback("framework", fw)

    # ══════════════════════════════════════════════════════════════════════
    # Addition placement flow — Change log: 2026-05-28
    # ══════════════════════════════════════════════════════════════════════

    def _ask_add_placement(self):
        """Ask user to confirm where to add their item (static, no LLM)."""
        item   = self.pending_add["item"]
        kind   = self.pending_add["kind"]
        target = self.pending_add["target"]

        if kind == "sub_bullet" and target:
            msg = (
                f"Good idea. Would you like to add **{item}** as a point under "
                f"**{target}**, or as a separate area of the framework?"
            )
        else:
            msg = (
                f"Good idea. Would you like to add **{item}** as a separate area "
                f"of the framework, or does it fit under one of the existing pillars?"
            )

        self.history.append(
            types.Content(role="model", parts=[types.Part(text=msg)])
        )
        log_agent_response(self.session_id, msg)
        yield msg

    def _resolve_pending_add(self, user_input: str):
        """Resolve user's confirmation reply for a pending addition."""
        from backend import knowledge_base as kb

        resolved = self._resolve_addition(user_input)
        item     = self.pending_add["item"]

        # ── User cancelled ─────────────────────────────────────────────────
        if not resolved["confirmed"]:
            self.pending_add = None
            logging.info(f"[ADD] cancelled: '{item}'")
            yield "No problem, let's leave that out for now.\n\n"
            concept = self._current_concept()
            if concept is None:
                yield from self._stream_summary()
            else:
                yield f"*Shall we move on to the next concept?*"
            return

        # ── CASE B: new pillar ─────────────────────────────────────────────
        if resolved["as_new_pillar"]:
            matched_pillar = self._match_withheld_pillar(item)
            if matched_pillar:
                self.walkthrough_concepts.append(matched_pillar)
                self.user_added_pillars.append(matched_pillar)
                log_concept_added(self.session_id, matched_pillar)
                self.pending_add = None
                logging.info(f"[ADD] new pillar matched withheld: '{matched_pillar}'")
                yield (
                    f"Good call. **{matched_pillar}** is an important area, "
                    f"we'll cover it later in the walkthrough.\n\n"
                    f"*Shall we move on to the next concept?*"
                )
            else:
                self.walkthrough_concepts.append(item)
                self.user_added_pillars.append(item)
                log_concept_added(self.session_id, item)
                self.pending_add = None
                logging.info(f"[ADD] new pillar (no match): '{item}'")
                yield (
                    f"Noted. We'll add **{item}** as a separate area "
                    f"toward the end of the walkthrough.\n\n"
                    f"*Shall we move on to the next concept?*"
                )
            return

        # ── CASE A: sub-bullet into a pillar ───────────────────────────────
        target = resolved["target"] or self.pending_add["target"] or self._current_concept()
        match  = self._match_key_question(item, target)

        if target not in self.user_sub_points:
            self.user_sub_points[target] = []

        if match:
            self.user_sub_points[target].append(match["question"])
            log_concept_added(self.session_id, item)
            self.pending_add = None
            logging.info(f"[ADD] sub-bullet matched key_question under '{target}'")
            sources = f"\n\n{match['sources']}" if match.get("sources") else ""
            yield (
                f"Good point, that's an important angle. I'll add it under "
                f"**{target}**:\n\n"
                f"- {match['question']}"
                f"{sources}\n\n"
                f"*Shall we move on to the next concept?*"
            )
        else:
            self.user_sub_points[target].append(item)
            log_concept_added(self.session_id, item)
            self.pending_add = None
            logging.info(f"[ADD] sub-bullet (no match) under '{target}'")
            yield (
                f"I don't have a source for this one, but it's a good addition. "
                f"I'll add it under **{target}**:\n\n"
                f"- {item}\n\n"
                f"*Shall we move on to the next concept?*"
            )

    def _stream_pushback(self, pending_type: str, detail: str):
        concept = self._current_concept() or "the current concept"

        if pending_type == "concept":
            instruction = (
                "You are a strategic consultant. The user wants to remove "
                "**" + detail + "** from the framework. "
                "Push back using counterfactual reasoning.\n\n"
                "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
                "1. One sentence: If we remove " + detail + ", then [consequence].\n"
                "2. One sentence grounding it in the case context.\n"
                "3. End with: Would you still like to remove it?\n\n"
                "─── RULES ──────────────────────────────────────────────────────────────────\n"
                "- Consulting reasoning only\n"
                "- Soft tone — consequence as information, not command\n"
                "- Do NOT refuse the removal\n"
                "- Do NOT present any other concept\n"
                "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
                "Concept questioned: **" + detail + "**\n"
                "Current concept: **" + concept + "**\n"
                "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
                "─────────────────────────────────────────────────────────────────────\n"
            )
        else:
            instruction = (
                "You are a strategic consultant. The user wants to switch to "
                "**" + detail + "** framework. "
                "Push back using counterfactual reasoning.\n\n"
                "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
                "1. One sentence: If we switch to " + detail + ", we would [consequence].\n"
                "2. One sentence: why current framework fits this case.\n"
                "3. End with: Would you still like to switch?\n\n"
                "─── RULES ──────────────────────────────────────────────────────────────────\n"
                "- Consulting reasoning only\n"
                "- Soft tone — consequence as information, not command\n"
                "- Do NOT refuse the switch\n"
                "- Do NOT present any concept yet\n"
                "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
                "Proposed: **" + detail + "**\n"
                "Current: **" + self.kg_context["framework"] + "**\n"
                "Case: " + self.kg_context["case_type"] + "\n"
                "─────────────────────────────────────────────────────────────────────\n"
            )

        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_concept(self, is_first: bool):
        concept = self._current_concept()
        if concept is None:
            yield from self._stream_summary()
            return

        is_wrong   = self._is_wrong_concept(concept)
        swap_block = self.concept_swap.get_system_prompt_block() if is_wrong else ""

        # ── Build static prefix from JSON ──────────────────────────────────
        from backend import knowledge_base as kb

        if not is_wrong:
            # Look up pillar by name directly (walkthrough uses pillar names)
            pillar = next(
                (p for p in kb.get_all_pillars() if p["name"].lower() == concept.lower()),
                None
            )
            if pillar is None:
                # Fall back to concept-level lookup for user-added concepts
                pillar = kb.get_pillar_for_concept_name(concept)
            if pillar is None:
                # User-added concept not in KB
                note = (
                    "\n> *ℹ️ This concept isn't in my knowledge base "
                    "— I can discuss it, but can't verify it with a source.*\n"
                )
                prefix = "**" + concept + "**\n" + note
            else:
                description = pillar.get("description", "")
                sub_bullets = pillar.get("sub_bullets", [])
                sources     = pillar.get("sub_bullets_sources", "")

                bullet_lines = "\n".join(f"- {b}" for b in sub_bullets)

                # Append any user-added sub-points for this pillar inline
                # (added earlier while on a different concept). Change log: 2026-05-29
                added_points = self.user_sub_points.get(concept, [])
                if added_points:
                    added_lines = "\n".join(f"- {p}" for p in added_points)
                    bullet_lines = bullet_lines + "\n" + added_lines

                sources_line = f"\n\n{sources}" if sources else ""

                prefix = (
                    f"**{concept}**\n\n"
                    f"{description}\n\n"
                    f"{bullet_lines}"
                    f"{sources_line}\n\n"
                    f"*Shall we move on to the next concept?*"
                )
        else:
            swap    = kb.get_swap_concept()
            bullets = swap.get("sub_bullets", []) if swap else []
            bullet_lines = "\n".join(f"- {b}" for b in bullets)
            prefix = (
                f"**{concept}**\n\n"
                f"{bullet_lines}\n\n"
                f"*Shall we move on to the next concept?*"
            )

        if is_first:
            prefix = "Here is how I would structure the analysis:\n\n" + prefix

        # ── Yield static text, append to history ───────────────────────────
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=f"[Present concept: {concept}]")])
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
                logging.info(f"[SWAP] concept presented at position={self.swap_position}")

        yield prefix

    def _stream_concept_qa(self, just_added: str | None = None):
        concept = self._current_concept() or "the current concept"
        on_swap = (self.walkthrough_index == self.swap_position
                   and self.swap_presented
                   and not self.concept_swap.is_detected)

        closing = (
            "End with exactly:\n"
            "*I can see why you'd question this — shall we include it or move on without it?*"
            if on_swap else
            "End with exactly:\n"
            "*Shall we move on to the next concept?*"
        )

        qa_prompt = CONCEPT_QA_PROMPT.format(on_swap=on_swap)

        added_note = ""
        if just_added:
            added_note = (
                "Good idea — I'll add **" + just_added
                + "** after we finish **" + concept + "**.\n\n"
            )

        instruction = (
            qa_prompt + "\n\n"
            "─── CLOSING INSTRUCTION ──────────────────────────────────────────────\n"
            + closing + "\n"
            "─── CONTEXT ──────────────────────────────────────────────────────────\n"
            "Current concept: **" + concept + "**\n"
            "On swap concept: " + str(on_swap) + "\n"
            "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
            "Framework concepts (in order): " + ", ".join(
                c for c in self.walkthrough_concepts
                if just_added is None or c.lower() != just_added.lower()
            ) + "\n"
            "──────────────────────────────────────────────────────────────────────\n"
        )

        yield from self._stream_with_instruction(instruction=instruction, prefix=added_note)

    def _stream_swap_caught(self):
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
        self.walkthrough_done = True

        from backend import knowledge_base as kb

        wrong          = self.concept_swap.config["wrong_concept"].lower()
        excluded_lower = [e.lower() for e in self.excluded_concepts] + [wrong] \
                         if self.concept_swap.is_detected else \
                         [e.lower() for e in self.excluded_concepts]

        added_lower = [p.lower() for p in self.user_added_pillars]

        # ── Group 1: completed original pillars (advanced PAST) ────────────
        completed_originals = [
            c for c in self.walkthrough_concepts[:self.walkthrough_index]
            if c.lower() not in excluded_lower
            and c.lower() not in added_lower
        ]

        # ── Group 2: user-added pillars (heading only) ─────────────────────
        added_pillars = [
            p for p in self.user_added_pillars
            if p.lower() not in excluded_lower
        ]

        logging.info(f"[SUMMARY] completed_originals={completed_originals}, "
                     f"added_pillars={added_pillars}")

        # ── Build the summary string DETERMINISTICALLY (no LLM) ───────────
        # Python owns rendering: the matching already happened during the
        # conversation and is stored in user_sub_points. Rendering known
        # state must be exact — an LLM would paraphrase. (Change log: 2026-05-28)
        lines = [
            "**Final Framework Summary**",
            "",
        ]

        for c in completed_originals:
            pillar = next(
                (p for p in kb.get_all_pillars() if p["name"].lower() == c.lower()),
                None
            )
            lines.append(f"**{c}**")

            collected_sources = []
            if pillar:
                # Static sub-bullets (keep inline [a][b] refs)
                for b in pillar.get("sub_bullets", []):
                    lines.append(f"- {b}")
                src = pillar.get("sub_bullets_sources", "")
                if src:
                    collected_sources.append(src)
            elif c.lower() == wrong:
                # Swap concept shown (user never caught it) — render its
                # sub-bullets like any other concept. Change log: 2026-05-29
                swap = kb.get_swap_concept()
                for b in (swap.get("sub_bullets", []) if swap else []):
                    lines.append(f"- {b}")

            # User-added sub-points for this pillar (already matched text + refs)
            for sp in self.user_sub_points.get(c, []):
                lines.append(f"- {sp}")

            if collected_sources:
                lines.append("")
                # sub_bullets_sources already begins with "Sources:"
                lines.append(collected_sources[0])
            lines.append("")

        # User-added areas — heading only, no sub-bullets
        for p in added_pillars:
            lines.append(f"**{p}**")
            for sp in self.user_sub_points.get(p, []):
                lines.append(f"- {sp}")
            lines.append("")

        summary_text = "\n".join(lines).rstrip()

        # Append to history + log + store answer, mirroring _stream_with_instruction
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=summary_text)])
        )
        update_answer(self.session_id, summary_text)
        log_agent_response(self.session_id, summary_text)

        yield summary_text

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
    # Session + system prompt
    # ══════════════════════════════════════════════════════════════════════

    def get_summary(self):
        """
        Public wrapper for app.py to stream the summary on End Session.
        Routes through _stream_summary() so the summary is built from explicit
        session state (completed originals + added pillars), identical to the
        natural walkthrough-end summary. Replaces inherited send_message().
        Change log: 2026-05-28
        """
        yield from self._stream_summary()

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