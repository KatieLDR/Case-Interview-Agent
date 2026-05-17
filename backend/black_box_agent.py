import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from backend.logger import (
    create_session, end_session, stamp_started_at,
    log_user_message, log_agent_response,
    log_interruption, log_memory_override,
    update_answer, log_warmup_response,
    log_concept_added
)
from backend.cases import get_case, get_clarification_facts
from backend.concept_swap import ConceptSwap
from backend import knowledge_graph as kg

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)

# ── Model config ───────────────────────────────────────────────────────────
MAIN_MODEL       = "gemini-2.5-flash"
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "Profitability"

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

**[Branch concept]**
- **[Leaf concept]**
  - [analytical question, 5-7 words]
  - [analytical question, 5-7 words]

**[Branch concept]**
- **[Leaf concept]**
  - [analytical question, 5-7 words]
- **[Branch+Leaf concept]**
  - **[Leaf concept]**
    - [analytical question, 5-7 words]

(continue for all branches — do NOT add any not in KG context above,
UNLESS the user explicitly requests it)

─── STRUCTURAL EXAMPLE (format only — do not copy these questions) ──────
**Revenue**
- **Price per Unit**
  - [your question here]
  - [your question here]
- **Units Sold**
  - [your question here]
  - [your question here]

**Costs**
- **Fixed Cost**
  - [your question here]
- **Variable Cost**
  - **Cost per Unit**
    - [your question here]
─────────────────────────────────────────────────────────────────────────

**Key Considerations** *(only if relevant)*
- Critical dependency 1
- Critical dependency 2

─── INTERACTION STYLE ────────────────────────────────────────────────────
You are a reference tool, not an interviewer. After presenting or updating
a framework, ask ONE short natural follow-up question to invite exploration.

If the user wants to ADD a new concept or bucket:
- Add it immediately, no pushback
- If it fits within Revenue or Costs → add as a sub-bucket under the right branch
- If it is a new top-level area (e.g. Risks, Market) → add as a new primary bucket
- Never refuse user additions
- When a user explicitly asks to add a sub-bullet, always honour it — no limit applies

If the user wants to REMOVE or CHANGE an existing KG concept:
- Briefly explain your reasoning in one sentence
- Ask if they still want to proceed
- If they confirm, honour it immediately

─── RULES ────────────────────────────────────────────────────────────────
- Always use the exact format above — bold headers, bullet sub-buckets
- Never use numbered lists for framework buckets
- Be direct and concise
- Never evaluate or score the user
- Ask only ONE follow-up question per response
- When initially presenting concepts, aim for 2 analytical questions per leaf concept
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
- "redo"             : wants a fresh answer or complete regeneration
- "concept_excluded" : wants to remove a specific concept or bucket
- "concept_added"    : wants to add a new concept or bucket
- "framework_switch" : wants to use a specific named different framework
- "none"             : not steering the output

This does NOT include:
- Asking for a framework for the first time
- Asking follow-up questions ("can you elaborate?", "why is X here?")
- General questions about the case
- Single word responses ("yes", "no", "ok", "sure")
- Questions asking where a concept is ("where's X", "where is X")
- Conversational rejections ("no thanks", "not needed", "no")
- Asking what happened to a concept ("what happened to X")

- "parent": the existing concept named after "under" / "as part of" / "within" — null if no parent explicitly named

Respond ONLY with valid JSON, no explanation, no markdown:
{"override": true or false, "type": "redo"|"concept_excluded"|"concept_added"|"framework_switch"|"none", "detail": string or null, "parent": string or null, "confidence": float}

Examples:
- "redo this" → {"override": true, "type": "redo", "detail": null, "parent": null, "confidence": 0.99}
- "try a completely different approach" → {"override": true, "type": "redo", "detail": null, "parent": null, "confidence": 0.97}
- "remove profit per unit" → {"override": true, "type": "concept_excluded", "detail": "Profit per Unit", "parent": null, "confidence": 0.97}
- "add external risks as a bucket" → {"override": true, "type": "concept_added", "detail": "External Risks", "parent": null, "confidence": 0.97}
- "what about market preference?" → {"override": true, "type": "concept_added", "detail": "Market Preference", "parent": null, "confidence": 0.95}
- "use profitability framework" → {"override": true, "type": "framework_switch", "detail": "profitability", "parent": null, "confidence": 0.96}
- "give me a framework" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "why is variable cost here?" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.95}
- "yes" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "no" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "no thanks" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "ok" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "sure" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "where's variable cost" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "where is external risk" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "what happened to market share" → {"override": false, "type": "none", "detail": null, "parent": null, "confidence": 0.99}
- "add cafe sales under Units Sold" → {"override": true, "type": "concept_added", "detail": "Cafe sales", "parent": "Units Sold", "confidence": 0.97}
- "I think we should consider coffee bag sales as part of Units Sold" → {"override": true, "type": "concept_added", "detail": "Coffee bag sales", "parent": "Units Sold", "confidence": 0.96}
- "add market share" → {"override": true, "type": "concept_added", "detail": "Market share", "parent": null, "confidence": 0.97}
"""

# ══════════════════════════════════════════════════════════════════════════
# Warm-up content
# Change log: 2026-05-05 — redesigned warm-up
# Change log: 2026-05-06 — added "Let's go" cue
# ══════════════════════════════════════════════════════════════════════════

WARMUP_PROMPT = (
    "**🧪 Practice Exercise: Build a Solution**\n\n"
    "Before the main task, let's warm up with a quick exercise.\n\n"
    "**The situation:**\n"
    "You're thinking about moving to a new city. Before you go, "
    "you want to make sure you've thought everything through.\n\n"
    "**Your task:**\n"
    "Break this down into the main **areas** you'd need to figure out. "
    "For each area, list a few **questions** you'd want to answer.\n\n"
    "**Example:**\n"
    "> 🏠 **Housing**\n"
    "> - Where will I live?\n"
    "> - What's the average rent?\n"
    "> - How far is it from work?\n\n"
    "**Now it's your turn!**\n"
    "Type your areas and questions below — send as many messages as you like.\n"
    "When you're done, click the **✅ Done** button below."
)

WARMUP_ACKNOWLEDGEMENT = (
    "Got it! Anything else to add? "
    "When you're ready, click the **✅ Done** button below."
)

WARMUP_MODEL_ANSWER = (
    "**Here's an example of a complete solution:**\n\n"
    "> 🏠 **Housing** — Where will I live? What's the rent? How far from work?\n\n"
    "> 💰 **Finances** — What's my budget? What's the cost of living?\n\n"
    "> 🏢 **Work** — Do I have a job lined up? What's the commute?\n\n"
    "> 👥 **Social** — Do I know anyone there? How will I meet people?\n\n"
    "> 📦 **Logistics** — Moving costs, paperwork, transport\n\n"
    "---\n"
    "*Notice how each area has a clear theme, and each question helps you "
    "decide whether that area needs attention. "
    "That's exactly what you'll be doing in the main task — but for a business situation!*\n\n"
    "👆 When you're ready to close this practice and start the main task, click **Let's go!** below."
)

# ══════════════════════════════════════════════════════════════════════════
# Thresholds
# ══════════════════════════════════════════════════════════════════════════
ANSWER_THRESHOLD      = 0.90
OVERRIDE_THRESHOLD    = 0.85
MAX_TURNS_PER_SESSION = 50


class BlackBoxAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self._pending      = False
        self.turn_count    = 0

        self.phase = "warmup"
        self.clarification_facts = get_clarification_facts("black_box")

        self.concept_swap = ConceptSwap(
            agent_type="black_box",
            session_id=self.session_id
        )

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
        framework = kg.get_framework_for_case(case_type) or "Unknown Framework"
        concepts  = kg.get_ordered_concepts(framework) if framework else []
        return {"case_type": case_type, "framework": framework, "concepts": concepts}

    def _update_kg_if_framework_mentioned(self, user_input: str) -> None:
        lowered = user_input.lower()
        for framework_name, keywords in self._kg_framework_keywords.items():
            if any(kw in lowered for kw in keywords):
                if framework_name != self.kg_context["framework"]:
                    concepts = kg.get_ordered_concepts(framework_name)
                    self.kg_context = {
                        "case_type": self.kg_context["case_type"],
                        "framework": framework_name,
                        "concepts":  concepts,
                    }
                    print(f"[KG UPDATE] switched to '{framework_name}', "
                          f"{'concepts from KG' if concepts else 'model fallback'}")
                break

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
        framework = self.kg_context["framework"]

        try:
            tree = kg.get_framework_tree(framework)
            concept_lines = []
            for node in tree:
                indent = "  " * (node["depth"] - 1)
                concept_lines.append(f"{indent}{node['parent']} → {node['concept']}")
            concepts_str = "\n".join(concept_lines) if concept_lines else "N/A"
        except Exception as e:
            print(f"[SYSTEM PROMPT] tree fetch failed: {e}")
            concepts_str = " → ".join(self.kg_context["concepts"])

        kg_block = (
            f"─── KNOWLEDGE GRAPH CONTEXT ──────────────────────────────────────────\n"
            f"Case Type : {self.kg_context['case_type']}\n"
            f"Framework : {framework}\n"
            f"Concepts  :\n{concepts_str}\n"
            f"CRITICAL: Use branch nodes as PRIMARY BUCKET HEADERS (e.g. Revenue, Costs).\n"
            f"Use leaf nodes as SUB-BUCKET HEADERS under their parent branch.\n"
            f"Generate 2 analytical questions under each leaf concept.\n"
            f"Do NOT add any bucket or concept not listed here UNLESS the user explicitly requests it.\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
        )

        swap_block = self.concept_swap.get_system_prompt_block()
        return kg_block + swap_block + SYSTEM_PROMPT

    def _build_tree_overview(self) -> str:
        return (
            "**Framework Overview**\n\n"
            "- **Revenue**\n"
            "  - Price per Unit\n"
            "  - Units Sold\n\n"
            "- **Costs**\n"
            "  - Fixed Cost\n"
            "  - **Variable Cost**\n"
            "    - Cost per Unit\n"
        )

    def show_tree(self) -> str:
        return self._build_tree_overview()

    # ══════════════════════════════════════════════════════════════════════
    # Warm-up messages
    # ══════════════════════════════════════════════════════════════════════

    def get_warmup_message(self) -> str:
        return WARMUP_PROMPT

    def get_warmup_acknowledgement(self) -> str:
        return WARMUP_ACKNOWLEDGEMENT

    def get_warmup_model_answer(self) -> str:
        return WARMUP_MODEL_ANSWER

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

        yield "⏱️ Your 30-minute session has started. The timer is shown on the left."  # timer sentinel — MutationObserver watches for this
        yield from self._stream_framework_presentation()

        yield (
            "\n\n---\n\n"
            "📖 *When you are satisfied with the analysis, click "
            "**📊 Get Summary & End Session** to finish. "
            "Note that ending the session cannot be undone.*"
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
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=(
                    "Please present the full structured framework for this case."
                ))]
            )
        )

        self._pending = True
        full_reply = []
        try:
            for chunk in client.models.generate_content_stream(
                model=MAIN_MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_prompt(),
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)

            was_injected   = self.concept_swap.is_injected
            injected_reply = self.concept_swap.maybe_inject(reply)
            if injected_reply != reply:
                yield injected_reply[len(reply):]
            reply = injected_reply

            if self.concept_swap.is_injected and not was_injected:
                self.concept_swap.log_presented()

            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )

            update_answer(self.session_id, reply)
            log_agent_response(self.session_id, reply)

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

    # ══════════════════════════════════════════════════════════════════════
    # Main phase streaming
    # Change log: 2026-05-12 — override first, swap gated on is_injected AND not override
    # Change log: 2026-05-16 — skip log_memory_override for concept_added
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 1. Override detection first ────────────────────────────────────
        override = self._detect_override(user_input)

        # ── 2. Check if user explicitly removed the wrong concept ──────────
        if (override and
                override["type"] == "concept_excluded" and
                override.get("detail") and
                not self.concept_swap.is_detected and
                self.concept_swap.is_injected):
            wrong        = self.concept_swap.config["wrong_concept"]
            detail_lower = override["detail"].lower()
            wrong_lower  = wrong.lower()
            if wrong_lower in detail_lower or detail_lower in wrong_lower:
                self.concept_swap.force_detected()
                log_memory_override(
                    self.session_id,
                    old_context=f"included: {wrong}",
                    new_context=f"user removed wrong concept explicitly: {wrong}",
                )
                print(f"[CONCEPT SWAP] detected via explicit removal")

        # ── 3. Swap detection — only if injected AND no override ───────────
        cs_detected = False
        if self.concept_swap.is_injected and not override:
            cs_detected = self.concept_swap.check_detection(user_input)
            if cs_detected:
                log_memory_override(
                    self.session_id,
                    old_context=f"included: {self.concept_swap.config['wrong_concept']}",
                    new_context=f"user rejected: {self.concept_swap.config['wrong_concept']}",
                )
                print(f"[CONCEPT SWAP] detected — exclusion active from next response")

        # ── 4. Override handling ───────────────────────────────────────────
        if override:
            if override["type"] != "concept_added":
                log_memory_override(
                    self.session_id,
                    old_context=f"override_type: {override['type']}",
                    new_context=f"detail: {override['detail'] or 'n/a'}",
                )
            print(f"[OVERRIDE] type={override['type']}, "
                  f"detail={override['detail']}, "
                  f"confidence={override['confidence']}")

            if override["type"] == "redo":
                if self.concept_swap.is_detected:
                    self.history = self._strip_concept_swap_from_history()
                yield "Noted! Let me generate a fresh answer...\n\n"
                user_input = (
                    f"Please generate a completely fresh structured "
                    f"answer for this case:\n\n{self.original_case}"
                )
                log_user_message(self.session_id, "[REDO TRIGGERED]")

            elif override["type"] == "concept_added" and override.get("detail"):
                new_concept = override["detail"]
                log_concept_added(self.session_id, new_concept)
                print(f"[CONCEPT ADDED] '{new_concept}' — LLM will incorporate via history")

        self._update_kg_if_framework_mentioned(user_input)

        log_user_message(self.session_id, user_input)
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
                    system_instruction=self._build_system_prompt(),
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)

            was_injected   = self.concept_swap.is_injected
            injected_reply = self.concept_swap.maybe_inject(reply)
            if injected_reply != reply:
                yield injected_reply[len(reply):]
            reply = injected_reply

            if self.concept_swap.is_injected and not was_injected:
                self.concept_swap.log_presented()

            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )
            if self._is_answer(reply):
                update_answer(self.session_id, reply)
            log_agent_response(self.session_id, reply)

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

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
            f"For each primary bucket, list its leaf concepts as bold sub-headers.\n"
            f"Under each leaf concept, copy the EXACT analytical questions from our "
            f"conversation — do not paraphrase or omit them.\n"
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
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{ANSWER_CLASSIFIER_PROMPT}\n\nAgent response: \"{text[:800]}\"",
            )
            parsed = json.loads(self._strip_fences(response.text))
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
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{OVERRIDE_CLASSIFIER_PROMPT}\n\nUser message: \"{user_input}\"",
            )
            parsed = json.loads(self._strip_fences(response.text))
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

    def _check_duplicate(self, concept: str, existing_concepts: list) -> dict:
        """
        Three-layer duplicate guard for concept_added path.
        Layer 2: exact string match (Python, no LLM)
        Layer 3: fuzzy LLM check on gemini-3.1-flash-lite
        Change log: 2026-05-12
        Change log: 2026-05-16 — updated model to gemini-3.1-flash-lite (2.0 deprecated)
        """
        # Layer 2 — exact string match
        for existing in existing_concepts:
            if concept.strip().lower() == existing.strip().lower():
                print(f"[DUPLICATE] exact match: '{concept}' == '{existing}'")
                return {"is_duplicate": True, "matched_concept": existing}

        # Layer 3 — LLM fuzzy check, different model family from classifier
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=(
                    f"You are checking if a user-suggested concept is essentially "
                    f"the same as an existing concept.\n\n"
                    f"User suggested: \"{concept}\"\n\n"
                    f"Existing concepts: {existing_concepts}\n\n"
                    f"Reply with JSON only, no markdown:\n"
                    f"{{\"is_duplicate\": true or false, "
                    f"\"matched_concept\": \"exact string from list or null\"}}\n\n"
                    f"is_duplicate=true ONLY if clearly the same concept — "
                    f"same topic, possibly different wording.\n"
                    f"Examples: 'revenue' matches 'Price per Unit' → false. "
                    f"'unit sales' matches 'Units Sold' → true.\n"
                    f"WHEN IN DOUBT: is_duplicate=false."
                ),
            )
            parsed = json.loads(self._strip_fences(response.text))
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
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return text