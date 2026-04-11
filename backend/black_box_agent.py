import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from backend.logger import (
    create_session, end_session, stamp_started_at,
    log_user_message, log_agent_response,
    log_interruption, log_memory_override,
    update_answer,
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
CASE_TYPE = "Market Entry"

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

# ── Clarification phase system prompt ─────────────────────────────────────
# Used only during self.phase == "clarification".
# Agent answers strictly from the facts sheet, infers when related,
# deflects politely when out of scope.
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

# ── Main phase system prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a strategic consultant specializing in structured frameworks. Your goal
is to provide a concise, high-level logical breakdown of business problems.

STRICT OUTPUT FORMAT — follow this exactly:

**Core Question**
One single question the framework aims to solve.

**The Framework**
**Primary Bucket 1**
- Sub-bucket detail
- Sub-bucket detail

**Primary Bucket 2**
- Sub-bucket detail
- Sub-bucket detail

(continue for all buckets)

**Key Considerations** *(only if relevant)*
- Critical dependency 1
- Critical dependency 2

─── INTERACTION STYLE ─────────────────────────────────────────────────────
You are a reference tool, not an interviewer. After presenting or updating
a framework, ask ONE short natural follow-up question to invite exploration.

If the user questions the framework or asks to change it:
- Briefly explain your reasoning in one sentence
- Ask if they still want to proceed
- If they confirm, honour it immediately

─── RULES ─────────────────────────────────────────────────────────────────
- Always use the exact format above — bold headers, bullet sub-buckets
- Never use numbered lists for framework buckets
- Be direct and concise
- Never evaluate or score the user
- Ask only ONE follow-up question per response
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
- "redo"               : wants a fresh answer or complete regeneration ("redo", "try again", "start over", "different approach")
- "concept_excluded"   : wants to remove a specific concept or bucket ("remove X", "exclude X", "don't include X")
- "framework_switch"   : wants to use a specific named different framework ("use profitability framework", "switch to market sizing")
- "none"               : not steering the output

This does NOT include:
- Asking for a framework for the first time ("give me a framework", "show me the framework")
- Asking follow-up questions ("can you elaborate?", "why is X here?")
- General questions about the case

Respond ONLY with valid JSON, no explanation, no markdown:
{"override": true or false, "type": "redo"|"concept_excluded"|"framework_switch"|"none", "detail": string or null, "confidence": float}

Examples:
- "redo this" → {"override": true, "type": "redo", "detail": null, "confidence": 0.99}
- "try a completely different approach" → {"override": true, "type": "redo", "detail": null, "confidence": 0.97}
- "remove profit per unit" → {"override": true, "type": "concept_excluded", "detail": "Profit per Unit", "confidence": 0.97}
- "use profitability framework" → {"override": true, "type": "framework_switch", "detail": "profitability", "confidence": 0.96}
- "give me a framework" → {"override": false, "type": "none", "detail": null, "confidence": 0.99}
- "show me the framework" → {"override": false, "type": "none", "detail": null, "confidence": 0.99}
- "why is variable cost here?" → {"override": false, "type": "none", "detail": null, "confidence": 0.95}
- "can you elaborate?" → {"override": false, "type": "none", "detail": null, "confidence": 0.98}
"""

# ══════════════════════════════════════════════════════════════════════════
# Thresholds
# ══════════════════════════════════════════════════════════════════════════
ANSWER_THRESHOLD   = 0.90
OVERRIDE_THRESHOLD = 0.85
MAX_TURNS_PER_SESSION = 50

class BlackBoxAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self._pending      = False
        self.turn_count = 0

        # ── Clarification phase ────────────────────────────────────────────
        # Phase starts as "clarification" and switches to "main" when the
        # user clicks "I'm Ready". Facts sheet is loaded from cases.py.
        self.phase               = "clarification"
        self.clarification_facts = get_clarification_facts("black_box")

        # ── Concept Swap experiment ────────────────────────────────────────
        # Active only during main phase. Injected via system prompt.
        self.concept_swap = ConceptSwap(
            agent_type="black_box",
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

        # ── Conversation history ───────────────────────────────────────────
        # Pre-load case so agent knows it's been presented.
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
        """Keyword match — update KG context if user mentions a different framework."""
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
        """
        Builds the system prompt for the clarification phase.
        Injects the facts sheet so the agent answers strictly from it.
        """
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
        """
        Builds the main phase system prompt:
          KG context + Concept Swap block + base SYSTEM_PROMPT
        """
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"

        kg_block = (
            f"─── KNOWLEDGE GRAPH CONTEXT ──────────────────────────────────────────\n"
            f"Case Type : {self.kg_context['case_type']}\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"(These are the correct ordered concepts. Ground your answer here.)\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
        )

        swap_block = self.concept_swap.get_system_prompt_block()

        return kg_block + swap_block + SYSTEM_PROMPT

    # ══════════════════════════════════════════════════════════════════════
    # Opening message
    # ══════════════════════════════════════════════════════════════════════

    def get_opening_message(self) -> str:
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
            f"---\n"
            f"Take your time to read it. Feel free to ask any clarifying questions "
            f"before you begin your analysis.\n\n"
            f"When you're ready to start, click **\"I'm Ready — Let's Start\"** below."
        )

    # ══════════════════════════════════════════════════════════════════════
    # Phase transition — called by app.py when user clicks "I'm Ready"
    # ══════════════════════════════════════════════════════════════════════

    def start_main_phase(self) -> str:
        """
        Transitions agent from clarification → main phase.
        Injects a system marker into history so the model knows the phase changed.
        Returns a confirmation message for Chainlit to display.
        """
        if self.phase == "main":
            return "⚠️ The session is already in progress."

        self.phase = "main"
        stamp_started_at(self.session_id)

        # Inject explicit handover marker into history
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
        return (
            "✅ **Clarification round closed.**\n\n"
            "You can now present your structured framework or ask for a reference answer!"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Main message handler
    # ══════════════════════════════════════════════════════════════════════

    def stream_message(self, user_input: str):
        """
        Routes to clarification or main flow based on self.phase.

        Clarification phase:
          - Answers strictly from facts sheet
          - No concept swap, no override detection, no KG update
          - Uses CLARIFICATION_SYSTEM_PROMPT

        Main phase:
          - Full flow: concept swap + override detection + KG + streaming
          - Uses _build_system_prompt()
        """
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── Route by phase ─────────────────────────────────────────────────
        if self.phase == "clarification":
            yield from self._stream_clarification(user_input)
        else:
            self.turn_count += 1
            if self.turn_count > MAX_TURNS_PER_SESSION:
                yield "⏱️ **Session limit reached.** You've completed the maximum number of turns for this session.\n\nGenerating your summary now..."
                yield from self._auto_end_session()
                return
            yield from self._stream_main(user_input)

    # ══════════════════════════════════════════════════════════════════════
    # Clarification phase streaming
    # ══════════════════════════════════════════════════════════════════════

    def _stream_clarification(self, user_input: str):
        """Answer from facts sheet only. No swap, no override, no KG."""
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
    # Main phase streaming
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        """
        Full main phase flow:
          1. Concept Swap detection
          2. Override detection → log for research
          3. KG context update
          4. Stream response
          5. Post-stream: concept swap injection tracking
        """
        # ── 1. Concept Swap detection ──────────────────────────────────────
        cs_detected = self.concept_swap.check_detection(user_input)
        if cs_detected:
            log_memory_override(
                self.session_id,
                old_context=f"included: {self.concept_swap.config['wrong_concept']}",
                new_context=f"user rejected: {self.concept_swap.config['wrong_concept']}",
            )
            print(f"[CONCEPT SWAP] detected — exclusion active from next response")

        # ── 2. Override detection → log for research ───────────────────────
        override = self._detect_override(user_input)
        if override:
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

        # ── 3. KG context update ───────────────────────────────────────────
        self._update_kg_if_framework_mentioned(user_input)

        # ── 4. Stream response ─────────────────────────────────────────────
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

            # ── 5. Concept Swap injection tracking ─────────────────────────
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
    # ══════════════════════════════════════════════════════════════════════

    def send_message(self, user_input: str) -> str:
        """Non-streaming fallback used for summary."""
        log_user_message(self.session_id, user_input)

        summary_prompt = (
            f"Based on our conversation, provide a summary in this exact format:\n\n"
            f"**Final Framework: [Framework Name]**\n\n"
            f"**The Framework:**\n"
            f"- Use bolding for Primary Buckets\n"
            f"- Use bullet points for specific Sub-buckets under each bucket\n\n"
            f"Then in 2-3 sentences: note any concepts the user removed and any "
            f"framework switches made during the session."
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

        if not self.concept_swap.is_detected:
            detected_at_end = self.concept_swap.check_detection(final_framework)
            print(f"[END SESSION] final concept swap check: detected={detected_at_end}")
        else:
            detected_at_end = True
            print("[END SESSION] concept swap already detected during chat")

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

    def _auto_end_session(self):
        """Auto-trigger summary + end when turn cap is hit."""
        summary = self.send_message(
            "Please summarise the session in the standard format."
        )
        yield f"\n\n{summary}"
        self.end_session()
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
                    "confidence": parsed["confidence"],
                }
        except Exception as e:
            print(f"[OVERRIDE] error: {e}")
        return None

    # ══════════════════════════════════════════════════════════════════════
    # History helpers
    # ══════════════════════════════════════════════════════════════════════

    def _strip_concept_swap_from_history(self) -> list:
        """Remove Concept Swap traces from history on redo after detection."""
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
    # Moved to BlackBoxAgent so all subclasses inherit it.
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