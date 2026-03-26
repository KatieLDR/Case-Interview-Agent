import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from backend.logger import (
    create_session, end_session,
    log_user_message, log_agent_response,
    log_interruption, log_memory_override,
    update_answer,
)
from backend.cases import get_case
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

# ── System Prompt (base) ───────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a strategic consultant specializing in structured frameworks. Your goal is to provide a concise, high-level logical breakdown of business problems.

STRICT OUTPUT FORMAT:

1. Core Question: One single question the framework aims to solve.

2. The Framework: A nested hierarchy of Buckets and Sub-buckets.

- Use bolding for Primary Buckets (e.g., Risk, Long-term Benefit).
- Use bullet points for specific Sub-buckets (e.g., redeploying staff, automation learnings).

3. Optional Context (Only if relevant):

- Key Considerations: 2-3 bullet points on critical dependencies.
- Framework Strengths: 1 sentence on why this logic works.
- Framework Improvements: 1 sentence on how to make it more robust.

─── FOLLOW-UP INTERACTIONS ────────────────────────────────────────────────
After the first response, the user may:
- Ask free-form questions about any part of the answer
- Ask for a deeper dive on a specific section
- Ask for alternative frameworks or approaches

Answer all follow-ups directly and concisely. Do NOT switch into interviewer
or coaching mode — you are a reference tool, not an interviewer.
Never ask the user questions back. Never guide or evaluate the user.

─── RULES ─────────────────────────────────────────────────────────────────
- Always lead with structure before detail
- Be direct — this is a reference answer, not a conversation
- Never ask follow-up questions to the user
- Never evaluate or score the user's responses
- Keep sample answers realistic and concise — as if spoken in an interview
- Focus on quantifiable or actionable sub-buckets.
- No long-winded paragraphs.
"""

# ── Classifier prompts ─────────────────────────────────────────────────────
REDO_CLASSIFIER_PROMPT = """
You are an intent classifier for a case interview tool researching user sense
of control.

Determine whether the user's message is actively steering or changing WHAT
the agent says — i.e. the content, direction, or perspective of the answer.

This includes:
- Explicit redo / retry requests
- Requests to use a different framework or approach
- Requests to add a new perspective or angle
- Expressing dissatisfaction and wanting something different
- Asking the agent to reconsider or rethink its answer

This does NOT include:
- Formatting requests ("make it shorter", "use bullet points")
- Style requests ("explain more simply", "be more concise")
- Passive follow-ups ("can you elaborate?", "what do you mean?")
- Clarification questions about the case

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"intent": true or false, "confidence": float between 0.0 and 1.0}

Examples:
- "can we redo this?" → {"intent": true, "confidence": 0.99}
- "add a new perspective" → {"intent": true, "confidence": 0.96}
- "use a different framework" → {"intent": true, "confidence": 0.97}
- "make it shorter" → {"intent": false, "confidence": 0.98}
- "can you elaborate on that?" → {"intent": false, "confidence": 0.97}
- "what is the market size?" → {"intent": false, "confidence": 0.99}
"""

ANSWER_CLASSIFIER_PROMPT = """
You are a classifier for a case interview tool.

Determine whether the following agent response contains a structured,
framework-based answer to a business case. It must include at least one of:
- A recommended framework (e.g. MECE tree, profitability tree, issue tree)
- Key hypotheses laid out in a structured format
- A structured sample answer with clear signposting

Short follow-up replies, clarifying questions, or feedback messages do NOT qualify.

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"is_answer": true or false, "confidence": float between 0.0 and 1.0}

Examples:
- A response with "## Recommended Framework"
  → {"is_answer": true, "confidence": 0.99}
- "Great point! Let's dig deeper into costs."
  → {"is_answer": false, "confidence": 0.99}
"""

EXCLUDE_CLASSIFIER_PROMPT = """
You are a classifier for a case interview tool.

Determine whether the user's message explicitly asks to remove, exclude, or
not include a specific concept or element from the framework or analysis.

If detected, extract the exact concept or topic name the user wants excluded.

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"excluded": true or false, "concept": string or null, "confidence": float between 0.0 and 1.0}

Examples:
- "don't include profit per unit" → {"excluded": true, "concept": "Profit per Unit", "confidence": 0.98}
- "remove market share from the analysis" → {"excluded": true, "concept": "Market Share", "confidence": 0.97}
- "exclude distribution challenges" → {"excluded": true, "concept": "Distribution Challenges", "confidence": 0.96}
- "I don't think we need fixed costs here" → {"excluded": true, "concept": "Fixed Costs", "confidence": 0.88}
- "can you elaborate on market size?" → {"excluded": false, "concept": null, "confidence": 0.98}
- "use a different framework" → {"excluded": false, "concept": null, "confidence": 0.96}
- "what is the market size?" → {"excluded": false, "concept": null, "confidence": 0.99}
"""

# ── Thresholds ─────────────────────────────────────────────────────────────
REDO_THRESHOLD    = 0.95
ANSWER_THRESHOLD  = 0.90
EXCLUDE_THRESHOLD = 0.85


class BlackBoxAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self._pending      = False
        self.swap          = ConceptSwap(agent_type="black_box", session_id=self.session_id)

        # ── KG context ────────────────────────────────────────────────────
        # Fetched once at init for the default case type.
        # Re-fetched per message if user mentions a different framework.
        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        print(f"[KG INIT] case_type={CASE_TYPE}, "
              f"framework={self.kg_context['framework']}, "
              f"concepts={self.kg_context['concepts']}")

        # Keyword map for fast per-message framework detection (no LLM needed)
        self._kg_framework_keywords = {
            "Economic Feasibility":    ["economic feasibility", "market entry", "market potential"],
            "Expanded Profit Formula": ["profit formula", "profitability", "revenue", "cost tree"],
            "Four-Pronged Strategy":   ["four-pronged", "pricing strategy", "price elasticity"],
            "Formulaic Breakdown":     ["formulaic breakdown", "guesstimate", "market sizing"],
            "Customized Issue Trees":  ["issue tree", "unconventional", "internal external"],
        }

        # Pre-load case into history so agent knows it's already been presented
        self.history = [
            types.Content(
                role="user",
                parts=[types.Part(text=self.original_case)]
            ),
            types.Content(
                role="model",
                parts=[types.Part(text=(
                    "I have received the case and am ready "
                    "to provide a structured reference answer."
                ))]
            ),
        ]

    # ══════════════════════════════════════════════════════════════════════
    # KG helpers
    # ══════════════════════════════════════════════════════════════════════

    def _fetch_kg_context(self, case_type: str) -> dict:
        """Fetch framework + ordered concepts from Neo4j for the given case type."""
        framework = kg.get_framework_for_case(case_type) or "Unknown Framework"
        concepts  = kg.get_ordered_concepts(framework) if framework else []
        return {"case_type": case_type, "framework": framework, "concepts": concepts}

    def _maybe_update_kg_context(self, user_input: str) -> None:
        """
        Keyword match user message against known framework names.
        If a different framework is mentioned, re-fetch KG context.
        """
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
                    print(f"[KG UPDATE] switched to framework={framework_name}, "
                          f"concepts={concepts}")
                break

    def _build_system_prompt(self) -> str:
        """
        Assemble system prompt fresh on every Gemini call:
          1. KG context block — correct framework + ordered concepts
          2. Swap exclusion block — only injected after swap is detected
          3. Base SYSTEM_PROMPT

        User concept exclusions (e.g. "don't include Profit per Unit") are NOT
        injected here. The model handles them naturally via conversation history.
        Only the swap concept needs an explicit exclusion instruction because it
        was injected outside the model's awareness.
        """
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"

        kg_block = (
            f"─── KNOWLEDGE GRAPH CONTEXT ──────────────────────────────────────────\n"
            f"Case Type : {self.kg_context['case_type']}\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"(These are the CORRECT, ordered concepts for this case. "
            f"Ground your answer in this structure.)\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
        )

        # Swap block toggles automatically:
        # before detection → injection instruction (include wrong concept)
        # after detection  → exclusion instruction (never mention it again)
        swap_block = self.swap.get_system_prompt_block()

        return kg_block + swap_block + SYSTEM_PROMPT

    # ══════════════════════════════════════════════════════════════════════
    # Opening message
    # ══════════════════════════════════════════════════════════════════════

    def get_opening_message(self) -> str:
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
            f"---\n"
            f"Take your time to read it. "
            f"Feel free to ask questions or request a structured answer!"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Main message handler
    # ══════════════════════════════════════════════════════════════════════

    def stream_message(self, user_input: str):
        """
        Generator that yields text chunks as Gemini streams them.

        Per-message order:
          1. check_detection   — did user catch the swap?
                                 if yes: clean history immediately
          2. _extract_excluded — did user ask to remove any concept?
                                 if yes: log memory_override only
          3. _is_redo          — does user want a fresh answer?
          4. _maybe_update_kg  — did user mention a different framework?
          5. stream → maybe_inject → append to history → log
        """
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # ── 1. Swap detection ──────────────────────────────────────────────
        # History stays intact — model needs full context to understand
        # the conversation arc around the swap concept.
        # System prompt switches to exclusion instruction automatically
        # on the next call via get_system_prompt_block().
        just_detected = self.swap.check_detection(user_input)
        if just_detected:
            print(f"[SWAP] detected — exclusion instruction active from next response")

        # ── 2. Concept exclusion — log only ────────────────────────────────
        # User exclusions are handled naturally by the model via history.
        # We only log here for the memory_override research metric.
        excluded = self._extract_excluded_concept(user_input)
        if excluded:
            log_memory_override(
                self.session_id,
                old_context=f"included: {excluded}",
                new_context=f"excluded: {excluded}",
            )
            print(f"[EXCLUDE] logged user exclusion: '{excluded}'")

        # ── 3. Redo intent ─────────────────────────────────────────────────
        if self._is_redo(user_input):
            yield "Noted! Let me generate a fresh answer for your case...\n\n"
            log_user_message(self.session_id, "[REDO TRIGGERED]")

            redo_input = (
                f"Please generate a completely fresh structured "
                f"answer for this case:\n\n{self.original_case}"
            )
            self.history.append(
                types.Content(role="user", parts=[types.Part(text=redo_input)])
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

                # Inject swap (skips if already detected).
                # Yield the injected tail so the user sees it.
                was_injected_before = self.swap.is_injected
                injected_reply = self.swap.maybe_inject(reply)
                if injected_reply != reply:
                    tail = injected_reply[len(reply):]
                    yield tail
                reply = injected_reply

                if self.swap.is_injected and not was_injected_before:
                    self.swap.log_presented()

                self.history.append(
                    types.Content(role="model", parts=[types.Part(text=reply)])
                )
                update_answer(self.session_id, reply)
                log_memory_override(
                    self.session_id,
                    old_context="[redo triggered]",
                    new_context=reply[:200],
                )
                log_agent_response(self.session_id, reply)

            except Exception as e:
                yield f"Sorry, I encountered an error: {str(e)}"
            finally:
                self._pending = False
            return

        # ── 4. KG context update ───────────────────────────────────────────
        self._maybe_update_kg_context(user_input)

        # ── 5. Normal message flow ─────────────────────────────────────────
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

            # Inject swap into completed response (skips if already detected).
            # If injection added content, yield it so the user sees it too.
            was_injected_before = self.swap.is_injected
            injected_reply = self.swap.maybe_inject(reply)
            if injected_reply != reply:
                tail = injected_reply[len(reply):]
                yield tail
            reply = injected_reply

            if self.swap.is_injected and not was_injected_before:
                self.swap.log_presented()

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
    # Non-streaming fallback (used for summary)
    # ══════════════════════════════════════════════════════════════════════

    def send_message(self, user_input: str) -> str:
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
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
        """
        Called when user clicks 'Get Summary & End Session'.
        1. Walk history in reverse — find last structured framework response.
        2. Run one final swap detection pass on it.
        3. Stamp Firestore: final_framework, swap_detected, swap_detected_at_end.
        4. Close the session.
        """
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
            print("[END SESSION] no structured framework found — using last model message")

        if not self.swap.is_detected:
            detected_at_end = self.swap.check_detection(final_framework)
            print(f"[END SESSION] final swap check: detected={detected_at_end}")
        else:
            detected_at_end = True
            print("[END SESSION] swap already detected during chat")

        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection("sessions").document(self.session_id).update({
                "final_framework":      final_framework[:1000],
                "swap_detected":        self.swap.is_detected,
                "swap_detected_at_end": detected_at_end,
            })
            print(f"[END SESSION] Firestore stamped for session={self.session_id}")
        except Exception as e:
            print(f"[END SESSION] Firestore stamp failed: {e}")

        end_session(self.session_id)

    # ══════════════════════════════════════════════════════════════════════
    # Classifiers
    # ══════════════════════════════════════════════════════════════════════

    def _is_answer(self, text: str) -> bool:
        """Is the agent reply a structured framework answer?"""
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{ANSWER_CLASSIFIER_PROMPT}\n\nAgent response: \"{text[:800]}\"",
            )
            parsed = json.loads(self._strip_fences(response.text))
            result = (
                parsed.get("is_answer", False)
                and parsed.get("confidence", 0.0) >= ANSWER_THRESHOLD
            )
            print(f"[ANSWER] is_answer={parsed.get('is_answer')}, "
                  f"confidence={parsed.get('confidence')}, stored={result}")
            return result
        except Exception as e:
            print(f"[ANSWER] error: {e}")
            return False

    def _is_redo(self, text: str) -> bool:
        """Did the user ask to regenerate or try a different approach?"""
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{REDO_CLASSIFIER_PROMPT}\n\nUser message: \"{text}\"",
            )
            parsed = json.loads(self._strip_fences(response.text))
            result = (
                parsed.get("intent", False)
                and parsed.get("confidence", 0.0) >= REDO_THRESHOLD
            )
            print(f"[REDO] intent={parsed.get('intent')}, "
                  f"confidence={parsed.get('confidence')}, triggered={result}")
            return result
        except Exception as e:
            print(f"[REDO] error: {e}")
            return False

    def _extract_excluded_concept(self, user_input: str) -> str | None:
        """
        Did the user ask to remove a specific concept?
        Returns concept name for logging only.
        The model handles the actual exclusion via conversation history.
        """
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{EXCLUDE_CLASSIFIER_PROMPT}\n\nUser message: \"{user_input}\"",
            )
            parsed = json.loads(self._strip_fences(response.text))
            if (parsed.get("excluded", False)
                    and parsed.get("confidence", 0.0) >= EXCLUDE_THRESHOLD
                    and parsed.get("concept")):
                concept = parsed["concept"]
                print(f"[EXCLUDE] concept='{concept}', "
                      f"confidence={parsed.get('confidence')}")
                return concept
        except Exception as e:
            print(f"[EXCLUDE] error: {e}")
        return None

    # ══════════════════════════════════════════════════════════════════════
    # History helpers
    # ══════════════════════════════════════════════════════════════════════

    def _strip_swap_from_history(self) -> list:
        """
        Remove all traces of the concept swap from in-memory history.
        Called immediately when swap is detected.
        Firestore retains the full unmodified record.

        Removes from each model message:
          1. The injection note tail (from '---\\n💡' onward)
          2. Any list line containing the wrong concept name
        """
        note_marker = "---\n💡"
        wrong       = self.swap.config["wrong_concept"].lower()
        cleaned     = []

        for msg in self.history:
            if msg.role == "model" and msg.parts:
                text = msg.parts[0].text

                if note_marker in text:
                    text = text[:text.index(note_marker)].rstrip()

                lines = [l for l in text.split("\n") if wrong not in l.lower()]
                text  = "\n".join(lines)

                msg = types.Content(
                    role="model",
                    parts=[types.Part(text=text)]
                )
            cleaned.append(msg)

        print(f"[STRIP] swap removed from history ({len(cleaned)} messages)")
        return cleaned

    def _summarize_history(self) -> str:
        lines = []
        for msg in self.history:
            role = msg.role.upper()
            text = msg.parts[0].text if msg.parts else ""
            lines.append(f"[{role}]: {text[:100]}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Strip markdown code fences from Gemini classifier responses."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return text