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

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Model config ───────────────────────────────────────────────────────────
MAIN_MODEL       = "gemini-2.5-flash"
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a BCG-style case interview reference assistant. Unlike an interviewer,
your role is to provide the user with a high-quality reference answer they can
study, compare against, and explore further.

─── WHEN PRESENTING A CASE ────────────────────────────────────────────────
Always structure your response in exactly this format:

## 🧠 Case Type & Core Question
Identify the case type and restate the core business problem in one sentence.

## 🔑 Key Hypotheses to Test
List 3-5 specific, hypothesis-driven statements the candidate should investigate.
Format: "H1: [Hypothesis]"

## 🗂️ Recommended Framework
State the best framework for this case and explain briefly why it fits.
Then show the full framework as a structured tree or numbered breakdown.

## 📝 Sample Structured Answer
Write a complete, well-structured sample answer as if delivered by a top BCG
candidate. Use clear signposting (e.g. "First...", "Moving to...", "In summary...").

## ⚠️ Common Mistakes to Avoid
List 3-4 specific mistakes candidates typically make on this type of case.

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

# ── Thresholds ─────────────────────────────────────────────────────────────
REDO_THRESHOLD   = 0.95
ANSWER_THRESHOLD = 0.90


class BlackBoxAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self._pending      = False
        self.swap = ConceptSwap(self.session_id, agent_type="black_box")

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

    def get_opening_message(self) -> str:
        """Return the case presentation message for Chainlit."""
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
            f"---\n"
            f"Take your time to read it. "
            f"Feel free to ask questions or request a structured answer!"
        )

    # ── Streaming chat method ──────────────────────────────────────────────
    def stream_message(self, user_input: str):
        """Generator that yields text chunks as Gemini streams them."""
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # 1. Check for redo intent
        if self._is_redo(user_input):
            yield "Noted! Let me generate a fresh answer for your case...\n\n"
            user_input = (
                f"Please generate a completely fresh structured "
                f"answer for this case:\n\n{self.original_case}"
            )
            log_user_message(self.session_id, "[REDO TRIGGERED]")
            self.history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_input)]
                )
            )
            self._pending = True
            full_reply = []
            try:
                system = SYSTEM_PROMPT + self.swap.get_instruction()
                for chunk in client.models.generate_content_stream(
                    model=MAIN_MODEL,
                    contents=self.history,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                    ),
                ):
                    token = chunk.text or ""
                    full_reply.append(token)
                    yield token

                reply = "".join(full_reply)
                self.history.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=reply)]
                    )
                )
                update_answer(self.session_id, reply)
                log_memory_override(
                    self.session_id,
                    old_context="[redo]",
                    new_context=reply[:200]
                )
                log_agent_response(self.session_id, reply)
            except Exception as e:
                yield f"Sorry, I encountered an error: {str(e)}"
            finally:
                self._pending = False
            return

        # 2. Check swap detection on every user message
        self.swap.check_message(user_input)

        # 3. Normal message flow
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )
        )

        self._pending = True
        full_reply = []
        try:
            system = SYSTEM_PROMPT + self.swap.get_instruction()
            for chunk in client.models.generate_content_stream(
                model=MAIN_MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)
            self.history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=reply)]
                )
            )
            if self._is_answer(reply):
                update_answer(self.session_id, reply)
                if not self.swap.detected:
                    self.swap.log_presented()
            log_agent_response(self.session_id, reply)

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

    # ── Non-streaming fallback (used for summary) ──────────────────────────
    def send_message(self, user_input: str) -> str:
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text=user_input)]
            )
        )
        try:
            response = client.models.generate_content(
                model=MAIN_MODEL,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            reply = response.text
            self.history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=reply)]
                )
            )
        except Exception as e:
            reply = f"Sorry, I encountered an error: {str(e)}"
        log_agent_response(self.session_id, reply)
        return reply

    # ── Session control ────────────────────────────────────────────────────
    def end_session(self) -> None:
        self.swap.check_history(self._summarize_history())
        end_session(self.session_id)

    # ── Intent classifiers ─────────────────────────────────────────────────
    def _is_answer(self, text: str) -> bool:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=(
                    f"{ANSWER_CLASSIFIER_PROMPT}\n\n"
                    f"Agent response: \"{text[:800]}\""
                ),
            )
            parsed = json.loads(response.text.strip())
            result = (
                parsed.get("is_answer", False)
                and parsed.get("confidence", 0.0) >= ANSWER_THRESHOLD
            )
            print(
                f"[ANSWER CLASSIFIER] is_answer={parsed.get('is_answer')}, "
                f"confidence={parsed.get('confidence')}, stored={result}"
            )
            return result
        except Exception as e:
            print(f"[ANSWER CLASSIFIER] error: {e}")
            return False

    def _is_redo(self, text: str) -> bool:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=(
                    f"{REDO_CLASSIFIER_PROMPT}\n\n"
                    f"User message: \"{text}\""
                ),
            )
            parsed = json.loads(response.text.strip())
            result = (
                parsed.get("intent", False)
                and parsed.get("confidence", 0.0) >= REDO_THRESHOLD
            )
            print(
                f"[REDO] intent={parsed.get('intent')}, "
                f"confidence={parsed.get('confidence')}, triggered={result}"
            )
            return result
        except Exception as e:
            print(f"[REDO] classifier error: {e}")
            return False

    # ── Memory helpers ─────────────────────────────────────────────────────
    def _summarize_history(self) -> str:
        lines = []
        for msg in self.history:
            role = msg.role.upper()
            text = msg.parts[0].text if msg.parts else ""
            lines.append(f"[{role}]: {text[:100]}")
        return "\n".join(lines)