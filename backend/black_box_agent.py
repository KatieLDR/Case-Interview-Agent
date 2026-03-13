import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from backend.logger import (
    create_session, end_session,
    log_user_message, log_agent_response,
    log_interruption, log_memory_override,
    save_original_case,
)

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

─── WHEN THE USER PASTES A CASE ───────────────────────────────────────────
Always structure your first response in exactly this format:

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
- Ask for a deeper dive on a specific section (e.g. "expand on the framework")
- Ask for alternative frameworks or approaches
- Ask "why" questions about your reasoning

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

# ── Override detection config ──────────────────────────────────────────────
REDO_CLASSIFIER_PROMPT = """
You are an intent classifier for a case interview tool.

Determine whether the user's message intends to:
- Redo, retry, or reattempt the current case
- Ask to see the case again or start over with the same case

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"intent": true or false, "confidence": float between 0.0 and 1.0}

Examples:
- "can we redo this?" → {"intent": true, "confidence": 0.99}
- "let me try again" → {"intent": true, "confidence": 0.98}
- "reiterate the case" → {"intent": true, "confidence": 0.97}
- "do it again" → {"intent": true, "confidence": 0.96}
- "what is the market size?" → {"intent": false, "confidence": 0.99}
- "can you explain that?" → {"intent": false, "confidence": 0.99}
"""

CASE_CONFIRMATION_PROMPT = """
You are a classifier for a case interview tool.

Determine whether the following agent response confirms it has received and 
understood a case problem from the user.

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"confirmed": true or false, "confidence": float between 0.0 and 1.0}

Examples:
- "Great, this is a profitability case..." → {"confirmed": true, "confidence": 0.99}
- "I can see this is a market entry case..." → {"confirmed": true, "confidence": 0.98}
- "Sure! Let's go back to the beginning..." → {"confirmed": false, "confidence": 0.99}
- "Can you clarify what you mean?" → {"confirmed": false, "confidence": 0.97}
"""
OVERRIDE_CLASSIFIER_PROMPT = """
You are an intent classifier for a case interview reference tool.

Your only job is to determine whether the user's message intends to:
- Reset or restart the conversation
- Start fresh with a new case

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"intent": true or false, "confidence": float between 0.0 and 1.0}

Examples:
- "let's start over" → {"intent": true, "confidence": 0.99}
- "new case" → {"intent": true, "confidence": 0.97}
- "can you expand on the framework?" → {"intent": false, "confidence": 0.99}
- "why did you choose that structure?" → {"intent": false, "confidence": 0.99}
"""
# ── Thresholds ─────────────────────────────────────────────────────────────
OVERRIDE_THRESHOLD     = 0.95
REDO_THRESHOLD         = 0.95
CASE_CONFIRM_THRESHOLD = 0.90

class BlackBoxAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.history       = []
        self._pending      = False
        self.original_case = None  # stores confirmed case problem

    # ── Streaming chat method ──────────────────────────────────────────────
    def stream_message(self, user_input: str):
        """Generator that yields text chunks as Gemini streams them."""
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        if self._is_override(user_input):
            old_ctx = self._summarize_history()
            self._soft_reset()
            log_memory_override(self.session_id, old_context=old_ctx, new_context="")
            yield "Sure! Let's go back to the beginning of this case. Feel free to re-approach it from scratch."
            return

        # 3. Check for redo intent — reuse original case silently
        if self.original_case and self._is_redo(user_input):
            yield "Restarting with your original case...\n\n"
            self._soft_reset()
            user_input = self.original_case

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
                    system_instruction=SYSTEM_PROMPT,
                ),
            ):
                token = chunk.text or ""
                full_reply.append(token)
                yield token

            reply = "".join(full_reply)
            self.history.append(
                types.Content(role="model", parts=[types.Part(text=reply)])
            )

            # Check if agent just confirmed receiving a case
            if self.original_case is None and self._is_case_confirmed(reply):
                last_user_msg = user_input
                self.original_case = last_user_msg
                save_original_case(self.session_id, last_user_msg)

            log_agent_response(self.session_id, reply)

        except Exception as e:
            yield f"Sorry, I encountered an error: {str(e)}"
        finally:
            self._pending = False

    # ── Non-streaming fallback (used for summary) ──────────────────────────
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
                    system_instruction=SYSTEM_PROMPT,
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

    # ── Session control ────────────────────────────────────────────────────
    def end_session(self) -> None:
        end_session(self.session_id)

    # ── Memory helpers ─────────────────────────────────────────────────────
    def _is_redo(self, text: str) -> bool:
        """Detect if user wants to redo the same case."""
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{REDO_CLASSIFIER_PROMPT}\n\nUser message: \"{text}\"",
            )
            parsed     = json.loads(response.text.strip())
            return parsed.get("intent", False) and parsed.get("confidence", 0.0) >= REDO_THRESHOLD
        except Exception:
            return False

    def _is_case_confirmed(self, agent_reply: str) -> bool:
        """Detect if agent response confirms it received a case problem."""
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{CASE_CONFIRMATION_PROMPT}\n\nAgent response: \"{agent_reply[:500]}\"",
            )
            parsed = json.loads(response.text.strip())
            return parsed.get("confirmed", False) and parsed.get("confidence", 0.0) >= CASE_CONFIRM_THRESHOLD
        except Exception:
            return False

    def _is_override(self, text: str) -> bool:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{OVERRIDE_CLASSIFIER_PROMPT}\n\nUser message: \"{text}\"",
            )
            raw    = response.text.strip()
            parsed = json.loads(raw)
            intent     = parsed.get("intent", False)
            confidence = parsed.get("confidence", 0.0)
            return intent and confidence >= OVERRIDE_THRESHOLD
        except Exception:
            return False

    def _soft_reset(self) -> None:
        """Keep full history but inject a reset marker so agent revisits the case."""
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text="[SYSTEM: The user wants to revisit this case from the beginning. Acknowledge and let them re-approach it without repeating your previous answers.)]")]
            )
        )

    def _reset_memory(self) -> None:
        """Full wipe — kept for internal use if needed."""
        self.history = []

    def _summarize_history(self) -> str:
        lines = []
        for msg in self.history:
            role = msg.role.upper()
            text = msg.parts[0].text if msg.parts else ""
            lines.append(f"[{role}]: {text[:100]}")
        return "\n".join(lines)