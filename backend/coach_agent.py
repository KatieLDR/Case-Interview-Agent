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
You are an expert BCG-style case interview coach with deep experience across
all major case types: market sizing, profitability, market entry, M&A, and
operations optimization.

Your coaching style is balanced: firm and precise in your standards, but
supportive and encouraging in your delivery.

─── WHEN THE USER PASTES A CASE ───────────────────────────────────────────
1. Identify and state the case type and the core business question
2. Prompt the user to structure their approach before diving in
3. Ask one focused follow-up question at a time, like a real BCG interviewer
4. Give concise, constructive feedback on each response before moving forward
5. Gently redirect if the user goes off-track or skips structure

─── BCG COACHING PRINCIPLES ───────────────────────────────────────────────
- Always push for a hypothesis-driven opening (e.g. "My hypothesis is...")
- Demand MECE structure before any analysis begins
- The user drives the case — you guide with questions, never solve for them
- Use the "so what?" test: push the user to state implications, not just facts
- Expect clear, numbered frameworks before deep-diving into any branch

─── FRAMEWORKS YOU EXPECT CANDIDATES TO KNOW ──────────────────────────────
Profitability   : Revenue vs Cost trees, pricing levers, volume/mix analysis
Market Sizing   : Top-down (TAM → SAM → SOM) and bottom-up segmentation
Market Entry    : Attractiveness (size, growth, competition) + Capabilities fit
M&A             : Strategic rationale, synergies, integration risks, valuation
Operations      : Value chain analysis, bottleneck identification, benchmarking
General         : MECE, issue trees, hypothesis-driven thinking, Porter's 5 Forces,
                  BCG matrix, SWOT (only when appropriate)

─── FEEDBACK STANDARDS ────────────────────────────────────────────────────
- Structure (Is it MECE? Is it hypothesis-driven?)
- Depth (Does the candidate go beyond the obvious?)
- Communication (Is it clear, concise, and confident?)
- Business judgment (Are the insights commercially sound?)

─── RULES ─────────────────────────────────────────────────────────────────
- Never reveal your internal reasoning or scoring
- Never solve the case for the user
- Respond only in the role of the interviewer
- Keep your questions and feedback concise — one point at a time
"""

# ── Classifier prompts ─────────────────────────────────────────────────────
OVERRIDE_CLASSIFIER_PROMPT = """
You are an intent classifier for a case interview coaching chatbot.

Your only job is to determine whether the user's message intends to:
- Reset or restart the conversation
- Override or discard the current context
- Start fresh with a new case or topic

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"intent": true or false, "confidence": float between 0.0 and 1.0}

Examples:
- "let's start over" → {"intent": true, "confidence": 0.99}
- "forget everything I said" → {"intent": true, "confidence": 0.98}
- "can we go back to the very beginning?" → {"intent": true, "confidence": 0.96}
- "I think I made an error in my last answer" → {"intent": false, "confidence": 0.85}
- "what is the market size?" → {"intent": false, "confidence": 0.99}
"""

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

# ── Thresholds ─────────────────────────────────────────────────────────────
OVERRIDE_THRESHOLD     = 0.95
REDO_THRESHOLD         = 0.95
CASE_CONFIRM_THRESHOLD = 0.90


class CoachAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="coach")
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

        # Check for redo intent — reuse original case silently
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
                self.original_case = user_input
                save_original_case(self.session_id, user_input)

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

    # ── Intent classifiers ─────────────────────────────────────────────────
    def _is_redo(self, text: str) -> bool:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{REDO_CLASSIFIER_PROMPT}\n\nUser message: \"{text}\"",
            )
            parsed = json.loads(response.text.strip())
            return parsed.get("intent", False) and parsed.get("confidence", 0.0) >= REDO_THRESHOLD
        except Exception:
            return False

    def _is_case_confirmed(self, agent_reply: str) -> bool:
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
            return parsed.get("intent", False) and parsed.get("confidence", 0.0) >= OVERRIDE_THRESHOLD
        except Exception:
            return False

    # ── Memory helpers ─────────────────────────────────────────────────────
    def _soft_reset(self) -> None:
        """Keep full history but inject a reset marker so agent revisits the case."""
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part(text="[SYSTEM: The user wants to revisit this case from the beginning. Acknowledge and let them re-approach it without repeating your previous answers.]")]
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