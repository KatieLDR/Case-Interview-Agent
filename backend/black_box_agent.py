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
from backend.knowledge_graph import KnowledgeGraph

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Model config ───────────────────────────────────────────────────────────
MAIN_MODEL       = "gemini-2.5-flash"
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

# ── System Prompt ──────────────────────────────────────────────────────────
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
You are an intent classifier for a case interview tool researching user sense of control.

Determine whether the user's message is actively steering or changing WHAT the agent says
— i.e. the content, direction, or perspective of the answer.

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
- "let me try again" → {"intent": true, "confidence": 0.98}
- "add a new perspective" → {"intent": true, "confidence": 0.96}
- "suggest another angle" → {"intent": true, "confidence": 0.96}
- "I'm not satisfied, try a different approach" → {"intent": true, "confidence": 0.97}
- "use a different framework" → {"intent": true, "confidence": 0.97}
- "rethink your answer" → {"intent": true, "confidence": 0.96}
- "make it shorter" → {"intent": false, "confidence": 0.98}
- "explain more simply" → {"intent": false, "confidence": 0.98}
- "can you elaborate on that?" → {"intent": false, "confidence": 0.97}
- "what is the market size?" → {"intent": false, "confidence": 0.99}
"""

OVERRIDE_CLASSIFIER_PROMPT = """
You are an intent classifier for a case interview tool.

Determine whether the user's message intends to:
- Reset or restart with a completely new case
- Discard the current case entirely

Respond ONLY with a valid JSON object, no explanation, no markdown:
{"intent": true or false, "confidence": float between 0.0 and 1.0}

Examples:
- "let's start over with a new case" → {"intent": true, "confidence": 0.99}
- "forget this case, give me a new one" → {"intent": true, "confidence": 0.98}
- "let's do this again" → {"intent": false, "confidence": 0.97}
- "what is the market size?" → {"intent": false, "confidence": 0.99}
"""

# ── Thresholds ─────────────────────────────────────────────────────────────
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
- A response with "## Recommended Framework" and numbered breakdown → {"is_answer": true, "confidence": 0.99}
- A response with "H1: hypothesis..." → {"is_answer": true, "confidence": 0.97}
- "Great point! Let's dig deeper into costs." → {"is_answer": false, "confidence": 0.99}
- "Can you clarify what you mean?" → {"is_answer": false, "confidence": 0.99}
"""

ANSWER_THRESHOLD = 0.90
REDO_THRESHOLD = 0.95
OVERRIDE_THRESHOLD = 0.95

class BlackBoxAgent:
    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="black_box")
        self.original_case = get_case("black_box")
        self._pending      = False

        # Load Knowledge Graph Context
        kg = KnowledgeGraph()
        kg_result = kg.get_context_for_case(self.original_case)
        kg.close()
        self.kg_context = kg_result

        # Pre-load case into history so agent knows it's already been presented
        self.history = [
            types.Content(role="user", parts=[types.Part(text=self.original_case)]),
            types.Content(role="model", parts=[types.Part(text="I have received the case and am ready to provide a structured reference answer.")]),
        ]

    def get_opening_message(self) -> str:
        """Return the case presentation message for Chainlit to display on start."""
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
            f"---\n"
            f"Take your time to read it. Feel free to ask questions or request a structured answer!"
        )
    
    def _build_system_prompt(self) -> str:
        """Inject KG context at the top of system prompt."""
        kg_text = f"""
    ─── KNOWLEDGE GRAPH CONTEXT ───────────────────────────────────────────────
    Case Type: {self.kg_context['case_type']}

    Relevant Frameworks and Concepts:
    """
        for framework, concepts in self.kg_context['context'].items():
            if concepts:
                kg_text += f"- {framework}: {', '.join(concepts)}\n"
            else:
                kg_text += f"- {framework}\n"

        kg_text += "────────────────────────────────────────────────────────────\n"

        return kg_text + SYSTEM_PROMPT    
    
    # ── Streaming chat method ──────────────────────────────────────────────
    def stream_message(self, user_input: str):
        """Generator that yields text chunks as Gemini streams them."""
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # 1. Check for redo intent → fetch case from Firestore, regenerate silently
        if self._is_redo(user_input):
            case = self.original_case
            yield "Noted! Let me generate a fresh answer for your case...\n\n"
            user_input = f"Please generate a completely fresh structured answer for this case:\n\n{case}"

            log_user_message(self.session_id, "[REDO TRIGGERED]")
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
                self.history.append(
                    types.Content(role="model", parts=[types.Part(text=reply)])
                )
                # Store answer and log override — redo intent confirmed
                update_answer(self.session_id, reply)
                log_memory_override(self.session_id, old_context="[redo]", new_context=reply[:200])
                log_agent_response(self.session_id, reply)
            except Exception as e:
                yield f"Sorry, I encountered an error: {str(e)}"
            finally:
                self._pending = False
            return

        # 2. Normal message flow
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
            # Only store if response is a structured framework answer
            if self._is_answer(reply):
                update_answer(self.session_id, reply)
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
    def _is_answer(self, text: str) -> bool:
        """Detect if agent response is a structured framework-based answer."""
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{ANSWER_CLASSIFIER_PROMPT}\n\nAgent response: \"{text[:800]}\"",
            )
            parsed = json.loads(response.text.strip())
            result = parsed.get("is_answer", False) and parsed.get("confidence", 0.0) >= ANSWER_THRESHOLD
            print(f"[ANSWER CLASSIFIER] is_answer={parsed.get('is_answer')}, confidence={parsed.get('confidence')}, stored={result}")
            return result
        except Exception as e:
            print(f"[ANSWER CLASSIFIER] error: {e}")
            return False

    def _is_redo(self, text: str) -> bool:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{REDO_CLASSIFIER_PROMPT}\n\nUser message: \"{text}\"",
            )
            parsed = json.loads(response.text.strip())
            result = parsed.get("intent", False) and parsed.get("confidence", 0.0) >= REDO_THRESHOLD
            print(f"[REDO] intent={parsed.get('intent')}, confidence={parsed.get('confidence')}, triggered={result}")
            return result
        except Exception as e:
            print(f"[REDO] classifier error: {e}")
            return False

    def _is_override(self, text: str) -> bool:
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{OVERRIDE_CLASSIFIER_PROMPT}\n\nUser message: \"{text}\"",
            )
            parsed = json.loads(response.text.strip())
            return parsed.get("intent", False) and parsed.get("confidence", 0.0) >= OVERRIDE_THRESHOLD
        except Exception:
            return False

    def _summarize_history(self) -> str:
        lines = []
        for msg in self.history:
            role = msg.role.upper()
            text = msg.parts[0].text if msg.parts else ""
            lines.append(f"[{role}]: {text[:100]}")
        return "\n".join(lines)