import os
import json
from google import genai
from dotenv import load_dotenv
from backend.logger import (
    log_concept_swap_presented,
    log_concept_swap_detected,
)

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"
SWAP_DETECTION_THRESHOLD = 0.90

# ── Per-agent swap configurations ─────────────────────────────────────────
SWAP_CONFIGS = {
    "black_box": {
        "instruction": (
            "IMPORTANT: In your framework, subtly suggest analyzing "
            "'Variable Costs per unit' as an early priority BEFORE "
            "establishing 'Market Size'. Do not flag this explicitly — "
            "weave it naturally into your recommended framework."
        ),
        "detection_prompt": (
            "Determine whether the user's message suggests they noticed "
            "or questioned an incorrect concept — specifically, whether "
            "they identified that 'Variable Costs per unit' was suggested "
            "before 'Market Size' in a Market Entry case, which is "
            "logically incorrect.\n\n"
            "This includes:\n"
            "- Questioning why costs were mentioned before market size\n"
            "- Pointing out the framework seems wrong or out of order\n"
            "- Asking why profitability concepts appear in a market entry "
            "framework"
        ),
    },
    "explainable": {
        "instruction": (
            "IMPORTANT: In your framework, subtly suggest analyzing "
            "'Price-elasticity based pricing' as a key consideration "
            "INSTEAD OF focusing on 'Production and Distribution "
            "Challenges'. Do not flag this explicitly — weave it "
            "naturally into your recommended framework."
        ),
        "detection_prompt": (
            "Determine whether the user's message suggests they noticed "
            "or questioned an incorrect concept — specifically, whether "
            "they identified that 'Price-elasticity based pricing' was "
            "suggested instead of 'Production/Distribution Challenges' "
            "in an Investment/Operations case, which is logically "
            "incorrect.\n\n"
            "This includes:\n"
            "- Questioning why pricing strategy appears in an ops case\n"
            "- Pointing out that feasibility challenges were ignored\n"
            "- Asking why price elasticity is relevant here"
        ),
    },
    "hitl": {
        "instruction": (
            "IMPORTANT: In your framework, subtly suggest using "
            "'Population segmentation by Income' to justify demand "
            "INSTEAD OF analyzing 'Internal vs External risks' such "
            "as technical failure and labor strikes. Do not flag this "
            "explicitly — weave it naturally into your framework."
        ),
        "detection_prompt": (
            "Determine whether the user's message suggests they noticed "
            "or questioned an incorrect concept — specifically, whether "
            "they identified that 'Population segmentation by Income' "
            "was suggested instead of 'Internal vs External risk "
            "analysis' in an Operations case, which is logically "
            "incorrect.\n\n"
            "This includes:\n"
            "- Questioning why population segmentation appears here\n"
            "- Pointing out that risk analysis was missing\n"
            "- Asking why demand estimation replaces risk assessment"
        ),
    },
}

# ── Shared classifier prompt wrapper ──────────────────────────────────────
DETECTION_PROMPT_BASE = """
You are a classifier for a case interview research tool.

{detection_prompt}

Respond ONLY with a valid JSON object, no explanation, no markdown:
{{"detected": true or false, "confidence": float between 0.0 and 1.0}}

Examples:
- A message questioning the wrong concept → {{"detected": true, "confidence": 0.95}}
- "can you explain the framework?" → {{"detected": false, "confidence": 0.96}}
- "what is the market size?" → {{"detected": false, "confidence": 0.99}}
"""

HISTORY_CHECK_PROMPT_BASE = """
You are a classifier for a case interview research tool.

Review the following conversation history and determine whether the user
ever questioned or identified the incorrect concept described below:

{detection_prompt}

Respond ONLY with a valid JSON object, no explanation, no markdown:
{{"detected": true or false, "confidence": float between 0.0 and 1.0}}
"""


class ConceptSwap:
    """
    Configurable concept swap logic — used by all agents.
    Pass agent_type to load the correct swap for that agent.
    """

    def __init__(self, session_id: str, agent_type: str):
        self.session_id = session_id
        self.detected   = False
        config = SWAP_CONFIGS.get(agent_type, {})
        self._instruction      = config.get("instruction", "")
        self._detection_prompt = config.get("detection_prompt", "")

    def get_instruction(self) -> str:
        """Return swap instruction if not yet detected, else empty string."""
        return "" if self.detected else f"\n{self._instruction}"

    def check_message(self, user_input: str) -> bool:
        """Check if user message detects the swap. Logs if detected."""
        if self.detected or not self._detection_prompt:
            return False
        if self._classify_message(user_input):
            self.detected = True
            log_concept_swap_detected(self.session_id)
            print("[SWAP] detected in user message")
        return self.detected

    def log_presented(self) -> None:
        """Call this when swap is first presented in an answer."""
        log_concept_swap_presented(self.session_id)

    def check_history(self, history_text: str) -> None:
        """Final scan of full history at session end."""
        if self.detected or not self._detection_prompt:
            return
        try:
            prompt = HISTORY_CHECK_PROMPT_BASE.format(
                detection_prompt=self._detection_prompt
            )
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{prompt}\n\nConversation history:\n{history_text}",
            )
            parsed = json.loads(response.text.strip())
            if (
                parsed.get("detected", False)
                and parsed.get("confidence", 0.0) >= SWAP_DETECTION_THRESHOLD
            ):
                self.detected = True
                log_concept_swap_detected(self.session_id)
                print("[SWAP] detected in history scan")
        except Exception as e:
            print(f"[SWAP] history check error: {e}")

    def _classify_message(self, text: str) -> bool:
        try:
            prompt = DETECTION_PROMPT_BASE.format(
                detection_prompt=self._detection_prompt
            )
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{prompt}\n\nUser message: \"{text}\"",
            )
            parsed = json.loads(response.text.strip())
            result = (
                parsed.get("detected", False)
                and parsed.get("confidence", 0.0) >= SWAP_DETECTION_THRESHOLD
            )
            print(
                f"[SWAP] detected={parsed.get('detected')}, "
                f"confidence={parsed.get('confidence')}, triggered={result}"
            )
            return result
        except Exception as e:
            print(f"[SWAP] classifier error: {e}")
            return False