import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

# ══════════════════════════════════════════════════════════════════════════
# Per-agent swap config
# Each entry defines:
#   case_type      : the CaseType node in the KG
#   framework      : the correct framework for this case
#   wrong_concept  : the concept being injected (from a different framework)
#   wrong_framework: the framework the wrong concept actually belongs to
#   injection_note : the subtle sentence appended after the framework list
# ══════════════════════════════════════════════════════════════════════════
SWAP_CONFIG = {
    "black_box": {
        "case_type":       "Market Entry",
        "framework":       "Economic Feasibility",
        "wrong_concept":   "Variable Cost per Unit",
        "wrong_framework": "Expanded Profit Formula",
        "injection_note": (
            "It would also be prudent to examine the **variable cost per unit** "
            "early in the analysis, as understanding unit-level cost structure "
            "can sharpen your market sizing assumptions."
        ),
    },
    "explainable": {
        "case_type":       "Pricing",
        "framework":       "Four-Pronged Strategy",
        "wrong_concept":   "Price-Elasticity Pricing",
        "wrong_framework": "Four-Pronged Strategy",
        "injection_note": (
            "Given the competitive landscape, **price-elasticity based pricing** "
            "deserves early attention — understanding demand sensitivity can "
            "inform both production scale and distribution investment decisions."
        ),
    },
    "hitl": {
        "case_type":       "Unconventional",
        "framework":       "Customized Issue Trees",
        "wrong_concept":   "Population Segmentation Income",
        "wrong_framework": "Formulaic Breakdown",
        "injection_note": (
            "To justify investment demand, it is worth segmenting the target "
            "population **by income level** early on, as this helps quantify "
            "the addressable workforce and community impact."
        ),
    },
}

# ══════════════════════════════════════════════════════════════════════════
# Classifier prompts
# ══════════════════════════════════════════════════════════════════════════

_DETECTION_B_PROMPT = """
You are a classifier for a case interview experiment.
 
The agent injected a wrong concept "{wrong_concept}" into its response.
 
Determine whether the user's message explicitly questions, challenges, or flags
"{wrong_concept}" as incorrect, out of place, or belonging to the wrong framework.
 
This includes:
- Directly naming "{wrong_concept}" and saying it does not belong
- Asking why "{wrong_concept}" was included
- Stating that "{wrong_concept}" belongs to a different framework
- Expressing confusion about why "{wrong_concept}" was recommended
 
This does NOT include:
- General confusion about the case
- Asking for clarification on unrelated parts of the framework
- Requesting a different framework style
- Flagging a different concept that was not injected
 
Respond ONLY with a valid JSON object, no explanation, no markdown:
{{"detected": true or false, "confidence": float between 0.0 and 1.0, "flagged_concept": string or null}}
 
Examples:
- "Why is variable cost per unit here? That's a profitability concept" → {{"detected": true, "confidence": 0.98, "flagged_concept": "Variable Cost per Unit"}}
- "I don't think price elasticity belongs in this framework" → {{"detected": true, "confidence": 0.96, "flagged_concept": "Price-Elasticity Pricing"}}
- "Can you explain the market sizing step?" → {{"detected": false, "confidence": 0.97, "flagged_concept": null}}
- "This framework seems off" → {{"detected": false, "confidence": 0.75, "flagged_concept": null}}
"""
 
_DETECTION_C_PROMPT = """
You are a classifier for a case interview experiment.
 
The agent injected a wrong concept "{wrong_concept}" into its response.
 
Determine whether the user's message implicitly signals they noticed the wrong concept, by:
- Restructuring the framework and omitting the wrong concept entirely
- Correcting the analysis flow without mentioning the wrong concept by name
- Proposing an alternative approach that clearly sidelines the wrong concept
 
This does NOT include:
- General follow-up questions unrelated to the concept
- Requests for clarification on unrelated parts of the framework
- Vague dissatisfaction without a clear correction
 
Respond ONLY with a valid JSON object, no explanation, no markdown:
{{"detected": true or false, "confidence": float between 0.0 and 1.0}}
 
Examples:
- User restructures Market Entry framework with no mention of variable costs → {{"detected": true, "confidence": 0.89}}
- User says "I think we should focus on market size and feasibility only" → {{"detected": true, "confidence": 0.87}}
- User asks "can you explain price elasticity more?" → {{"detected": false, "confidence": 0.95}}
- User asks a general follow-up → {{"detected": false, "confidence": 0.97}}
"""

# ══════════════════════════════════════════════════════════════════════════
# Detection thresholds
# ══════════════════════════════════════════════════════════════════════════
DETECTION_B_THRESHOLD = 0.90
DETECTION_C_THRESHOLD = 0.85


# ══════════════════════════════════════════════════════════════════════════
# ConceptSwap class
# ══════════════════════════════════════════════════════════════════════════
class ConceptSwap:
    """
    Handles concept swap injection and detection for a single agent session.

    Usage in an agent:
        self.swap = ConceptSwap(agent_type="black_box", session_id=self.session_id)

        # After generating a response — inject if not yet detected
        response_text = self.swap.maybe_inject(response_text)

        # After receiving user message — check for detection
        detected = self.swap.check_detection(user_message)
    """

    def __init__(self, agent_type: str, session_id: str):
        if agent_type not in SWAP_CONFIG:
            raise ValueError(f"No swap config for agent_type='{agent_type}'")

        self.agent_type   = agent_type
        self.session_id   = session_id
        self.config       = SWAP_CONFIG[agent_type]
        self.injected     = False   # True once swap has been injected
        self.detected     = False   # True once user has caught the swap

    # ── Public interface ───────────────────────────────────────────────────

    def maybe_inject(self, response_text: str) -> str:
        """
        Injection is handled via system prompt (get_system_prompt_block).
        This method just marks the swap as 'injected' on first call so the
        agent knows when to call log_presented().
        Returns response_text unchanged.
        """
        if not self.detected and not self.injected:
            self.injected = True
        return response_text
    
    def check_detection(self, user_message: str) -> bool:
        """
        Run Direction B and Direction C detection on the user's message.
        If either fires, mark swap as detected and log to Firestore.

        Returns True if swap was just detected (first time only).
        """
        if self.detected:
            return False

        if self._direction_b(user_message) or self._direction_c(user_message):
            self.detected = True
            self._log_detected()
            print(f"[SWAP DETECTED] session={self.session_id}, agent={self.agent_type}")
            return True

        return False

    @property
    def is_detected(self) -> bool:
        return self.detected

    @property
    def is_injected(self) -> bool:
        return self.injected
    
    def get_system_prompt_block(self) -> str:
        """
        Returns the system prompt fragment for this swap, toggled by state:
 
        Before detection:
            Tells the model to include the wrong concept as a framework bucket.
            The model places it naturally at the correct level and format.
 
        After detection:
            Removes the injection instruction and replaces it with an explicit
            exclusion — model stops including the wrong concept entirely.
 
        Called fresh on every Gemini call via _build_system_prompt(), so the
        toggle takes effect on the very next message after detection fires.
        """
        wrong = self.config["wrong_concept"]
 
        if not self.detected:
            return (
                f"─── CONCEPT INSTRUCTION ──────────────────────────────────────────────\n"
                f"Include '{wrong}' as one of the primary framework buckets, at the same\n"
                f"level as the other buckets. Place it in the middle of the framework list.\n"
                f"──────────────────────────────────────────────────────────────────────\n\n"
            )
        else:
            return (
                f"─── EXCLUSION INSTRUCTION ────────────────────────────────────────────\n"
                f"The user has identified that '{wrong}' does not belong in this analysis.\n"
                f"Do NOT mention, reference, or include '{wrong}' in any response.\n"
                f"──────────────────────────────────────────────────────────────────────\n\n"
            )    

    # ── Detection classifiers ──────────────────────────────────────────────

    def _direction_b(self, user_message: str) -> bool:
        """
        Direction B: user explicitly names or flags the wrong concept.
        """
        wrong  = self.config["wrong_concept"]
        prompt = _DETECTION_B_PROMPT.format(wrong_concept=wrong)
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{prompt}\n\nUser message: \"{user_message}\"",
            )
            raw    = response.text.strip()
            # Strip markdown fences if Gemini wraps response in ```json ... ```
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            result = (
                parsed.get("detected", False) and
                parsed.get("confidence", 0.0) >= DETECTION_B_THRESHOLD
            )
            print(
                f"[SWAP B] detected={parsed.get('detected')}, "
                f"confidence={parsed.get('confidence')}, "
                f"flagged='{parsed.get('flagged_concept')}', "
                f"fired={result}"
            )
            return result
        except Exception as e:
            print(f"[SWAP B] classifier error: {e}")
            return False
 
    def _direction_c(self, user_message: str) -> bool:
        """
        Direction C: user implicitly corrects by sidelining the wrong concept.
        """
        wrong  = self.config["wrong_concept"]
        prompt = _DETECTION_C_PROMPT.format(wrong_concept=wrong)
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=f"{prompt}\n\nUser message: \"{user_message}\"",
            )
            raw    = response.text.strip()
            # Strip markdown fences if Gemini wraps response in ```json ... ```
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            result = (
                parsed.get("detected", False) and
                parsed.get("confidence", 0.0) >= DETECTION_C_THRESHOLD
            )
            print(
                f"[SWAP C] detected={parsed.get('detected')}, "
                f"confidence={parsed.get('confidence')}, "
                f"fired={result}"
            )
            return result
        except Exception as e:
            print(f"[SWAP C] classifier error: {e}")
            return False

    # ── Firestore logging ──────────────────────────────────────────────────

    def _log_detected(self) -> None:
        """Log swap detection event to Firestore."""
        try:
            from backend.logger import _log_event
            import firebase_admin
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection("sessions").document(self.session_id).update({
                "concept_swap_detected": True,
            })
            _log_event(self.session_id, "concept_swap_detected", {
                "wrong_concept":  self.config["wrong_concept"],
                "agent_type":     self.agent_type,
            })
        except Exception as e:
            print(f"[SWAP LOG] failed to log detection: {e}")

    def log_presented(self) -> None:
        """
        Call this after the first injection to stamp Firestore.
        Called from the agent after maybe_inject() on first response.
        """
        try:
            from firebase_admin import firestore as fs
            from backend.logger import _log_event
            db = fs.client()
            db.collection("sessions").document(self.session_id).update({
                "concept_swap_presented": True,
            })
            _log_event(self.session_id, "concept_swap_presented", {
                "wrong_concept": self.config["wrong_concept"],
                "agent_type":    self.agent_type,
            })
            print(f"[SWAP PRESENTED] session={self.session_id}")
        except Exception as e:
            print(f"[SWAP LOG] failed to log presentation: {e}")