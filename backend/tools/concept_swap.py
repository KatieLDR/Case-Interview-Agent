import logging
import os
import json
from google import genai
from dotenv import load_dotenv
import re

load_dotenv()

# ── LLM (centralised in backend.llm) ───────────────────────────────────────
from backend.llm import client, CLASSIFIER_MODEL, classify_json, DETECTION_B_THRESHOLD, DETECTION_C_THRESHOLD

# ══════════════════════════════════════════════════════════════════════════
# Per-agent swap config
#
# Fields:
#   wrong_concept  : the concept being injected (from a different framework)
#   wrong_framework: the framework the wrong concept actually belongs to
#                    Used in _stream_swap_caught() context block.
#
# Removed fields (change log: 2026-03-31):
#   case_type  — was never read by any method; removed to avoid misleading
#   framework  — was never read by any method; the active framework is now
#                read dynamically from self.kg_context["framework"] in the
#                agent. Hardcoding it here broke when users switched frameworks.
#
# NOTE: wrong_concept does NOT need to exist in the KG.
#       Injection is purely via system prompt string.
#       All wrong concepts are currently outside the KG — handle separately
#       in the Framework Alignment Accuracy scoring script.
# ══════════════════════════════════════════════════════════════════════════
SWAP_CONFIG = {
    "black_box": {
        "wrong_concept":   "Average number of steps walked per day by the IT team",
        "wrong_framework": "Employee Wellness Analytics",
        "match_terms":     ["steps walked", "step walked", "step-walked",
                            "steps per day", "steps a day", "steps each day", "daily steps",
                            "number of steps", "number of step", "how many steps",
                            "amount of steps", "step count", "steps metric", "step metric",
                            "steps data", "walking metric", "team walks", "walk per day",
                            "how much they walk", "physical activity", "physical movement",
                            "physical engagement", "wellness metric", "wellness analytics",
                            "fitness metric", "fitness tracker", "pedometer"],
        "match_stems":     ["step", "walk"],
    },
    "explainable": {
        "wrong_concept":   "Average number of steps walked per day by the IT team",
        "wrong_framework": "Employee Wellness Analytics",
        "match_terms":     ["steps walked", "step walked", "step-walked",
                            "steps per day", "steps a day", "steps each day", "daily steps",
                            "number of steps", "number of step", "how many steps",
                            "amount of steps", "step count", "steps metric", "step metric",
                            "steps data", "walking metric", "team walks", "walk per day",
                            "how much they walk", "physical activity", "physical movement",
                            "physical engagement", "wellness metric", "wellness analytics",
                            "fitness metric", "fitness tracker", "pedometer"],
        "match_stems":     ["step", "walk"],
    },
    "hitl": {
        "wrong_concept":   "Average number of steps walked per day by the IT team",
        "wrong_framework": "Employee Wellness Analytics",
        "match_terms":     ["steps walked", "step walked", "step-walked",
                            "steps per day", "steps a day", "steps each day", "daily steps",
                            "number of steps", "number of step", "how many steps",
                            "amount of steps", "step count", "steps metric", "step metric",
                            "steps data", "walking metric", "team walks", "walk per day",
                            "how much they walk", "physical activity", "physical movement",
                            "physical engagement", "wellness metric", "wellness analytics",
                            "fitness metric", "fitness tracker", "pedometer"],
        "match_stems":     ["step", "walk"],
    },
}

# ══════════════════════════════════════════════════════════════════════════
# Classifier prompts
# ══════════════════════════════════════════════════════════════════════════

# Change log: 2026-06-01 — narrowed detection to REJECTION only; a bare "why is
# X here?" is now a question (handled by the swap_questioned path), not detection.
# Removes the format confound where naming the concept (BlackBox) detected but a
# pronoun (walkthrough) did not. PENDING ALEX SIGN-OFF — changes the DV definition
# uniformly across all three agents.
_DETECTION_B_PROMPT = """
You are a classifier for a case interview experiment.

The agent injected a wrong concept "{wrong_concept}" into its response.

Determine whether the user's message REJECTS or FLAGS "{wrong_concept}" as not
belonging — i.e. as incorrect, out of place, irrelevant, or belonging to a
different type of analysis.

Detection requires a REJECTION signal, NOT mere curiosity. The user must
indicate the concept should not be there.

detected = true:
- Naming "{wrong_concept}" and saying it does not belong / is irrelevant / is out of place
- Stating "{wrong_concept}" belongs to a different framework or type of analysis
- Asking to remove, exclude, or drop "{wrong_concept}"
- A "why is this here?" COMBINED with a reason it does not belong
  (e.g. "why is X here? that's a wellness metric")

detected = false:
- A bare question asking why "{wrong_concept}" is included, what it means, or how
  it applies — WITHOUT asserting it does not belong. This is a request for
  justification, not a rejection.
- Expressing confusion or uncertainty about "{wrong_concept}" without rejecting it
- General confusion about the case
- Asking for clarification on unrelated parts of the framework
- Requesting a different framework style
- Flagging a different concept that was not injected

If in doubt between a question and a rejection, choose detected = false.

Respond ONLY with a valid JSON object, no explanation, no markdown:
{{"detected": true or false, "confidence": float between 0.0 and 1.0, "flagged_concept": string or null}}

Examples:
- "Why is debt-to-equity ratio here? That's a finance concept" → {{"detected": true, "confidence": 0.97, "flagged_concept": "Debt-to-Equity Ratio"}}
- "Why is debt-to-equity ratio here?" → {{"detected": false, "confidence": 0.92, "flagged_concept": null}}
- "Why is debt-to-equity relevant for this case?" → {{"detected": false, "confidence": 0.90, "flagged_concept": null}}
- "Can you explain why debt-to-equity is included?" → {{"detected": false, "confidence": 0.93, "flagged_concept": null}}
- "I'm not sure what debt-to-equity is doing here" → {{"detected": false, "confidence": 0.82, "flagged_concept": null}}
- "I don't think debt-to-equity belongs in a market entry framework" → {{"detected": true, "confidence": 0.96, "flagged_concept": "Debt-to-Equity Ratio"}}
- "Debt-to-equity belongs to financial analysis, not this" → {{"detected": true, "confidence": 0.95, "flagged_concept": "Debt-to-Equity Ratio"}}
- "Remove debt-to-equity" → {{"detected": true, "confidence": 0.96, "flagged_concept": "Debt-to-Equity Ratio"}}
- "Can you explain the market sizing step?" → {{"detected": false, "confidence": 0.97, "flagged_concept": null}}
- "This framework seems off" → {{"detected": false, "confidence": 0.75, "flagged_concept": null}}
"""

_DETECTION_C_PROMPT = """
You are a classifier for a case interview experiment.

The agent injected a wrong concept "{wrong_concept}" into its response.

Determine whether the user's message implicitly signals they have DECIDED to
remove "{wrong_concept}", by:
- Restructuring the framework and omitting "{wrong_concept}" entirely
- Correcting the analysis flow in a way that drops "{wrong_concept}" without
  naming it
- Proposing an alternative approach that clearly sidelines "{wrong_concept}"

Detection requires an actual decision to remove or sideline the concept, NOT
mere curiosity or a request for explanation.

detected = false:
- A bare question asking why "{wrong_concept}" is included, what it means, or how
  it applies — WITHOUT restructuring or dropping it. This is a request for
  justification, not a decision to remove. Treat as NOT detected.
- Expressing confusion or uncertainty about "{wrong_concept}" without correcting it
- General follow-up questions unrelated to the concept
- Requests for clarification on unrelated parts of the framework
- Vague dissatisfaction without a clear correction

If in doubt between a question and a decision to remove, choose detected = false.

Respond ONLY with a valid JSON object, no explanation, no markdown:
{{"detected": true or false, "confidence": float between 0.0 and 1.0}}

Examples:
- User restructures Market Entry framework with no mention of debt-to-equity → {{"detected": true, "confidence": 0.89}}
- User says "I think we should focus on market size and feasibility only" → {{"detected": true, "confidence": 0.87}}
- "Why is debt-to-equity here?" → {{"detected": false, "confidence": 0.93}}
- "Why are we considering debt-to-equity?" → {{"detected": false, "confidence": 0.93}}
- "Can you explain why debt-to-equity is included?" → {{"detected": false, "confidence": 0.94}}
- User asks "can you explain price elasticity more?" → {{"detected": false, "confidence": 0.95}}
- User asks a general follow-up → {{"detected": false, "confidence": 0.97}}
"""

# ══════════════════════════════════════════════════════════════════════════
# Detection thresholds
# ══════════════════════════════════════════════════════════════════════════
# DETECTION_B_THRESHOLD / DETECTION_C_THRESHOLD now imported from backend.llm


# ══════════════════════════════════════════════════════════════════════════
# ConceptSwap class
# ══════════════════════════════════════════════════════════════════════════
class ConceptSwap:
    """
    Handles concept swap injection and detection for a single agent session.

    Injection is done via system prompt (get_system_prompt_block), not by
    modifying response text. This ensures the wrong concept appears naturally
    in the framework at the correct level and format.

    Usage in an agent:
        self.swap = ConceptSwap(agent_type="black_box", session_id=self.session_id)

        # In _build_system_prompt() — inject or exclude based on state
        swap_block = self.swap.get_system_prompt_block()

        # After generating first framework response
        self.swap.maybe_inject(reply)  # marks injected=True, call log_presented() after

        # After receiving user message — check for detection
        detected = self.swap.check_detection(user_message)
    """

    def __init__(self, agent_type: str, session_id: str):
        if agent_type not in SWAP_CONFIG:
            raise ValueError(f"No swap config for agent_type='{agent_type}'")

        self.agent_type = agent_type
        self.session_id = session_id
        self.config     = SWAP_CONFIG[agent_type]
        self.injected   = False  # True after first framework response shown
        self.detected   = False  # True after user catches the swap

    # ── Public interface ───────────────────────────────────────────────────

    def get_system_prompt_block(self) -> str:
        """
        Returns the system prompt fragment for this swap, toggled by state.

        Before detection:
            Tells the model to include the wrong concept as a framework bucket.
            The model places it naturally at the correct level and format.

        After detection:
            Removes the injection instruction and adds an explicit exclusion —
            model stops including the wrong concept entirely.

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

    def maybe_inject(self, response_text: str) -> str:
        """
        Injection is handled via system prompt (get_system_prompt_block).
        This method only marks self.injected = True on the first framework
        response so the agent knows when to call log_presented().
        Returns response_text unchanged.
        """
        if not self.detected and not self.injected:
            self.injected = True
        return response_text

    def check_detection(self, user_message: str) -> bool:
        """
        Run Direction B and C detection on the user's message.
        Direction B also verifies the flagged concept matches wrong_concept
        to avoid false positives when users remove legitimate concepts.

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

    def log_presented(self) -> None:
        """Call after first framework response to stamp Firestore. Step 5: routed
        through the shared §3.6 layer (swap_presented) so the swap vocabulary is
        unified; the injection/detection MECHANISM below is untouched (§0 #4)."""
        try:
            from backend.logging import events as ev
            from backend.logging.sink import firestore_sink as sink
            ev.swap_presented(
                ev.EventContext(self.session_id, agent_type=self.agent_type,
                                wrong_concept=self.config["wrong_concept"]),
                sink,
            )
            print(f"[SWAP PRESENTED] session={self.session_id}")
        except Exception as e:
            print(f"[SWAP LOG] failed to log presentation: {e}")
            
    def force_detected(self) -> None:
        """
        Force swap detection without running the LLM classifier.
        Used when detection is triggered by a button click (HITL Reject
        confirmation) rather than natural language input.

        Python owns the state decision — no LLM call needed here.
        Consistent with principle: Python owns state, LLM owns semantics.

        Change log: 2026-04-09 — added for HITLAgent button-triggered detection.
        """
        if not self.detected:
            self.detected = True
            self._log_detected()
            print(
                f"[SWAP] force_detected() — non-text-triggered, "
                f"session={self.session_id}, agent={self.agent_type}"
            )

    @property
    def is_detected(self) -> bool:
        return self.detected

    @property
    def is_injected(self) -> bool:
        return self.injected

    def matches(self, text: str) -> bool:
        """
        Does `text` refer to the swap concept? Deterministic routing decision — no LLM.
        Three layers: canonical-name substring, literal match_terms, then a stem
        signature (ALL required stems present in some token, so "step walk" / "steps
        walked" / "walking" all resolve). Single home for "is this the swap?" — used by
        removal (BlackBox _begin_removal, HITL §3b) and _direction_b's concept_match.
        Change log: 2026-06-01
        """
        if not text:
            return False
        norm  = self._normalize(text)
        wrong = self._normalize(self.config["wrong_concept"])
        if len(norm) >= 5 and (norm in wrong or wrong in norm):
            return True
        if any(self._normalize(t) in norm for t in self.config.get("match_terms", [])):
            return True
        stems = self.config.get("match_stems", [])
        if stems:
            tokens = norm.split()
            return all(any(stem in tok for tok in tokens) for stem in stems)
        return False

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())   # strip hyphens/punctuation
        return re.sub(r"\s+", " ", text).strip()
    # ── Detection classifiers ──────────────────────────────────────────────

    def _direction_b(self, user_message: str) -> bool:
        """
        Direction B: user explicitly names or flags the wrong concept.
        Includes concept_match guard — only fires if the flagged concept
        actually matches wrong_concept, preventing false positives when
        users remove legitimate framework concepts.
        """
        wrong  = self.config["wrong_concept"]
        prompt = _DETECTION_B_PROMPT.format(wrong_concept=wrong)
        try:
            parsed = classify_json(f"{prompt}\n\nUser message: \"{user_message}\"")

            detected   = parsed.get("detected", False)
            confidence = parsed.get("confidence", 0.0)
            flagged    = parsed.get("flagged_concept") or ""

            # Verify flagged concept refers to the swap before firing — single predicate.
            concept_match = self.matches(flagged)

            result = (
                detected and
                confidence >= DETECTION_B_THRESHOLD and
                concept_match
            )
            print(
                f"[SWAP B] detected={detected}, confidence={confidence}, "
                f"flagged='{flagged}', concept_match={concept_match}, fired={result}"
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
            parsed = classify_json(f"{prompt}\n\nUser message: \"{user_message}\"")
            result = (
                parsed.get("detected", False) and
                parsed.get("confidence", 0.0) >= DETECTION_C_THRESHOLD
            )
            print(
                f"[SWAP C] detected={parsed.get('detected')}, "
                f"confidence={parsed.get('confidence')}, fired={result}"
            )
            return result
        except Exception as e:
            print(f"[SWAP C] classifier error: {e}")
            return False

    # ── Firestore logging ──────────────────────────────────────────────────

    def _log_detected(self) -> None:
        """Log swap detection event. Step 5: routed through the shared §3.6 layer
        (swap_detected). The Direction B/C detection MECHANISM that CALLS this is
        untouched (§0 #4) — only the logging vocabulary is unified."""
        try:
            from backend.logging import events as ev
            from backend.logging.sink import firestore_sink as sink
            ev.swap_detected(
                ev.EventContext(self.session_id, agent_type=self.agent_type,
                                wrong_concept=self.config["wrong_concept"]),
                sink,
            )
        except Exception as e:
            print(f"[SWAP LOG] failed to log detection: {e}")

    # ── Utility ────────────────────────────────────────────────────────────

