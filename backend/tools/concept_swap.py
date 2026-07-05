import re

from dotenv import load_dotenv

from backend.llm import (
    classify_json, DETECTION_B_THRESHOLD, DETECTION_C_THRESHOLD
)
from backend.tools.prompts.concept_swap import (
    _DETECTION_B_PROMPT, _DETECTION_C_PROMPT
)

load_dotenv()

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
# NOTE: wrong_concept is injected purely via system-prompt / walkthrough string; it does
#       NOT need to exist in the Neo4j KG (cypher/ has no swap node). The swap DOES live in
#       knowledge_base.json (the "swap": true entry) for its sub-bullets/explanation, and
#       `wrong_concept` here MUST stay identical to that entry's `name` (see the name-sync
#       assert in backend/test/swap_recognition_gate.py). Keep the external Framework
#       Alignment Accuracy scoring script's concept list updated for `genai_use_cases_submitted`.
# ══════════════════════════════════════════════════════════════════════════
SWAP_CONFIG = {
    "black_box": {
        "wrong_concept":   "Total number of GenAI use cases submitted company-wide this year",
        "wrong_framework": "Company-wide Innovation Metrics",
        "match_terms":     ["use cases submitted", "use case submitted", "cases submitted",
                            "submitted company-wide", "submitted company wide",
                            "company-wide submission", "companywide submission",
                            "submission count", "submission volume", "number of submissions",
                            "submissions this year", "total number of use cases",
                            "total use cases submitted", "innovation pipeline",
                            "pipeline volume", "pipeline metric", "portfolio volume",
                            "aggregate count", "aggregate number", "company-wide total",
                            # H1: GenAI-anchored aggregate phrasings (recall) — each keeps a
                            # count/submission anchor so no bare standalone word can match.
                            "total genai use cases", "number of genai use cases",
                            "total number of genai use cases", "genai use case count",
                            "genai use cases total", "genai submissions",
                            "use case submissions", "submissions metric", "submission rate",
                            "company-wide use case count", "companywide use case count"],
        "match_stems":     [],
    },
    "explainable": {
        "wrong_concept":   "Total number of GenAI use cases submitted company-wide this year",
        "wrong_framework": "Company-wide Innovation Metrics",
        "match_terms":     ["use cases submitted", "use case submitted", "cases submitted",
                            "submitted company-wide", "submitted company wide",
                            "company-wide submission", "companywide submission",
                            "submission count", "submission volume", "number of submissions",
                            "submissions this year", "total number of use cases",
                            "total use cases submitted", "innovation pipeline",
                            "pipeline volume", "pipeline metric", "portfolio volume",
                            "aggregate count", "aggregate number", "company-wide total",
                            # H1: GenAI-anchored aggregate phrasings (recall) — each keeps a
                            # count/submission anchor so no bare standalone word can match.
                            "total genai use cases", "number of genai use cases",
                            "total number of genai use cases", "genai use case count",
                            "genai use cases total", "genai submissions",
                            "use case submissions", "submissions metric", "submission rate",
                            "company-wide use case count", "companywide use case count"],
        "match_stems":     [],
    },
    "hitl": {
        "wrong_concept":   "Total number of GenAI use cases submitted company-wide this year",
        "wrong_framework": "Company-wide Innovation Metrics",
        "match_terms":     ["use cases submitted", "use case submitted", "cases submitted",
                            "submitted company-wide", "submitted company wide",
                            "company-wide submission", "companywide submission",
                            "submission count", "submission volume", "number of submissions",
                            "submissions this year", "total number of use cases",
                            "total use cases submitted", "innovation pipeline",
                            "pipeline volume", "pipeline metric", "portfolio volume",
                            "aggregate count", "aggregate number", "company-wide total",
                            # H1: GenAI-anchored aggregate phrasings (recall) — each keeps a
                            # count/submission anchor so no bare standalone word can match.
                            "total genai use cases", "number of genai use cases",
                            "total number of genai use cases", "genai use case count",
                            "genai use cases total", "genai submissions",
                            "use case submissions", "submissions metric", "submission rate",
                            "company-wide use case count", "companywide use case count"],
        "match_stems":     [],
    },
}


class ConceptSwap:
    def __init__(self, agent_type: str, session_id: str):
        if agent_type not in SWAP_CONFIG:
            raise ValueError(f"No swap config for agent_type='{agent_type}'")

        self.agent_type = agent_type
        self.session_id = session_id
        self.config     = SWAP_CONFIG[agent_type]
        self.injected   = False  # True after first framework response shown
        self.detected   = False  # True after user catches the swap


    # Public interface
    def get_system_prompt_block(self) -> str:
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
        if not self.detected and not self.injected:
            self.injected = True
        return response_text


    def check_detection(self, user_message: str) -> bool:
        
        if self.detected:
            return False

        if self._direction_b(user_message) or self._direction_c(user_message):
            self.detected = True
            self._log_detected()
            print(f"[SWAP DETECTED] session={self.session_id}, agent={self.agent_type}")
            return True

        return False


    def log_presented(self) -> None:
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
        if not text:
            return False
        norm  = self._normalize(text)
        wrong = self._normalize(self.config["wrong_concept"])
        if len(wrong) >= 5 and wrong in norm:
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
        text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        return re.sub(r"\s+", " ", text).strip()


    def _direction_b(self, user_message: str) -> bool:
        wrong  = self.config["wrong_concept"]
        prompt = _DETECTION_B_PROMPT.format(wrong_concept=wrong)
        try:
            parsed = classify_json(f"{prompt}\n\nUser message: \"{user_message}\"")

            detected   = parsed.get("detected", False)
            confidence = parsed.get("confidence", 0.0)
            flagged    = parsed.get("flagged_concept") or ""

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


    def _log_detected(self) -> None:
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
