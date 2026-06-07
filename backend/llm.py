"""
backend/llm.py — single source of truth for the Gemini client and all
classifier/matcher plumbing.

WHY THIS FILE EXISTS (REFACTOR_PLAN §4 Step 1, F-DET):
Before this module, the genai client, model strings, score thresholds and the
JSON-fence stripper were each redefined in 3-4 places (black_box_agent.py,
concept_swap.py, rag_explainer.py, plus an inlined copy in hitl_agent.py). One
black_box duplicate-check even called a stray 3.1-flash-lite model while every
other arm used the 2.5-flash-lite classifier — a cross-condition confound, not
just untidiness. This file collapses all of that to one definition each.

DETERMINISM CONTRACT (Invariant I-7):
``classify_json`` pins ``temperature=0`` on every classifier/matcher call so the
same input takes the same path. This is the determinism anchor every later §4
gate leans on; it is also the one Step-1 change that *intentionally* shifts raw
LLM output versus the pre-pin baseline (gate against verified labels, not raw
text — see REFACTOR_PLAN §5).

SCOPE: classifier/matcher calls only. Persona *generation/rendering* calls
(streamed MAIN_MODEL output) are deliberately NOT funnelled here — their
temperature is left untouched in Step 1 so prose style does not move.
"""

import json
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Gemini client (THE only client construction in the codebase) ────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)

# ── Models (defined once; every arm uses the same constants) ────────────────
# NOTE: a stray 3.1-flash-lite model used to power black_box/HITL fuzzy
# duplicate checks (F-DET). It is gone — every classifier/matcher call now
# resolves to CLASSIFIER_MODEL so all three arms classify identically.
MAIN_MODEL       = "gemini-2.5-flash"        # persona generation / streamed prose
CLASSIFIER_MODEL = "gemini-2.5-flash-lite"   # all intent / matching / duplicate / swap classifiers

# ── Score thresholds (consolidated from the three modules) ──────────────────
# black_box_agent.py
ANSWER_THRESHOLD        = 0.90
OVERRIDE_THRESHOLD      = 0.85
ADD_MATCH_THRESHOLD     = 0.75
CONCEPT_MATCH_THRESHOLD = 0.85
# concept_swap.py
DETECTION_B_THRESHOLD   = 0.90
DETECTION_C_THRESHOLD   = 0.85
# rag_explainer.py
FAITHFULNESS_THRESHOLD  = 0.85
PILLAR_MATCH_THRESHOLD  = 0.80

# Deterministic config reused by every classifier/matcher call.
_CLASSIFIER_CONFIG = types.GenerateContentConfig(temperature=0.0)


def _strip_fences(text: str) -> str:
    """Strip a leading ```/```json code fence from a Gemini JSON reply.

    Byte-for-byte identical to the four copies it replaces — no behaviour
    change, just one definition.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


# Public alias (the underscore name is kept for drop-in compatibility with the
# existing ``self._strip_fences`` / ``_strip_fences`` call sites).
strip_fences = _strip_fences


def classify_json(prompt: str, *, model: str = CLASSIFIER_MODEL) -> dict:
    """Run one deterministic classifier/matcher call and return parsed JSON.

    The single funnel for every "ask Gemini, get JSON back" call. Pins
    ``temperature=0`` (I-7) and goes through the one shared client + model.

    ERROR POLICY (deliberate): on a transport error (e.g. a 503) or a JSON
    parse failure this RAISES rather than returning a silent default. Each
    caller keeps its own ``try/except`` and applies its own fallback, because
    "what to do when the classifier fails" is persona/handler policy, not
    plumbing. This also keeps the error out of the question/intent counts
    (F-I3: a 503 must not be logged as a genuine ``question``).
    """
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=_CLASSIFIER_CONFIG,
    )
    return json.loads(_strip_fences(response.text))
