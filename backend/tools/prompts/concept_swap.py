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