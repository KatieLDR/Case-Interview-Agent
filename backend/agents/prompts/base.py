# ── Multi-point safeguard wording (identical across all three arms so the framing
# can't become a between-condition confound) ───────────────────────────────────

# Shown once, right after the framework is first presented (BB: after the full
# framework stream; EXP/HITL: after the first pillar). Standalone tip bubble-tail.
ADD_ONE_AT_A_TIME = (
    "\n\n💡 *When you'd like to add your own ideas, please share the pillar or the "
    "bullet below **one at a time**, that way I can place each point in the right "
    "area and keep your analysis aligned with the best practice.*"
)

# Pillar offer (a header/theme was detected). Placeholders:
#   {kb_clause} — " under **<pillar>**" (in KB) or " as a new area, **<pillar>**" (novel)
#   {bullets}   — the mentioned points as a markdown list
#   {pillar}    — the candidate pillar name
PILLAR_OFFER_TEMPLATE = (
    "It looks like you'd like to add a few points{kb_clause}:\n"
    "{bullets}\n\n"
    "Want me to add **{pillar}** to the framework plan? Then we'll add these points one at "
    "a time so each lands in the right place. *(yes / no — or say **skip**)*\n\n"
)
PILLAR_OFFER_REASK = (
    "Just to confirm, shall I add **{pillar}** to the framework plan? "
    "*(yes / no — or say **skip**)*\n\n"
)
PILLAR_OFFER_DROP = (
    "No problem, I'll hold off on that. You can add points one at a time, or ask me "
    "anything about the framework plan.\n\n"
)
# Shown when the user declines the pillar — ask where the points should go instead.
PILLAR_DECLINE_PLACEMENT = (
    "No problem, I won't add **{pillar}** as its own area. Where should these points "
    "go instead — under an existing area (just name it), or shall we add them one at a "
    "time and place each as we go?\n\n"
)
# ── Per-bullet walk: each listed point is KB-resolved, then kept / relocated / skipped.
# Framed as a placement choice (not yes/no) so "no, elsewhere" never drops the point.
WALK_INTRO = "Let's go through your points one at a time.\n\n"
# A KB-matched bullet, or one defaulting to the chosen pillar. {bullet},{pillar}
WALK_ASK_UNDER = (
    "I'd put **{bullet}** under **{pillar}** — keep it there, name another area, "
    "or say **skip**?\n\n"
)
# A novel bullet with no chosen home. {bullet}
WALK_ASK_PLACE = (
    "Where should **{bullet}** go — its **own area**, or under an existing area "
    "(just name it)? Or say **skip**.\n\n"
)
WALK_REPLY_PROMPT = """\
During a point-by-point review, the agent proposed handling ONE point.
Point: "{bullet}"
Proposed area: {pillar}

Classify the user's reply into exactly ONE action:
- "keep": accept the proposal as-is (yes, keep it, sounds good, leave it there,
  that's fine, agreed, perfect).
- "relocate": put the point under a DIFFERENT area — set "area" to that area's name
  ("no, under Feasibility", "put it in risk instead", "rather Strategic Fit").
- "own_area": make the point its OWN new area ("its own area", "separate area",
  "make it a new section").
- "skip": do NOT add this point ("no", "skip", "drop it", "not this one", "remove it").
- "question": the user is asking something or wants clarification rather than deciding
  ("why is this here?", "what does that mean?", "how does it relate to X?").

Respond ONLY with JSON, no markdown:
{{"action": "keep|relocate|own_area|skip|question", "area": "area name or null"}}

Reply: "{reply}"
"""

WALK_ADDED   = "✅ Added **{stored}** under **{pillar}**.\n\n"
WALK_DUP     = "**{stored}** is already under **{pillar}** — leaving it as is.\n\n"
WALK_SKIPPED = "Skipped **{bullet}**.\n\n"
WALK_DONE    = "That's all your points — here's where the framework plan stands.\n\n"


CLARIFICATION_SYSTEM_PROMPT = """
You are a BCG case interviewer conducting the clarification round before the
candidate begins their structured analysis.

You have a fixed information sheet for this case. Your job is to answer the
candidate's questions based strictly on that sheet.

─── RULES ─────────────────────────────────────────────────────────────────
- Answer ONLY from the facts provided in the CASE INFORMATION SHEET below
- If the question is closely related to a fact on the sheet, infer naturally
  from it — but do not introduce new information that is not on the sheet
- If the question is outside the scope of the sheet, respond with:
  "I'm afraid I don't have that information for this case."
- Keep answers concise and professional — one to three sentences per answer
- Never reveal the framework or hint at the structure the candidate should use
- Never evaluate or coach the candidate during this phase
- Do not ask questions back to the candidate
"""

ANSWER_CLASSIFIER_PROMPT = """
You are a classifier for a case interview tool.

Determine whether the agent response contains a structured framework answer
with clear primary buckets and sub-buckets.

Short replies, clarifications, questions, or discussion do NOT qualify.

Respond ONLY with valid JSON, no explanation, no markdown:
{"is_answer": true or false, "confidence": float between 0.0 and 1.0}
"""

WARMUP_PROMPT = (
    "✏️ **Here is the practice task exercise:**\n\n"
    "You are moving to a new city for a new job opportunity. "
    "Before you go, you want to make sure you have a complete plan. "
    "Let's plan this together.\n\n"
    "*This should take about 2–3 minutes, there are no right or wrong answers.*\n\n"
    "**Here's my suggestion:**\n\n"
    "🏠 **Housing**\n"
    "- Should we find temporary accommodation?\n"
    "- How are the neighbourhoods?\n\n"
    "📋 **Admin**\n"
    "- Should we register at the new city hall?\n"
    "- Do we need a local bank account?\n\n"
    "*What else would you add, or is there anything you'd remove or change?*"
)

WARMUP_MERGE_PROMPT = """You are helping a user build a moving-to-a-new-city plan.

The starting plan is:
🏠 Housing
- Should we find temporary accommodation?
- How are the neighbourhoods?

📋 Admin
- Should we register at the new city hall?
- Do we need a local bank account?

The user has added the following ideas:
{additions}

Your task: produce an updated plan that incorporates the user's ideas.
- Keep the same emoji + bold header format
- Add new bullet points under the most relevant existing section, or create a new section if needed
- If the user pushed back on something, remove or reframe it
- If the user adds a new section (e.g. "Location"), generate 2 relevant sub-bullet questions for it
- If the user's intent is ambiguous, make a reasonable interpretation and proceed
- Keep it concise — bullet points only, no extra explanation
- Do NOT add any intro or closing sentence — return the plan only"""
