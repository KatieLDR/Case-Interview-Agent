import json
import logging
import re
from google.genai import types
from backend.black_box_agent import (
    BlackBoxAgent, CLASSIFIER_MODEL, MAIN_MODEL, client, classify_json,
    ANSWER_THRESHOLD, OVERRIDE_THRESHOLD,
)
from backend.cases import get_case, get_clarification_facts
from backend.concept_swap import ConceptSwap
from backend.logger import (
    create_session, log_user_message, log_agent_response,
    log_interruption, log_memory_override, update_answer,
    log_concept_added,
    log_question, log_add_pillar, log_add_sub_bullet, log_delete, log_swap_questioned
)
from backend.rag_explainer import build_citation_header, check_and_append_warning

# ── Source display: letter scheme kept, entries shown as named links ────────
# Change log: 2026-05-30
SOURCE_NAMES = {
    # GDPR
    "https://gdpr-info.eu/art-4-gdpr/": "GDPR Article 4",
    "https://gdpr-info.eu/art-5-gdpr/": "GDPR Article 5",
    "https://gdpr-info.eu/art-9-gdpr/": "GDPR Article 9",
    "https://gdpr-info.eu/art-25-gdpr/": "GDPR Article 25",
    "https://gdpr-info.eu/art-28-gdpr/": "GDPR Article 28",
    "https://gdpr-info.eu/art-35-gdpr/": "GDPR Article 35",
    "https://gdpr-info.eu/art-44-gdpr/": "GDPR Article 44",
    # EU AI Act
    "https://artificialintelligenceact.eu/article/3/": "EU AI Act Article 3",
    "https://artificialintelligenceact.eu/article/4/": "EU AI Act Article 4",
    "https://artificialintelligenceact.eu/article/6/": "EU AI Act Article 6",
    "https://artificialintelligenceact.eu/article/10/": "EU AI Act Article 10",
    "https://artificialintelligenceact.eu/article/11/": "EU AI Act Article 11",
    "https://artificialintelligenceact.eu/article/14/": "EU AI Act Article 14",
    "https://artificialintelligenceact.eu/article/50/": "EU AI Act Article 50",
    "https://artificialintelligenceact.eu/": "EU AI Act",
    # NIST
    "https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10": "NIST AI RMF 1.0",
    "https://airc.nist.gov/airmf-resources/airmf/5-sec-core/": "NIST AI RMF MANAGE 2.4",
    "https://airc.nist.gov/airmf-resources/playbook/govern/": "NIST AI RMF GOVERN",
    "https://airc.nist.gov/airmf-resources/playbook/manage/": "NIST AI RMF MANAGE",
    # McKinsey
    "https://www.mckinsey.com/capabilities/quantumblack/our-insights/one-year-of-agentic-ai-six-lessons-from-the-people-doing-the-work": "McKinsey: One Year of Agentic AI",
    "https://www.mckinsey.com/capabilities/quantumblack/our-insights/from-promise-to-impact-how-companies-can-measure-and-realize-the-full-value-of-ai": "McKinsey QuantumBlack: From Promise to Impact",
    "https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/overcoming-two-issues-that-are-sinking-gen-ai-programs": "McKinsey: Overcoming Two Issues Sinking Gen AI",
    "https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/moving-past-gen-ais-honeymoon-phase-seven-hard-truths-for-cios-to-get-from-pilot-to-scale": "McKinsey: Seven Hard Truths for CIOs",
    "https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/mlops-so-ai-can-scale": "McKinsey: MLOps So AI Can Scale",
    "https://www.mckinsey.com/capabilities/mckinsey-technology/our-insights/recalibrating-technology-budgets-for-the-ai-era": "McKinsey: Recalibrating Tech Budgets",
    "https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-new-economics-of-enterprise-technology-in-an-ai-world": "McKinsey: New Economics of Enterprise Tech",
    "https://www.mckinsey.com/capabilities/risk-and-resilience/our-insights/trust-in-the-age-of-agents": "McKinsey: Trust in the Age of Agents",
    "https://www.mckinsey.com/au/our-insights/australia-and-new-zealand-perspectives/accelerating-impact-from-ai": "McKinsey: Accelerating Impact from AI",
    # BCG
    "https://www.bcg.com/publications/2026/ai-risk-management-needs-a-better-model": "BCG: AI Risk Management",
    "https://www.bcg.com/publications/2025/strategies-tackle-ai-skills-gap": "BCG: Strategies to Tackle AI Skills Gap",
    "https://www.bcg.com/publications/2025/wont-get-gen-ai-right-if-human-oversight-wrong": "BCG: Human Oversight",
    "https://media-publications.bcg.com/global-scaling-strategic-workforce-planning-bcg-allianz.pdf": "BCG Strategic Workforce Planning",
    # Gartner
    "https://www.gartner.com/en/articles/when-not-to-use-generative-ai": "Gartner: When Not to Use GenAI",
    "https://www.gartner.com/en/articles/deploying-ai": "Gartner: Build, Buy or Blend",
    "https://www.gartner.com/en/newsroom/press-releases/2026-04-07-gartner-says-artificial-intelligence-projects-in-infrastructure-and-operations-stall-ahead-of-meaningful-roi-returns": "Gartner: AI Projects in I&O Stall",
    "https://www.gartner.com/en/newsroom/press-releases/2026-05-13-gartner-predicts-by-2027-50-percent-of-enterprises-without-a-people-centric-ai-strategy-will-lose-their-top-ai-talent": "Gartner: People-Centric AI Strategy",
    # Deloitte
    "https://www.deloitte.com/us/en/about/press-room/state-of-ai-report-2026.html": "Deloitte State of AI 2026",
    "https://www.deloitte.com/content/dam/assets-shared/docs/about/2025/state-of-ai-2026-global.pdf": "Deloitte State of AI in the Enterprise 2026",
    "https://www.deloitte.com/us/en/what-we-do/capabilities/applied-artificial-intelligence/blogs/pulse-check-series-latest-ai-developments/ai-adoption-challenges-ai-trends.html": "Deloitte: AI Adoption Challenges",
    # Other standards & frameworks
    "https://www.zenml.io/llmops-database/enterprise-genai-implementation-strategies-across-industries": "ZenML Panel",
    "https://www.hackingthecaseinterview.com/pages/ai-implementation-case-interview": "Hacking the Case Interview",
    "https://www.allianz.com/en/mediacenter/news/articles/260318-responsible-use-of-ai-at-allianz.html": "Allianz Responsible AI",
    "https://www.allianz.com/en/about-us/strategy-values/responsible-use-of-artificial-intelligence.html": "Allianz Responsible AI Principles",
    "https://www.isms.online/iso-27002/control-5-12-classification-of-information/": "ISO/IEC 27002 Control 5.12",
    "https://iso25000.com/index.php/en/iso-25000-standards/iso-25010": "ISO/IEC 25010 SQuaRE",
    "https://kpmg.com/ch/en/insights/artificial-intelligence/iso-iec-42001.html": "ISO/IEC 42001:2023",
    "https://www.digital-operational-resilience-act.com/Article_28.html": "DORA Article 28",
    "https://www.wolterskluwer.com/en/news/indicator-survey-finds-lower-concern-levels-following-significant-drop-in-regulatory-penalties": "Wolters Kluwer Regulatory Indicator 2025",
    "https://www.ibm.com/think/insights/building-evaluating-ai-agents-real-world": "IBM: Building AI Agents",
    "https://www.slideshare.net/slideshow/state-of-ai-in-business-2025-mit-nanda/282804851": "MIT NANDA: State of AI in Business 2025",
    "https://arxiv.org/pdf/2004.05785": "Lu et al.: Learning under Concept Drift",
    "https://arxiv.org/abs/2202.01523": "Jabrayilzade et al.: Bus Factor In Practice",
    "https://doi.org/10.1145/3449287": "Buçinca et al.: Overreliance on AI",
    "https://genai.owasp.org/llmrisk/llm01-prompt-injection/": "OWASP LLM01: Prompt Injection",
    "https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/": "OWASP LLM02: Sensitive Information Disclosure",
    "https://genai.owasp.org/llmrisk/llm092025-misinformation/": "OWASP LLM09: Misinformation",
}

_SRC_ENTRY_RE  = re.compile(r"\[([a-z])\]\(([^)]+)\)")   # [a](url) entries
_INLINE_REF_RE = re.compile(r"\[([a-z])\]")               # [a] inline refs in bullets

def _parse_source_line(line: str) -> list[tuple[str, str]]:
    """'Sources: [a](url) · [b](url)' → [('a', url), ('b', url)]."""
    return _SRC_ENTRY_RE.findall(line or "")

def _format_named_sources(entries: list[tuple[str, str]]) -> str:
    """Ordered (letter, url) → 'Sources: a [Name](url) · b [Name](url)'."""
    if not entries:
        return ""
    parts = [f"{letter} [{SOURCE_NAMES.get(url, url)}]({url})" for letter, url in entries]
    return "Sources: " + " · ".join(parts)

# ── Case config ────────────────────────────────────────────────────────────
CASE_TYPE = "AI Implementation"

# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

SINGLE_CONCEPT_PROMPT = """
You are a strategic consultant walking a user through a framework ONE concept at a time.

The concept name, explanation, and sources have already been presented to the user above.
YOUR ONLY JOB: Output the sub-bullets and closing question — nothing else.

EXACT OUTPUT FORMAT — copy this structure precisely:

- [analytical question, 5-7 words, specific to this case]
- [analytical question, 5-7 words, specific to this case]

*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*

─── EXAMPLE (for Strategic Fit) ────────────────────────────────────────────
- Is the workflow problem frequent enough to justify governance overhead?
- Would a simpler rules-based solution achieve the same result?

*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*
─────────────────────────────────────────────────────────────────────────────

─── STRICT RULE ─────────────────────────────────────────────────────────────
Answer ONLY about the concept named in CURRENT CONCEPT below.
If the user mentions a different concept by name:
- Check the FRAMEWORK CONTEXT below for already-planned concepts.
  Only say a concept is already planned if you are highly confident it
  matches one from the list — exact name or a clear, unambiguous synonym.
  If uncertain, do not mention it — just answer about the CURRENT CONCEPT.
- If it is clearly not in the list, briefly acknowledge it in one clause
  ("we can look at that next") — then answer about the CURRENT CONCEPT only

You do NOT control the framework. NEVER claim to add, remove, keep, include, or
leave anything out (no "I'll leave that out", no "I've added that"). NEVER
present, name, or describe another concept, and NEVER output a concept heading
or its sub-bullets. Answer only about the CURRENT CONCEPT — the system makes all
changes separately.
─────────────────────────────────────────────────────────────────────────────
"""

CONCEPT_QA_PROMPT = """
You are a strategic consultant who just introduced one concept from a framework.
The user has a question about it.

Answer in 2–3 sentences. Stay grounded in the current case context.
Plain language only — no jargon, no technical terms.

─── STRICT RULE ─────────────────────────────────────────────────────────────
Answer ONLY about the concept named in CURRENT CONCEPT below.
If the user mentions a different concept by name:
- Check the FRAMEWORK CONTEXT below for already-planned concepts.
  Only say a concept is already planned if you are highly confident it
  matches one from the list — exact name or a clear, unambiguous synonym.
  If uncertain, do not mention it — just answer about the CURRENT CONCEPT.
- If it is clearly not in the list, briefly acknowledge it in one clause
  ("we can look at that next") — then answer about the CURRENT CONCEPT only

If the user wants to remove a specific sub-point or sub-bullet:
- Acknowledge it clearly in one sentence: "Understood — I'll leave that out
  of the final summary."
- Do NOT remove the parent concept
- Do NOT ask for confirmation — honour it immediately
- Then end with the normal closing question
─────────────────────────────────────────────────────────────────────────────

─── CONDITIONAL FORMAT ───────────────────────────────────────────────────────
ON SWAP CONCEPT: {on_swap}

IF on_swap is False AND the user is asking WHY this concept matters, WHY it
is relevant, or WHY it belongs here — structure your answer using this format:
  "If we don't consider [concept], then [specific consequence for this case].
   Since [one sentence grounding it in the case], this concept is essential."

IF on_swap is True OR the user is asking anything other than why — answer
naturally in 2–3 sentences without the if-then format.
─────────────────────────────────────────────────────────────────────────────

After answering, use the closing specified in CLOSING INSTRUCTION below.
"""

SUMMARY_PROMPT = """
You are a strategic consultant presenting a final framework summary.

FORMAT:
**Full Framework Summary**

**[Bucket 1]**
- sub-bullet (5–7 words)
- sub-bullet (5–7 words)

**[Bucket 2]**
- sub-bullet (5–7 words)
- sub-bullet (5–7 words)

(continue for all buckets)

─── RULES ─────────────────────────────────────────────────────────────────
- Include ONLY concepts listed in CONCEPTS TO INCLUDE below
- No rationale sentences — summary only
- Do NOT add a follow-up question — end after the last concept

─── SUB-BULLET EXCLUSIONS ────────────────────────────────────────────────
Review the full conversation history. If the user explicitly asked to remove
a specific sub-point or sub-bullet during the walkthrough, exclude it from
the sub-bullets you generate for that concept. Do not mention the exclusion
— simply omit it.
─────────────────────────────────────────────────────────────────────────────
"""

SWAP_CAUGHT_PROMPT = """
The user has flagged that the concept you introduced does not belong here.

Respond by:
1. Acknowledging their catch warmly (one sentence)
2. Explaining in plain language why that concept belongs to a different type
   of analysis — no jargon, no technical terms
3. One short closing sentence confirming you are moving on — then STOP

STRICT RULES:
- Do NOT end with a question
- Do NOT say "shall we move on"
- Do NOT introduce, name, or preview the next concept
- Do NOT present any new framework bucket in this response
- Your response ends after the closing sentence — next concept follows separately
"""

ADVANCE_CLASSIFIER_PROMPT = """
You are a classifier for a case interview walkthrough tool.

The agent just introduced one concept and asked "Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important." Determine whether the user is ready to advance OR still has
a question or concern.

Ready to advance:
- Explicit yes ("yes", "sure", "ok", "next", "move on", "let's go")
- Implicit acceptance ("got it", "makes sense", "clear", "understood")
- Short affirmations with no follow-up question

NOT ready to advance:
- Any question about the current concept
- Expressing confusion or disagreement
- Asking to remove or change the current concept

Respond ONLY with valid JSON, no explanation, no markdown:
{"advance": true or false, "confidence": float between 0.0 and 1.0}
"""

ADVANCE_THRESHOLD = 0.75

CLARIFY_DOUBT_PROMPT = """
You classify a user's message during a framework walkthrough.

The agent presented a concept and asked whether to move on. Decide whether the
user is expressing DOUBT about whether this concept belongs — WITHOUT a clear
remove command and WITHOUT a clear information question.

doubt = true:
- "I don't think it makes sense", "this doesn't seem relevant", "not sure this belongs",
  "is this really necessary?", "this seems off", "I'm not convinced", "this feels wrong"
doubt = false:
- Clear removal command ("remove this", "skip this", "drop it", "take it out")
- Genuine information question ("what does this mean?", "how does this apply?",
  "why is this here?", "can you explain?")
- Agreement / advance ("makes sense", "ok", "sure", "next", "move on")

Respond ONLY with valid JSON, no markdown:
{"doubt": true or false, "confidence": float}
"""

CLARIFY_DOUBT_THRESHOLD = 0.7

CLARIFY_RESOLVE_PROMPT = """
The agent asked whether the user wants to REMOVE the current concept from the
framework, or have it EXPLAINED.

User reply: "{reply}"

Classify:
- "remove"  : wants it removed / left out ("remove it", "take it out", "move on without it")
- "explain" : wants to understand why it's included ("explain", "why", "tell me more")
- "advance" : fine with it, keep and continue ("it's fine", "keep it", "ok move on")

Respond ONLY with valid JSON, no markdown:
{{"decision": "remove" | "explain" | "advance", "confidence": float}}
"""

# Removal granularity — Change log: 2026-06-04
# Decides whether a removal request targets a WHOLE pillar or ONE sub-bullet, and
# which. Sees the current pillar's points, all pillar names, and the agent's last
# message so a vague "remove it" binds to the just-discussed point.
REMOVAL_TARGET_PROMPT = """
You classify WHAT a user wants to remove during a framework walkthrough.

Decide whether they mean:
- "pillar"     : a whole top-level area/pillar
- "sub_bullet" : ONE specific point/bullet inside a pillar

─── CURRENT PILLAR ───────────────────────────────────────────────────────────
{current_pillar}
Its points:
{current_bullets}
────────────────────────────────────────────────────────────────────────────
─── ALL PILLAR NAMES ─────────────────────────────────────────────────────────
{all_pillars}
────────────────────────────────────────────────────────────────────────────
─── WHAT THE AGENT LAST SAID ─────────────────────────────────────────────────
{last_agent}
────────────────────────────────────────────────────────────────────────────
─── USER MESSAGE ─────────────────────────────────────────────────────────────
{user_msg}
────────────────────────────────────────────────────────────────────────────

Rules:
- Names/refers to a whole pillar or area → level="pillar", pillar=that name.
- Refers to one specific point ("the responsible AI part", "the second point",
  "that bullet about transparency") → level="sub_bullet", pillar=its pillar,
  bullet = the EXACT matching point text copied verbatim from the points list.
- Vague "remove it / this / that": if the agent's last message was about ONE
  specific point, treat as that sub_bullet; otherwise treat as the whole current pillar.
- "bullet" MUST be copied verbatim from the points list above, or null for a pillar.

Respond ONLY with valid JSON, no markdown:
{{"level": "pillar" | "sub_bullet", "pillar": "name or null", "bullet": "exact point text or null", "confidence": float}}
"""

# Single intent router — Change log: 2026-06-04
# Replaces the old cascade (_detect_override + _is_ready_to_advance + _is_removal_doubt)
# for the walkthrough phase. Returns ONE primary intent so an add can never be
# mis-caught as a remove/doubt/question.
INTENT_ROUTER_PROMPT = """
You route a user's message during a one-concept-at-a-time framework walkthrough.
Return the SINGLE best intent.

Intents:
- "advance"  : ready to move to the next pillar ("yes", "ok", "next", "move on",
               "makes sense", "got it", "sounds good") with no other request.
- "add"      : wants to ADD a new point or area — including proposals phrased as
               "we should consider X", "we need to think about X", "what about X",
               "how about X", "can we also look at X", "add X", "include X".
               detail = the thing to add. A leading "no" does NOT cancel an add
               ("no, add X" is still add).
- "remove"   : wants to remove a whole pillar OR one specific point
               ("remove X", "drop this", "take out the part about Y", "delete that bullet").
               detail = what to remove.
- "question" : asking to understand the current concept ("what is X?", "why is this
               here?", "how does this apply?", "can you explain?").
- "doubt"    : vaguely doubts the current concept belongs, WITHOUT a clear remove
               command ("I'm not sure this fits", "this seems off", "is this necessary?").
- "redo"     : restart the whole walkthrough from scratch.
- "none"     : none of the above / unclear.

KEY DISAMBIGUATION:
- Proposing a NEW consideration to include is "add", even when phrased as
  "consider if X ...", "we need to understand X", "we need to account for X",
  "we should capture X". ("We need to consider if the team strategy aligns" → add,
  detail "team strategy alignment".)
- "add" vs "question": proposing something to include → add; asking to understand
  what is already shown → question.
- Naming an existing pillar AS THE PLACE for a new point is still "add" — the pillar
  is just the location, the new point is the thing being added. detail = the new
  point. ("Let's revisit Strategic Fit, we need to understand the frequency of the
  opportunity" → add, detail "frequency of the opportunity".)
- Only treat naming an existing pillar as "question" when the user asks to UNDERSTAND
  its existing content (e.g. "remind me what Strategic Fit covers"), with no new point.
- "doubt" vs "remove": doubt is hesitation with no command; remove is a clear instruction.

─── CURRENT PILLAR ───────────────────────────────────────────────────────────
{current_pillar}
Its points:
{current_bullets}
────────────────────────────────────────────────────────────────────────────
─── WALKTHROUGH PILLARS ──────────────────────────────────────────────────────
{concepts}
────────────────────────────────────────────────────────────────────────────
─── WHAT THE AGENT LAST SAID ─────────────────────────────────────────────────
{last_agent}
────────────────────────────────────────────────────────────────────────────
─── USER MESSAGE ─────────────────────────────────────────────────────────────
{user_msg}
────────────────────────────────────────────────────────────────────────────

Respond ONLY with valid JSON, no markdown:
{{"intent": "advance|add|remove|question|doubt|redo|none", "detail": "string or null", "confidence": float}}
"""

# ══════════════════════════════════════════════════════════════════════════
# Addition flow prompts — Change log: 2026-05-28
# ══════════════════════════════════════════════════════════════════════════

ADD_CLASSIFY_PROMPT = """
You are a classifier for a case interview framework tool.

The user wants to add something to the framework. Determine whether they are adding:
- "sub_bullet" : a specific point/question that belongs UNDER one of the existing pillars
- "pillar"     : a whole new top-level area of analysis

Then identify the best-guess target pillar from the list below (for sub_bullet),
or null (for a new pillar).

─── EXISTING PILLARS ─────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────

─── USER WANTS TO ADD ──────────────────────────────────────────────────────
{item}
────────────────────────────────────────────────────────────────────────────

Respond ONLY with valid JSON, no explanation, no markdown:
{{"kind": "sub_bullet" or "pillar", "target": "pillar name or null", "confidence": float}}
"""

ADD_RESOLVE_PROMPT = """
You are a classifier for a case interview framework tool.

The user asked to add "{item}". The agent suggested it belongs under the pillar
**{target}** and asked: add it THERE (under {target}), or as its OWN SEPARATE AREA?

─── ALL PILLAR NAMES ─────────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────
─── USER REPLY ──────────────────────────────────────────────────────────────
{reply}
────────────────────────────────────────────────────────────────────────────

Classify the reply into ONE choice:
- "under_target" : add it under {target}
                   ("add it there", "there", "under it", "yes", "ok", "sure",
                    "the first one", "yes please", "add to {target}")
- "separate"     : add it as its own separate area / new pillar
                   ("separate area", "its own area", "as a new pillar", "the second one",
                    "no, separate", "make it separate")
- "cancel"       : changed their mind, do not add it
                   ("no", "never mind", "forget it", "cancel", "actually no", "drop it")

Rules:
- A bare "yes" / "ok" / "sure" / "yeah" → under_target (the agent's primary suggestion).
- A bare "no" with no alternative → cancel. But "no, as its own area" → separate.
- If the user names a DIFFERENT pillar than {target}, set choice="under_target" and put
  that pillar name in "target_override".

Respond ONLY with valid JSON, no explanation, no markdown:
{{"choice": "under_target" | "separate" | "cancel", "target_override": "pillar name or null"}}
"""

ADD_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.

The user added "{item}" under the pillar "{pillar}".
Check whether it matches one of the pillar's existing key questions below.

─── KEY QUESTIONS FOR {pillar} ──────────────────────────────────────────────
{key_questions}
────────────────────────────────────────────────────────────────────────────

A match means the user's addition is essentially the same point as one of the
key questions above — same topic, possibly different wording.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched": true or false, "matched_index": integer or null, "confidence": float}}
"""

# Enriched: candidate areas now carry their description so semantically-adjacent
# terms ("IT Budget" → "Financial Impact") resolve. Mirrors BlackBox. Change log: 2026-06-02
ADD_PILLAR_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.

The user wants to add a new area: "{item}".
Each area below is given as "Name: description". Check whether the user's new
area is essentially the SAME AREA of analysis as one of them — same topic or
clearly the same scope, possibly different wording.

─── AREAS ───────────────────────────────────────────────────────────────────
{pillars}
────────────────────────────────────────────────────────────────────────────
Match only at the level of the whole AREA. If the user's text is just one
specific point that would sit *inside* an area (rather than naming the area
itself), set matched=false.

Respond ONLY with valid JSON, no explanation, no markdown:
{{"matched": true or false, "matched_pillar": "pillar name or null", "confidence": float}}
"""

ADD_MATCH_THRESHOLD = 0.75

# Whole-KB concept search — searches every analytical concept (name + explanation)
# across all pillars (swap excluded). On a confident hit, the caller resolves the
# parent pillar via pillar_id. Higher bar than the pillar matcher because a false hit
# REVEALS a withheld pillar = stimulus contamination. Mirrors BlackBox. Change log: 2026-06-02
CONCEPT_MATCH_THRESHOLD = 0.85

ADD_CONCEPT_MATCH_PROMPT = """
You are a classifier for a case interview framework tool.
The user wrote: "{item}".

Below is a numbered list of analytical concepts, each as "Name: explanation".
Decide whether the user's text clearly refers to ONE of these concepts — i.e.
it is essentially that concept, or a specific instance of it, possibly worded
differently.

─── CONCEPTS ────────────────────────────────────────────────────────────────
{concepts}
────────────────────────────────────────────────────────────────────────────
Match ONLY if you are confident it is essentially that concept. If the user's
text is only loosely or topically related, set matched=false.

Respond ONLY with valid JSON, no markdown:
{{"matched": true or false, "matched_index": integer or null, "confidence": float}}
"""

SUB_BULLET_FORMAT_PROMPT = """
You reformat a user's note into the style of a sub-bullet in this case framework.

Rules:
- Keep the user's MEANING exactly — add no new content, examples, or sources.
- Output ONE concise line in the framework's voice: a short topic followed by a
  clarifying question or qualifying clause. Phrasing it as a natural question is fine.
- Drop filler ("I think", "we should also", "maybe", "consider").
- No leading dash, no inline source markers. End with a question mark only if it is a question.
- Match the style of these existing framework sub-bullets:
    "Does the IT team currently log this data as part of their productivity tracking?"
    "Data classification is data correctly classified under company data handling standards?"
    "Single-developer dependency critical point of failure if original developer leaves or moves"
    "Prototype scope and governability is the use case tightly scoped enough to be reviewable?"

User note: "{item}"

Output ONLY the reformatted sub-bullet, nothing else.
"""

class ExplainableAgent(BlackBoxAgent):
    """
    Explainable agent — stateful concept-by-concept walkthrough.

    Inherits from BlackBoxAgent:
      - Clarification phase (_stream_clarification)
      - _detect_override(), _is_answer(), _strip_fences()
      - _strip_concept_swap_from_history()
      - send_message(), end_session()
      - KG infrastructure (_fetch_kg_context, _update_kg_if_framework_mentioned)
      - show_tree(), _build_tree_overview()
      - _check_duplicate()

    Overrides:
      - __init__                  : explainable case + walkthrough state variables
      - get_opening_message       : tailored intro
      - get_pre_analysis_instruction : agent-specific instruction
      - begin_analysis            : tree/button flow — streams first concept
      - _build_system_prompt      : minimal fallback (used by inherited send_message)
      - _stream_main              : stateful walkthrough logic
      - _detect_override          : walkthrough-aware override classifier

    Change log: 2026-05-12 — begin_analysis() replaces start_main_phase()
    Change log: 2026-05-16 — concept_added double-log fix; _resolve_pending
                             log_memory_override removed; WALKTHROUGH_OVERRIDE_PROMPT
                             negative examples added
    Change log: 2026-05-17 — timer sentinel; UX note moved before first concept
    Change log: 2026-05-25 — citation block updated for JSON knowledge base;
                             CASE_TYPE updated to AI Implementation;
                             WALKTHROUGH_OVERRIDE_PROMPT examples updated for Allianz case
    Change log: 2026-06-02 — add-flow matcher broadened to parity with BlackBox:
                             enriched pillar match (descriptions), whole-KB concept
                             search (_match_concept), and the add_sub_bullet
                             granularity rule. Placement-confirmation UX retained.
    Change log: 2026-06-03 — CONCEPT-FIRST precedence (most-specific wins): the concept
                             search runs before the area match, so a leaf point
                             ('payback period') resolves to a sub-bullet, not a whole
                             pillar. Empty-summary fix: added KB pillars now render
                             their KB sub-bullets in the End-Session summary.
    Change log: 2026-06-04 — a withheld pillar surfaced via a sub-concept is now INSERTED
                             right after the current concept and walked normally (no inline
                             block dump, shown once). Acknowledge-only on add.
    Change log: 2026-06-04b — sub-bullet removal (Bug 1): _classify_removal_target decides
                             pillar-vs-bullet; sub-bullet removal gets a reasoned pushback +
                             3-way confirm, logs delete at sub-bullet level, and render +
                             summary omit excluded bullets (excluded_sub_bullets state).
    Change log: 2026-06-04c — routing consolidated into a SINGLE intent router
                             (_classify_intent): advance|add|remove|question|doubt|
                             redo|switch|none. Replaces _detect_override (walkthrough
                             phase) + _is_ready_to_advance + _is_removal_doubt, so an add
                             ("we should consider X") can't be mis-caught as remove/doubt.
                             Swap detection kept identical (2a explicit remove + 2b semantic
                             backstop). question → KB-grounded QA (_concept_grounding).
    Change log: 2026-06-04d — adding to a not-yet-reached pillar now lifts it to the
                             next slot and discusses it immediately (_move_pillar_to_next;
                             swap shifts back one, still shown). Current/passed pillars
                             still update in place. Placement question glosses a
                             not-yet-presented target pillar with one sentence and drops
                             the 'bring in / existing pillar' wording.
    Change log: 2026-06-04e — placement resolution (_resolve_addition) is now TARGET-AWARE:
                             a 3-way choice (under_target | separate | cancel) that receives
                             the suggested pillar, so 'add it there' / 'yes' / 'the first one'
                             bind to it instead of cancelling or misfiring as a new pillar
                             (which had double-logged add_pillar + add_sub_bullet and created
                             a junk pillar). Errors default to add-under-target, not cancel.
    Change log: 2026-06-04f — live walkthrough and summary now share one body renderer
                             (_render_bullets_and_sources), so a user-added bullet's source
                             refs are re-lettered and merged into the Sources line on screen
                             too — no more dangling [e][d] markers mid-walkthrough.
    """

    def __init__(self, user_id: str = "anonymous"):
        self.user_id       = user_id
        self.session_id    = create_session(user_id, agent_type="explainable")
        self.original_case = get_case("explainable")
        self._pending      = False
        self.turn_count    = 0
        self.has_main_contribution = False   # gates End Session button (app.py reads this)

        self.phase = "warmup"
        self.clarification_facts = get_clarification_facts("explainable")

        self.concept_swap = ConceptSwap(
            agent_type="explainable",
            session_id=self.session_id
        )

        self.kg_context = self._fetch_kg_context(CASE_TYPE)
        logging.info(f"[KG INIT] case_type={CASE_TYPE}, "
              f"framework={self.kg_context['framework']}, "
              f"concepts={self.kg_context['concepts']}")

        # ── Walkthrough state ──────────────────────────────────────────────
        self.walkthrough_concepts = []
        self.walkthrough_index    = 0
        self.walkthrough_active   = False
        self.walkthrough_done     = False
        self.excluded_concepts    = []
        self.swap_presented       = False
        self.swap_position        = 0

        # ── Pending confirmation states ────────────────────────────────────
        self.pending_excl = None
        self.pending_add  = None   # {"item": str, "kind": str, "target": str}
        self.pending_clarify = None
        self.pending_sub_excl = None  # {"pillar": str, "bullet": str} — Change log: 2026-06-04

        # ── User sub-points — populated by duplicate guard path ───────────
        # Change log: 2026-05-12
        self.user_sub_points = {}

        # ── Removed sub-bullets — pillar name → list of removed bullet texts.
        # Render + summary omit any bullet whose normalised text is in here.
        # Change log: 2026-06-04
        self.excluded_sub_bullets = {}

        # ── User-added pillars — explicit tracking for summary ─────────────
        # Change log: 2026-05-28 — separate from walkthrough_concepts so summary
        # can distinguish "user added" from "not yet reached"
        self.user_added_pillars = []

        self.history = [
            types.Content(
                role="user",
                parts=[types.Part(text=self.original_case)]
            ),
            types.Content(
                role="model",
                parts=[types.Part(text=(
                    "I have received the case. We are now in the clarification round. "
                    "Please feel free to ask any questions about the case before you begin."
                ))]
            ),
        ]

    # ══════════════════════════════════════════════════════════════════════
    # Opening message
    # ══════════════════════════════════════════════════════════════════════

    def get_opening_message(self) -> str:
        return (
            f"📋 **Here is your case:**\n\n"
            f"{self.original_case}\n\n"
            f"---\n"
            f"Take your time to read it. Feel free to ask any clarifying questions "
            f"before you begin. When you're ready, I'll walk you through the framework "
            f"one concept at a time — you can ask questions at each step.\n\n"
        )

    def get_pre_analysis_instruction(self) -> str:
        return (
            "📖 *After you click the button below, I'll walk you through "
            "each concept one at a time — you can ask questions or suggest "
            "changes at any step.*"
        )

    def begin_analysis(self):
        """
        Generator — called when user clicks 'Got it, show me the full analysis'.
        Replaces start_main_phase().
        Change log: 2026-05-12
        Change log: 2026-05-17 — timer sentinel as first yield (split in app.py);
                                  UX note moved before first concept
        """
        self._start_main_phase_setup()

        yield (
            "⚠️ Your goal is to build a structured plan for this case. "
            "Review each factor below, share your thoughts, and you **should not only read it** but also add or remove anything you think is missing."
        )

        yield "⏱️ Your 20-minute session has started. The timer is shown on the left."
        self.walkthrough_concepts = self._build_walkthrough_concepts()
        self.walkthrough_active   = True
        self.walkthrough_index    = 0
        self.swap_presented       = False

        yield (
            "💡 *When you're finished, click **‼️End Session** "
            "to close your session. **Note: this cannot be undone.***\n\n---\n\n"
        )

        yield from self._stream_concept(is_first=True)

    # ══════════════════════════════════════════════════════════════════════
    # Walkthrough state helpers
    # ══════════════════════════════════════════════════════════════════════

    def _build_walkthrough_concepts(self) -> list:
        base     = list(self.kg_context["concepts"])
        wrong    = self.concept_swap.config["wrong_concept"]
        position = len(base) // 2
        base.insert(position, wrong)
        self.swap_position = position
        logging.info(f"[WALKTHROUGH] built={base}, swap_position={position}, "
                     f"framework={self.kg_context['framework']}")
        return base

    def _current_concept(self) -> str | None:
        excluded_lower = [e.lower() for e in self.excluded_concepts]
        while self.walkthrough_index < len(self.walkthrough_concepts):
            concept = self.walkthrough_concepts[self.walkthrough_index]
            if concept.lower() not in excluded_lower:
                return concept
            logging.debug(f"[WALKTHROUGH] skipping excluded: {concept}")
            self.walkthrough_index += 1
        return None

    def _is_wrong_concept(self, concept: str) -> bool:
        return concept.lower() == self.concept_swap.config["wrong_concept"].lower()

    def _is_ready_to_advance(self, user_input: str) -> bool:
        try:
            parsed = classify_json(f"{ADVANCE_CLASSIFIER_PROMPT}\n\nUser reply: \"{user_input}\"")
            result = (
                parsed.get("advance", False) and
                parsed.get("confidence", 0.0) >= ADVANCE_THRESHOLD
            )
            logging.info(f"[ADVANCE] advance={parsed.get('advance')}, confidence={parsed.get('confidence'):.2f}, proceed={result}")
            return result
        except Exception as e:
            logging.warning(f"[ADVANCE] classifier error: {e}")
            return False

    # ══════════════════════════════════════════════════════════════════════
    # Addition flow helpers — Change log: 2026-05-28
    # ══════════════════════════════════════════════════════════════════════

    def _classify_addition(self, item: str) -> dict:
        """LLM: is this a sub-bullet or a new pillar? Best-guess target pillar."""
        from backend import knowledge_base as kb
        pillars = ", ".join(p["name"] for p in kb.get_shown_pillars())
        prompt = ADD_CLASSIFY_PROMPT.format(pillars=pillars, item=item)
        try:
            parsed = classify_json(prompt)
            return {
                "kind":   parsed.get("kind", "sub_bullet"),
                "target": parsed.get("target"),
            }
        except Exception as e:
            logging.warning(f"[ADD CLASSIFY] error: {e}")
            return {"kind": "sub_bullet", "target": self._current_concept()}

    def _resolve_add_target(self, item: str) -> dict:
        """Pick the BEST target to offer in the placement question. CONCEPT-FIRST
        (most-specific wins) — mirrors BlackBox. Change log: 2026-06-03
          Stage 1: whole-KB concept search → kind='sub_bullet', target=parent pillar.
                   (A specific leaf like 'payback period' resolves here, NOT as a pillar.)
          Stage 2: area-level pillar match (enriched) → kind='pillar'.
                   (A broad area name like 'IT Budget' that matches no single concept.)
          Stage 3: shown-only LLM fallback for genuinely-new items.
        The user still confirms placement — this only supplies better options."""
        concept_parent = self._match_concept(item)
        if concept_parent:
            return {"kind": "sub_bullet", "target": concept_parent}
        matched_pillar = self._match_pillar(item)
        if matched_pillar:
            return {"kind": "pillar", "target": matched_pillar}
        return self._classify_addition(item)

    def _resolve_addition(self, reply: str) -> dict:
        """Interpret the placement reply. Target-aware 3-way choice so deictic replies
        ('add it there', 'yes', 'the first one') bind to the suggested pillar instead of
        cancelling. Change log: 2026-06-04"""
        from backend import knowledge_base as kb
        pillars = ", ".join(p["name"] for p in kb.get_all_pillars())
        item    = self.pending_add["item"]   if self.pending_add else ""
        target  = self.pending_add["target"] if self.pending_add else None
        prompt  = ADD_RESOLVE_PROMPT.format(
            pillars=pillars, item=item, target=target or "(unspecified)", reply=reply,
        )
        try:
            parsed = classify_json(prompt)
            choice   = parsed.get("choice", "under_target")
            override = parsed.get("target_override")
            if choice == "cancel":
                return {"confirmed": False, "target": None, "as_new_pillar": False}
            if choice == "separate":
                return {"confirmed": True, "target": None, "as_new_pillar": True}
            # under_target (default) — honour a named different pillar if given
            return {"confirmed": True, "target": override or target, "as_new_pillar": False}
        except Exception as e:
            # The user is mid-add — default to adding under the suggested target rather
            # than silently dropping it. Change log: 2026-06-04
            logging.warning(f"[ADD RESOLVE] error: {e} — defaulting to under_target")
            return {"confirmed": True, "target": target, "as_new_pillar": False}

    def _match_key_question(self, item: str, pillar_name: str) -> dict | None:
        """LLM: does item match a key_question in this pillar? Return matched text+sources."""
        from backend import knowledge_base as kb
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == pillar_name.lower()),
            None
        )
        if pillar is None:
            return None
        key_questions = pillar.get("key_questions", [])
        if not key_questions:
            return None

        kq_block = "\n".join(f"{i}. {q}" for i, q in enumerate(key_questions))
        prompt   = ADD_MATCH_PROMPT.format(
            item=item, pillar=pillar_name, key_questions=kq_block
        )
        try:
            parsed = classify_json(prompt)
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD):
                idx = parsed.get("matched_index")
                if idx is not None and 0 <= idx < len(key_questions):
                    return {
                        "question": key_questions[idx],
                        "sources":  pillar.get("key_questions_sources", ""),
                    }
        except Exception as e:
            logging.warning(f"[ADD MATCH] error: {e}")
        return None

    def _match_pillar(self, item: str) -> str | None:
        """LLM: does the user's text name one of the framework's AREAS (any pillar,
        shown or withheld)? Candidate block carries each pillar's description so
        semantically-adjacent terms resolve. Mirrors BlackBox. Change log: 2026-06-02"""
        from backend import knowledge_base as kb
        pillars = kb.get_all_pillars()
        if not pillars:
            return None
        block = "\n".join(
            f"- {p['name']}: {(p.get('description') or '').strip()}" for p in pillars
        )
        prompt = ADD_PILLAR_MATCH_PROMPT.format(item=item, pillars=block)
        try:
            parsed = classify_json(prompt)
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD):
                return parsed.get("matched_pillar")
        except Exception as e:
            logging.warning(f"[PILLAR MATCH] error: {e}")
        return None

    def _match_concept(self, item: str) -> str | None:
        """LLM: search EVERY analytical concept (name + explanation) across all pillars
        (swap excluded). On a confident hit, return the parent pillar's name via
        pillar_id. Lets a sub-concept of a withheld pillar ("breakeven point") resolve
        to its parent ("Financial Impact"). Mirrors BlackBox. Change log: 2026-06-02"""
        from backend import knowledge_base as kb
        concepts = [c for c in kb.get_all_concepts()
                    if not c.get("swap", False) and c.get("pillar_id")]
        if not concepts:
            return None
        block = "\n".join(
            f"{i}. {c['name']}: {(c.get('explanation') or '').strip()}"
            for i, c in enumerate(concepts)
        )
        prompt = ADD_CONCEPT_MATCH_PROMPT.format(item=item, concepts=block)
        try:
            parsed = classify_json(prompt)
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= CONCEPT_MATCH_THRESHOLD):
                idx = parsed.get("matched_index")
                if idx is not None and 0 <= idx < len(concepts):
                    pillar = kb.get_pillar_by_id(concepts[idx]["pillar_id"])
                    if pillar:
                        return pillar["name"]
        except Exception as e:
            logging.warning(f"[CONCEPT MATCH] error: {e}")
        return None

    def _match_withheld_pillar(self, item: str) -> str | None:
        """LLM: does new pillar match a withheld pillar (Financial Impact / Risks)?
        Used by the explicit 'separate area' branch. Now description-enriched.
        Change log: 2026-06-02"""
        from backend import knowledge_base as kb
        shown_ids = {p["id"] for p in kb.get_shown_pillars()}
        withheld  = [p for p in kb.get_all_pillars() if p["id"] not in shown_ids]
        if not withheld:
            return None

        pillars_block = "\n".join(
            f"- {p['name']}: {(p.get('description') or '').strip()}" for p in withheld
        )
        prompt = ADD_PILLAR_MATCH_PROMPT.format(item=item, pillars=pillars_block)
        try:
            parsed = classify_json(prompt)
            if (parsed.get("matched") and
                    parsed.get("confidence", 0.0) >= ADD_MATCH_THRESHOLD):
                return parsed.get("matched_pillar")
        except Exception as e:
            logging.warning(f"[ADD PILLAR MATCH] error: {e}")
        return None

    def _reveal_withheld_pillar(self, name: str) -> bool:
        """If `name` is a withheld KB pillar not yet revealed, INSERT it into the
        walkthrough right after the current concept so it is presented next as a normal
        walk-through step (it is NOT rendered as a block on add). Returns True if it was
        newly revealed, False otherwise. No add_pillar — granularity rule.
        Change log: 2026-06-04 — insert-after-current (was append); returns bool."""
        from backend import knowledge_base as kb
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == name.lower()),
            None
        )
        if pillar is None:
            return False                             # not a KB pillar — nothing to reveal
        shown_ids = {p["id"] for p in kb.get_shown_pillars()}
        if pillar["id"] in shown_ids:
            return False                             # already shown in the walkthrough
        if pillar["name"].lower() in [c.lower() for c in self.walkthrough_concepts]:
            return False                             # already revealed earlier
        insert_pos = self.walkthrough_index + 1
        self.walkthrough_concepts.insert(insert_pos, pillar["name"])
        # Keep swap_position accurate for logging if we inserted before the swap.
        if self.swap_position is not None and insert_pos <= self.swap_position:
            self.swap_position += 1
        if pillar["name"] not in self.user_added_pillars:
            self.user_added_pillars.append(pillar["name"])
        logging.info(f"[ADD] withheld pillar '{pillar['name']}' inserted at "
                     f"index {insert_pos} (next walk-through step)")
        return True
        # deliberately NO log_add_pillar — granularity rule (logged as add_sub_bullet)

    def _move_pillar_to_next(self, target: str) -> None:
        """Lift an already-scheduled but NOT-yet-reached pillar to the next slot so it
        is discussed immediately. Keeps swap_position accurate. The swap is never
        removed — it just shifts one slot later. Change log: 2026-06-04"""
        if target not in self.walkthrough_concepts:
            return
        old_idx = self.walkthrough_concepts.index(target)
        new_idx = self.walkthrough_index + 1
        if old_idx <= new_idx:
            return                                  # already next (or earlier) — nothing to move
        self.walkthrough_concepts.pop(old_idx)
        self.walkthrough_concepts.insert(new_idx, target)
        # Elements in [new_idx, old_idx-1] shift one later; bump the swap if it's among them.
        if new_idx <= self.swap_position < old_idx:
            self.swap_position += 1
        logging.info(f"[REORDER] '{target}' → index {new_idx} (discuss now); "
                     f"swap_position={self.swap_position}")

    # ── Sub-bullet removal helpers — Change log: 2026-06-04 ─────────────────

    @staticmethod
    def _norm(text: str) -> str:
        """Normalise a bullet for comparison: strip inline [a] refs, collapse space, lower."""
        return re.sub(r"\s+", " ", _INLINE_REF_RE.sub("", text or "")).strip().lower()

    def _is_excluded_bullet(self, pillar_name: str, bullet: str) -> bool:
        removed = {self._norm(b) for b in self.excluded_sub_bullets.get(pillar_name, [])}
        return self._norm(bullet) in removed

    def _last_agent_message(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts and c.parts[0].text:
                return c.parts[0].text[:600]
        return ""

    def _classify_removal_target(self, user_input: str) -> dict:
        """Decide whether a removal targets a whole pillar or one sub-bullet, and which.
        Falls back to pillar-level (the safe, pre-existing behaviour) on any error."""
        from backend import knowledge_base as kb
        current = self._current_concept() or ""

        # Visible points of the current pillar (KB sub_bullets + user points, minus removed)
        pillar = next((p for p in kb.get_all_pillars()
                       if p["name"].lower() == current.lower()), None)
        bullets = []
        if pillar:
            bullets += list(pillar.get("sub_bullets", []))
        bullets += self.user_sub_points.get(current, [])
        bullets = [b for b in bullets if not self._is_excluded_bullet(current, b)]
        bullets_block = "\n".join(f"- {b}" for b in bullets) or "(none)"

        all_pillars = ", ".join(
            sorted({p["name"] for p in kb.get_all_pillars()}
                   | set(self.walkthrough_concepts))
        )
        prompt = REMOVAL_TARGET_PROMPT.format(
            current_pillar=current or "(none)",
            current_bullets=bullets_block,
            all_pillars=all_pillars,
            last_agent=self._last_agent_message() or "(nothing yet)",
            user_msg=user_input,
        )
        try:
            parsed = classify_json(prompt)
            level = parsed.get("level", "pillar")
            return {
                "level":  level if level in ("pillar", "sub_bullet") else "pillar",
                "pillar": parsed.get("pillar") or current,
                "bullet": parsed.get("bullet"),
            }
        except Exception as e:
            logging.warning(f"[REMOVAL TARGET] error: {e} — defaulting to pillar")
            return {"level": "pillar", "pillar": current, "bullet": None}

    def _classify_intent(self, user_input: str) -> dict:
        """Single primary-intent classifier for a walkthrough turn. Replaces the old
        cascade. Defaults to 'question' (safe — never silently advances or removes).
        Change log: 2026-06-04"""
        from backend import knowledge_base as kb
        current = self._current_concept() or "(none)"
        pillar = next((p for p in kb.get_all_pillars()
                       if p["name"].lower() == current.lower()), None)
        bullets = []
        if pillar:
            bullets += list(pillar.get("sub_bullets", []))
        bullets += self.user_sub_points.get(current, [])
        bullets = [b for b in bullets if not self._is_excluded_bullet(current, b)]
        bullets_block = "\n".join(f"- {b}" for b in bullets) or "(none)"
        concepts = ", ".join(self.walkthrough_concepts) or "(none)"
        prompt = INTENT_ROUTER_PROMPT.format(
            current_pillar=current,
            current_bullets=bullets_block,
            concepts=concepts,
            last_agent=self._last_agent_message() or "(nothing yet)",
            user_msg=user_input,
        )
        try:
            parsed = classify_json(prompt)
            intent = parsed.get("intent", "question")
            valid = {"advance", "add", "remove", "question", "doubt", "redo", "none"}
            if intent not in valid:
                intent = "question"
            return {"intent": intent, "detail": parsed.get("detail")}
        except Exception as e:
            logging.warning(f"[INTENT] error: {e} — defaulting to question")
            return {"intent": "question", "detail": None}

    def _format_sub_bullet(self, item: str) -> str:
        """Reformat an unmatched user sub-point into terse bullet style. Style only —
        no new content, no source. Falls back to raw input on error. Change log: 2026-05-30"""
        try:
            response = client.models.generate_content(
                model=CLASSIFIER_MODEL,
                contents=SUB_BULLET_FORMAT_PROMPT.format(item=item),
                config=types.GenerateContentConfig(temperature=0.0),
            )
            out = self._strip_fences(response.text).strip().strip("-• ").rstrip(".")
            if out:
                logging.info(f"[SUB-POINT FORMAT] '{item}' → '{out}'")
                return out
        except Exception as e:
            logging.warning(f"[SUB-POINT FORMAT] error: {e} — keeping raw")
        return item.strip()

    def _render_bullets_and_sources(self, concept: str) -> tuple[str, str]:
        """Shared body renderer used by BOTH the live walkthrough (_stream_concept) and
        the summary (_render_pillar_block) so their citations agree. Static sub-bullets
        keep their refs; user sub-points have their matched refs re-lettered to continue
        after the static ones (deduped by URL) and merged into one named Sources line.
        Returns (bullets_text, sources_line). Read-only. Change log: 2026-06-04"""
        from backend import knowledge_base as kb
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == concept.lower()),
            None
        )
        bullet_lines = []

        # Non-KB concept — just the user's own points, no sources.
        if pillar is None:
            for sp in self.user_sub_points.get(concept, []):
                if not self._is_excluded_bullet(concept, sp):
                    bullet_lines.append(f"- {sp}")
            return "\n".join(bullet_lines), ""

        # Static sources keep their original letters/order
        merged        = _parse_source_line(pillar.get("sub_bullets_sources", ""))
        url_to_letter = {url: letter for letter, url in merged}
        next_ord      = (max(ord(l) for l, _ in merged) + 1) if merged else ord("a")

        # Static bullets — refs untouched; omit any the user removed
        for b in pillar.get("sub_bullets", []):
            if not self._is_excluded_bullet(concept, b):
                bullet_lines.append(f"- {b}")

        # User sub-points — matched refs resolve via key_questions_sources, continue letters
        kq_map = dict(_parse_source_line(pillar.get("key_questions_sources", "")))

        def _repl(m):
            nonlocal next_ord
            url = kq_map.get(m.group(1))
            if not url:
                return ""                       # drop a ref with no resolvable source
            if url in url_to_letter:
                return f"[{url_to_letter[url]}]"  # dedup by URL
            letter = chr(next_ord); next_ord += 1
            url_to_letter[url] = letter
            merged.append((letter, url))
            return f"[{letter}]"

        for sp in self.user_sub_points.get(concept, []):
            if self._is_excluded_bullet(concept, sp):
                continue
            bullet_lines.append(f"- {_INLINE_REF_RE.sub(_repl, sp)}")

        return "\n".join(bullet_lines), _format_named_sources(merged)

    def _render_pillar_block(self, concept: str) -> str:
        """Heading + bullets + Sources line. Used by live walkthrough."""
        bullets, src = self._render_bullets_and_sources(concept)
        lines = [f"**{concept}**"]
        if bullets:
            lines.append(bullets)
        if src:
            lines.append("")
            lines.append(src)
        return "\n".join(lines)

    def _render_pillar_block_no_sources(self, concept: str) -> str:
        """Heading + bullets only — no Sources line. Used by summary."""
        bullets, _ = self._render_bullets_and_sources(concept)
        # Also strip any inline [a] [b] refs that survive inside bullet text
        bullets = _INLINE_REF_RE.sub("", bullets).strip()
        lines = [f"**{concept}**"]
        if bullets:
            lines.append(bullets)
        return "\n".join(lines)
    
    def _concept_grounding(self, concept: str) -> str:
        """Q&A grounding: description + key-questions, refs stripped. Change log: 2026-05-31"""
        from backend import knowledge_base as kb
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == concept.lower()),
            None
        )
        if pillar:
            pts = [re.sub(r"\s+", " ", _INLINE_REF_RE.sub("", q)).strip()
                for q in pillar.get("key_questions", [])]
            parts = []
            if pillar.get("description"):
                parts.append(pillar["description"])
            if pts:
                parts.append("\n".join(f"- {p}" for p in pts))
            return "\n\n".join(parts).strip()
        swap = kb.get_swap_concept()
        if swap and concept.lower() == self.concept_swap.config["wrong_concept"].lower():
            return "\n".join(f"- {b}" for b in swap.get("sub_bullets", []))
        return ""

    def _is_removal_doubt(self, user_input: str) -> bool:
        """Vague doubt about whether the current concept belongs. Change log: 2026-05-31"""
        try:
            parsed = classify_json(f"{CLARIFY_DOUBT_PROMPT}\n\nUser message: \"{user_input}\"")
            return bool(parsed.get("doubt")) and parsed.get("confidence", 0.0) >= CLARIFY_DOUBT_THRESHOLD
        except Exception as e:
            logging.warning(f"[CLARIFY] doubt classifier error: {e}")
            return False

    def _classify_clarify(self, reply: str) -> str:
        """remove | explain | advance. Change log: 2026-05-31"""
        try:
            parsed = classify_json(CLARIFY_RESOLVE_PROMPT.format(reply=reply))
            decision = parsed.get("decision", "explain")
            return decision if decision in ("remove", "explain", "advance") else "explain"
        except Exception as e:
            logging.warning(f"[CLARIFY] resolve classifier error: {e}")
            return "explain"

    def _ask_clarify(self):
        concept = self._current_concept() or "this concept"
        msg = (
            f"Just to make sure I understand — would you like to remove "
            f"**{concept}** from the framework, or shall I explain why it's included?"
        )
        self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
        log_agent_response(self.session_id, msg)
        yield msg

    def _resolve_pending_clarify(self, user_input: str):
        concept = self.pending_clarify
        self.pending_clarify = None
        decision = self._classify_clarify(user_input)
        wrong    = self.concept_swap.config["wrong_concept"]
        on_swap  = (concept is not None and concept.lower() == wrong.lower()
                    and self.swap_presented and not self.concept_swap.is_detected)
        logging.info(f"[CLARIFY] concept='{concept}' decision={decision} on_swap={on_swap}")

        if decision == "remove":
            if on_swap:
                self.concept_swap.force_detected()
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)
                self.walkthrough_index += 1
                log_memory_override(self.session_id, old_context=f"included: {wrong}",
                                    new_context=f"user removed wrong concept via clarify: {wrong}")
                log_delete(self.session_id, wrong, "text")
                yield from self._stream_swap_caught()
                yield "\n\n"
                nxt = self._current_concept()
                yield from (self._stream_summary() if nxt is None
                            else self._stream_concept(is_first=False))
            else:
                self.pending_excl = self._resolve_to_concept_name(concept)
                log_delete(self.session_id, self.pending_excl, "text")
                yield from self._stream_pushback("concept", self.pending_excl)
        elif decision == "advance":
            self.walkthrough_index += 1
            nxt = self._current_concept()
            yield from (self._stream_summary() if nxt is None
                        else self._stream_concept(is_first=False))
        else:  # explain — grounded answer, stay on concept
            yield from self._stream_concept_qa()
    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # Change log: 2026-05-12 — override first, swap gated
    # Change log: 2026-05-16 — skip log_memory_override for concept_added
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # Gate flag — flips True on the user's FIRST main-phase message, before any
        # early-return branch (pending_add / pending_clarify / pending_excl / fw), so
        # every walkthrough message counts. app.py reads this agent-agnostically.
        # Change log: 2026-05-31
        self.has_main_contribution = True

        # ── 0. Resolve pending state ───────────────────────────────────────
        if self.pending_add is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(
                types.Content(role="user", parts=[types.Part(text=user_input)])
            )
            yield from self._resolve_pending_add(user_input)
            return
        
        if self.pending_clarify is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            yield from self._resolve_pending_clarify(user_input)
            return
        
        if self.pending_sub_excl is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
            yield from self._resolve_pending_sub_excl(user_input)
            return

        if self.pending_excl is not None:
            log_user_message(self.session_id, user_input)
            self.history.append(
                types.Content(role="user", parts=[types.Part(text=user_input)])
            )
            yield from self._resolve_pending(user_input)
            return

        # ── 1. Single intent router (replaces _detect_override + advance + doubt).
        #    Change log: 2026-06-04 ─────────────────────────────────────────────
        just_added_concept = None
        intent_obj = self._classify_intent(user_input)
        intent = intent_obj["intent"]
        detail = intent_obj.get("detail")
        logging.info(f"[INTENT] {intent} — detail={detail!r}")

        override = None   # reused to drive the existing step-6 routing branches
        wrong = self.concept_swap.config["wrong_concept"]

        # ── 2. Swap detection — UNCHANGED behaviour, now fed by the intent.
        #    (Katie 2026-06-04: keep swap detection identical; router only changes
        #    non-swap routing.) Explicit 'remove' on the swap = 2a (old 1b); the
        #    semantic detector is the backstop for non-steering intents = 2b (old step 2).
        cs_detected = False

        # 2a. Explicit rejection of the swap.
        if (intent == "remove" and self.swap_presented
                and not self.concept_swap.is_detected):
            detail_lower = (detail or "").lower().strip()
            wrong_lower  = wrong.lower().strip()
            current      = self._current_concept()
            names_swap = (
                detail_lower == wrong_lower or
                (len(detail_lower) >= 5 and detail_lower in wrong_lower) or
                (len(wrong_lower) >= 5 and wrong_lower in detail_lower)
            )
            on_swap_now = current is not None and current.lower() == wrong_lower
            if names_swap or on_swap_now:
                self.concept_swap.force_detected()
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)
                self.walkthrough_index += 1
                log_memory_override(
                    self.session_id,
                    old_context=f"included: {wrong}",
                    new_context=f"user removed wrong concept explicitly: {wrong}",
                )
                logging.info(f"[CONCEPT SWAP] detected via explicit removal")
                cs_detected = True

        # 2b. Semantic backstop — runs for non-steering intents while the swap is live
        #    (mirrors the old 'swap_presented and not override').
        if (not cs_detected and self.swap_presented
                and not self.concept_swap.is_detected
                and intent in ("question", "doubt", "advance", "none")):
            cs_detected = self.concept_swap.check_detection(user_input)
            if cs_detected:
                log_memory_override(
                    self.session_id,
                    old_context=f"included: {wrong}",
                    new_context=f"user rejected: {wrong}",
                )
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)
                self.walkthrough_index += 1
                logging.info(f"[SWAP] caught — index→{self.walkthrough_index}")

        # ── 3. Translate intent → routing setup (skipped if the swap was caught) ──
        if not cs_detected:
            if intent == "redo":
                self.walkthrough_active   = False
                self.walkthrough_done     = False
                self.walkthrough_index    = 0
                self.walkthrough_concepts = []
                self.excluded_concepts    = []
                self.swap_presented       = False
                self.swap_position        = 0
                self.pending_excl         = None
                self.pending_add          = None
                self.pending_clarify = None
                self.pending_sub_excl     = None
                self.excluded_sub_bullets = {}
                self.has_main_contribution = False
                self.user_added_pillars   = []
                self.user_sub_points      = {}
                if self.concept_swap.is_detected:
                    self.history = self._strip_concept_swap_from_history()
                yield "Noted! Let me start the walkthrough fresh...\n\n"
                log_user_message(self.session_id, "[REDO TRIGGERED]")

            elif intent == "remove":
                # Disambiguate: whole pillar, or one sub-bullet? Change log: 2026-06-04
                rt = self._classify_removal_target(user_input)
                if rt["level"] == "sub_bullet" and rt.get("bullet"):
                    self.pending_sub_excl = {
                        "pillar": rt.get("pillar") or self._current_concept(),
                        "bullet": rt["bullet"],
                    }
                    override = {"type": "pending_sub_excl_set"}
                    logging.info(f"[OVERRIDE] sub-bullet removal pending: "
                                 f"'{rt['bullet']}' under '{self.pending_sub_excl['pillar']}'")
                    log_delete(self.session_id, rt["bullet"], "text")   # intent
                else:
                    excl = rt.get("pillar") or detail or self._current_concept()
                    if excl:
                        self.pending_excl = excl
                        override = {"type": "pending_excl_set"}
                        logging.info("[OVERRIDE] concept exclusion pending: " + excl)
                        log_delete(self.session_id, excl, "text")   # intent (#1)

            elif intent == "add" and detail:
                # Concept-first resolver picks the best placement target; user confirms.
                classification = self._resolve_add_target(detail)
                self.pending_add = {
                    "item":   detail,
                    "kind":   classification["kind"],
                    "target": classification["target"],
                }
                override = {"type": "pending_add_set"}
                logging.info(f"[ADD] pending: '{detail}' "
                             f"kind={self.pending_add['kind']} "
                             f"target={self.pending_add['target']}")

        # ── 4. Log and append user message ────────────────────────────────
        log_user_message(self.session_id, user_input)
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        # ── 5. Routing log ─────────────────────────────────────────────────
        logging.info(f"[ROUTE] active={self.walkthrough_active}, "
                     f"done={self.walkthrough_done}, "
                     f"swap_presented={self.swap_presented}, "
                     f"index={self.walkthrough_index}, "
                     f"swap_position={self.swap_position}, "
                     f"cs_detected={cs_detected}")

        # ── 6. Route ───────────────────────────────────────────────────────
        if not self.walkthrough_active:
            self.walkthrough_concepts = self._build_walkthrough_concepts()
            self.walkthrough_active   = True
            self.walkthrough_index    = 0
            self.swap_presented       = False
            yield from self._stream_concept(is_first=True)

        elif self.walkthrough_done:
            # Check if user wants to revisit a past pillar before falling through to freeform.
            revisit = self._match_pillar(user_input)
            past = [c for c in self.walkthrough_concepts[:self.walkthrough_index + 1]
                    if c.lower() not in [e.lower() for e in self.excluded_concepts]]
            if revisit and revisit in past:
                self.walkthrough_index = self.walkthrough_concepts.index(revisit)
                yield f"Going back to **{revisit}** — here's where we left off.\n\n"
                yield from self._stream_concept(is_first=False)
            else:
                yield from self._stream_freeform(cs_detected)

        elif override and override["type"] == "pending_excl_set":
            yield from self._stream_pushback("concept", self.pending_excl)

        elif override and override["type"] == "pending_sub_excl_set":
            yield from self._stream_sub_bullet_pushback(
                self.pending_sub_excl["pillar"], self.pending_sub_excl["bullet"]
            )

        elif override and override["type"] == "pending_add_set":
            yield from self._ask_add_placement()

        elif cs_detected:
            yield from self._stream_swap_caught()
            yield "\n\n"
            next_concept = self._current_concept()
            if next_concept is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)

        elif intent == "advance":
            self.walkthrough_index += 1
            concept = self._current_concept()
            if concept is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)

        elif intent == "doubt":
            current  = self._current_concept()
            _on_swap = (self.swap_presented
                        and not self.concept_swap.is_detected
                        and current is not None
                        and self._is_wrong_concept(current))
            log_question(self.session_id, "text", detail=user_input[:200])
            if _on_swap:
                log_swap_questioned(self.session_id, "text", detail=user_input[:200])
            self.pending_clarify = current
            logging.info(f"[CLARIFY] doubt on '{self.pending_clarify}' — asking")
            yield from self._ask_clarify()

        else:
            # question / none / add-without-detail → KB-grounded Q&A.
            # _stream_concept_qa grounds on the concept's real description +
            # key questions from knowledge_base.json (via _concept_grounding).
            current  = self._current_concept()
            _on_swap = (self.swap_presented
                        and not self.concept_swap.is_detected
                        and current is not None
                        and self._is_wrong_concept(current))
            log_question(self.session_id, "text", detail=user_input[:200])
            if _on_swap:
                log_swap_questioned(self.session_id, "text", detail=user_input[:200])
            yield from self._stream_concept_qa(just_added=just_added_concept)

    # ══════════════════════════════════════════════════════════════════════
    # Pending resolution + pushback
    # Change log: 2026-05-16 — removed log_memory_override from _resolve_pending
    # ══════════════════════════════════════════════════════════════════════

    def _resolve_to_concept_name(self, name: str) -> str:
        for c in self.walkthrough_concepts:
            if c.lower() == name.lower():
                return c
        fallback = self._current_concept() or name
        logging.info(f"[RESOLVE] '{name}' not in walkthrough_concepts — fallback: '{fallback}'")
        return fallback

    def _resolve_pending(self, user_input: str):
        # 3-way gate (parity with BlackBox _resolve_pending_excl): confirm | decline |
        # other. Fixes the bug where an advance-like reply ("next"/"move on") after a
        # DECLINED removal silently confirmed it. Change log: 2026-06-04
        cls      = self._classify_confirmation(user_input)   # confirm | decline | other
        decision = cls["decision"]

        # ── Pending concept removal ────────────────────────────────────────
        if self.pending_excl is not None:
            excl = self.pending_excl

            # confirm → remove
            if decision == "confirm":
                excl = self._resolve_to_concept_name(excl)
                if excl not in self.excluded_concepts:
                    self.excluded_concepts.append(excl)
                self.pending_excl = None
                logging.info("[PENDING] exclusion confirmed: " + excl)
                self.walkthrough_index += 1
                yield "Understood — removing **" + excl + "** from the framework. Let's continue.\n\n"
                concept = self._current_concept()
                if concept is None:
                    yield from self._stream_summary()
                else:
                    yield from self._stream_concept(is_first=False)

            # decline → KEEP (clear pending, stay on concept)
            elif decision == "decline":
                self.pending_excl = None
                logging.info("[PENDING] exclusion declined — keeping: " + excl)
                msg = (
                    "No problem — I'll keep **" + excl + "** in the framework.\n\n"
                    "*Would you like to add, change, or question anything here? "
                    "Or shall we move on to the next pillar? Feel free to raise any "
                    "pillar you think is important.*"
                )
                self.history.append(
                    types.Content(role="model", parts=[types.Part(text=msg)])
                )
                log_agent_response(self.session_id, msg)
                yield msg

            # other → re-offer with reasoned counterfactual, KEEP pending
            else:
                if cls.get("is_question"):
                    log_question(self.session_id, "text", detail=user_input[:200])
                logging.info("[PENDING] exclusion unresolved (other) — re-offering: " + excl)
                yield from self._stream_pushback("concept", excl)

        # ── Pending framework switch (same 3-way) ──────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    # Addition placement flow — Change log: 2026-05-28
    # ══════════════════════════════════════════════════════════════════════

    def _ask_add_placement(self):
        """Ask user to confirm where to add their item (static, no LLM).
        Change log: 2026-06-04 — when the target pillar hasn't been presented yet,
        gloss it with one sentence so the placement choice isn't opaque, and drop the
        'bring in / existing pillar' wording (the target often already IS a pillar)."""
        from backend import knowledge_base as kb
        item   = self.pending_add["item"]
        target = self.pending_add["target"]

        # If the target is a real pillar the user hasn't been shown yet, add one
        # sentence describing it so the choice isn't about an opaque name.
        gloss = ""
        if target:
            tp = next((p for p in kb.get_all_pillars()
                       if p["name"].lower() == target.lower()), None)
            presented = (target in self.walkthrough_concepts
                         and self.walkthrough_concepts.index(target) <= self.walkthrough_index)
            if tp and tp.get("description") and not presented:
                first = tp["description"].split(". ")[0].strip().rstrip(".")
                if first:
                    gloss = f" {first}."

        if target:
            msg = (
                f"Good idea. **{item}** sounds like it belongs under **{target}**.{gloss} "
                f"Would you like to add it under **{target}**, or as its own separate area?"
            )
        else:
            msg = (
                f"Good idea. Would you like to add **{item}** as a separate area "
                f"of the framework, or does it fit under one of the existing pillars?"
            )

        self.history.append(
            types.Content(role="model", parts=[types.Part(text=msg)])
        )
        log_agent_response(self.session_id, msg)
        yield msg

    def _resolve_pending_add(self, user_input: str):
        """Resolve user's confirmation reply for a pending addition."""
        from backend import knowledge_base as kb

        resolved = self._resolve_addition(user_input)
        item     = self.pending_add["item"]

        # ── User cancelled ─────────────────────────────────────────────────
        if not resolved["confirmed"]:
            self.pending_add = None
            logging.info(f"[ADD] cancelled: '{item}'")
            yield "No problem, let's leave that out for now.\n\n"
            concept = self._current_concept()
            if concept is None:
                yield from self._stream_summary()
            else:
                yield f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
            return

        # ── CASE B: new pillar (user chose 'separate area') ────────────────
        # Naming the AREA → add_pillar (consistent with BlackBox Stage 1).
        if resolved["as_new_pillar"]:
            matched_pillar = self._match_withheld_pillar(item)
            if matched_pillar:
                self.walkthrough_concepts.append(matched_pillar)
                self.user_added_pillars.append(matched_pillar)
                log_concept_added(self.session_id, matched_pillar)
                self.pending_add = None
                logging.info(f"[ADD] new pillar matched withheld: '{matched_pillar}'")
                log_add_pillar(self.session_id, matched_pillar, "text")
                yield (
                    f"Good call. **{matched_pillar}** is an important area, "
                    f"we'll cover it later in the walkthrough.\n\n"
                    f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
                )
            else:
                self.walkthrough_concepts.append(item)
                self.user_added_pillars.append(item)
                log_concept_added(self.session_id, item)
                log_add_pillar(self.session_id, item, "text")
                self.pending_add = None
                logging.info(f"[ADD] new pillar (no match): '{item}'")
                yield (
                    f"Noted. We'll add **{item}** as a separate area "
                    f"toward the end of the walkthrough.\n\n"
                    f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
                )
            return

        # ── CASE A: sub-bullet into a pillar → add_sub_bullet ──────────────
        # Naming a POINT inside an area. If the resolved target is a withheld pillar,
        # INSERT it right after the current concept so it is walked next as a normal
        # step (NO block dumped here, NO add_pillar). If it's an already-shown pillar,
        # we show the updated block inline as before. Change log: 2026-06-04
        target   = resolved["target"] or self.pending_add["target"] or self._current_concept()
        revealed = self._reveal_withheld_pillar(target)
        match    = self._match_key_question(item, target)

        if target not in self.user_sub_points:
            self.user_sub_points[target] = []

        if match:
            stored = match["question"]                  # keeps its inline refs
            kbp = next((p for p in kb.get_all_pillars() if p["name"].lower() == target.lower()), None)
            already_shown = any(
                _INLINE_REF_RE.sub("", stored).strip().lower()
                == _INLINE_REF_RE.sub("", b).strip().lower()
                for b in (kbp.get("sub_bullets", []) if kbp else [])
            )
            if not already_shown and stored not in self.user_sub_points[target]:
                self.user_sub_points[target].append(stored)
        else:
            stored = self._format_sub_bullet(item)      # terse, no raw input
            if stored not in self.user_sub_points[target]:
                self.user_sub_points[target].append(stored)

        log_concept_added(self.session_id, item)
        log_add_sub_bullet(self.session_id, stored, "text")
        self.pending_add = None
        logging.info(f"[ADD] sub-bullet under '{target}' "
                     f"(matched={bool(match)}, revealed={revealed})")

        if revealed:
            # Withheld pillar just surfaced — discuss it NOW. It was inserted right
            # after the current concept (swap stays ahead), so advance to it and present
            # it as a normal walk-through step. Change log: 2026-06-04
            self.walkthrough_index += 1
            yield f"Good point — let's bring in **{target}** now.\n\n"
            yield from self._stream_concept(is_first=False)
        else:
            target_idx = (self.walkthrough_concepts.index(target)
                          if target in self.walkthrough_concepts
                          else self.walkthrough_index)
            if target_idx > self.walkthrough_index:
                # Not yet reached → lift it to the next slot and discuss it now
                # (Katie 2026-06-04). Swap just shifts back one — it still shows.
                self._move_pillar_to_next(target)
                self.walkthrough_index += 1
                yield f"Good point — that fits **{target}**; let's bring it forward now.\n\n"
                yield from self._stream_concept(is_first=False)
            else:
                # current or passed pillar → update in place, stay here
                intro = ("Good point, that's an important angle. " if match else
                         "I don't have a source for this one, but it's a good addition. ")
                yield (
                    f"{intro}I've added it under **{target}**. Here's how it looks now:\n\n"
                    f"{self._render_pillar_block(target)}\n\n"
                    f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
                )

    def _stream_pushback(self, pending_type: str, detail: str):
        concept = self._current_concept() or "the current concept"

        if pending_type == "concept":
            instruction = (
                "You are a strategic consultant. The user wants to remove "
                "**" + detail + "** from the framework. "
                "Push back using counterfactual reasoning.\n\n"
                "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
                "1. One sentence: If we remove " + detail + ", then [consequence].\n"
                "2. One sentence grounding it in the case context.\n"
                "3. End with: Would you still like to remove it?\n\n"
                "─── RULES ──────────────────────────────────────────────────────────────────\n"
                "- Consulting reasoning only\n"
                "- Soft tone — consequence as information, not command\n"
                "- Do NOT refuse the removal\n"
                "- Do NOT present any other concept\n"
                "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
                "Concept questioned: **" + detail + "**\n"
                "Current concept: **" + concept + "**\n"
                "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
                "─────────────────────────────────────────────────────────────────────\n"
            )
        else:
            instruction = (
                "You are a strategic consultant. The user wants to switch to "
                "**" + detail + "** framework. "
                "Push back using counterfactual reasoning.\n\n"
                "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
                "1. One sentence: If we switch to " + detail + ", we would [consequence].\n"
                "2. One sentence: why current framework fits this case.\n"
                "3. End with: Would you still like to switch?\n\n"
                "─── RULES ──────────────────────────────────────────────────────────────────\n"
                "- Consulting reasoning only\n"
                "- Soft tone — consequence as information, not command\n"
                "- Do NOT refuse the switch\n"
                "- Do NOT present any concept yet\n"
                "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
                "Proposed: **" + detail + "**\n"
                "Current: **" + self.kg_context["framework"] + "**\n"
                "Case: " + self.kg_context["case_type"] + "\n"
                "─────────────────────────────────────────────────────────────────────\n"
            )

        yield from self._stream_with_instruction(instruction=instruction)

    # ── Sub-bullet removal: reasoned pushback + 3-way confirm ──────────────
    # Change log: 2026-06-04 — mirrors the pillar-removal flow at sub-bullet level.

    def _stream_sub_bullet_pushback(self, pillar: str, bullet: str):
        """Reasoned counterfactual for dropping ONE point (not the whole pillar)."""
        concept = self._current_concept() or pillar
        instruction = (
            "You are a strategic consultant. The user wants to remove ONE specific "
            "point from the **" + pillar + "** pillar (not the whole pillar). "
            "Push back briefly using counterfactual reasoning.\n\n"
            "─── RESPONSE FORMAT ──────────────────────────────────────────────────────\n"
            "1. One sentence: If we drop this point, then [specific consequence].\n"
            "2. End with exactly: Would you still like to remove it?\n\n"
            "─── RULES ──────────────────────────────────────────────────────────────────\n"
            "- Consulting reasoning only; soft tone — consequence as information.\n"
            "- This is about ONE point, NOT the whole pillar — do not threaten to remove the pillar.\n"
            "- Do NOT present any other concept.\n"
            "─── CONTEXT ──────────────────────────────────────────────────────────────────\n"
            "Point to remove: \"" + bullet + "\"\n"
            "Pillar: **" + pillar + "** | Current concept: **" + concept + "**\n"
            "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
            "─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _resolve_pending_sub_excl(self, user_input: str):
        """3-way gate for a pending sub-bullet removal: confirm | decline | other."""
        pillar = self.pending_sub_excl["pillar"]
        bullet = self.pending_sub_excl["bullet"]
        cls      = self._classify_confirmation(user_input)   # confirm | decline | other
        decision = cls["decision"]

        # confirm → record the exclusion (render + summary will omit it)
        if decision == "confirm":
            self.excluded_sub_bullets.setdefault(pillar, [])
            if not self._is_excluded_bullet(pillar, bullet):
                self.excluded_sub_bullets[pillar].append(bullet)
            self.pending_sub_excl = None
            log_memory_override(self.session_id,
                old_context=f"sub-bullet in {pillar}: {bullet}",
                new_context="user confirmed removal")
            logging.info(f"[PENDING] sub-bullet removal confirmed under '{pillar}'")
            yield (
                f"Done — I've removed that point from **{pillar}**. Here's how it looks now:\n\n"
                f"{self._render_pillar_block(pillar)}\n\n"
                f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
            )

        # decline → keep the point
        elif decision == "decline":
            self.pending_sub_excl = None
            logging.info(f"[PENDING] sub-bullet removal declined — keeping point under '{pillar}'")
            msg = (
                f"No problem — I'll keep that point in **{pillar}**.\n\n"
                f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
            )
            self.history.append(types.Content(role="model", parts=[types.Part(text=msg)]))
            log_agent_response(self.session_id, msg)
            yield msg

        # other → re-offer the reasoned pushback, KEEP pending
        else:
            if cls.get("is_question"):
                log_question(self.session_id, "text", detail=user_input[:200])
            logging.info(f"[PENDING] sub-bullet removal unresolved (other) — re-offering")
            yield from self._stream_sub_bullet_pushback(pillar, bullet)

    def _stream_concept(self, is_first: bool):
        concept = self._current_concept()
        if concept is None:
            yield from self._stream_summary()
            return

        is_wrong   = self._is_wrong_concept(concept)
        swap_block = self.concept_swap.get_system_prompt_block() if is_wrong else ""

        # ── Build static prefix from JSON ──────────────────────────────────
        from backend import knowledge_base as kb

        if not is_wrong:
            # Look up pillar by name directly (walkthrough uses pillar names)
            pillar = next(
                (p for p in kb.get_all_pillars() if p["name"].lower() == concept.lower()),
                None
            )
            if pillar is None:
                # Fall back to concept-level lookup for user-added concepts
                pillar = kb.get_pillar_for_concept_name(concept)
            if pillar is None:
                # User-added concept not in KB
                note = (
                    "\n> *ℹ️ This concept isn't in my knowledge base "
                    "— I can discuss it, but can't verify it with a source.*\n"
                )
                prefix = "**" + concept + "**\n" + note
            else:
                description = pillar.get("description", "")
                # Shared body renderer — user-added bullets get their refs merged into
                # the Sources line, identical to the summary (no dangling [e][d] markers
                # mid-walkthrough). Change log: 2026-06-04
                bullet_lines, named_sources = self._render_bullets_and_sources(concept)
                sources_line  = f"\n\n{named_sources}" if named_sources else ""

                prefix = (
                    f"**{concept}**\n\n"
                    f"{description}\n\n"
                    f"{bullet_lines}"
                    f"{sources_line}\n\n"
                    f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
                )
        else:
            swap    = kb.get_swap_concept()
            bullets = swap.get("sub_bullets", []) if swap else []
            bullet_lines = "\n".join(f"- {_INLINE_REF_RE.sub('', b).strip()}" for b in bullets)
            prefix = (
                f"**{concept}**\n\n"
                f"{bullet_lines}\n\n"
                f"*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
            )

        if is_first:
            prefix = "Here is how I would structure the analysis:\n\n" + prefix

        # ── Yield static text, append to history ───────────────────────────
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=f"[Present concept: {concept}]")])
        )
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=prefix)])
        )
        log_agent_response(self.session_id, prefix)

        if is_wrong:
            self.swap_presented = True
            if not self.concept_swap.is_injected:
                self.concept_swap.maybe_inject(prefix)
                self.concept_swap.log_presented()
                logging.info(f"[SWAP] concept presented at position={self.swap_position}")

        yield prefix

    def _stream_concept_qa(self, just_added: str | None = None):
        current = self._current_concept()
        concept = current or "the current concept"
        on_swap = (self.swap_presented
                   and not self.concept_swap.is_detected
                   and current is not None
                   and self._is_wrong_concept(current))

        closing = (
            "End with exactly:\n"
            "*I can see why you'd question this — shall we include it or move on without it?*"
            if on_swap else
            "End with exactly:\n"
            "*Would you like to add, change, or question anything here? Or shall we move on to the next pillar? Feel free to raise any pillar you think is important.*"
        )

        qa_prompt = CONCEPT_QA_PROMPT.format(on_swap=on_swap)

        grounding = self._concept_grounding(concept)
        grounding_block = ""
        if grounding:
            grounding_block = (
                "─── KNOWN POINTS FOR THIS CONCEPT (ground your answer here) ──────────\n"
                + grounding + "\n"
                "─── GROUNDING RULE ──────────────────────────────────────────────────\n"
                "Base your answer on the KNOWN POINTS above and the case. You may explain\n"
                "or apply them to this case, but do NOT introduce regulations, statistics,\n"
                "named sources, or framework concepts that are not among the known points.\n"
                "Do NOT output bracketed letter markers like [a].\n"
                "──────────────────────────────────────────────────────────────────────\n"
            )


        added_note = ""
        if just_added:
            added_note = (
                "Good idea — I'll add **" + just_added
                + "** after we finish **" + concept + "**.\n\n"
            )

        instruction = (
            qa_prompt + "\n\n"
            "─── CLOSING INSTRUCTION ──────────────────────────────────────────────\n"
            + closing + "\n"
            + grounding_block
            + "─── CONTEXT ──────────────────────────────────────────────────────────\n"
            "Current concept: **" + concept + "**\n"
            "On swap concept: " + str(on_swap) + "\n"
            "Framework: " + self.kg_context["framework"] + " | Case: " + self.kg_context["case_type"] + "\n"
            "Framework concepts (in order): " + ", ".join(
                c for c in self.walkthrough_concepts
                if just_added is None or c.lower() != just_added.lower()
            ) + "\n"
            "──────────────────────────────────────────────────────────────────────\n"
        )
        # logging.info("[QA GROUNDING] present=%s concept=%s",
        #      "KNOWN POINTS FOR THIS CONCEPT" in instruction, concept)
        yield from self._stream_with_instruction(instruction=instruction, prefix=added_note)

    def _stream_swap_caught(self):
        wrong       = self.concept_swap.config["wrong_concept"]
        wrong_fw    = self.concept_swap.config["wrong_framework"]
        active_fw   = self.kg_context["framework"]
        active_case = self.kg_context["case_type"]
        instruction = (
            f"{SWAP_CAUGHT_PROMPT}\n\n"
            f"─── CONTEXT ──────────────────────────────────────────────────────────\n"
            f"Wrong concept flagged: **{wrong}**\n"
            f"It belongs to: {wrong_fw} (a different type of analysis)\n"
            f"This case is: {active_case} analysis — {active_fw}\n"
            f"─────────────────────────────────────────────────────────────────────\n"
        )
        yield from self._stream_with_instruction(instruction=instruction)

    def _stream_summary(self):
        self.walkthrough_done = True

        from backend import knowledge_base as kb

        wrong          = self.concept_swap.config["wrong_concept"].lower()
        excluded_lower = [e.lower() for e in self.excluded_concepts] + [wrong] \
                         if self.concept_swap.is_detected else \
                         [e.lower() for e in self.excluded_concepts]

        added_lower = [p.lower() for p in self.user_added_pillars]

        # ── Group 1: original pillars SHOWN to the user (incl. the one on
        #    screen at End Session). [:index+1] includes the current concept,
        #    fixing blank summaries when ending mid-walkthrough. No-op on the
        #    natural-end path (index already == len). Change log: 2026-06-02
        completed_originals = [
            c for c in self.walkthrough_concepts[:self.walkthrough_index + 1]
            if c.lower() not in excluded_lower
            and c.lower() not in added_lower
        ]

        # ── Group 2: user-added pillars (heading only) ─────────────────────
        added_pillars = [
            p for p in self.user_added_pillars
            if p.lower() not in excluded_lower
        ]

        logging.info(f"[SUMMARY] completed_originals={completed_originals}, "
                     f"added_pillars={added_pillars}")

        # ── Build the summary string DETERMINISTICALLY (no LLM) ───────────
        # Python owns rendering: the matching already happened during the
        # conversation and is stored in user_sub_points. Rendering known
        # state must be exact — an LLM would paraphrase. (Change log: 2026-05-28)
        lines = [
            "**Final Framework Summary**",
            "",
        ]

        for c in completed_originals:
            pillar = next(
                (p for p in kb.get_all_pillars() if p["name"].lower() == c.lower()),
                None
            )
            if pillar:
                lines.append(self._render_pillar_block_no_sources(c))
                lines.append("")
            else:
                lines.append(f"**{c}**")
                if c.lower() == wrong:
                    swap = kb.get_swap_concept()
                    for b in (swap.get("sub_bullets", []) if swap else []):
                        if not self._is_excluded_bullet(c, b):
                            lines.append(f"- {_INLINE_REF_RE.sub('', b).strip()}")
                for sp in self.user_sub_points.get(c, []):
                    if not self._is_excluded_bullet(c, sp):
                        lines.append(f"- {_INLINE_REF_RE.sub('', sp).strip()}")
                lines.append("")

        # User-added areas — render KB sub-bullets if it's a KB pillar (empty-summary
        # fix), else heading + the user's own points. Change log: 2026-06-03
        for p in added_pillars:
            pillar = next(
                (x for x in kb.get_all_pillars() if x["name"].lower() == p.lower()),
                None
            )
            if pillar:
                lines.append(self._render_pillar_block_no_sources(p))
                lines.append("")
            else:
                lines.append(f"**{p}**")
                for sp in self.user_sub_points.get(p, []):
                    if not self._is_excluded_bullet(p, sp):
                        lines.append(f"- {_INLINE_REF_RE.sub('', sp).strip()}")
                lines.append("")

        summary_text = "\n".join(lines).rstrip()

        # Append to history + log + store answer, mirroring _stream_with_instruction
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=summary_text)])
        )
        update_answer(self.session_id, summary_text)
        log_agent_response(self.session_id, summary_text)

        yield summary_text

        closing = (
            "\n\n---\n\n"
            "Is there anything you'd like to revisit, add to, or discuss further? "
            "You can name any pillar and I'll take you back to it — "
            "or click **‼️End Session** whenever you're ready to finish."
        )
        self.history.append(
            types.Content(role="model", parts=[types.Part(text=closing)])
        )
        log_agent_response(self.session_id, closing)
        yield closing

    def _stream_freeform(self, cs_detected: bool):
        concepts_str = " → ".join(self.kg_context["concepts"])
        instruction  = (
            f"─── FRAMEWORK CONTEXT ────────────────────────────────────────────────\n"
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n"
            f"──────────────────────────────────────────────────────────────────────\n\n"
            f"You are a strategic consultant. The user has seen the full framework. "
            f"Answer their question concisely in plain language — no jargon, no "
            f"mention of technical systems. Ask ONE follow-up question after answering."
        )
        if cs_detected:
            instruction += f"\n\n{SWAP_CAUGHT_PROMPT}"
        yield from self._stream_with_instruction(instruction=instruction)

    # ══════════════════════════════════════════════════════════════════════
    # Session + system prompt
    # ══════════════════════════════════════════════════════════════════════

    def get_summary(self):
        """
        Public wrapper for app.py to stream the summary on End Session.
        Routes through _stream_summary() so the summary is built from explicit
        session state (completed originals + added pillars), identical to the
        natural walkthrough-end summary. Replaces inherited send_message().
        Change log: 2026-05-28
        """
        yield from self._stream_summary()

    def end_session(self) -> None:
        from backend.logger import end_session as _end_session
        try:
            from firebase_admin import firestore as fs
            db = fs.client()
            db.collection('sessions').document(self.session_id).update({
                'concept_swap_detected': self.concept_swap.is_detected,
                'swap_detected_at_end':  self.concept_swap.is_detected,
            })
            logging.info(f'[END SESSION] stamped session={self.session_id}, '
                         f'swap_detected={self.concept_swap.is_detected}')
        except Exception as e:
            logging.warning(f'[END SESSION] Firestore stamp failed: {e}')
        _end_session(self.session_id)

    def _build_system_prompt(self) -> str:
        concepts_str = " → ".join(self.kg_context["concepts"]) \
                       if self.kg_context["concepts"] else "N/A"
        return (
            f"Framework : {self.kg_context['framework']}\n"
            f"Concepts  : {concepts_str}\n\n"
            f"You are a strategic consultant. Answer concisely in plain language. "
            f"No jargon. No mention of databases or technical systems."
        )