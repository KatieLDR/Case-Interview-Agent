import json
import logging
import re
from google.genai import types
from backend.agents.base import BaseAgent           # Step 6b: sibling of BaseAgent (F-ARCH2)
from backend.llm import (
    CLASSIFIER_MODEL, MAIN_MODEL, client, classify_json, ANSWER_THRESHOLD,
)
from backend.knowledge.cases import get_case, get_clarification_facts
from backend.tools.concept_swap import ConceptSwap
from backend.logger import (
    create_session, log_user_message, log_agent_response,
    log_interruption, update_answer,
)
from backend.logging import events as ev
from backend.logging.sink import firestore_sink as _sink
from backend.tools.rag_explainer import build_citation_header, check_and_append_warning
from backend.domain import matching, grounding          # Step 2: shared KB matchers + grounding
from backend.interaction import intents                      # Step 3: unified intent taxonomy (I-2)
from backend.interaction import handlers                     # Step 4c: shared handlers + PendingAction (I-1)

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

# Single intent router retired (Step 3): routing now goes through the shared
# backend.interaction.intents.classify_intent (ONE taxonomy across all arms, I-1/I-2).
# The local router prompt + the per-arm intent method were removed here; the
# canonical prompt lives in the interaction layer. See REFACTOR_PLAN §S Step 3.

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

# ── ADD matchers moved to domain/ (Step 2) ─────────────────────────────────
# ADD_MATCH_PROMPT / ADD_PILLAR_MATCH_PROMPT / ADD_CONCEPT_MATCH_PROMPT and the
# ADD_MATCH_THRESHOLD / CONCEPT_MATCH_THRESHOLD thresholds are now single-sourced:
# the prompts live in backend.domain.matching, the thresholds in backend.llm.
# _match_key_question below delegates to
# matching.*; ADD_PILLAR_MATCH_PROMPT + ADD_MATCH_THRESHOLD are imported at the
# top of this module solely so the Explainable-only _match_withheld_pillar (a
# withheld-candidate variant, not one of the shared matchers) keeps the canonical
# prompt/threshold rather than a private copy.

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

class ExplainableAgent(BaseAgent):
    """
    Explainable agent — stateful concept-by-concept walkthrough.

    Inherits from BaseAgent (Step 6b; method list below may predate later steps):
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
    Change log: 2026-06-08 — Step 3: routing switched to the SHARED interaction-layer
                             classifier (one taxonomy across arms, I-1/I-2). Local
                             router prompt + per-arm intent method retired; redo
                             removed (W4, §0 no session wipe); swap gate re-expressed
                             as non-steering (W2); new shared intents handled —
                             ask_agent_to_suggest via the shared handlers.suggest_handler
                             (grounded, not free-form), revisit navigates a passed pillar.
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
        self._init_flow_state()
        self.session_id    = create_session(user_id, agent_type="explainable")
        self.original_case = get_case("explainable")
        self.has_main_contribution = False   # gates End Session button (app.py reads this)

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
        self.swap_presented       = False
        self.swap_position        = 0

        # ── Step 4c: shared HandlerSession flow state (replaces pending_excl /
        #    pending_add / pending_clarify / pending_sub_excl). interaction/handlers.py
        #    owns add/remove/suggest/question + the two-turn removal loop now. The
        #    walkthrough cursor stays EXP persona state, driven by the renderers. ──

        # ── User sub-points — populated by duplicate guard path ───────────
        # Change log: 2026-05-12

        # ── Removed sub-bullets — pillar name → list of removed bullet texts.
        # Render + summary omit any bullet whose normalised text is in here.
        # Change log: 2026-06-04

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
        position = min(1, len(base))   # C3: fixed EARLY 2nd slot (dropout-safe); EXP appends adds (no shift needed)
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

    # ══════════════════════════════════════════════════════════════════════
    # Addition flow helpers — Change log: 2026-05-28
    # ══════════════════════════════════════════════════════════════════════

    def _match_key_question(self, item: str, pillar_name: str) -> dict | None:
        """Matching DECISION → shared domain matcher (Step 2). Explainable then recovers
        the VERBATIM KB question (inline [a] refs KEPT) plus its sources so citations
        render — the shared matcher returns ref-stripped text, so we re-find the raw
        question by its stripped form (matching._strip_source_refs == the matcher's own
        stripping). Contract unchanged: returns {"question", "sources"} or None."""
        text, _score = matching.match_key_question(item, pillar_name)
        if not text:
            return None
        from backend.knowledge import knowledge_base as kb
        pillar = next(
            (p for p in kb.get_all_pillars() if p["name"].lower() == pillar_name.lower()),
            None
        )
        if pillar is None:
            return None
        raw = next(
            (q for q in pillar.get("key_questions", [])
             if matching._strip_source_refs(q) == text),
            text,
        )
        return {"question": raw, "sources": pillar.get("key_questions_sources", "")}

    # _match_pillar / _match_concept collapsed (Step 6c, I-3). _match_key_question
    # is kept above: EXP citation recovery (persona); its decision still delegates
    # to matching.match_key_question.

    def _is_excluded_bullet(self, pillar_name: str, bullet: str) -> bool:
        """→ shared domain predicate (Step 2; excluded-map passed by value). The local
        _norm was retired — matching owns ref-insensitive normalisation now."""
        return matching.is_excluded_bullet(self.excluded_sub_bullets, pillar_name, bullet)

    def _last_agent_message(self) -> str:
        for c in reversed(self.history):
            if c.role == "model" and c.parts and c.parts[0].text:
                return c.parts[0].text[:600]
        return ""

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
        from backend.knowledge import knowledge_base as kb
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
        """Q&A grounding → shared grounding.ground_pillar (Step 2): description +
        key-questions (refs stripped), plus the planted-swap fallback. Behaviour-
        preserving — the shared fallback keys off the KB swap concept's own name (==
        this arm's wrong_concept) and the swap sub-bullets carry no [a] refs, so the
        shared ref-strip is a no-op. Sources stay an Explainable-only render concern,
        layered elsewhere, never in grounding."""
        return grounding.ground_pillar(concept)

    # ══════════════════════════════════════════════════════════════════════
    # Main phase — stateful walkthrough router
    # Change log: 2026-05-12 — override first, swap gated
    # Change log: 2026-05-16 — skip log_memory_override for concept_added
    # ══════════════════════════════════════════════════════════════════════

    def _stream_main(self, user_input: str):
        if self._pending:
            log_interruption(self.session_id, context=user_input)

        # Gate flag — flips True on the user's FIRST main-phase message; app.py reads
        # it agent-agnostically. Change log: 2026-05-31
        self.has_main_contribution = True

        # ── 1. Unified intent (Step 3). Context = current concept + its non-excluded
        #      points + walkthrough pillars + last agent msg (W8: no confidence floor). ──
        from backend.knowledge import knowledge_base as kb
        _cur    = self.current_pillar() or "(none)"
        _pillar = next((p for p in kb.get_all_pillars()
                        if p["name"].lower() == _cur.lower()), None)
        _bul = list(_pillar.get("sub_bullets", [])) if _pillar else []
        _bul += self.user_sub_points.get(_cur, [])
        _bul = [b for b in _bul if not self._is_excluded_bullet(_cur, b)]
        res = intents.classify_intent(
            user_input,
            current_pillar=_cur,
            current_bullets="\n".join(f"- {b}" for b in _bul) or "(none)",
            walkthrough_pillars=", ".join(self.walkthrough_concepts) or "(none)",
            last_agent=self._last_agent_message() or "(nothing yet)",
        )
        intent = res.intent
        logging.info(f"[INTENT] {intent} — detail={res.detail!r} parent={res.parent!r}")

        # ── 1a. Swap semantic backstop — W2: only on a FRESH non-steering turn (no parked
        #      removal/suggestion), the faithful equivalent of the old 'swap_presented and
        #      not override' gate. A NAMED swap removal is intent==remove and is detected
        #      inside removal_handler (challenge → confirm → detection, §0 — no delete),
        #      so it is correctly excluded here. ──
        if (self.pending is None and self.pending_suggestion is None
                and self.swap_presented and not self.concept_swap.is_detected
                and intent not in ("add", "remove", "question")):
            if self.concept_swap.check_detection(user_input):       # sets is_detected
                wrong = self.concept_swap.config["wrong_concept"]
                if wrong not in self.excluded_concepts:
                    self.excluded_concepts.append(wrong)            # skip it in the walk
                # check_detection() already fired §3.6 swap_detected via ConceptSwap._log_detected.
                logging.info("[SWAP] caught via semantic backstop")
                log_user_message(self.session_id, user_input)
                self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
                yield from self._stream_swap_caught()
                yield "\n\n"
                if self.current_pillar() is None:
                    yield from self._stream_summary()
                else:
                    yield from self._stream_concept(is_first=False)
                return

        # ── 2. Log + append the user turn (parity with baseline ordering). ──
        log_user_message(self.session_id, user_input)
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        # ── 3. Shared handler dispatch (Step 4c). A parked removal / pending suggestion is
        #      resolved inside dispatch; snapshot it first so the renderer can recover
        #      pillar/type after the PendingAction machine clears it. ──
        was_pending = self.pending is not None
        pa_snapshot = self.pending
        self._last_intent = intent          # 6h-2: render_fallback reads this (intent dropped from the public seam)
        outcome = handlers.dispatch(res, self, user_text=user_input)
        yield from self._render_outcome(outcome, user_input,
                                        was_pending=was_pending, pa=pa_snapshot)

    # ══════════════════════════════════════════════════════════════════════
    # Step 4c — HandlerSession adapter (D-H3) + outcome renderer.
    #   The shared layer (interaction/handlers.py) does the invariant work and
    #   returns a structured Outcome; Explainable renders it (citations +
    #   counterfactual + the walkthrough cursor) and fires the §3.6 events
    #   DRIVEN BY the outcome (D-H1). Delete fires only at stage="confirmed"
    #   (F-R1). Swap removal = DETECTION, never a delete (§0).
    # ══════════════════════════════════════════════════════════════════════

    # ── HandlerSession queries ─────────────────────────────────────────────
    def _presented_concepts(self) -> list:
        """Walkthrough concepts presented so far (up to & incl. the current slot),
        minus excluded. Read-only (never advances the cursor)."""
        excluded = [e.lower() for e in self.excluded_concepts]
        return [c for c in self.walkthrough_concepts[:self.walkthrough_index + 1]
                if c.lower() not in excluded]

    def presented_pillars(self) -> list:
        return self._presented_concepts()

    def presented_sub_bullets(self) -> dict:
        """{concept -> [non-excluded bullet texts]} for each presented concept (KB
        sub-bullets ref-stripped + user sub-points). Drives the removal existence guard."""
        from backend.knowledge import knowledge_base as kb
        out = {}
        for name in self._presented_concepts():
            kbp = next((p for p in kb.get_all_pillars()
                        if p["name"].lower() == name.lower()), None)
            bl = []
            if kbp:
                bl += [matching._strip_source_refs(b) for b in kbp.get("sub_bullets", [])
                       if not self._is_excluded_bullet(name, b)]
            bl += [_INLINE_REF_RE.sub("", sp).strip()
                   for sp in self.user_sub_points.get(name, [])
                   if not self._is_excluded_bullet(name, sp)]
            out[name] = bl
        return out

    def surfaced_pillar_names(self) -> set:
        """Everything already surfaced (shown KB pillars / in the walk / user-added /
        excluded). suggest_handler offers the first WITHHELD pillar NOT in this set."""
        from backend.knowledge import knowledge_base as kb
        names = {p["name"].lower() for p in kb.get_shown_pillars()}
        names |= {c.lower() for c in self.walkthrough_concepts}
        names |= {n.lower() for n in self.user_added_pillars}
        names |= {e.lower() for e in self.excluded_concepts}
        return names

    def current_pillar(self):
        """The current walkthrough concept (read-only; skips excluded WITHOUT mutating
        the cursor). None once the walkthrough is exhausted/done."""
        excluded = [e.lower() for e in self.excluded_concepts]
        idx = self.walkthrough_index
        while idx < len(self.walkthrough_concepts):
            c = self.walkthrough_concepts[idx]
            if c.lower() not in excluded:
                return c
            idx += 1
        return None

    # ── HandlerSession mutators (pure state; logging is render-driven, D-H1) ──
    def surface_pillar(self, name: str) -> None:
        """Reveal a withheld pillar OR create a novel area — both append to the walk +
        user_added for Explainable. Stash is_new so the render logs add_pillar once. A
        concept already shown in the walk (e.g. a shown-but-unreached pillar) is is_new
        =False → acknowledged, not counted."""
        in_walk = name.lower() in [c.lower() for c in self.walkthrough_concepts]
        already = name.lower() in [p.lower() for p in self.user_added_pillars]
        if in_walk or already:
            self._last_surface = {"name": name, "is_new": False}
            return
        self.walkthrough_concepts.append(name)
        self.user_added_pillars.append(name)
        self._last_surface = {"name": name, "is_new": True}

    def add_sub_point(self, pillar: str, text: str) -> None:
        """Store a new sub-point under `pillar`. Citation-preserving: a key-question match
        keeps the verbatim KB question (inline refs kept); otherwise the terse formatted
        text (Fork-B: log the STORED text). Dedup against existing user points + static KB
        bullets. Stash stored/raw/is_new so the render logs add_sub_bullet from the outcome."""
        from backend.knowledge import knowledge_base as kb
        match = self._match_key_question(text, pillar)
        if match:
            stored = match["question"]
        else:
            stored = self._format_sub_bullet(text)
        kbp = next((p for p in kb.get_all_pillars()
                    if p["name"].lower() == pillar.lower()), None)
        already_static = any(
            _INLINE_REF_RE.sub("", stored).strip().lower()
            == _INLINE_REF_RE.sub("", b).strip().lower()
            for b in (kbp.get("sub_bullets", []) if kbp else [])
        )
        self.user_sub_points.setdefault(pillar, [])
        is_new = (not already_static
                  and stored not in self.user_sub_points[pillar]
                  and not self._is_excluded_bullet(pillar, stored))
        if is_new:
            self.user_sub_points[pillar].append(stored)
        self._last_sub_add = {"pillar": pillar, "stored": stored, "raw": text, "is_new": is_new}

    # ── swap channel (PRESERVED per-arm, §0 #4) ─────────────────────────────

    def _on_swap_now(self) -> bool:
        cur = self.current_pillar()
        return cur is not None and self._is_wrong_concept(cur)


    def _extra_swap_signal(self, km, user_text: str) -> bool:
        """6e: walkthrough arms also fire when the CURRENT concept is the swap."""
        cur = self.current_pillar()
        return bool(cur and self._is_wrong_concept(cur))


    def mark_swap_detected(self) -> None:
        """Swap DETECTED on confirm — force_detected + exclude it from the walk so the
        cursor skips it. NEVER a delete event (§0); the render logs no delete."""
        self.concept_swap.force_detected()
        wrong = self.concept_swap.config["wrong_concept"]
        if wrong not in self.excluded_concepts:
            self.excluded_concepts.append(wrong)


    # ── outcome renderer ────────────────────────────────────────────────────
    # C4 (Decision 6 / F5): _swap_question_signal override REMOVED — EXP now inherits the
    # ONE shared base signal (is_injected & not detected & _classify_swap_question), so the
    # questioned-vs-detected line is drawn by one classifier across all arms and
    # count_swap_questioned is cross-arm comparable. The old EXP-only `on_swap` positional
    # term is retired. (_on_swap_now stays — still used by the doubt/consequence paths.)
    #
    # REVISED (locked swap_questioned definition): the cursor arms are POSITIONAL —
    # swap_questioned fires on ANY question while the swap is the CURRENT concept,
    # regardless of wording (matches the doubt path + HITL). C4's removal of this
    # override was the wrong call; restored. BB stays content-based (base.matches).
    def _swap_question_signal(self, outcome, user_input: str) -> bool:
        return (self.swap_presented and not self.concept_swap.is_detected
                and self._on_swap_now())

    _NEXT_AFFORD = ("\n\n*Add a point here, raise a new area, or question anything \u2014 "
                    "or say \"next\" to move on.*")

    def render_add(self, o):
        # revisit -> navigation to a PASSED pillar (the router never lets revisit carry
        # new content). Jump the cursor if it's a passed, non-excluded concept; else
        # non-destructive grounded Q&A.
        if o.action == "navigated":
            target = o.pillar
            past = [c for c in self.walkthrough_concepts[:self.walkthrough_index + 1]
                    if c.lower() not in [e.lower() for e in self.excluded_concepts]]
            if target and target in past:
                self.walkthrough_index = self.walkthrough_concepts.index(target)
                self.walkthrough_done = False
                yield f"Going back to **{target}** — here's where we left off.\n\n"
                yield from self._stream_concept(is_first=False)
            else:
                ev.question(self._evctx(), _sink)   # revisit -> grounded Q&A (no turn outcome)
                yield from self._stream_concept_qa()
            return

        if o.action == "duplicate":
            if o.level == "pillar" and o.pillar:
                msg = f"**{o.pillar}** is already part of the framework." + self._NEXT_AFFORD
            elif o.pillar and o.matched_text:
                msg = (f"That's already covered under **{o.pillar}** as *{o.matched_text}*. "
                       f"Want to adjust it?" + self._NEXT_AFFORD)
            elif o.pillar:
                msg = f"That's already covered under **{o.pillar}**." + self._NEXT_AFFORD
            else:
                msg = "That's already in the framework."
            self._emit(msg); yield msg; return

        if o.action == "revealed" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                msg = (f"Good call — **{o.pillar}** is an important area; "
                       f"we'll cover it in the walkthrough." + self._NEXT_AFFORD)
            else:
                msg = (f"**{o.pillar}** is already part of the framework — we'll get to it."
                       + self._NEXT_AFFORD)
            self._emit(msg); yield msg; return

        if o.action == "added_new" and o.level == "sub_bullet":
            st = self._last_sub_add or {}
            if st.get("is_new"):
                if o.also_covered:
                    gist = f" \u2014 {o.explanation}" if o.explanation else ""
                    also = (f" It also relates to **{o.also_covered}**{gist} "
                            f"Say the word if you'd rather it sit there.")
                else:
                    also = ""
                msg = (f"Good point — I've added it under **{o.pillar}**.{also} "
                       f"Here's how it looks now:\n\n"
                       f"{self._render_pillar_block(o.pillar)}" + self._NEXT_AFFORD)
            else:
                msg = f"That's already noted under **{o.pillar}**." + self._NEXT_AFFORD
            self._emit(msg); yield msg; return

        if o.action == "added_new" and o.level == "pillar":
            st = self._last_surface or {}
            if st.get("is_new"):
                msg = (f"Noted — I've added **{o.pillar}** as a separate area. "
                       f"What points would you like under it? Add them one at a time."
                       + self._NEXT_AFFORD)
            else:
                msg = f"**{o.pillar}** is already part of the framework."
            self._emit(msg); yield msg; return

        msg = "Noted."; self._emit(msg); yield msg          # defensive

    def render_removal(self, o, user_input, *, was_pending=False, pa=None):
        stage = o.stage
        if stage == "confirmed":
            if o.is_swap:
                # §0 — swap is DETECTION, never a delete. swap_detected+swap_removed fired by
                # _fire_turn(RemovalOutcome is_swap); mark_swap_detected ran in _confirm_removal.
                yield from self._stream_swap_caught()
                yield "\n\n"
                if self.current_pillar() is None:
                    yield from self._stream_summary()
                else:
                    yield from self._stream_concept(is_first=False)
                return
            if pa and pa.type == "remove_sub_bullet":
                msg = (f"Done — I've removed that point from **{pa.pillar}**. "
                       f"Here's how it looks now:\n\n"
                       f"{self._render_pillar_block(pa.pillar)}" + self._NEXT_AFFORD)
                self._emit(msg); yield msg; return
            # whole pillar — _confirm_removal already excluded it; the cursor auto-skips.
            yield f"Understood — removing **{o.target}** from the framework. Let's continue.\n\n"
            if self.current_pillar() is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)
            return

        if stage == "abandoned":
            if pa and pa.type == "remove_sub_bullet":
                msg = f"No problem — I'll keep that point in **{pa.pillar}**." + self._NEXT_AFFORD
            else:
                tgt = pa.target if pa else o.target
                msg = f"No problem — I'll keep **{tgt}** in the framework." + self._NEXT_AFFORD
            self._emit(msg); yield msg; return

        if stage == "nothing_to_remove":
            if o.suggest_add_alternative:
                msg = (f"**{o.suggest_add_alternative}** isn't in the current framework. "
                       f"Did you mean to *add* it? Reply **yes** to add it.")
            else:
                msg = (f"**{o.target or 'That'}** isn't part of the current framework, "
                       f"so there's nothing to remove there.")
            self._emit(msg); yield msg; return

        if stage == "needs_disambiguation":
            msg = ("Which part would you like to remove — the current concept, or a "
                   "specific point within it? You can name it.\n\n"
                   "*(Or say **never mind** to keep everything as is.)*")
            self._emit(msg); yield msg; return

        if stage == "challenged":
            if was_pending:
                if self._reply_is_question(user_input):
                    # §3.6 question (+ swap_questioned W9) already fired by _fire_turn.
                    yield from self._stream_concept_qa(); return
                msg = f"No rush — reply **yes** to remove **{o.target}**, or **no** to keep it."
                self._emit(msg); yield msg; return
            # first challenge -> Explainable's counterfactual pushback (persona render).
            if self.pending and self.pending.type == "remove_sub_bullet":
                yield from self._stream_sub_bullet_pushback(self.pending.pillar, o.target)
            else:
                yield from self._stream_pushback("concept", o.target)
            return

        msg = f"No rush — reply **yes** to remove **{o.target}**, or **no** to keep it."
        self._emit(msg); yield msg                                              # defensive

    def render_question(self, user_input):
        # §3.6 question (+ swap_questioned W9) already fired by _fire_turn at the boundary.
        if self.walkthrough_done:
            yield from self._stream_freeform(cs_detected=False)
        else:
            yield from self._stream_concept_qa()

    def render_next_steps(self, o):
        if getattr(o, "revealed", False):
            # D7 accept: the withheld pillar is now surfaced (dispatch -> surface_pillar
            # appended it to the walk). No DV (ask_agent_suggestion is Step 5).
            msg = (f"Good point — I've brought in **{o.suggested_item}**; "
                   f"we'll cover it in the walkthrough." + self._NEXT_AFFORD)
            self._emit(msg); yield msg; return
        if not getattr(o, "suggested_item", None):
            msg = ("You've surfaced the main areas I'd flag — feel free to revisit, add, "
                   "remove, or question any part of the framework.")
            self._emit(msg); yield msg; return
        why = (o.grounding or "").split("\n")[0].strip()
        msg = (f"One area we haven't covered yet is **{o.suggested_item}**"
               + (f" — {why}" if why else "")
               + ".\n\nIt's worth considering whether it applies to your case. "
                 "Shall I bring it in?")
        self._emit(msg); yield msg

    def render_fallback(self, outcome=None):
        # 6h-2: unified public seam (replaces private _render_advance + _render_fallback).
        # None -> suggest-exhausted line; AdvanceOutcome -> advance the walk; Fallback (or
        # advance-past-end) -> doubt-as-question / non-destructive affordance. Intent is read
        # from self._last_intent (the seam no longer receives it). Behavior-preserving.
        if outcome is None:                      # suggest_handler: nothing left to suggest
            msg = ("You've surfaced the main areas I'd flag — feel free to revisit, add, "
                   "remove, or question any part of the framework.")
            self._emit(msg); yield msg; return
        if isinstance(outcome, handlers.AdvanceOutcome) and not self.walkthrough_done:
            self.walkthrough_index += 1
            if self.current_pillar() is None:
                yield from self._stream_summary()
            else:
                yield from self._stream_concept(is_first=False)
            return
        # Fallback, or AdvanceOutcome past the end of the walk (intent != "doubt"):
        # doubt -> non-destructive grounded explanation (W5); W9: doubt on swap logs questioned.
        if getattr(self, "_last_intent", "none") == "doubt":
            ev.question(self._evctx(), _sink)   # doubt-as-question (FallbackOutcome has no map)
            on_swap = (self.swap_presented and not self.concept_swap.is_detected
                       and self._on_swap_now())
            if on_swap:
                ev.swap_questioned(self._evctx(), _sink)
            if self.walkthrough_done:
                yield from self._stream_freeform(cs_detected=False)
            else:
                yield from self._stream_concept_qa()
            return
        # none / 'start over' / unclear / advance-past-end -> affordance (§0: no session wipe).
        msg = ("I want to make sure I help with the right thing. You can **add** a point, "
               "**remove** something, **question** any part of the framework, ask me to "
               "**suggest** what else to consider, or say **move on** to continue.")
        self._emit(msg); yield msg

    def render_summary(self):
        # 6h-2 contract seam — EXP's terminal I-6 summary (delegates to the existing streamer).
        yield from self._stream_summary()

    def render_framework(self, preamble=""):
        # 6h-2 contract seam (7-seam symmetry; not invoked by the base router for walkthrough
        # arms). Faithful 'show current state': current concept, else the summary.
        if preamble:
            yield preamble
        if self.current_pillar() is None:
            yield from self._stream_summary()
        else:
            yield from self._stream_concept(is_first=False)

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

    # ══════════════════════════════════════════════════════════════════════
    # Addition placement flow — Change log: 2026-05-28
    # ══════════════════════════════════════════════════════════════════════

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

    def _stream_concept(self, is_first: bool):
        concept = self._current_concept()
        if concept is None:
            yield from self._stream_summary()
            return

        is_wrong   = self._is_wrong_concept(concept)
        swap_block = self.concept_swap.get_system_prompt_block() if is_wrong else ""

        # ── Build static prefix from JSON ──────────────────────────────────
        from backend.knowledge import knowledge_base as kb

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
        """#2 — NEUTRAL swap handling: no "sharp catch" praise or LLM commentary that could
        lead the participant. A plain acknowledgement; the caller advances to the next pillar
        like any removal. swap_detected / swap_removed already fired upstream."""
        wrong = self.concept_swap.config["wrong_concept"]
        msg = f"Understood — we'll set **{wrong}** aside and continue."
        self._emit(msg)
        yield msg

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
        # #2 — no swap-catch praise even in freeform; answer neutrally (swap_detected
        # already fired upstream). cs_detected retained for signature/compat.
        _ = cs_detected
        yield from self._stream_with_instruction(instruction=instruction)

    # ══════════════════════════════════════════════════════════════════════
    # Session + system prompt
    # ══════════════════════════════════════════════════════════════════════

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
