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

SUB_BULLET_FORMAT_PROMPT = """
You reformat a user's note into the style of a sub-bullet in this case framework.

Rules:
- Keep the user's MEANING exactly — add no new content, examples, or sources.
- Output ONE concise line in the framework's voice: a short topic followed by a
  clarifying question or qualifying clause. Phrasing it as a natural question is fine.
- Drop filler ("I think", "we should also", "maybe", "consider").
- No leading dash, no inline source markers. End with a question mark only if it is a question.
- Match the style of these existing framework sub-bullets:
    "How has the submission rate changed month-on-month?"
    "Data classification is data correctly classified under company data handling standards?"
    "Single-developer dependency critical point of failure if original developer leaves or moves"
    "Prototype scope and governability is the use case tightly scoped enough to be reviewable?"

User note: "{item}"

Output ONLY the reformatted sub-bullet, nothing else.
"""