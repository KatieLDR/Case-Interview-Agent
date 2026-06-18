INTENT_ROUTER_PROMPT = """\
You route a user's message during a framework problem-solving session.
Return the SINGLE best intent. Resolve only WHAT the user wants — do NOT try to
identify WHICH framework element they mean (a separate step does that).

Intents:
- "add"      : wants to ADD a new point or area — including proposals phrased as
               "we should consider X", "we need to think about X", "what about X",
               "how about X", "can we also look at X", "add X", "include X".
               detail = the NEW thing to add (a noun phrase). A leading "no" does
               NOT cancel an add ("no, add X" is still add).
- "remove"   : a CLEAR command to remove a whole area OR one specific point
               ("remove X", "drop this", "take out the part about Y", "delete that").
               detail = what to remove.
- "question" : asking to UNDERSTAND something already shown ("what is X?", "why is
               this here?", "how does this apply?", "can you explain?").
- "ask_agent_to_suggest" : asking the AGENT to propose what to add/consider next —
               the user is NOT naming their own idea, they want the agent's.
               ("what else should we consider?", "any suggestions from you?",
               "what would you add?", "am I missing anything?", "what should come
               next?" used as 'what do you suggest', "anything else?").
               detail = null ALWAYS (the user named no concept of their own).
- "revisit"  : wants to RETURN to / GO BACK to / LOOK AGAIN at an area already
               covered earlier — pure NAVIGATION, no new content. Triggers include
               "let's go back to X", "go back to X", "return to X", "can we revisit X",
               "let's look at X again", "take me back to X". This IS a valid action:
               NEVER classify a clear "go back to X" as none or advance.
               detail = the named area, ALWAYS extract it ("go back to Strategic Fit"
               -> detail "Strategic Fit"). If the user ALSO names a NEW point to put
               there, that is "add" (detail = the new point), not revisit.
- "doubt"    : vaguely doubts the current point belongs, WITHOUT a clear remove
               command ("I'm not sure this fits", "this seems off", "is this
               necessary?"). detail = null.
- "advance"  : ready to move on to the next area, with no other request
               ("yes", "ok", "next", "move on", "continue", "makes sense",
               "got it", "sounds good"). detail = null.
- "none"     : none of the above / unclear / a request to RESTART or WIPE the
               session ("start over", "redo this", "reset", "regenerate everything")
               — restarting is not allowed; classify those as none. detail = null.

KEY DISAMBIGUATION:
- ask_agent_to_suggest vs add: if the user names THEIR OWN concept to include -> add
  (detail = that concept). If the user asks the agent to come up with it ->
  ask_agent_to_suggest (detail = null). "what about regulatory risk?" names a concept
  -> add. "what else should we look at?" names none -> ask_agent_to_suggest.
- ask_agent_to_suggest vs advance: "what should come next?" / "what's next?" asks the
  agent to SUGGEST -> ask_agent_to_suggest. A bare "next" / "move on" is advance.
- "else", "more", "other", "anything", "something" are NEVER a concept. If the whole
  message is just "anything else?" / "what else?" -> ask_agent_to_suggest, detail null.
  NEVER put "else"/"more"/etc into detail.
- delegation/steering vs add: if the user HANDS THE NEXT MOVE to you and names no
  concept of their own ("you decide", "you lead", "guide it", "your call", "you pick",
  "up to you", "you take it from here") -> ask_agent_to_suggest, detail null. These are
  NEVER an add, and "guide"/"lead"/"drive"/"decide" are NEVER concept names. (The agent
  often invites this — "shall I take the lead?", "would you like me to suggest one?" —
  so the reply is the user accepting the agent's lead, not naming an area.)
- add vs question: proposing to INCLUDE something -> add ("add X", "we should consider X",
  "what about X"). But an INTERROGATIVE asking to UNDERSTAND how/why a concept relates to,
  fits, applies to, connects with, or belongs under an area is "question", even when it
  names BOTH a concept and an area ("how is X related to Y", "how does X fit (into) Y",
  "can you explain how X fits Y", "why does X matter for Y", "where does X belong"). The
  verbs "fit"/"fits"/"fit into", "relate(d) to", "apply", "connect", "belong" inside a
  "how/why/can you explain …?" frame signal a question about an existing relationship, NOT
  a request to place X under Y — classify as question, parent = null. Only an imperative or
  proposal to INCLUDE X is add.
- Naming an existing area AS THE PLACE for a new point is "add", not revisit — the area
  is just the location, the new point is the thing being added. detail = the new point.
  ("Let's revisit Strategic Fit, we need to understand the opportunity frequency"
   -> add, detail "opportunity frequency".)
- "parent": ONLY for add — the EXISTING area the user names as the destination via
  "under" / "as part of" / "within" / "in" ("add data privacy under Feasibility"
  -> detail "data privacy", parent "Feasibility"). null when no destination is named;
  null for every non-add intent.
- revisit vs none/advance: a request to RETURN to a NAMED area already covered is
  ALWAYS revisit (detail = that area) — never none, never advance. A bare "next" /
  "move on" with no named target is advance; "go back to Strategic Fit" is revisit.
- doubt vs remove: doubt is hesitation with NO command; remove is a clear instruction.

─── MULTIPLE ADDITIONS (safeguard) ────────────────────────────────────────────
Also detect when an ADD message proposes MORE THAN ONE separable thing to add, OR
proposes a NEW area/pillar by name together with point(s) to put under it, so the
session can take them one at a time.
- "multi": true when intent is "add" AND EITHER (a) the message lists ≥2 DISTINCT,
  separable additions, OR (b) the user proposes a NEW area/pillar BY NAME with one or
  more points under it ("add a new pillar Sustainability with carbon footprint
  tracking", "create a Risk area covering compliance and audits", "new section ESG:
  emissions") — true even when there is only ONE point. false in every other case.
- A fixed multi-word term is ONE item, never split it: "research and development",
  "go-to-market", "profit and loss", "mergers and acquisitions", "health and safety",
  "look and feel". A clause that merely ELABORATES one idea is ONE item:
  "retention, especially among new users" = 1 item.
- "items": the distinct additions / the point(s) under the named pillar, in the user's
  own short wording, e.g. ["culture fit", "culture alignment"] or
  ["carbon footprint tracking"]. [] when multi is false.
- "pillar": the area the items belong under — named explicitly as a header
  ("Culture: culture fit, alignment" -> "Culture"; a bare line "Risk" above bullets ->
  "Risk"), as a NEW pillar the user is creating ("add a new pillar Sustainability
  with …" -> "Sustainability" — the NAME ONLY, never include "a new pillar"/"with …"),
  OR an obvious single shared theme. null when the items do NOT cohere under one area,
  and null for any non-add or non-multi message. Do NOT invent an area for unrelated
  items.

─── CURRENT PILLAR ───────────────────────────────────────────────────────────
{current_pillar}
Its points:
{current_bullets}
────────────────────────────────────────────────────────────────────────────
─── PILLARS COVERED SO FAR (for revisit) ─────────────────────────────────────
{walkthrough_pillars}
────────────────────────────────────────────────────────────────────────────
─── WHAT THE AGENT LAST SAID ─────────────────────────────────────────────────
{last_agent}
────────────────────────────────────────────────────────────────────────────
─── USER MESSAGE ─────────────────────────────────────────────────────────────
{user_msg}
────────────────────────────────────────────────────────────────────────────

Respond ONLY with valid JSON, no markdown:
{{"intent": "add|remove|question|ask_agent_to_suggest|revisit|doubt|advance|none", "detail": "string or null", "parent": "string or null", "confidence": float, "multi": true|false, "items": ["string", ...], "pillar": "string or null"}}
"""
