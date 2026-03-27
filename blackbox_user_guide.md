# Case Interview Assistant — User Guide

## What This Tool Does

This assistant helps you explore structured frameworks for business case interviews. Present it with a case and it will break down the problem into a logical hierarchy of buckets and sub-buckets — the kind of structure a top consulting candidate would use.

It is a **reference tool**, not an interviewer. It will not evaluate or score you.

---

## Getting Started

1. **Select Agent 1** from the welcome screen
2. Read the case presented to you carefully
3. When ready, ask for a framework — for example:
   - *"Give me a framework for this case"*
   - *"Walk me through a structured approach"*
   - *"What framework would you use here?"*

---

## What You Can Do

### Explore the framework
Ask follow-up questions about any part of the response:
- *"Can you go deeper on the Market Size bucket?"*
- *"What sub-buckets would you add under Production Challenges?"*
- *"Why does this framework fit a Market Entry case?"*

### Remove a concept
If you disagree with a bucket or concept:
- *"Remove Profit per Unit from the framework"*
- *"Exclude Distribution Challenges"*
- *"Don't include that last bucket"*

The agent will briefly explain why it included the concept and ask you to confirm before removing it.

### Switch frameworks
If you think a different framework fits better:
- *"Use a profitability framework instead"*
- *"Try a market sizing approach"*
- *"Switch to an issue tree framework"*

The agent will explain its reasoning and ask you to confirm before switching.

### Request a fresh answer
If you want to start over with a different angle:
- *"Can we redo this?"*
- *"Try a completely different approach"*
- *"Start fresh"*

### End the session
Click **📊 Get Summary & End Session** at any time to receive a summary of your final framework and end the session.

---

## Tips for Best Results

**Be specific when removing concepts**
Say *"remove Profit per Unit"* rather than *"remove that last thing"* — the agent responds best to clear concept names.

**Confirm explicitly when asked**
When the agent asks *"do you still want to proceed?"*, reply with a clear yes or no. Vague responses may be interpreted as continuing the discussion.

**One change at a time**
The agent handles one framework change per message best. If you want to remove multiple concepts, do them one at a time.

**Switching frameworks works best with a name**
*"Use profitability framework"* works well. *"Use something else"* is too vague for the agent to know what to switch to.

---

## What to Expect

| You say | Agent does |
|---|---|
| "Give me a framework" | Presents full structured framework |
| "Why is X here?" | Explains the reasoning |
| "Remove X" | Pushes back briefly, asks to confirm |
| "Yes, remove it" | Removes and presents updated framework |
| "Use profitability framework" | Explains current choice, asks to confirm switch |
| "Yes switch it" | Switches framework using updated knowledge |
| "Redo this" | Generates a fresh answer from scratch |
| Summary button | Shows final framework + session changes |

---

## Known Limitations

- **Framework switching requires a known framework name** — the agent recognises Market Entry, Profitability, Pricing, Market Sizing, and Issue Tree frameworks. Other frameworks (e.g. Porter's Five Forces) will be handled using the agent's own knowledge without structured knowledge graph grounding.

- **The agent may occasionally push back on a valid removal** — if this happens, simply confirm again and it will honour your choice.

- **Session is not saved** — once you close or refresh the page, the session ends. Use the summary button before leaving to capture your final framework.

- **One session per page load** — to start a new session, refresh the page.

---

## Session ID

At the start of each session you are given a **Session ID** (e.g. `f3511929-...`). Keep this for reference — it links your session to any follow-up support or research tracking.
