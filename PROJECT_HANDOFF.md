# Case Interview Agent — Project Handoff

## 🗂️ Project Overview
An experimental multi-agent case interview system designed to research **user sense of control** over AI interactions. Users are assigned one of two agents (blind) and work through a BCG-style case interview. All behavioral events are logged to Firebase Firestore for analysis.

---

## 🏗️ Architecture

```
case-interview-agent/
│
├── .env                          ← API keys (never commit)
├── firebase_key.json             ← Firebase service account (never commit)
├── pyproject.toml                ← Poetry dependencies
│
├── frontend/
│   └── app.py                    ← Chainlit UI + agent selector
│
└── backend/
    ├── cases.py                  ← Static case bank (one per agent)
    ├── logger.py                 ← Firebase Firestore logging
    ├── concept_swap.py           ← Shared concept swap logic (all agents)
    ├── black_box_agent.py        ← Agent 1: reference answer generator
    ├── coach_agent.py            ← Agent 2: interactive BCG coach
    ├── explainable_agent.py      ← Agent 3: RAG + Neo4j KG (TODO)
    └── hitl_agent.py             ← Agent 4: Human-in-the-Loop (TODO)
```

---

## 🧱 Class Hierarchy

```
BlackBoxAgent                         ← Agent 1 (done)
CoachAgent                            ← Agent 2 (done)
ExplainableAgent(BlackBoxAgent)       ← Agent 3 (TODO)
HITLAgent(BlackBoxAgent)              ← Agent 4 (TODO)
```

---

## ✅ What's Built

### Agent 1 — BlackBoxAgent
- Presents Ghost Restaurant case automatically on session start
- Generates structured reference answer (framework + hypotheses + sample answer + mistakes)
- Detects redo/dissatisfaction intent via Gemini classifier → regenerates answer
- Injects Concept Swap (Variable Costs before Market Size) until user detects it
- Logs all events to Firestore

### Agent 2 — CoachAgent
- Presents Burger Store Taipei case automatically on session start
- BCG-style interactive coaching — one question at a time
- Detects redo intent → restarts coaching from beginning
- Logs all events to Firestore

### Shared Infrastructure
- **`logger.py`** — Firestore logging for all agents
- **`concept_swap.py`** — per-agent configurable swap injection + detection
- **`cases.py`** — static case bank
- **`frontend/app.py`** — Chainlit UI with agent selector buttons (Agent 1 / Agent 2)

---

## 📊 Firestore Data Structure

### Session Document (`sessions/{session_id}`)
```
user_id:                  "user-a1b2c3d4"
agent_type:               "black_box" | "coach"
started_at:               timestamp
ended_at:                 timestamp
original_answer:          string (first structured answer generated)
current_answer:           string (latest structured answer)
concept_swap_presented:   boolean
concept_swap_detected:    boolean
count_user_messages:      integer
count_agent_responses:    integer
count_interruptions:      integer
count_memory_overrides:   integer
count_answer_updates:     integer
```

### Events Subcollection (`sessions/{session_id}/events/{event_id}`)
```
type:       "user_message" | "agent_response" | "interruption" |
            "memory_override" | "concept_swap_presented" | "concept_swap_detected"
timestamp:  timestamp
message:    string (for user_message)
response:   string (for agent_response)
context:    string (for interruption)
```

---

## 🔑 Environment Variables (.env)

```
GEMINI_API_KEY=your_gemini_api_key
FIREBASE_KEY_PATH=firebase_key.json
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

---

## 📦 Dependencies (pyproject.toml)

```toml
[tool.poetry.dependencies]
python = ">=3.10,<4.0.0"
chainlit = ">=2.6.3"
google-genai = "*"
firebase-admin = ">=7.2.0,<8.0.0"
python-dotenv = ">=1.2.2,<2.0.0"
neo4j = "*"
```

---

## 🚀 How to Run

```bash
# From project root
poetry install
poetry run chainlit run frontend/app.py -w
```

App opens at: `http://localhost:8000`

---

## 🤖 Gemini Models Used

| Purpose | Model |
|---|---|
| Main agent (answers) | `gemini-2.5-flash` |
| Classifiers (intent, swap, answer) | `gemini-2.5-flash-lite` |

---

## 🔀 Concept Swap Config (concept_swap.py)

| Agent | Wrong Concept Injected |
|---|---|
| `black_box` | Variable Costs per unit before Market Size |
| `explainable` | Price-elasticity pricing instead of Production/Distribution Challenges |
| `hitl` | Population segmentation by Income instead of Internal vs External risks |

Detection uses **Direction B+C** (contextual deviation + framework extraction) — **C is pending KG completion**.

---

## 🗃️ Neo4j Knowledge Graph (TODO)

### Status: ⚠️ Connection issue — credentials need to be reset on AuraDB console

### Next Steps:
1. Fix Neo4j AuraDB credentials (delete + recreate instance if needed)
2. Design KG schema:
   - Nodes: `CaseType`, `Framework`, `Concept`
   - Relationships: `USES_FRAMEWORK`, `HAS_CONCEPT`, `PRECEDES`
3. Populate KG with MECE framework data from the doc's Knowledge Taxonomy table
4. Build `backend/knowledge_graph.py` — query interface
5. Build `backend/explainable_agent.py` — RAG using KG
6. Complete Concept Swap Classifier C using KG's correct framework order

---

## 🧪 Evaluation Metrics (from design doc)

| Metric | Status | Field in Firestore |
|---|---|---|
| Overwrite Frequency | ✅ Done | `count_memory_overrides` |
| Interruption Frequency | ✅ Done | `count_interruptions` |
| Error Detection Rate | ⚠️ Partial (B only, C pending KG) | `concept_swap_detected` |
| Framework Performance | ❌ TODO (needs KG + RocketBlocks) | — |
| Agency Delta | ❌ TODO (needs KG) | — |
| Control Perception | ❌ Outside scope (post-session survey) | — |

---

## 📋 Remaining TODO List

- [ ] Fix Neo4j AuraDB credentials
- [ ] Design and populate Knowledge Graph schema
- [ ] Build `backend/knowledge_graph.py`
- [ ] Build `backend/explainable_agent.py` (RAG + KG + Concept Swap C)
- [ ] Build `backend/hitl_agent.py` (HITL + Concept Swap C)
- [ ] Complete Classifier C for swap detection using KG
- [ ] Deploy to Google Cloud Run
- [ ] Write Dockerfile for deployment
- [ ] Post-session survey integration (Control Perception metric)

---

## 📁 Git

- **Main branch:** `main`
- **Dev branch:** `feature/agent-dev`
- **Never commit:** `.env`, `firebase_key.json`

---

## 🔗 Key Resources

- [Chainlit Docs](https://docs.chainlit.io)
- [Google GenAI Python SDK](https://github.com/googleapis/python-genai)
- [Firebase Admin Python](https://firebase.google.com/docs/admin/setup)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [AuraDB Console](https://console.neo4j.io)