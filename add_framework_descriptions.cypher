// ── add_framework_descriptions.cypher ────────────────────────────────────────
// Run this in your AuraDB Browser AFTER seed_knowledge_graph.cypher
// Adds a description field to each Framework node for LLM-based resolution.
// Safe to re-run: SET is idempotent.
// Change log: 2026-03-30 — added to support _resolve_framework() in explainable_agent.py

MATCH (f:Framework {name: "Economic Feasibility"})
SET f.description = "Assesses market size, competitive share, and implementation challenges for entering a new market";

MATCH (f:Framework {name: "Expanded Profit Formula"})
SET f.description = "Breaks down profitability into revenue drivers (volume, price) and cost drivers (variable, fixed)";

MATCH (f:Framework {name: "Four-Pronged Strategy"})
SET f.description = "Evaluates pricing decisions through cost, competition, customer value, and price elasticity";

MATCH (f:Framework {name: "Formulaic Breakdown"})
SET f.description = "Estimates unknown quantities using supply constraints, demand estimation, and population segmentation";

MATCH (f:Framework {name: "Customized Issue Trees"})
SET f.description = "Structures ambiguous problems using internal vs external factors or custom logical splits";

// VERIFY — run separately
// MATCH (f:Framework) RETURN f.name, f.description ORDER BY f.name;
