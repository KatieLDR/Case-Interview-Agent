// ── restructure_knowledge_graph.cypher ────────────────────────────────────
// Run this in your AuraDB Browser AFTER seed_knowledge_graph.cypher
// and add_framework_descriptions.cypher.
//
// What this script does:
//   1. Deletes all PRECEDES relationships (replaced by HAS_CHILD tree edges)
//   2. Adds intermediate branch nodes (Revenue, Costs, etc.) as Concept nodes
//   3. Adds HAS_CONCEPT edges for new branch nodes
//   4. Adds HAS_CHILD edges to express the real consulting tree structure
//   5. Leaves all CaseType, Framework, HAS_CONCEPT (flat), USES_FRAMEWORK untouched
//
// Safe to re-run: MERGE prevents duplicate nodes/relationships.
//
// Change log: 2026-03-31 — replaced linear PRECEDES chain with HAS_CHILD tree
//   structure to reflect real consulting framework hierarchy. Motivation: the
//   flat PRECEDES chain incorrectly implied sequential analysis across parallel
//   branches (e.g. Revenue vs Cost side of Expanded Profit Formula). The new
//   schema supports Framework Alignment Accuracy scoring by preserving branch
//   membership and depth, and makes NL citations accurate (sub-bucket of X,
//   not "follows X").

// ══════════════════════════════════════════════════════════════════════════
// STEP 1 — Delete all PRECEDES relationships
// ══════════════════════════════════════════════════════════════════════════
MATCH ()-[r:PRECEDES]->()
DELETE r;

// ══════════════════════════════════════════════════════════════════════════
// STEP 2 — Add intermediate branch nodes
// These are parent nodes that group related concepts within a framework.
// They are added as Concept nodes so the tree is uniform.
// ══════════════════════════════════════════════════════════════════════════

// ── Expanded Profit Formula (Profitability) branch nodes ──────────────────
MERGE (:Concept {name: "Revenue",        framework: "Expanded Profit Formula", branch: true})
MERGE (:Concept {name: "Costs",          framework: "Expanded Profit Formula", branch: true})
MERGE (:Concept {name: "Variable Costs", framework: "Expanded Profit Formula", branch: true})
MERGE (:Concept {name: "Value Chain",    framework: "Expanded Profit Formula", branch: true});

// ── Economic Feasibility (Market Entry) branch nodes ──────────────────────
MERGE (:Concept {name: "Market Potential",  framework: "Economic Feasibility", branch: true})
MERGE (:Concept {name: "Implementation",    framework: "Economic Feasibility", branch: true});

// ── Formulaic Breakdown (Guesstimates) branch nodes ───────────────────────
MERGE (:Concept {name: "Demand-side",            framework: "Formulaic Breakdown", branch: true})
MERGE (:Concept {name: "Population Segmentation", framework: "Formulaic Breakdown", branch: true});

// ── Four-Pronged Strategy and Customized Issue Trees are flat ─────────────
// No intermediate branch nodes needed — all concepts are parallel pillars.

// ══════════════════════════════════════════════════════════════════════════
// STEP 3 — Add HAS_CONCEPT edges for new branch nodes
// ══════════════════════════════════════════════════════════════════════════

// ── Expanded Profit Formula ───────────────────────────────────────────────
MATCH (f:Framework {name: "Expanded Profit Formula"})
MATCH (c:Concept   {framework: "Expanded Profit Formula", branch: true})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ── Economic Feasibility ──────────────────────────────────────────────────
MATCH (f:Framework {name: "Economic Feasibility"})
MATCH (c:Concept   {framework: "Economic Feasibility", branch: true})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ── Formulaic Breakdown ───────────────────────────────────────────────────
MATCH (f:Framework {name: "Formulaic Breakdown"})
MATCH (c:Concept   {framework: "Formulaic Breakdown", branch: true})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ══════════════════════════════════════════════════════════════════════════
// STEP 4 — Add HAS_CHILD edges (tree structure)
// Format: MATCH parent, child → MERGE (parent)-[:HAS_CHILD]->(child)
// ══════════════════════════════════════════════════════════════════════════

// ────────────────────────────────────────────────────────────────────────
// Expanded Profit Formula
//
//   Profit
//   ├── Revenue
//   │   ├── Volume
//   │   └── Price per Unit
//   └── Costs
//       ├── Fixed Costs
//       └── Variable Costs
//           ├── Variable Cost per Unit
//           └── Value Chain
//               ├── Production
//               ├── Distribution
//               └── Customer Pull
// ────────────────────────────────────────────────────────────────────────

// Framework root → top-level branches
MATCH (f:Framework {name: "Expanded Profit Formula"})
MATCH (c:Concept   {name: "Revenue",  framework: "Expanded Profit Formula"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Expanded Profit Formula"})
MATCH (c:Concept   {name: "Costs",    framework: "Expanded Profit Formula"})
MERGE (f)-[:HAS_CHILD]->(c);

// Revenue → leaf concepts
MATCH (p:Concept {name: "Revenue",      framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Volume",       framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Revenue",        framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Price per Unit", framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

// Costs → Fixed Costs + Variable Costs branch
MATCH (p:Concept {name: "Costs",       framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Fixed Costs", framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Costs",          framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Variable Costs", framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

// Variable Costs → Variable Cost per Unit + Value Chain branch
MATCH (p:Concept {name: "Variable Costs",          framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Variable Cost per Unit",  framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Variable Costs", framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Value Chain",    framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

// Value Chain → leaf concepts
MATCH (p:Concept {name: "Value Chain",  framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Production",   framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Value Chain",  framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Distribution", framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Value Chain",    framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Customer Pull",  framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Economic Feasibility
//
//   Feasibility
//   ├── Market Potential
//   │   ├── Market Size
//   │   ├── Market Share
//   │   └── Profit per Unit
//   └── Implementation
//       ├── Production Challenges
//       ├── Distribution Challenges
//       └── Marketing Challenges
// ────────────────────────────────────────────────────────────────────────

// Framework root → top-level branches
MATCH (f:Framework {name: "Economic Feasibility"})
MATCH (c:Concept   {name: "Market Potential", framework: "Economic Feasibility"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Economic Feasibility"})
MATCH (c:Concept   {name: "Implementation",   framework: "Economic Feasibility"})
MERGE (f)-[:HAS_CHILD]->(c);

// Market Potential → leaf concepts
MATCH (p:Concept {name: "Market Potential", framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Market Size",      framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Market Potential", framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Market Share",     framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Market Potential", framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Profit per Unit",  framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

// Implementation → leaf concepts
MATCH (p:Concept {name: "Implementation",          framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Production Challenges",   framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Implementation",           framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Distribution Challenges",  framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Implementation",          framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Marketing Challenges",    framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Four-Pronged Strategy (Pricing)
// All four pillars are parallel — Framework root → leaf concepts directly
//
//   Pricing Strategy
//   ├── Cost-based Pricing
//   ├── Competitor-based Pricing
//   ├── Value-to-Customer Pricing
//   └── Price-Elasticity Pricing
// ────────────────────────────────────────────────────────────────────────

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Cost-based Pricing",        framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Competitor-based Pricing",  framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Value-to-Customer Pricing", framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Price-Elasticity Pricing",  framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Formulaic Breakdown (Guesstimates)
//
//   Estimation
//   ├── Supply-side Constraints
//   └── Demand-side
//       ├── Demand-side Estimation
//       └── Population Segmentation
//           ├── Population Segmentation Age
//           ├── Population Segmentation Gender
//           └── Population Segmentation Income
// ────────────────────────────────────────────────────────────────────────

// Framework root → top-level branches
MATCH (f:Framework {name: "Formulaic Breakdown"})
MATCH (c:Concept   {name: "Supply-side Constraints", framework: "Formulaic Breakdown"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Formulaic Breakdown"})
MATCH (c:Concept   {name: "Demand-side",             framework: "Formulaic Breakdown"})
MERGE (f)-[:HAS_CHILD]->(c);

// Demand-side → Demand-side Estimation + Population Segmentation branch
MATCH (p:Concept {name: "Demand-side",            framework: "Formulaic Breakdown"})
MATCH (c:Concept {name: "Demand-side Estimation", framework: "Formulaic Breakdown"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Demand-side",              framework: "Formulaic Breakdown"})
MATCH (c:Concept {name: "Population Segmentation",  framework: "Formulaic Breakdown"})
MERGE (p)-[:HAS_CHILD]->(c);

// Population Segmentation → leaf concepts
MATCH (p:Concept {name: "Population Segmentation",        framework: "Formulaic Breakdown"})
MATCH (c:Concept {name: "Population Segmentation Age",    framework: "Formulaic Breakdown"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Population Segmentation",           framework: "Formulaic Breakdown"})
MATCH (c:Concept {name: "Population Segmentation Gender",    framework: "Formulaic Breakdown"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Population Segmentation",           framework: "Formulaic Breakdown"})
MATCH (c:Concept {name: "Population Segmentation Income",    framework: "Formulaic Breakdown"})
MERGE (p)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Customized Issue Trees (Unconventional)
// All three splits are parallel — Framework root → leaf concepts directly
//
//   Issue Tree
//   ├── Internal Factors
//   ├── External Factors
//   └── Mathematical Parameter Division
// ────────────────────────────────────────────────────────────────────────

MATCH (f:Framework {name: "Customized Issue Trees"})
MATCH (c:Concept   {name: "Internal Factors",             framework: "Customized Issue Trees"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Customized Issue Trees"})
MATCH (c:Concept   {name: "External Factors",             framework: "Customized Issue Trees"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Customized Issue Trees"})
MATCH (c:Concept   {name: "Mathematical Parameter Division", framework: "Customized Issue Trees"})
MERGE (f)-[:HAS_CHILD]->(c);

// ══════════════════════════════════════════════════════════════════════════
// STEP 5 — VERIFY (run these separately in AuraDB browser to check)
// ══════════════════════════════════════════════════════════════════════════

// Check tree structure for one framework:
// MATCH (f:Framework {name: "Expanded Profit Formula"})-[:HAS_CHILD]->(branch)
// OPTIONAL MATCH (branch)-[:HAS_CHILD*1..]->(leaf)
// RETURN f.name, branch.name, collect(leaf.name) AS children
// ORDER BY branch.name;

// Check PRECEDES is fully gone:
// MATCH ()-[r:PRECEDES]->() RETURN count(r) AS remaining_precedes;
// Expected: 0

// Check all HAS_CHILD edges:
// MATCH (parent)-[:HAS_CHILD]->(child)
// RETURN parent.name AS parent, child.name AS child
// ORDER BY parent.name, child.name;
