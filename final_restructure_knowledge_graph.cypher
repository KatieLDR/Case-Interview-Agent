// ── final_restructure_knowledge_graph.cypher ──────────────────────────────
// Run this in your AuraDB Browser after seed_knowledge_graph.cypher
// and add_framework_descriptions.cypher.
//
// What this script does:
//   1. Deletes PRECEDES and HAS_CHILD relationships (clean slate)
//   2. Renames/removes old Profitability nodes (align to Victor Cheng)
//   3. Adds new concept nodes for Profitability, M&A, Capacity Change
//   4. Adds HAS_CONCEPT flat edges for all new nodes
//   5. Adds HAS_CHILD tree edges for all 7 frameworks
//   6. Adds ALSO_DRIVES edge (Units Sold → Variable Cost)
//   7. Adds CaseType + Framework nodes for M&A and Capacity Change
//   8. Adds framework descriptions for new frameworks
//
// Source authority per framework:
//   Expanded Profit Formula  → Victor Cheng (caseinterview.com PDF)
//   Economic Feasibility     → Original design doc taxonomy
//   Four-Pronged Strategy    → Original design doc taxonomy
//   Formulaic Breakdown      → Original design doc taxonomy
//   Customized Issue Trees   → Original design doc taxonomy
//   MA Fit Framework         → Victor Cheng (caseinterview.com PDF)
//   Capacity Change          → Victor Cheng (caseinterview.com PDF)
//
// Change log: 2026-03-31 — full KG restructure
//   - Replaced linear PRECEDES chain with HAS_CHILD tree structure
//   - Profitability rebuilt from Victor Cheng source (removed Value Chain
//     branch; Production/Distribution/Customer Pull were segmentation
//     techniques, not structural framework buckets)
//   - Renamed: Volume → Units Sold, Variable Cost per Unit → Cost per Unit
//   - Added ALSO_DRIVES relationship for shared Units Sold node
//   - Added M&A Fit Framework and Capacity Change Framework
// ══════════════════════════════════════════════════════════════════════════


// ══════════════════════════════════════════════════════════════════════════
// STEP 1 — Clean slate: delete PRECEDES and HAS_CHILD relationships
// ══════════════════════════════════════════════════════════════════════════

MATCH ()-[r:PRECEDES]->() DELETE r;
MATCH ()-[r:HAS_CHILD]->() DELETE r;
MATCH ()-[r:ALSO_DRIVES]->() DELETE r;


// ══════════════════════════════════════════════════════════════════════════
// STEP 2 — Remove old Profitability nodes that are renamed or deleted
//
// Removed:  Volume (→ renamed to Units Sold)
//           Variable Cost per Unit (→ renamed to Cost per Unit)
//           Production, Distribution, Customer Pull (value chain — removed)
//           Variable Costs branch node (→ renamed to Variable Cost)
//           Value Chain branch node (removed entirely)
// ══════════════════════════════════════════════════════════════════════════

MATCH (c:Concept {name: "Volume",                framework: "Expanded Profit Formula"}) DETACH DELETE c;
MATCH (c:Concept {name: "Variable Cost per Unit",framework: "Expanded Profit Formula"}) DETACH DELETE c;
MATCH (c:Concept {name: "Production",            framework: "Expanded Profit Formula"}) DETACH DELETE c;
MATCH (c:Concept {name: "Distribution",          framework: "Expanded Profit Formula"}) DETACH DELETE c;
MATCH (c:Concept {name: "Customer Pull",         framework: "Expanded Profit Formula"}) DETACH DELETE c;
MATCH (c:Concept {name: "Variable Costs",        framework: "Expanded Profit Formula"}) DETACH DELETE c;
MATCH (c:Concept {name: "Value Chain",           framework: "Expanded Profit Formula"}) DETACH DELETE c;

// Remove old branch nodes added in previous restructure attempt
MATCH (c:Concept {name: "Revenue",        framework: "Expanded Profit Formula", branch: true}) DETACH DELETE c;
MATCH (c:Concept {name: "Costs",          framework: "Expanded Profit Formula", branch: true}) DETACH DELETE c;


// ══════════════════════════════════════════════════════════════════════════
// STEP 3 — Add/update Profitability concept nodes (Victor Cheng structure)
// ══════════════════════════════════════════════════════════════════════════

// Branch nodes
MERGE (:Concept {name: "Revenue",       framework: "Expanded Profit Formula", branch: true})
MERGE (:Concept {name: "Costs",         framework: "Expanded Profit Formula", branch: true})
MERGE (:Concept {name: "Variable Cost", framework: "Expanded Profit Formula", branch: true});

// Leaf nodes
MERGE (:Concept {name: "Price per Unit", framework: "Expanded Profit Formula", branch: false})
MERGE (:Concept {name: "Units Sold",     framework: "Expanded Profit Formula", branch: false})
MERGE (:Concept {name: "Cost per Unit",  framework: "Expanded Profit Formula", branch: false})
MERGE (:Concept {name: "Fixed Cost",     framework: "Expanded Profit Formula", branch: false});

// Note: Fixed Costs → Fixed Cost (singular, matching Victor Cheng)
// Delete old Fixed Costs node if it exists
MATCH (c:Concept {name: "Fixed Costs", framework: "Expanded Profit Formula"}) DETACH DELETE c;


// ══════════════════════════════════════════════════════════════════════════
// STEP 4 — Add new CaseType + Framework nodes for M&A and Capacity Change
// ══════════════════════════════════════════════════════════════════════════

MERGE (ct6:CaseType {name: "M&A"})
MERGE (ct7:CaseType {name: "Capacity Change"});

MERGE (f6:Framework {name: "MA Fit Framework"})
MERGE (f7:Framework {name: "Capacity Change Framework"});

// CaseType → Framework
MATCH (ct6:CaseType {name: "M&A"}),
      (f6:Framework  {name: "MA Fit Framework"})
MERGE (ct6)-[:USES_FRAMEWORK]->(f6);

MATCH (ct7:CaseType {name: "Capacity Change"}),
      (f7:Framework  {name: "Capacity Change Framework"})
MERGE (ct7)-[:USES_FRAMEWORK]->(f7);

// Framework descriptions
MATCH (f:Framework {name: "MA Fit Framework"})
SET f.description = "Evaluates strategic fit of a merger or acquisition across customers, products, company capabilities, and competitive landscape";

MATCH (f:Framework {name: "Capacity Change Framework"})
SET f.description = "Assesses whether to add or reduce capacity by analysing demand sustainability, industry supply dynamics, and cost of expansion";


// ══════════════════════════════════════════════════════════════════════════
// STEP 5 — Add concept nodes for M&A and Capacity Change (all leaf, flat)
// ══════════════════════════════════════════════════════════════════════════

// M&A Fit Framework
MERGE (:Concept {name: "Customer",          framework: "MA Fit Framework",          branch: false})
MERGE (:Concept {name: "Company",           framework: "MA Fit Framework",          branch: false})
MERGE (:Concept {name: "Product",           framework: "MA Fit Framework",          branch: false})
MERGE (:Concept {name: "Competition",       framework: "MA Fit Framework",          branch: false});

// Capacity Change Framework
MERGE (:Concept {name: "Demand",            framework: "Capacity Change Framework", branch: false})
MERGE (:Concept {name: "Supply",            framework: "Capacity Change Framework", branch: false})
MERGE (:Concept {name: "Cost of Expansion", framework: "Capacity Change Framework", branch: false});


// ══════════════════════════════════════════════════════════════════════════
// STEP 6 — HAS_CONCEPT flat edges (all concepts → their framework)
// Ensures get_concepts() and concept_belongs_to_framework() work correctly.
// ══════════════════════════════════════════════════════════════════════════

// Expanded Profit Formula — re-attach all nodes
MATCH (f:Framework {name: "Expanded Profit Formula"})
MATCH (c:Concept   {framework: "Expanded Profit Formula"})
MERGE (f)-[:HAS_CONCEPT]->(c);

// MA Fit Framework
MATCH (f:Framework {name: "MA Fit Framework"})
MATCH (c:Concept   {framework: "MA Fit Framework"})
MERGE (f)-[:HAS_CONCEPT]->(c);

// Capacity Change Framework
MATCH (f:Framework {name: "Capacity Change Framework"})
MATCH (c:Concept   {framework: "Capacity Change Framework"})
MERGE (f)-[:HAS_CONCEPT]->(c);


// ══════════════════════════════════════════════════════════════════════════
// STEP 7 — HAS_CHILD tree edges
// ══════════════════════════════════════════════════════════════════════════

// ────────────────────────────────────────────────────────────────────────
// Expanded Profit Formula (Victor Cheng)
//
//   Profit
//   ├── Revenue
//   │   ├── Price per Unit
//   │   └── Units Sold  ← also drives Variable Cost via ALSO_DRIVES
//   └── Costs
//       ├── Variable Cost
//       │   ├── Cost per Unit
//       │   └── Units Sold  ← shared node (ALSO_DRIVES, not HAS_CHILD here)
//       └── Fixed Cost
// ────────────────────────────────────────────────────────────────────────

// Framework root → top-level branches
MATCH (f:Framework {name: "Expanded Profit Formula"})
MATCH (c:Concept   {name: "Revenue", framework: "Expanded Profit Formula"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Expanded Profit Formula"})
MATCH (c:Concept   {name: "Costs",   framework: "Expanded Profit Formula"})
MERGE (f)-[:HAS_CHILD]->(c);

// Revenue → leaf concepts
MATCH (p:Concept {name: "Revenue",       framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Price per Unit",framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Revenue",   framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Units Sold",framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

// Costs → Variable Cost branch + Fixed Cost leaf
MATCH (p:Concept {name: "Costs",         framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Variable Cost", framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Costs",      framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Fixed Cost", framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

// Variable Cost → Cost per Unit
// Units Sold is NOT HAS_CHILD here — it is linked via ALSO_DRIVES (Step 8)
MATCH (p:Concept {name: "Variable Cost", framework: "Expanded Profit Formula"})
MATCH (c:Concept {name: "Cost per Unit", framework: "Expanded Profit Formula"})
MERGE (p)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Economic Feasibility (design doc)
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

MATCH (f:Framework {name: "Economic Feasibility"})
MATCH (c:Concept   {name: "Market Potential", framework: "Economic Feasibility"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Economic Feasibility"})
MATCH (c:Concept   {name: "Implementation",   framework: "Economic Feasibility"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Market Potential", framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Market Size",      framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Market Potential", framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Market Share",     framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Market Potential", framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Profit per Unit",  framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Implementation",         framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Production Challenges",  framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Implementation",          framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Distribution Challenges", framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Implementation",        framework: "Economic Feasibility"})
MATCH (c:Concept {name: "Marketing Challenges",  framework: "Economic Feasibility"})
MERGE (p)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Four-Pronged Strategy (design doc — flat, all parallel pillars)
// ────────────────────────────────────────────────────────────────────────

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Cost-based Pricing", framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Competitor-based Pricing", framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Value-to-Customer Pricing", framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {name: "Price-Elasticity Pricing", framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Formulaic Breakdown (design doc)
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

MATCH (f:Framework {name: "Formulaic Breakdown"})
MATCH (c:Concept   {name: "Supply-side Constraints", framework: "Formulaic Breakdown"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Formulaic Breakdown"})
MATCH (c:Concept   {name: "Demand-side",             framework: "Formulaic Breakdown"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Demand-side",            framework: "Formulaic Breakdown"})
MATCH (c:Concept {name: "Demand-side Estimation", framework: "Formulaic Breakdown"})
MERGE (p)-[:HAS_CHILD]->(c);

MATCH (p:Concept {name: "Demand-side",             framework: "Formulaic Breakdown"})
MATCH (c:Concept {name: "Population Segmentation", framework: "Formulaic Breakdown"})
MERGE (p)-[:HAS_CHILD]->(c);

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
// Customized Issue Trees (design doc — flat, all parallel splits)
// ────────────────────────────────────────────────────────────────────────

MATCH (f:Framework {name: "Customized Issue Trees"})
MATCH (c:Concept   {name: "Internal Factors", framework: "Customized Issue Trees"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Customized Issue Trees"})
MATCH (c:Concept   {name: "External Factors", framework: "Customized Issue Trees"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Customized Issue Trees"})
MATCH (c:Concept   {name: "Mathematical Parameter Division", framework: "Customized Issue Trees"})
MERGE (f)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// MA Fit Framework (Victor Cheng — flat, all parallel pillars)
// ────────────────────────────────────────────────────────────────────────

MATCH (f:Framework {name: "MA Fit Framework"})
MATCH (c:Concept   {name: "Customer",    framework: "MA Fit Framework"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "MA Fit Framework"})
MATCH (c:Concept   {name: "Company",     framework: "MA Fit Framework"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "MA Fit Framework"})
MATCH (c:Concept   {name: "Product",     framework: "MA Fit Framework"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "MA Fit Framework"})
MATCH (c:Concept   {name: "Competition", framework: "MA Fit Framework"})
MERGE (f)-[:HAS_CHILD]->(c);

// ────────────────────────────────────────────────────────────────────────
// Capacity Change Framework (Victor Cheng — flat, all parallel pillars)
// ────────────────────────────────────────────────────────────────────────

MATCH (f:Framework {name: "Capacity Change Framework"})
MATCH (c:Concept   {name: "Demand",            framework: "Capacity Change Framework"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Capacity Change Framework"})
MATCH (c:Concept   {name: "Supply",            framework: "Capacity Change Framework"})
MERGE (f)-[:HAS_CHILD]->(c);

MATCH (f:Framework {name: "Capacity Change Framework"})
MATCH (c:Concept   {name: "Cost of Expansion", framework: "Capacity Change Framework"})
MERGE (f)-[:HAS_CHILD]->(c);


// ══════════════════════════════════════════════════════════════════════════
// STEP 8 — ALSO_DRIVES relationship
// Units Sold drives both Revenue (via HAS_CHILD) and Variable Cost (via this).
// This reflects Victor Cheng: the same volume number appears on both sides
// of the profit formula. HAS_CHILD tree stays clean (Units Sold has one
// structural parent: Revenue). ALSO_DRIVES captures the cross-branch
// analytical dependency without polluting the tree.
//
// Change log: 2026-03-31 — new relationship type, replaces shared-node
//   approach which would have broken get_concept_parent() and tree traversal.
// ══════════════════════════════════════════════════════════════════════════

MATCH (units:Concept  {name: "Units Sold",     framework: "Expanded Profit Formula"})
MATCH (varcost:Concept{name: "Variable Cost",  framework: "Expanded Profit Formula"})
MERGE (units)-[:ALSO_DRIVES]->(varcost);


// ══════════════════════════════════════════════════════════════════════════
// STEP 9 — VERIFY (run these separately in AuraDB browser)
// ══════════════════════════════════════════════════════════════════════════

// 1. Confirm PRECEDES is gone:
// MATCH ()-[r:PRECEDES]->() RETURN count(r) AS remaining;
// Expected: 0

// 2. Check all HAS_CHILD direct edges:
// MATCH (parent)-[:HAS_CHILD]->(child)
// RETURN parent.name AS parent, child.name AS child
// ORDER BY parent.name, child.name;

// 3. Check ALSO_DRIVES:
// MATCH (a)-[:ALSO_DRIVES]->(b) RETURN a.name, b.name;
// Expected: Units Sold → Variable Cost

// 4. Check full Profitability tree:
// MATCH (f:Framework {name: "Expanded Profit Formula"})-[:HAS_CHILD]->(branch)
// OPTIONAL MATCH (branch)-[:HAS_CHILD]->(leaf)
// RETURN branch.name AS branch, collect(leaf.name) AS direct_children
// ORDER BY branch.name;

// 5. Check new frameworks exist:
// MATCH (ct:CaseType)-[:USES_FRAMEWORK]->(f:Framework)
// RETURN ct.name, f.name ORDER BY ct.name;
// Expected: 7 rows (5 original + M&A + Capacity Change)
