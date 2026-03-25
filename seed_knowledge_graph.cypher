// ── seed_knowledge_graph.cypher ────────────────────────────────────────────
// Run this in your AuraDB Browser (console.neo4j.io → Open → Query)
// Safe to re-run: MERGE prevents duplicate nodes/relationships

// ══════════════════════════════════════════════════════════════════════════
// 1. CASE TYPES
// ══════════════════════════════════════════════════════════════════════════
MERGE (ct1:CaseType {name: "Market Entry"})
MERGE (ct2:CaseType {name: "Profitability"})
MERGE (ct3:CaseType {name: "Pricing"})
MERGE (ct4:CaseType {name: "Guesstimates"})
MERGE (ct5:CaseType {name: "Unconventional"});

// ══════════════════════════════════════════════════════════════════════════
// 2. FRAMEWORKS
// ══════════════════════════════════════════════════════════════════════════
MERGE (f1:Framework {name: "Economic Feasibility"})
MERGE (f2:Framework {name: "Expanded Profit Formula"})
MERGE (f3:Framework {name: "Four-Pronged Strategy"})
MERGE (f4:Framework {name: "Formulaic Breakdown"})
MERGE (f5:Framework {name: "Customized Issue Trees"});

// ══════════════════════════════════════════════════════════════════════════
// 3. CASETYPE → FRAMEWORK
// ══════════════════════════════════════════════════════════════════════════
MATCH (ct1:CaseType {name: "Market Entry"}),    (f1:Framework {name: "Economic Feasibility"})
MERGE (ct1)-[:USES_FRAMEWORK]->(f1);

MATCH (ct2:CaseType {name: "Profitability"}),   (f2:Framework {name: "Expanded Profit Formula"})
MERGE (ct2)-[:USES_FRAMEWORK]->(f2);

MATCH (ct3:CaseType {name: "Pricing"}),         (f3:Framework {name: "Four-Pronged Strategy"})
MERGE (ct3)-[:USES_FRAMEWORK]->(f3);

MATCH (ct4:CaseType {name: "Guesstimates"}),    (f4:Framework {name: "Formulaic Breakdown"})
MERGE (ct4)-[:USES_FRAMEWORK]->(f4);

MATCH (ct5:CaseType {name: "Unconventional"}),  (f5:Framework {name: "Customized Issue Trees"})
MERGE (ct5)-[:USES_FRAMEWORK]->(f5);

// ══════════════════════════════════════════════════════════════════════════
// 4. CONCEPTS
// ══════════════════════════════════════════════════════════════════════════

// ── Market Entry ──
MERGE (c_ms:Concept   {name: "Market Size",                 framework: "Economic Feasibility"})
MERGE (c_msh:Concept  {name: "Market Share",                framework: "Economic Feasibility"})
MERGE (c_ppu:Concept  {name: "Profit per Unit",             framework: "Economic Feasibility"})
MERGE (c_prod:Concept {name: "Production Challenges",       framework: "Economic Feasibility"})
MERGE (c_dist:Concept {name: "Distribution Challenges",     framework: "Economic Feasibility"})
MERGE (c_mkt:Concept  {name: "Marketing Challenges",        framework: "Economic Feasibility"});

// ── Profitability ──
MERGE (c_vol:Concept  {name: "Volume",                      framework: "Expanded Profit Formula"})
MERGE (c_price:Concept{name: "Price per Unit",              framework: "Expanded Profit Formula"})
MERGE (c_vc:Concept   {name: "Variable Cost per Unit",      framework: "Expanded Profit Formula"})
MERGE (c_fc:Concept   {name: "Fixed Costs",                 framework: "Expanded Profit Formula"})
MERGE (c_vcp:Concept  {name: "Production",                  framework: "Expanded Profit Formula"})
MERGE (c_vcd:Concept  {name: "Distribution",                framework: "Expanded Profit Formula"})
MERGE (c_cp:Concept   {name: "Customer Pull",               framework: "Expanded Profit Formula"});

// ── Pricing ──
MERGE (c_cb:Concept   {name: "Cost-based Pricing",          framework: "Four-Pronged Strategy"})
MERGE (c_compb:Concept{name: "Competitor-based Pricing",    framework: "Four-Pronged Strategy"})
MERGE (c_vtc:Concept  {name: "Value-to-Customer Pricing",   framework: "Four-Pronged Strategy"})
MERGE (c_pe:Concept   {name: "Price-Elasticity Pricing",    framework: "Four-Pronged Strategy"});

// ── Guesstimates ──
MERGE (c_ss:Concept   {name: "Supply-side Constraints",     framework: "Formulaic Breakdown"})
MERGE (c_de:Concept   {name: "Demand-side Estimation",      framework: "Formulaic Breakdown"})
MERGE (c_psa:Concept  {name: "Population Segmentation Age", framework: "Formulaic Breakdown"})
MERGE (c_psg:Concept  {name: "Population Segmentation Gender", framework: "Formulaic Breakdown"})
MERGE (c_psi:Concept  {name: "Population Segmentation Income", framework: "Formulaic Breakdown"});

// ── Unconventional ──
MERGE (c_if:Concept   {name: "Internal Factors",            framework: "Customized Issue Trees"})
MERGE (c_ef:Concept   {name: "External Factors",            framework: "Customized Issue Trees"})
MERGE (c_mpd:Concept  {name: "Mathematical Parameter Division", framework: "Customized Issue Trees"});

// ══════════════════════════════════════════════════════════════════════════
// 5. FRAMEWORK → CONCEPT  (HAS_CONCEPT)
// ══════════════════════════════════════════════════════════════════════════

// ── Market Entry ──
MATCH (f:Framework {name: "Economic Feasibility"})
MATCH (c:Concept   {framework: "Economic Feasibility"})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ── Profitability ──
MATCH (f:Framework {name: "Expanded Profit Formula"})
MATCH (c:Concept   {framework: "Expanded Profit Formula"})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ── Pricing ──
MATCH (f:Framework {name: "Four-Pronged Strategy"})
MATCH (c:Concept   {framework: "Four-Pronged Strategy"})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ── Guesstimates ──
MATCH (f:Framework {name: "Formulaic Breakdown"})
MATCH (c:Concept   {framework: "Formulaic Breakdown"})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ── Unconventional ──
MATCH (f:Framework {name: "Customized Issue Trees"})
MATCH (c:Concept   {framework: "Customized Issue Trees"})
MERGE (f)-[:HAS_CONCEPT]->(c);

// ══════════════════════════════════════════════════════════════════════════
// 6. CONCEPT → CONCEPT  (PRECEDES — correct ordering within each framework)
// ══════════════════════════════════════════════════════════════════════════

// ── Market Entry (must analyse Market Size before anything else) ──
MATCH (a:Concept {name: "Market Size"}),          (b:Concept {name: "Market Share"})          MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Market Share"}),         (b:Concept {name: "Profit per Unit"})        MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Profit per Unit"}),      (b:Concept {name: "Production Challenges"})  MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Production Challenges"}),(b:Concept {name: "Distribution Challenges"})MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Distribution Challenges"}),(b:Concept {name: "Marketing Challenges"}) MERGE (a)-[:PRECEDES]->(b);

// ── Profitability ──
MATCH (a:Concept {name: "Volume"}),               (b:Concept {name: "Price per Unit"})         MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Price per Unit"}),       (b:Concept {name: "Variable Cost per Unit"}) MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Variable Cost per Unit"}),(b:Concept {name: "Fixed Costs"})            MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Fixed Costs"}),          (b:Concept {name: "Production"})             MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Production"}),           (b:Concept {name: "Distribution"})           MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Distribution"}),         (b:Concept {name: "Customer Pull"})          MERGE (a)-[:PRECEDES]->(b);

// ── Pricing ──
MATCH (a:Concept {name: "Cost-based Pricing"}),      (b:Concept {name: "Competitor-based Pricing"})  MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Competitor-based Pricing"}),(b:Concept {name: "Value-to-Customer Pricing"}) MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Value-to-Customer Pricing"}),(b:Concept {name: "Price-Elasticity Pricing"}) MERGE (a)-[:PRECEDES]->(b);

// ── Guesstimates ──
MATCH (a:Concept {name: "Supply-side Constraints"}),    (b:Concept {name: "Demand-side Estimation"})         MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Demand-side Estimation"}),     (b:Concept {name: "Population Segmentation Age"})    MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Population Segmentation Age"}),(b:Concept {name: "Population Segmentation Gender"}) MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "Population Segmentation Gender"}),(b:Concept {name: "Population Segmentation Income"}) MERGE (a)-[:PRECEDES]->(b);

// ── Unconventional ──
MATCH (a:Concept {name: "Internal Factors"}), (b:Concept {name: "External Factors"})              MERGE (a)-[:PRECEDES]->(b);
MATCH (a:Concept {name: "External Factors"}), (b:Concept {name: "Mathematical Parameter Division"}) MERGE (a)-[:PRECEDES]->(b);

// ══════════════════════════════════════════════════════════════════════════
// 7. VERIFY — run these separately to check your data
// ══════════════════════════════════════════════════════════════════════════
// MATCH (ct:CaseType)-[:USES_FRAMEWORK]->(f:Framework)-[:HAS_CONCEPT]->(c:Concept)
// RETURN ct.name, f.name, c.name ORDER BY ct.name, c.name;

// MATCH (a:Concept)-[:PRECEDES]->(b:Concept)
// RETURN a.name AS from, b.name AS to ORDER BY from;
