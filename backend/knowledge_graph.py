import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ── Neo4j connection ───────────────────────────────────────────────────────
_driver = None

def _get_driver():
    global _driver
    if _driver is None:
        uri      = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        _driver  = GraphDatabase.driver(uri, auth=(username, password))
    return _driver


def close():
    """Call this on app shutdown to cleanly close the connection."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# ── Internal helper ────────────────────────────────────────────────────────
def _run(query: str, **params) -> list[dict]:
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


# ══════════════════════════════════════════════════════════════════════════
# Existing flat query methods — unchanged
# ══════════════════════════════════════════════════════════════════════════

def get_concepts(case_type: str) -> list[str]:
    """
    Return all concept names belonging to the given CaseType (flat).

    Example:
        get_concepts("Market Entry")
        → ["Market Size", "Market Share", "Profit per Unit", ...]
    """
    rows = _run(
        """
        MATCH (ct:CaseType {name: $case_type})
              -[:USES_FRAMEWORK]->(f:Framework)
              -[:HAS_CONCEPT]->(c:Concept)
        RETURN c.name AS concept
        """,
        case_type=case_type,
    )
    return [r["concept"] for r in rows]


def concept_belongs_to_framework(concept: str, framework: str) -> bool:
    """
    Return True if the concept belongs to the given framework (flat check).

    Example:
        concept_belongs_to_framework("Variable Cost per Unit", "Economic Feasibility")
        → False
    """
    rows = _run(
        """
        MATCH (f:Framework {name: $framework})-[:HAS_CONCEPT]->(c:Concept {name: $concept})
        RETURN count(c) AS found
        """,
        framework=framework,
        concept=concept,
    )
    return rows[0]["found"] > 0 if rows else False


def get_ordered_concepts(framework: str) -> list[str]:
    """
    Return leaf concept names for the given framework in depth-first order.

    Change log: 2026-03-31 — updated to use HAS_CHILD tree traversal instead
    of PRECEDES chain. Branch nodes (branch: true) are excluded — only leaf
    concepts are returned, preserving backward compatibility with all callers.

    Depth-first traversal ensures parent branches are visited before their
    children, which reflects natural consulting analysis order.

    Example:
        get_ordered_concepts("Expanded Profit Formula")
        → ["Volume", "Price per Unit", "Fixed Costs",
           "Variable Cost per Unit", "Production", "Distribution", "Customer Pull"]
    """
    rows = _run(
        """
        MATCH (f:Framework {name: $framework})
        CALL apoc.path.subgraphNodes(f, {
            relationshipFilter: 'HAS_CHILD>',
            minLevel: 1
        }) YIELD node AS c
        WHERE (c:Concept) AND NOT coalesce(c.branch, false)
        RETURN c.name AS concept
        """,
        framework=framework,
    )

    # Fallback: if APOC not available, use recursive Cypher
    if not rows:
        rows = _run(
            """
            MATCH (f:Framework {name: $framework})-[:HAS_CHILD*1..]->(c:Concept)
            WHERE NOT coalesce(c.branch, false)
            RETURN DISTINCT c.name AS concept
            """,
            framework=framework,
        )

    return [r["concept"] for r in rows]


def comes_before(concept_a: str, concept_b: str) -> bool:
    """
    Return True if concept_a is an ancestor of concept_b in the HAS_CHILD tree
    (directly or transitively within the same framework).

    Change log: 2026-03-31 — replaced PRECEDES traversal with HAS_CHILD ancestry
    check. Cross-framework comparisons always return False (correct behaviour).

    Example:
        comes_before("Variable Costs", "Production")   → True
        comes_before("Revenue", "Fixed Costs")         → False (different branch)
        comes_before("Market Size", "Volume")          → False (different frameworks)
    """
    rows = _run(
        """
        MATCH (a:Concept {name: $concept_a})-[:HAS_CHILD*1..]->(b:Concept {name: $concept_b})
        RETURN count(*) AS found
        """,
        concept_a=concept_a,
        concept_b=concept_b,
    )
    return rows[0]["found"] > 0 if rows else False


def get_correct_first_concept(case_type: str) -> str | None:
    """
    Return the name of the first top-level branch/concept for a given CaseType.
    For frameworks with a tree structure, this is the first HAS_CHILD target
    from the Framework node.

    Change log: 2026-03-31 — updated to use HAS_CHILD instead of PRECEDES root.

    Example:
        get_correct_first_concept("Profitability")
        → "Revenue"  (first branch under Expanded Profit Formula)
    """
    rows = _run(
        """
        MATCH (ct:CaseType {name: $case_type})
              -[:USES_FRAMEWORK]->(f:Framework)
              -[:HAS_CHILD]->(c:Concept)
        RETURN c.name AS concept
        LIMIT 1
        """,
        case_type=case_type,
    )
    return rows[0]["concept"] if rows else None


def get_framework_for_case(case_type: str) -> str | None:
    """
    Return the framework name for a given CaseType.

    Example:
        get_framework_for_case("Market Entry")
        → "Economic Feasibility"
    """
    rows = _run(
        """
        MATCH (ct:CaseType {name: $case_type})-[:USES_FRAMEWORK]->(f:Framework)
        RETURN f.name AS framework
        LIMIT 1
        """,
        case_type=case_type,
    )
    return rows[0]["framework"] if rows else None


def get_all_frameworks() -> list[dict]:
    """
    Return all frameworks in the KG with their case type and description.
    Used by ExplainableAgent._resolve_framework().

    Change log: 2026-03-30 — initial implementation.
    """
    rows = _run(
        """
        MATCH (ct:CaseType)-[:USES_FRAMEWORK]->(f:Framework)
        RETURN f.name        AS framework,
               ct.name       AS case_type,
               f.description AS description
        ORDER BY f.name
        """
    )
    return [
        {
            "framework":   r["framework"],
            "case_type":   r["case_type"],
            "description": r.get("description") or "",
        }
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════
# New tree-aware query methods
# Change log: 2026-03-31 — added to support HAS_CHILD tree structure
# ══════════════════════════════════════════════════════════════════════════

def get_concept_parent(concept_name: str, framework: str) -> str | None:
    """
    Return the parent of the given concept in the HAS_CHILD tree.

    Returns:
      - Parent Concept name if parent is a Concept node
      - Framework name string if concept is a direct child of the Framework root
      - None if concept is not found in the KG at all

    Example:
        get_concept_parent("Price per Unit", "Expanded Profit Formula")
        → "Revenue"

        get_concept_parent("Revenue", "Expanded Profit Formula")
        → "Expanded Profit Formula"  (direct child of Framework root)

        get_concept_parent("Net Promoter Score", "Expanded Profit Formula")
        → None  (not in KG)
    """
    # Check if parent is a Concept node
    rows = _run(
        """
        MATCH (parent:Concept {framework: $framework})-[:HAS_CHILD]->
              (c:Concept {name: $concept_name, framework: $framework})
        RETURN parent.name AS parent
        LIMIT 1
        """,
        concept_name=concept_name,
        framework=framework,
    )
    if rows:
        return rows[0]["parent"]

    # Check if parent is the Framework root itself
    rows_fw = _run(
        """
        MATCH (f:Framework {name: $framework})-[:HAS_CHILD]->
              (c:Concept {name: $concept_name})
        RETURN f.name AS parent
        LIMIT 1
        """,
        concept_name=concept_name,
        framework=framework,
    )
    return rows_fw[0]["parent"] if rows_fw else None


def get_concept_children(concept_name: str, framework: str) -> list[str]:
    """
    Return the direct children of a concept in the HAS_CHILD tree.
    Returns empty list for leaf concepts.

    Example:
        get_concept_children("Revenue", "Expanded Profit Formula")
        → ["Volume", "Price per Unit"]

        get_concept_children("Volume", "Expanded Profit Formula")
        → []
    """
    rows = _run(
        """
        MATCH (p:Concept {name: $concept_name, framework: $framework})
              -[:HAS_CHILD]->(c:Concept)
        RETURN c.name AS concept
        """,
        concept_name=concept_name,
        framework=framework,
    )
    return [r["concept"] for r in rows]


def get_concept_depth(concept_name: str, framework: str) -> int:
    """
    Return the depth of a concept in the HAS_CHILD tree.
    Direct children of Framework root = depth 1.
    Their children = depth 2. And so on.

    Used by Framework Alignment Accuracy scoring to compare structural
    depth between user's framework and the KG ground truth.

    Example:
        get_concept_depth("Revenue", "Expanded Profit Formula")     → 1
        get_concept_depth("Volume", "Expanded Profit Formula")      → 2
        get_concept_depth("Production", "Expanded Profit Formula")  → 4
    """
    rows = _run(
        """
        MATCH path = (f:Framework {name: $framework})-[:HAS_CHILD*1..]->(c:Concept {name: $concept_name})
        RETURN length(path) AS depth
        ORDER BY depth
        LIMIT 1
        """,
        concept_name=concept_name,
        framework=framework,
    )
    return rows[0]["depth"] if rows else 0


def get_framework_tree(framework: str) -> list[dict]:
    """
    Return the full tree structure of a framework as a flat list of
    {concept, parent, depth} dicts, ordered by depth then concept name.

    Used by Framework Alignment Accuracy scoring to compare the user's
    submitted framework against the KG ground truth tree.

    Example:
        get_framework_tree("Expanded Profit Formula")
        → [
            {"concept": "Revenue",               "parent": "Expanded Profit Formula", "depth": 1},
            {"concept": "Costs",                 "parent": "Expanded Profit Formula", "depth": 1},
            {"concept": "Fixed Costs",           "parent": "Costs",                   "depth": 2},
            {"concept": "Variable Costs",        "parent": "Costs",                   "depth": 2},
            ...
          ]

    Change log: 2026-03-31 — initial implementation
    """
    rows = _run(
        """
        MATCH path = (f:Framework {name: $framework})-[:HAS_CHILD*1..]->(c:Concept)
        WITH c, length(path) AS depth, path
        MATCH (parent)-[:HAS_CHILD]->(c)
        RETURN c.name      AS concept,
               parent.name AS parent,
               depth
        ORDER BY depth, c.name
        """,
        framework=framework,
    )
    return [
        {
            "concept": r["concept"],
            "parent":  r["parent"],
            "depth":   r["depth"],
        }
        for r in rows
    ]


def get_also_drives(concept_name: str, framework: str) -> list[str]:
    """
    Return concepts that the given concept ALSO_DRIVES (cross-branch dependencies).

    Currently only Units Sold → Variable Cost in Expanded Profit Formula.
    Added as a general method so future cross-branch drivers can be added
    to the KG without code changes.

    Change log: 2026-03-31 — added to support ALSO_DRIVES relationship
      introduced for Units Sold shared-driver pattern (Victor Cheng).

    Example:
        get_also_drives("Units Sold", "Expanded Profit Formula")
        → ["Variable Cost"]

        get_also_drives("Price per Unit", "Expanded Profit Formula")
        → []
    """
    rows = _run(
        """
        MATCH (a:Concept {name: $concept_name, framework: $framework})
              -[:ALSO_DRIVES]->(b:Concept)
        RETURN b.name AS concept
        """,
        concept_name=concept_name,
        framework=framework,
    )
    return [r["concept"] for r in rows]


def is_branch_node(concept_name: str, framework: str) -> bool:
    """
    Return True if the concept is an intermediate branch node (not a leaf).
    Branch nodes have branch: true property set in the KG.

    Used by rag_explainer.py to adjust citation language —
    branch nodes get "this bucket groups..." phrasing,
    leaf nodes get "this concept analyses..." phrasing.

    Example:
        is_branch_node("Revenue", "Expanded Profit Formula")        → True
        is_branch_node("Volume", "Expanded Profit Formula")         → False
    """
    rows = _run(
        """
        MATCH (c:Concept {name: $concept_name, framework: $framework})
        RETURN coalesce(c.branch, false) AS is_branch
        LIMIT 1
        """,
        concept_name=concept_name,
        framework=framework,
    )
    return rows[0]["is_branch"] if rows else False


# ── Quick smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== get_ordered_concepts(Expanded Profit Formula) ===")
    print(get_ordered_concepts("Expanded Profit Formula"))

    print("\n=== get_concept_parent(Price per Unit, Expanded Profit Formula) ===")
    print(get_concept_parent("Price per Unit", "Expanded Profit Formula"))
    # Expected: Revenue

    print("\n=== get_concept_parent(Revenue, Expanded Profit Formula) ===")
    print(get_concept_parent("Revenue", "Expanded Profit Formula"))
    # Expected: None (direct child of Framework root)

    print("\n=== get_concept_children(Costs, Expanded Profit Formula) ===")
    print(get_concept_children("Costs", "Expanded Profit Formula"))
    # Expected: ['Variable Cost', 'Fixed Cost']

    print("\n=== get_also_drives(Units Sold, Expanded Profit Formula) ===")
    print(get_also_drives("Units Sold", "Expanded Profit Formula"))
    # Expected: ['Variable Cost']

    print("\n=== get_framework_tree(Expanded Profit Formula) ===")
    for row in get_framework_tree("Expanded Profit Formula"):
        print(f"  depth={row['depth']}  parent={row['parent']}  →  {row['concept']}")

    print("\n=== get_all_frameworks() — should include M&A and Capacity Change ===")
    for f in get_all_frameworks():
        print(f"  {f['case_type']} → {f['framework']}")

    print("\n=== is_branch_node(Revenue, Expanded Profit Formula) ===")
    print(is_branch_node("Revenue", "Expanded Profit Formula"))
    # Expected: True

    print("\n=== is_branch_node(Units Sold, Expanded Profit Formula) ===")
    print(is_branch_node("Units Sold", "Expanded Profit Formula"))
    # Expected: False

    close()
