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
# Public query methods
# ══════════════════════════════════════════════════════════════════════════

def get_concepts(case_type: str) -> list[str]:
    """
    Return all concept names belonging to the given CaseType.

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
    Return True if the concept belongs to the given framework.

    Example:
        concept_belongs_to_framework("Variable Cost per Unit", "Economic Feasibility")
        → False   ← this is the swap detection check
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
    Return concepts in correct PRECEDES order for the given framework.
    Starts from the concept with no incoming PRECEDES edge (the root).

    Example:
        get_ordered_concepts("Economic Feasibility")
        → ["Market Size", "Market Share", "Profit per Unit",
           "Production Challenges", "Distribution Challenges", "Marketing Challenges"]
    """
    rows = _run(
        """
        MATCH (f:Framework {name: $framework})-[:HAS_CONCEPT]->(c:Concept)
        WHERE NOT ()-[:PRECEDES]->(c)  // root: no predecessor
        MATCH path = (c)-[:PRECEDES*0..]->(next:Concept)
        WHERE (next)<-[:HAS_CONCEPT]-(f)
        WITH next, length(path) AS depth
        ORDER BY depth
        RETURN next.name AS concept
        """,
        framework=framework,
    )
    return [r["concept"] for r in rows]


def comes_before(concept_a: str, concept_b: str) -> bool:
    """
    Return True if concept_a PRECEDES concept_b (directly or transitively).

    Example:
        comes_before("Market Size", "Variable Cost per Unit")
        → False  ← they are in different frameworks entirely

        comes_before("Market Size", "Market Share")
        → True
    """
    rows = _run(
        """
        MATCH (a:Concept {name: $concept_a})-[:PRECEDES*1..]->(b:Concept {name: $concept_b})
        RETURN count(*) AS found
        """,
        concept_a=concept_a,
        concept_b=concept_b,
    )
    return rows[0]["found"] > 0 if rows else False


def get_correct_first_concept(case_type: str) -> str | None:
    """
    Return the concept that should come FIRST for a given CaseType
    (i.e. the root concept — the one with no incoming PRECEDES edge
    within that framework).

    This is the primary check for Concept Swap detection:
    if the agent suggests anything other than this concept first,
    it may be injecting a swap.

    Example:
        get_correct_first_concept("Market Entry")
        → "Market Size"

        get_correct_first_concept("Profitability")
        → "Volume"
    """
    rows = _run(
        """
        MATCH (ct:CaseType {name: $case_type})
              -[:USES_FRAMEWORK]->(f:Framework)
              -[:HAS_CONCEPT]->(c:Concept)
        WHERE NOT ()-[:PRECEDES]->(c)
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


# Change log: 2026-03-30 — added for _resolve_framework() in explainable_agent.py


def get_all_frameworks() -> list[dict]:
    """
    Return all frameworks in the KG with their case type and description.
    Used by ExplainableAgent._resolve_framework() to resolve user's framework
    mention to a KG framework name via LLM matching.

    Returns list of dicts:
        [{"framework": str, "case_type": str, "description": str}, ...]

    Falls back gracefully if description field is not yet set on nodes.
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
# ── Quick smoke test (run directly: python -m backend.knowledge_graph) ─────
if __name__ == "__main__":
    print("=== get_concepts(Market Entry) ===")
    print(get_concepts("Market Entry"))

    print("\n=== get_ordered_concepts(Economic Feasibility) ===")
    print(get_ordered_concepts("Economic Feasibility"))

    print("\n=== get_correct_first_concept(Market Entry) ===")
    print(get_correct_first_concept("Market Entry"))

    print("\n=== concept_belongs_to_framework(Variable Cost per Unit, Economic Feasibility) ===")
    print(concept_belongs_to_framework("Variable Cost per Unit", "Economic Feasibility"))

    print("\n=== comes_before(Market Size, Market Share) ===")
    print(comes_before("Market Size", "Market Share"))

    print("\n=== comes_before(Variable Cost per Unit, Market Size) ===")
    print(comes_before("Variable Cost per Unit", "Market Size"))

    close()