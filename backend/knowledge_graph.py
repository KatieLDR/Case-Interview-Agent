import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

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
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def _run(query: str, **params) -> list[dict]:
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


def get_concepts(case_type: str) -> list[str]:
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
    Change log: 2026-03-31 — updated to use HAS_CHILD tree traversal.
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
    """Change log: 2026-03-30 — initial implementation."""
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


def get_concept_parent(concept_name: str, framework: str) -> str | None:
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
    """Change log: 2026-03-31 — initial implementation."""
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
        {"concept": r["concept"], "parent": r["parent"], "depth": r["depth"]}
        for r in rows
    ]


def get_also_drives(concept_name: str, framework: str) -> list[str]:
    """Change log: 2026-03-31 — added for ALSO_DRIVES relationship."""
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


def get_concept_full_data(concept_name: str, framework: str) -> dict:
    """
    Return all concept data in a single KG query.
    Change log: 2026-03-31 — added to batch KG queries.
    """
    rows = _run(
        """
        MATCH (c:Concept {name: $concept_name, framework: $framework})
        OPTIONAL MATCH (parent)-[:HAS_CHILD]->(c)
        OPTIONAL MATCH (parent)-[:HAS_CHILD]->(sibling:Concept {framework: $framework})
        WHERE sibling.name <> $concept_name
        OPTIONAL MATCH (c)-[:HAS_CHILD]->(child:Concept)
        RETURN
            coalesce(c.description, '')    AS description,
            coalesce(c.branch, false)      AS is_branch,
            parent.name                    AS parent,
            collect(DISTINCT sibling.name) AS siblings,
            collect(DISTINCT child.name)   AS children
        LIMIT 1
        """,
        concept_name=concept_name,
        framework=framework,
    )
    if not rows:
        return {"description": "", "parent": None, "siblings": [], "children": [], "is_branch": False}
    r = rows[0]
    return {
        "description": r["description"] or "",
        "parent":      r["parent"],
        "siblings":    r["siblings"] or [],
        "children":    r["children"] or [],
        "is_branch":   r["is_branch"] or False,
    }


def get_concept_description(concept_name: str, framework: str) -> str:
    rows = _run(
        """
        MATCH (c:Concept {name: $concept_name, framework: $framework})
        RETURN coalesce(c.description, '') AS description
        LIMIT 1
        """,
        concept_name=concept_name,
        framework=framework,
    )
    return rows[0]["description"] if rows else ""


def get_concept_siblings(concept_name: str, framework: str) -> list[str]:
    rows = _run(
        """
        MATCH (parent)-[:HAS_CHILD]->(c:Concept {name: $concept_name, framework: $framework})
        MATCH (parent)-[:HAS_CHILD]->(sibling:Concept {framework: $framework})
        WHERE sibling.name <> $concept_name
        RETURN sibling.name AS concept
        """,
        concept_name=concept_name,
        framework=framework,
    )
    return [r["concept"] for r in rows]


def is_branch_node(concept_name: str, framework: str) -> bool:
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


# Change log: 2026-04-02 — added get_framework_for_concept()
def get_framework_for_concept(concept_name: str) -> list[dict]:
    """
    Return all frameworks containing the given concept, across the entire KG.

    Used when a concept is not found in the current active framework —
    allows the citation layer to show a cross-framework note with correct
    provenance instead of fabricating a source or showing a generic unverified note.

    Returns list of dicts: [{"framework": str, "case_type": str}]
    Returns empty list if concept not found anywhere in KG.

    Example:
        get_framework_for_concept("Market Share")
        -> [{"framework": "Economic Feasibility", "case_type": "Market Entry"}]

        get_framework_for_concept("unknown concept")
        -> []
    """
    rows = _run(
        """
        MATCH (f:Framework)-[:HAS_CONCEPT]->(c:Concept {name: $concept_name})
        MATCH (ct:CaseType)-[:USES_FRAMEWORK]->(f)
        RETURN f.name  AS framework,
               ct.name AS case_type
        """,
        concept_name=concept_name,
    )
    return [
        {"framework": r["framework"], "case_type": r["case_type"]}
        for r in rows
    ]


if __name__ == "__main__":
    print("=== get_ordered_concepts(Expanded Profit Formula) ===")
    print(get_ordered_concepts("Expanded Profit Formula"))

    print("\n=== get_framework_for_concept(Market Share) ===")
    print(get_framework_for_concept("Market Share"))

    print("\n=== get_framework_for_concept(unknown) ===")
    print(get_framework_for_concept("unknown concept"))

    close()