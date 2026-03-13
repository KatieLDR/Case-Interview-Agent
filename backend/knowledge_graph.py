import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from google import genai
from google.genai import types

CLASSIFIER_MODEL = "gemini-2.5-flash-lite"

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def verify_connection(self):
        with self.driver.session() as session:
            result = session.run("RETURN 1")
            print("[NEO4J] Connection successful!")
            return result
        
    def populate(self):
        with self.driver.session() as session:
            session.execute_write(self._create_graph)
            print("[NEO4J] Knowledge graph populated!")

    def get_frameworks_for_case_type(self, case_type: str) -> list:
        """Given a case type, return all frameworks."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:CaseType {name: $case_type})-[:USES_FRAMEWORK]->(f:Framework)
                RETURN f.name AS framework
            """, case_type=case_type)
            return [record["framework"] for record in result]

    def get_concepts_for_framework(self, framework: str) -> list:
        """Given a framework, return all concepts."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:Framework {name: $framework})-[:HAS_CONCEPT]->(c:Concept)
                RETURN c.name AS concept
            """, framework=framework)
            return [record["concept"] for record in result]

    def get_full_context(self, case_type: str) -> dict:
        """Return frameworks and concepts for a given case type."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:CaseType {name: $case_type})-[:USES_FRAMEWORK]->(f:Framework)
                OPTIONAL MATCH (f)-[:HAS_CONCEPT]->(con:Concept)
                RETURN f.name AS framework, collect(con.name) AS concepts
            """, case_type=case_type)
            return {
                record["framework"]: record["concepts"]
                for record in result
            }

    # The Case should be one or more types?
    def detect_case_type(self, case_text: str) -> str:
            """Use Gemini to classify the case type from the text."""
            prompt = f"""
    You are a case interview classifier with deep BCG consulting experience.

    Classify the following case into EXACTLY one of these types:
    - Profitability
    - Market Entry
    - Market Sizing
    - M&A
    - Operations

    Definitions:
    - Profitability: focused on improving profits, reducing costs, or hitting financial targets
    - Market Entry: evaluating whether and how to enter a new market or geography
    - Market Sizing: estimating the size or potential of a market
    - M&A: evaluating a merger, acquisition, or partnership
    - Operations: improving efficiency, processes, or supply chain

    Examples:
    - "Client wants to break even on a new store in year one" → Profitability
    - "Should our client enter the Japanese market?" → Market Entry
    - "How many coffee shops are there in Germany?" → Market Sizing
    - "Should our client acquire a competitor?" → M&A
    - "Client's factory output has dropped 20%" → Operations

    Respond ONLY with the case type name, nothing else.

    Case:
    {case_text}
    """
            try:
                response = client.models.generate_content(
                    model=CLASSIFIER_MODEL,
                    contents=prompt,
                )
                case_type = response.text.strip()
                print(f"[KG] Detected case type: {case_type}")
                return case_type
            except Exception as e:
                print(f"[KG] Case type detection failed: {e}")
                return "Profitability"  # safe default   
            
    def get_context_for_case(self, case_text: str) -> dict:
        """Main method agents will call — detects case type and returns full context."""
        case_type = self.detect_case_type(case_text)
        context = self.get_full_context(case_type)
        return {
            "case_type": case_type,
            "context": context
        }
    # ─────────────────Should Check the structure with Alex─────────────────────────────────────────────────────────────
    @staticmethod
    def _create_graph(tx):
        tx.run("""
        // Case Types
        MERGE (p:CaseType {name: "Profitability"})
        MERGE (me:CaseType {name: "Market Entry"})
        MERGE (ms:CaseType {name: "Market Sizing"})
        MERGE (ma:CaseType {name: "M&A"})
        MERGE (op:CaseType {name: "Operations"})

        // Frameworks
        MERGE (pt:Framework {name: "Profitability Tree"})
        MERGE (mece:Framework {name: "MECE Issue Tree"})
        MERGE (td:Framework {name: "Top-Down Market Sizing"})
        MERGE (bu:Framework {name: "Bottom-Up Market Sizing"})
        MERGE (attr:Framework {name: "Attractiveness & Capabilities"})
        MERGE (syn:Framework {name: "Synergies & Integration"})
        MERGE (vc:Framework {name: "Value Chain Analysis"})

        // Concepts — Profitability
        MERGE (rev:Concept {name: "Revenue"})
        MERGE (cost:Concept {name: "Cost"})
        MERGE (price:Concept {name: "Pricing"})
        MERGE (vol:Concept {name: "Volume"})
        MERGE (fixed:Concept {name: "Fixed Costs"})
        MERGE (var:Concept {name: "Variable Costs"})

        // Concepts — Market Entry
        MERGE (msize:Concept {name: "Market Size"})
        MERGE (mgrowth:Concept {name: "Market Growth"})
        MERGE (comp:Concept {name: "Competition"})
        MERGE (cap:Concept {name: "Capabilities"})

        // Concepts — Market Sizing
        MERGE (tam:Concept {name: "TAM"})
        MERGE (sam:Concept {name: "SAM"})
        MERGE (som:Concept {name: "SOM"})

        // Concepts — M&A
        MERGE (strat:Concept {name: "Strategic Rationale"})
        MERGE (syn2:Concept {name: "Synergies"})
        MERGE (risk:Concept {name: "Integration Risk"})

        // Concepts — Operations
        MERGE (bottle:Concept {name: "Bottleneck"})
        MERGE (bench:Concept {name: "Benchmarking"})

        // CaseType → Framework
        MERGE (p)-[:USES_FRAMEWORK]->(pt)
        MERGE (p)-[:USES_FRAMEWORK]->(mece)
        MERGE (me)-[:USES_FRAMEWORK]->(attr)
        MERGE (ms)-[:USES_FRAMEWORK]->(td)
        MERGE (ms)-[:USES_FRAMEWORK]->(bu)
        MERGE (ma)-[:USES_FRAMEWORK]->(syn)
        MERGE (op)-[:USES_FRAMEWORK]->(vc)

        // Framework → Concept
        MERGE (pt)-[:HAS_CONCEPT]->(rev)
        MERGE (pt)-[:HAS_CONCEPT]->(cost)
        MERGE (rev)-[:HAS_CONCEPT]->(price)
        MERGE (rev)-[:HAS_CONCEPT]->(vol)
        MERGE (cost)-[:HAS_CONCEPT]->(fixed)
        MERGE (cost)-[:HAS_CONCEPT]->(var)
        MERGE (attr)-[:HAS_CONCEPT]->(msize)
        MERGE (attr)-[:HAS_CONCEPT]->(mgrowth)
        MERGE (attr)-[:HAS_CONCEPT]->(comp)
        MERGE (attr)-[:HAS_CONCEPT]->(cap)
        MERGE (td)-[:HAS_CONCEPT]->(tam)
        MERGE (td)-[:HAS_CONCEPT]->(sam)
        MERGE (td)-[:HAS_CONCEPT]->(som)
        MERGE (syn)-[:HAS_CONCEPT]->(strat)
        MERGE (syn)-[:HAS_CONCEPT]->(syn2)
        MERGE (syn)-[:HAS_CONCEPT]->(risk)
        MERGE (vc)-[:HAS_CONCEPT]->(bottle)
        MERGE (vc)-[:HAS_CONCEPT]->(bench)
        """)
    