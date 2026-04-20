"""
keep_alive.py — AuraDB Free Tier Keep-Alive
Runs a minimal Cypher query to prevent AuraDB Free from pausing after 72h inactivity.
Designed to be executed by Cloud Run Job, triggered every 24h via Cloud Scheduler.

Change log: 2026-04-16 — initial build
Change log: 2026-04-16 — switched RETURN 1 to MATCH (n) RETURN count(n) to ensure real graph READ activity
"""

import os
import sys
import logging
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def ping_neo4j():
    uri      = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not uri or not password:
        logging.error("NEO4J_URI or NEO4J_PASSWORD not set — aborting.")
        sys.exit(1)

    logging.info(f"Connecting to AuraDB at {uri} ...")

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS alive LIMIT 1")
            record = result.single()
            if record and record["alive"] is not None:
                logging.info("✅ AuraDB ping successful — instance is alive.")
            else:
                logging.warning("⚠️ Unexpected response from AuraDB.")
        driver.close()
    except Exception as e:
        logging.error(f"❌ AuraDB ping failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    ping_neo4j()