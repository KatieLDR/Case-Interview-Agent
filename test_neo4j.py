import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

print(f"URI: {os.getenv('NEO4J_URI')}")
print(f"Username: {os.getenv('NEO4J_USERNAME')}")
print(f"Password: {os.getenv('NEO4J_PASSWORD')}")

try:
    driver.verify_connectivity()
    print("Successfully connected to Neo4j database.")
except Exception as e:
    print(f"Failed to connect to Neo4j database: {e}")
finally:
    driver.close()
