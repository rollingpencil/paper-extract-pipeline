from typing import Dict, Any, List
import os
from neo4j import GraphDatabase
from models.models import (
    PaperExtractedData,
    PaperMetadata
)

ENDPOINT_LINK = os.getenv("NEO4J_ENDPOINT")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def find_or_create_vector_node(
    tx,
    label: str,
    id_property: str,
    node_id: str,
    title: str,
    description: str,
    embedding: List[float],
    threshold: float = 0.94,
):
    """
    Uses Neo4j vector index to find semantically similar node.
    Deduplicates using explicit ID or high cosine similarity.
    """

    # Check if node exists
    exact_match = tx.run(
        f"MATCH (n:{label} {{{id_property}: $node_id}}) RETURN n LIMIT 1",
        node_id=node_id,
    ).data()
    if exact_match:
        return exact_match[0]["n"]

    # Check if similar node with extremely high similarity score exists
    index_name = f"{label.lower()}_vector_index"
    query = f"""
    CALL db.index.vector.queryNodes('{index_name}', 1, $embedding)
    YIELD node, score
    RETURN node, score
    """
    result = tx.run(query, embedding=embedding).data()

    if result and result[0]["score"] >= threshold:
        return result[0]["node"]

    # Create a fresh node if node not already in
    tx.run(
        f"""
        CREATE (n:{label} {{
            {id_property}: $node_id,
            title: $title,
            description: $description,
            embedding: $embedding
        }})
        """,
        node_id=node_id,
        title=title,
        description=description,
        embedding=embedding,
    )
    return {"id": node_id}


def create_similarity_links(tx, label: str, node_id: str, embedding: List[float], top_k=5, threshold=0.9):
    """
    Creates :SIMILAR_TO relationships between semantically close nodes
    using the label's vector index.
    """
    index_name = f"{label.lower()}_vector_index"

    tx.run(
        f"""
        MATCH (src:{label} {{{'id' if label == 'Paper' else 'title'}: $node_id}})
        CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
        YIELD node, score
        WHERE ({'node.id' if label == 'Paper' else 'node.title'}) <> $node_id AND score > $threshold
        MERGE (src)-[:SIMILAR_TO {{score: score}}]->(node)
        """,
        node_id=node_id,
        embedding=embedding,
        k=top_k,
        threshold=threshold,
    )


def get_neo4j_driver():
    """Returns a Neo4j driver instance."""
    return GraphDatabase.driver(ENDPOINT_LINK, auth=(USER, PASSWORD))


def store_paper_node(session, meta: PaperMetadata):
    """Creates or updates the Paper node."""
    print("Creating/updating paper node...")
    session.run(
        """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.summary = $summary,
            p.date_published = $date_published,
            p.date_updated = $date_updated,
            p.embedding = $embedding
        """,
        id=meta.id,
        title=meta.title,
        summary=meta.summary,
        date_published=meta.date_published,
        date_updated=meta.date_updated,
        embedding=meta.embedding,
    )
    print("Created/updated paper node.")


def store_authors(session, meta: PaperMetadata):
    """Creates Author nodes and links them to the Paper."""
    print("Creating author node...")
    for author in meta.authors:
        session.run(
            """
            MERGE (a:Author {name: $name})
            WITH a
            MATCH (p:Paper {id: $paper_id})
            MERGE (p)-[:WRITTEN_BY]->(a)
            """,
            name=author,
            paper_id=meta.id,
        )
    print("Created author node.")


def store_content_chunks(session, pdf_data: PaperExtractedData, paper_id: str):
    """Creates Content chunk nodes and links them to the Paper."""
    print("Creating chunk nodes...")
    for chunk in pdf_data.content:
        session.run(
            """
            CREATE (c:Content {description: $description, embedding: $embedding})
            WITH c
            MATCH (p:Paper {id: $paper_id})
            MERGE (p)-[:CONTAINS_CHUNK]->(c)
            """,
            description=chunk.description,
            embedding=chunk.embedding,
            paper_id=paper_id,
        )
    print("Created chunk nodes.")


def store_semantic_entities(session, pdf_data: PaperExtractedData, paper_id: str):
    """Creates Dataset, Model, Method, and Task nodes with semantic links."""
    print("Creating dataset, model, method, task nodes...")
    mapping = [
        ("Dataset", "USES_DATASET", pdf_data.datasets, "title"),
        ("Model", "USES_MODEL", pdf_data.models, "title"),
        ("Method", "USES_METHOD", pdf_data.methods, "title"),
        ("Task", "SOLVES_TASK", pdf_data.tasking, "title"),
    ]

    for label, rel, node_list, id_prop in mapping:
        print(f"Creating {label} nodes, with the {rel} relation...")
        for node in node_list:
            # Create or find similar node (by title or embedding)
            session.execute_write(
                find_or_create_vector_node,
                label,
                id_prop,
                node.title,
                node.title,
                node.description,
                node.embedding,
                0.95,
            )
            # Link to the paper
            session.run(
                f"""
                MATCH (p:Paper {{id: $paper_id}})
                MATCH (n:{label} {{{id_prop}: $node_id}})
                MERGE (p)-[:{rel}]->(n)
                """,
                paper_id=paper_id,
                node_id=node.title,
            )

            # Build similarity links among same-label nodes
            session.execute_write(
                create_similarity_links,
                label,
                node.title,
                node.embedding,
                5,
                0.9,
            )
        print(f"Created {label} nodes, with the {rel} relation.")


def store_paper_similarity_links(session, meta: PaperMetadata):
    """Creates similarity links between papers."""
    print("Building similarity links between papers...")
    session.execute_write(
        create_similarity_links,
        "Paper",
        meta.id,
        meta.embedding,
        5,
        0.9,
    )
    print("Built similarity links between papers.")
