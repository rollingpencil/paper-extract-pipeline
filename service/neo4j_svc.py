from typing import Any, List

from neo4j import Driver, GraphDatabase, Query
from neo4j.time import Date, DateTime, Duration, Time

from models.models import EmbeddingVector, PaperExtractedData, PaperMetadata
from utils.logger import log
from utils.utils import get_envvar

DATABASE_NAME = get_envvar("NEO4J_DATABASE_NAME")

VECTOR_SEARCH_CYPHER_TEXT = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
    YIELD node, score
    RETURN node, score
    ORDER BY score DESC
"""


def get_neo4j_driver() -> Driver:
    """Returns a Neo4j driver instance."""
    return GraphDatabase.driver(
        get_envvar("NEO4J_ENDPOINT"),
        auth=(get_envvar("NEO4J_USER"), get_envvar("NEO4J_PASSWORD")),
    )


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


def create_similarity_links(
    tx, label: str, node_id: str, embedding: List[float], top_k=5, threshold=0.9
):
    """
    Creates :SIMILAR_TO relationships between semantically close nodes
    using the label's vector index.
    """
    index_name = f"{label.lower()}_vector_index"

    tx.run(
        f"""
        MATCH (src:{label} {{{"id" if label == "Paper" else "title"}: $node_id}})
        CALL db.index.vector.queryNodes('{index_name}', $k, $embedding)
        YIELD node, score
        WHERE ({"node.id" if label == "Paper" else "node.title"}) <> $node_id AND score > $threshold
        MERGE (src)-[:SIMILAR_TO {{score: score}}]->(node)
        """,
        node_id=node_id,
        embedding=embedding,
        k=top_k,
        threshold=threshold,
    )


def store_paper_node(session, meta: PaperMetadata):
    """Creates or updates the Paper node."""
    log.info("Creating/updating paper node...")
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
    log.info("Created/updated paper node.")


def store_authors(session, meta: PaperMetadata):
    """Creates Author nodes and links them to the Paper."""
    log.info("Creating author node...")
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
    log.info("Created author node.")


def store_content_chunks(session, pdf_data: PaperExtractedData, paper_id: str):
    """Creates Content chunk nodes and links them to the Paper."""
    log.info("Creating chunk nodes...")
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
    log.info("Created chunk nodes.")


def store_semantic_entities(session, pdf_data: PaperExtractedData, paper_id: str):
    """Creates Dataset, Model, Method, and Task nodes with semantic links."""
    log.info("Creating dataset, model, method, task nodes...")
    mapping = [
        ("Dataset", "USES_DATASET", pdf_data.datasets, "title"),
        ("Model", "USES_MODEL", pdf_data.models, "title"),
        ("Method", "USES_METHOD", pdf_data.methods, "title"),
        ("Task", "SOLVES_TASK", pdf_data.tasking, "title"),
    ]

    for label, rel, node_list, id_prop in mapping:
        log.info(f"Creating {label} nodes, with the {rel} relation...")
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
        log.info(f"Created {label} nodes, with the {rel} relation.")


def store_paper_similarity_links(session, meta: PaperMetadata):
    """Creates similarity links between papers."""
    log.info("Building similarity links between papers...")
    session.execute_write(
        create_similarity_links,
        "Paper",
        meta.id,
        meta.embedding,
        5,
        0.9,
    )
    log.info("Built similarity links between papers.")


def serialize_neo4j_types(obj: Any) -> Any:
    """Convert Neo4j types to JSON-serializable Python types"""
    if isinstance(obj, (DateTime, Date, Time)):
        return obj.iso_format()
    elif isinstance(obj, Duration):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_neo4j_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_neo4j_types(item) for item in obj]
    else:
        return obj


def get_graph_schema() -> str:
    """Retrieve Neo4j graph schema with property mappings"""
    driver = get_neo4j_driver()
    with driver.session(database=DATABASE_NAME) as session:
        # Get detailed schema for each node label
        schema_parts = ["Graph Schema:\n"]

        # Get node labels
        node_result = session.run("CALL db.labels()")
        labels = [record[0] for record in node_result]

        # For each label, get sample properties
        schema_parts.append("Node Types:")
        for label in labels:
            # Get a sample node to see its properties
            sample = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1")
            record = sample.single()
            if record:
                props = list(record["n"].keys())
                schema_parts.append(f"  - {label}: {{{', '.join(props)}}}")
            else:
                schema_parts.append(f"  - {label}")

        # Get relationship types with start/end node info
        schema_parts.append("\nRelationship Types:")
        rel_query = """
            MATCH (a)-[r]->(b)
            WITH type(r) as rel_type, labels(a)[0] as from_label, labels(b)[0] as to_label
            RETURN DISTINCT rel_type, from_label, to_label
            ORDER BY rel_type
            """
        rel_result = session.run(rel_query)
        for record in rel_result:
            schema_parts.append(
                f"  - ({record['from_label']})-[{record['rel_type']}]->({record['to_label']})"
            )

        # Get vector indexes
        try:
            index_result = session.run("SHOW INDEXES")
            vector_indexes = []
            for record in index_result:
                if record.get("type") == "VECTOR":
                    index_name = record.get("name")
                    # Get label from index name or labelsOrTypes field
                    labels_or_types = record.get("labelsOrTypes", [])
                    label = labels_or_types[0] if labels_or_types else index_name
                    vector_indexes.append(f"{index_name} (on {label})")

            if vector_indexes:
                schema_parts.append("\nVector Indexes:")
                for idx in vector_indexes:
                    schema_parts.append(f"  - {idx}")
        except Exception:
            # If SHOW INDEXES fails, skip vector index info
            pass

        schema = "\n".join(schema_parts)
    return schema


def process_query(cypher_query: str, cache_func=None) -> list[dict[str, Any]]:
    driver = get_neo4j_driver()
    with driver.session(database=DATABASE_NAME) as session:
        result = session.run(cypher_query)
        records = [serialize_neo4j_types(dict(record)) for record in result]

        if cache_func:
            cache_func(cypher_query, records)

        log.debug(f"Query returned {len(records)} record(s)")
        if records:
            log.debug(f"Sample result: {records[0]}")

        return records


def process_vector_search(
    index_name: str,
    top_k: int,
    query_embedding: EmbeddingVector,
    query_text: str,
    cache_func=None,
) -> list[dict[str, Any]]:
    driver = get_neo4j_driver()
    # Perform vector search using Neo4j's vector index
    driver = get_neo4j_driver()
    with driver.session(database=DATABASE_NAME) as session:
        cypher_query = Query(VECTOR_SEARCH_CYPHER_TEXT)

        result = session.run(
            cypher_query,
            index_name=index_name,
            top_k=top_k,
            query_embedding=query_embedding,
        )
        records = [
            {
                "node": serialize_neo4j_types(dict(record["node"])),
                "score": record["score"],
            }
            for record in result
        ]

        if cache_func:
            cache_query = f"{query_text}|{index_name}|{top_k}"
            cache_func(cache_query, records)

        log.debug(f"Vector search returned {len(records)} result(s)")
        if records:
            log.debug(f"Top result (score: {records[0]['score']:.4f}):")
            # Show key fields from the node
            node = records[0]["node"]
            for key in ["title", "name", "content", "abstract"]:
                if key in node:
                    value = node[key]
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    log.debug(f"  {key}: {value}")

        return records


def check_paper_exists(paper_id: str) -> bool:
    query = f'MATCH (p:Paper) WHERE p.id CONTAINS "{paper_id}" RETURN *'
    driver = get_neo4j_driver()
    with driver.session(database=DATABASE_NAME) as session:
        result = session.run(query)
        return result.single(strict=False) is not None
