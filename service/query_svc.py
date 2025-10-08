import json
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from service.embed_svc import embed_content
from service.neo4j_svc import get_graph_schema, process_query, process_vector_search
from utils.logger import log
from utils.utils import get_envvar

NEO4J_QUERY_SYSTEM_PROMPT = f"""
You are a Neo4j graph database expert with access to both Cypher queries and vector similarity search.

Graph Schema:
{get_graph_schema()}

When asked a question:
1. Analyze what information is needed
2. If the question requires finding nodes based on semantic meaning (e.g., "papers about X", "datasets used for Y"), use vector_search FIRST to find relevant nodes
3. Once you have specific node IDs from vector search, use query_neo4j to traverse relationships and gather additional information
4. For direct graph traversals with known entities, use query_neo4j directly
5. You can combine both tools: vector_search to find starting nodes, then query_neo4j for multi-hop traversals (up to 3 hops)
6. Always use correct labels, relationship types, and property names from the schema
7. Provide a clear natural language answer based on the results

Always ensure your Cypher syntax is correct and optimized.
"""


class GraphQueryResult(BaseModel):
    answer: str = Field(description="Natural language answer to the question")


log.info(f"Query Model set: {get_envvar('QUERY_MODEL_NAME')}")
llm_model = OpenAIChatModel(
    get_envvar("QUERY_MODEL_NAME"),
    provider=OpenAIProvider(
        base_url=get_envvar("OPENAI_COMPAT_API_ENDPOINT"),
        api_key=get_envvar("OPENAI_COMPAT_API_KEY"),
    ),
)

query_agent = Agent(
    llm_model,
    system_prompt=NEO4J_QUERY_SYSTEM_PROMPT,
    output_type=GraphQueryResult,
)


async def query(question: str) -> GraphQueryResult:
    """
    Query Neo4j using natural language (async).

    Args:
        question: Natural language question

    Returns:
        GraphQueryResult with answer and data
    """

    result = await query_agent.run(
        question, model_settings=json.loads(get_envvar("QUERY_MODEL_SETTINGS"))
    )
    return result.output


@query_agent.tool
def query_neo4j(ctx: RunContext[None], cypher_query: str) -> list[dict[str, Any]]:
    """
    Execute a Cypher query against the Neo4j database.

    Args:
        cypher_query: The Cypher query to execute

    Returns:
        List of records as dictionaries
    """
    log.debug(f"{'=' * 60}")
    log.debug("GENERATED CYPHER QUERY:")
    log.debug(f"{'=' * 60}")
    log.debug(cypher_query)
    log.debug(f"{'=' * 60}\n")

    return process_query(cypher_query)


@query_agent.tool
async def vector_search(
    ctx: RunContext[None], query_text: str, index_name: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """
    Search for nodes using vector similarity on their embeddings.

    Args:
        query_text: The text to search for semantically (e.g., "transforms unstructured PDFs into a queryable ontology")
        index_name: Name of the vector index to search. Available indexes are shown in the schema.
        top_k: Number of top results to return (default 5)

    Returns:
        List of matching nodes with their properties and similarity scores
    """

    log.debug(f"{'=' * 60}")
    log.debug("VECTOR SEARCH:")
    log.debug(f"{'=' * 60}")
    log.debug(f"Query: {query_text}")
    log.debug(f"Index: {index_name}")
    log.debug(f"Top K: {top_k}")
    log.debug(f"{'=' * 60}\n")

    # Generate embedding for the query text
    query_embedding = await embed_content(query_text)
    log.debug(f"Generated embedding (dim: {len(query_embedding)})")

    # Execute the vector search query
    return process_vector_search(index_name, top_k, query_embedding)
