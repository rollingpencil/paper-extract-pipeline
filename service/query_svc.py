import hashlib
import json
import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from service.embed_svc import embed_content
from service.neo4j_svc import get_graph_schema, process_query, process_vector_search
from utils.logger import log
from utils.utils import get_envvar

CACHE_ENABLE: bool = bool(get_envvar("QUERY_CACHE_ENABLE"))
CACHE_TTL: int = int(get_envvar("QUERY_CACHE_TTL"))

NEO4J_QUERY_SYSTEM_PROMPT = f"""
You are a Neo4j graph database expert with access to both Cypher queries and vector similarity search.

CRITICAL: After each tool call, check if you have enough information to answer the question. If yes, IMMEDIATELY return the answer without making additional tool calls.

{get_graph_schema()}

When asked a question:
1. Analyze what information is needed to answer the question
2. If the question requires finding nodes based on semantic meaning (e.g., "papers about X", "datasets used for Y"), use vector_search FIRST to find relevant nodes
3. Once you have specific node IDs from vector search, use query_neo4j to traverse relationships and gather additional information
4. For direct graph traversals with known entities, use query_neo4j directly
5. You can combine both tools: vector_search to find starting nodes, then query_neo4j for multi-hop traversals (up to 3 hops)
6. Always use correct labels, relationship types, and property names from the schema
7. ALWAYS show the complete reasoning process with traversal path in your response

FORMAT REQUIREMENTS for your response:
- In the 'reasoning' field: Show the complete traversal path using arrows. For each hop, include the node's key identifying information and its type in parentheses.
  Example: "Multi-Hop Reasoning Dataset (Dataset)" -> "Knowledge Graph Question Answering (Paper)" -> "John Smith (Author)"
- If vector search was used, mention it in reasoning: "Found via vector search: [node info], then traversed to..."
- In the 'answer' field: Provide the final natural language answer to the question

IMPORTANT - Query Efficiency Rules:
- ONLY make tool calls if you need MORE information to answer the question
- ONCE you have sufficient data to answer the user's question, STOP and provide the answer immediately
- DO NOT make redundant queries that retrieve the same information
- DO NOT query for additional details unless specifically requested
- If a query returns empty results, do NOT retry with the same or similar query
- Prefer ONE comprehensive query over multiple smaller queries when possible

8. Before providing the answer, please provide the hops taken to get information. For example, if the hop is from the chunk with description "This is A" to the paper with the title "Paper 1" to the author "Author 1", represent it as ""This is A (Content)" -> "Paper 1 (Paper)" -> "Author 1 (Author)"" where the items in the bracket is the type of node.

Always ensure your Cypher syntax is correct and optimized. Provide a clear natural language answer based on the results.
"""


class GraphQueryResult(BaseModel):
    reasoning: str = Field(
        description="The reasoning process including traversal hops in format: 'Node1 (Type)' -> 'Node2 (Type)' -> 'Node3 (Type)'"
    )
    answer: str = Field(description="Natural language answer to the question")


_query_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
_vector_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}

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
    retries=int(get_envvar("QUERY_RETRIES")),
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

    # Clean expired cache entries periodically
    _clean_expired_cache()

    # Check cache first
    cache_func = None
    if CACHE_ENABLE:
        cache_key = _get_cache_key(cypher_query)
        if cache_key in _query_cache:
            cached_result, timestamp = _query_cache[cache_key]
            if _is_cache_valid(timestamp):
                log.debug(f"\n{'=' * 60}")
                log.debug("CACHE HIT - Using cached result")
                log.debug(f"{'=' * 60}")
                log.debug(f"Query: {cypher_query}")
                log.debug(f"{'=' * 60}\n")
                return cached_result
        cache_func = _query_cache_func

    log.debug(f"{'=' * 60}")
    log.debug("GENERATED CYPHER QUERY:")
    log.debug(f"{'=' * 60}")
    log.debug(cypher_query)
    log.debug(f"{'=' * 60}\n")

    return process_query(cypher_query, cache_func=cache_func)


def _query_cache_func(cypher_query, records):
    log.debug(f"{'=' * 60}")
    log.debug("QUERY CACHE MISS - Caching result")
    log.debug(f"{'=' * 60}")
    cache_key = _get_cache_key(cypher_query)
    _query_cache[cache_key] = (records, time.time())


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

    # Clean expired cache entries periodically
    _clean_expired_cache()

    # Check cache first
    cache_func = None
    cache_query = f"{query_text}|{index_name}|{top_k}"
    if CACHE_ENABLE:
        cache_key = _get_cache_key(cache_query)
        if cache_key in _vector_cache:
            cached_result, timestamp = _vector_cache[cache_key]
            if _is_cache_valid(timestamp):
                log.debug(f"\n{'=' * 60}")
                log.debug("CACHE HIT - Using cached vector search result")
                log.debug(f"{'=' * 60}")
                log.debug(f"Query: {query_text}")
                log.debug(f"Index: {index_name}")
                log.debug(f"Top K: {top_k}")
                log.debug(f"{'=' * 60}\n")
                return cached_result
        cache_func = _vector_search_cache_func

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
    return process_vector_search(
        index_name, top_k, query_embedding, query_text, cache_func=cache_func
    )


def _vector_search_cache_func(cache_query, records):
    log.debug(f"{'=' * 60}")
    log.debug("VECTOR CACHE MISS - Caching result")
    log.debug(f"{'=' * 60}")
    cache_key = _get_cache_key(cache_query)
    _vector_cache[cache_key] = (records, time.time())


def _get_cache_key(query: str) -> str:
    """Generate a cache key from a query string"""
    return hashlib.md5(query.encode()).hexdigest()


def _is_cache_valid(timestamp: float) -> bool:
    """Check if a cached entry is still valid"""
    return (time.time() - timestamp) < CACHE_TTL


def _clean_expired_cache():
    """Remove expired entries from both caches"""
    current_time = time.time()

    # Clean query cache
    expired_keys = [
        key
        for key, (_, timestamp) in _query_cache.items()
        if (current_time - timestamp) >= CACHE_TTL
    ]
    for key in expired_keys:
        del _query_cache[key]

    # Clean vector cache
    expired_keys = [
        key
        for key, (_, timestamp) in _vector_cache.items()
        if (current_time - timestamp) >= CACHE_TTL
    ]
    for key in expired_keys:
        del _vector_cache[key]


def clear_cache():
    """Manually clear all caches"""
    _query_cache.clear()
    _vector_cache.clear()
    log.debug("Cache cleared")
