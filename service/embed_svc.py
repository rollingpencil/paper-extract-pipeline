from fastapi import HTTPException
from openai import AsyncOpenAI, PermissionDeniedError

from models.models import EmbeddingVector
from utils.logger import log
from utils.utils import get_envvar

embedding_client = AsyncOpenAI(
    base_url=get_envvar("OPENAI_COMPAT_EMBED_API_ENDPOINT"),
    api_key=get_envvar("OPENAI_COMPAT_EMBED_API_KEY"),
)


async def embed_content(content: str) -> EmbeddingVector:
    log.debug(f"Embedding: {len(content) > 40 and content[:40] + '...' or content}")
    try:
        embedding = await embedding_client.embeddings.create(
            input=content,
            model=get_envvar("EMBED_MODEL_NAME"),
        )
    except PermissionDeniedError:
        raise HTTPException(status_code=500, detail=f"Failed to embed: {content}")

    return embedding.data[0].embedding
