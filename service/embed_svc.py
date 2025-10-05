import os

from dotenv import load_dotenv
from fastapi import HTTPException
from openai import AsyncOpenAI, PermissionDeniedError

from models.models import EmbeddingVector

load_dotenv()

embedding_client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_COMPAT_EMBED_API_ENDPOINT"),
    api_key=os.getenv("OPENAI_COMPAT_EMBED_API_KEY"),
)


async def embed_content(content: str) -> EmbeddingVector:
    print("E", end='')
    try:
        embedding = await embedding_client.embeddings.create(
            input=content,
            model=os.getenv("EMBED_MODEL_NAME"),
        )
    except PermissionDeniedError:
        raise HTTPException(status_code=500, detail=f"Failed to embed: {content}")

    return embedding.data[0].embedding
