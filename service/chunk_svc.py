from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.models import NodeRecord
from service.embed_svc import embed_content
from utils.logger import log

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50, length_function=len, is_separator_regex=False
)


async def chunk_and_embed_text(content: str) -> list[NodeRecord]:
    log.info("Chunking and embedding")
    docs = text_splitter.split_text(content)
    docs_content_with_embed = []
    for chunk in docs:
        embedding = await embed_content(chunk)
        docs_content_with_embed.append(
            NodeRecord(title=None, description=chunk, embedding=embedding)
        )
    return docs_content_with_embed
