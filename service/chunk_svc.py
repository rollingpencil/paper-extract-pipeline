# from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.models import NodeRecord
from service.embed_svc import embed_content

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512, chunk_overlap=50, length_function=len, is_separator_regex=False
# )


async def chunk_and_embed_text(content: str) -> list[NodeRecord]:
    # docs = text_splitter.split_text(content)
    # docs_content_with_embed = []
    # for d in docs:
    #     embedding = await embed_content(d)
    #     docs_content_with_embed.append(
    #         NodeRecord(title=None, description=d, embedding=embedding)
    #     )
    # return docs_content_with_embed
    return []
