from models.exceptions import SourceTypeError
from service.arxiv_svc import fetch_paper_metadata, fetch_pdf_content
from service.openrouter_svc import (
    extract_paper_dataset,
    extract_paper_methods,
    extract_paper_models,
)
from utils.constants import SourceType


def retrievePaperMetadata(source: SourceType, paper_id: str) -> dict:
    result = {}
    match source:
        case SourceType.ARXIV:
            result["content"] = fetch_paper_metadata(paper_id)
        case _:
            raise SourceTypeError
    return result


def retrievePaperContent(pdf_url: str) -> dict:
    result = {}
    result["content"] = fetch_pdf_content(pdf_url)
    return result


def retrievePaperMetadataContent(source: SourceType, paper_id: str) -> dict:
    meta_result = retrievePaperMetadata(source, paper_id)

    content_result = retrievePaperContent(meta_result["content"].get("pdf_url", ""))

    meta_result["content"]["full_text"] = content_result["content"]

    return meta_result


async def retrievePaperDatasetList(pdf_url: str) -> dict:
    full_text_result = retrievePaperContent(pdf_url)
    full_text_result["datasets"] = await extract_paper_dataset(
        full_text_result["content"]
    )
    full_text_result["models"] = await extract_paper_models(full_text_result["content"])
    full_text_result["methods"] = await extract_paper_methods(
        full_text_result["content"]
    )
    return full_text_result
