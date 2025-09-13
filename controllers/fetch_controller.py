from models.exceptions import SourceTypeError
from service.arxiv_svc import fetch_paper_metadata, fetch_pdf_content
from utils.constants import SourceType


def retrievePaperMetadata(source: SourceType, paper_id: str) -> dict:
    result = {}
    match source:
        case SourceType.ARXIV:
            result["content"] = fetch_paper_metadata(paper_id)
        case _:
            raise SourceTypeError
    return result


def retrievePaperContent(source: SourceType, pdf_url: str) -> dict:
    result = {}
    match source:
        case SourceType.ARXIV:
            result["content"] = fetch_pdf_content(pdf_url)
        case _:
            raise SourceTypeError
    return result


def retrievePaperMetadataContent(source: SourceType, paper_id: str) -> dict:
    meta_result = retrievePaperMetadata(source, paper_id)

    content_result = retrievePaperContent(
        source, meta_result["content"].get("pdf_url", "")
    )

    meta_result["content"]["full_text"] = content_result["content"]

    return meta_result
