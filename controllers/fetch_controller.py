from service.arxiv_svc import fetch_paper_metadata
from utils.constants import SourceType


def retrievePaperMetadata(source: SourceType, paper_id: str) -> dict:
    result = {"error": False}
    match source:
        case SourceType.ARXIV:
            result["content"] = fetch_paper_metadata(paper_id)
        case _:
            result["error"] = True
            result["message"] = "Unsupported source type"

    return result
