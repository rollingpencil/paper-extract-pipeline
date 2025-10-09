from models.exceptions import SourceTypeError
from models.models import Paper, PaperExtractedData, PaperMetadata
from service.arxiv_svc import fetch_paper_metadata, fetch_pdf_content
from service.chunk_svc import chunk_and_embed_text
from service.extract_svc import (
    extract_paper_dataset,
    extract_paper_methods,
    extract_paper_models,
    extract_paper_tasking,
)
from utils.constants import SourceType


async def retrieve_paper_metadata(source: SourceType, paper_id: str) -> PaperMetadata:
    match source:
        case SourceType.ARXIV:
            return await fetch_paper_metadata(paper_id)
        case _:
            raise SourceTypeError


async def retrieve_paper_extracted_data(pdf_url: str) -> PaperExtractedData:
    pdf_text = fetch_pdf_content(pdf_url)
    datasets = await extract_paper_dataset(pdf_text)
    models = await extract_paper_models(pdf_text)
    methods = await extract_paper_methods(pdf_text)
    tasking = await extract_paper_tasking(pdf_text)
    content = await chunk_and_embed_text(pdf_text)

    return PaperExtractedData(
        content=content,
        datasets=datasets,
        models=models,
        methods=methods,
        tasking=tasking,
    )


async def retrieve_paper(source: SourceType, paper_id: str) -> Paper:
    match source:
        case SourceType.ARXIV:
            return await _retrieve_paper_arxiv(paper_id)
        case _:
            raise SourceTypeError


async def _retrieve_paper_arxiv(paper_id: str) -> Paper:
    metadata = await fetch_paper_metadata(paper_id)
    pdf_data = await retrieve_paper_extracted_data(metadata.pdf_url)
    return Paper(metadata=metadata, pdf_data=pdf_data)
