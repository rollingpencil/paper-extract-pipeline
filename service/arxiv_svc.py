from datetime import datetime

import feedparser
import pymupdf
import requests

from models.exceptions import PaperFetchError
from models.models import PaperMetadata
from service.embed_svc import embed_content

ARXIV_BASE_URL = "http://export.arxiv.org/api/query?"
ARXIV_BASE_PDF_URL = "https://arxiv.org/pdf/"


async def fetch_paper_metadata(paper_id: str) -> PaperMetadata:
    paper_meta = {}
    query_params = "id_list="
    response = requests.get(f"{ARXIV_BASE_URL}{query_params}{paper_id}")
    if response.status_code != 200:
        raise PaperFetchError("Failed to fetch paper metadata from arXiv.")

    feed = feedparser.parse(response.text)
    if len(feed.entries) != 1:
        raise PaperFetchError("Paper not found or multiple entries returned.")

    paper_id = feed.entries[0].id.split("/")[-1]
    title = feed.entries[0].title.replace("\n", " ").strip()
    authors = [author.name for author in feed.entries[0].authors]
    date_published = datetime.fromisoformat(feed.entries[0].published)
    date_updated = datetime.fromisoformat(feed.entries[0].updated)
    summary = feed.entries[0].summary.replace("\n", " ").strip()

    pdf_link = None
    for link in feed.entries[0].links:
        if link.type == "application/pdf":
            paper_meta["pdf_url"] = link.href
            pdf_link = link.href
    if pdf_link is None:
        raise PaperFetchError("Paper PDF Link not found.")

    summary_embedding = await embed_content(summary)

    paper_meta = PaperMetadata(
        id=paper_id,
        title=title,
        authors=authors,
        date_published=date_published,
        date_updated=date_updated,
        summary=summary,
        pdf_url=pdf_link,
        embedding=summary_embedding,
    )
    return paper_meta


def fetch_pdf_content(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise PaperFetchError(
            "Failed to fetch PDF content.", status_code=response.status_code
        )
    doc = pymupdf.Document(stream=response.content)
    doc_text = chr(12).join([page.get_text() for page in doc])
    return doc_text


def fetch_document_id_by_topic(topic: str, num_papers: int) -> list:
    documents = []
    start = 0
    query = f"search_query=all:{topic}&start={start}&max_results={num_papers}"
    response = requests.get(f"{ARXIV_BASE_URL}{query}")

    if response.status_code != 200:
        raise PaperFetchError("Failed to fetch paper metadata from arXiv.")

    feed = feedparser.parse(response.text)

    # Extract the document IDs per entry and formulate the PDF link
    for entry in feed.entries:
        doc = {}
        doc["id"] = entry.id.split("/abs/")[-1]
        doc["pdf_link"] = f"{ARXIV_BASE_PDF_URL}{doc['id']}"
        documents.append(doc)

    return documents
