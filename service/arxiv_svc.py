import feedparser
import requests


ARXIV_BASE_URL = "http://export.arxiv.org/api/query?"


def fetch_paper_metadata(paper_id: str) -> dict:
    paper_meta = {"error": False}
    query_params = "id_list="
    response = requests.get(f"{ARXIV_BASE_URL}{query_params}{paper_id}")
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    if len(feed.entries) != 1:
        paper_meta["error"] = True
        paper_meta["message"] = "Paper not found or multiple entries returned."
        return paper_meta

    paper_meta["title"] = feed.entries[0].title.replace("\n", " ").strip()
    paper_meta["authors"] = [author.name for author in feed.entries[0].authors]
    paper_meta["summary"] = feed.entries[0].summary.replace("\n", " ").strip()
    for link in feed.entries[0].links:
        if link.type == "application/pdf":
            paper_meta["pdf_url"] = link.href
    return paper_meta
