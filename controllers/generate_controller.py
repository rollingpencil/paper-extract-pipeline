from controllers.fetch_controller import (
    retrievePaperMetadata,
    retrievePaperDatasetList
)

from service.arxiv_svc import (
    fetch_document_id_by_topic,
)
from utils.constants import SourceType

def retrieveDocumentIds(topic: str, num_papers: int) -> list:
    ids = fetch_document_id_by_topic(topic, num_papers)
    for id in ids:
        id["source"] = SourceType.ARXIV

    return ids

async def generate_ontology_graph(topic: str, num_papers: int):
    data = {}
    ids =  retrieveDocumentIds(topic, num_papers)

    for id in ids:
        document_data = {}
        
        document_data["metadata"] = retrievePaperMetadata(id["source"], id["id"])["content"]
        document_data["pdf_data"]  = await retrievePaperDatasetList(id["pdf_link"])

        data[id["id"]] = document_data
    
    return data

