from fastapi import FastAPI, Response

from controllers.fetch_controller import (
    retrieve_paper,
    retrieve_paper_extracted_data,
    retrieve_paper_metadata,
)
from controllers.generate_controller import (
    add_to_graph,
    check_paper_exists_graph,
    generate_ontology_graph,
    query_database,
)
from models.exceptions import PaperAlreadyExistsError
from models.models import Paper
from models.route_model import (
    BuildGraphModel,
    ExtractModel,
    GetPaperModel,
    QueryModel,
)
from service.eval_svc import evaluate_query_relevance, evaluate_with_judge
from utils.logger import log

app = FastAPI()


@app.get("/")
async def main():
    log.info("Health Check")
    return {"message": "Working"}


@app.post("/getpapermetadata/")
async def get_paper_metadata(req: GetPaperModel, res: Response):
    log.info("Processing Get Paper Metadata Request")
    data = retrieve_paper_metadata(req.source, req.paper_id)
    return data


@app.post("/extractpaperdata/")
async def extract_paper_data(req: ExtractModel, res: Response):
    log.info("Processing Extract Paper Data Request")
    data = await retrieve_paper_extracted_data(req.paper_pdf_url)
    return data


@app.post("/buildgraph/")
async def buildgraph(req: BuildGraphModel, res: Response):
    log.info("Processing Build a networkx graph Request")
    data = await generate_ontology_graph(req.topic, req.num_papers)
    return data


@app.post("/extractpaper/")
async def extract_paper(req: GetPaperModel, res: Response):
    log.info("Processing Extract Paper Metadata and Data Request")
    data = await retrieve_paper(req.source, req.paper_id)
    return data


@app.post("/addtograph/")
async def add_paper_to_graph(paper: Paper, res: Response):
    log.info(f"Adding paper '{paper.metadata.title}' to Neo4j graph")
    add_to_graph(paper)
    return {"message": f"Paper '{paper.metadata.title}' successfully added to graph"}


@app.post("/importpaper/")
async def import_paper_to_graph(req: GetPaperModel, res: Response):
    log.info("Processing Extract Paper Metadata and Data Request")
    if check_paper_exists_graph(req.paper_id):
        raise PaperAlreadyExistsError(detail=f"Paper with {req.paper_id} already exist")
    paper = await retrieve_paper(req.source, req.paper_id)
    log.info(f"Adding paper '{paper.metadata.title}' to Neo4j graph")
    add_to_graph(paper)
    return {"message": f"Paper '{paper.metadata.title}' successfully added to graph"}


@app.post("/query/")
async def send_query_database(req: QueryModel, res: Response):
    log.info("Processing Query")
    return await query_database(req.qns)


@app.post("/eval/relevance/")
async def eval_simple(res: Response):
    """Call evaluate_query_relevance with a dummy query and dummy embeddings."""
    dummy_query = "test query about contrastive learning"
    # small dummy embeddings (length mismatch with real model is acceptable for quick smoke test)
    dummy_embeddings = [[0.1] * 8, [0.2] * 8]
    result = await evaluate_query_relevance(dummy_query, dummy_embeddings)
    return result


@app.post("/eval/judge/")
async def eval_judge(res: Response):
    """Call evaluate_with_judge with dummy query/answer/evidence and return judge output."""
    dummy_query = "What datasets are commonly used for contrastive learning in vision?"
    dummy_answer = "Common datasets include ImageNet and CIFAR-10; many works also use COCO."
    dummy_evidence = [
        {"id": "paper1", "text": "We evaluated on ImageNet and CIFAR-10.", "embedding": [0.1] * 8},
        {"id": "paper2", "text": "COCO was used for additional experiments.", "embedding": [0.2] * 8},
    ]
    result = await evaluate_with_judge(dummy_query, dummy_answer, dummy_evidence)
    return result
