from fastapi import FastAPI, Response

from controllers.fetch_controller import (
    retrievePaper,
    retrievePaperExtractedData,
    retrievePaperMetadata,
)
from controllers.generate_controller import add_to_graph, generate_ontology_graph
from models.models import Paper
from models.route_model import (
    BuildGraphModel,
    ExtractModel,
    GetPaperModel,
)
from utils.logger import log

app = FastAPI()


@app.get("/")
async def main():
    log.info("Health Check")
    return {"message": "Working"}


@app.post("/getpapermetadata/")
async def submit(req: GetPaperModel, res: Response):
    log.info("Processing Get Paper Metadata Request")
    data = retrievePaperMetadata(req.source, req.paper_id)
    return data


@app.post("/extractpaperdata/")
async def extract(req: ExtractModel, res: Response):
    log.info("Processing Extract Paper Data Request")
    data = await retrievePaperExtractedData(req.paper_pdf_url)
    return data


@app.post("/buildgraph/")
async def buildgraph(req: BuildGraphModel, res: Response):
    log.info("Processing Build a networkx graph Request")
    data = await generate_ontology_graph(req.topic, req.num_papers)
    return data


@app.post("/extractpaper/")
async def extractPaper(req: GetPaperModel, res: Response):
    log.info("Processing Extract Paper Metadata and Data Request")
    data = await retrievePaper(req.source, req.paper_id)
    return data


@app.post("/addtograph/")
def addToGraph(paper: Paper, res: Response):
    log.info(f"Adding paper '{paper.metadata.title}' to Neo4j graph")
    add_to_graph(paper)
    return {"message": f"Paper '{paper.metadata.title}' successfully added to graph"}
