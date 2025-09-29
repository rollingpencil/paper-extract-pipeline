from dotenv import load_dotenv
from fastapi import FastAPI, Response

from controllers.fetch_controller import (
    retrievePaperDatasetList,
    retrievePaperMetadataContent,
)
from controllers.generate_controller import generate_ontology_graph
from models.route_model import BuildGraphModel, ExtractModel, GetPaperModel

load_dotenv()
app = FastAPI()


@app.get("/")
async def main():
    return {"message": "Working"}


@app.post("/getpaper/")
async def submit(req: GetPaperModel, res: Response):
    print("Processing Get Paper Request")
    data = retrievePaperMetadataContent(req.source, req.paper_id)
    return data


@app.post("/extract/")
async def extract(req: ExtractModel, res: Response):
    print("Processing Extract Request")
    data = await retrievePaperDatasetList(req.paper_pdf_url)
    return data


@app.post("/buildgraph/")
async def buildgraph(req: BuildGraphModel, res: Response):
    data = await generate_ontology_graph(req.topic, req.num_papers)
    return data
