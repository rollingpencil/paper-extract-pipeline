from controllers.fetch_controller import (
    retrievePaperDatasetList,
    retrievePaperMetadataContent,
)
from fastapi import FastAPI, Response
from models.route_model import ExtractModel, GetPaperModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()


@app.get("/")
async def main():
    return {"message": "Working"}


@app.post("/getpaper/")
async def submit(req: GetPaperModel, res: Response):
    data = retrievePaperMetadataContent(req.source, req.paper_id)
    return data


@app.post("/extract/")
async def extract(req: ExtractModel, res: Response):
    data = await retrievePaperDatasetList(req.paper_pdf_url)
    return data
