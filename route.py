from controllers.fetch_controller import retrievePaperMetadataContent
from fastapi import FastAPI, Response
from models.route_model import SubmitModel

app = FastAPI()


@app.get("/")
async def main():
    return {"message": "Working"}


@app.post("/submit/")
async def submit(req: SubmitModel, res: Response):
    data = retrievePaperMetadataContent(req.source, req.paper_id)
    return data
