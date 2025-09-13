from controllers.fetch_controller import retrievePaperMetadata
from fastapi import FastAPI
from models.route_model import SubmitModel

app = FastAPI()


@app.get("/")
async def main():
    return {"message": "Working"}


@app.post("/submit/")
async def submit(req: SubmitModel):
    data = retrievePaperMetadata(req.source, req.paper_id)
    return {"content": data}
