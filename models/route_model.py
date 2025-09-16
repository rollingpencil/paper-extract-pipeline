from pydantic import BaseModel, Field
from utils.constants import SourceType


class GetPaperModel(BaseModel):
    source: SourceType = Field(..., example="arxiv")
    paper_id: str = Field(..., example="2506.00664")


class ExtractModel(BaseModel):
    paper_pdf_url: str = Field(..., example="https://arxiv.org/pdf/2506.00664")
