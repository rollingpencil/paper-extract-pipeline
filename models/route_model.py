from pydantic import BaseModel, Field
from utils.constants import SourceType


class SubmitModel(BaseModel):
    source: SourceType = Field(..., example="arxiv")
    paper_id: str = Field(..., example="2506.00664")
