from typing import TypeAlias

from pydantic import BaseModel

EmbeddingVector: TypeAlias = list[float]


class PaperMetadata(BaseModel):
    title: str
    authors: list[str]
    summary: str
    pdf_url: str
    embedding: EmbeddingVector


class NodeRecord(BaseModel):
    title: str | None
    description: str
    embedding: EmbeddingVector


class PaperExtractedData(BaseModel):
    content: list[NodeRecord]
    datasets: list[NodeRecord]
    models: list[NodeRecord]
    methods: list[NodeRecord]
    tasking: list[NodeRecord]


class Paper(BaseModel):
    metadata: PaperMetadata
    pdf_data: PaperExtractedData
