from datetime import datetime
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field

EmbeddingVector: TypeAlias = list[float]


class PaperMetadata(BaseModel):
    id: str
    title: str
    authors: list[str]
    date_published: datetime
    date_updated: datetime
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


class QueryAnswerPair(BaseModel):
    query: Annotated[str, Field(min_length=1)]
    expected_answer: Annotated[str, Field(min_length=1)]
    actual_reasoning: Annotated[
        str,
        Field(
            description="The reasoning process including traversal hops in format: 'Node1 (Type)' -> 'Node2 (Type)' -> 'Node3 (Type)'",
        ),
    ]
    actual_answer: Annotated[
        str, Field(min_length=1, description="Natural language answer to the question")
    ]


class GroundednessCheckModel(BaseModel):
    support_claims: Annotated[
        int, Field(description="Number of sentences supported by any evidence snippet")
    ]
    total_claims: Annotated[int, Field(description="Total number of sentences")]
    grounded_ratio: Annotated[
        float, Field(description="Ratio of grounded claims to total claims")
    ]
    unsupported_examples: Annotated[
        list[str], Field(description="List of unsupported sentences from the answer")
    ]


class RelevanceCheckModel(BaseModel):
    score: Annotated[float, Field(ge=0, le=1, description="Relevance score")]
    reasoning: Annotated[
        str,
        Field(
            description="Reasoning for the relevance score",
        ),
    ]


class CompletenessCheckModel(BaseModel):
    score: Annotated[float, Field(ge=0, le=1, description="Relevance score")]
    missing: Annotated[list[str], Field(description="List of missing keywords")]


class QAEvaluationModel(BaseModel):
    groundedness_check: GroundednessCheckModel
    relevance_check: RelevanceCheckModel
    completeness_check: CompletenessCheckModel


class QAResultModel(BaseModel):
    qapair: QueryAnswerPair
    evaluation: QAEvaluationModel
