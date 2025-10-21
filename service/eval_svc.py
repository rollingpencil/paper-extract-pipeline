from pydantic import BaseModel
from service.embed_svc import embed_content
from math import sqrt
from typing import List, Dict, Any, Optional

from service.judge_svc import evaluate_response


class EvaluationOutput(BaseModel):
    relevance_score: str
    judge: Optional[Dict[str, Any]] = None


def _cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python).
    Returns 0.0 on error or zero-length vectors."""
    try:
        dot = sum(x * y for x, y in zip(a, b))
        na = sqrt(sum(x * x for x in a))
        nb = sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0
    except Exception:
        return 0.0


async def evaluate_query_relevance(query_text: str, answer_embeddings: list[list[float]]) -> EvaluationOutput:
    """Embed query, compute cosine vs candidates, return best score as string."""
    q_emb = await embed_content(query_text)
    best = 0.0
    for emb in answer_embeddings:
        score = _cosine(q_emb, emb)
        if score > best:
            best = score
    return EvaluationOutput(relevance_score=f"{best:.6f}")




async def evaluate_with_judge(query_text: str, system_answer: str, evidence: List[Dict[str, Any]]) -> EvaluationOutput:
    """Call judge_svc.evaluate_response and package result into EvaluationOutput."""
    judge_res = await evaluate_response(query_text, system_answer, evidence)
    rel = judge_res.get("relevance", {}).get("score") if isinstance(judge_res, dict) else None
    rel_str = f"{rel:.6f}" if isinstance(rel, float) else str(rel or "0.0")
    return EvaluationOutput(relevance_score=rel_str, judge=judge_res)
