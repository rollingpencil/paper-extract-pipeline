import re
from typing import Any, Dict, List

from fastapi import HTTPException
from pydantic_ai import Agent, ModelHTTPError, NativeOutput, PromptedOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models.models import (
    CompletenessCheckModel,
    GroundednessCheckModel,
    QAEvaluationModel,
    QueryAnswerPair,
    RelevanceCheckModel,
)
from utils.logger import log
from utils.utils import get_envvar

log.info(f"Judge Model set: {get_envvar('JUDGE_MODEL_NAME')}")
llm_model = OpenAIChatModel(
    get_envvar("JUDGE_MODEL_NAME"),
    provider=OpenAIProvider(
        base_url=get_envvar("OPENAI_COMPAT_API_ENDPOINT"),
        api_key=get_envvar("OPENAI_COMPAT_API_KEY"),
    ),
)
judge_agent = Agent(
    llm_model,
    system_prompt="You are an objective evaluator that evaluates the quality of a response to a question. You have three tools available: groundedness_check, completeness_check, and relevance_check. You must use all three tools to evaluate the response.",
    output_type=NativeOutput(QAEvaluationModel),
)


@judge_agent.tool
def groundedness_check(
    ctx: RunContext[None], system_answer: str, evidence: List[Dict[str, Any]]
) -> GroundednessCheckModel:
    """Count sentences supported by any evidence snippet (simple substring heuristic)."""
    sents = [s.strip() for s in re.split(r"[.!?]\s*", system_answer) if s.strip()]
    supported, unsupported = 0, []
    for s in sents:
        s_l = s.lower()
        if any(s_l in (ev.get("text", "").lower()) for ev in evidence):
            supported += 1
        else:
            unsupported.append(s[:120])
    total = len(sents) or 1
    return GroundednessCheckModel(
        support_claims=supported,
        total_claims=total,
        grounded_ratio=supported / total,
        unsupported_examples=unsupported[:3],
    )


@judge_agent.tool
def relevance_check(
    ctx: RunContext[None], query_text: str, system_answer: str
) -> RelevanceCheckModel:
    """Simple token-overlap relevance score (0-1) and short reason."""
    q = set(w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 2)
    a = set(w for w in re.findall(r"\w+", system_answer.lower()) if len(w) > 2)
    if not q:
        return RelevanceCheckModel(score=0.00, reasoning="no query tokens")
    inter = q & a
    score = len(inter) / len(q)
    return RelevanceCheckModel(
        score=round(score, 3), reasoning=f"{len(inter)}/{len(q)} query tokens present"
    )


@judge_agent.tool
def completeness_check(
    ctx: RunContext[None], query_text: str, system_answer: str
) -> CompletenessCheckModel:
    """Heuristic completeness: fraction of query keywords present; list missing keywords."""
    q = [w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 3]
    a = set(w for w in re.findall(r"\w+", system_answer.lower()) if len(w) > 3)
    if not q:
        return CompletenessCheckModel(score=0.00, missing=[])
    missing = [w for w in q if w not in a]
    score = 1 - (len(missing) / len(set(q)))
    return CompletenessCheckModel(score=round(score, 3), missing=missing[:6])


# async def evaluate_response(
#     query_text: str, system_answer: str, evidence: List[Dict[str, Any]]
# ) -> Dict[str, Any]:
#     """Run the three simple tools and return combined result."""
#     try:
#         g = groundedness_check(None, system_answer, evidence)
#         r = relevance_check(None, query_text, system_answer)
#         c = completeness_check(None, query_text, system_answer)
#         out = {"groundedness": g, "relevance": r, "completeness": c}
#         log.info(
#             f"Judge result: grounded={g['grounded_ratio']:.3f} rel={r['score']:.3f} comp={c['score']:.3f}"
#         )
#         return out
#     except Exception as e:
#         log.debug(f"evaluate_response failed: {e}")
#         return {"error": str(e)}


async def evaluate_response(qa_pair: QueryAnswerPair) -> QAEvaluationModel:
    prompt = ""
    try:
        result = await judge_agent.run(prompt)
        evaluation = result.output
        log.info(
            f"Judge result: grounded={evaluation.groundedness_check.grounded_ratio:.3f}\nrel={evaluation.relevance_check.score:.3f}\ncomp={evaluation.completeness_check.score:.3f}"
        )
    except ModelHTTPError as e:
        log.debug(e)
        raise HTTPException(status_code=500, detail=str(e))
    return result.output
