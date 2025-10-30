import re
from typing import Any, Dict, List

from fastapi import HTTPException
from pydantic_ai import Agent, ModelHTTPError, NativeOutput, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models.models import (
    AccuracyCheckModel,
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
    system_prompt="""You are an objective evaluator that evaluates the quality of a response to a question.

You have FOUR tools available: groundedness_check, relevance_check, completeness_check, and accuracy_check.

CRITICAL INSTRUCTIONS:
1. You MUST call ALL FOUR tools
2. After calling all four tools, return a JSON object with EXACTLY this structure:
{
    "groundedness_check": {
        "support_claims": <number>,
        "total_claims": <number>,
        "grounded_ratio": <float between 0-1>,
        "unsupported_examples": [<list of strings>]
    },
    "relevance_check": {
        "score": <float between 0-1>,
        "reasoning": "<string>"
    },
    "completeness_check": {
        "score": <float between 0-1>,
        "missing": [<list of strings>]
    },
    "accuracy_check": {
        "is_accurate": <boolean>,
        "reasoning": "<string>"
    }
}

Do NOT add any extra fields. Do NOT add explanatory text outside the JSON structure. Simply call all four tools and populate the JSON with their results.""",
    output_type=NativeOutput(QAEvaluationModel),
    retries=3,
)


@judge_agent.tool
def groundedness_check(
    ctx: RunContext[None], system_answer: str, evidence: List[Dict[str, Any]]
) -> GroundednessCheckModel:
    """Count sentences supported by any evidence snippet (simple substring heuristic)."""
    log.info(
        f"groundedness_check tool called with system_answer: {system_answer[:100]}..."
    )
    log.info(f"evidence count: {len(evidence)}")
    sents = [s.strip() for s in re.split(r"[.!?]\s*", system_answer) if s.strip()]
    supported, unsupported = 0, []
    for s in sents:
        s_l = s.lower()
        if any(s_l in (ev.get("text", "").lower()) for ev in evidence):
            supported += 1
        else:
            unsupported.append(s[:120])
    total = len(sents) or 1
    result = GroundednessCheckModel(
        support_claims=supported,
        total_claims=total,
        grounded_ratio=supported / total,
        unsupported_examples=unsupported[:3],
    )
    log.info(f"groundedness_check returning: {result}")
    return result


@judge_agent.tool
def relevance_check(
    ctx: RunContext[None], query_text: str, system_answer: str
) -> RelevanceCheckModel:
    """Simple token-overlap relevance score (0-1) and short reason."""
    log.info(f"relevance_check tool called with query_text: {query_text}")
    log.info(f"system_answer: {system_answer[:100]}...")
    q = set(w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 2)
    a = set(w for w in re.findall(r"\w+", system_answer.lower()) if len(w) > 2)
    if not q:
        result = RelevanceCheckModel(score=0.00, reasoning="no query tokens")
        log.info(f"relevance_check returning (no query tokens): {result}")
        return result
    inter = q & a
    score = len(inter) / len(q)
    result = RelevanceCheckModel(
        score=round(score, 3), reasoning=f"{len(inter)}/{len(q)} query tokens present"
    )
    log.info(f"relevance_check returning: {result}")
    return result


@judge_agent.tool
def completeness_check(
    ctx: RunContext[None], query_text: str, system_answer: str
) -> CompletenessCheckModel:
    """Heuristic completeness: fraction of query keywords present; list missing keywords."""
    log.info(f"completeness_check tool called with query_text: {query_text}")
    log.info(f"system_answer: {system_answer[:100]}...")
    q = [w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 3]
    a = set(w for w in re.findall(r"\w+", system_answer.lower()) if len(w) > 3)
    if not q:
        result = CompletenessCheckModel(score=0.00, missing=[])
        log.info(f"completeness_check returning (no query tokens): {result}")
        return result
    missing = [w for w in q if w not in a]
    score = 1 - (len(missing) / len(set(q)))
    result = CompletenessCheckModel(score=round(score, 3), missing=missing[:6])
    log.info(f"completeness_check returning: {result}")
    return result


@judge_agent.tool
def accuracy_check(
    ctx: RunContext[None], expected_answer: str, actual_answer: str
) -> AccuracyCheckModel:
    """Check if the actual answer matches the expected answer semantically."""
    log.info(f"accuracy_check tool called")
    log.info(f"expected_answer: {expected_answer[:100]}...")
    log.info(f"actual_answer: {actual_answer[:100]}...")

    # Simple heuristic: check token overlap
    exp_tokens = set(w for w in re.findall(r"\w+", expected_answer.lower()) if len(w) > 2)
    act_tokens = set(w for w in re.findall(r"\w+", actual_answer.lower()) if len(w) > 2)

    if not exp_tokens:
        result = AccuracyCheckModel(
            is_accurate=False,
            reasoning="Expected answer has no valid tokens"
        )
        log.info(f"accuracy_check returning (no expected tokens): {result}")
        return result

    overlap = exp_tokens & act_tokens
    overlap_ratio = len(overlap) / len(exp_tokens)

    # Consider accurate if >60% token overlap
    is_accurate = overlap_ratio > 0.6
    reasoning = f"Token overlap: {len(overlap)}/{len(exp_tokens)} ({overlap_ratio:.1%})"

    result = AccuracyCheckModel(is_accurate=is_accurate, reasoning=reasoning)
    log.info(f"accuracy_check returning: {result}")
    return result


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
    log.info(f"Starting evaluation for query: {qa_pair.query}")
    log.info(f"Expected answer: {qa_pair.expected_answer}")
    log.info(f"Actual answer: {qa_pair.actual_answer}")
    log.info(f"Actual reasoning: {qa_pair.actual_reasoning}")

    # Construct a comprehensive prompt with all evaluation data
    prompt = f"""Please evaluate the quality of the following response to a question using all four available tools:

QUESTION: {qa_pair.query}

EXPECTED ANSWER: {qa_pair.expected_answer}

SYSTEM ANSWER: {qa_pair.actual_answer}

SYSTEM REASONING: {qa_pair.actual_reasoning}

Your task:
1. Use the groundedness_check tool to evaluate how well the system answer is supported by evidence
2. Use the relevance_check tool to evaluate how relevant the system answer is to the question
3. Use the completeness_check tool to evaluate how complete the system answer is
4. Use the accuracy_check tool to compare the system answer against the expected answer

For the groundedness_check tool, since no specific evidence is provided, use a placeholder empty list for the evidence parameter.

Please call all four tools to complete the evaluation."""

    try:
        log.info("Calling judge agent with proper prompt...")
        result = await judge_agent.run(prompt)
        evaluation = result.output
        log.info(f"Agent returned evaluation: {evaluation}")
        log.info(
            f"Judge result: grounded={evaluation.groundedness_check.grounded_ratio:.3f} | "
            f"rel={evaluation.relevance_check.score:.3f} | "
            f"comp={evaluation.completeness_check.score:.3f} | "
            f"accurate={evaluation.accuracy_check.is_accurate}"
        )
    except ModelHTTPError as e:
        log.debug(e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error(f"Unexpected error in evaluate_response: {e}")
        log.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    return result.output
