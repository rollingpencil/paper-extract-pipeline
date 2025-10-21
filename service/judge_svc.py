import re
from typing import Any, Dict, List
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from utils.utils import get_envvar
from utils.logger import log

llm_model = OpenAIChatModel(
    get_envvar("JUDGE_MODEL_NAME"),
    provider=OpenAIProvider(
        base_url=get_envvar("OPENAI_COMPAT_API_ENDPOINT"),
        api_key=get_envvar("OPENAI_COMPAT_API_KEY"),
    ),
)
judge_agent = Agent(llm_model, system_prompt="You are an objective evaluator.")

@judge_agent.tool
def groundedness_check(ctx: RunContext[None], system_answer: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count sentences supported by any evidence snippet (simple substring heuristic)."""
    sents = [s.strip() for s in re.split(r'[.!?]\s*', system_answer) if s.strip()]
    supported, unsupported = 0, []
    for s in sents:
        s_l = s.lower()
        if any(s_l in (ev.get("text","").lower()) for ev in evidence):
            supported += 1
        else:
            unsupported.append(s[:120])
    total = len(sents) or 1
    return {"supported_claims": supported, "total_claims": total, "grounded_ratio": supported / total, "unsupported_examples": unsupported[:3]}

@judge_agent.tool
def relevance_check(ctx: RunContext[None], query_text: str, system_answer: str) -> Dict[str, Any]:
    """Simple token-overlap relevance score (0-1) and short reason."""
    q = set(w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 2)
    a = set(w for w in re.findall(r"\w+", system_answer.lower()) if len(w) > 2)
    if not q:
        return {"score": 0.0, "reason": "no query tokens"}
    inter = q & a
    score = len(inter) / len(q)
    return {"score": round(score, 3), "reason": f"{len(inter)}/{len(q)} query tokens present"}

@judge_agent.tool
def completeness_check(ctx: RunContext[None], query_text: str, system_answer: str) -> Dict[str, Any]:
    """Heuristic completeness: fraction of query keywords present; list missing keywords."""
    q = [w for w in re.findall(r"\w+", query_text.lower()) if len(w) > 3]
    a = set(w for w in re.findall(r"\w+", system_answer.lower()) if len(w) > 3)
    if not q:
        return {"score": 0.0, "missing": []}
    missing = [w for w in q if w not in a]
    score = 1 - (len(missing) / len(set(q)))
    return {"score": round(max(0.0, score), 3), "missing": missing[:6]}

async def evaluate_response(query_text: str, system_answer: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run the three simple tools and return combined result."""
    try:
        g = groundedness_check(None, system_answer, evidence)
        r = relevance_check(None, query_text, system_answer)
        c = completeness_check(None, query_text, system_answer)
        out = {"groundedness": g, "relevance": r, "completeness": c}
        log.info(f"Judge result: grounded={g['grounded_ratio']:.3f} rel={r['score']:.3f} comp={c['score']:.3f}")
        return out
    except Exception as e:
        log.debug(f"evaluate_response failed: {e}")
        return {"error": str(e)}