import json
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from models.exceptions import ExtractionError

load_dotenv()

DATA_EXTRACTION_SYSTEM_PROMPT = ""

gpt_oss_model = OpenAIChatModel(
    "openai/gpt-oss-20b:free",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")),
)

data_extraction_agent = Agent(
    gpt_oss_model,
    system_prompt=DATA_EXTRACTION_SYSTEM_PROMPT,
)


async def _extract_paper_content(prompt: str, error_message: str) -> str:
    result = await data_extraction_agent.run(prompt)
    trimmed_result = result.output.strip()
    if trimmed_result.startswith("```json"):
        trimmed_result = trimmed_result[len("```json") :].strip()
    if trimmed_result.endswith("```"):
        trimmed_result = trimmed_result[: -len("```")].strip()
    try:
        return json.loads(trimmed_result)
    except ValueError:
        raise ExtractionError(error_message)


async def extract_paper_dataset(paper_text: str) -> str:
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nextract the datasets used in the paper and output in the form of json list with each element containing its name and description\n\nList:"
    return await _extract_paper_content(
        prompt, "Failed to parse extracted datasets as JSON."
    )


async def extract_paper_models(paper_text: str) -> str:
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nextract the LLM models used, LLM models used for comparison in the paper and output in the form of json list with each element containing its name and description if provided\n\nList:"
    return await _extract_paper_content(
        prompt, "Failed to parse extracted models as JSON."
    )


async def extract_paper_methods(paper_text: str) -> str:
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nextract the terms of techniques used in the paper and output in the form of json list with each element containing its name and description if provided\n\nList:"
    return await _extract_paper_content(
        prompt, "Failed to parse extracted models as JSON."
    )
