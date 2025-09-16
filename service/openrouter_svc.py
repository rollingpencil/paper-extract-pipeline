import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

DATA_EXTRACTION_SYSTEM_PROMPT = "You are a highly skilled data extraction specialist. Your task is to extract datasets, techniques/methods covered as well as models used from research papers. Methods refer to techniques or approaches used in the research, while models refer to specific implementations or algorithms. Datasets refer to collections of data or benchmarks used for training or evaluation."


class ExtractionOutput(BaseModel):
    name: str
    description: str


gpt_oss_model = OpenAIChatModel(
    # "openai/gpt-oss-20b:free",
    "deepseek/deepseek-chat-v3.1:free",
    provider=OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")),
)

data_extraction_agent = Agent(
    gpt_oss_model,
    system_prompt=DATA_EXTRACTION_SYSTEM_PROMPT,
    output_type=PromptedOutput(list[ExtractionOutput]),
)


async def _extract_paper_content(
    prompt: str, error_message: str
) -> list[ExtractionOutput]:
    result = await data_extraction_agent.run(prompt)
    return result.output


async def extract_paper_dataset(paper_text: str) -> list[ExtractionOutput]:
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nExtract datasets and benchmarks used for training or evaluation in the paper."
    return await _extract_paper_content(
        prompt, "Failed to parse extracted datasets as JSON."
    )


async def extract_paper_models(paper_text: str) -> list[ExtractionOutput]:
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nExtract the models referenced, such as language models, rerank models, embed models, models that implements a technique or models used for comparison, in the paper. Exclude methods, benchmarks and framework."
    return await _extract_paper_content(
        prompt, "Failed to parse extracted models as JSON."
    )


async def extract_paper_methods(paper_text: str) -> list[ExtractionOutput]:
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nExtract the terms of techniques used in the paper. Exclude language models, rerank models, embed models."
    return await _extract_paper_content(
        prompt, "Failed to parse extracted models as JSON."
    )
