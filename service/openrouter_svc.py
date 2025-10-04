import os
import time

from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic_ai import Agent, ModelHTTPError, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models.models import NodeRecord
from service.embed_svc import embed_content

load_dotenv()

DATA_EXTRACTION_SYSTEM_PROMPT = "You are a highly skilled data extraction specialist. Your task is to extract datasets, techniques/methods covered as well as models used from research papers. Methods refer to techniques or approaches used in the research, while models refer to specific implementations or algorithms. Datasets refer to collections of data or benchmarks used for training or evaluation."


class ExtractionOutput(BaseModel):
    name: str
    description: str


llm_model = OpenAIChatModel(
    os.getenv("MODEL_NAME") or "",
    provider=OpenAIProvider(
        base_url=os.getenv("OPENAI_COMPAT_API_ENDPOINT"),
        api_key=os.getenv("OPENAI_COMPAT_API_KEY"),
    ),
)

data_extraction_agent = Agent(
    llm_model,
    system_prompt=DATA_EXTRACTION_SYSTEM_PROMPT,
    output_type=PromptedOutput(list[ExtractionOutput]),
)


async def _extract_paper_content(prompt: str, error_message: str) -> list[NodeRecord]:
    time_start = time.perf_counter()

    try:
        result = await data_extraction_agent.run(prompt)
    except ModelHTTPError:
        raise HTTPException(status_code=500, detail=error_message)

    extracted_term = result.output
    extracted_term_with_embed = []
    for term in extracted_term:
        term_embed = await embed_content(term.name)
        extracted_term_with_embed.append(
            NodeRecord(
                title=term.name, description=term.description, embedding=term_embed
            )
        )
    print()
    time_end = time.perf_counter()
    print(f"Completed in {time_end - time_start:.2f} seconds")

    return extracted_term_with_embed


async def extract_paper_dataset(paper_text: str) -> list[NodeRecord]:
    print("Extracting datasets")
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nExtract datasets and benchmarks used for training or evaluation in the paper."
    return await _extract_paper_content(
        prompt, "Failed to parse extracted datasets as JSON."
    )


async def extract_paper_models(paper_text: str) -> list[NodeRecord]:
    print("Extracting models")
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nExtract the models referenced, such as language models, rerank models, embed models, models that implements a technique or models used for comparison, in the paper. Exclude methods, benchmarks and framework."
    return await _extract_paper_content(
        prompt, "Failed to parse extracted models as JSON."
    )


async def extract_paper_methods(paper_text: str) -> list[NodeRecord]:
    print("Extracting methods")
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nExtract the terms of techniques used in the paper. Exclude language models, rerank models, embed models."
    return await _extract_paper_content(
        prompt, "Failed to parse extracted methods as JSON."
    )


async def extract_paper_tasking(paper_text: str) -> list[NodeRecord]:
    print("Extracting tasking")
    prompt = f"Given the following academic paper text:\n\n{paper_text}\n\nExtract the tasks/use cases covered in the paper."
    return await _extract_paper_content(
        prompt, "Failed to parse extracted tasking as JSON."
    )
