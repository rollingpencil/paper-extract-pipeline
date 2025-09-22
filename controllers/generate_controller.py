import networkx as nx
import matplotlib.pyplot as plt

from controllers.fetch_controller import (
    retrievePaperMetadata,
    retrievePaperDatasetList
)

from service.arxiv_svc import (
    fetch_document_id_by_topic,
)
from utils.constants import SourceType

def retrieveDocumentIds(topic: str, num_papers: int) -> list:
    ids = fetch_document_id_by_topic(topic, num_papers)
    for id in ids:
        id["source"] = SourceType.ARXIV

    return ids

async def generate_ontology_graph(topic: str, num_papers: int):
    data = {}
    ids =  retrieveDocumentIds(topic, num_papers)

    # for id in ids:
    #     document_data = {}
        
    #     document_data["metadata"] = retrievePaperMetadata(id["source"], id["id"])["content"]
    #     document_data["pdf_data"]  = await retrievePaperDatasetList(id["pdf_link"])

    #     data[id["id"]] = document_data
    
    plot_nodes(data)

def plot_nodes(data: dict):
    color_map = {
        "paper": "lightgray",
        "author": "pink",
        "dataset": "skyblue",
        "model": "lightgreen",
        "method": "salmon"
    }

    data = {
        "document 1" : {
            "metadata" : {
                "authors" : ["author2", "author3"]
            },
            "pdf_data": {
                 "datasets": [
                    {
                    "name": "Podcast Transcripts Dataset",
                    "description": "Corpus of public transcripts from the Microsoft 'Behind the Tech' podcast featuring conversations with thought leaders, totaling approximately 1 million tokens."
                    },
                    {
                    "name": "News Articles Dataset",
                    "description": "Corpus of news articles published from September 2013 to December 2023 across categories such as entertainment, business, sports, technology, health, and science, totaling around 1.7 million tokens."
                    },
                    {
                    "name": "MultiHop-RAG Benchmark Dataset",
                    "description": "Reference dataset used in the paper for demonstrating community detection in graph construction, containing multi-hop question-answer pairs."
                    }
                ],
                "models": [
                    {
                    "name": "GPT-4",
                    "description": "Large language model developed by OpenAI used for knowledge extraction and answer generation in the paper."
                    },
                    {
                    "name": "GPT-4 Turbo",
                    "description": "Optimized version of GPT‑4 used for LLM inference during graph indexing and evaluation."
                    },
                    {
                    "name": "Llama 2",
                    "description": "Open‑source large language model referenced as a potential LLM in the work."
                    },
                    {
                    "name": "Gemini",
                    "description": "Large multimodal language model from Google used as an LLM reference in the study."
                    },
                    {
                    "name": "ChatGPT",
                    "description": "User‑facing LLM service built on GPT‑4, employed as the primary inference model."
                    },
                    {
                    "name": "G‑Retriever",
                    "description": "Retrieval‑augmented generation model that implements a graph‑aware retrieval strategy, used for comparative evaluation."
                    }
                ],
                "methods": [
                    {
                    "name": "GraphRAG",
                    "description": "Graph‑based Retrieval-Augmented Generation combining knowledge‑graph extraction with query‑focused summarization to answer global sense‑making queries."
                    },
                    {
                    "name": "Entity & Relationship Extraction",
                    "description": "LLM prompting to identify named entities and their relationships, producing graph nodes and edges."
                    },
                    {
                    "name": "Claim Extraction",
                    "description": "Detection of factual claims within text to provide verifiable statements for index creation and evaluation."
                    },
                    {
                    "name": "Leiden Community Detection",
                    "description": "Hierarchical modular community detection algorithm applied to the knowledge graph."
                    },
                    {
                    "name": "Community Summarization",
                    "description": "Hierarchical summarization of community nodes, edges, and claims, producing concise reports at multiple granularity levels."
                    },
                    {
                    "name": "Map‑Reduce Summarization",
                    "description": "Parallel generation of partial answers from community summaries, followed by aggregation into a global answer."
                    },
                    {
                    "name": "Vector RAG",
                    "description": "Standard retrieval‑augmented generation using text embeddings to return semantically similar chunks."
                    },
                    {
                    "name": "Chunking with Overlap",
                    "description": "Splitting source documents into token‑sized chunks with overlap to enable local information extraction."
                    },
                    {
                    "name": "Self‑Reflection Prompting",
                    "description": "Iterative LLM prompting to verify and augment entity extraction, improving recall."
                    },
                    {
                    "name": "Adaptive Benchmarking for RAG",
                    "description": "Generating domain‑specific global sense‑making questions via persona and task prompts for evaluation."
                    },
                    {
                    "name": "LLM‑as‑a‑Judge Evaluation",
                    "description": "Using an LLM to compare two system responses according to predefined criteria (comprehensiveness, diversity, directness, empowerment)."
                    }
                ]
                }
            },
        "document 2": {
            "metadata" : {
                "authors" : ["author1", "author2"]
            },
            "pdf_data": {
                 "datasets": [
                    {
                    "name": "Podcast Transcripts Dataset",
                    "description": "Corpus of public transcripts from the Microsoft 'Behind the Tech' podcast featuring conversations with thought leaders, totaling approximately 1 million tokens."
                    },
                    {
                    "name": "News Articles Dataset",
                    "description": "Corpus of news articles published from September 2013 to December 2023 across categories such as entertainment, business, sports, technology, health, and science, totaling around 1.7 million tokens."
                    },
                    {
                    "name": "MultiHop-RAG Benchmark Dataset",
                    "description": "Reference dataset used in the paper for demonstrating community detection in graph construction, containing multi-hop question-answer pairs."
                    }
                ],
                "models": [
                    {
                    "name": "GPT-4",
                    "description": "Large language model developed by OpenAI used for knowledge extraction and answer generation in the paper."
                    },
                    {
                    "name": "Gemini",
                    "description": "Large multimodal language model from Google used as an LLM reference in the study."
                    },
                    {
                    "name": "ChatGPT",
                    "description": "User‑facing LLM service built on GPT‑4, employed as the primary inference model."
                    },
                    {
                    "name": "G‑Retriever",
                    "description": "Retrieval‑augmented generation model that implements a graph‑aware retrieval strategy, used for comparative evaluation."
                    }
                ],
                "methods": [
                    {
                    "name": "GraphRAG",
                    "description": "Graph‑based Retrieval-Augmented Generation combining knowledge‑graph extraction with query‑focused summarization to answer global sense‑making queries."
                    },
                    {
                    "name": "Entity & Relationship Extraction",
                    "description": "LLM prompting to identify named entities and their relationships, producing graph nodes and edges."
                    },
                    {
                    "name": "Claim Extraction",
                    "description": "Detection of factual claims within text to provide verifiable statements for index creation and evaluation."
                    },
                    {
                    "name": "Leiden Community Detection",
                    "description": "Hierarchical modular community detection algorithm applied to the knowledge graph."
                    },
                    {
                    "name": "Community Summarization",
                    "description": "Hierarchical summarization of community nodes, edges, and claims, producing concise reports at multiple granularity levels."
                    },
                    {
                    "name": "Map‑Reduce Summarization",
                    "description": "Parallel generation of partial answers from community summaries, followed by aggregation into a global answer."
                    },
                    {
                    "name": "Adaptive Benchmarking for RAG",
                    "description": "Generating domain‑specific global sense‑making questions via persona and task prompts for evaluation."
                    },
                ]
                }
        }
    }
    graph = nx.Graph()

    for document_id, document_data in data.items():

        # Add the document node
        graph.add_node(document_id, type = "paper")

        # Add the author nodes and link it to the current paper
        for author in document_data["metadata"]['authors']:
            graph.add_node(author, type = "author")
            graph.add_edge(document_id, author, relation = "written by")
        
        # Add the database nodes and link it to the current paper
        for database in document_data["pdf_data"]['datasets']:
            db_name = database["name"].lower()
            graph.add_node(db_name, type = "dataset")
            graph.add_edge(document_id, db_name, relation = "dataset trained on")
        
        # Add the model nodes and link it to the current paper
        for model in document_data["pdf_data"]['models']:
            model_name = model["name"].lower()
            graph.add_node(model_name, type = "model")
            graph.add_edge(document_id, model_name, relation = "model used")
        
        # Add the model nodes and link it to the current paper
        for method in document_data["pdf_data"]['methods']:
            method_name = method["name"].lower()
            graph.add_node(method_name, type = "method")
            graph.add_edge(document_id, method_name, relation = "method used")
    
    # Extract colors based on node attribute
    node_colors = [color_map[graph.nodes[n].get("type", "paper")] for n in graph.nodes]

    plt.figure(figsize=(12, 12))  # Bigger canvas
    pos = nx.spring_layout(graph, k=0.6, iterations=100)  # Spread nodes more
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=1200, font_size=6)
    plt.savefig("graph.png")