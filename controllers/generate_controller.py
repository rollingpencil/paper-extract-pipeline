import matplotlib.pyplot as plt
import networkx as nx

from controllers.fetch_controller import (
    retrievePaper,
)
from service.arxiv_svc import (
    fetch_document_id_by_topic,
)
from utils.constants import SourceType


def _retrieveDocumentIds(topic: str, num_papers: int) -> list:
    ids = fetch_document_id_by_topic(topic, num_papers)
    for id in ids:
        id["source"] = SourceType.ARXIV

    return ids


async def generate_ontology_graph(topic: str, num_papers: int):
    data = {}
    ids = _retrieveDocumentIds(topic, num_papers)
    print(f"Processing ids: {[id["id"] for id in ids]}")

    for id in ids:
        print(f"Processing paper with id: {id["id"]}")
        data[id["id"]] = await retrievePaper(id["source"], id["id"])

    _plot_nodes(data)


def _plot_nodes(data: dict):
    color_map = {
        "paper": "lightgray",
        "author": "pink",
        "dataset": "skyblue",
        "model": "lightgreen",
        "method": "salmon",
        "tasking": "purple"
    }

    graph = nx.Graph()

    for document_id, document_data in data.items():
        # Add the document node
        graph.add_node(document_id, type="paper")

        # Add the author nodes and link it to the current paper
        for author in document_data.metadata.authors:
            graph.add_node(author, type="author")
            graph.add_edge(document_id, author, relation="written by")

        # Add the database nodes and link it to the current paper
        for dataset in document_data.pdf_data.datasets:
            ds_name = dataset.title.lower()
            graph.add_node(ds_name, type="dataset")
            graph.add_edge(document_id, ds_name, relation="dataset trained on")

        # Add the model nodes and link it to the current paper
        for model in document_data.pdf_data.models:
            model_name = model.title.lower()
            graph.add_node(model_name, type="model")
            graph.add_edge(document_id, model_name, relation="model used")

        # Add the model nodes and link it to the current paper
        for method in document_data.pdf_data.methods:
            method_name = method.title.lower()
            graph.add_node(method_name, type="method")
            graph.add_edge(document_id, method_name, relation="method used")

        # Add the tasking nodes and link it to the current paper
        for task in document_data.pdf_data.tasking:
            task_name = task.title.lower()
            graph.add_node(task_name, type="tasking")
            graph.add_edge(document_id, task_name, relation="tasking performed")

    # Extract colors based on node attribute
    node_colors = [color_map[graph.nodes[n].get("type", "paper")] for n in graph.nodes]

    plt.figure(figsize=(15, 15))  # Bigger canvas
    pos = nx.spring_layout(graph, k=0.7, iterations=100)  # Spread nodes more
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=1200,
        font_size=8,
    )
    plt.savefig("graph.png")
