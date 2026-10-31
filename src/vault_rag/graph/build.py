"""Directed link graph between notes: an edge A -> B means note A wikilinks to note B."""

import json
from pathlib import Path

import networkx as nx

from vault_rag.ingest.parse import resolve_link, wikilinks


def build_graph(notes: dict[str, str]) -> nx.DiGraph:
    """Nodes are vault-relative paths with `.md`, matching chunk payloads and the golden set."""
    ids = {path.removesuffix(".md") for path in notes}
    graph = nx.DiGraph()
    graph.add_nodes_from(notes)
    for path, text in notes.items():
        source = path.removesuffix(".md")
        for target in wikilinks(text):
            resolved = resolve_link(target, source, ids)
            if resolved and resolved != source:
                graph.add_edge(path, f"{resolved}.md")
    return graph


def save_graph(graph: nx.DiGraph, path: Path) -> None:
    path.write_text(json.dumps(nx.node_link_data(graph, edges="edges"), indent=1))


def load_graph(path: Path) -> nx.DiGraph:
    return nx.node_link_graph(json.loads(path.read_text()), directed=True, edges="edges")
