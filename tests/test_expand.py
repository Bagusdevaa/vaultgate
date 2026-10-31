import networkx as nx
from qdrant_client import QdrantClient

from vault_rag import store
from vault_rag.graph.expand import expand, link_kind, pick_neighbors, seed_notes
from vault_rag.retrieve.embed import Embedding
from vault_rag.trace import Candidate


def cand(note: str, n: int = 0) -> Candidate:
    return Candidate(chunk_id=f"{note}#{n}", note=note, heading_path=[], text="", source="hybrid")


def star() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from([("P", "B"), ("P", "D"), ("P", "K"), ("K2", "P"), ("Q", "B")])
    return graph


def test_seed_notes_are_distinct_and_ordered():
    assert seed_notes([cand("a"), cand("a", 1), cand("b"), cand("c")], 2) == ["a", "b"]


def test_neighbors_rank_by_similarity_and_respect_cap():
    best = {"B": ("B#0", 0.9), "D": ("D#0", 0.5), "K": ("K#0", 0.7), "K2": ("K2#0", 0.8)}
    # K2 is a backlink of P, and still counts as a neighbor.
    assert pick_neighbors(star(), ["P"], best, 2, set()) == [("P", "B"), ("P", "K2")]


def test_duplicates_are_skipped_without_using_the_cap():
    best = {"B": ("B#0", 0.9), "D": ("D#0", 0.5), "K": ("K#0", 0.7), "K2": ("K2#0", 0.8)}
    picked = pick_neighbors(star(), ["P", "Q"], best, 2, known_chunks={"B#0"})
    assert picked == [("P", "K2"), ("P", "K")]  # Q's only neighbor B is already a candidate


def test_expand_adds_best_chunk_marked_as_graph():
    client = QdrantClient(":memory:")
    store.reset_collection(client, "vault")
    vec = lambda i: [1.0 if j == i else 0.0 for j in range(store.DENSE_DIM)]  # noqa: E731
    rows = [("P", 0, vec(0)), ("B", 0, vec(5)), ("B", 1, vec(0)), ("D", 0, vec(6))]
    payloads = [{"chunk_id": f"{n}#{i}", "note": n, "heading_path": [n], "text": f"{n}{i}"}
                for n, i, _ in rows]
    store.upsert_chunks(client, "vault", payloads, [Embedding(v, {1: 1.0}) for *_, v in rows])

    added = expand(client, "vault", star(), [cand("P")], vec(0), seed_count=5, cap=5)
    by_note = {c.note: c for c in added}
    assert by_note["B"].chunk_id == "B#1"  # the chunk closest to the query, not the first one
    assert by_note["B"].source == "graph" and by_note["B"].via == "P"
    assert by_note["B"].via_link == "link"
    assert set(by_note) == {"B", "D"}


def test_link_kind_names_the_direction():
    graph = star()
    graph.add_edge("B", "P")
    assert link_kind(graph, "P", "D") == "link"
    assert link_kind(graph, "P", "K2") == "backlink"
    assert link_kind(graph, "P", "B") == "both"
