import pytest
from qdrant_client import QdrantClient, models

from vault_rag import store
from vault_rag.retrieve.embed import Embedding
from vault_rag.retrieve.hybrid import build_filter, hybrid_search, rrf


def test_rrf_scores_by_hand():
    fused = dict(rrf([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)  # found by only one list
    assert fused["c"] == pytest.approx(1 / 63)


def test_rrf_order_and_ties():
    assert [item for item, _ in rrf([["a", "b", "c"], ["b", "d"]], k=60)] == ["b", "a", "d", "c"]
    # a and x tie (rank 1 in one list each): the one seen first wins.
    assert [item for item, _ in rrf([["a"], ["x"]], k=60)] == ["a", "x"]


def test_filter_matches_any_value_per_field():
    flt = build_filter({"folder": "Bugs", "tags": ["a", "b"]})
    assert [c.key for c in flt.must] == ["folder", "tags"]
    assert flt.must[0].match.any == ["Bugs"]
    assert build_filter(None) is None
    with pytest.raises(ValueError):
        build_filter({"colour": "red"})


def unit(i: int) -> list[float]:
    vec = [0.0] * store.DENSE_DIM
    vec[i % 4] = 1.0
    vec[4 + i] = 0.5
    return vec


def make_index(n: int = 12) -> QdrantClient:
    client = QdrantClient(":memory:")
    store.reset_collection(client, "vault")
    payloads, embeddings = [], []
    for i in range(n):
        folder = "Bugs" if i % 2 else "Knowledge"
        payloads.append({"chunk_id": f"n{i}.md#0", "note": f"n{i}.md", "heading_path": [f"n{i}"],
                         "text": f"text {i}", "folder": folder, "tags": [], "device": "dev"})
        embeddings.append(Embedding(unit(i), {i: 1.0, 100 + i % 3: 0.3}))
    store.upsert_chunks(client, "vault", payloads, embeddings)
    return client


def test_matches_qdrant_server_side_rrf():
    client = make_index()
    query = Embedding(unit(1), {1: 1.0, 5: 0.8, 102: 0.4})
    ours = hybrid_search(client, "vault", query, top_k=6, rrf_k=60)
    sparse = models.SparseVector(indices=list(query.sparse), values=list(query.sparse.values()))
    theirs = client.query_points("vault", prefetch=[
        models.Prefetch(query=query.dense, using=store.DENSE, limit=6),
        models.Prefetch(query=sparse, using=store.SPARSE, limit=6),
    ], query=models.RrfQuery(rrf=models.Rrf(k=60)), limit=6, with_payload=True).points
    assert [c.chunk_id for c in ours] == [p.payload["chunk_id"] for p in theirs]
    # Qdrant counts ranks from 0, so its k=60 scores equal ours with k=59.
    shifted = hybrid_search(client, "vault", query, top_k=6, rrf_k=59)
    assert [c.rrf_score for c in shifted] == pytest.approx([p.score for p in theirs])


def test_candidates_carry_both_ranks_and_respect_filter():
    client = make_index()
    query = Embedding(unit(1), {1: 1.0})
    top = hybrid_search(client, "vault", query, top_k=4, rrf_k=60)[0]
    assert (top.chunk_id, top.dense_rank, top.sparse_rank, top.source) == ("n1.md#0", 1, 1, "hybrid")
    bugs = hybrid_search(client, "vault", query, top_k=10, rrf_k=60, filters={"folder": "Bugs"})
    assert bugs and all(int(c.note[1:-3]) % 2 for c in bugs)
