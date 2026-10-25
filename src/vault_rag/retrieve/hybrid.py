"""Hybrid search: dense and sparse top-k from Qdrant, fused with reciprocal rank fusion.

Fusion happens here rather than in Qdrant so each candidate keeps its rank in both lists
for the trace. The order matches Qdrant's own RRF with the same k; its scores are
slightly higher because it counts ranks from 0 (see tests).
"""

from collections import defaultdict

from qdrant_client import QdrantClient, models

from vault_rag.retrieve.embed import Embedding
from vault_rag.store import DENSE, SPARSE
from vault_rag.trace import Candidate

FILTER_FIELDS = ("folder", "tags", "device")


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """score(item) = sum over lists of 1 / (k + rank), rank starting at 1.

    Ties keep the order in which items first appear, scanning the lists in turn.
    """
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1 / (k + rank)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def build_filter(filters: dict | None) -> models.Filter | None:
    """{"folder": ["Bugs"], "tags": ["pbpipe"]}: every field must match one of its values."""
    if not filters:
        return None
    unknown = set(filters) - set(FILTER_FIELDS)
    if unknown:
        raise ValueError(f"unknown filter fields: {sorted(unknown)}")
    must = []
    for key, values in filters.items():
        values = [values] if isinstance(values, str) else list(values)
        must.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
    return models.Filter(must=must)


def sparse_vector(embedding: Embedding) -> models.SparseVector:
    return models.SparseVector(indices=list(embedding.sparse), values=list(embedding.sparse.values()))


def hybrid_search(client: QdrantClient, collection: str, query: Embedding, top_k: int,
                  rrf_k: int, filters: dict | None = None) -> list[Candidate]:
    flt = build_filter(filters)
    dense, sparse = client.query_batch_points(collection, requests=[
        models.QueryRequest(query=query.dense, using=DENSE, filter=flt, limit=top_k, with_payload=True),
        models.QueryRequest(query=sparse_vector(query), using=SPARSE, filter=flt, limit=top_k,
                            with_payload=True),
    ])
    payloads, dense_hits, sparse_hits = {}, {}, {}
    for hits, response in ((dense_hits, dense), (sparse_hits, sparse)):
        for rank, point in enumerate(response.points, start=1):
            payloads[point.payload["chunk_id"]] = point.payload
            hits[point.payload["chunk_id"]] = (rank, point.score)

    candidates = []
    for chunk_id, score in rrf([list(dense_hits), list(sparse_hits)], rrf_k)[:top_k]:
        payload = payloads[chunk_id]
        d_rank, d_score = dense_hits.get(chunk_id, (None, None))
        s_rank, s_score = sparse_hits.get(chunk_id, (None, None))
        candidates.append(Candidate(
            chunk_id=chunk_id, note=payload["note"], heading_path=payload["heading_path"],
            text=payload["text"], source="hybrid", dense_rank=d_rank, dense_score=d_score,
            sparse_rank=s_rank, sparse_score=s_score, rrf_score=score))
    return candidates
