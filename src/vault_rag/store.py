"""Qdrant collection holding one point per chunk, with a dense and a sparse named vector."""

import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from vault_rag.retrieve.embed import Embedding

DENSE, SPARSE = "dense", "sparse"
DENSE_DIM = 1024  # bge-m3
UPSERT_BATCH = 64


def open_client(qdrant: dict, index_dir: Path) -> QdrantClient:
    """Qdrant server when `url` is set (Docker), otherwise local mode on disk."""
    if qdrant.get("url"):
        return QdrantClient(url=qdrant["url"])
    return QdrantClient(path=str(index_dir / "qdrant"))


def reset_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
    )


def point_id(chunk_id: str) -> str:
    """Stable UUID per chunk id, since Qdrant ids must be ints or UUIDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def upsert_chunks(client: QdrantClient, name: str, payloads: list[dict],
                  embeddings: list[Embedding]) -> None:
    points = [
        models.PointStruct(
            id=point_id(payload["chunk_id"]),
            vector={DENSE: emb.dense,
                    SPARSE: models.SparseVector(indices=list(emb.sparse), values=list(emb.sparse.values()))},
            payload=payload,
        )
        for payload, emb in zip(payloads, embeddings, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[start:start + UPSERT_BATCH])


def delete_note(client: QdrantClient, name: str, note: str) -> None:
    match = models.FieldCondition(key="note", match=models.MatchValue(value=note))
    client.delete(name, points_selector=models.FilterSelector(filter=models.Filter(must=[match])))


def count(client: QdrantClient, name: str) -> int:
    return client.count(name).count
