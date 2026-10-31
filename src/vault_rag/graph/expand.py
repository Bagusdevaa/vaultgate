"""1-hop graph expansion: add the best chunk of notes linked to or from the top hits."""

import networkx as nx
from qdrant_client import QdrantClient, models

from vault_rag.store import DENSE
from vault_rag.trace import Candidate


def seed_notes(candidates: list[Candidate], n: int) -> list[str]:
    """The first n distinct notes, in candidate order."""
    return list(dict.fromkeys(c.note for c in candidates))[:n]


def neighbors(graph: nx.DiGraph, note: str) -> set[str]:
    """Links and backlinks."""
    if note not in graph:
        return set()
    return set(graph.successors(note)) | set(graph.predecessors(note))


def link_kind(graph: nx.DiGraph, seed: str, note: str) -> str:
    """How `note` is connected to `seed`: the seed links to it, it links back, or both."""
    forward, backward = graph.has_edge(seed, note), graph.has_edge(note, seed)
    return "both" if forward and backward else "link" if forward else "backlink"


def pick_neighbors(graph: nx.DiGraph, seeds: list[str], best: dict[str, tuple[str, float]],
                   cap: int, known_chunks: set[str]) -> list[tuple[str, str]]:
    """(seed, neighbor) pairs, up to `cap` per seed, most query-similar neighbor first.

    `best` maps a note to its most query-similar chunk id and score. A neighbor whose best
    chunk is already a candidate, or was added from an earlier seed, is skipped and does
    not use up the cap.
    """
    picked, taken = [], set(known_chunks)
    for seed in seeds:
        ranked = sorted((n for n in neighbors(graph, seed) if n in best),
                        key=lambda n: (-best[n][1], n))
        added = 0
        for note in ranked:
            if added == cap:
                break
            if best[note][0] in taken:
                continue
            taken.add(best[note][0])
            picked.append((seed, note))
            added += 1
    return picked


def best_chunks(client: QdrantClient, collection: str, query_dense: list[float],
                notes: set[str]) -> dict[str, tuple[dict, float]]:
    """Each note's chunk closest to the query (dense), as payload and score."""
    if not notes:
        return {}
    only = models.Filter(must=[models.FieldCondition(key="note", match=models.MatchAny(any=sorted(notes)))])
    result = client.query_points_groups(collection, query=query_dense, using=DENSE, group_by="note",
                                        group_size=1, limit=len(notes), query_filter=only,
                                        with_payload=True)
    return {group.hits[0].payload["note"]: (group.hits[0].payload, group.hits[0].score)
            for group in result.groups}


def expand(client: QdrantClient, collection: str, graph: nx.DiGraph, candidates: list[Candidate],
           query_dense: list[float], seed_count: int, cap: int) -> list[Candidate]:
    seeds = seed_notes(candidates, seed_count)
    reachable = set().union(*(neighbors(graph, s) for s in seeds)) if seeds else set()
    best = best_chunks(client, collection, query_dense, reachable)
    scores = {note: (payload["chunk_id"], score) for note, (payload, score) in best.items()}
    known = {c.chunk_id for c in candidates}
    added = []
    for seed, note in pick_neighbors(graph, seeds, scores, cap, known):
        payload, score = best[note]
        added.append(Candidate(chunk_id=payload["chunk_id"], note=note,
                               heading_path=payload["heading_path"], text=payload["text"],
                               source="graph", via=seed, via_link=link_kind(graph, seed, note),
                               graph_score=score))
    return added
