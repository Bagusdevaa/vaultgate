"""retrieve -> expand -> gate -> (retry) -> trace.

    pipeline = build_pipeline(load_config(), gate="jev", use_graph=True)
    trace = pipeline.run("why did the gateway mark openrouter unhealthy?")
"""

import copy
import time

import networkx as nx
from qdrant_client import QdrantClient

from vault_rag import store
from vault_rag.config import REPO_ROOT, Config
from vault_rag.gate.base import Gate, context_order
from vault_rag.graph.build import load_graph
from vault_rag.graph.expand import expand
from vault_rag.retrieve.embed import Embedder, Embedding, from_config
from vault_rag.retrieve.hybrid import hybrid_search
from vault_rag.trace import Attempt, Candidate, Trace


class Pipeline:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder,
                 graph: nx.DiGraph, gate: Gate, retrieval: dict, use_graph: bool):
        self.client, self.collection = client, collection
        self.embedder, self.graph, self.gate = embedder, graph, gate
        self.settings, self.use_graph = retrieval, use_graph

    def with_options(self, gate: Gate | None = None, use_graph: bool | None = None) -> "Pipeline":
        """A copy that shares the index, embedder and graph but uses another gate or graph setting."""
        other = copy.copy(self)
        if gate is not None:
            other.gate = gate
        if use_graph is not None:
            other.use_graph = use_graph
        return other

    def close(self) -> None:
        # Local-mode Qdrant otherwise closes itself during interpreter shutdown and prints
        # a traceback.
        self.client.close()

    def run(self, query: str, filters: dict | None = None) -> Trace:
        """First attempt as asked; on abstain, retry without filters and with a larger top_k."""
        start = time.perf_counter()
        trace = Trace(query=query, gate_name=self.gate.name, use_graph=self.use_graph)
        embedding = self.embedder.encode([query])[0]
        embed_seconds = time.perf_counter() - start

        top_k = self.settings["top_k"]
        for number in range(self.settings["max_retries"] + 1):
            if number:
                top_k, filters = self.settings["retry_top_k"], None
            attempt, context = self.attempt(query, embedding, top_k, filters)
            trace.attempts.append(attempt)
            if attempt.gate.sufficient:
                trace.status, trace.context = "answer", context
                break
        trace.attempts[0].timings["embed"] = embed_seconds
        trace.seconds = time.perf_counter() - start
        return trace

    def attempt(self, query: str, embedding: Embedding, top_k: int,
                filters: dict | None) -> tuple[Attempt, list[Candidate]]:
        timings, clock = {}, time.perf_counter()
        candidates = hybrid_search(self.client, self.collection, embedding, top_k,
                                   self.settings["rrf_k"], filters)
        timings["search"] = time.perf_counter() - clock
        if self.use_graph:
            clock = time.perf_counter()
            candidates += expand(self.client, self.collection, self.graph, candidates,
                                 embedding.dense, self.settings["graph_seed_notes"],
                                 self.settings["graph_max_neighbors_per_note"])
            timings["expand"] = time.perf_counter() - clock

        passages = [c.gate_text() for c in candidates]
        gate = self.gate.score(query, passages)
        timings["gate"] = gate.seconds
        chosen = context_order(gate.scores, gate.kept, [len(p) for p in passages],
                               self.settings["context_max_chars"])
        for i, (candidate, score, kept) in enumerate(zip(candidates, gate.scores, gate.kept)):
            candidate.gate_score, candidate.kept = score, kept
            if not kept:
                candidate.drop_reason = f"gate score {score:.3f} < threshold {gate.threshold}"
            elif i not in chosen:
                candidate.drop_reason = "kept, but outside the context budget"
        return Attempt(top_k, filters, candidates, gate, timings), [candidates[i] for i in chosen]


def make_gate(cfg: Config, name: str, context_max_chars: int) -> Gate:
    if name == "none":
        from vault_rag.gate.none_gate import NoneGate
        return NoneGate()
    if name == "bge":
        from vault_rag.gate.bge_gate import BgeGate, bge_reranker
        bge = cfg["gate"]["bge"]
        return BgeGate(bge_reranker(bge["model"], cfg["embedding"]["use_fp16"]), bge["threshold"])
    if name == "jev":
        from vault_rag.gate.jev_gate import JevClient, JevGate
        if not cfg.openrouter_api_key:
            raise RuntimeError("the Jev gate needs OPENROUTER_API_KEY in .env")
        jev = cfg["gate"]["jev"]
        return JevGate(JevClient(cfg.openrouter_api_key, jev["model"], jev["endpoint"]),
                       jev["threshold"], jev["sufficiency_threshold"], jev["max_request_bytes"],
                       context_max_chars)
    raise ValueError(f"unknown gate: {name}")


def build_pipeline(cfg: Config, gate: str | None = None, use_graph: bool | None = None) -> Pipeline:
    """Defaults come from config.yaml; eval variants override gate and graph."""
    retrieval = cfg["retrieval"]
    index_dir = REPO_ROOT / cfg["paths"]["index_dir"]
    return Pipeline(
        client=store.open_client(cfg["qdrant"], index_dir),
        collection=cfg["qdrant"]["collection"],
        embedder=from_config(cfg["embedding"]),
        graph=load_graph(index_dir / "graph.json"),
        gate=make_gate(cfg, gate or cfg["gate"]["type"], retrieval["context_max_chars"]),
        retrieval=retrieval,
        use_graph=retrieval["graph"] if use_graph is None else use_graph,
    )
