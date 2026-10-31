import networkx as nx
from qdrant_client import QdrantClient

from vault_rag import store
from vault_rag.pipeline import Pipeline
from vault_rag.retrieve.embed import Embedding
from vault_rag.trace import GateResult

SETTINGS = {"top_k": 3, "rrf_k": 60, "graph_seed_notes": 2, "graph_max_neighbors_per_note": 2,
            "max_retries": 1, "retry_top_k": 6, "context_max_chars": 10_000}


def vec(i: int) -> list[float]:
    return [1.0 if j == i else 0.1 for j in range(store.DENSE_DIM)]


class OneVector:
    def encode(self, texts):
        return [Embedding(vec(0), {0: 1.0}) for _ in texts]


class ScriptedGate:
    """Keeps passages containing `keep`; sufficiency answers follow the script in order."""
    name = "scripted"

    def __init__(self, sufficient: list[bool], keep: str = ""):
        self.sufficient, self.keep, self.seen = list(sufficient), keep, []

    def score(self, query, passages):
        self.seen.append(passages)
        kept = [self.keep in p for p in passages]
        return GateResult(scores=[0.9 if k else 0.1 for k in kept], kept=kept,
                          sufficient=self.sufficient.pop(0), threshold=0.5)


def make_pipeline(gate, use_graph=False, settings=SETTINGS):
    client = QdrantClient(":memory:")
    store.reset_collection(client, "vault")
    notes = [f"n{i}.md" for i in range(8)]
    payloads = [{"chunk_id": f"{n}#0", "note": n, "heading_path": [n], "text": f"text of {n}",
                 "folder": "Bugs" if i < 2 else "Knowledge", "tags": [], "device": "dev"}
                for i, n in enumerate(notes)]
    store.upsert_chunks(client, "vault", payloads, [Embedding(vec(i), {i: 1.0}) for i in range(8)])
    graph = nx.DiGraph([("n0.md", "n7.md")])
    return Pipeline(client, "vault", OneVector(), graph, gate, settings, use_graph)


def test_answer_on_first_attempt_has_context():
    trace = make_pipeline(ScriptedGate([True], keep="n0")).run("q")
    assert trace.status == "answer" and len(trace.attempts) == 1
    assert [c.note for c in trace.context] == ["n0.md"]
    dropped = [c for c in trace.attempts[0].candidates if not c.kept]
    assert dropped and all("threshold" in c.drop_reason for c in dropped)


def test_abstain_retries_once_without_filter_and_with_larger_top_k():
    gate = ScriptedGate([False, True], keep="n")
    trace = make_pipeline(gate).run("q", filters={"folder": "Bugs"})
    first, second = trace.attempts
    assert (first.top_k, first.filters, len(first.candidates)) == (3, {"folder": "Bugs"}, 2)
    assert (second.top_k, second.filters, len(second.candidates)) == (6, None, 6)
    assert trace.status == "answer" and "embed" in first.timings


def test_gives_up_after_max_retries():
    trace = make_pipeline(ScriptedGate([False, False])).run("q")
    assert trace.status == "abstain" and len(trace.attempts) == 2 and trace.context == []


def test_graph_candidates_join_before_the_gate():
    gate = ScriptedGate([True], keep="n")
    trace = make_pipeline(gate, use_graph=True, settings=dict(SETTINGS, top_k=1)).run("q")
    graph_added = [c for c in trace.attempts[0].candidates if c.source == "graph"]
    assert [(c.note, c.via) for c in graph_added] == [("n7.md", "n0.md")]
    assert any("n7.md" in p for p in gate.seen[0])


def test_context_is_cut_at_the_budget():
    settings = dict(SETTINGS, context_max_chars=40)
    trace = make_pipeline(ScriptedGate([True], keep="n"), settings=settings).run("q")
    assert len(trace.context) == 1
    assert any(c.drop_reason == "kept, but outside the context budget"
               for c in trace.attempts[0].candidates)
    assert trace.to_dict()["status"] == "answer"


def test_with_options_shares_resources_but_not_settings():
    base = make_pipeline(ScriptedGate([True]))
    other = base.with_options(gate=ScriptedGate([False, False]), use_graph=True)
    assert other.client is base.client and other.embedder is base.embedder
    assert other.use_graph and not base.use_graph
    assert base.run("q").status == "answer" and other.run("q").status == "abstain"
