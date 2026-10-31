from fastapi.testclient import TestClient

from vault_rag.api import create_app
from vault_rag.generate import Answer
from vault_rag.trace import GateResult

from test_pipeline import make_pipeline


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, query, context):
        self.calls.append((query, context))
        return Answer(text=f"answer [{context[0].note}]", citations=[context[0].note])


class AlwaysGate:
    def __init__(self, name: str, sufficient: bool):
        self.name, self.sufficient = name, sufficient

    def score(self, query, passages):
        return GateResult(scores=[0.9] * len(passages), kept=[True] * len(passages),
                          sufficient=self.sufficient, sufficiency_score=0.9 if self.sufficient else 0.1)


def client_for(default_sufficient=True):
    pipeline = make_pipeline(AlwaysGate("jev", default_sufficient))
    made = []

    def make_gate(name):
        made.append(name)
        return AlwaysGate(name, True)

    generator = FakeGenerator()
    app = create_app(pipeline, make_gate, generator, models={"generator": "fake"})
    return TestClient(app), generator, made


def test_health_reports_index_size():
    client, _, _ = client_for()
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 8 and body["gate"] == "jev"


def test_search_returns_a_trace():
    client, generator, _ = client_for()
    body = client.post("/search", json={"query": "q"}).json()
    assert body["status"] == "answer" and body["attempts"] and body["reason"]
    assert generator.calls == []


def test_ask_answers_with_citations():
    client, generator, _ = client_for()
    body = client.post("/ask", json={"query": "q"}).json()
    assert body["answer"]["citations"] == [body["trace"]["context"][0]["note"]]
    assert len(generator.calls) == 1


def test_ask_abstains_without_calling_the_llm():
    client, generator, _ = client_for(default_sufficient=False)
    body = client.post("/ask", json={"query": "q"}).json()
    assert body["trace"]["status"] == "abstain" and len(body["trace"]["attempts"]) == 2
    assert "not have enough information" in body["answer"]["text"]
    assert generator.calls == []


def test_gate_override_builds_that_gate_once():
    client, _, made = client_for(default_sufficient=False)
    for _ in range(2):
        body = client.post("/search", json={"query": "q", "gate": "bge"}).json()
        assert body["gate_name"] == "bge" and body["status"] == "answer"
    assert made == ["bge"]
    assert client.post("/search", json={"query": "q"}).json()["gate_name"] == "jev"


def test_bad_requests_are_rejected():
    client, _, _ = client_for()
    assert client.post("/search", json={"query": "q", "filters": {"colour": "red"}}).status_code == 422
    assert client.post("/search", json={"query": "q", "gate": "gpt"}).status_code == 422
    assert client.post("/search", json={"query": ""}).status_code == 422
