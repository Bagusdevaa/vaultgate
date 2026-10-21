import json

import httpx

from vault_rag.gate.jev_gate import JevClient


def fake_jev(probabilities: dict[str, float], seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = {
            "model": "typesafe/jev-1.14-20261001",
            "answers": {k: {"type": "noul", "noul": p} for k, p in probabilities.items()},
            "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.0000004},
        }
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def make_client(probabilities, seen):
    return JevClient("test-key", "typesafe/jev-1.14", "https://example.test/decisions",
                     transport=fake_jev(probabilities, seen))


def test_relevance_packs_one_question_per_passage_and_keeps_order():
    seen = []
    # Answers arrive in a different key order than asked; scores must follow input order.
    client = make_client({"p2": 0.1, "p0": 0.9, "p1": 0.4}, seen)
    call = client.relevance("q", ["a", "b", "c"])

    assert call.scores == [0.9, 0.4, 0.1]
    assert len(seen) == 1
    payload = json.loads(seen[0].content)
    assert payload["model"] == "typesafe/jev-1.14"
    assert list(payload["questions"]) == ["p0", "p1", "p2"]
    assert payload["state"]["passages"] == {"p0": "a", "p1": "b", "p2": "c"}
    assert seen[0].headers["Authorization"] == "Bearer test-key"


def test_sufficiency_asks_a_single_question():
    seen = []
    call = make_client({"sufficient": 0.8}, seen).sufficiency("q", ["a"])
    assert call.scores == [0.8]
    assert list(json.loads(seen[0].content)["questions"]) == ["sufficient"]
    assert call.usage["cost"] == 0.0000004
