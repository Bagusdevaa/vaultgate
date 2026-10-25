import json

import httpx
import pytest

from vault_rag.gate.base import context_order, keep_flags
from vault_rag.gate.bge_gate import BgeGate
from vault_rag.gate.jev_gate import JevClient, JevGate, byte_batches, request_bytes
from vault_rag.gate.none_gate import NoneGate


def test_threshold_boundary_is_kept():
    assert keep_flags([0.19, 0.2, 0.21], 0.2) == [False, True, True]


def test_context_order_ranks_kept_and_stops_at_budget():
    scores, kept = [0.3, 0.9, 0.5, 0.8], [True, True, False, True]
    assert context_order(scores, kept, [10, 10, 10, 10], max_chars=100) == [1, 3, 0]
    assert context_order(scores, kept, [10, 10, 10, 10], max_chars=25) == [1, 3]
    # The best passage always goes in, even alone over the budget.
    assert context_order(scores, kept, [10, 50, 10, 10], max_chars=25) == [1]
    assert context_order(scores, [False] * 4, [10] * 4, max_chars=100) == []


def test_none_gate_keeps_everything():
    result = NoneGate().score("q", ["a", "b"])
    assert result.kept == [True, True] and result.sufficient
    assert not NoneGate().score("q", []).sufficient


def test_bge_gate_is_sufficient_when_any_chunk_passes():
    gate = BgeGate(lambda pairs: [0.9 if "yes" in p else 0.1 for _, p in pairs], threshold=0.5)
    result = gate.score("q", ["yes here", "no"])
    assert result.kept == [True, False] and result.sufficient and result.sufficiency_score == 0.9
    assert not gate.score("q", ["no", "nope"]).sufficient


def test_byte_batches_stay_under_the_cap_and_keep_order():
    passages = [f"passage {i} " + "x" * 900 for i in range(12)]
    batches = byte_batches("query", passages, max_bytes=4000)
    assert [i for batch in batches for i in batch] == list(range(12))
    assert len(batches) > 1
    assert all(request_bytes("query", [passages[i] for i in b]) <= 4000 for b in batches)


def jev_transport(seen: list):
    """Relevance 0.9 for passages containing "good", else 0.05; sufficiency 0.8."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        passages = body["state"]["passages"]
        if "sufficient" in body["questions"]:
            answers = {"sufficient": {"noul": 0.8}}
        else:
            answers = {k: {"noul": 0.9 if "good" in passages[k] else 0.05} for k in body["questions"]}
        return httpx.Response(200, json={"model": "jev", "answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 1, "cost": 0.001}})
    return httpx.MockTransport(handler)


def make_gate(seen, max_bytes=30000):
    client = JevClient("k", "jev", "https://example.test", transport=jev_transport(seen))
    return JevGate(client, threshold=0.2, sufficiency_threshold=0.5,
                   max_request_bytes=max_bytes, context_max_chars=12000)


def test_jev_gate_merges_batches_in_order_then_asks_sufficiency():
    seen = []
    passages = [("good " if i % 3 == 0 else "bad ") + "y" * 900 for i in range(9)]
    result = make_gate(seen, max_bytes=4000).score("q", passages)
    relevance_calls = [b for b in seen if "sufficient" not in b["questions"]]
    assert len(relevance_calls) > 1
    assert result.scores == [0.9 if i % 3 == 0 else 0.05 for i in range(9)]
    assert result.kept == [i % 3 == 0 for i in range(9)]
    assert list(seen[-1]["state"]["passages"].values()) == [passages[0], passages[3], passages[6]]
    assert result.sufficient and result.sufficiency_score == 0.8
    assert result.cost == pytest.approx(0.001 * len(seen))


def test_jev_gate_skips_sufficiency_when_nothing_is_kept():
    seen = []
    result = make_gate(seen).score("q", ["bad one", "bad two"])
    assert len(seen) == 1 and not result.sufficient and result.sufficiency_score is None
