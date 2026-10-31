import pytest

from eval.metrics import percentile, recall_at_k, reciprocal_rank, summarize


def test_recall_counts_distinct_notes_and_unique_gold():
    ranked = ["a", "a", "b", "c", "d", "e", "f"]
    assert recall_at_k(ranked, ["b", "f"], 5) == 0.5     # f is the 6th distinct note
    assert recall_at_k(ranked, ["b", "b"], 5) == 1.0
    assert recall_at_k([], ["a"], 5) == 0.0
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank_uses_distinct_notes():
    assert reciprocal_rank(["a", "a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["x", "a"]) == 1.0
    assert reciprocal_rank(["a"], ["x"]) == 0.0


def test_percentile_is_nearest_rank():
    values = [5, 1, 4, 2, 3]
    assert percentile(values, 50) == 3
    assert percentile(values, 95) == 5
    assert percentile(values, 0) == 1
    assert percentile([], 50) == 0.0


def record(category, correct, declined, ranked=("g",), graph_only=(), seconds=1.0, cost=0.01):
    return {"category": category, "gold_notes": ["g"],
            "ranked_notes": None if ranked is None else list(ranked),
            "graph_only_gold": list(graph_only), "seconds": seconds, "tokens": 100, "cost": cost,
            "judge": {"correct": correct, "faithful": 1, "declined": declined}}


def test_summarize_splits_answerable_and_unanswerable():
    rows = [record("keyword", 1, 0),
            record("multihop", 1, 0, ranked=("x", "g"), graph_only=("g",)),
            record("semantic", 0, 1, ranked=("x",)),
            record("unanswerable", 1, 1),
            record("unanswerable", 0, 0)]
    s = summarize(rows)
    assert s["recall@5"] == pytest.approx(2 / 3)
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert s["correct"] == pytest.approx(3 / 5)
    assert s["correct_answerable"] == pytest.approx(2 / 3)
    assert s["false_abstain"] == pytest.approx(1 / 3)
    assert s["correct_abstain"] == 0.5 and s["hallucination"] == 0.5
    assert s["graph_only_correct"] == 0.5    # 1 of the 2 correct answerable answers
    assert s["cost"] == pytest.approx(0.01)


def test_summarize_without_retrieval_has_no_retrieval_metrics():
    s = summarize([record("keyword", 0, 1, ranked=None)])
    assert s["recall@5"] is None and s["mrr"] is None
