"""Metric functions over per-question eval records. Pure functions, unit tested."""

import math
from collections import defaultdict


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    """Share of the gold notes found in the first k distinct ranked notes."""
    if not gold:
        return 0.0
    top = list(dict.fromkeys(ranked))[:k]
    return sum(note in top for note in set(gold)) / len(set(gold))


def reciprocal_rank(ranked: list[str], gold: list[str]) -> float:
    """1 / rank of the first gold note among distinct ranked notes, 0 if none appears."""
    for rank, note in enumerate(dict.fromkeys(ranked), start=1):
        if note in gold:
            return 1 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def answerable(record: dict) -> bool:
    return record["category"] != "unanswerable"


def summarize(records: list[dict]) -> dict:
    """One row of metrics for a group of records (usually one variant)."""
    ans = [r for r in records if answerable(r)]
    una = [r for r in records if not answerable(r)]
    retrieved = [r for r in ans if r["ranked_notes"] is not None]
    correct = [r for r in ans if r["judge"]["correct"]]
    return {
        "questions": len(records),
        "recall@5": mean([recall_at_k(r["ranked_notes"], r["gold_notes"], 5) for r in retrieved])
        if retrieved else None,
        "mrr": mean([reciprocal_rank(r["ranked_notes"], r["gold_notes"]) for r in retrieved])
        if retrieved else None,
        "correct": mean([r["judge"]["correct"] for r in records]),
        "correct_answerable": mean([r["judge"]["correct"] for r in ans]),
        "faithful": mean([r["judge"]["faithful"] for r in records if not r["judge"]["declined"]])
        if any(not r["judge"]["declined"] for r in records) else None,
        "correct_abstain": mean([r["judge"]["declined"] for r in una]) if una else None,
        "hallucination": mean([1 - r["judge"]["declined"] for r in una]) if una else None,
        "false_abstain": mean([r["judge"]["declined"] for r in ans]) if ans else None,
        "graph_only_correct": mean([bool(r["graph_only_gold"]) for r in correct]) if correct else 0.0,
        "latency_p50": percentile([r["seconds"] for r in records], 50),
        "latency_p95": percentile([r["seconds"] for r in records], 95),
        "tokens": mean([r["tokens"] for r in records]),
        "cost": mean([r["cost"] for r in records]),
    }


def by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return dict(groups)
