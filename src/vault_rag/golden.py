"""Split the golden draft into a tuning set and a held-out test set."""

import random
from collections import defaultdict


def dev_quota(counts: dict[str, int], dev_size: int) -> dict[str, int]:
    """Dev questions per category, proportional to the draft (largest remainder)."""
    total = sum(counts.values())
    exact = {c: n * dev_size / total for c, n in counts.items()}
    quota = {c: int(x) for c, x in exact.items()}
    by_remainder = sorted(counts, key=lambda c: (-(exact[c] - quota[c]), c))
    for category in by_remainder[: dev_size - sum(quota.values())]:
        quota[category] += 1
    return quota


def split(items: list[dict], dev_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified by category and deterministic for a given seed; both keep draft order."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[item["category"]].append(index)
    quota = dev_quota({c: len(ix) for c, ix in groups.items()}, dev_size)
    rng = random.Random(seed)
    dev = set()
    for category in sorted(groups):
        dev.update(rng.sample(groups[category], quota[category]))
    return ([item for i, item in enumerate(items) if i in dev],
            [item for i, item in enumerate(items) if i not in dev])
