from collections import Counter

from vault_rag.golden import dev_quota, split


def draft(counts: dict[str, int]) -> list[dict]:
    return [{"id": f"{c}{i}", "category": c} for c, n in counts.items() for i in range(n)]


def test_quota_is_proportional_and_sums_to_dev_size():
    # 15 of 75: exact shares 3.8, 4.0, 3.8, 3.4, so the two .8 remainders round up.
    counts = {"keyword": 19, "semantic": 20, "multihop": 19, "unanswerable": 17}
    assert dev_quota(counts, 15) == {"keyword": 4, "semantic": 4, "multihop": 4, "unanswerable": 3}


def test_split_is_disjoint_complete_and_stratified():
    items = draft({"keyword": 19, "semantic": 20, "multihop": 19, "unanswerable": 17})
    dev, test = split(items, 15, seed=7)
    assert len(dev) == 15 and len(test) == 60
    assert {i["id"] for i in dev}.isdisjoint(i["id"] for i in test)
    assert Counter(i["category"] for i in dev) == {"keyword": 4, "semantic": 4,
                                                   "multihop": 4, "unanswerable": 3}


def test_same_seed_same_split():
    items = draft({"keyword": 10, "semantic": 10})
    assert split(items, 5, seed=1) == split(items, 5, seed=1)
    assert split(items, 5, seed=1) != split(items, 5, seed=2)
