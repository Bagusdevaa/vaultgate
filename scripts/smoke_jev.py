"""Jev smoke test: one rerank() and one relevance_rerank() call on 5 toy documents."""

import sys
import time

from jev_reranker import JevReranker

from vault_rag.config import load_config

QUERY = "How long do I have to return an online order to ACME Shop?"
DOCUMENTS = [
    "ACME Shop accepts online returns within 30 days of delivery.",
    "ACME Shop in-store purchases can be returned within 14 days of purchase.",
    "ACME Shop products come with a one-year repair warranty.",
    "FooBar Shop accepts online returns within 60 days of delivery.",
    "Bread is made from flour, water, yeast and salt.",
]


def show(label: str, response: dict, seconds: float) -> None:
    print(f"\n== {label} ({seconds:.2f}s)")
    for text, doc in zip(DOCUMENTS, response["detail"]["documents"]):
        mark = "keep" if doc["passes_threshold"] else "drop"
        print(f"  {doc['score']:.3f}  {mark}  {text}")
    print(f"  usage: {response['detail'].get('usage')}")


def main() -> int:
    cfg = load_config()
    if not cfg.jev_api_key:
        print("SKIPPED: JEV_API_KEY not set")
        return 0
    jev_cfg = cfg["gate"]["jev"]
    reranker = JevReranker(api_key=cfg.jev_api_key, mode=jev_cfg["mode"])

    start = time.perf_counter()
    ranked = reranker.rerank(QUERY, DOCUMENTS, detail=True)
    show("rerank", ranked, time.perf_counter() - start)

    start = time.perf_counter()
    gated = reranker.relevance_rerank(QUERY, DOCUMENTS, threshold=jev_cfg["threshold"], detail=True)
    show(f"relevance_rerank (threshold {jev_cfg['threshold']})", gated, time.perf_counter() - start)

    sufficient = len(gated["results"]) > 0
    print(f"\nsufficient evidence: {sufficient}")
    return 0 if sufficient else 1


if __name__ == "__main__":
    sys.exit(main())
