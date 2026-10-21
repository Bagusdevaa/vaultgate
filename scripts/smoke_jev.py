"""Jev smoke test: rerank(), relevance_rerank() and a native sufficiency question."""

import sys
import time

import httpx
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
# Only the in-store and wrong-shop passages: related, but they cannot answer the query.
NO_ANSWER = [DOCUMENTS[1], DOCUMENTS[3]]

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
SUFFICIENCY_QUESTION = {
    "type": "noul",
    "instructions": "Do the passages in `documents`, taken together, contain enough "
    "information to fully answer `query`?",
    "criteria": {
        "true": "The passages state the facts needed to answer the query",
        "false": "The answer is missing, only partly covered, or about a different subject",
    },
}


def ask_sufficient(api_key: str, docs: list[str]) -> tuple[float, float]:
    """Ask Jev one yes/no question; return P(yes) and latency in seconds."""
    payload = {
        "model": "jev-latest",
        "state": {"query": QUERY, "documents": {f"doc_{i}": d for i, d in enumerate(docs)}},
        "questions": {"sufficient": SUFFICIENCY_QUESTION},
    }
    start = time.perf_counter()
    reply = httpx.post(
        JEV_ENDPOINT, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=60
    )
    reply.raise_for_status()
    return reply.json()["answers"]["sufficient"]["noul"], time.perf_counter() - start


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

    cutoff = jev_cfg["sufficiency_threshold"]
    kept = [item["text"] for item in gated["results"]]
    p_kept, s_kept = ask_sufficient(cfg.jev_api_key, kept)
    p_none, s_none = ask_sufficient(cfg.jev_api_key, NO_ANSWER)
    print(f"\n== sufficiency (cutoff {cutoff})")
    print(f"  kept passages:      P(sufficient)={p_kept:.3f}  ({s_kept:.2f}s)")
    print(f"  no-answer passages: P(sufficient)={p_none:.3f}  ({s_none:.2f}s)")
    return 0 if p_kept >= cutoff > p_none else 1


if __name__ == "__main__":
    sys.exit(main())
