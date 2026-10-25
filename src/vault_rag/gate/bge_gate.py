"""Open-source gate: bge-reranker-v2-m3 relevance scores with a fixed threshold."""

import time
from collections.abc import Callable

from vault_rag.gate.base import keep_flags
from vault_rag.trace import GateResult

Scorer = Callable[[list[tuple[str, str]]], list[float]]


class BgeGate:
    name = "bge"

    def __init__(self, scorer: Scorer, threshold: float):
        self.scorer = scorer
        self.threshold = threshold

    def score(self, query: str, passages: list[str]) -> GateResult:
        start = time.perf_counter()
        scores = self.scorer([(query, p) for p in passages]) if passages else []
        kept = keep_flags(scores, self.threshold)
        # A reranker only scores pairs; it has no "is this enough?" signal of its own.
        return GateResult(scores=scores, kept=kept, sufficient=any(kept),
                          sufficiency_score=max(scores, default=None), threshold=self.threshold,
                          seconds=time.perf_counter() - start,
                          note="bge has no sufficiency signal: answer when any chunk passes")


def bge_reranker(model_id: str, use_fp16: bool, max_length: int = 1024) -> Scorer:
    """FlagReranker with normalize=True, so scores are sigmoid probabilities in 0-1."""
    from FlagEmbedding import FlagReranker

    from vault_rag.hf import local_model_path

    model = FlagReranker(local_model_path(model_id), use_fp16=use_fp16)

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        scores = model.compute_score(pairs, normalize=True, max_length=max_length)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]

    return scorer
