"""Gate interface: score every candidate for relevance, then decide answer vs abstain."""

from typing import Protocol

from vault_rag.trace import GateResult


class Gate(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> GateResult: ...


def keep_flags(scores: list[float], threshold: float) -> list[bool]:
    """A score exactly at the threshold is kept."""
    return [score >= threshold for score in scores]


def context_order(scores: list[float], kept: list[bool], lengths: list[int],
                  max_chars: int) -> list[int]:
    """Kept passages, best score first, while the total length fits in max_chars.

    The pipeline builds the answer context with this, and the Jev gate asks its
    sufficiency question over the same passages, so both see the same evidence.
    """
    ranked = sorted((i for i, k in enumerate(kept) if k), key=lambda i: -scores[i])
    chosen, total = [], 0
    for i in ranked:
        if chosen and total + lengths[i] > max_chars:
            break
        chosen.append(i)
        total += lengths[i]
    return chosen
