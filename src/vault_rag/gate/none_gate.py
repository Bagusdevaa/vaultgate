"""Passthrough gate for the baseline variants: keeps everything, never abstains."""

from vault_rag.trace import GateResult


class NoneGate:
    name = "none"

    def score(self, query: str, passages: list[str]) -> GateResult:
        return GateResult(scores=[1.0] * len(passages), kept=[True] * len(passages),
                          sufficient=bool(passages), note="no gate: every candidate is kept")
