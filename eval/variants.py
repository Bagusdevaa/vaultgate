"""The six systems compared in the evaluation, and how to read retrieval results off a trace."""

from dataclasses import dataclass

from vault_rag.trace import Trace


@dataclass(frozen=True)
class Variant:
    name: str
    retrieval: str             # "none", "grep" or "pipeline"
    gate: str | None = None    # pipeline only: none, bge, jev
    graph: bool = False        # pipeline only: 1-hop expansion


VARIANTS = [
    Variant("no_retrieval", "none"),
    Variant("grep_baseline", "grep"),
    Variant("hybrid", "pipeline", gate="none"),
    Variant("hybrid_bge_gate", "pipeline", gate="bge"),
    Variant("hybrid_jev_gate", "pipeline", gate="jev"),
    Variant("hybrid_graph_jev_gate", "pipeline", gate="jev", graph=True),   # the main system
]

BY_NAME = {v.name: v for v in VARIANTS}


def ranked_notes(trace: Trace) -> list[str]:
    """Notes of the last attempt's candidates: kept before dropped, then by gate score.

    With a gate this is the gate's reranking; without one it is the RRF order. It is
    read even when the pipeline abstained, so retrieval quality and the abstain
    decision are measured separately.
    """
    candidates = trace.attempts[-1].candidates
    order = sorted(range(len(candidates)),
                   key=lambda i: (not candidates[i].kept, -(candidates[i].gate_score or 0.0), i))
    return list(dict.fromkeys(candidates[i].note for i in order))


def graph_only_gold(trace: Trace, gold: list[str]) -> list[str]:
    """Gold notes in the answer context that hybrid search alone did not return."""
    hybrid = {c.note for c in trace.attempts[-1].candidates if c.source == "hybrid"}
    in_context = {c.note for c in trace.context}
    return sorted(n for n in set(gold) if n in in_context and n not in hybrid)
