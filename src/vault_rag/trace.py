"""What one query went through: every candidate with its score at each stage.

The CLI, the API, the inspector and the evaluation all read this object.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class Candidate:
    chunk_id: str
    note: str
    heading_path: list[str]
    text: str
    source: str                      # "hybrid" or "graph"
    via: str | None = None           # seed note a graph candidate was reached from
    dense_rank: int | None = None    # 1-based; None when absent from that list
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    graph_score: float | None = None  # dense similarity of a graph candidate to the query
    gate_score: float | None = None
    kept: bool | None = None
    drop_reason: str | None = None

    def gate_text(self) -> str:
        """What a gate reads: where the chunk comes from, then the chunk itself."""
        return f"[{' > '.join([self.note, *self.heading_path])}]\n{self.text}"


@dataclass
class GateResult:
    scores: list[float]
    kept: list[bool]
    sufficient: bool
    sufficiency_score: float | None = None
    threshold: float | None = None
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    note: str = ""                   # how sufficiency was decided, for the trace reader


@dataclass
class Attempt:
    top_k: int
    filters: dict | None
    candidates: list[Candidate]
    gate: GateResult
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class Trace:
    query: str
    gate_name: str
    use_graph: bool
    attempts: list[Attempt] = field(default_factory=list)
    status: str = "abstain"          # "answer" or "abstain"
    context: list[Candidate] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def cost(self) -> float:
        return sum(a.gate.cost for a in self.attempts)

    @property
    def input_tokens(self) -> int:
        return sum(a.gate.input_tokens for a in self.attempts)

    @property
    def reason(self) -> str:
        """One line on why the pipeline answered or abstained."""
        if not self.attempts:
            return "no attempt was made"
        gate, tries = self.attempts[-1].gate, len(self.attempts)
        if self.status == "answer":
            return f"{len(self.context)} chunks passed the {self.gate_name} gate"
        if not any(gate.kept):
            return f"no chunk passed the {self.gate_name} gate after {tries} attempt(s)"
        score = "" if gate.sufficiency_score is None else f" (sufficiency {gate.sufficiency_score:.2f})"
        return f"the kept chunks were not enough to answer{score} after {tries} attempt(s)"

    def to_dict(self) -> dict:
        return asdict(self) | {"cost": self.cost, "input_tokens": self.input_tokens,
                               "reason": self.reason}
