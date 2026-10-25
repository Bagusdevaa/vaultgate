"""Jev (TypeSafe) through OpenRouter's Decisions API.

The endpoint is alpha, so every Jev-specific request and response detail stays in
this file. Jev answers typed questions about a JSON `state`; we only use `noul`
(yes/no) questions, whose answer is the probability of yes.
"""

import json
import time
from dataclasses import dataclass

import httpx

from vault_rag.gate.base import context_order, keep_flags
from vault_rag.trace import GateResult

RELEVANCE_CRITERIA = {
    "true": "The passage states a fact that answers the query, part of it, or a linking "
    "fact needed to reach the answer.",
    "false": "The passage only shares the topic or keywords, is about a different subject, "
    "or gives no usable evidence.",
}

SUFFICIENCY_QUESTION = {
    "type": "noul",
    "instructions": "Do the passages in `passages`, taken together, contain enough "
    "information to fully answer `query`?",
    "criteria": {
        "true": "The passages state the facts needed to answer the query.",
        "false": "The answer is missing, only partly covered, or about a different subject.",
    },
}


@dataclass
class JevCall:
    scores: list[float]   # P(yes), one per question, in the order asked
    seconds: float
    usage: dict           # input_tokens, output_tokens, cost (USD)
    model: str            # dated snapshot that served the request


def passages_state(query: str, passages: list[str]) -> dict:
    return {"query": query, "passages": {f"p{i}": text for i, text in enumerate(passages)}}


def relevance_questions(count: int) -> dict[str, dict]:
    return {
        f"p{i}": {
            "type": "noul",
            "instructions": f"Does `passages.p{i}` help answer `query`?",
            "criteria": RELEVANCE_CRITERIA,
        }
        for i in range(count)
    }


def request_bytes(query: str, passages: list[str]) -> int:
    """Size of a relevance request body; the questions repeat per passage, so they count too."""
    body = {"state": passages_state(query, passages), "questions": relevance_questions(len(passages))}
    return len(json.dumps(body).encode("utf-8"))


def byte_batches(query: str, passages: list[str], max_bytes: int) -> list[list[int]]:
    """Passage indices split into consecutive batches whose requests stay under max_bytes."""
    batches: list[list[int]] = []
    current: list[int] = []
    for i in range(len(passages)):
        trial = current + [i]
        if current and request_bytes(query, [passages[j] for j in trial]) > max_bytes:
            batches.append(current)
            trial = [i]
        current = trial
    return batches + [current] if current else batches


class JevClient:
    def __init__(self, api_key: str, model: str, endpoint: str,
                 timeout: float = 60, transport: httpx.BaseTransport | None = None):
        self.model = model
        self.endpoint = endpoint
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout, transport=transport
        )

    def ask(self, state: dict, questions: dict[str, dict]) -> JevCall:
        start = time.perf_counter()
        reply = self._http.post(
            self.endpoint, json={"model": self.model, "state": state, "questions": questions}
        )
        reply.raise_for_status()
        data = reply.json()
        scores = [float(data["answers"][key]["noul"]) for key in questions]
        return JevCall(scores, time.perf_counter() - start, data["usage"], data["model"])

    def relevance(self, query: str, passages: list[str]) -> JevCall:
        """One noul question per passage, all packed into a single request."""
        return self.ask(passages_state(query, passages), relevance_questions(len(passages)))

    def sufficiency(self, query: str, passages: list[str]) -> JevCall:
        return self.ask(passages_state(query, passages), {"sufficient": SUFFICIENCY_QUESTION})


class JevGate:
    """Relevance per chunk in byte-capped batches, then one sufficiency question over the
    passages that would form the answer context."""

    name = "jev"

    def __init__(self, client: JevClient, threshold: float, sufficiency_threshold: float,
                 max_request_bytes: int, context_max_chars: int):
        self.client = client
        self.threshold = threshold
        self.sufficiency_threshold = sufficiency_threshold
        self.max_request_bytes = max_request_bytes
        self.context_max_chars = context_max_chars

    def score(self, query: str, passages: list[str]) -> GateResult:
        start = time.perf_counter()
        result = GateResult(scores=[0.0] * len(passages), kept=[], sufficient=False,
                            threshold=self.threshold)
        for batch in byte_batches(query, passages, self.max_request_bytes):
            call = self.client.relevance(query, [passages[i] for i in batch])
            for i, score in zip(batch, call.scores, strict=True):
                result.scores[i] = score
            add_usage(result, call)
        result.kept = keep_flags(result.scores, self.threshold)

        chosen = context_order(result.scores, result.kept, [len(p) for p in passages],
                               self.context_max_chars)
        if not chosen:
            result.note = "no chunk passed the relevance threshold; sufficiency not asked"
        else:
            call = self.client.sufficiency(query, [passages[i] for i in chosen])
            add_usage(result, call)
            result.sufficiency_score = call.scores[0]
            result.sufficient = call.scores[0] >= self.sufficiency_threshold
            result.note = (f"sufficiency asked over the top {len(chosen)} of "
                           f"{sum(result.kept)} kept chunks (the answer context)")
        result.seconds = time.perf_counter() - start
        return result


def add_usage(result: GateResult, call: JevCall) -> None:
    result.input_tokens += call.usage.get("input_tokens") or 0
    result.output_tokens += call.usage.get("output_tokens") or 0
    result.cost += call.usage.get("cost") or 0.0
