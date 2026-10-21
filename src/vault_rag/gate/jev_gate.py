"""Jev (TypeSafe) through OpenRouter's Decisions API.

The endpoint is alpha, so every Jev-specific request and response detail stays in
this file. Jev answers typed questions about a JSON `state`; we only use `noul`
(yes/no) questions, whose answer is the probability of yes.
"""

import time
from dataclasses import dataclass

import httpx

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
        questions = {
            f"p{i}": {
                "type": "noul",
                "instructions": f"Does `passages.p{i}` help answer `query`?",
                "criteria": RELEVANCE_CRITERIA,
            }
            for i in range(len(passages))
        }
        return self.ask(passages_state(query, passages), questions)

    def sufficiency(self, query: str, passages: list[str]) -> JevCall:
        return self.ask(passages_state(query, passages), {"sufficient": SUFFICIENCY_QUESTION})
