"""LLM judge: correctness against the reference answer and faithfulness to the context.

The judge model must come from a different family than the generator (config check in
run_eval). One call per answer, JSON out.
"""

import json
from dataclasses import dataclass

JUDGE_PROMPT = """You grade an answer produced by a question-answering system over personal notes.

You get the question, whether the notes contain an answer, a reference answer, the sources the system was given, and the system's answer.

Return a JSON object with these keys:
- "declined": 1 if the answer says the notes do not contain the answer (or refuses to answer) instead of giving one, else 0.
- "correct": for an answerable question, 1 if the answer gives the key facts of the reference answer without contradicting it, else 0 (a declined answer is 0). For an unanswerable question, 1 if the answer declines, else 0.
- "faithful": 1 if every claim in the answer is supported by the given sources, else 0. A declined answer is 1.
- "reason": one short sentence explaining the grades.

Judge meaning, not wording. The answer may be in a different language than the reference."""


@dataclass
class Verdict:
    correct: int
    faithful: int
    declined: int
    reason: str
    cost: float = 0.0


def judge_message(question: str, answerable: bool, reference: str, sources: str, answer: str) -> str:
    return (f"Question: {question}\n\n"
            f"Notes contain an answer: {'yes' if answerable else 'no'}\n\n"
            f"Reference answer: {reference}\n\n"
            f"Sources given to the system:\n{sources or '(none)'}\n\n"
            f"System answer:\n{answer}")


def parse_verdict(text: str) -> Verdict:
    """Read the judge's JSON; tolerate a ```json fence around it."""
    body = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    data = json.loads(body)
    return Verdict(correct=int(data["correct"]), faithful=int(data["faithful"]),
                   declined=int(data["declined"]), reason=str(data.get("reason", "")))


class Judge:
    def __init__(self, client, model: str, max_tokens: int = 300):
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def grade(self, question: str, answerable: bool, reference: str, sources: str,
              answer: str) -> Verdict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": judge_message(question, answerable, reference,
                                                                sources, answer)}],
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            extra_body={"usage": {"include": True}, "reasoning": {"enabled": False}},
        )
        verdict = parse_verdict(response.choices[0].message.content or "")
        verdict.cost = getattr(response.usage, "cost", 0.0) or 0.0
        return verdict
