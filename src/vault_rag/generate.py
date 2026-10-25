"""Answer a question from the gated context with an OpenAI-compatible model, citing notes."""

import re
import time
from dataclasses import dataclass, field

from vault_rag.trace import Candidate

SYSTEM_PROMPT = """You answer questions about the user's personal notes.

Rules:
- Use only the sources given in the user message. Do not add facts from general knowledge.
- After every claim, cite the note it comes from as its path in square brackets, for example [bagusdeva/Bugs/pbpipe.md].
- If the sources do not contain the answer, say plainly that the notes do not say, and stop.
- Answer in the language of the question. Be concise."""

CITATION = re.compile(r"\[([^\[\]]+?\.md)\]")


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)          # cited notes found in the context
    unknown_citations: list[str] = field(default_factory=list)  # cited notes that were not given
    model: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


def format_sources(context: list[Candidate]) -> str:
    blocks = []
    for c in context:
        heading = " > ".join(c.heading_path) or "(top of note)"
        blocks.append(f"Source: {c.note}\nSection: {heading}\n\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def split_citations(text: str, context: list[Candidate]) -> tuple[list[str], list[str]]:
    given = {c.note for c in context}
    cited = list(dict.fromkeys(CITATION.findall(text)))
    return [n for n in cited if n in given], [n for n in cited if n not in given]


class Generator:
    def __init__(self, client, model: str, temperature: float, reasoning: bool, max_tokens: int = 800):
        self.client = client  # openai.OpenAI or anything with the same chat.completions.create
        self.model, self.temperature = model, temperature
        self.reasoning, self.max_tokens = reasoning, max_tokens

    def generate(self, query: str, context: list[Candidate]) -> Answer:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n\n{format_sources(context)}\n\nQuestion: {query}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            # Qwen3.6 thinks by default and can spend all of max_tokens before answering.
            extra_body={"usage": {"include": True}, "reasoning": {"enabled": self.reasoning}},
        )
        text = (response.choices[0].message.content or "").strip()
        known, unknown = split_citations(text, context)
        usage = response.usage
        return Answer(text=text, citations=known, unknown_citations=unknown, model=response.model,
                      seconds=time.perf_counter() - start,
                      prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                      completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                      # OpenRouter adds `cost` (USD) when usage accounting is requested.
                      cost=getattr(usage, "cost", 0.0) or 0.0)


def from_config(llm: dict, api_key: str) -> Generator:
    from openai import OpenAI

    client = OpenAI(base_url=llm["base_url"], api_key=api_key)
    return Generator(client, llm["generator"], llm["temperature"], llm["reasoning"],
                     llm.get("max_tokens", 800))
