from types import SimpleNamespace

from vault_rag.generate import SYSTEM_PROMPT, Generator, split_citations
from vault_rag.trace import Candidate


def cand(note: str, text: str = "fact") -> Candidate:
    return Candidate(chunk_id=f"{note}#0", note=note, heading_path=["t", "Sessions", "day"],
                     text=text, source="hybrid")


class FakeClient:
    def __init__(self, reply: str):
        self.calls = []
        completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=completions)
        self.reply = reply

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, cost=0.0002)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage,
                               model="qwen/qwen3.6-35b-a3b-20260901")


def test_prompt_carries_rules_sources_and_question():
    client = FakeClient("ok [a/Bugs/x.md]")
    Generator(client, "qwen", 0, reasoning=False).generate("why?", [cand("a/Bugs/x.md", "the cause")])
    call = client.calls[0]
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = call["messages"][1]["content"]
    assert "Source: a/Bugs/x.md" in user and "Section: t > Sessions > day" in user
    assert "the cause" in user and user.endswith("Question: why?")
    assert call["temperature"] == 0
    assert call["extra_body"]["reasoning"] == {"enabled": False}


def test_answer_splits_known_and_invented_citations():
    reply = "Cause one [a/Bugs/x.md]. Also [a/Bugs/x.md] and [b/Made/up.md]."
    answer = Generator(FakeClient(reply), "qwen", 0, False).generate("q", [cand("a/Bugs/x.md")])
    assert answer.citations == ["a/Bugs/x.md"]
    assert answer.unknown_citations == ["b/Made/up.md"]
    assert (answer.prompt_tokens, answer.completion_tokens, answer.cost) == (120, 30, 0.0002)


def test_brackets_that_are_not_note_paths_are_ignored():
    assert split_citations("see [1] and [link](x) and [a/b.md]", [cand("a/b.md")]) == (["a/b.md"], [])
