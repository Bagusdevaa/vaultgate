from types import SimpleNamespace

from eval.judge import Judge, judge_message, parse_verdict


def test_parse_verdict_accepts_a_fenced_reply():
    v = parse_verdict('```json\n{"correct": 1, "faithful": true, "declined": 0, "reason": "ok"}\n```')
    assert (v.correct, v.faithful, v.declined, v.reason) == (1, 1, 0, "ok")


def test_message_marks_unanswerable_and_missing_sources():
    text = judge_message("q?", False, "ref", "", "the notes do not say")
    assert "Notes contain an answer: no" in text and "(none)" in text


class FakeClient:
    def __init__(self, reply):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.reply = reply

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(cost=0.0002))


def test_grade_asks_for_json_and_records_cost():
    client = FakeClient('{"correct": 0, "faithful": 1, "declined": 1, "reason": "declined"}')
    v = Judge(client, "judge-model").grade("q", True, "ref", "src", "no idea")
    assert (v.correct, v.declined, v.cost) == (0, 1, 0.0002)
    call = client.calls[0]
    assert call["model"] == "judge-model" and call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
