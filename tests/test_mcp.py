import json

import httpx

from vault_rag.mcp_server import compact, search

ANSWER = {"status": "answer", "reason": "1 chunks passed the jev gate",
          "context": [{"note": "a/Bugs/x.md", "heading_path": ["x", "Sessions", "day"], "text": "fact",
                       "chunk_id": "a/Bugs/x.md#0", "gate_score": 0.9}]}
ABSTAIN = {"status": "abstain", "reason": "no chunk passed the jev gate after 2 attempt(s)", "context": []}


def test_compact_keeps_only_what_an_agent_needs():
    assert compact(ANSWER) == {"status": "ok", "sources": [
        {"note": "a/Bugs/x.md", "heading": "x > Sessions > day", "text": "fact"}]}
    assert compact(ABSTAIN) == {"status": "abstain", "reason": ABSTAIN["reason"]}


def test_search_posts_query_and_filters():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=ANSWER)

    result = search("http://api.test", "why?", folder=["Bugs"], transport=httpx.MockTransport(handler))
    assert seen == [("/search", {"query": "why?", "filters": {"folder": ["Bugs"]}})]
    assert result["status"] == "ok"


def test_api_down_or_failing_gives_a_clear_message():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    down = search("http://api.test", "q", transport=httpx.MockTransport(refuse))
    assert down["status"] == "error" and "make serve" in down["reason"]
    failing = search("http://api.test", "q",
                     transport=httpx.MockTransport(lambda r: httpx.Response(503, text="no key")))
    assert failing["status"] == "error" and "503" in failing["reason"]
