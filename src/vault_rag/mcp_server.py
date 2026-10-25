"""MCP server with one tool, `search_vault`, so Claude Code (or Hermes) can query the vault.

    claude mcp add vaultgate -s user -- uv --directory ~/WORK/vault-rag run python -m vault_rag.mcp_server

The tool forwards to the API's /search instead of opening the index itself: local-mode
Qdrant allows one process per index, and the API already holds the index and models.
No answer is generated here; the calling agent writes the answer from the sources.
"""

import os

import httpx
from mcp.server.mcpserver import MCPServer

from vault_rag.config import load_config

server = MCPServer(
    name="vaultgate",
    instructions="search_vault searches the user's Obsidian notes. Cite the note paths it "
    "returns. When it says abstain, tell the user the notes do not cover the question.",
)


def compact(trace: dict) -> dict:
    """The parts of a trace an agent needs: the gated sources, or why there are none."""
    if trace["status"] != "answer":
        return {"status": "abstain", "reason": trace["reason"]}
    sources = [{"note": c["note"], "heading": " > ".join(c["heading_path"]), "text": c["text"]}
               for c in trace["context"]]
    return {"status": "ok", "sources": sources}


def search(api_url: str, query: str, folder: list[str] | None = None, tags: list[str] | None = None,
           transport: httpx.BaseTransport | None = None) -> dict:
    filters = {k: v for k, v in (("folder", folder), ("tags", tags)) if v} or None
    try:
        with httpx.Client(base_url=api_url, timeout=120, transport=transport) as http:
            reply = http.post("/search", json={"query": query, "filters": filters})
            reply.raise_for_status()
    except httpx.ConnectError:
        return {"status": "error", "reason": f"vaultgate API not reachable at {api_url}; "
                "start it with `make serve` in ~/WORK/vault-rag"}
    except httpx.HTTPStatusError as exc:
        return {"status": "error",
                "reason": f"vaultgate API returned {exc.response.status_code}: {exc.response.text[:300]}"}
    return compact(reply.json())


def api_url() -> str:
    mcp = load_config()["mcp"]  # also loads .env, which may set VAULTGATE_API_URL
    return os.getenv("VAULTGATE_API_URL") or mcp["api_url"]


@server.tool()
def search_vault(query: str, folder: list[str] | None = None, tags: list[str] | None = None) -> dict:
    """Search the user's Obsidian vault (PersonalBrain: projects, bugs, decisions, patterns,
    knowledge notes from past Claude Code sessions) with hybrid search, link-graph expansion
    and a relevance gate.

    Returns {"status": "ok", "sources": [{note, heading, text}]} with only the passages that
    passed the gate, or {"status": "abstain", "reason": ...} when the vault does not contain
    enough to answer. Optional filters: folder (e.g. ["Bugs", "Decisions"]) and tags.
    """
    return search(api_url(), query, folder, tags)


if __name__ == "__main__":
    server.run("stdio")
