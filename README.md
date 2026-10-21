# vault-rag

Retrieval over an Obsidian vault: hybrid search (bge-m3 dense + sparse in Qdrant), one-hop
expansion through the vault's own wikilinks, and a Jev relevance gate that abstains when the
vault does not contain the answer. Final project for Sertifikasi AI Super Class. See `PLAN.md`.

## Setup

```sh
uv sync
cp .env.example .env   # then fill in JEV_API_KEY and OPENROUTER_API_KEY
make smoke             # Jev, OpenRouter and bge-m3 smoke tests
make test
```

A smoke test whose key is missing prints `SKIPPED` instead of failing.
