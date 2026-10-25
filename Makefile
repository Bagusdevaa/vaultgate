.PHONY: smoke smoke-jev smoke-llm smoke-embed test index query serve mcp

smoke: smoke-jev smoke-llm smoke-embed

smoke-jev:
	uv run python scripts/smoke_jev.py

smoke-llm:
	uv run python scripts/smoke_llm.py

smoke-embed:
	uv run python scripts/smoke_embed.py

test:
	uv run pytest -q

index:
	uv run python scripts/index_vault.py

# make query Q="your question" ARGS="--gate bge --no-graph"
query:
	uv run python scripts/query.py "$(Q)" $(ARGS)

serve:
	uv run uvicorn --factory vault_rag.api:build_app --host 127.0.0.1 --port 8000

mcp:
	uv run python -m vault_rag.mcp_server
