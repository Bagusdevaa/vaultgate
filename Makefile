.PHONY: smoke smoke-jev smoke-llm smoke-embed test index query

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
