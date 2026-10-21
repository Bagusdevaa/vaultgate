.PHONY: smoke smoke-jev smoke-llm smoke-embed test

smoke: smoke-jev smoke-llm smoke-embed

smoke-jev:
	uv run python scripts/smoke_jev.py

smoke-llm:
	uv run python scripts/smoke_llm.py

smoke-embed:
	uv run python scripts/smoke_embed.py

test:
	uv run pytest -q
