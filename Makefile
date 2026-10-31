.PHONY: smoke smoke-jev smoke-llm smoke-embed test index query serve mcp inspector tune eval report

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

# needs the API (make serve); opens on http://127.0.0.1:8501
inspector:
	uv run streamlit run inspector/app.py --server.address 127.0.0.1 --server.port 8501

# Thresholds are tuned on dev only; the test run uses the frozen config.
tune:
	uv run python -m eval.tune

# make eval ARGS="--split dev --limit 3"
eval:
	uv run python -m eval.run_eval $(ARGS)

# make report RUN=eval/results/<run>
report:
	uv run python -m eval.report $(RUN)
