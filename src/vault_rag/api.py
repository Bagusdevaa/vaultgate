"""HTTP API: /search returns the retrieval trace, /ask adds a cited answer, /health.

    uv run uvicorn --factory vault_rag.api:build_app --host 127.0.0.1 --port 8000
"""

import threading
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vault_rag import store
from vault_rag.gate.base import Gate
from vault_rag.generate import Answer, Generator
from vault_rag.pipeline import Pipeline
from vault_rag.retrieve.hybrid import build_filter
from vault_rag.trace import Trace


class Query(BaseModel):
    query: str = Field(min_length=1)
    filters: dict[str, list[str] | str] | None = None   # folder, tags, device
    gate: Literal["none", "bge", "jev"] | None = None    # default: config gate.type
    graph: bool | None = None                            # default: config retrieval.graph


def abstain_answer(trace: Trace) -> Answer:
    return Answer(text=f"The vault does not have enough information to answer this: {trace.reason}.")


def create_app(pipeline: Pipeline, make_gate: Callable[[str], Gate], generator: Generator,
               models: dict) -> FastAPI:
    app = FastAPI(title="vaultgate")
    gates: dict[str, Gate] = {pipeline.gate.name: pipeline.gate}
    # One query at a time: a single-user demo does not need concurrent embedding on MPS.
    lock = threading.Lock()

    def run(req: Query) -> Trace:
        try:
            build_filter(req.filters)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        with lock:
            if req.gate and req.gate not in gates:
                try:
                    gates[req.gate] = make_gate(req.gate)
                except RuntimeError as exc:
                    raise HTTPException(503, str(exc)) from exc
            variant = pipeline.with_options(gates.get(req.gate) if req.gate else None, req.graph)
            return variant.run(req.query, req.filters)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": store.count(pipeline.client, pipeline.collection),
                "notes": pipeline.graph.number_of_nodes(), "gate": pipeline.gate.name,
                "graph": pipeline.use_graph, "models": models}

    @app.post("/search")
    def search(req: Query) -> dict:
        return run(req).to_dict()

    @app.post("/ask")
    def ask(req: Query) -> dict:
        trace = run(req)
        # Abstaining never calls the LLM: the answer is fixed and costs nothing.
        answer = generator.generate(req.query, trace.context) if trace.status == "answer" \
            else abstain_answer(trace)
        return {"trace": trace.to_dict(), "answer": asdict(answer)}

    return app


def build_app() -> FastAPI:
    from vault_rag.config import load_config
    from vault_rag.generate import from_config
    from vault_rag.pipeline import build_pipeline, make_gate

    cfg = load_config()
    if not cfg.openrouter_api_key:
        raise RuntimeError("the API needs OPENROUTER_API_KEY in .env for /ask")
    pipeline = build_pipeline(cfg)
    pipeline.embedder.encode(["warm up"])  # load bge-m3 now, not on the first request
    budget = cfg["retrieval"]["context_max_chars"]
    models = {"embedding": cfg["embedding"]["model"], "generator": cfg["llm"]["generator"],
              "jev": cfg["gate"]["jev"]["model"], "bge": cfg["gate"]["bge"]["model"]}
    return create_app(pipeline, lambda name: make_gate(cfg, name, budget),
                      from_config(cfg["llm"], cfg.openrouter_api_key), models)
