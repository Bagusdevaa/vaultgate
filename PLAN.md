# Vault RAG: Graph-Expanded Hybrid RAG with Jev Gate over Obsidian

Implementation plan for Claude Code. Final project for Sertifikasi AI Super Class (LLM track). Deadline: 2026-11-01.

## How to use this plan (instructions for Claude Code)

- Work phase by phase. At the end of each phase, run the acceptance checks, summarize what was built, and STOP for my review before starting the next phase.
- Use plan mode at the start of each phase. Propose the file changes first, then implement.
- NEVER modify the original Obsidian vault. Read-only. All copies go to `./data/demo-vault/`.
- NEVER commit secrets. `.env` must be in `.gitignore` from the first commit. Provide `.env.example`.
- Every component that calls an external API (Jev, OpenRouter) must be behind an interface so it can be swapped or mocked.
- Write small tests for anything that produces numbers (metrics, fusion, thresholds). Metric bugs are silent.
- Prefer simple, readable code over clever abstractions. I need to explain this in a presentation.

## Goal

A retrieval system over my Obsidian vault that:
1. Retrieves context with hybrid search (dense + sparse) using an open-source embedding model.
2. Expands candidates through the vault's own link graph (wikilinks + backlinks), a lightweight Graph RAG.
3. Gates candidates with Jev (TypeSafe AI): per-chunk relevance probability + "is the evidence sufficient?" decision. Abstains when the answer is not in the vault.
4. Is usable from my real workflow (Claude Code / Hermes via an MCP tool) and from a standalone `/ask` endpoint for the demo.
5. Is evaluated against baselines on a golden set, with results visible in a retrieval inspector UI.

The instructor explicitly asked to highlight the retrieval features. The inspector and the evaluation are the main deliverables, not the chat answer.

## Stack

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.11+, `uv` | |
| Embedding | `BAAI/bge-m3`, self-hosted | Dense + sparse from one model (FlagEmbedding). Open source. |
| Vector DB | Qdrant | Local mode during dev, Docker on home server for deploy. Dense + sparse in one collection, server-side RRF fusion. |
| Graph | NetworkX | Built from wikilinks, backlinks. |
| Gate (main) | Jev `typesafe/jev-1.14` via OpenRouter Decisions API | `POST https://openrouter.ai/api/alpha/decisions`. Same `OPENROUTER_API_KEY` as the generator, no TypeSafe account. NOT the chat completions endpoint. Pin the version, never `jev-latest`. Endpoint is alpha: keep all Jev code inside `gate/jev_gate.py`. |
| Gate (fallback) | `BAAI/bge-reranker-v2-m3` + threshold | Self-hosted, open source. Same interface as Jev gate. |
| Generator (experiments) | Qwen open-weights model via OpenRouter | OpenAI-compatible client, temperature 0. Slug in config, never changed mid-experiment. |
| Judge (experiments) | Non-Qwen model via OpenRouter | Must differ from the generator family. |
| API | FastAPI | `/search`, `/ask`, `/health` |
| Workflow integration | MCP server exposing `search_vault` | Returns context + sources or abstain signal. Generation done by the harness LLM. |
| Inspector UI | Streamlit | |
| Deploy | Docker Compose on home server (Ubuntu Server) | |

## Repo structure

```
vault-rag/
  PLAN.md
  README.md
  pyproject.toml
  .env.example
  config.yaml                 # thresholds, top_k, model slugs, graph caps
  docker-compose.yml
  data/
    demo-vault/               # sanitized copy, gitignored except a small sample
    golden/
      dev.jsonl               # 10-15 questions, for threshold tuning only
      test.jsonl              # 40-60 questions, never used for tuning
  src/vault_rag/
    ingest/
      parse.py                # frontmatter, headings, wikilinks, tags
      chunk.py                # chunk per heading, metadata
      index.py                # incremental indexing by file hash
    graph/
      build.py                # NetworkX graph from links
      expand.py               # 1-hop expansion with caps
    retrieve/
      embed.py                # bge-m3 dense + sparse
      hybrid.py               # Qdrant hybrid query with RRF
    gate/
      base.py                 # Gate interface: score(query, candidates) -> scored, sufficient
      jev_gate.py
      bge_gate.py
      none_gate.py            # passthrough for baseline variants
    pipeline.py               # retrieve -> expand -> gate -> (retry) -> result
    generate.py               # OpenAI-compatible generator, citations required
    api.py                    # FastAPI
    mcp_server.py             # search_vault tool
  eval/
    variants.py               # variant definitions
    grep_baseline.py          # emulates current harness behavior (keyword grep + read files)
    run_eval.py
    metrics.py
    judge.py
    report.py                 # tables + charts
  inspector/
    app.py                    # Streamlit
  scripts/
    build_demo_vault.py
    scan_sensitive.py
  tests/
```

## Phase 0: Setup and risk check (Oct 21)

Tasks:
- Init repo, `uv`, `.gitignore`, `.env.example`, `config.yaml`.
- Jev smoke test via OpenRouter Decisions API (`typesafe/jev-1.14`): read OpenRouter's Jev tutorial and Decisions API reference first. With a toy query and 5 toy documents, send (a) one noul question per document ("Does this passage help answer the query?") in a single packed request, and (b) one sufficiency question over the kept passages. Log per-document probabilities, latency, and token usage/cost.
- OpenRouter smoke test: one call to the chosen Qwen open-weights model and one to the judge model.
- bge-m3 smoke test: encode 3 sentences, print dense dim and sparse token count. Report RAM usage.

Acceptance:
- All three smoke tests pass and are runnable via `make smoke` (or `uv run`).
- Report: Jev latency, any pricing/rate-limit info found, bge-m3 RAM usage.

STOP: if the Decisions API is blocked or its schema is unclear, switch the main gate to `bge_gate` and keep Jev as optional.

## Phase 1: Demo vault (Oct 21, parallel with golden set work)

Tasks:
- `build_demo_vault.py`: survey vault (folders, note counts, link density) by sampling, not reading everything.
- Propose 2-3 seed projects that are personal/technical, not Investland, not client work, not personal life/finance/family. Show 1-hop subgraph size and link count for each. STOP and wait for my choice.
- Copy seed notes + 1-hop neighbors (max 5 neighbors per note, skip Daily notes) to `./data/demo-vault/`, preserve folders. Target 50-150 notes.
- `scan_sensitive.py`: regex scan for emails, phone numbers, API keys/tokens, IPs, internal URLs, currency amounts, and a deny-list of terms from config. Output `flagged_report.md`. Do not auto-redact.
- Report broken links (pointing outside demo vault).

Acceptance:
- Demo vault exists, note count in range, flagged report generated.
- I manually review flagged files before anything is sent to external APIs.

## Phase 2: Ingest and index (Oct 22)

Tasks:
- Parse: frontmatter, headings, wikilinks (`[[note]]`, `[[note|alias]]`, `[[note#heading]]`), tags, folder.
- Chunk per heading. Very short sections merge with the next; very long sections split with overlap. Each chunk carries: note path, heading path, folder, tags, outgoing links, modified time, chunk id.
- Embed with bge-m3 (dense + sparse), upsert to Qdrant with payload.
- Incremental indexing: file hash table, only re-embed changed files, delete chunks of removed files.
- Build link graph (NetworkX), persist to disk.

Acceptance:
- Full index of demo vault completes, report chunk count and time.
- Re-running index with no changes does zero embedding work.
- Editing one note re-indexes only that note.
- Unit tests for wikilink parsing edge cases.

## Phase 3: Retrieval, graph expansion, gate (Oct 23-24)

Tasks:
- Hybrid query: dense + sparse in Qdrant, RRF fusion, top 20 candidates. Optional metadata filter (folder, tags).
- Graph expansion: from top 3-5 notes, take 1-hop neighbors (links + backlinks), cap neighbors per note (config), and from each neighbor take only the chunk most similar to the query. Mark these candidates with `source="graph"`.
- Gate interface with three implementations: none, bge-reranker + threshold, Jev.
  - Per candidate: relevance probability/score, keep/drop by threshold.
  - Overall: sufficiency decision (answer vs abstain).
- Retry loop: if abstain, retry once with relaxed settings (no metadata filter, larger top_k). Max retries in config. Then final abstain.
- `pipeline.py` returns a full trace object: candidates with scores at every stage, which were dropped and why, graph-added candidates, gate decisions, retry attempts, timings, token counts. The inspector and the eval both consume this trace.

Acceptance:
- One CLI command runs a query and prints the trace.
- Same query through all gate implementations works via config switch.
- Tests for RRF fusion and threshold logic.

## Phase 4: Generation, API, MCP (Oct 25)

Tasks:
- `generate.py`: OpenAI-compatible client, base URL and model from config. System prompt: answer only from provided context, cite note paths, say clearly when context is insufficient. Temperature 0.
- FastAPI:
  - `POST /search` -> trace (context + sources, or abstain)
  - `POST /ask` -> trace + generated answer (uses Qwen)
  - `GET /health`
- MCP server with one tool `search_vault(query, filters?)` that returns compact context + sources, or `{"status": "abstain", "reason": ...}`. No generation inside the tool; the harness LLM generates.
- Register the MCP server in my local Claude Code and verify one real query end to end.

Acceptance:
- `/ask` returns an answer with citations for an answerable question and abstains for an unanswerable one.
- Claude Code can call `search_vault` and use the result.

## Phase 5: Retrieval inspector (Oct 26)

Streamlit app, single page:
- Query box, variant selector (gate type, graph on/off).
- Stage 1: hybrid candidates table (rank, note, heading, dense score, sparse score, fused rank).
- Stage 2: graph-added candidates, highlighted, with the link that brought them in.
- Stage 3: gate scores per candidate with threshold line, kept vs dropped.
- Decision banner: answer / retry / abstain.
- Final context sent to LLM, with source paths, and the generated answer.
- Timings and token counts per stage.
- A "compare variants" mode: same query, side by side.

Acceptance:
- I can demo 3 scenarios: answer with citations, correct abstain, answer only found through graph expansion.

## Phase 6: Evaluation (Oct 27-28)

Golden set format (`jsonl`), written or reviewed by me, not generated end to end by AI:
```json
{"id": "q001", "question": "...", "category": "keyword|semantic|multihop|unanswerable", "gold_notes": ["Decisions/x.md"], "reference_answer": "..."}
```
Category mix for test set: roughly 25% keyword, 25% semantic, 30% multi-hop, 20% unanswerable.

Variants:
1. `no_retrieval`: LLM answers with no context.
2. `grep_baseline`: emulates current harness behavior (keyword grep over files, read top matching files, same token budget cap).
3. `hybrid`
4. `hybrid_bge_gate`
5. `hybrid_jev_gate`
6. `hybrid_graph_jev_gate` (main system)

Rules for fairness:
- Same generator, prompt, temperature 0, and context token budget for all variants.
- Thresholds tuned on `dev.jsonl` only. `test.jsonl` is run once per variant after tuning is frozen.
- Judge model differs from generator family.

Metrics:
- Retrieval (deterministic, no LLM): Recall@5, MRR against `gold_notes`.
- Answer: correctness vs reference (judge, 0/1 or 1-3 rubric), faithfulness (claims supported by context, judge).
- Gate: on unanswerable questions, correct abstain rate and hallucination rate. On answerable questions, false abstain rate.
- Graph contribution: % of correct answers where the gold note entered only via graph expansion.
- Ops: latency p50/p95, tokens per query, cost per query.

Output:
- `eval/results/<timestamp>/` with raw per-question records (jsonl), summary table (markdown + csv), and charts (PNG): metrics per variant, breakdown per category, cost vs correctness scatter.
- Judge validation: export 20% random sample with judge verdicts to a markdown file for my manual check.

Acceptance:
- Full eval runs with one command.
- Metric functions have unit tests on tiny hand-made cases.

## Phase 7: Deploy to home server (Oct 29)

Tasks:
- `docker-compose.yml`: `qdrant`, `api` (FastAPI + embedding + reranker in-process), `inspector` (Streamlit). Vault mounted read-only. Volumes for Qdrant data and HF model cache.
- Check RAM on the server before deploying (bge-m3 + bge-reranker + Qdrant need several GB). If tight, make the bge gate lazy-loaded or disabled in prod config.
- Indexer runs as a scheduled job or watcher container.
- Expose only on LAN (or behind Tailscale). No public ports.
- Health checks, restart policies, `.env` on server only.

Acceptance:
- `docker compose up -d` brings everything up, `/health` OK, inspector reachable from my laptop on LAN.
- MCP server on my laptop can point to the home server API.

## Phase 8: Docs and presentation material (Oct 30)

- README: problem, architecture diagram, how to run, results table, limitations.
- Limitations section must include: small golden set (differences of a few % may not be significant), Jev is a proprietary API (open-source fallback provided), demo vault is a sanitized subset.
- Export key charts for slides.

## Oct 31: Buffer, LinkedIn post, Classroom upload. Shut down anything not needed.

## Non-goals

- No fine-tuning.
- No Microsoft-style GraphRAG (LLM entity extraction, community summaries). We use the vault's existing link graph.
- No query rewriting with an LLM.
- No public deployment.
