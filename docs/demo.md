# Demo script

Start the API and the inspector (`make serve`, `make inspector`), or the Docker stack
(`docker compose up -d`), then open http://127.0.0.1:8501. With the golden files present,
scenarios 1 and 2 are in the "Golden question" picker.

## 1. Answer with citations: golden d034 (semantic)

The question asks, in Indonesian, why the morning digest sends each news item as its own message. The answer exists only in the vault, so general knowledge cannot produce it.

- Mode: single variant, gate `jev`, graph on, generate answer on.
- Show stage 3: the chunk from `bagusdeva/Patterns/mira.md` passes the red threshold line, and the sufficiency score clears 0.5.
- The answer says each item collects its own 👍/👎 reactions, which Mira reads the next morning as feedback, and cites `[bagusdeva/Patterns/mira.md]`. The citation is the gold note.
- Then switch to compare mode: in the test run, `hybrid (no gate)` said the notes do not cover it and the bge gate abstained. Only the Jev variants ranked the pattern note into the context.

## 2. Correct abstain: golden d074 (unanswerable)

- Mode: compare variants, all four.
- The table shows the difference:
  - `hybrid (no gate)` and `hybrid + bge gate` both send context to Qwen.
  - Both Jev variants abstain after 2 attempts (sufficiency ~0.1) and never call the LLM, so they cost nothing to answer.
- Point out the retry tab: attempt 2 ran without filters and with top 40 before giving up.

## 3. Evidence found only through graph expansion

- Query: `API key read-only Healthchecks bisa ngapain aja, ping URL ikut kelihatan gak?`
- Folder filter: `Projects`.
- Gate `jev`. Run it once with graph on, then once with graph off (or use compare mode).

| | Graph on | Graph off |
|---|---|---|
| Attempt 1 | **answers** (sufficiency 0.96) | nothing in `Projects` passes the gate |
| Evidence | one chunk from `Knowledge/healthchecks.md`, added by graph via the link from `Projects/mira.md` | – |
| Result | answer on the first try | answers only after the unfiltered retry |

Stage 2 shows the graph row highlighted, with `via bagusdeva/Projects/mira.md`.

Why this scenario uses a filter: in this vault, project notes summarize their linked notes, so unfiltered hybrid search usually reaches the answer by itself. The evaluation shows that too (README, "What the numbers say").

## 4. In the real workflow: Claude Code

With `make serve` running, ask Claude Code something the vault knows, e.g. *"why did the Hermes gateway mark openrouter unhealthy? check my vault"*. Claude calls `search_vault`, gets the gated passages with note paths, and answers with citations. Ask about something the vault does not cover, and the tool returns `abstain`, so Claude says the notes do not cover it.
