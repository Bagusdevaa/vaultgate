"""Run one query through the pipeline and print its trace.

    uv run python scripts/query.py "why was openrouter marked unhealthy?"
    uv run python scripts/query.py "..." --gate bge --no-graph --folder Bugs
    uv run python scripts/query.py "..." --json > trace.json
"""

import argparse
import json
import sys

from vault_rag.config import load_config
from vault_rag.pipeline import build_pipeline
from vault_rag.trace import Attempt, Candidate, Trace


def fmt(value, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


def where(c: Candidate, width: int = 70) -> str:
    text = " > ".join([c.note, *c.heading_path[1:]])
    return text if len(text) <= width else text[: width - 1] + "…"


def print_attempt(number: int, attempt: Attempt) -> None:
    print(f"\n== attempt {number}  top_k={attempt.top_k}  filters={attempt.filters}")
    print(f"{'#':>3}  {'source':6}  {'dense':>5}  {'sparse':>6}  {'rrf':>6}  {'gate':>5}  "
          f"{'':4}  note > heading")
    for i, c in enumerate(attempt.candidates, start=1):
        verdict = "keep" if c.kept else "drop"
        print(f"{i:>3}  {c.source:6}  {fmt(c.dense_rank):>5}  {fmt(c.sparse_rank):>6}  "
              f"{fmt(c.rrf_score or None, '.4f'):>6}  {fmt(c.gate_score, '.3f'):>5}  {verdict:4}  "
              f"{where(c)}" + (f"   (via {c.via}, {c.via_link})" if c.via else ""))
    gate = attempt.gate
    print(f"gate: kept {sum(gate.kept)}/{len(gate.kept)}  sufficiency {fmt(gate.sufficiency_score, '.3f')}"
          f"  -> {'answer' if gate.sufficient else 'abstain'}   ({gate.note})")
    print("timings: " + "  ".join(f"{k} {v:.2f}s" for k, v in attempt.timings.items()))


def print_trace(trace: Trace) -> None:
    print(f"query: {trace.query}")
    print(f"gate: {trace.gate_name}   graph: {'on' if trace.use_graph else 'off'}")
    for number, attempt in enumerate(trace.attempts, start=1):
        print_attempt(number, attempt)
    chars = sum(len(c.gate_text()) for c in trace.context)
    print(f"\nstatus: {trace.status.upper()}   context: {len(trace.context)} chunks, {chars} chars   "
          f"Jev tokens: {trace.input_tokens}  cost: ${trace.cost:.5f}   total: {trace.seconds:.2f}s")
    for c in trace.context:
        print(f"  - {where(c, 100)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--gate", choices=["none", "bge", "jev"], help="default: config gate.type")
    parser.add_argument("--no-graph", action="store_true", help="skip graph expansion")
    parser.add_argument("--folder", action="append", help="metadata filter, repeatable")
    parser.add_argument("--tag", action="append", help="metadata filter, repeatable")
    parser.add_argument("--json", action="store_true", help="print the full trace as JSON")
    args = parser.parse_args()

    filters = {k: v for k, v in (("folder", args.folder), ("tags", args.tag)) if v} or None
    pipeline = build_pipeline(load_config(), gate=args.gate,
                              use_graph=False if args.no_graph else None)
    trace = pipeline.run(args.query, filters)
    pipeline.close()
    if args.json:
        print(json.dumps(trace.to_dict(), indent=1, ensure_ascii=False))
    else:
        print_trace(trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
