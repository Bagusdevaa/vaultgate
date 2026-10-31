"""Run every variant over a golden split, judge the answers, and write the report.

    uv run python -m eval.run_eval --split test
    uv run python -m eval.run_eval --split dev --variants hybrid,hybrid_jev_gate --limit 3
    uv run python -m eval.run_eval --resume eval/results/20261031-120000   # finish a run

Records are appended one per (variant, question) as they finish, so a crashed run can
be resumed without paying for finished questions again.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eval.grep_baseline import rank_files, read_files
from eval.judge import Judge, Verdict
from eval.report import write_report
from eval.variants import BY_NAME, VARIANTS, Variant, graph_only_gold, ranked_notes
from vault_rag.api import abstain_answer
from vault_rag.config import REPO_ROOT, Config, load_config
from vault_rag.gate.base import Gate
from vault_rag.generate import Answer, Generator, format_sources
from vault_rag.ingest.index import read_vault
from vault_rag.pipeline import Pipeline, build_pipeline, make_gate


@dataclass
class Resources:
    cfg: Config
    pipeline: Pipeline
    generator: Generator
    judge: Judge
    notes: dict[str, str]
    gates: dict[str, Gate] = field(default_factory=dict)

    def gate(self, name: str) -> Gate:
        if name not in self.gates:
            self.gates[name] = make_gate(self.cfg, name, self.budget)
        return self.gates[name]

    @property
    def budget(self) -> int:
        return self.cfg["retrieval"]["context_max_chars"]


def load_resources(cfg: Config) -> Resources:
    from openai import OpenAI

    from vault_rag.generate import from_config

    llm = cfg["llm"]
    if "qwen" in llm["judge"].lower():
        raise SystemExit("the judge must not be from the generator's (Qwen) family")
    vault = read_vault(REPO_ROOT / cfg["paths"]["demo_vault"])
    pipeline = build_pipeline(cfg, gate="none")
    client = OpenAI(base_url=llm["base_url"], api_key=cfg.openrouter_api_key)
    return Resources(cfg, pipeline, from_config(llm, cfg.openrouter_api_key),
                     Judge(client, llm["judge"]), {p: f.text for p, f in vault.items()})


def retrieve(variant: Variant, item: dict, res: Resources) -> dict:
    """Context and retrieval fields for one question; no generation yet."""
    question = item["question"]
    if variant.retrieval == "none":
        return {"status": "answer", "context": [], "ranked_notes": None, "attempts": 0,
                "graph_only_gold": [], "gate_tokens": 0, "gate_cost": 0.0, "trace": None}
    if variant.retrieval == "grep":
        ranking = rank_files(res.notes, question)
        return {"status": "answer", "context": read_files(res.notes, ranking, res.budget),
                "ranked_notes": [p for p, _ in ranking], "attempts": 1, "graph_only_gold": [],
                "gate_tokens": 0, "gate_cost": 0.0, "trace": None}
    trace = res.pipeline.with_options(res.gate(variant.gate), variant.graph).run(question)
    return {"status": trace.status, "context": trace.context, "ranked_notes": ranked_notes(trace),
            "attempts": len(trace.attempts), "graph_only_gold": graph_only_gold(trace, item["gold_notes"]),
            "gate_tokens": sum(a.gate.input_tokens + a.gate.output_tokens for a in trace.attempts),
            "gate_cost": trace.cost, "trace": trace}


def run_question(variant: Variant, item: dict, res: Resources) -> dict:
    start = time.perf_counter()
    got = retrieve(variant, item, res)
    retrieval_seconds = time.perf_counter() - start
    answerable = item["category"] != "unanswerable"
    sources = format_sources(got["context"])
    if got["status"] == "answer":
        answer = res.generator.generate(item["question"], got["context"])
        verdict = res.judge.grade(item["question"], answerable, item["reference_answer"],
                                  sources, answer.text)
    else:
        # The pipeline abstained, so no LLM ran: the answer is the fixed abstain text.
        answer: Answer = abstain_answer(got["trace"])
        verdict = Verdict(correct=int(not answerable), faithful=1, declined=1,
                          reason="pipeline abstained; not sent to the judge")
    return {
        "id": item["id"], "category": item["category"], "variant": variant.name,
        "question": item["question"], "gold_notes": item["gold_notes"],
        "reference_answer": item["reference_answer"], "status": got["status"],
        "attempts": got["attempts"], "ranked_notes": got["ranked_notes"],
        "context_notes": list(dict.fromkeys(c.note for c in got["context"])),
        "context_chars": len(sources), "graph_only_gold": got["graph_only_gold"],
        "answer": answer.text, "citations": answer.citations,
        "unknown_citations": answer.unknown_citations,
        "seconds_retrieval": retrieval_seconds, "seconds_generate": answer.seconds,
        "seconds": retrieval_seconds + answer.seconds,
        "tokens": got["gate_tokens"] + answer.prompt_tokens + answer.completion_tokens,
        "cost": got["gate_cost"] + answer.cost,
        "judge": asdict(verdict),
    }


def done_keys(records_path: Path) -> set[tuple[str, str]]:
    if not records_path.exists():
        return set()
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    return {(r["variant"], r["id"]) for r in rows}


def run(out_dir: Path, split: str, variants: list[Variant], limit: int | None) -> None:
    cfg = load_config()
    items = [json.loads(line) for line in
             (REPO_ROOT / cfg["paths"]["golden_dir"] / f"{split}.jsonl").read_text().splitlines()
             if line.strip()][:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"split": split, "variants": [v.name for v in variants], "limit": limit,
                **{k: cfg[k] for k in ("llm", "gate", "retrieval", "embedding", "chunking")}}
    (out_dir / "run.json").write_text(json.dumps(settings, indent=1))
    records_path = out_dir / "records.jsonl"
    done = done_keys(records_path)
    res = load_resources(cfg)
    try:
        with records_path.open("a") as out:
            for variant in variants:
                for n, item in enumerate(items, start=1):
                    if (variant.name, item["id"]) in done:
                        continue
                    record = run_question(variant, item, res)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"{variant.name:22} {n:>3}/{len(items)} {item['id']} {record['status']:7} "
                          f"correct={record['judge']['correct']} {record['seconds']:.1f}s", flush=True)
    finally:
        res.pipeline.close()
    write_report(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--variants", help="comma-separated names; default: all six")
    parser.add_argument("--limit", type=int, help="first N questions only (smoke runs)")
    parser.add_argument("--resume", type=Path, help="results folder of an unfinished run")
    args = parser.parse_args()

    if args.resume:
        saved = json.loads((args.resume / "run.json").read_text())
        args.split, args.limit = saved["split"], saved["limit"]
        args.variants = args.variants or ",".join(saved["variants"])
    variants = [BY_NAME[n] for n in args.variants.split(",")] if args.variants else VARIANTS
    out_dir = args.resume or REPO_ROOT / "eval" / "results" / (
        time.strftime("%Y%m%d-%H%M%S") + f"-{args.split}")
    run(out_dir, args.split, variants, args.limit)
    print(f"\nresults: {out_dir}")


if __name__ == "__main__":
    main()
