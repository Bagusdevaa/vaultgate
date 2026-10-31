"""Summary tables, charts and the judge sample from a run's records.jsonl.

    uv run python -m eval.report eval/results/<run>
"""

import csv
import json
import random
import sys
from pathlib import Path

from eval.metrics import by, summarize
from eval.variants import VARIANTS

CATEGORIES = ["keyword", "semantic", "multihop", "unanswerable"]
COLUMNS = [  # (key, header, format)
    ("recall@5", "Recall@5", ".2f"), ("mrr", "MRR", ".2f"), ("correct", "Correct", ".0%"),
    ("correct_answerable", "Correct (answerable)", ".0%"), ("faithful", "Faithful", ".0%"),
    ("correct_abstain", "Correct abstain", ".0%"), ("hallucination", "Hallucination", ".0%"),
    ("false_abstain", "False abstain", ".0%"), ("graph_only_correct", "Graph-only", ".0%"),
    ("latency_p50", "p50 s", ".1f"), ("latency_p95", "p95 s", ".1f"),
    ("tokens", "Tokens/q", ",.0f"), ("cost", "Cost/q $", ".5f"),
]


def load(run_dir: Path) -> list[dict]:
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def ordered_variants(records: list[dict]) -> list[str]:
    present = {r["variant"] for r in records}
    return [v.name for v in VARIANTS if v.name in present]


def cell(value, spec: str) -> str:
    return "–" if value is None else format(value, spec)


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def tables(records: list[dict]) -> tuple[dict[str, dict], str]:
    groups, names = by(records, "variant"), ordered_variants(records)
    summary = {name: summarize(groups[name]) for name in names}
    main = markdown_table(["Variant", "n"] + [h for _, h, _ in COLUMNS],
                          [[name, str(summary[name]["questions"])] +
                           [cell(summary[name][k], f) for k, _, f in COLUMNS] for name in names])
    per_cat = []
    for name in names:
        cats = by(groups[name], "category")
        per_cat.append([name] + [cell(summarize(cats[c])["correct"], ".0%") if c in cats else "–"
                                 for c in CATEGORIES])
    category = markdown_table(["Variant"] + CATEGORIES, per_cat)
    counts = {c: len(by(groups[names[0]], "category").get(c, [])) for c in CATEGORIES}
    text = (f"# Evaluation results\n\nQuestions per category: "
            + ", ".join(f"{c} {n}" for c, n in counts.items())
            + f"\n\n## All metrics\n\n{main}\n\n## Correct answers per category\n\n{category}\n\n"
            "Recall@5 and MRR are over answerable questions, against `gold_notes`. Faithful is "
            "over answers that did not decline. Graph-only is the share of correct answerable "
            "answers whose gold note reached the context only through graph expansion. Cost "
            "covers the gate and the generator, not the judge.\n")
    return summary, text


def write_csv(path: Path, summary: dict[str, dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "questions"] + [k for k, _, _ in COLUMNS])
        for name, row in summary.items():
            writer.writerow([name, row["questions"]] + [row[k] for k, _, _ in COLUMNS])


def judge_sample(records: list[dict], share: float = 0.2, seed: int = 20261101) -> str:
    """A random share of the judged answers, for checking the judge by hand."""
    judged = [r for r in records if r["status"] == "answer"]
    picked = random.Random(seed).sample(judged, max(1, round(share * len(judged)))) if judged else []
    parts = [f"# Judge sample ({len(picked)} of {len(judged)} judged answers)\n\n"
             "Mark each verdict you disagree with. Agreement rate = 1 - disagreements / total.\n"]
    for r in sorted(picked, key=lambda r: (r["variant"], r["id"])):
        j = r["judge"]
        parts.append(f"\n## {r['id']} · {r['variant']} · {r['category']}\n\n"
                     f"**Question:** {r['question']}\n\n**Reference:** {r['reference_answer']}\n\n"
                     f"**Answer:** {r['answer']}\n\n**Context notes:** {', '.join(r['context_notes']) or '(none)'}\n\n"
                     f"**Judge:** correct={j['correct']} faithful={j['faithful']} declined={j['declined']}"
                     f" — {j['reason']}\n\n- [ ] I disagree\n")
    return "".join(parts)


def write_report(run_dir: Path) -> None:
    from eval.charts import write_charts

    records = load(run_dir)
    summary, text = tables(records)
    (run_dir / "summary.md").write_text(text)
    write_csv(run_dir / "summary.csv", summary)
    (run_dir / "judge_sample.md").write_text(judge_sample(records))
    write_charts(run_dir, records, summary)
    print(text)


if __name__ == "__main__":
    write_report(Path(sys.argv[1]))
