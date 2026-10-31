"""PNG charts for the slides: metrics per variant, correctness per category, cost vs correct."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

from eval.metrics import by, summarize  # noqa: E402

CATEGORIES = ["keyword", "semantic", "multihop", "unanswerable"]
MAIN = "hybrid_graph_jev_gate"


def grouped_bars(ax, labels: list[str], series: dict[str, list[float]]) -> None:
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        xs = [x + i * width - 0.4 + width / 2 for x in range(len(labels))]
        ax.bar(xs, values, width, label=name)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def metrics_chart(path: Path, summary: dict[str, dict]) -> None:
    names = list(summary)
    keys = {"Recall@5": "recall@5", "Correct": "correct", "Correct abstain": "correct_abstain",
            "Faithful": "faithful"}
    series = {label: [summary[n][k] or 0.0 for n in names] for label, k in keys.items()}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    grouped_bars(ax, names, series)
    ax.set_title("Metrics per variant")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def category_chart(path: Path, records: list[dict], names: list[str]) -> None:
    groups = by(records, "variant")
    series = {}
    for name in names:
        cats = by(groups[name], "category")
        series[name] = [summarize(cats[c])["correct"] if c in cats else 0.0 for c in CATEGORIES]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    grouped_bars(ax, CATEGORIES, series)
    ax.set_title("Correct answers per question category")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_chart(path: Path, summary: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, row in summary.items():
        ax.scatter(row["cost"] * 1000, row["correct"], s=80 if name == MAIN else 40,
                   marker="*" if name == MAIN else "o")
        ax.annotate(name, (row["cost"] * 1000, row["correct"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Cost per query (USD × 1000, gate + generator)")
    ax.set_ylabel("Correct")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.set_title("Cost vs correctness")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_charts(run_dir: Path, records: list[dict], summary: dict[str, dict]) -> None:
    metrics_chart(run_dir / "metrics_per_variant.png", summary)
    category_chart(run_dir / "correct_per_category.png", records, list(summary))
    cost_chart(run_dir / "cost_vs_correct.png", summary)
