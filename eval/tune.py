"""Pick gate thresholds on dev.jsonl only; test.jsonl is never used for tuning.

    uv run python -m eval.tune

Each gate scores the first attempt's candidates once; thresholds are then swept offline.
bge: the relevance threshold (it answers when any chunk passes). Jev: the sufficiency
threshold, with the relevance threshold kept at its config value.
"""

import json
import time

from eval.variants import BY_NAME
from vault_rag.config import REPO_ROOT, load_config
from vault_rag.gate.base import context_order, keep_flags
from vault_rag.pipeline import build_pipeline, make_gate

BGE_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
JEV_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def first_attempt(pipeline, variant, gate, item, budget):
    query = item["question"]
    embedding = pipeline.embedder.encode([query])[0]
    run = pipeline.with_options(gate, variant.graph)
    attempt, _ = run.attempt(query, embedding, pipeline.settings["top_k"], None)
    lengths = [len(c.gate_text()) for c in attempt.candidates]
    return {"id": item["id"], "answerable": item["category"] != "unanswerable",
            "gold": item["gold_notes"], "notes": [c.note for c in attempt.candidates],
            "scores": attempt.gate.scores, "lengths": lengths, "budget": budget,
            "sufficiency": attempt.gate.sufficiency_score}


def bge_outcome(row, threshold):
    kept = keep_flags(row["scores"], threshold)
    answer = any(kept)
    chosen = context_order(row["scores"], kept, row["lengths"], row["budget"])
    return answer, any(row["notes"][i] in row["gold"] for i in chosen)


def jev_outcome(row, threshold, relevance_threshold):
    kept = keep_flags(row["scores"], relevance_threshold)
    chosen = context_order(row["scores"], kept, row["lengths"], row["budget"])
    answer = row["sufficiency"] is not None and row["sufficiency"] >= threshold
    return answer, any(row["notes"][i] in row["gold"] for i in chosen)


def sweep(rows, grid, outcome) -> list[dict]:
    table = []
    for t in grid:
        results = [(row, *outcome(row, t)) for row in rows]
        table.append({"threshold": t,
                      "decision_accuracy": sum(a == r["answerable"] for r, a, _ in results) / len(rows),
                      "false_abstain": sum(r["answerable"] and not a for r, a, _ in results),
                      "wrong_answer": sum(not r["answerable"] and a for r, a, _ in results),
                      "gold_in_context": sum(r["answerable"] and g for r, _, g in results)})
    return table


def main() -> None:
    cfg = load_config()
    items = [json.loads(line) for line in
             (REPO_ROOT / cfg["paths"]["golden_dir"] / "dev.jsonl").read_text().splitlines() if line]
    budget = cfg["retrieval"]["context_max_chars"]
    pipeline = build_pipeline(cfg, gate="none")
    relevance = cfg["gate"]["jev"]["threshold"]
    try:
        bge_rows = [first_attempt(pipeline, BY_NAME["hybrid_bge_gate"], make_gate(cfg, "bge", budget),
                                  item, budget) for item in items]
        jev_gate = make_gate(cfg, "jev", budget)
        jev_rows = [first_attempt(pipeline, BY_NAME["hybrid_graph_jev_gate"], jev_gate, item, budget)
                    for item in items]
    finally:
        pipeline.close()
    answerable = sum(r["answerable"] for r in bge_rows)
    result = {"dev_questions": len(items), "answerable": answerable,
              "bge": sweep(bge_rows, BGE_GRID, bge_outcome),
              "jev": sweep(jev_rows, JEV_GRID, lambda r, t: jev_outcome(r, t, relevance))}
    for gate in ("bge", "jev"):
        print(f"\n{gate}: threshold  accuracy  false_abstain  wrong_answer  gold_in_context/{answerable}")
        for row in result[gate]:
            print(f"     {row['threshold']:>9}  {row['decision_accuracy']:>8.0%}  {row['false_abstain']:>13}"
                  f"  {row['wrong_answer']:>12}  {row['gold_in_context']:>15}")
    out = REPO_ROOT / "eval" / "results" / f"tune-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result | {"bge_rows": bge_rows, "jev_rows": jev_rows}, indent=1))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
