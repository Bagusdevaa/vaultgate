import json

from eval.report import judge_sample, tables, write_report
from eval.variants import graph_only_gold, ranked_notes
from vault_rag.trace import Attempt, Candidate, GateResult, Trace


def cand(note, source="hybrid", score=None, kept=None):
    return Candidate(chunk_id=f"{note}#0", note=note, heading_path=[], text="t", source=source,
                     gate_score=score, kept=kept)


def trace_with(candidates, context):
    gate = GateResult(scores=[], kept=[], sufficient=True)
    return Trace(query="q", gate_name="jev", use_graph=True,
                 attempts=[Attempt(20, None, candidates, gate)], status="answer", context=context)


def test_ranked_notes_puts_kept_first_by_gate_score():
    cs = [cand("a", score=0.1, kept=False), cand("b", score=0.5, kept=True),
          cand("c", source="graph", score=0.9, kept=True), cand("b", score=0.3, kept=True)]
    assert ranked_notes(trace_with(cs, [])) == ["c", "b", "a"]


def test_graph_only_gold_ignores_notes_hybrid_already_found():
    cs = [cand("a"), cand("g1", source="graph"), cand("g2", source="graph"), cand("g2")]
    trace = trace_with(cs, context=[cs[1], cs[2]])
    assert graph_only_gold(trace, ["g1", "g2", "missing"]) == ["g1"]


def record(variant, category, correct, status="answer"):
    return {"id": f"{variant}-{category}", "variant": variant, "category": category,
            "question": "q", "reference_answer": "r", "answer": "a", "context_notes": [],
            "gold_notes": ["g"], "ranked_notes": ["g"], "graph_only_gold": [], "status": status,
            "seconds": 1.0, "tokens": 10, "cost": 0.001,
            "judge": {"correct": correct, "faithful": 1, "declined": int(status != "answer"),
                      "reason": "r"}}


RECORDS = [record("hybrid", "keyword", 1), record("hybrid", "unanswerable", 0),
           record("no_retrieval", "keyword", 0), record("no_retrieval", "unanswerable", 1, "abstain")]


def test_tables_follow_variant_order_and_split_categories():
    summary, text = tables(RECORDS)
    assert list(summary) == ["no_retrieval", "hybrid"]
    assert "| hybrid | 100% | – | – | 0% |" in text


def test_judge_sample_only_uses_judged_answers():
    text = judge_sample(RECORDS, share=1.0)
    assert "3 of 3 judged answers" in text and "abstain" not in text.split("\n")[0]


def test_write_report_creates_every_file(tmp_path):
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in RECORDS))
    write_report(tmp_path)
    for name in ("summary.md", "summary.csv", "judge_sample.md", "metrics_per_variant.png",
                 "correct_per_category.png", "cost_vs_correct.png"):
        assert (tmp_path / name).stat().st_size > 0
