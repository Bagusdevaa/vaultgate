import json

from vault_rag.inspector.data import (candidate_rows, compare_row, decision, gate_rows,
                                      load_golden, request_body, stage_metrics)


def cand(i, source="hybrid", kept=True, score=0.5, note=None):
    return {"chunk_id": f"n{i}#0", "note": note or f"a/Bugs/n{i}.md", "heading_path": [f"n{i}", "Fix"],
            "text": "t", "source": source, "via": "a/Bugs/n0.md" if source == "graph" else None,
            "via_link": "backlink" if source == "graph" else None, "dense_rank": i, "dense_score": 0.8,
            "sparse_rank": None, "sparse_score": None, "rrf_score": 0.016, "graph_score": 0.6,
            "gate_score": score, "kept": kept, "drop_reason": None if kept else "gate score low"}


GATE = {"scores": [], "kept": [True, False, True], "sufficient": True, "sufficiency_score": 0.8,
        "threshold": 0.2, "sufficiency_threshold": 0.5, "input_tokens": 900, "output_tokens": 9,
        "cost": 0.0006, "note": "n", "seconds": 1.0}
ATTEMPT = {"top_k": 20, "filters": None, "timings": {"search": 0.1, "gate": 1.0}, "gate": GATE,
           "candidates": [cand(1, score=0.9), cand(2, kept=False, score=0.1), cand(3, "graph", score=0.7)]}


def test_request_body_drops_empty_filters():
    assert request_body("q", "jev", True, [], [])["filters"] is None
    assert request_body("q", "bge", False, ["Bugs"], [])["filters"] == {"folder": ["Bugs"]}


def test_decision_tells_answer_retry_and_abstain_apart():
    one, two = [ATTEMPT], [ATTEMPT, ATTEMPT]
    assert decision({"status": "answer", "attempts": one, "reason": "r"})[0] == "answer"
    assert decision({"status": "answer", "attempts": two, "reason": "r"})[0] == "retry"
    kind, text = decision({"status": "abstain", "attempts": two, "reason": "r"})
    assert kind == "abstain" and text == "ABSTAIN: r"


def test_candidate_rows_split_stages_and_mark_gold():
    hybrid, graph = candidate_rows(ATTEMPT, gold={"a/Bugs/n3.md"})
    assert [r["rank"] for r in hybrid] == [1, 2]
    assert graph == [{"#": 3, "note": "a/Bugs/n3.md", "heading": "Fix", "gold": True, "gate": 0.7,
                      "kept": True, "via": "a/Bugs/n0.md", "link": "backlink", "similarity": 0.6}]


def test_gate_rows_sort_by_score_and_label_context():
    rows = gate_rows(ATTEMPT, context_ids={"n1#0"})
    assert [r["status"] for r in rows] == ["in context", "kept", "dropped"]
    assert rows[-1]["reason"] == "gate score low"


def test_stage_metrics_and_compare_row():
    m = stage_metrics(ATTEMPT)
    assert m["kept"] == "2/3" and m["sufficiency"] == "0.80" and m["gate tokens"] == "909"
    trace = {"status": "answer", "attempts": [ATTEMPT], "reason": "r", "seconds": 2.0,
             "cost": 0.0006, "context": [ATTEMPT["candidates"][0], ATTEMPT["candidates"][2]]}
    row = compare_row("v", {"trace": trace, "answer": {"seconds": 1.0, "cost": 0.0004}})
    assert row == {"variant": "v", "decision": "answer", "attempts": 1, "context notes": 2,
                   "graph in context": 1, "seconds": 3.0, "cost $": 0.001}


def test_load_golden_reads_existing_splits(tmp_path):
    assert load_golden(tmp_path) == []
    (tmp_path / "dev.jsonl").write_text(json.dumps({"id": "d1"}) + "\n")
    assert load_golden(tmp_path) == [{"id": "d1", "split": "dev"}]


def test_app_renders_and_reports_a_missing_api(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("VAULTGATE_API_URL", "http://127.0.0.1:9")   # nothing listens there
    app = Path(__file__).resolve().parents[1] / "inspector" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)
