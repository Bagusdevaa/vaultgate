"""Pure helpers for the inspector: API calls and turning a trace dict into table rows."""

import json
from pathlib import Path

import httpx

# Named variants for the compare mode: (gate, graph).
VARIANTS = {
    "hybrid (no gate)": ("none", False),
    "hybrid + bge gate": ("bge", False),
    "hybrid + Jev gate": ("jev", False),
    "hybrid + graph + Jev gate": ("jev", True),
}


def call(api_url: str, path: str, body: dict | None = None, timeout: float = 180) -> dict:
    with httpx.Client(base_url=api_url, timeout=timeout) as http:
        reply = http.post(path, json=body) if body is not None else http.get(path)
        reply.raise_for_status()
        return reply.json()


def request_body(query: str, gate: str, graph: bool, folders: list[str], tags: list[str]) -> dict:
    filters = {k: v for k, v in (("folder", folders), ("tags", tags)) if v} or None
    return {"query": query, "gate": gate, "graph": graph, "filters": filters}


def load_golden(golden_dir: Path) -> list[dict]:
    """dev + test questions if the (gitignored) files exist, else nothing."""
    items = []
    for split in ("dev", "test"):
        path = golden_dir / f"{split}.jsonl"
        if path.exists():
            items += [json.loads(line) | {"split": split}
                      for line in path.read_text().splitlines() if line.strip()]
    return items


def decision(trace: dict) -> tuple[str, str]:
    """("answer" | "retry" | "abstain", one-line banner text)."""
    tries = len(trace["attempts"])
    if trace["status"] == "answer" and tries > 1:
        return "retry", f"ANSWER after retry ({tries} attempts): {trace['reason']}"
    if trace["status"] == "answer":
        return "answer", f"ANSWER: {trace['reason']}"
    return "abstain", f"ABSTAIN: {trace['reason']}"


def heading(c: dict) -> str:
    return " > ".join(c["heading_path"][1:]) or "(top)"


def candidate_rows(attempt: dict, gold: set[str]) -> tuple[list[dict], list[dict]]:
    """Stage 1 (hybrid) and stage 2 (graph) rows, in pipeline order."""
    hybrid, graph = [], []
    for i, c in enumerate(attempt["candidates"], start=1):
        common = {"note": c["note"], "heading": heading(c), "gold": c["note"] in gold,
                  "gate": c["gate_score"], "kept": bool(c["kept"])}
        if c["source"] == "graph":
            graph.append({"#": i, **common, "via": c["via"], "link": c["via_link"],
                          "similarity": c["graph_score"]})
        else:
            hybrid.append({"rank": i, **common, "dense rank": c["dense_rank"],
                           "dense": c["dense_score"], "sparse rank": c["sparse_rank"],
                           "sparse": c["sparse_score"], "rrf": c["rrf_score"]})
    return hybrid, graph


def gate_rows(attempt: dict, context_ids: set[str]) -> list[dict]:
    """One bar per candidate for the stage 3 chart, best score first."""
    rows = []
    for i, c in enumerate(attempt["candidates"], start=1):
        status = "in context" if c["chunk_id"] in context_ids else ("kept" if c["kept"] else "dropped")
        rows.append({"label": f"{i:02d} {c['note'].split('/')[-1]} › {heading(c)}"[:70],
                     "score": c["gate_score"] or 0.0, "status": status, "source": c["source"],
                     "reason": c["drop_reason"] or ""})
    return sorted(rows, key=lambda r: -r["score"])


def stage_metrics(attempt: dict) -> dict[str, str]:
    gate = attempt["gate"]
    out = {f"{k} s": f"{v:.2f}" for k, v in attempt["timings"].items()}
    out["kept"] = f"{sum(gate['kept'])}/{len(gate['kept'])}"
    if gate["sufficiency_score"] is not None:
        out["sufficiency"] = f"{gate['sufficiency_score']:.2f}"
    out["gate tokens"] = f"{gate['input_tokens'] + gate['output_tokens']:,}"
    out["gate $"] = f"{gate['cost']:.5f}"
    return out


def compare_row(name: str, result: dict) -> dict:
    trace, answer = result["trace"], result.get("answer") or {}
    return {"variant": name, "decision": decision(trace)[0], "attempts": len(trace["attempts"]),
            "context notes": len({c["note"] for c in trace["context"]}),
            "graph in context": sum(c["source"] == "graph" for c in trace["context"]),
            "seconds": round(trace["seconds"] + answer.get("seconds", 0.0), 2),
            "cost $": round(trace["cost"] + answer.get("cost", 0.0), 5)}
