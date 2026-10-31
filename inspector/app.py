"""Retrieval inspector: what every stage of the pipeline did for one query.

    make inspector      # needs the API: make serve

The inspector only talks to the API over HTTP, so it runs the same against the local
API and the home-server deploy (VAULTGATE_API_URL).
"""

import os

import httpx
import pandas as pd
import streamlit as st

from vault_rag.config import REPO_ROOT, load_config
from vault_rag.inspector.data import VARIANTS, call, compare_row, load_golden, request_body
from vault_rag.inspector.views import context_and_answer, trace_view

st.set_page_config(page_title="vaultgate inspector", layout="wide")
cfg = load_config()
API = os.getenv("VAULTGATE_API_URL") or cfg["mcp"]["api_url"]


@st.cache_data(ttl=30)
def health(api_url: str) -> dict | None:
    try:
        return call(api_url, "/health", timeout=5)
    except httpx.HTTPError:
        return None


def sidebar() -> dict:
    with st.sidebar:
        status = health(API)
        if status:
            st.success(f"API {API} · {status['chunks']} chunks · {status['notes']} notes")
        else:
            st.error(f"API not reachable at {API}. Start it with `make serve`.")
        mode = st.radio("Mode", ["Single variant", "Compare variants"])
        if mode == "Single variant":
            gate = st.selectbox("Gate", ["jev", "bge", "none"])
            graph = st.toggle("Graph expansion", value=True)
            chosen = {f"{gate} gate, graph {'on' if graph else 'off'}": (gate, graph)}
        else:
            names = st.multiselect("Variants", list(VARIANTS), default=list(VARIANTS))
            chosen = {n: VARIANTS[n] for n in names}
        folders = st.multiselect("Folder filter", ["Bugs", "Decisions", "Knowledge", "Patterns",
                                                   "Projects", "Handoffs"])
        tags = [t.strip() for t in st.text_input("Tag filter (comma-separated)").split(",") if t.strip()]
        generate = st.toggle("Generate answer (Qwen)", value=True)
    return {"mode": mode, "variants": chosen, "folders": folders, "tags": tags, "generate": generate}


def question_box() -> tuple[str, dict | None]:
    golden = load_golden(REPO_ROOT / cfg["paths"]["golden_dir"])
    item = None
    if golden:
        labels = ["(type your own)"] + [f"{g['id']} · {g['category']} · {g['question'][:90]}" for g in golden]
        pick = st.selectbox("Golden question", labels)
        item = golden[labels.index(pick) - 1] if pick != labels[0] else None
    query = st.text_area("Query", value=item["question"] if item else "", height=80)
    if item:
        st.caption(f"Gold notes: {', '.join(item['gold_notes']) or '(none: unanswerable)'} · "
                   f"reference: {item['reference_answer']}")
    return query.strip(), item


def run(query: str, gate: str, graph: bool, opts: dict) -> dict:
    body = request_body(query, gate, graph, opts["folders"], opts["tags"])
    if opts["generate"]:
        return call(API, "/ask", body)
    return {"trace": call(API, "/search", body), "answer": None}


def main() -> None:
    st.title("vaultgate · retrieval inspector")
    opts = sidebar()
    query, item = question_box()
    gold = set(item["gold_notes"]) if item else set()
    if not st.button("Run", type="primary", disabled=not query or not opts["variants"]):
        return
    results = {}
    for name, (gate, graph) in opts["variants"].items():
        with st.spinner(f"Running {name}…"):
            try:
                results[name] = run(query, gate, graph, opts)
            except httpx.HTTPError as exc:
                st.error(f"{name}: {exc}")
    if opts["mode"] == "Single variant":
        for result in results.values():
            trace_view(result["trace"], result["answer"], gold)
        return
    st.dataframe(pd.DataFrame([compare_row(n, r) for n, r in results.items()]), hide_index=True)
    for col, (name, result) in zip(st.columns(len(results) or 1), results.items()):
        with col:
            st.markdown(f"#### {name}")
            context_and_answer(result["trace"], result["answer"])
    st.subheader("Traces")
    for tab, result in zip(st.tabs(list(results) or ["–"]), results.values()):
        with tab:
            trace_view(result["trace"], None, gold, with_context=False)


main()
