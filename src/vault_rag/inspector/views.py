"""Streamlit components for one trace: decision, the three stages, context and answer."""

import altair as alt
import pandas as pd
import streamlit as st

from vault_rag.inspector.data import candidate_rows, decision, gate_rows, stage_metrics

STATUS_COLORS = alt.Scale(domain=["in context", "kept", "dropped"],
                          range=["#2e7d32", "#9ccc65", "#bdbdbd"])
SCORE = st.column_config.NumberColumn(format="%.3f")


def banner(trace: dict) -> None:
    kind, text = decision(trace)
    {"answer": st.success, "retry": st.warning, "abstain": st.error}[kind](text)


def stage_tables(attempt: dict, gold: set[str]) -> None:
    hybrid, graph = candidate_rows(attempt, gold)
    st.markdown(f"**Stage 1 · Hybrid search** — top {attempt['top_k']}, filters: "
                f"`{attempt['filters'] or 'none'}`. Dense and sparse lists fused with RRF.")
    st.dataframe(pd.DataFrame(hybrid), hide_index=True,
                 column_config={"dense": SCORE, "sparse": SCORE, "gate": SCORE,
                                "rrf": st.column_config.NumberColumn(format="%.4f")})
    st.markdown("**Stage 2 · Graph expansion** — best chunk of each linked or backlinking note "
                "of the top hits.")
    if graph:
        st.dataframe(pd.DataFrame(graph).style.set_properties(**{"background-color": "#fff8e1"}),
                     hide_index=True,
                     column_config={"similarity": SCORE, "gate": SCORE})
    else:
        st.caption("No graph candidates (graph off, or every neighbor was already a candidate).")


def gate_chart(attempt: dict, context_ids: set[str]) -> None:
    gate = attempt["gate"]
    st.markdown(f"**Stage 3 · Gate** — {gate['note']}")
    rows = pd.DataFrame(gate_rows(attempt, context_ids))
    if rows.empty:
        st.caption("No candidates.")
        return
    bars = alt.Chart(rows).mark_bar().encode(
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="gate score"),
        y=alt.Y("label:N", sort=None, title=None),
        color=alt.Color("status:N", scale=STATUS_COLORS, legend=alt.Legend(orient="bottom")),
        tooltip=["label", alt.Tooltip("score:Q", format=".3f"), "status", "source", "reason"])
    layers = [bars]
    if gate["threshold"] is not None:
        layers.append(alt.Chart(pd.DataFrame({"x": [gate["threshold"]]}))
                      .mark_rule(color="#c62828", strokeDash=[4, 3]).encode(x="x:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=max(160, 18 * len(rows))), width="stretch")
    threshold = "–" if gate["threshold"] is None else gate["threshold"]
    line = f"Relevance threshold (red line): {threshold}."
    if gate["sufficiency_score"] is not None:
        need = gate.get("sufficiency_threshold")
        line += (f" Sufficiency: {gate['sufficiency_score']:.2f}"
                 + (f" vs threshold {need}" if need is not None else "")
                 + f" → {'answer' if gate['sufficient'] else 'abstain'}.")
    st.caption(line)


def attempt_view(attempt: dict, gold: set[str], context_ids: set[str]) -> None:
    stage_tables(attempt, gold)
    gate_chart(attempt, context_ids)
    cols = st.columns(len(metrics := stage_metrics(attempt)))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)


def context_and_answer(trace: dict, answer: dict | None) -> None:
    chars = sum(len(c["text"]) for c in trace["context"])
    st.subheader(f"Context sent to the LLM · {len(trace['context'])} chunks, {chars:,} chars")
    if not trace["context"]:
        st.caption("Empty: the pipeline abstained, so the LLM is not called.")
    for c in trace["context"]:
        tag = f" · graph via {c['via'].split('/')[-1]} ({c['via_link']})" if c["source"] == "graph" else ""
        with st.expander(f"{c['note']} › {' > '.join(c['heading_path'][1:]) or '(top)'}{tag}"):
            st.text(c["text"])
    if answer is None:
        return
    st.subheader("Answer")
    st.markdown(answer["text"])
    if answer.get("unknown_citations"):
        st.warning("Cites notes that were not in the context: " + ", ".join(answer["unknown_citations"]))
    if answer.get("model"):
        st.caption(f"{answer['model']} · {answer['seconds']:.1f}s · "
                   f"{answer['prompt_tokens'] + answer['completion_tokens']:,} tokens · ${answer['cost']:.5f}")


def trace_view(trace: dict, answer: dict | None, gold: set[str], with_context: bool = True) -> None:
    banner(trace)
    context_ids = {c["chunk_id"] for c in trace["context"]}
    attempts = trace["attempts"]
    tabs = st.tabs([f"Attempt {i}" + (" (retry: no filters, larger top_k)" if i > 1 else "")
                    for i in range(1, len(attempts) + 1)])
    for tab, attempt in zip(tabs, attempts):
        with tab:
            attempt_view(attempt, gold, context_ids if attempt is attempts[-1] else set())
    st.caption(f"Total retrieval {trace['seconds']:.2f}s · gate tokens {trace['input_tokens']:,} in · "
               f"gate cost ${trace['cost']:.5f}")
    if with_context:
        context_and_answer(trace, answer)
