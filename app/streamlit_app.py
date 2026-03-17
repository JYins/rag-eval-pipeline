"""Streamlit dashboard for retrieval result inspection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_METRICS_PATH = ROOT_DIR / "results" / "metrics_summary.csv"
DEFAULT_PER_QUERY_PATH = ROOT_DIR / "results" / "per_query_results.json"


@st.cache_data
def load_metrics(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_queries(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda item: (
            item["retrieval_metrics"]["recall@3"],
            item["retrieval_metrics"]["mrr"],
            item["answer_quality"]["f1"],
        ),
    )
    return ordered


def format_results_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    table_rows = []
    for item in results:
        row = {
            "rank": item["rank"],
            "doc_id": item["doc_id"],
            "chunk_id": item["chunk_id"],
            "is_supporting": item["is_supporting"],
            "text": item["text"],
        }
        if "score" in item:
            row["score"] = item["score"]
        if "hybrid_score" in item:
            row["hybrid_score"] = item["hybrid_score"]
            row["bm25_rank"] = item.get("bm25_rank")
            row["dense_rank"] = item.get("dense_rank")
        table_rows.append(row)
    return pd.DataFrame(table_rows)


def show_summary(metrics_df: pd.DataFrame) -> None:
    st.subheader("Cross-Configuration Summary")
    st.dataframe(metrics_df, use_container_width=True)

    chart_df = metrics_df.set_index("config_name")[["recall@3", "mrr", "hit_rate"]]
    st.subheader("Main Retrieval Metrics")
    st.bar_chart(chart_df)


def show_query_inspector(query_rows: list[dict[str, Any]]) -> None:
    st.subheader("Per-Query Inspector")

    query_map: dict[str, list[dict[str, Any]]] = {}
    for row in query_rows:
        query_map.setdefault(row["query_id"], []).append(row)

    options = [
        (query_id, rows[0]["question"])
        for query_id, rows in query_map.items()
    ]
    labels = {
        query_id: f"{question[:90]}{'...' if len(question) > 90 else ''}"
        for query_id, question in options
    }

    selected_query_id = st.selectbox(
        "Pick a question",
        options=[item[0] for item in options],
        format_func=lambda item: labels[item],
    )
    compare_rows = query_map[selected_query_id]
    compare_rows.sort(key=lambda item: item["config_name"])

    base_row = compare_rows[0]
    st.markdown(f"**Question:** {base_row['question']}")
    st.markdown(f"**Answer:** {base_row['answer']}")
    st.markdown(f"**Gold Docs:** {', '.join(base_row['gold_doc_ids'])}")

    compare_table = []
    for row in compare_rows:
        compare_table.append(
            {
                "config_name": row["config_name"],
                **row["retrieval_metrics"],
                **row["answer_quality"],
            }
        )
    st.dataframe(pd.DataFrame(compare_table), use_container_width=True)

    config_names = [row["config_name"] for row in compare_rows]
    picked_configs = st.multiselect(
        "Configs to inspect",
        options=config_names,
        default=config_names,
    )

    for row in compare_rows:
        if row["config_name"] not in picked_configs:
            continue
        with st.expander(row["config_name"], expanded=False):
            left, right = st.columns(2)
            left.metric("Recall@3", f"{row['retrieval_metrics']['recall@3']:.3f}")
            left.metric("MRR", f"{row['retrieval_metrics']['mrr']:.3f}")
            right.metric("Hit Rate", f"{row['retrieval_metrics']['hit_rate']:.3f}")
            right.metric("Answer F1", f"{row['answer_quality']['f1']:.3f}")
            st.dataframe(format_results_table(row["results"]), use_container_width=True)


def show_failure_cases(query_rows: list[dict[str, Any]]) -> None:
    st.subheader("Failure Cases")

    config_names = sorted({row["config_name"] for row in query_rows})
    picked_config = st.selectbox("Config", options=config_names)
    limit = st.slider("How many failure cases", min_value=3, max_value=20, value=8)

    rows = [row for row in query_rows if row["config_name"] == picked_config]
    worst_rows = build_failure_rows(rows)[:limit]

    for row in worst_rows:
        title = (
            f"{row['question']} "
            f"(Recall@3={row['retrieval_metrics']['recall@3']:.2f}, "
            f"MRR={row['retrieval_metrics']['mrr']:.2f}, "
            f"F1={row['answer_quality']['f1']:.2f})"
        )
        with st.expander(title, expanded=False):
            st.markdown(f"**Answer:** {row['answer']}")
            st.markdown(f"**Gold Docs:** {', '.join(row['gold_doc_ids'])}")
            st.dataframe(format_results_table(row["results"]), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="RAG Eval Dashboard",
        page_icon=":books:",
        layout="wide",
    )

    st.title("RAG Evaluation Dashboard")
    st.caption("Compare retrieval configs, inspect single-query behavior, and review failure cases.")

    metrics_path = st.sidebar.text_input("Metrics CSV", str(DEFAULT_METRICS_PATH))
    per_query_path = st.sidebar.text_input("Per-query JSON", str(DEFAULT_PER_QUERY_PATH))

    metrics_df = load_metrics(metrics_path)
    query_rows = load_queries(per_query_path)

    st.sidebar.markdown(f"**Configs loaded:** {metrics_df.shape[0]}")
    st.sidebar.markdown(f"**Query rows loaded:** {len(query_rows)}")

    summary_tab, query_tab, failure_tab = st.tabs(
        ["Summary", "Query Inspector", "Failure Cases"]
    )

    with summary_tab:
        show_summary(metrics_df)

    with query_tab:
        show_query_inspector(query_rows)

    with failure_tab:
        show_failure_cases(query_rows)


if __name__ == "__main__":
    main()
