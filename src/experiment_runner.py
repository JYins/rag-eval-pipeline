"""Run retrieval eval across config-defined experiments."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from src.answer_quality import (
    mean_answer_scores,
    score_answer_overlap,
    score_ragas_context_recall,
)
from src.data_loader import load_hotpot_subset
from src.data_loader_sermon import build_sermon_rows
from src.eval_metrics import get_gold_doc_ids, mean_metrics, score_query
from src.retriever_bm25 import build_bm25_retriever
from src.retriever_dense import load_encoder
from src.retriever_dense import build_dense_retriever
from src.retriever_hybrid import build_hybrid_retriever
from src.utils import load_yaml, resolve_path, save_csv, save_json


def chunk_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    chunking = config.get("chunking", {})
    strategy = chunking.get("strategy", "sentence")
    kwargs: dict[str, Any] = {}
    if strategy == "fixed":
        kwargs["chunk_size"] = int(chunking.get("chunk_size", 120))
        kwargs["overlap"] = int(chunking.get("overlap", 20))
    if strategy == "sentence":
        kwargs["max_sentences"] = int(chunking.get("max_sentences", 2))
    return kwargs


def build_experiment_name(config: dict[str, Any]) -> str:
    if config.get("name"):
        return str(config["name"])

    parts = [
        str(config["retrieval_mode"]),
        str(config.get("chunking", {}).get("strategy", "sentence")),
        f"top{config.get('top_k', 3)}",
    ]
    if config.get("model_name"):
        parts.append(str(config["model_name"]).split("/")[-1])
    if config.get("dense_backend") and config["dense_backend"] != "faiss":
        parts.append(str(config["dense_backend"]))
    return "_".join(parts)


def expand_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    base = deepcopy(config.get("base_experiment", {}))
    grid = config.get("grid", {})
    if not grid:
        return rows

    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for picks in product(*values):
        row = deepcopy(base)
        for key, value in zip(keys, picks):
            if key in {"chunk_strategy", "chunk_size", "overlap", "max_sentences"}:
                row.setdefault("chunking", {})
                chunk_key = key.replace("chunk_", "")
                row["chunking"][chunk_key] = value
            else:
                row[key] = value
        row["name"] = build_experiment_name(row)
        rows.append(row)
    return rows


def load_experiments(config_path: str) -> dict[str, Any]:
    config = load_yaml(config_path)
    experiments = [dict(item) for item in config.get("experiments", [])]
    experiments.extend(expand_grid(config))
    if not experiments:
        raise ValueError("config should include experiments or grid")

    for item in experiments:
        item["name"] = build_experiment_name(item)
    return {
        "dataset": dict(config.get("dataset", {})),
        "output": dict(config.get("output", {})),
        "experiments": experiments,
    }


def get_rows(dataset_config: dict[str, Any], limit_override: int | None = None) -> list[dict[str, Any]]:
    dataset_name = dataset_config.get("name", "hotpotqa")
    if dataset_name == "hotpotqa":
        path = dataset_config.get("path", "data/eval/hotpotqa_subset.json")
        rows = load_hotpot_subset(resolve_path(path))
    elif dataset_name == "sermon":
        rows = build_sermon_rows(
            questions_path=dataset_config.get("questions_path", "data/eval/sermon_questions.csv"),
            sermon_root=dataset_config.get("sermon_dir", "data/raw/sermons"),
        )
    else:
        raise ValueError(f"unknown dataset: {dataset_name}")

    if limit_override is not None:
        limit = int(limit_override)
    else:
        limit = int(dataset_config.get("limit", len(rows)))
    return rows[:limit]


def get_encoder_cache(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    cache: dict[str, Any] = {}
    for item in experiments:
        if item["retrieval_mode"] not in {"dense", "hybrid"}:
            continue
        model_name = item["model_name"]
        if model_name not in cache:
            cache[model_name] = load_encoder(model_name)
    return cache


def use_ragas(config: dict[str, Any]) -> bool:
    answer_quality = config.get("answer_quality", {})
    return bool(answer_quality.get("use_ragas", False))


def build_retriever(
    docs: list[dict[str, Any]],
    config: dict[str, Any],
    encoder_cache: dict[str, Any],
) -> Any:
    strategy = config.get("chunking", {}).get("strategy", "sentence")
    kwargs = chunk_kwargs_from_config(config)
    mode = config["retrieval_mode"]

    if mode == "bm25":
        return build_bm25_retriever(docs, strategy=strategy, **kwargs)
    if mode == "dense":
        model_name = config["model_name"]
        return build_dense_retriever(
            docs,
            strategy=strategy,
            model_name=model_name,
            encoder=encoder_cache[model_name],
            backend=str(config.get("dense_backend", "faiss")),
            **kwargs,
        )
    if mode == "hybrid":
        model_name = config["model_name"]
        return build_hybrid_retriever(
            docs,
            strategy=strategy,
            model_name=model_name,
            encoder=encoder_cache[model_name],
            dense_backend=str(config.get("dense_backend", "faiss")),
            bm25_weight=float(config.get("bm25_weight", 0.5)),
            dense_weight=float(config.get("dense_weight", 0.5)),
            **kwargs,
        )
    raise ValueError(f"unknown retrieval_mode: {mode}")


def search_docs(retriever: Any, query: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    retrieve_k = max(int(config.get("top_k", 3)), 5)
    if config["retrieval_mode"] == "hybrid":
        return retriever.search(query, top_k=retrieve_k, candidate_k=retrieve_k)
    return retriever.search(query, top_k=retrieve_k)


def trim_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        row = {
            "chunk_id": item["chunk_id"],
            "doc_id": item["doc_id"],
            "rank": item["rank"],
            "text": item["text"],
            "is_supporting": item.get("is_supporting", False),
        }
        if "score" in item:
            row["score"] = item["score"]
        if "hybrid_score" in item:
            row["hybrid_score"] = item["hybrid_score"]
            row["bm25_rank"] = item.get("bm25_rank")
            row["dense_rank"] = item.get("dense_rank")
        rows.append(row)
    return rows


def build_docs_key(docs: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple((doc["doc_id"], doc["text"]) for doc in docs)


def run_experiment(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    encoder_cache: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_rows = []
    metric_rows = []
    answer_rows = []
    ragas_rows = []
    retriever_cache: dict[tuple[tuple[str, str], ...], Any] = {}

    for row in rows:
        docs_key = build_docs_key(row["documents"])
        if docs_key not in retriever_cache:
            retriever_cache[docs_key] = build_retriever(row["documents"], config, encoder_cache)
        retriever = retriever_cache[docs_key]
        results = search_docs(retriever, row["question"], config)
        gold_doc_ids = get_gold_doc_ids(row)
        retrieval_scores = score_query(results, gold_doc_ids)

        top_k = int(config.get("top_k", 3))
        context_text = " ".join(item["text"] for item in results[:top_k])
        answer_scores = score_answer_overlap(context_text, row["answer"])
        ragas_scores = None
        if use_ragas(config):
            retrieved_ids = [item["doc_id"] for item in results[:top_k]]
            ragas_scores = score_ragas_context_recall(retrieved_ids, gold_doc_ids)

        metric_rows.append(retrieval_scores)
        answer_rows.append(answer_scores)
        if ragas_scores:
            ragas_rows.append(ragas_scores)
        query_rows.append(
            {
                "config_name": config["name"],
                "query_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "gold_doc_ids": sorted(gold_doc_ids),
                "retrieval_metrics": retrieval_scores,
                "answer_quality": answer_scores,
                "ragas_metrics": ragas_scores or {},
                "results": trim_results(results),
            }
        )

    retrieval_summary = mean_metrics(metric_rows)
    answer_summary = mean_answer_scores(answer_rows)
    ragas_summary = mean_metrics(ragas_rows) if ragas_rows else {}
    summary = {
        "config_name": config["name"],
        "retrieval_mode": config["retrieval_mode"],
        "chunk_strategy": config.get("chunking", {}).get("strategy", "sentence"),
        "top_k": int(config.get("top_k", 3)),
        "model_name": config.get("model_name", ""),
        "dense_backend": config.get("dense_backend", ""),
        "n_queries": len(rows),
        **retrieval_summary,
        **answer_summary,
        **ragas_summary,
    }
    return summary, query_rows


def run_eval(config_path: str, limit_override: int | None = None) -> dict[str, Any]:
    payload = load_experiments(config_path)
    rows = get_rows(payload["dataset"], limit_override=limit_override)
    encoder_cache = get_encoder_cache(payload["experiments"])

    summary_rows = []
    per_query_rows = []
    for config in payload["experiments"]:
        summary, query_rows = run_experiment(rows, config, encoder_cache)
        summary_rows.append(summary)
        per_query_rows.extend(query_rows)

    output = payload["output"]
    save_csv(summary_rows, output.get("metrics_path", "results/metrics_summary.csv"))
    save_json(per_query_rows, output.get("per_query_path", "results/per_query_results.json"))

    return {
        "summary_rows": summary_rows,
        "per_query_rows": per_query_rows,
        "metrics_path": output.get("metrics_path", "results/metrics_summary.csv"),
        "per_query_path": output.get("per_query_path", "results/per_query_results.json"),
    }
