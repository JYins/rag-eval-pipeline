"""Retrieval metrics for experiment eval."""

from __future__ import annotations

from statistics import mean
from typing import Any


def get_gold_doc_ids(row: dict[str, Any]) -> set[str]:
    return {item["title"] for item in row.get("supporting_facts", [])}


def top_doc_ids(results: list[dict[str, Any]], top_k: int) -> list[str]:
    if top_k <= 0:
        raise ValueError("top_k should be > 0")
    return [item["doc_id"] for item in results[:top_k]]


def recall_at_k(
    results: list[dict[str, Any]],
    gold_doc_ids: set[str],
    top_k: int,
) -> float:
    if not gold_doc_ids:
        return 0.0
    picked = set(top_doc_ids(results, top_k))
    return len(picked & gold_doc_ids) / len(gold_doc_ids)


def reciprocal_rank(
    results: list[dict[str, Any]],
    gold_doc_ids: set[str],
) -> float:
    for rank, item in enumerate(results, start=1):
        if item["doc_id"] in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(
    results: list[dict[str, Any]],
    gold_doc_ids: set[str],
    top_k: int,
) -> float:
    return 1.0 if recall_at_k(results, gold_doc_ids, top_k) > 0 else 0.0


def score_query(
    results: list[dict[str, Any]],
    gold_doc_ids: set[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    metrics = {
        "mrr": reciprocal_rank(results, gold_doc_ids),
        "hit_rate": hit_rate_at_k(results, gold_doc_ids, top_k=max(ks)),
    }
    for top_k in ks:
        metrics[f"recall@{top_k}"] = recall_at_k(results, gold_doc_ids, top_k=top_k)
    return metrics


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("rows should not be empty")
    keys = rows[0].keys()
    return {key: mean(row[key] for row in rows) for key in keys}
