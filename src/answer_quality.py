"""Small answer-quality proxy metrics."""

from __future__ import annotations

import asyncio
import importlib
import re
from statistics import mean

from src.cleaning import clean_text


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize_words(text: str) -> list[str]:
    value = clean_text(text).lower()
    return WORD_RE.findall(value) + CJK_RE.findall(value)


def score_answer_overlap(predicted: str, gold: str) -> dict[str, float]:
    predicted_tokens = tokenize_words(predicted)
    gold_tokens = tokenize_words(gold)

    if not predicted_tokens or not gold_tokens:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "keyword_hit": 0.0,
        }

    predicted_set = set(predicted_tokens)
    gold_set = set(gold_tokens)
    overlap = predicted_set & gold_set

    precision = len(overlap) / len(predicted_set)
    recall = len(overlap) / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "keyword_hit": 1.0 if gold_set <= predicted_set else 0.0,
    }


def mean_answer_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("rows should not be empty")
    keys = rows[0].keys()
    return {key: mean(row[key] for row in rows) for key in keys}


def score_ragas_context_recall(
    retrieved_ids: list[str],
    gold_doc_ids: set[str],
) -> dict[str, float]:
    """Optional RAGAS metric using document ids only."""
    try:
        dataset_schema = importlib.import_module("ragas.dataset_schema")
        metrics_mod = importlib.import_module("ragas.metrics")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ragas is required for optional RAGAS metrics. Install requirements first."
        ) from exc

    sample = dataset_schema.SingleTurnSample(
        retrieved_context_ids=retrieved_ids,
        reference_context_ids=sorted(gold_doc_ids),
    )
    metric = metrics_mod.IDBasedContextRecall()
    score = asyncio.run(metric.single_turn_ascore(sample))
    return {"ragas_context_recall": float(score)}
