import sys
from types import SimpleNamespace

from src.answer_quality import (
    mean_answer_scores,
    score_answer_overlap,
    score_ragas_context_recall,
)
from src.eval_metrics import (
    get_gold_doc_ids,
    hit_rate_at_k,
    mean_metrics,
    recall_at_k,
    reciprocal_rank,
    score_query,
)


def test_recall_at_k_uses_gold_doc_coverage() -> None:
    results = [
        {"doc_id": "doc_a"},
        {"doc_id": "doc_x"},
        {"doc_id": "doc_b"},
    ]
    gold_doc_ids = {"doc_a", "doc_b"}

    assert recall_at_k(results, gold_doc_ids, top_k=1) == 0.5
    assert recall_at_k(results, gold_doc_ids, top_k=3) == 1.0


def test_reciprocal_rank_and_hit_rate() -> None:
    results = [
        {"doc_id": "doc_x"},
        {"doc_id": "doc_b"},
        {"doc_id": "doc_a"},
    ]
    gold_doc_ids = {"doc_a", "doc_b"}

    assert reciprocal_rank(results, gold_doc_ids) == 0.5
    assert hit_rate_at_k(results, gold_doc_ids, top_k=1) == 0.0
    assert hit_rate_at_k(results, gold_doc_ids, top_k=3) == 1.0


def test_score_query_and_mean_metrics() -> None:
    rows = [
        score_query(
            results=[{"doc_id": "doc_a"}, {"doc_id": "doc_x"}],
            gold_doc_ids={"doc_a", "doc_b"},
            ks=(1, 3),
        ),
        score_query(
            results=[{"doc_id": "doc_x"}, {"doc_id": "doc_b"}],
            gold_doc_ids={"doc_a", "doc_b"},
            ks=(1, 3),
        ),
    ]

    summary = mean_metrics(rows)

    assert rows[0]["recall@1"] == 0.5
    assert rows[1]["mrr"] == 0.5
    assert summary["recall@1"] == 0.25
    assert summary["hit_rate"] == 1.0


def test_get_gold_doc_ids_reads_supporting_facts() -> None:
    row = {
        "supporting_facts": [
            {"title": "Doc A", "sentence_id": 0},
            {"title": "Doc B", "sentence_id": 1},
            {"title": "Doc A", "sentence_id": 3},
        ]
    }

    assert get_gold_doc_ids(row) == {"Doc A", "Doc B"}


def test_answer_overlap_scores_token_match() -> None:
    scores = score_answer_overlap(
        predicted="Shirley Temple was a diplomat and actress",
        gold="Shirley Temple diplomat",
    )

    assert scores["precision"] > 0.0
    assert scores["recall"] == 1.0
    assert scores["f1"] > 0.0
    assert scores["keyword_hit"] == 1.0


def test_answer_overlap_handles_cjk_text() -> None:
    scores = score_answer_overlap(
        predicted="这篇讲道强调盼望喜乐与平安",
        gold="盼望喜乐与平安",
    )

    assert scores["precision"] > 0.0
    assert scores["recall"] == 1.0
    assert scores["f1"] > 0.0
    assert scores["keyword_hit"] == 1.0


def test_mean_answer_scores_averages_rows() -> None:
    summary = mean_answer_scores(
        [
            {"precision": 1.0, "recall": 0.5, "f1": 0.66, "keyword_hit": 1.0},
            {"precision": 0.5, "recall": 1.0, "f1": 0.66, "keyword_hit": 0.0},
        ]
    )

    assert summary["precision"] == 0.75
    assert summary["recall"] == 0.75
    assert summary["keyword_hit"] == 0.5


def test_score_ragas_context_recall_uses_id_metric(monkeypatch) -> None:
    class FakeSingleTurnSample:
        def __init__(self, retrieved_context_ids, reference_context_ids):
            self.retrieved_context_ids = retrieved_context_ids
            self.reference_context_ids = reference_context_ids

    class FakeIDBasedContextRecall:
        async def single_turn_ascore(self, sample):
            gold = set(sample.reference_context_ids)
            picked = set(sample.retrieved_context_ids)
            return len(gold & picked) / len(gold)

    monkeypatch.setitem(
        sys.modules,
        "ragas.dataset_schema",
        SimpleNamespace(SingleTurnSample=FakeSingleTurnSample),
    )
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        SimpleNamespace(IDBasedContextRecall=FakeIDBasedContextRecall),
    )

    scores = score_ragas_context_recall(
        retrieved_ids=["doc_a", "doc_x", "doc_b"],
        gold_doc_ids={"doc_a", "doc_b"},
    )

    assert scores["ragas_context_recall"] == 1.0
