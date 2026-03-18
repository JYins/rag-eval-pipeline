import src.experiment_runner as experiment_runner
from src.experiment_runner import (
    dedupe_results_by_doc_id,
    load_experiments,
    rerank_results_with_doc_penalty,
    run_eval,
    run_experiment,
)


def test_load_experiments_reads_sermon_bm25_config() -> None:
    payload = load_experiments("configs/sermon.yaml")

    assert payload["dataset"]["name"] == "sermon"
    assert len(payload["experiments"]) == 3
    assert payload["experiments"][0]["name"] == "bm25_sentence_top3_sermon"
    assert payload["experiments"][0]["retrieval_mode"] == "bm25"
    assert payload["experiments"][1]["retrieval_mode"] == "dense"
    assert payload["experiments"][2]["retrieval_mode"] == "hybrid"


def test_load_experiments_reads_chromadb_ragas_config() -> None:
    payload = load_experiments("configs/sermon_chromadb_ragas.yaml")

    assert payload["dataset"]["name"] == "sermon"
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["dense_backend"] == "chromadb"
    assert payload["experiments"][0]["answer_quality"]["use_ragas"] is True


def test_load_experiments_reads_doc_dedup_config() -> None:
    payload = load_experiments("configs/sermon_doc_dedup.yaml")

    assert payload["dataset"]["name"] == "sermon"
    assert len(payload["experiments"]) == 3
    assert payload["experiments"][0]["dedupe_docs"] is True
    assert payload["experiments"][1]["dedupe_docs"] is True
    assert payload["experiments"][2]["dedupe_docs"] is True


def test_load_experiments_reads_doc_penalty_config() -> None:
    payload = load_experiments("configs/sermon_doc_penalty.yaml")

    assert payload["dataset"]["name"] == "sermon"
    assert len(payload["experiments"]) == 3
    assert payload["experiments"][0]["doc_repeat_penalty"] == 2.0
    assert payload["experiments"][1]["doc_repeat_penalty"] == 2.0
    assert payload["experiments"][2]["doc_repeat_penalty"] == 2.0


def test_dedupe_results_by_doc_id_keeps_best_chunk_per_doc() -> None:
    results = [
        {"chunk_id": "a_1", "doc_id": "doc_a", "rank": 1, "text": "a1"},
        {"chunk_id": "a_2", "doc_id": "doc_a", "rank": 2, "text": "a2"},
        {"chunk_id": "b_1", "doc_id": "doc_b", "rank": 3, "text": "b1"},
        {"chunk_id": "c_1", "doc_id": "doc_c", "rank": 4, "text": "c1"},
    ]

    picked = dedupe_results_by_doc_id(results, top_k=3)

    assert [item["chunk_id"] for item in picked] == ["a_1", "b_1", "c_1"]
    assert [item["rank"] for item in picked] == [1, 2, 3]


def test_rerank_results_with_doc_penalty_softens_duplicates() -> None:
    results = [
        {"chunk_id": "a_1", "doc_id": "doc_a", "rank": 1, "text": "a1"},
        {"chunk_id": "a_2", "doc_id": "doc_a", "rank": 2, "text": "a2"},
        {"chunk_id": "b_1", "doc_id": "doc_b", "rank": 3, "text": "b1"},
        {"chunk_id": "c_1", "doc_id": "doc_c", "rank": 4, "text": "c1"},
    ]

    picked = rerank_results_with_doc_penalty(results, top_k=4, penalty=2.0)

    assert [item["chunk_id"] for item in picked] == ["a_1", "b_1", "a_2", "c_1"]
    assert [item["rank"] for item in picked] == [1, 2, 3, 4]


def test_run_experiment_reuses_retriever_for_same_docs(monkeypatch) -> None:
    docs = [
        {
            "doc_id": "doc_1",
            "text": "讲得救的确信",
            "sentences": ["讲得救的确信"],
            "source": "sermon",
            "title": "doc_1",
            "is_supporting": False,
            "supporting_sentence_ids": [],
        }
    ]
    rows = [
        {
            "id": "q1",
            "question": "第一个问题",
            "answer": "得救的确信",
            "supporting_facts": [{"title": "doc_1", "sentence_id": 0}],
            "documents": docs,
        },
        {
            "id": "q2",
            "question": "第二个问题",
            "answer": "得救的确信",
            "supporting_facts": [{"title": "doc_1", "sentence_id": 0}],
            "documents": docs,
        },
    ]
    config = {
        "name": "bm25_sentence_top3_sermon",
        "retrieval_mode": "bm25",
        "top_k": 3,
        "chunking": {"strategy": "sentence", "max_sentences": 2},
    }
    calls = {"build_retriever": 0}

    def fake_build_retriever(docs, config, encoder_cache):
        calls["build_retriever"] += 1
        return object()

    def fake_search_docs(retriever, query, config):
        return [
            {
                "chunk_id": "doc_1_chunk_0",
                "doc_id": "doc_1",
                "rank": 1,
                "text": "讲得救的确信",
                "is_supporting": True,
                "score": 1.0,
            }
        ]

    monkeypatch.setattr(experiment_runner, "build_retriever", fake_build_retriever)
    monkeypatch.setattr(experiment_runner, "search_docs", fake_search_docs)

    summary, query_rows = run_experiment(rows, config, encoder_cache={})

    assert calls["build_retriever"] == 1
    assert summary["n_queries"] == 2
    assert len(query_rows) == 2


def test_run_experiment_adds_optional_ragas_scores(monkeypatch) -> None:
    rows = [
        {
            "id": "q1",
            "question": "第一个问题",
            "answer": "得救的确信",
            "gold_doc_ids": ["doc_1"],
            "documents": [
                {
                    "doc_id": "doc_1",
                    "text": "讲得救的确信",
                    "sentences": ["讲得救的确信"],
                    "source": "sermon",
                    "title": "doc_1",
                    "is_supporting": False,
                    "supporting_sentence_ids": [],
                }
            ],
        }
    ]
    config = {
        "name": "dense_sentence_top3_chromadb",
        "retrieval_mode": "dense",
        "top_k": 3,
        "model_name": "all-MiniLM-L6-v2",
        "dense_backend": "chromadb",
        "answer_quality": {"use_ragas": True},
        "chunking": {"strategy": "sentence", "max_sentences": 2},
    }

    def fake_build_retriever(docs, config, encoder_cache):
        return object()

    def fake_search_docs(retriever, query, config):
        return [
            {
                "chunk_id": "doc_1_chunk_0",
                "doc_id": "doc_1",
                "rank": 1,
                "text": "讲得救的确信",
                "is_supporting": True,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(experiment_runner, "build_retriever", fake_build_retriever)
    monkeypatch.setattr(experiment_runner, "search_docs", fake_search_docs)
    monkeypatch.setattr(
        experiment_runner,
        "score_ragas_context_recall",
        lambda retrieved_ids, gold_doc_ids: {"ragas_context_recall": 1.0},
    )

    summary, query_rows = run_experiment(rows, config, encoder_cache={"all-MiniLM-L6-v2": object()})

    assert summary["dense_backend"] == "chromadb"
    assert summary["ragas_context_recall"] == 1.0
    assert query_rows[0]["ragas_metrics"]["ragas_context_recall"] == 1.0


def test_run_eval_can_skip_unavailable_configs(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    metrics_path = tmp_path / "metrics.csv"
    per_query_path = tmp_path / "per_query.json"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  name: hotpotqa",
                "  limit: 1",
                "output:",
                f"  metrics_path: {metrics_path}",
                f"  per_query_path: {per_query_path}",
                "experiments:",
                "  - name: bm25_ok",
                "    retrieval_mode: bm25",
                "    top_k: 3",
                "    chunking:",
                "      strategy: sentence",
                "      max_sentences: 2",
                "  - name: dense_missing",
                "    retrieval_mode: dense",
                "    model_name: all-MiniLM-L6-v2",
                "    top_k: 3",
                "    chunking:",
                "      strategy: sentence",
                "      max_sentences: 2",
            ]
        ),
        encoding="utf-8",
    )

    rows = [
        {
            "id": "q1",
            "question": "question",
            "answer": "answer",
            "supporting_facts": [{"title": "doc_1", "sentence_id": 0}],
            "documents": [
                {
                    "doc_id": "doc_1",
                    "text": "answer text",
                    "sentences": ["answer text"],
                    "source": "hotpotqa",
                    "title": "doc_1",
                    "is_supporting": True,
                    "supporting_sentence_ids": [0],
                }
            ],
        }
    ]

    monkeypatch.setattr(experiment_runner, "get_rows", lambda dataset_config, limit_override=None: rows)

    def fake_run_experiment(rows, config, encoder_cache):
        if config["name"] == "dense_missing":
            raise RuntimeError("failed to load dense embedding model")
        return (
            {
                "config_name": config["name"],
                "retrieval_mode": config["retrieval_mode"],
                "chunk_strategy": "sentence",
                "top_k": 3,
                "model_name": "",
                "dense_backend": "",
                "n_queries": 1,
                "recall@1": 1.0,
                "recall@3": 1.0,
                "recall@5": 1.0,
                "mrr": 1.0,
                "hit_rate": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "keyword_hit": 1.0,
            },
            [{"config_name": config["name"], "query_id": "q1"}],
        )

    monkeypatch.setattr(experiment_runner, "run_experiment", fake_run_experiment)

    result = run_eval(str(config_path), skip_unavailable=True)

    assert len(result["summary_rows"]) == 2
    assert result["summary_rows"][0]["config_name"] == "bm25_ok"
    assert result["summary_rows"][1]["status"] == "skipped"
    assert "failed to load dense embedding model" in result["summary_rows"][1]["error"]
    assert len(result["per_query_rows"]) == 1
    assert "dense_missing" not in per_query_path.read_text(encoding="utf-8")
