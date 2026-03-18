import src.experiment_runner as experiment_runner
from src.experiment_runner import load_experiments, run_experiment


def test_load_experiments_reads_sermon_bm25_config() -> None:
    payload = load_experiments("configs/sermon.yaml")

    assert payload["dataset"]["name"] == "sermon"
    assert len(payload["experiments"]) == 3
    assert payload["experiments"][0]["name"] == "bm25_sentence_top3_sermon"
    assert payload["experiments"][0]["retrieval_mode"] == "bm25"
    assert payload["experiments"][1]["retrieval_mode"] == "dense"
    assert payload["experiments"][2]["retrieval_mode"] == "hybrid"


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
