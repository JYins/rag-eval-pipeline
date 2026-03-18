from src.experiment_runner import load_experiments


def test_load_experiments_reads_sermon_bm25_config() -> None:
    payload = load_experiments("configs/sermon.yaml")

    assert payload["dataset"]["name"] == "sermon"
    assert len(payload["experiments"]) == 1
    assert payload["experiments"][0]["name"] == "bm25_sentence_top3_sermon"
    assert payload["experiments"][0]["retrieval_mode"] == "bm25"
