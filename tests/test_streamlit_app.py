from app.streamlit_app import (
    DEFAULT_METRICS_PATH,
    DEFAULT_PER_QUERY_PATH,
    SERMON_METRICS_PATH,
    SERMON_PER_QUERY_PATH,
    get_dataset_paths,
)


def test_get_dataset_paths_returns_hotpot_defaults() -> None:
    metrics_path, per_query_path = get_dataset_paths("HotpotQA")

    assert metrics_path == DEFAULT_METRICS_PATH
    assert per_query_path == DEFAULT_PER_QUERY_PATH


def test_get_dataset_paths_returns_sermon_defaults() -> None:
    metrics_path, per_query_path = get_dataset_paths("Sermon")

    assert metrics_path == SERMON_METRICS_PATH
    assert per_query_path == SERMON_PER_QUERY_PATH
