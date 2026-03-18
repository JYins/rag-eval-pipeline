from app.streamlit_app import (
    DEFAULT_METRICS_PATH,
    DEFAULT_PER_QUERY_PATH,
    SERMON_DOC_DEDUP_METRICS_PATH,
    SERMON_DOC_DEDUP_PER_QUERY_PATH,
    SERMON_DOC_PENALTY_METRICS_PATH,
    SERMON_DOC_PENALTY_PER_QUERY_PATH,
    SERMON_METADATA_RERANK_METRICS_PATH,
    SERMON_METADATA_RERANK_PER_QUERY_PATH,
    SERMON_TITLE_AWARE_METRICS_PATH,
    SERMON_TITLE_AWARE_PER_QUERY_PATH,
    SERMON_CHROMADB_RAGAS_METRICS_PATH,
    SERMON_CHROMADB_RAGAS_PER_QUERY_PATH,
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


def test_get_dataset_paths_returns_sermon_doc_dedup_defaults() -> None:
    metrics_path, per_query_path = get_dataset_paths("Sermon (Doc Dedupe Study)")

    assert metrics_path == SERMON_DOC_DEDUP_METRICS_PATH
    assert per_query_path == SERMON_DOC_DEDUP_PER_QUERY_PATH


def test_get_dataset_paths_returns_sermon_doc_penalty_defaults() -> None:
    metrics_path, per_query_path = get_dataset_paths("Sermon (Doc Penalty Study)")

    assert metrics_path == SERMON_DOC_PENALTY_METRICS_PATH
    assert per_query_path == SERMON_DOC_PENALTY_PER_QUERY_PATH


def test_get_dataset_paths_returns_sermon_title_aware_defaults() -> None:
    metrics_path, per_query_path = get_dataset_paths("Sermon (Title-Aware Study)")

    assert metrics_path == SERMON_TITLE_AWARE_METRICS_PATH
    assert per_query_path == SERMON_TITLE_AWARE_PER_QUERY_PATH


def test_get_dataset_paths_returns_sermon_metadata_rerank_defaults() -> None:
    metrics_path, per_query_path = get_dataset_paths("Sermon (Metadata Rerank Study)")

    assert metrics_path == SERMON_METADATA_RERANK_METRICS_PATH
    assert per_query_path == SERMON_METADATA_RERANK_PER_QUERY_PATH


def test_get_dataset_paths_returns_sermon_chromadb_ragas_defaults() -> None:
    metrics_path, per_query_path = get_dataset_paths("Sermon (ChromaDB + RAGAS)")

    assert metrics_path == SERMON_CHROMADB_RAGAS_METRICS_PATH
    assert per_query_path == SERMON_CHROMADB_RAGAS_PER_QUERY_PATH
