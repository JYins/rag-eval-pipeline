from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from src.retriever_bm25 import BM25Retriever, build_bm25_retriever
from src.retriever_dense import DenseRetriever, build_dense_retriever
from src.retriever_hybrid import HybridRetriever, build_hybrid_retriever


def make_doc(doc_id: str, text: str, sentences: list[str] | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "title": doc_id,
        "source": "unit-test",
        "text": text,
        "sentences": sentences or [text],
        "is_supporting": False,
        "supporting_sentence_ids": [],
    }


def test_bm25_search_returns_most_relevant_chunk() -> None:
    chunks = [
        {"chunk_id": "c1", "doc_id": "a", "source": "unit-test", "text": "apple banana fruit market"},
        {"chunk_id": "c2", "doc_id": "b", "source": "unit-test", "text": "basketball team wins playoff game"},
        {"chunk_id": "c3", "doc_id": "c", "source": "unit-test", "text": "banana smoothie with apple and milk"},
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("apple banana", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rank"] == 1
    assert results[0]["score"] >= results[1]["score"]


def test_build_bm25_retriever_from_docs_uses_chunking() -> None:
    docs = [
        make_doc(
            "doc_1",
            "Shirley Temple was also a diplomat.",
            sentences=[
                "Shirley Temple was an actress.",
                "Shirley Temple was also a diplomat.",
            ],
        ),
        make_doc(
            "doc_2",
            "Janet Waldo was a radio actress.",
            sentences=[
                "Janet Waldo was a radio actress.",
            ],
        ),
    ]

    retriever = build_bm25_retriever(docs, strategy="sentence", max_sentences=1)
    results = retriever.search("which actress was also a diplomat", top_k=2)

    assert len(retriever.chunks) == 3
    assert results[0]["doc_id"] == "doc_1"
    assert "diplomat" in results[0]["text"].lower()


class FakeEncoder:
    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        rows = []
        for text in texts:
            value = text.lower()
            rows.append(
                [
                    float("apple" in value or "banana" in value),
                    float("basketball" in value or "playoff" in value),
                    float("diplomat" in value),
                ]
            )
        return np.asarray(rows, dtype="float32")


class FakeIndexFlatIP:
    def __init__(self, dim: int):
        self.dim = dim
        self.rows: np.ndarray | None = None

    def add(self, rows: np.ndarray) -> None:
        self.rows = rows

    def search(self, query_rows: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.rows is not None
        scores = query_rows @ self.rows.T
        order = np.argsort(-scores, axis=1)[:, :top_k]
        picked_scores = np.take_along_axis(scores, order, axis=1)
        return picked_scores, order


def install_fake_faiss(monkeypatch) -> None:
    fake_module = SimpleNamespace(IndexFlatIP=FakeIndexFlatIP)
    monkeypatch.setitem(sys.modules, "faiss", fake_module)


class FakeChromaCollection:
    def __init__(self):
        self.rows = []

    def add(self, ids, documents, metadatas, embeddings) -> None:
        for item in zip(ids, documents, metadatas, embeddings):
            row_id, text, metadata, vector = item
            self.rows.append(
                {
                    "id": row_id,
                    "text": text,
                    "metadata": metadata,
                    "vector": np.asarray(vector, dtype="float32"),
                }
            )

    def query(self, query_embeddings, n_results, include):
        query = np.asarray(query_embeddings[0], dtype="float32")
        rows = []
        for row in self.rows:
            score = float(query @ row["vector"])
            rows.append((score, row))
        rows.sort(key=lambda item: item[0], reverse=True)
        picked = rows[:n_results]
        return {
            "documents": [[item[1]["text"] for item in picked]],
            "metadatas": [[item[1]["metadata"] for item in picked]],
            "distances": [[1.0 - item[0] for item in picked]],
        }


class FakeChromaClient:
    def get_or_create_collection(self, name: str):
        return FakeChromaCollection()


def install_fake_chromadb(monkeypatch) -> None:
    fake_module = SimpleNamespace(Client=lambda: FakeChromaClient())
    monkeypatch.setitem(sys.modules, "chromadb", fake_module)


def test_dense_search_returns_most_similar_chunk(monkeypatch) -> None:
    install_fake_faiss(monkeypatch)

    chunks = [
        {"chunk_id": "c1", "doc_id": "a", "source": "unit-test", "text": "apple banana fruit market"},
        {"chunk_id": "c2", "doc_id": "b", "source": "unit-test", "text": "basketball team wins playoff game"},
        {"chunk_id": "c3", "doc_id": "c", "source": "unit-test", "text": "apple diplomat speech"},
    ]

    retriever = DenseRetriever(chunks, encoder=FakeEncoder())
    results = retriever.search("apple banana", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rank"] == 1
    assert results[0]["score"] >= results[1]["score"]


def test_build_dense_retriever_from_docs_uses_chunking(monkeypatch) -> None:
    install_fake_faiss(monkeypatch)

    docs = [
        make_doc(
            "doc_1",
            "Shirley Temple was also a diplomat.",
            sentences=[
                "Shirley Temple was an actress.",
                "Shirley Temple was also a diplomat.",
            ],
        ),
        make_doc(
            "doc_2",
            "Janet Waldo was a radio actress.",
            sentences=[
                "Janet Waldo was a radio actress.",
            ],
        ),
    ]

    retriever = build_dense_retriever(
        docs,
        strategy="sentence",
        max_sentences=1,
        model_name="all-MiniLM-L6-v2",
        encoder=FakeEncoder(),
    )
    results = retriever.search("which actress was also a diplomat", top_k=2)

    assert retriever.model_name == "all-MiniLM-L6-v2"
    assert len(retriever.chunks) == 3
    assert results[0]["doc_id"] == "doc_1"
    assert "diplomat" in results[0]["text"].lower()


def test_dense_search_supports_chromadb_backend(monkeypatch) -> None:
    install_fake_chromadb(monkeypatch)

    chunks = [
        {"chunk_id": "c1", "doc_id": "a", "source": "unit-test", "text": "apple banana fruit market"},
        {"chunk_id": "c2", "doc_id": "b", "source": "unit-test", "text": "basketball team wins playoff game"},
        {"chunk_id": "c3", "doc_id": "c", "source": "unit-test", "text": "apple diplomat speech"},
    ]

    retriever = DenseRetriever(chunks, encoder=FakeEncoder(), backend="chromadb")
    results = retriever.search("apple banana", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["rank"] == 1
    assert results[0]["score"] >= results[1]["score"]


class StubRetriever:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.rows[:top_k]


def test_hybrid_search_fuses_bm25_and_dense_ranks() -> None:
    bm25_rows = [
        {"chunk_id": "c1", "doc_id": "doc_1", "text": "apple banana", "rank": 1, "score": 8.0},
        {"chunk_id": "c2", "doc_id": "doc_2", "text": "apple diplomat", "rank": 2, "score": 5.0},
    ]
    dense_rows = [
        {"chunk_id": "c2", "doc_id": "doc_2", "text": "apple diplomat", "rank": 1, "score": 0.9},
        {"chunk_id": "c3", "doc_id": "doc_3", "text": "basketball playoff", "rank": 2, "score": 0.6},
    ]

    retriever = HybridRetriever(
        bm25_retriever=StubRetriever(bm25_rows),
        dense_retriever=StubRetriever(dense_rows),
        bm25_weight=0.3,
        dense_weight=0.7,
    )
    results = retriever.search("apple diplomat", top_k=3)

    assert len(results) == 3
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["bm25_rank"] == 2
    assert results[0]["dense_rank"] == 1
    assert results[0]["hybrid_score"] > results[1]["hybrid_score"]


def test_build_hybrid_retriever_from_docs_uses_shared_chunks(monkeypatch) -> None:
    install_fake_faiss(monkeypatch)

    docs = [
        make_doc(
            "doc_1",
            "Shirley Temple was also a diplomat.",
            sentences=[
                "Shirley Temple was an actress.",
                "Shirley Temple was also a diplomat.",
            ],
        ),
        make_doc(
            "doc_2",
            "Janet Waldo was a radio actress.",
            sentences=[
                "Janet Waldo was a radio actress.",
            ],
        ),
    ]

    retriever = build_hybrid_retriever(
        docs,
        strategy="sentence",
        max_sentences=1,
        model_name="all-MiniLM-L6-v2",
        encoder=FakeEncoder(),
        bm25_weight=0.4,
        dense_weight=0.6,
    )
    results = retriever.search("which actress was also a diplomat", top_k=2)

    assert len(retriever.bm25_retriever.chunks) == 3
    assert len(retriever.dense_retriever.chunks) == 3
    assert results[0]["doc_id"] == "doc_1"
    assert results[0]["dense_rank"] is not None
