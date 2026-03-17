from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from src.retriever_bm25 import BM25Retriever, build_bm25_retriever
from src.retriever_dense import DenseRetriever, build_dense_retriever


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
