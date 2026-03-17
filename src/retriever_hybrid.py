"""Hybrid retrieval with simple weighted rank fusion."""

from __future__ import annotations

from typing import Any

from src.chunking import chunk_docs
from src.retriever_bm25 import BM25Retriever
from src.retriever_dense import DenseRetriever


def rank_score(rank: int) -> float:
    if rank <= 0:
        raise ValueError("rank should be > 0")
    return 1.0 / rank


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k should be > 0")

        pick_k = candidate_k or max(top_k, 5)
        bm25_results = self.bm25_retriever.search(query, top_k=pick_k)
        dense_results = self.dense_retriever.search(query, top_k=pick_k)

        merged: dict[str, dict[str, Any]] = {}

        for item in bm25_results:
            key = item["chunk_id"]
            row = merged.setdefault(key, dict(item))
            row["bm25_rank"] = item["rank"]
            row["bm25_score"] = item["score"]
            row["dense_rank"] = None
            row["dense_score"] = None
            row["hybrid_score"] = self.bm25_weight * rank_score(item["rank"])

        for item in dense_results:
            key = item["chunk_id"]
            row = merged.setdefault(key, dict(item))
            row["dense_rank"] = item["rank"]
            row["dense_score"] = item["score"]
            if "bm25_rank" not in row:
                row["bm25_rank"] = None
                row["bm25_score"] = None
                row["hybrid_score"] = 0.0
            row["hybrid_score"] += self.dense_weight * rank_score(item["rank"])

        rows = list(merged.values())
        rows.sort(key=lambda item: item["hybrid_score"], reverse=True)

        results = []
        for rank, item in enumerate(rows[:top_k], start=1):
            row = dict(item)
            row["rank"] = rank
            results.append(row)
        return results


def build_hybrid_retriever(
    docs: list[dict[str, Any]],
    strategy: str = "sentence",
    model_name: str = "all-MiniLM-L6-v2",
    encoder: Any | None = None,
    bm25_weight: float = 0.5,
    dense_weight: float = 0.5,
    **chunk_kwargs: Any,
) -> HybridRetriever:
    chunks = chunk_docs(docs, strategy=strategy, **chunk_kwargs)
    bm25_retriever = BM25Retriever(chunks)
    dense_retriever = DenseRetriever(chunks, model_name=model_name, encoder=encoder)
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )
