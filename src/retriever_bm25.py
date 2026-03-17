"""BM25 retrieval baseline."""

from __future__ import annotations

import argparse
from typing import Any

from rank_bm25 import BM25Okapi

from src.chunking import chunk_docs
from src.cleaning import clean_text
from src.data_loader import load_hotpot_subset


def tokenize(text: str) -> list[str]:
    text = clean_text(text).lower()
    return text.split()


class BM25Retriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        if not chunks:
            raise ValueError("chunks should not be empty")

        self.chunks = chunks
        self.corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.index = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k should be > 0")

        tokens = tokenize(query)
        scores = self.index.get_scores(tokens)
        pairs = list(enumerate(scores))
        pairs.sort(key=lambda item: item[1], reverse=True)

        results = []
        for rank, (index, score) in enumerate(pairs[:top_k], start=1):
            chunk = dict(self.chunks[index])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)
        return results


def build_bm25_retriever(
    docs: list[dict[str, Any]],
    strategy: str = "sentence",
    **chunk_kwargs: Any,
) -> BM25Retriever:
    chunks = chunk_docs(docs, strategy=strategy, **chunk_kwargs)
    return BM25Retriever(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval on one HotpotQA sample")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--max-sentences", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_hotpot_subset()
    row = rows[args.sample_index]

    chunk_kwargs: dict[str, Any] = {}
    if args.strategy == "sentence":
        chunk_kwargs["max_sentences"] = args.max_sentences

    retriever = build_bm25_retriever(
        row["documents"],
        strategy=args.strategy,
        **chunk_kwargs,
    )
    results = retriever.search(row["question"], top_k=args.top_k)

    print(f"question: {row['question']}")
    for item in results:
        print(
            f"rank={item['rank']} score={item['score']:.4f} "
            f"doc_id={item['doc_id']} support={item['is_supporting']}"
        )
        print(item["text"][:200])
        print("---")


if __name__ == "__main__":
    main()
