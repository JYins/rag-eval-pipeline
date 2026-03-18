"""Dense retrieval with sentence-transformers + FAISS."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.chunking import chunk_docs
from src.data_loader import load_hotpot_subset
from src.indexing import build_faiss_index, search_index


EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
}


def load_encoder(model_name: str) -> Any:
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"unknown embedding model: {model_name}")

    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentence-transformers is required for dense retrieval. Install requirements first."
        ) from exc

    try:
        return sentence_transformers.SentenceTransformer(EMBEDDING_MODELS[model_name])
    except Exception as exc:
        raise RuntimeError(
            "failed to load dense embedding model. "
            "Make sure the model is cached locally or the machine can reach Hugging Face."
        ) from exc


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
        encoder: Any | None = None,
    ):
        if not chunks:
            raise ValueError("chunks should not be empty")

        self.chunks = chunks
        self.model_name = model_name
        self.encoder = encoder or load_encoder(model_name)
        texts = [chunk["text"] for chunk in chunks]
        self.vectors = self.encode_texts(texts)
        self.index = build_faiss_index(self.vectors)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(texts, convert_to_numpy=True)
        return np.asarray(vectors, dtype="float32")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k should be > 0")

        query_vector = self.encode_texts([query])
        scores, ids = search_index(self.index, query_vector, top_k=top_k)

        results = []
        for rank, (index, score) in enumerate(zip(ids[0], scores[0]), start=1):
            chunk = dict(self.chunks[int(index)])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)
        return results


def build_dense_retriever(
    docs: list[dict[str, Any]],
    strategy: str = "sentence",
    model_name: str = "all-MiniLM-L6-v2",
    encoder: Any | None = None,
    **chunk_kwargs: Any,
) -> DenseRetriever:
    chunks = chunk_docs(docs, strategy=strategy, **chunk_kwargs)
    return DenseRetriever(chunks, model_name=model_name, encoder=encoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense retrieval on one HotpotQA sample")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--strategy", default="sentence")
    parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    parser.add_argument("--max-sentences", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_hotpot_subset()
    row = rows[args.sample_index]

    chunk_kwargs: dict[str, Any] = {}
    if args.strategy == "sentence":
        chunk_kwargs["max_sentences"] = args.max_sentences

    retriever = build_dense_retriever(
        row["documents"],
        strategy=args.strategy,
        model_name=args.model_name,
        **chunk_kwargs,
    )
    results = retriever.search(row["question"], top_k=args.top_k)

    print(f"question: {row['question']}")
    print(f"model: {retriever.model_name}")
    for item in results:
        print(
            f"rank={item['rank']} score={item['score']:.4f} "
            f"doc_id={item['doc_id']} support={item['is_supporting']}"
        )
        print(item["text"][:200])
        print("---")


if __name__ == "__main__":
    main()
