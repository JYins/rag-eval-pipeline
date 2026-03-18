"""Small helpers for dense vector indexing."""

from __future__ import annotations

import importlib
from uuid import uuid4
from typing import Any

import numpy as np


def get_faiss() -> Any:
    """Import faiss only when dense retrieval is used."""
    try:
        return importlib.import_module("faiss")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss-cpu is required for dense retrieval. Install requirements first."
        ) from exc


def get_chromadb() -> Any:
    """Import chromadb only when the Chroma path is used."""
    try:
        return importlib.import_module("chromadb")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "chromadb is required for the Chroma dense backend. Install requirements first."
        ) from exc


def to_float32(vectors: np.ndarray | list[list[float]]) -> np.ndarray:
    rows = np.asarray(vectors, dtype="float32")
    if rows.ndim != 2:
        raise ValueError("vectors should be a 2D array")
    if rows.shape[0] == 0:
        raise ValueError("vectors should not be empty")
    return rows


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_faiss_index(
    vectors: np.ndarray | list[list[float]],
    normalize: bool = True,
) -> Any:
    """Build a simple inner-product FAISS index."""
    rows = to_float32(vectors)
    if normalize:
        rows = normalize_rows(rows)

    faiss = get_faiss()
    index = faiss.IndexFlatIP(rows.shape[1])
    index.add(rows)
    return index


def search_index(
    index: Any,
    query_vectors: np.ndarray | list[list[float]],
    top_k: int = 5,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k should be > 0")

    rows = to_float32(query_vectors)
    if normalize:
        rows = normalize_rows(rows)

    scores, ids = index.search(rows, top_k)
    return scores, ids


def iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size should be > 0")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_chroma_collection(
    vectors: np.ndarray | list[list[float]],
    chunks: list[dict[str, Any]],
    collection_name: str | None = None,
) -> Any:
    """Build an in-memory Chroma collection from precomputed embeddings."""
    rows = to_float32(vectors)
    chromadb = get_chromadb()
    client = chromadb.Client()
    name = collection_name or f"rag_eval_{uuid4().hex}"
    collection = client.get_or_create_collection(name=name)
    batch_size = client.get_max_batch_size() if hasattr(client, "get_max_batch_size") else len(chunks)

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source": chunk.get("source", ""),
            "is_supporting": bool(chunk.get("is_supporting", False)),
        }
        for chunk in chunks
    ]

    embedding_rows = rows.tolist()
    id_batches = iter_batches(ids, batch_size)
    document_batches = iter_batches(documents, batch_size)
    metadata_batches = iter_batches(metadatas, batch_size)
    embedding_batches = iter_batches(embedding_rows, batch_size)

    for id_batch, document_batch, metadata_batch, embedding_batch in zip(
        id_batches,
        document_batches,
        metadata_batches,
        embedding_batches,
    ):
        collection.add(
            ids=id_batch,
            documents=document_batch,
            metadatas=metadata_batch,
            embeddings=embedding_batch,
        )
    return collection


def search_chroma_collection(
    collection: Any,
    query_vectors: np.ndarray | list[list[float]],
    top_k: int = 5,
) -> dict[str, list[list[Any]]]:
    if top_k <= 0:
        raise ValueError("top_k should be > 0")

    rows = to_float32(query_vectors)
    return collection.query(
        query_embeddings=rows.tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
