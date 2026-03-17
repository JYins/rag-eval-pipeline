"""Small helpers for dense vector indexing."""

from __future__ import annotations

import importlib
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
