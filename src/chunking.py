"""Simple chunking helpers for retrieval eval."""

from __future__ import annotations

from typing import Any

from src.cleaning import clean_lines, clean_text


def make_chunk(
    doc: dict[str, Any],
    text: str,
    chunk_index: int,
    strategy: str,
    sentence_ids: list[int] | None = None,
    paragraph_index: int | None = None,
) -> dict[str, Any]:
    chunk_text = clean_text(text)
    if not chunk_text:
        raise ValueError("chunk text should not be empty")

    return {
        "chunk_id": f"{doc['doc_id']}_chunk_{chunk_index}",
        "chunk_index": chunk_index,
        "doc_id": doc["doc_id"],
        "title": doc.get("title", ""),
        "source": doc.get("source", ""),
        "text": chunk_text,
        "strategy": strategy,
        "sentence_ids": sentence_ids or [],
        "paragraph_index": paragraph_index,
        "is_supporting": doc.get("is_supporting", False),
        "supporting_sentence_ids": doc.get("supporting_sentence_ids", []),
    }


def chunk_fixed_size(
    doc: dict[str, Any],
    chunk_size: int = 120,
    overlap: int = 20,
) -> list[dict[str, Any]]:
    """Split one doc into fixed-size word chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size should be > 0")
    if overlap < 0:
        raise ValueError("overlap should be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap should be smaller than chunk_size")

    words = clean_text(doc.get("text", "")).split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for chunk_index, start in enumerate(range(0, len(words), step)):
        piece = words[start : start + chunk_size]
        if not piece:
            continue
        chunks.append(
            make_chunk(
                doc=doc,
                text=" ".join(piece),
                chunk_index=chunk_index,
                strategy="fixed",
            )
        )
    return chunks


def chunk_by_sentence(
    doc: dict[str, Any],
    max_sentences: int = 3,
) -> list[dict[str, Any]]:
    """Group nearby sentences into one chunk."""
    if max_sentences <= 0:
        raise ValueError("max_sentences should be > 0")

    sentences = clean_lines(doc.get("sentences"))
    if not sentences:
        text = clean_text(doc.get("text", ""))
        if not text:
            return []
        sentences = [text]

    chunks = []
    for chunk_index, start in enumerate(range(0, len(sentences), max_sentences)):
        piece = sentences[start : start + max_sentences]
        sentence_ids = list(range(start, start + len(piece)))
        chunks.append(
            make_chunk(
                doc=doc,
                text=" ".join(piece),
                chunk_index=chunk_index,
                strategy="sentence",
                sentence_ids=sentence_ids,
            )
        )
    return chunks


def chunk_by_paragraph(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split doc text by blank lines."""
    text = str(doc.get("text", ""))
    if not text.strip():
        return []

    parts = [clean_text(part) for part in text.split("\n\n")]
    parts = [part for part in parts if part]

    if not parts:
        return []

    chunks = []
    for chunk_index, part in enumerate(parts):
        chunks.append(
            make_chunk(
                doc=doc,
                text=part,
                chunk_index=chunk_index,
                strategy="paragraph",
                paragraph_index=chunk_index,
            )
        )
    return chunks


def chunk_doc(
    doc: dict[str, Any],
    strategy: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Dispatch one strategy for one doc."""
    if strategy == "fixed":
        return chunk_fixed_size(doc, **kwargs)
    if strategy == "sentence":
        return chunk_by_sentence(doc, **kwargs)
    if strategy == "paragraph":
        return chunk_by_paragraph(doc)
    raise ValueError(f"unknown chunk strategy: {strategy}")


def chunk_docs(
    docs: list[dict[str, Any]],
    strategy: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Chunk a list of docs and flatten result."""
    chunks = []
    for doc in docs:
        chunks.extend(chunk_doc(doc, strategy=strategy, **kwargs))
    return chunks
