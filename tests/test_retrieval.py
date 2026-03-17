from src.retriever_bm25 import BM25Retriever, build_bm25_retriever


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

