from src.chunking import chunk_by_paragraph, chunk_by_sentence, chunk_docs, chunk_fixed_size


def make_doc(text: str, sentences: list[str] | None = None) -> dict:
    return {
        "doc_id": "doc_1",
        "title": "Test Doc",
        "source": "unit-test",
        "text": text,
        "sentences": sentences or [],
        "is_supporting": True,
        "supporting_sentence_ids": [0, 2],
    }


def test_fixed_chunk_keeps_metadata_and_overlap() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve"
    doc = make_doc(text)

    chunks = chunk_fixed_size(doc, chunk_size=5, overlap=2)

    assert len(chunks) == 4
    assert chunks[0]["chunk_id"] == "doc_1_chunk_0"
    assert chunks[0]["doc_id"] == "doc_1"
    assert chunks[0]["source"] == "unit-test"
    assert chunks[0]["strategy"] == "fixed"
    assert chunks[0]["is_supporting"] is True
    assert chunks[0]["text"] == "one two three four five"
    assert chunks[1]["text"] == "four five six seven eight"


def test_sentence_chunk_groups_sentences() -> None:
    sentences = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
        "Fourth sentence.",
        "Fifth sentence.",
    ]
    doc = make_doc(" ".join(sentences), sentences=sentences)

    chunks = chunk_by_sentence(doc, max_sentences=2)

    assert len(chunks) == 3
    assert chunks[0]["strategy"] == "sentence"
    assert chunks[0]["sentence_ids"] == [0, 1]
    assert chunks[1]["sentence_ids"] == [2, 3]
    assert chunks[2]["sentence_ids"] == [4]
    assert chunks[0]["text"] == "First sentence. Second sentence."


def test_sentence_chunk_can_prefix_title() -> None:
    sentences = ["First sentence.", "Second sentence."]
    doc = make_doc(" ".join(sentences), sentences=sentences)

    chunks = chunk_by_sentence(doc, max_sentences=2, include_title=True)

    assert chunks[0]["text"] == "Test Doc First sentence. Second sentence."


def test_paragraph_chunk_splits_on_blank_lines() -> None:
    text = "Para one line.\nStill para one.\n\nPara two here.\n\nPara three."
    doc = make_doc(text)

    chunks = chunk_by_paragraph(doc)

    assert len(chunks) == 3
    assert chunks[0]["strategy"] == "paragraph"
    assert chunks[0]["paragraph_index"] == 0
    assert chunks[1]["paragraph_index"] == 1
    assert chunks[2]["text"] == "Para three."


def test_chunk_docs_flattens_multiple_docs() -> None:
    docs = [
        make_doc("one two three four five six"),
        {
            **make_doc("alpha beta gamma delta epsilon"),
            "doc_id": "doc_2",
        },
    ]

    chunks = chunk_docs(docs, strategy="fixed", chunk_size=3, overlap=0)

    assert len(chunks) == 4
    assert chunks[0]["doc_id"] == "doc_1"
    assert chunks[-1]["doc_id"] == "doc_2"
