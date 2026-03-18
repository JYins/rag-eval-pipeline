from src.data_loader import (
    build_subset,
    build_support_map,
    parse_context,
    parse_example,
    sample_rows,
)


def make_raw_row() -> dict:
    return {
        "_id": "row_1",
        "question": " Who played Corliss Archer? ",
        "answer": " Shirley Temple ",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": [
            ["Kiss and Tell (1945 film)", 0],
            ["Shirley Temple", 1],
        ],
        "context": [
            ["Kiss and Tell (1945 film)", ["Shirley Temple played Corliss Archer.", "Another line."]],
            ["Shirley Temple", ["Shirley Temple was an actress.", "She was also a diplomat."]],
        ],
    }


def test_build_support_map_groups_sentence_ids() -> None:
    support_map = build_support_map(
        [
            ["Doc A", 0],
            ["Doc B", 1],
            ["Doc A", 3],
        ]
    )

    assert support_map == {"Doc A": [0, 3], "Doc B": [1]}


def test_parse_context_marks_supporting_docs() -> None:
    row = make_raw_row()
    docs = parse_context(row["context"], row["supporting_facts"])

    assert len(docs) == 2
    assert docs[0]["doc_id"] == "Kiss and Tell (1945 film)"
    assert docs[0]["is_supporting"] is True
    assert docs[0]["supporting_sentence_ids"] == [0]
    assert docs[1]["doc_id"] == "Shirley Temple"
    assert docs[1]["supporting_sentence_ids"] == [1]


def test_parse_example_returns_clean_fields() -> None:
    row = parse_example(make_raw_row())

    assert row["id"] == "row_1"
    assert row["question"] == "Who played Corliss Archer?"
    assert row["answer"] == "Shirley Temple"
    assert len(row["documents"]) == 2
    assert row["supporting_facts"][0]["title"] == "Kiss and Tell (1945 film)"


def test_sample_rows_is_deterministic() -> None:
    rows = [{"id": f"row_{index}"} for index in range(10)]

    first = sample_rows(rows, limit=4, seed=7)
    second = sample_rows(rows, limit=4, seed=7)

    assert first == second
    assert len(first) == 4


def test_build_subset_parses_sampled_rows() -> None:
    rows = [make_raw_row() for _ in range(3)]
    subset = build_subset(rows, limit=2, seed=7)

    assert len(subset) == 2
    assert subset[0]["question"] == "Who played Corliss Archer?"
