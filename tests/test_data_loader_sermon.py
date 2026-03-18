from __future__ import annotations

from pathlib import Path

from docx import Document

from scripts.prepare_sermon_data import stage_sermon_files
from src.data_loader_sermon import build_sermon_rows, load_sermon_documents, parse_gold_doc_ids


def write_docx(path: Path, paragraphs: list[str]) -> None:
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(path)


def test_load_sermon_documents_reads_docx_content(tmp_path: Path) -> None:
    sermon_dir = tmp_path / "sermons"
    sermon_dir.mkdir()
    write_docx(
        sermon_dir / "得救的确信.docx",
        [
            "第一段。这里讲得救的根据。",
            "第二段。这里讲救恩的确信。",
        ],
    )

    docs = load_sermon_documents(sermon_dir)

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "得救的确信"
    assert docs[0]["file_type"] == "docx"
    assert "得救的根据" in docs[0]["text"]
    assert len(docs[0]["sentences"]) >= 2


def test_build_sermon_rows_uses_gold_doc_ids(tmp_path: Path) -> None:
    sermon_dir = tmp_path / "sermons"
    sermon_dir.mkdir()
    write_docx(
        sermon_dir / "得救的确信.docx",
        [
            "得救不是靠行为，乃是靠神的恩典。",
        ],
    )

    questions_path = tmp_path / "sermon_questions.csv"
    questions_path.write_text(
        (
            "question_id,question,answer,gold_doc_ids,notes\n"
            "q1,得救的根据是什么,神的恩典,得救的确信,\n"
        ),
        encoding="utf-8",
    )

    rows = build_sermon_rows(questions_path=questions_path, sermon_root=sermon_dir)

    assert len(rows) == 1
    assert rows[0]["id"] == "q1"
    assert rows[0]["gold_doc_ids"] == ["得救的确信"]
    assert rows[0]["documents"][0]["doc_id"] == "得救的确信"


def test_parse_gold_doc_ids_splits_multiple_values() -> None:
    assert parse_gold_doc_ids("a;b|c") == ["a", "b", "c"]


def test_stage_sermon_files_replaces_stale_links(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    first_group = source_dir / "初信者话语 视频转文本"
    second_group = source_dir / "jsk牧师视频转文本 处理文件"
    first_group.mkdir(parents=True)
    second_group.mkdir(parents=True)

    write_docx(first_group / "第一讲.docx", ["第一篇。"])
    write_docx(second_group / "第二讲.docx", ["第二篇。"])

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    stale = target_dir / "旧文件.docx"
    stale.symlink_to(first_group / "第一讲.docx")

    stage_sermon_files(source_dir, target_dir)

    paths = sorted(path.name for path in target_dir.iterdir())
    assert "旧文件.docx" not in paths
    assert len(paths) == 2
