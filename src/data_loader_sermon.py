"""Helpers for loading local sermon transcripts and eval labels."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from src.cleaning import clean_text, has_cjk
from src.utils import resolve_path


ROOT_DIR = Path(__file__).resolve().parents[1]
SERMON_DIR = ROOT_DIR / "data" / "raw" / "sermons"
QUESTIONS_FILE = ROOT_DIR / "data" / "eval" / "sermon_questions.csv"

ZH_SENTENCE_RE = re.compile(r"(?<=[。！？；!?])")
EN_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def resolve_input_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return resolve_path(str(value))


def list_sermon_files(root: str | Path = SERMON_DIR) -> list[Path]:
    root_path = resolve_input_path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"sermon dir not found: {root_path}")

    files = []
    for path in root_path.rglob("*"):
        if path.name.startswith(".") or not path.is_file():
            continue
        if path.suffix.lower() not in {".docx", ".pdf"}:
            continue
        files.append(path)

    if not files:
        raise ValueError(f"no sermon files found in {root_path}")
    return sorted(files)


def split_sermon_sentences(text: str) -> list[str]:
    value = clean_text(text)
    if not value:
        return []

    if has_cjk(value):
        parts = ZH_SENTENCE_RE.split(value)
    else:
        parts = EN_SENTENCE_RE.split(value)
    return [clean_text(part) for part in parts if clean_text(part)]


def load_docx_text(path: str | Path) -> str:
    doc = Document(str(path))
    parts = [clean_text(paragraph.text) for paragraph in doc.paragraphs]
    parts = [part for part in parts if part]
    return "\n\n".join(parts)


def load_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = clean_text(page.extract_text() or "")
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def load_sermon_text(path: str | Path) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        return load_docx_text(file_path)
    if suffix == ".pdf":
        return load_pdf_text(file_path)
    raise ValueError(f"unsupported sermon file: {file_path}")


def make_doc_id(path: Path, used_ids: set[str]) -> str:
    base = clean_text(path.stem) or clean_text(path.name)
    if base in used_ids:
        parent_name = clean_text(path.parent.name)
        if parent_name:
            base = f"{parent_name}__{base}"

    doc_id = base
    index = 2
    while doc_id in used_ids:
        doc_id = f"{base}_{index}"
        index += 1

    used_ids.add(doc_id)
    return doc_id


def load_sermon_documents(root: str | Path = SERMON_DIR) -> list[dict]:
    root_path = resolve_input_path(root)
    files = list_sermon_files(root_path)
    used_ids: set[str] = set()
    docs = []

    for path in files:
        text = load_sermon_text(path)
        if not text:
            continue

        paragraphs = [clean_text(part) for part in text.split("\n\n")]
        paragraphs = [part for part in paragraphs if part]
        if not paragraphs:
            continue

        doc_id = make_doc_id(path, used_ids)
        docs.append(
            {
                "doc_id": doc_id,
                "title": clean_text(path.stem),
                "source": "sermon",
                "text": "\n\n".join(paragraphs),
                "sentences": split_sermon_sentences("\n".join(paragraphs)),
                "is_supporting": False,
                "supporting_sentence_ids": [],
                "file_name": path.name,
                "relative_path": path.relative_to(root_path).as_posix(),
                "file_type": path.suffix.lower().lstrip("."),
            }
        )

    if not docs:
        raise ValueError(f"no readable sermon docs found in {root_path}")
    return docs


def parse_gold_doc_ids(value: str) -> list[str]:
    picks = re.split(r"[;|]", str(value))
    return [clean_text(item) for item in picks if clean_text(item)]


def load_sermon_questions(path: str | Path = QUESTIONS_FILE) -> list[dict[str, str]]:
    file_path = resolve_input_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"sermon question file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def build_sermon_rows(
    questions_path: str | Path = QUESTIONS_FILE,
    sermon_root: str | Path = SERMON_DIR,
) -> list[dict]:
    docs = load_sermon_documents(sermon_root)
    questions = load_sermon_questions(questions_path)
    if not questions:
        raise ValueError("sermon question file has no labeled rows yet")

    doc_ids = {doc["doc_id"] for doc in docs}
    rows = []
    for index, item in enumerate(questions, start=1):
        question = clean_text(item.get("question"))
        gold_doc_ids = parse_gold_doc_ids(item.get("gold_doc_ids", ""))
        if not question:
            raise ValueError(f"question should not be empty at row {index}")
        if not gold_doc_ids:
            raise ValueError(f"gold_doc_ids should not be empty at row {index}")

        missing = [doc_id for doc_id in gold_doc_ids if doc_id not in doc_ids]
        if missing:
            raise ValueError(f"unknown sermon doc ids at row {index}: {missing}")

        rows.append(
            {
                "id": clean_text(item.get("question_id")) or f"sermon_{index}",
                "question": question,
                "answer": clean_text(item.get("answer", "")),
                "supporting_facts": [
                    {"title": doc_id, "sentence_id": 0}
                    for doc_id in gold_doc_ids
                ],
                "gold_doc_ids": gold_doc_ids,
                "documents": docs,
            }
        )
    return rows
