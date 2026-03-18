# ruff: noqa: E402

"""Prepare local sermon files for Phase B experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader_sermon import load_sermon_documents


DEFAULT_SOURCE_DIR = ROOT_DIR / "sermon处理文档"
DEFAULT_TARGET_DIR = ROOT_DIR / "data" / "raw" / "sermons"
DEFAULT_QUESTIONS_FILE = ROOT_DIR / "data" / "eval" / "sermon_questions.csv"
DEFAULT_DOC_INDEX = ROOT_DIR / "data" / "eval" / "sermon_doc_index.csv"


def resolve_cli_path(path: str, follow_links: bool = True) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT_DIR / value
    if follow_links:
        return value.resolve()
    return value


def pick_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"source dir not found: {source_dir}")

    files = []
    for path in source_dir.rglob("*.docx"):
        if path.name.startswith("."):
            continue
        if "视频转文本" not in str(path.parent):
            continue
        files.append(path)

    if not files:
        raise ValueError(f"no sermon transcript docx files found in {source_dir}")
    return sorted(files)


def make_target_name(source_path: Path, used_names: set[str]) -> str:
    name = source_path.name
    if name not in used_names:
        used_names.add(name)
        return name

    name = f"{source_path.parent.name}__{source_path.name}"
    if name not in used_names:
        used_names.add(name)
        return name

    index = 2
    picked = name
    while picked in used_names:
        picked = f"{source_path.stem}_{index}{source_path.suffix}"
        index += 1
    used_names.add(picked)
    return picked


def stage_sermon_files(source_dir: Path, target_dir: Path) -> None:
    source_files = pick_source_files(source_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.is_symlink():
        target_dir.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)

    used_names = {path.name for path in target_dir.iterdir()}
    for source_path in source_files:
        target_name = make_target_name(source_path, used_names)
        target_path = target_dir / target_name
        if target_path.exists():
            continue
        target_path.symlink_to(source_path)


def write_question_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question_id", "question", "answer", "gold_doc_ids", "notes"],
        )
        writer.writeheader()


def write_doc_index(path: Path, docs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doc_id", "title", "file_type", "relative_path"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "file_type": doc["file_type"],
                "relative_path": doc["relative_path"],
            }
            for doc in docs
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare sermon files and labeling template")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--target-dir", default=str(DEFAULT_TARGET_DIR))
    parser.add_argument("--questions-path", default=str(DEFAULT_QUESTIONS_FILE))
    parser.add_argument("--doc-index-path", default=str(DEFAULT_DOC_INDEX))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve_cli_path(args.source_dir)
    target_dir = resolve_cli_path(args.target_dir, follow_links=False)
    questions_path = resolve_cli_path(args.questions_path)
    doc_index_path = resolve_cli_path(args.doc_index_path)

    stage_sermon_files(source_dir, target_dir)
    docs = load_sermon_documents(target_dir)
    write_question_template(questions_path)
    write_doc_index(doc_index_path, docs)

    print(f"linked sermon source: {target_dir}")
    print(f"sermon docs found: {len(docs)}")
    print(f"question template ready: {questions_path}")
    print(f"doc index ready: {doc_index_path}")


if __name__ == "__main__":
    main()
