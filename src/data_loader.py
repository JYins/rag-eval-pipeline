"""Helpers for loading and parsing HotpotQA data."""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.cleaning import clean_lines, clean_text, join_lines


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
EVAL_DIR = ROOT_DIR / "data" / "eval"
RAW_FILE = RAW_DIR / "hotpot_dev_distractor_v1.json"
SUBSET_FILE = EVAL_DIR / "hotpotqa_subset.json"


def load_json(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_hotpot_raw(path: str | Path = RAW_FILE) -> list[dict]:
    return load_json(path)


def build_support_map(supporting_facts: list[list]) -> dict[str, list[int]]:
    support_map: dict[str, list[int]] = {}
    for item in supporting_facts:
        title = clean_text(item[0])
        sent_id = int(item[1])
        support_map.setdefault(title, []).append(sent_id)
    return support_map


def parse_context(context: list[list], supporting_facts: list[list]) -> list[dict]:
    support_map = build_support_map(supporting_facts)
    docs = []

    for index, item in enumerate(context):
        title = clean_text(item[0])
        lines = clean_lines(item[1])
        docs.append(
            {
                "doc_id": title or f"doc_{index}",
                "title": title,
                "source": "hotpotqa",
                "text": join_lines(lines),
                "sentences": lines,
                "is_supporting": title in support_map,
                "supporting_sentence_ids": support_map.get(title, []),
            }
        )

    return docs


def parse_example(row: dict) -> dict:
    return {
        "id": row["_id"],
        "question": clean_text(row["question"]),
        "answer": clean_text(row["answer"]),
        "type": clean_text(row.get("type", "")),
        "level": clean_text(row.get("level", "")),
        "supporting_facts": [
            {
                "title": clean_text(item[0]),
                "sentence_id": int(item[1]),
            }
            for item in row["supporting_facts"]
        ],
        "documents": parse_context(row["context"], row["supporting_facts"]),
    }


def parse_rows(rows: list[dict]) -> list[dict]:
    return [parse_example(row) for row in rows]


def sample_rows(rows: list[dict], limit: int = 500, seed: int = 7) -> list[dict]:
    if limit >= len(rows):
        return rows

    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in picks]


def build_subset(
    rows: list[dict],
    limit: int = 500,
    seed: int = 7,
) -> list[dict]:
    picked_rows = sample_rows(rows, limit=limit, seed=seed)
    return parse_rows(picked_rows)


def load_hotpot_subset(path: str | Path = SUBSET_FILE) -> list[dict]:
    return load_json(path)
