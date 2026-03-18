"""Small helpers for config and file output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_path(path: str) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return ROOT_DIR / value


def load_yaml(path: str) -> dict[str, Any]:
    file_path = resolve_path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_json(data: Any, path: str) -> None:
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        raise ValueError("rows should not be empty")

    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
