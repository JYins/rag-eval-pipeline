"""Basic text cleaning helpers."""

from __future__ import annotations

import re
import unicodedata


SPACE_RE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def clean_text(text: str | None) -> str:
    """Normalize text a bit for later chunking and retrieval."""
    if text is None:
        return ""

    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\u00a0", " ")
    value = value.replace("\u200b", " ")
    value = SPACE_RE.sub(" ", value)
    return value.strip()


def clean_lines(lines: list[str] | None) -> list[str]:
    """Clean sentence list and drop empty ones."""
    if not lines:
        return []

    cleaned = []
    for line in lines:
        text = clean_text(line)
        if text:
            cleaned.append(text)
    return cleaned


def join_lines(lines: list[str] | None) -> str:
    """Join cleaned lines into one retrieval text."""
    return " ".join(clean_lines(lines))


def has_cjk(text: str | None) -> bool:
    """Small helper for Chinese/Japanese/Korean text handling."""
    if text is None:
        return False
    return bool(CJK_RE.search(str(text)))
