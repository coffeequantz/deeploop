"""Small shared helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def truncate_middle(text: str, max_chars: int, marker: str = "\n… [{removed} chars omitted] …\n") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    removed = len(text) - head - tail
    return text[:head] + marker.format(removed=removed) + text[-tail:]


def truncate_tail(text: str, max_chars: int, marker: str = "\n… [{removed} earlier chars omitted]\n") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    removed = len(text) - max_chars
    return marker.format(removed=removed) + text[-max_chars:]


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]


SEPARATOR_CHARS = set("=-_*#~+. ")


def first_error_line(output: str, limit: int = 160) -> str:
    fallback = ""
    for line in (output or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        alnum_ratio = sum(char.isalnum() for char in stripped) / len(stripped)
        if alnum_ratio < 0.4:
            continue
        lowered = stripped.lower()
        markers = ("assert", "error", "failed", "failure", "traceback", "exception")
        if any(token in lowered for token in markers):
            if "assert" in lowered:
                return stripped[:limit]
            fallback = fallback or stripped[:limit]
    return fallback


def one_line(text: str, limit: int = 160) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:limit]


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}
