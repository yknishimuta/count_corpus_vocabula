from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


_STRIP_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


def load_ref_tag_set(path: Path) -> set[str]:
    """Load ref-tag abbreviations from a text file (one per line, # comments)."""
    items: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        items.add(s.lower())
    return items


def _normalize_for_match(key: str) -> str:
    """Normalize a counter key for ref-tag matching.

    Steps: strip whitespace → lowercase → remove edge punctuation → remove trailing dot.
    """
    t = key.strip().lower()
    t = _STRIP_EDGE_PUNCT.sub("", t)
    if t.endswith("."):
        t = t[:-1]
        t = _STRIP_EDGE_PUNCT.sub("", t)
    return t


def build_ref_tag_detector(ref_tags: set[str]) -> Callable[[str], str]:
    """Return a detector function: key -> matched tag name or empty string."""

    def _detect(key: str) -> str:
        t = _normalize_for_match(key)
        if t in ref_tags:
            return t
        return ""

    return _detect
