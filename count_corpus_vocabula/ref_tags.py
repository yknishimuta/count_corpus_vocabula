from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from collections import Counter


@dataclass(frozen=True)
class RefTagPattern:
    name: str
    regex: str
    compiled: re.Pattern


def load_ref_tag_patterns(path: Path) -> List[RefTagPattern]:
    """
    Load ref tag patterns from a text file.

    Supported formats per line:
      - name<TAB>regex
      - name: regex
      - regex   (auto-named as pattern_N)
    """
    txt = path.read_text(encoding="utf-8")
    out: List[RefTagPattern] = []

    auto_i = 1
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        name = ""
        regex = ""

        if "\t" in line:
            a, b = line.split("\t", 1)
            name, regex = a.strip(), b.strip()
        elif ":" in line:
            a, b = line.split(":", 1)
            # avoid accidentally splitting "http://..."-like cases by requiring non-empty name
            if a.strip():
                name, regex = a.strip(), b.strip()
            else:
                regex = line
        else:
            regex = line

        if not regex:
            continue

        if not name:
            name = f"pattern_{auto_i}"
            auto_i += 1

        comp = re.compile(regex)
        out.append(RefTagPattern(name=name, regex=regex, compiled=comp))

    return out


def strip_and_count_ref_tags(text: str, patterns: Iterable[RefTagPattern]) -> tuple[str, Counter]:
    """
    Remove ref tags from text and count them.

    Returns: (cleaned_text, tag_counter)
    """
    c = Counter()
    cleaned = text

    for p in patterns:
        cleaned, n = p.compiled.subn(" ", cleaned)
        if n:
            c[p.name] += int(n)

    return cleaned, c
