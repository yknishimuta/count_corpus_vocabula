from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Mapping


def write_frequency_csv(path: Path, freq: Mapping[str, int] | Counter[str]) -> None:
    """
    Write frequency CSV with columns: lemma,count

    - Sorted by count desc, then lemma asc
    - UTF-8
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # normalize to dict[str, int]
    items = []
    for k, v in (freq.items() if hasattr(freq, "items") else []):  # type: ignore[truthy-bool]
        try:
            key = str(k)
            val = int(v)
        except Exception:
            continue
        items.append((key, val))

    # sort
    items.sort(key=lambda kv: (-kv[1], kv[0]))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lemma", "count"])
        w.writerows(items)
