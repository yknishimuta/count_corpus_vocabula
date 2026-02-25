from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

def write_frequency_csv(
    path: Path,
    freq: Mapping[str, int] | Counter[str],
    *,
    header: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, int]] = []
    for k, v in freq.items():
        items.append((str(k), int(v)))

    items.sort(key=lambda kv: (-kv[1], kv[0]))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        w.writerows(items)
