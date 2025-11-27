from __future__ import annotations
from collections import Counter
from typing import Dict, Tuple

def compose_all(counters: Dict[str, Counter]) -> Tuple[Counter]:
    total = sum(counters.values(), Counter())
    return total
