from __future__ import annotations
from typing import Dict, Tuple
from collections import Counter
from typing import Iterable
from pathlib import Path
from nlpo_toolkit.nlp import count_nouns_streaming, load_vocab as _load_vocab

def load_vocab(vocab_path) -> dict:
    return _load_vocab(vocab_path)

def load_exclude_list(path: str | Path) -> set[str]:
    p = Path(path)
    items: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        items.add(s.lower())
    return items

def filter_counter(counter: Counter, *, exclude: Iterable[str]) -> Counter:
    ex = {w.lower() for w in exclude}
    return Counter({k: v for k, v in counter.items() if k.lower() not in ex})

def count_group(text: str, nlp, label: str = "", exclude_lemmas: set[str] | None = None,
               trace_kwargs: dict | None = None) -> Counter:
    total = count_nouns_streaming(
        text,
        nlp,
        use_lemma=True,
        upos_targets={"NOUN"},
        chunk_chars=200_000,
        label=label,
        **(trace_kwargs or {}),
    )
    if exclude_lemmas:
        total = filter_counter(total, exclude=exclude_lemmas)
    return total

def merge_counters(a: Counter, b: Counter) -> Counter:
    return a + b

def sum_all(groups: Dict[str, Counter]) -> Counter:
    total = Counter()
    for c in groups.values():
        total.update(c)
    return total
