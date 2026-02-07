from __future__ import annotations
from typing import Dict, Tuple
from collections import Counter
from nlpo_toolkit.nlp import count_nouns_streaming, load_vocab as _load_vocab

def load_vocab(vocab_path) -> dict:
    return _load_vocab(vocab_path)

def count_group(text: str, nlp,  label: str = "") -> Counter:
    """Return counter for NOUN lemmas"""
    total = count_nouns_streaming(
        text,
        nlp,
        use_lemma=True,
        upos_targets={"NOUN"},
        chunk_chars=200_000,
        label=label,
    )
    return total

def merge_counters(a: Counter, b: Counter) -> Counter:
    return a + b

def sum_all(groups: Dict[str, Counter]) -> Counter:
    total = Counter()
    for c in groups.values():
        total.update(c)
    return total
