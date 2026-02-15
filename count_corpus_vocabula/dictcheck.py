from __future__ import annotations
import re
import string
import csv
from pathlib import Path
from typing import Iterable, Set, Tuple

from nlpo_toolkit.nlp import load_vocab, normalize_token

_STRIP_RE = re.compile(
    rf"^[{re.escape(string.punctuation)}“”‘’«»…—–\-­]+|"
    rf"[{re.escape(string.punctuation)}“”‘’«»…—–\-­]+$"
)

_ROMAN_RE = re.compile(r"^(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))$", re.I)

def _dictcheck_key(s: str, *, normalize: bool) -> str:
    t = s.strip()
    t = _STRIP_RE.sub("", t)
    t = normalize_token(t) if normalize else t
    if _ROMAN_RE.fullmatch(t):
        return ""
    if len(t) < 2 or (not t.isalpha()):
        return ""
    return t

def load_lemma_normalize_map(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split("\t")
        if len(parts) != 2:
            raise ValueError(f"lemma normalize TSV must have 2 columns: {path} line={line!r}")
        src, dst = parts[0].strip(), parts[1].strip()
        if src and dst:
            m[src] = dst
    return m

def split_frequency_csv(
    freq_csv: Path,
    wordlist_path: Path,
    out_known_csv: Path,
    out_unknown_csv: Path,
    *,
    lemma_col: str = "lemma",
    count_col: str = "count",
    normalize: bool = True,
    normalize_map_path: Path | None = None
) -> Tuple[int, int]:
    """
    Split noun_frequency CSV into known/unknown by checking membership in a wordlist.

    Returns: (known_rows, unknown_rows)
    """
    lemma_map: dict[str, str] = {}
    if normalize_map_path is not None:
        lemma_map = load_lemma_normalize_map(normalize_map_path)

    raw_vocab: Set[str] = load_vocab(wordlist_path)
    vocab: Set[str] = set()
    for w in raw_vocab:
        k = _dictcheck_key(w, normalize=normalize)
        if k:
            vocab.add(k)

    known_rows: list[dict] = []
    unknown_rows: list[dict] = []

    with freq_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])

        if lemma_col not in fieldnames or count_col not in fieldnames:
            raise ValueError(
                f"CSV must have columns '{lemma_col}' and '{count_col}'. got={fieldnames}"
            )

        for row in r:
            lemma = (row.get(lemma_col) or "").strip()
            if not lemma:
                continue

            lemma = lemma_map.get(lemma, lemma)
            key = _dictcheck_key(lemma, normalize=normalize)
            if not key:
                continue

            if key in vocab:
                known_rows.append(row)
            else:
                unknown_rows.append(row)

    # write outputs (keep same columns & order)
    if not fieldnames:
        fieldnames = [lemma_col, count_col]

    def write(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as g:
            w = csv.DictWriter(g, fieldnames=fieldnames)
            w.writeheader()
            for rr in rows:
                w.writerow(rr)

    write(out_known_csv, known_rows, fieldnames)
    write(out_unknown_csv, unknown_rows, fieldnames)

    return (len(known_rows), len(unknown_rows))
