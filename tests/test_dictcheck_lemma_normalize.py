from __future__ import annotations
from pathlib import Path
import csv

from count_corpus_vocabula.dictcheck import split_frequency_csv


def test_dictcheck_applies_lemma_normalize_map(tmp_path: Path):
    # wordlist contains normalized lemma
    wordlist = tmp_path / "latin_words.txt"
    wordlist.write_text("materia\n", encoding="utf-8")

    # freq csv contains unnormalized lemma
    freq = tmp_path / "noun_frequency.csv"
    freq.write_text("word,frequency\nmaterium,10\n", encoding="utf-8")

    # normalize tsv maps materium -> materia
    norm = tmp_path / "lemma_normalize.tsv"
    norm.write_text("materium\tmateria\n", encoding="utf-8")

    known = tmp_path / "known.csv"
    unknown = tmp_path / "unknown.csv"

    k, u = split_frequency_csv(
        freq_csv=freq,
        wordlist_path=wordlist,
        out_known_csv=known,
        out_unknown_csv=unknown,
        lemma_col="word",
        count_col="frequency",
        normalize=True,
        normalize_map_path=norm,
    )

    assert k == 1
    assert u == 0

