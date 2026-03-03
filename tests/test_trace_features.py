from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from collections import Counter

from nlpo_toolkit.nlp import count_nouns_streaming  # type: ignore
from count_corpus_vocabula.counters import count_group  # type: ignore


# -------------------------
# Minimal fake stanza-like NLP
# -------------------------

@dataclass
class FakeWord:
    text: str
    lemma: str
    upos: str


@dataclass
class FakeToken:
    text: str
    start_char: int
    words: List[FakeWord]


@dataclass
class FakeSentence:
    text: str
    tokens: List[FakeToken]


@dataclass
class FakeDoc:
    sentences: List[FakeSentence]


class FakeNLP:
    """
    A tiny stanza-like callable:
    - returns a FakeDoc with a single sentence
    - each whitespace-separated token becomes one FakeToken
    - all tokens are tagged as NOUN, lemma = lowercased surface
    - token.start_char is computed within the given chunk
    """

    def __call__(self, text: str) -> FakeDoc:
        s = text
        tokens: List[FakeToken] = []

        i = 0
        for raw in s.split():
            start = s.find(raw, i)
            if start < 0:
                start = i
            i = start + len(raw)

            w = FakeWord(text=raw, lemma=raw.lower(), upos="NOUN")
            tok = FakeToken(text=raw, start_char=start, words=[w])
            tokens.append(tok)

        sent = FakeSentence(text=s, tokens=tokens)
        return FakeDoc(sentences=[sent])


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r)


# -------------------------
# Tests
# -------------------------

def test_trace_only_keys_writes_only_selected_in_trace_but_counts_all(tmp_path: Path):
    nlp = FakeNLP()
    text = "rosa aqua rosa terra"
    trace_path = tmp_path / "trace.tsv"

    out = count_nouns_streaming(
        text,
        nlp,
        use_lemma=True,
        upos_targets={"NOUN"},
        trace_tsv=trace_path,
        trace_only_keys={"rosa"},
        trace_max_rows=0,
        trace_write_truncation_marker=False,  # marker不要
    )

    # counting includes all nouns
    assert out == Counter({"rosa": 2, "aqua": 1, "terra": 1})

    rows = _read_tsv_rows(trace_path)

    # trace should include only rows where lemma==rosa
    assert len(rows) == 2
    assert all(r["lemma"].strip().lower() == "rosa" for r in rows)
    assert all(r["upos"] == "NOUN" for r in rows)


def test_trace_truncation_marker_written_once(tmp_path: Path):
    nlp = FakeNLP()
    text = "rosa aqua terra rosa"
    trace_path = tmp_path / "trace.tsv"

    out = count_nouns_streaming(
        text,
        nlp,
        use_lemma=True,
        upos_targets={"NOUN"},
        trace_tsv=trace_path,
        trace_max_rows=2,
        trace_write_truncation_marker=True,
    )

    # counting continues for the whole text
    assert out == Counter({"rosa": 2, "aqua": 1, "terra": 1})

    rows = _read_tsv_rows(trace_path)

    # 2 normal rows + 1 marker row
    assert len(rows) == 3

    marker_rows = [r for r in rows if r.get("upos") == "TRACE_TRUNCATED"]
    assert len(marker_rows) == 1
    assert marker_rows[0]["token"] == "(trace stopped; counting continues)"


def test_count_group_passes_trace_kwargs_to_trace_only_keys(tmp_path: Path):
    """
    Ensure count_group forwards trace_kwargs to count_nouns_streaming.
    """
    nlp = FakeNLP()
    text = "rosa aqua rosa terra"
    trace_path = tmp_path / "group_trace.tsv"

    out = count_group(
        text,
        nlp,
        label="g1",
        exclude_lemmas=None,
        upos_targets={"NOUN"},
        trace_kwargs={
            "trace_tsv": trace_path,
            "trace_only_keys": {"rosa"},
            "trace_max_rows": 0,
            "trace_write_truncation_marker": False,
        },
        use_lemma=True,
    )

    assert out == Counter({"rosa": 2, "aqua": 1, "terra": 1})

    rows = _read_tsv_rows(trace_path)
    assert len(rows) == 2
    assert all(r["lemma"].strip().lower() == "rosa" for r in rows)