from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path

import pytest

from nlpo_toolkit.nlp import count_nouns, count_nouns_streaming
from count_corpus_vocabula.counters import count_group


# Minimal Stanza-like objects
@dataclass
class DummyWord:
    upos: str
    lemma: str | None = None
    text: str | None = None


@dataclass
class DummySentence:
    words: list[DummyWord]


@dataclass
class DummyDoc:
    sentences: list


class DummyNLP:
    """
    nlp(text) -> DummyDoc
    """
    def __init__(self, doc: DummyDoc):
        self._doc = doc
        self.calls: list[str] = []

    def __call__(self, text: str) -> DummyDoc:
        self.calls.append(text)
        return self._doc


# tokens-only fallback objects
@dataclass
class DummyToken:
    words: list[DummyWord] | None = None


@dataclass
class DummySentenceTokensOnly:
    words: list[DummyWord] | None = None
    tokens: list[DummyToken] | None = None


# Tests
def test_count_nouns_lemma_lowercase_and_upos_filter():
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="Puella", text="Puella"),
                        DummyWord(upos="VERB", lemma="amo", text="amat"),
                        DummyWord(upos="NOUN", lemma="Rosa", text="rosam"),
                        DummyWord(upos="ADJ", lemma="pulcher", text="pulchra"),
                        DummyWord(upos="NOUN", lemma=None, text="LIBER"),  # lemma無し→textへfallback
                    ]
                )
            ]
        )
    )

    out = count_nouns("whatever", nlp, use_lemma=True, upos_targets={"NOUN"})
    assert out == Counter({"puella": 1, "rosa": 1, "liber": 1})


def test_count_nouns_surface_when_use_lemma_false():
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="puella", text="Puella"),
                        DummyWord(upos="NOUN", lemma="rosa", text="rosam"),
                    ]
                )
            ]
        )
    )
    out = count_nouns("whatever", nlp, use_lemma=False, upos_targets={"NOUN"})
    assert out == Counter({"puella": 1, "rosam": 1})


def test_count_nouns_streaming_calls_nlp_multiple_times():
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(words=[DummyWord(upos="NOUN", lemma="rosa", text="rosa")])
            ]
        )
    )

    text = "rosa " * 50
    out = count_nouns_streaming(
        text,
        nlp,
        use_lemma=True,
        upos_targets={"NOUN"},
        chunk_chars=20,
        label="",
    )

    assert out["rosa"] >= 1
    assert len(nlp.calls) >= 2


def test_count_group_counts_only_nouns_and_applies_exclude():
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="rosa", text="rosam"),
                        DummyWord(upos="NOUN", lemma="puella", text="Puella"),
                        DummyWord(upos="VERB", lemma="amo", text="amat"),
                    ]
                )
            ]
        )
    )

    out = count_group("anything", nlp, label="g1", exclude_lemmas={"puella"})
    assert out == Counter({"rosa": 1})


def test_count_nouns_falls_back_to_tokens_when_words_missing():
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentenceTokensOnly(
                    words=None,
                    tokens=[
                        DummyToken(words=[DummyWord(upos="NOUN", lemma="Rosa", text="rosam")]),
                        DummyToken(words=[DummyWord(upos="VERB", lemma="amo", text="amat")]),
                    ],
                )
            ]
        )
    )

    out = count_nouns("whatever", nlp, use_lemma=True, upos_targets={"NOUN"})
    assert out == Counter({"rosa": 1})


def test_count_nouns_streaming_trace_stops_writing_after_limit_but_keeps_counting(tmp_path):
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="rosa", text="rosam"),
                        DummyWord(upos="NOUN", lemma="puella", text="Puella"),
                        DummyWord(upos="VERB", lemma="amo", text="amat"),
                    ]
                )
            ]
        )
    )

    trace_path = tmp_path / "trace.tsv"
    text = "rosa " * 50

    out = count_nouns_streaming(
        text,
        nlp,
        use_lemma=True,
        upos_targets={"NOUN"},
        chunk_chars=20,
        label="",
        trace_tsv=trace_path,
        trace_max_rows=1,
    )

    assert len(nlp.calls) >= 2

    assert out["rosa"] == len(nlp.calls)
    assert out["puella"] == len(nlp.calls)

    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "label\tchunk\tsent_idx\ttoken_idx\tsentence\ttoken\tlemma\tupos\tref_tag\tglobal_row"
    assert len(lines) == 2


def test_count_group_writes_trace_when_trace_kwargs_given(tmp_path):
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="rosa", text="rosam"),
                        DummyWord(upos="VERB", lemma="amo", text="amat"),
                    ]
                )
            ]
        )
    )

    trace_path = tmp_path / "group_trace.tsv"

    out = count_group(
        "anything",
        nlp,
        label="g1",
        exclude_lemmas=None,
        trace_kwargs={"trace_tsv": trace_path, "trace_max_rows": 1},
    )

    assert out == Counter({"rosa": 1})

    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "label\tchunk\tsent_idx\ttoken_idx\tsentence\ttoken\tlemma\tupos\tref_tag\tglobal_row"
    assert len(lines) == 2


def test_count_nouns_streaming_no_trace_does_not_create_tsv(tmp_path):
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(words=[DummyWord(upos="NOUN", lemma="rosa", text="rosa")])
            ]
        )
    )
    trace_path = tmp_path / "trace.tsv"

    out = count_nouns_streaming(
        "rosa " * 10,
        nlp,
        trace_tsv=None,
        chunk_chars=20,
    )

    assert out["rosa"] >= 1
    assert not trace_path.exists()
