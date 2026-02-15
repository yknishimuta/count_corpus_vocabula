from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

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
    sentences: list[DummySentence]


class DummyNLP:
    """
    nlp(text) -> DummyDoc。
    """
    def __init__(self, doc: DummyDoc):
        self._doc = doc
        self.calls: list[str] = []

    def __call__(self, text: str) -> DummyDoc:
        self.calls.append(text)
        return self._doc

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
            sentences=[DummySentence(words=[DummyWord(upos="NOUN", lemma="rosa", text="rosa")])]
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

