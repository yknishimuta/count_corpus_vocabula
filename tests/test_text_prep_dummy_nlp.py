from __future__ import annotations

from dataclasses import dataclass
from count_corpus_vocabula.text_prep import one_sentence_per_line


@dataclass
class DummySentence:
    text: str


@dataclass
class DummyDoc:
    sentences: list[DummySentence]


class DummyNLP:
    def __init__(self, sentences: list[str]):
        self._doc = DummyDoc([DummySentence(s) for s in sentences])
        self.calls: list[str] = []

    def __call__(self, text: str):
        self.calls.append(text)
        return self._doc


def test_one_sentence_per_line_uses_stanza_sentence_text_and_adds_newlines():
    nlp = DummyNLP(["Puella rosam amat.", "Rosa pulchra est."])
    raw = "Puella rosam amat.\nRosa pulchra est.\n"  # 入力はどうでもよい（分割は nlp が返す）
    out = one_sentence_per_line(raw, nlp)

    assert out == "Puella rosam amat.\nRosa pulchra est.\n"
    assert len(nlp.calls) == 1
