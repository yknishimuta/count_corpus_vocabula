from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from count_corpus_vocabula.ref_tags import (
    load_ref_tag_set,
    build_ref_tag_detector,
    _normalize_for_match,
)
from nlpo_toolkit.nlp import count_nouns, count_nouns_streaming


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_tag_file(tmp_path: Path) -> Path:
    p = tmp_path / "ref_tags.txt"
    p.write_text(
        "# comment line\n"
        "metaphys\n"
        "physic\n"
        "\n"
        "cap\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def ref_tags() -> set[str]:
    return {"metaphys", "physic", "cap"}


# ---------------------------------------------------------------------------
# load_ref_tag_set
# ---------------------------------------------------------------------------

def test_load_ref_tag_set(ref_tag_file: Path):
    tags = load_ref_tag_set(ref_tag_file)
    assert tags == {"metaphys", "physic", "cap"}


def test_load_ref_tag_set_skips_comments_and_blanks(ref_tag_file: Path):
    tags = load_ref_tag_set(ref_tag_file)
    assert "#" not in "".join(tags)
    assert "" not in tags


# ---------------------------------------------------------------------------
# _normalize_for_match
# ---------------------------------------------------------------------------

def test_normalize_strips_trailing_dot():
    assert _normalize_for_match("metaphys.") == "metaphys"


def test_normalize_strips_edge_punct():
    assert _normalize_for_match(".cap.") == "cap"


def test_normalize_lowercases():
    assert _normalize_for_match("Metaphys.") == "metaphys"


def test_normalize_plain_word():
    assert _normalize_for_match("rosa") == "rosa"


# ---------------------------------------------------------------------------
# build_ref_tag_detector
# ---------------------------------------------------------------------------

def test_detector_matches_with_dot(ref_tags: set[str]):
    detect = build_ref_tag_detector(ref_tags)
    assert detect("metaphys.") == "metaphys"


def test_detector_matches_without_dot(ref_tags: set[str]):
    detect = build_ref_tag_detector(ref_tags)
    assert detect("metaphys") == "metaphys"


def test_detector_no_match(ref_tags: set[str]):
    detect = build_ref_tag_detector(ref_tags)
    assert detect("rosa") == ""


def test_detector_case_insensitive(ref_tags: set[str]):
    detect = build_ref_tag_detector(ref_tags)
    assert detect("Metaphys.") == "metaphys"


# ---------------------------------------------------------------------------
# Integration: count_nouns with ref_tag_detector
# ---------------------------------------------------------------------------

from dataclasses import dataclass


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
    def __init__(self, doc: DummyDoc):
        self._doc = doc

    def __call__(self, text: str) -> DummyDoc:
        return self._doc


def test_count_nouns_excludes_ref_tags(ref_tags: set[str]):
    detect = build_ref_tag_detector(ref_tags)
    ref_counter = Counter()

    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="rosa", text="rosam"),
                        DummyWord(upos="NOUN", lemma="Metaphys.", text="Metaphys."),
                        DummyWord(upos="NOUN", lemma="puella", text="Puella"),
                    ]
                )
            ]
        )
    )

    result = count_nouns(
        "whatever", nlp,
        use_lemma=True, upos_targets={"NOUN"},
        ref_tag_detector=detect,
        ref_tag_counter=ref_counter,
    )

    assert result == Counter({"rosa": 1, "puella": 1})
    assert "metaphys." not in result
    assert ref_counter["metaphys"] == 1


def test_count_nouns_streaming_ref_tag_counter(ref_tags: set[str]):
    detect = build_ref_tag_detector(ref_tags)
    ref_counter = Counter()

    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="rosa", text="rosa"),
                        DummyWord(upos="NOUN", lemma="cap.", text="cap."),
                    ]
                )
            ]
        )
    )

    result = count_nouns_streaming(
        "rosa " * 10,
        nlp,
        use_lemma=True, upos_targets={"NOUN"},
        chunk_chars=20,
        ref_tag_detector=detect,
        ref_tag_counter=ref_counter,
    )

    assert "cap." not in result
    assert result["rosa"] >= 1
    assert ref_counter["cap"] >= 1


def test_count_nouns_streaming_trace_has_ref_tag_column(ref_tags: set[str], tmp_path: Path):
    detect = build_ref_tag_detector(ref_tags)
    ref_counter = Counter()

    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="rosa", text="rosam"),
                        DummyWord(upos="NOUN", lemma="Physic.", text="Physic."),
                    ]
                )
            ]
        )
    )

    trace_path = tmp_path / "trace.tsv"

    count_nouns_streaming(
        "rosa ",
        nlp,
        use_lemma=True, upos_targets={"NOUN"},
        chunk_chars=200_000,
        trace_tsv=trace_path,
        trace_max_rows=0,
        ref_tag_detector=detect,
        ref_tag_counter=ref_counter,
    )

    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    assert "ref_tag" in header

    # Check that the ref_tag column has the correct value for the Physic. row
    cols = header.split("\t")
    ref_tag_idx = cols.index("ref_tag")

    for line in lines[1:]:
        fields = line.split("\t")
        token_field = fields[cols.index("token")]
        ref_tag_field = fields[ref_tag_idx]
        if token_field == "Physic.":
            assert ref_tag_field == "physic"
        elif token_field == "rosam":
            assert ref_tag_field == ""


def test_count_nouns_no_ref_tag_backward_compat():
    """Without ref_tag params, behavior is identical to before."""
    nlp = DummyNLP(
        DummyDoc(
            sentences=[
                DummySentence(
                    words=[
                        DummyWord(upos="NOUN", lemma="rosa", text="rosam"),
                        DummyWord(upos="NOUN", lemma="Metaphys.", text="Metaphys."),
                    ]
                )
            ]
        )
    )

    result = count_nouns("whatever", nlp, use_lemma=True, upos_targets={"NOUN"})
    # Without ref_tag_detector, Metaphys. should be counted normally
    assert result == Counter({"rosa": 1, "metaphys.": 1})
