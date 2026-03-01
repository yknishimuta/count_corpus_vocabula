from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from count_corpus_vocabula.ref_tags import (
    load_ref_tag_patterns,
    strip_and_count_ref_tags,
    RefTagPattern,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_tag_file(tmp_path: Path) -> Path:
    p = tmp_path / "ref_tags.txt"
    p.write_text(
        "# comment\n"
        "metaphys\tmetaphys\\.\n"
        "physic: physic\\.\n"
        "\n"
        "cap\\.\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# load_ref_tag_patterns
# ---------------------------------------------------------------------------

def test_load_ref_tag_patterns_parses_formats(ref_tag_file: Path):
    patterns = load_ref_tag_patterns(ref_tag_file)

    assert len(patterns) == 3
    names = {p.name for p in patterns}

    # explicit names
    assert "metaphys" in names
    assert "physic" in names

    # auto-named pattern
    auto = [p for p in patterns if p.name.startswith("pattern_")]
    assert len(auto) == 1


def test_load_ref_tag_patterns_compiles_regex(ref_tag_file: Path):
    patterns = load_ref_tag_patterns(ref_tag_file)
    for p in patterns:
        assert isinstance(p.compiled.pattern, str)


# ---------------------------------------------------------------------------
# strip_and_count_ref_tags
# ---------------------------------------------------------------------------

def test_strip_and_count_ref_tags_basic(ref_tag_file: Path):
    patterns = load_ref_tag_patterns(ref_tag_file)

    text = "Rosa metaphys. et physic. cap."
    cleaned, counter = strip_and_count_ref_tags(text, patterns)

    # tags removed
    assert "metaphys." not in cleaned
    assert "physic." not in cleaned
    assert "cap." not in cleaned

    # counts correct
    assert counter["metaphys"] == 1
    assert counter["physic"] == 1
    # auto-named pattern
    auto_keys = [k for k in counter if k.startswith("pattern_")]
    assert len(auto_keys) == 1
    assert counter[auto_keys[0]] == 1


def test_strip_and_count_multiple_occurrences(tmp_path: Path):
    p = tmp_path / "ref_tags.txt"
    p.write_text("cap\tcap\\.\n", encoding="utf-8")

    patterns = load_ref_tag_patterns(p)
    text = "cap. cap. cap."
    cleaned, counter = strip_and_count_ref_tags(text, patterns)

    assert counter["cap"] == 3
    assert "cap." not in cleaned


def test_strip_and_count_no_match(tmp_path: Path):
    p = tmp_path / "ref_tags.txt"
    p.write_text("cap\tcap\\.\n", encoding="utf-8")

    patterns = load_ref_tag_patterns(p)
    text = "rosa puella"
    cleaned, counter = strip_and_count_ref_tags(text, patterns)

    assert cleaned == text
    assert counter == Counter()