from __future__ import annotations

from count_corpus_vocabula.text_prep import normalize_linebreaks_and_hyphens


def test_normalize_joins_hyphen_newline():
    raw = "prae-\nexistit"
    assert normalize_linebreaks_and_hyphens(raw) == "praeexistit"


def test_normalize_folds_single_newlines_keeps_blank_lines():
    raw = "a\nb\n\nc\nd\n"
    assert normalize_linebreaks_and_hyphens(raw) == "a b\n\nc d"


def test_normalize_collapses_spaces():
    raw = "a   b\t\tc\n\nd"
    assert normalize_linebreaks_and_hyphens(raw) == "a b c\n\nd"
