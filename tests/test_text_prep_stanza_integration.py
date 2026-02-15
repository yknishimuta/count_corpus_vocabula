from __future__ import annotations

import os
import pytest

from count_corpus_vocabula.text_prep import one_sentence_per_line
from nlpo_toolkit.nlp import build_sentence_splitter


@pytest.mark.stanza_integration
def test_one_sentence_per_line_with_real_stanza_tokenizer():
    if os.environ.get("RUN_STANZA_TESTS") != "1":
        pytest.skip("Set RUN_STANZA_TESTS=1 to run stanza integration tests")

    pytest.importorskip("stanza")

    splitter = build_sentence_splitter(language="la", stanza_package="perseus", cpu_only=True)

    raw = "Puella rosam amat.\nRosa pulchra est."
    out = one_sentence_per_line(raw, splitter)

    lines = [x for x in out.splitlines() if x.strip()]
    assert len(lines) >= 2
    assert lines[0].endswith(".")
