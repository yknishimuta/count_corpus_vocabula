from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional


_RE_HYPHEN_NL = re.compile(r"(\w)-\n(\w)")

_RE_SINGLE_NL_IN_PARA = re.compile(r"(?<!\n)\n(?!\n)")


def normalize_linebreaks_and_hyphens(raw: str) -> str:
    """
    Stable normalization (unit-test target):
      - normalize newlines
      - join hyphen+newline word breaks
      - fold single newlines within paragraphs into spaces (keep blank lines)
      - compress whitespace
    """
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = _RE_HYPHEN_NL.sub(r"\1\2", s)
    s = _RE_SINGLE_NL_IN_PARA.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def sentences_via_stanza_tokenize(text: str, nlp) -> list[str]:
    """
    Sentence split using stanza pipeline's tokenizer.
    Assumes nlp(text) returns doc with doc.sentences, each having tokens/words.
    """
    doc = nlp(text)
    out: list[str] = []
    for sent in getattr(doc, "sentences", []):
        # Stanza Sentence has .text in newer versions; fallback to token/word concat
        st = getattr(sent, "text", None)
        if st:
            st = st.strip()
            if st:
                out.append(st)
            continue

        # fallback: join token texts if available
        toks = []
        for w in getattr(sent, "words", []):
            t = getattr(w, "text", None)
            if t:
                toks.append(t)
        if toks:
            out.append(" ".join(toks).strip())
    return [x for x in out if x]


def one_sentence_per_line(raw: str, nlp) -> str:
    """
    Main helper:
      normalize_linebreaks_and_hyphens(raw)  -> stable normalized text
      stanza tokenizer sentence split        -> 1 sentence per line
    """
    normalized = normalize_linebreaks_and_hyphens(raw)
    sents = sentences_via_stanza_tokenize(normalized, nlp)
    return "\n".join(sents) + ("\n" if sents else "")