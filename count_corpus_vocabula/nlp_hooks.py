# count_corpus_vocabula/nlp_hooks.py
from __future__ import annotations

from collections import Counter
from typing import List


def build_pipeline(language: str, stanza_package: str, cpu_only: bool):
    """
    Production pipeline builder (Stanza via nlpo_toolkit).
    Returns (nlp, package).
    """
    from nlpo_toolkit.nlp import build_stanza_pipeline  # type: ignore

    processors = "tokenize,pos,lemma"
    nlp = build_stanza_pipeline(
        lang=language,
        processors=processors,
        package=stanza_package,
        use_gpu=(not cpu_only),
    )
    return nlp, stanza_package


def build_sentence_splitter(language: str, stanza_package: str, cpu_only: bool):
    # Optional: tests may monkeypatch this on count_corpus_vocabula_local,
    # but runner will accept any callable. Default is "not provided".
    raise RuntimeError("build_sentence_splitter is optional and should not be required in tests")

def count_group(text: str, nlp, **kwargs) -> Counter:
    """
    Production counter: count noun lemmas using nlpo_toolkit.
    """
    from nlpo_toolkit.nlp import count_nouns_streaming  # type: ignore

    use_lemma = bool(kwargs.get("use_lemma", True))

    upos_targets = kwargs.get("upos_targets", {"NOUN"})
    if isinstance(upos_targets, set):
        upos_targets = frozenset(upos_targets)

    chunk_chars = int(kwargs.get("chunk_chars", 200_000))
    label = str(kwargs.get("label", ""))
    min_token_length = int(kwargs.get("min_token_length", 0))

    ref_tag_detector = kwargs.get("ref_tag_detector")
    ref_tag_counter = kwargs.get("ref_tag_counter")

    # ---- trace-related options ----
    trace_tsv = kwargs.get("trace_tsv")
    trace_max_rows = int(kwargs.get("trace_max_rows", 0))
    trace_only_keys = kwargs.get("trace_only_keys")
    trace_write_truncation_marker = bool(
        kwargs.get("trace_write_truncation_marker", True)
    )

    # normalize trace_only_keys (key is lowercased in counter)
    if trace_only_keys is not None:
        trace_only_keys = {
            str(x).strip().lower()
            for x in trace_only_keys
            if str(x).strip()
        }

    return count_nouns_streaming(
        text,
        nlp,
        use_lemma=use_lemma,
        upos_targets=upos_targets,
        chunk_chars=chunk_chars,
        ref_tag_detector=ref_tag_detector,
        ref_tag_counter=ref_tag_counter,
        label=label,
        trace_tsv=trace_tsv,
        trace_max_rows=trace_max_rows,
        trace_only_keys=trace_only_keys,
        min_token_length=min_token_length,
        trace_write_truncation_marker=trace_write_truncation_marker,
    )

def render_stanza_package_table(nlp, pkg: str) -> List[str]:
    return [f"package={pkg}"]