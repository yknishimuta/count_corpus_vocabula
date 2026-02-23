# count_corpus_vocabula_local.py
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

# ----------------------------
# Compatibility hooks for tests
# ----------------------------

# tests expect: mod.clean_mod.main to exist and be monkeypatchable
try:
    from count_corpus_vocabula import clean as clean_mod  # type: ignore
except Exception:
    try:
        from count_corpus_vocabula import cleaner as clean_mod  # type: ignore
    except Exception:
        clean_mod = SimpleNamespace(main=lambda argv: 0)

# tests expect: mod.load_config to exist and be monkeypatchable
from count_corpus_vocabula.config import load_config  # noqa: F401

# tests expect these names on THIS module (they monkeypatch them here)
from count_corpus_vocabula.nlp_hooks import (  # noqa: F401
    build_pipeline,
    build_sentence_splitter,
    count_group,
    render_stanza_package_table,
)

# tests also call this helper directly
from count_corpus_vocabula.preprocess import expand_cleaned_dir_placeholders as _expand_cleaned_dir_placeholders  # noqa: F401


def main() -> int:
    """
    Thin shim: keeps the monkeypatch surface in this module,
    but runs the real logic in count_corpus_vocabula.runner.
    """
    from count_corpus_vocabula.runner import run

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config" / "groups.config.yml"

    return run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config,  # may be monkeypatched on this module
        clean_mod=clean_mod,         # may be monkeypatched on clean_mod.main
        build_pipeline_fn=build_pipeline,
        build_sentence_splitter_fn=build_sentence_splitter,
        count_group_fn=count_group,
        render_stanza_package_table_fn=render_stanza_package_table,
    )


if __name__ == "__main__":
    raise SystemExit(main())