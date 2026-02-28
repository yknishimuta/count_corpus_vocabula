# tests/test_runner_run.py
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

import count_corpus_vocabula.runner as runner_mod  # 実際のファイル名に合わせて


def test_run_minimal_success(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    # --- fake config
    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "language": "la",
            "stanza_package": "perseus",
            "cpu_only": True,
            "groups": {
                "g1": {"files": ["a.txt"]},
                "g2": {"files": ["b.txt"]},
            },
        }

    # --- stub preprocess
    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)

    # --- stub IO
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path(p) for p in patterns])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "AAA BBB")

    # --- stub NLP & counter
    def build_pipeline_fn(language, stanza_package, cpu_only):
        return object(), stanza_package

    def count_group_fn(text, nlp, use_lemma=True):
        assert text  # joined
        return Counter({"deus": 2, "angelus": 1})

    # --- capture CSV writes
    calls = []
    def write_frequency_csv(path, counter, header):
        calls.append((Path(path), dict(counter), header))

    monkeypatch.setattr(runner_mod, "write_frequency_csv", write_frequency_csv)

    # --- meta
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {"py": "x"})
    meta_calls = []
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: meta_calls.append((meta, Path(out_dir))))

    # --- stanza package table
    def render_stanza_package_table_fn(nlp, stanza_package):
        return ["table: ok"]

    rc = runner_mod.run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=build_pipeline_fn,
        build_sentence_splitter_fn=None,
        count_group_fn=count_group_fn,
        render_stanza_package_table_fn=render_stanza_package_table_fn,
    )
    assert rc == 0

    out_dir = tmp_path / "output"
    assert (out_dir / "summary.txt").exists()

    # CSV: groups分
    assert len(calls) == 2
    assert calls[0][0].name == "noun_frequency_g1.csv"
    assert calls[1][0].name == "noun_frequency_g2.csv"

    # meta
    assert len(meta_calls) == 1
    meta, meta_out_dir = meta_calls[0]
    assert meta_out_dir == out_dir
    assert meta["analysis_unit"] == "lemma"
    assert meta["environment"] == {"py": "x"}

def test_run_analysis_unit_surface_passes_use_lemma_false(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "analysis_unit": "surface",
            "groups": {"g": {"files": ["a.txt"]}},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path("a.txt")])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "X")

    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)

    monkeypatch.setattr(runner_mod, "write_frequency_csv", lambda *a, **k: None)

    seen = {}
    def count_group_fn(text, nlp, use_lemma=True):
        seen["use_lemma"] = use_lemma
        return Counter({"X": 1})

    rc = runner_mod.run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=count_group_fn,
        render_stanza_package_table_fn=lambda *a, **k: [],
    )
    assert rc == 0
    assert seen["use_lemma"] is False

def test_run_dictcheck_requires_wordlist(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"g": {"files": ["a.txt"]}},
            "dictcheck": {"enabled": True},  # wordlistなし
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path("a.txt")])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "X")
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)

    with pytest.raises(ValueError):
        runner_mod.run(
            script_dir=script_dir,
            config_path=config_path,
            load_config_fn=load_config_fn,
            clean_mod=object(),
            build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
            build_sentence_splitter_fn=None,
            count_group_fn=lambda *a, **k: Counter({"a": 1}),
            render_stanza_package_table_fn=lambda *a, **k: [],
        )


def test_run_dictcheck_writes_known_unknown(tmp_path: Path, monkeypatch):
    script_dir = tmp_path
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("dummy", encoding="utf-8")

    wl = tmp_path / "wordlist.txt"
    wl.write_text("deus\n", encoding="utf-8")

    def load_config_fn(_p: Path):
        return {
            "out_dir": "output",
            "groups": {"g": {"files": ["a.txt"]}},
            "dictcheck": {"enabled": True, "wordlist": str(wl)},
        }

    monkeypatch.setattr(runner_mod, "run_preprocess_if_needed", lambda **kwargs: None)
    monkeypatch.setattr(runner_mod, "expand_cleaned_dir_placeholders", lambda patterns, cleaned_dir: patterns)
    monkeypatch.setattr(runner_mod, "expand_globs", lambda patterns: [Path("a.txt")])
    monkeypatch.setattr(runner_mod, "read_concat", lambda files: "X")
    monkeypatch.setattr(runner_mod, "build_run_meta", lambda **kwargs: {})
    monkeypatch.setattr(runner_mod, "collect_runtime_environment", lambda _sd: {})
    monkeypatch.setattr(runner_mod, "write_run_meta", lambda meta, out_dir: None)

    calls = []
    monkeypatch.setattr(runner_mod, "write_frequency_csv",
                        lambda path, c, header: calls.append(Path(path).name))

    rc = runner_mod.run(
        script_dir=script_dir,
        config_path=config_path,
        load_config_fn=load_config_fn,
        clean_mod=object(),
        build_pipeline_fn=lambda *a, **k: (object(), "perseus"),
        build_sentence_splitter_fn=None,
        count_group_fn=lambda *a, **k: Counter({"deus": 2, "angelus": 1}),
        render_stanza_package_table_fn=lambda *a, **k: [],
    )
    assert rc == 0

    assert calls == [
        "noun_frequency_g.csv",
        "noun_frequency_g.known.csv",
        "noun_frequency_g.unknown.csv",
    ]
