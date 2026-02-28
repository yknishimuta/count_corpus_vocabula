# tests/test_outputs.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from count_corpus_vocabula.outputs import (
    write_frequency_csv,
    build_run_meta,
    write_run_meta,
    collect_runtime_environment,
)


def test_write_frequency_csv_writes_header_and_sorts(tmp_path: Path) -> None:
    out = tmp_path / "freq.csv"
    freq = Counter({"b": 1, "a": 2, "c": 2})

    write_frequency_csv(out, freq, header=("lemma", "count"))

    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "lemma,count"

    # count desc, then key asc => a(2), c(2), b(1)
    assert text[1:] == ["a,2", "c,2", "b,1"]


def test_build_run_meta_shape_and_groups_files() -> None:
    meta = build_run_meta(groups_files={"g1": ["x.txt", "y.txt"], "g2": []}, hash_inputs=False)

    assert "generated_at" in meta
    assert meta["hash_inputs"] is False
    assert meta["groups_files"] == {"g1": ["x.txt", "y.txt"], "g2": []}


def test_write_run_meta_writes_json(tmp_path: Path) -> None:
    meta = {"hello": "world", "n": 1}
    p = write_run_meta(meta, tmp_path)

    assert p.name == "run_meta.json"
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded == meta


def test_collect_runtime_environment_does_not_crash_without_git(monkeypatch, tmp_path: Path) -> None:
    import subprocess as _subprocess

    def _boom(*args, **kwargs):
        raise _subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(_subprocess, "check_output", _boom)

    env = collect_runtime_environment(tmp_path)

    assert "python_version" in env

    assert "git_commit" in env
    assert env["git_commit"] is None