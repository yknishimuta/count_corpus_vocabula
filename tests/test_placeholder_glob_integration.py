from __future__ import annotations

from pathlib import Path
import sys

# count_corpus_vocabula_local を import できるようにする
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import count_corpus_vocabula_local as local_mod
from count_corpus_vocabula.io_utils import expand_globs


def test_placeholder_expand_then_glob_finds_files(tmp_path: Path):
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # matched
    f1 = cleaned_dir / "a.txt"
    f2 = cleaned_dir / "b.txt"
    f1.write_text("a", encoding="utf-8")
    f2.write_text("b", encoding="utf-8")

    # not matched (suffix違い)
    (cleaned_dir / "c.md").write_text("c", encoding="utf-8")

    patterns = ["{cleaned_dir}/*.txt"]

    expanded = local_mod._expand_cleaned_dir_placeholders(patterns, cleaned_dir)
    files = expand_globs(expanded)

    assert files == sorted([f1.resolve(), f2.resolve()])


def test_placeholder_expand_then_glob_recursive(tmp_path: Path):
    cleaned_dir = tmp_path / "cleaned"
    (cleaned_dir / "sub").mkdir(parents=True, exist_ok=True)

    # matched recursively
    f1 = cleaned_dir / "sub" / "x.txt"
    f1.write_text("x", encoding="utf-8")

    patterns = ["{cleaned_dir}/**/*.txt"]

    expanded = local_mod._expand_cleaned_dir_placeholders(patterns, cleaned_dir)
    files = expand_globs(expanded)

    assert files == [f1.resolve()]
