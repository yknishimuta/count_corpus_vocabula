from __future__ import annotations

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import count_corpus_vocabula_local as mod


def test_resolve_cleaner_output_dir_relative(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    cleaner_cfg = cfg_dir / "cleaner.yml"
    cleaner_cfg.write_text(
        "\n".join(
            [
                "kind: corpus_corporum",
                "input: input",
                "output: cleaned", #relative
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    got = mod._resolve_cleaner_output_dir(cleaner_cfg)
    assert got == (cfg_dir / "cleaned").resolve()


def test_resolve_cleaner_output_dir_absolute(tmp_path: Path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    out_abs = (tmp_path / "somewhere" / "cleaned_abs").resolve()
    cleaner_cfg = cfg_dir / "cleaner.yml"
    cleaner_cfg.write_text(
        "\n".join(
            [
                "kind: corpus_corporum",
                "input: input",
                f"output: {out_abs}",  # absolute
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    got = mod._resolve_cleaner_output_dir(cleaner_cfg)
    assert got == out_abs
