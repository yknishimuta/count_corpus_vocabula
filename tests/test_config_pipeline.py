from __future__ import annotations

from pathlib import Path
import pytest

from count_corpus_vocabula.config import load_config


def test_load_config_accepts_pipeline_mode(tmp_path: Path):
    """
    load_config() should accept pipeline-mode config (no 'groups', has 'cleaner_config').
    """
    cfg_path = tmp_path / "pipeline.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "cleaner_config: cleaners/config/sample.yml",
                "out_dir: output",
                "language: la",
                "stanza_package: perseus",
                "cpu_only: true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert "groups" not in cfg  # pipeline mode: groups not required at load time
    assert cfg["cleaner_config"] == "cleaners/config/sample.yml"
    assert cfg.get("language") == "la"
    assert cfg.get("stanza_package") == "perseus"
    assert cfg.get("cpu_only") is True


def test_load_config_rejects_missing_groups_and_cleaner_config(tmp_path: Path):
    """
    load_config() should reject configs that have neither 'groups' nor 'cleaner_config'.
    """
    cfg_path = tmp_path / "invalid.yml"
    cfg_path.write_text("out_dir: output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="either 'groups'|Either 'groups'|cleaner_config"):
        load_config(cfg_path)
