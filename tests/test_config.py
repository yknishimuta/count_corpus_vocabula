from __future__ import annotations

from pathlib import Path
import pytest

from count_corpus_vocabula.config import load_config


def test_load_config_accepts_preprocess_and_groups(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "preprocess:",
                "  kind: cleaner",
                "  config: cleaners/config/sample.yml",
                "groups:",
                "  text:",
                "    files:",
                "      - cleaned/*.txt",
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

    assert "preprocess" in cfg
    assert cfg["preprocess"]["kind"] == "cleaner"
    assert cfg["preprocess"]["config"] == "cleaners/config/sample.yml"
    assert "groups" in cfg
    assert "text" in cfg["groups"]
    assert cfg["groups"]["text"]["files"] == ["cleaned/*.txt"]


def test_load_config_normalizes_group_to_groups(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        "\n".join(
            [
                "group:",
                "  name: text",
                "  files:",
                "    - input/*.txt",
                "out_dir: output",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert "groups" in cfg
    assert "text" in cfg["groups"]
    assert cfg["groups"]["text"]["files"] == ["input/*.txt"]


def test_load_config_rejects_missing_groups_and_group(tmp_path: Path):
    cfg_path = tmp_path / "invalid.yml"
    cfg_path.write_text("out_dir: output\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"define 'groups' or 'group'"):
        load_config(cfg_path)


def test_load_config_rejects_deprecated_cleaner_config(tmp_path: Path):
    cfg_path = tmp_path / "old.yml"
    cfg_path.write_text("cleaner_config: cleaners/config/sample.yml\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Deprecated key 'cleaner_config'"):
        load_config(cfg_path)

