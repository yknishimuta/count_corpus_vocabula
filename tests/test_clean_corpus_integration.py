from __future__ import annotations

from pathlib import Path

import pytest

from nlpo_toolkit.latin.cleaners import run_clean_corpus


def test_run_clean_corpus_uses_external_rules_path(tmp_path: Path, monkeypatch):
    """
    Integration test from count_corpus side:
    - config.yml and rules.yml live under count_corpus (tmp_path)
    - run_clean_corpus resolves rules_path and lexicon_map_path relative to config.yml
    - cleaned output is produced
    - ref_events.tsv is produced
    - lexicon map is applied as the 3rd layer (ipsus -> ipse)
    """

    # count_corpus-like layout under tmp_path
    base = tmp_path / "count_corpus"
    cfg_dir = base / "config"
    rules_dir = cfg_dir / "latin_cleaners"
    inp_dir = base / "input"
    out_dir = base / "output"

    rules_dir.mkdir(parents=True, exist_ok=True)
    inp_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # 1) rules yaml (externalized)
    rules_path = rules_dir / "corpus_corporum.yml"
    rules_path.write_text(
        r"""
substitute_patterns:
  - name: inline_footnote_reference
    enabled: true
    pattern: '\[\d+\]'
    repl: ''
""".lstrip(),
        encoding="utf-8",
    )

    # 1b) lexicon map tsv (externalized)
    lexicon_map_path = rules_dir / "lexicon_map.tsv"
    lexicon_map_path.write_text(
        "# from\tto\n"
        "ipsus\tipse\n",
        encoding="utf-8",
    )

    # 2) input text: [12] should be removed by rules, and ipsus should become ipse by lexicon map
    (inp_dir / "a.txt").write_text("abc ipsus [12] def\n", encoding="utf-8")

    # 3) clean config yaml that lives in count_corpus/config
    config_path = cfg_dir / "clean.yml"
    config_path.write_text(
        """
kind: corpus_corporum
input: ../input
output: ../output
output_filename_template: "{stem}.cleaned.{ext}"
ref_tsv: ref_events.tsv
doc_id_prefix: TEST
rules_path: latin_cleaners/corpus_corporum.yml
lexicon_map_path: latin_cleaners/lexicon_map.tsv
""".lstrip(),
        encoding="utf-8",
    )

    # IMPORTANT: change CWD to ensure paths are resolved from config.yml dir, not CWD
    monkeypatch.chdir(tmp_path)

    # run
    rc = run_clean_corpus.main(argv=[str(config_path)])
    assert rc == 0

    # output file exists
    cleaned = out_dir / "a.cleaned.txt"
    assert cleaned.exists()

    out = cleaned.read_text(encoding="utf-8")

    # rules_path substitution applied
    assert "[12]" not in out

    # lexicon map applied (3rd layer)
    assert "ipse" in out
    assert "ipsus" not in out

    # ref_events.tsv should be written next to config file (cfg_dir)
    ref_tsv = cfg_dir / "ref_events.tsv"
    assert ref_tsv.exists()
    assert "inline_footnote_reference" in ref_tsv.read_text(encoding="utf-8")

def test_run_clean_corpus_bad_rules_path_raises(tmp_path):
    base = tmp_path / "count_corpus"
    cfg_dir = base / "config"
    inp_dir = base / "input"
    out_dir = base / "output"
    inp_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    (inp_dir / "a.txt").write_text("abc [12] def\n", encoding="utf-8")

    config_path = cfg_dir / "clean.yml"
    config_path.write_text(
        """
kind: corpus_corporum
input: ../input
output: ../output
rules_path: latin_cleaners/NO_SUCH.yml
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        run_clean_corpus.main(argv=[str(config_path)])
