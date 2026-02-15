from __future__ import annotations

from pathlib import Path
from collections import Counter
import csv

import count_corpus_vocabula_local as mod


def test_dictcheck_enabled_creates_known_unknown(tmp_path, monkeypatch):
    # --- Arrange: fake "repo" layout that main() expects ---
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    # Create a real input file to match glob
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    # Wordlist (relative path case): data/wordlists/latin_words.txt
    wordlist_dir = data_dir / "wordlists"
    wordlist_dir.mkdir(parents=True, exist_ok=True)
    wordlist_path = wordlist_dir / "latin_words.txt"
    wordlist_path.write_text(
        "\n".join(["rosa"]),  # only "rosa" is known
        encoding="utf-8",
    )

    cfg = {
        "groups": {
            "text": {"files": [str(data_dir / "*.txt")]}
        },
        "out_dir": str(out_dir),
        "language": "la",
        "stanza_package": "perseus",
        "cpu_only": True,
        "dictcheck": {
            "enabled": True,
            # IMPORTANT: main() resolves relative to script_dir
            "wordlist": "data/wordlists/latin_words.txt",
            # optional: you can omit these because we default to word/frequency in the helper
            # "lemma_column": "word",
            # "count_column": "frequency",
        },
    }

    # main() reads: script_dir/config/groups.config.yml
    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg

    monkeypatch.setattr(mod, "load_config", fake_load_config)

    # Make main() think config exists
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "groups.config.yml":
            return True
        return real_exists(self)

    monkeypatch.setattr(mod.Path, "exists", fake_exists)

    # Patch __file__ so script_dir resolves to our tmp runner_dir
    monkeypatch.setattr(mod, "__file__", str(script_dir / "count_corpus_vocabula_local.py"))

    # Stub NLP build + counting so we don't download Stanza models
    monkeypatch.setattr(mod, "build_pipeline", lambda **kwargs: (object(), {"package": "stub"}))
    monkeypatch.setattr(
        mod,
        "count_group",
        lambda text, nlp, **kwargs: Counter({"rosa": 2, "puella": 1}),
    )
    monkeypatch.setattr(mod, "render_stanza_package_table", lambda nlp, pkg: ["[stanza stub]"])

    # --- Act ---
    rc = mod.main()
    assert rc == 0

    # --- Assert ---
    base_csv = out_dir / "noun_frequency_text.csv"
    known_csv = out_dir / "noun_frequency_text.known.csv"
    unknown_csv = out_dir / "noun_frequency_text.unknown.csv"

    assert base_csv.exists()
    assert known_csv.exists()
    assert unknown_csv.exists()

    # base csv header is ["word","frequency"]
    rows = list(csv.reader(base_csv.open(encoding="utf-8")))
    assert rows[0] == ["word", "frequency"]

    # known should contain rosa, unknown should contain puella
    known_rows = list(csv.DictReader(known_csv.open(encoding="utf-8")))
    unknown_rows = list(csv.DictReader(unknown_csv.open(encoding="utf-8")))

    assert [r["word"] for r in known_rows] == ["rosa"]
    assert [r["word"] for r in unknown_rows] == ["puella"]


def test_dictcheck_enabled_requires_wordlist(tmp_path, monkeypatch):
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "language": "la",
        "stanza_package": "perseus",
        "cpu_only": True,
        "dictcheck": {
            "enabled": True,
            # wordlist missing on purpose
        },
    }

    def fake_load_config(path: Path):
        assert path.name == "groups.config.yml"
        return cfg

    monkeypatch.setattr(mod, "load_config", fake_load_config)

    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "groups.config.yml":
            return True
        return real_exists(self)

    monkeypatch.setattr(mod.Path, "exists", fake_exists)
    monkeypatch.setattr(mod, "__file__", str(script_dir / "count_corpus_vocabula_local.py"))

    monkeypatch.setattr(mod, "build_pipeline", lambda **kwargs: (object(), {"package": "stub"}))
    monkeypatch.setattr(mod, "count_group", lambda text, nlp, **kwargs: Counter({"rosa": 1}))
    monkeypatch.setattr(mod, "render_stanza_package_table", lambda nlp, pkg: ["[stanza stub]"])

    import pytest
    with pytest.raises(ValueError, match=r"dictcheck\.wordlist"):
        mod.main()
