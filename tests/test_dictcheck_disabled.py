from __future__ import annotations

from pathlib import Path
from collections import Counter

import count_corpus_vocabula_local as mod


def test_dictcheck_disabled_does_not_create_known_unknown(tmp_path, monkeypatch):
    # --- Arrange: fake "repo" layout that main() expects ---
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    # Create a real input file to match glob
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    # Config returned by load_config (new design)
    cfg = {
        "groups": {
            "text": {"files": [str(data_dir / "*.txt")]}
        },
        "out_dir": str(out_dir),
        "language": "la",
        "stanza_package": "perseus",
        "cpu_only": True,
        "dictcheck": {
            "enabled": False,
            # even if present, must not be used when enabled=False
            "wordlist": str(script_dir / "data" / "wordlists" / "latin_words.txt"),
        },
    }

    def fake_load_config(path: Path):
        # main() reads: script_dir/config/groups.config.yml
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
    monkeypatch.setattr(mod, "count_group", lambda text, nlp, **kwargs: Counter({"rosa": 2}))
    monkeypatch.setattr(mod, "render_stanza_package_table", lambda nlp, pkg: ["[stanza stub]"])

    # --- Act ---
    rc = mod.main()
    assert rc == 0

    # --- Assert ---
    # Base CSV should exist
    assert (out_dir / "noun_frequency_text.csv").exists()
    assert (out_dir / "summary.txt").exists()

    # But dictcheck outputs must NOT exist when enabled=False
    assert not (out_dir / "noun_frequency_text.known.csv").exists()
    assert not (out_dir / "noun_frequency_text.unknown.csv").exists()

