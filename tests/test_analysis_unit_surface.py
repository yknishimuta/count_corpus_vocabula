import csv
from collections import Counter
from pathlib import Path

import count_corpus_vocabula_local as mod


def test_analysis_unit_surface_writes_word_frequency_and_passes_use_lemma_false(tmp_path, monkeypatch):
    # --- Arrange: fake "repo" layout that main() expects ---
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    # Create a real input file to match glob
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "analysis_unit": "surface",
        "language": "la",
        "stanza_package": "perseus",
        "cpu_only": True,
        "dictcheck": {"enabled": False},
    }

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

    # Stub NLP build so we don't download Stanza models
    monkeypatch.setattr(mod, "build_pipeline", lambda *a, **k: (object(), "perseus"))
    monkeypatch.setattr(mod, "render_stanza_package_table", lambda *a, **k: ["[stanza stub]"])

    # Assert use_lemma=False is passed into count_group()
    def fake_count_group(text, nlp, **kwargs):
        assert kwargs.get("use_lemma") is False
        return Counter({"Rosa": 2, "puella": 1})

    monkeypatch.setattr(mod, "count_group", fake_count_group)

    # --- Act ---
    rc = mod.main()
    assert rc == 0

    # --- Assert: header changes for surface ---
    csv_path = out_dir / "noun_frequency_text.csv"
    assert csv_path.exists()

    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    assert rows[0] == ["word", "frequency"]
