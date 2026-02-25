from collections import Counter
from pathlib import Path

import pytest

import count_corpus_vocabula_local as mod


def test_analysis_unit_invalid_raises_value_error(tmp_path, monkeypatch):
    script_dir = tmp_path / "runner_dir"
    (script_dir / "config").mkdir(parents=True, exist_ok=True)

    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "t1.txt").write_text("Puella rosam amat.\n", encoding="utf-8")

    out_dir = script_dir / "output"

    cfg = {
        "groups": {"text": {"files": [str(data_dir / "*.txt")]}},
        "out_dir": str(out_dir),
        "analysis_unit": "LEMMAAAA",  # invalid on purpose
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

    monkeypatch.setattr(mod, "__file__", str(script_dir / "count_corpus_vocabula_local.py"))

    monkeypatch.setattr(mod, "build_pipeline", lambda *a, **k: (object(), "perseus"))
    monkeypatch.setattr(mod, "count_group", lambda *a, **k: Counter({"x": 1}))
    monkeypatch.setattr(mod, "render_stanza_package_table", lambda *a, **k: ["[stanza stub]"])

    with pytest.raises(ValueError, match=r"analysis_unit"):
        mod.main()
