from __future__ import annotations

from pathlib import Path
from collections import Counter
import csv
import sys

import pytest

# Make sure project root is on sys.path so we can import the script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import count_corpus_vocabula_local as mod


def test_e2e_small_input_creates_csv_and_summary(tmp_path, monkeypatch):
    """
    End-to-end test with a tiny real input corpus.

    This test:
      - creates a small input/ directory under tmp_path
      - writes a single Latin text file
      - provides a fake config via load_config
      - runs main() without mocking I/O or NLP
      - verifies that a CSV and summary.txt are actually created.
    """

    # Prepare a tiny input corpus under tmp_path
    input_dir = tmp_path / "input" / "G1"
    input_dir.mkdir(parents=True, exist_ok=True)

    text_path = input_dir / "g1_sample.txt"
    text_path.write_text(
        "Puella rosam amat. Rosa pulchra est.\n",
        encoding="utf-8",
    )

    # Output directory for this test run
    out_dir = tmp_path / "output"

    # Fake configuration returned by load_config
    cfg = {
        "out_dir": str(out_dir),
        "language": "la",
        "stanza_package": "perseus",
        "cpu_only": True,
        "groups": {
            # Use a glob pattern pointing to the file we just created
            "G1": {
                "files": [str(input_dir / "*.txt")],
            },
        },
    }

    def fake_load_config(path: Path):
        # main() passes script_dir / "groups.config.yml" here;
        # we ignore the actual contents and return our test config.
        assert path.name == "groups.config.yml"
        return cfg

    monkeypatch.setattr(mod, "load_config", fake_load_config)

    # Make groups.config.yml "exist" so main() does not raise
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name == "groups.config.yml":
            return True
        return real_exists(self)

    monkeypatch.setattr(mod.Path, "exists", fake_exists)

    # Run main()
    rc = mod.main()
    assert rc == 0

    # Verify that CSV and summary.txt were actually created
    csv_path = out_dir / "noun_frequency_G1.csv"
    summary_path = out_dir / "summary.txt"

    assert csv_path.is_file(), f"Expected CSV not found: {csv_path}"
    assert summary_path.is_file(), f"Expected summary not found: {summary_path}"

    # Check CSV header and that there is at least one data row
    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    assert rows, "CSV should not be empty"
    header = rows[0]
    assert header == ["word", "frequency"]

    # There should be at least one noun counted
    data_rows = rows[1:]
    assert len(data_rows) >= 1

    # Each data row should have (word, frequency) with an integer frequency
    for word, freq in data_rows:
        assert isinstance(word, str)
        assert word  # non-empty
        # frequency is written as string; ensure it parses as int
        int(freq)

    # Check that summary contains an entry for G1
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "G1:" in summary_text

def test_pipeline_mode_runs_cleaner_and_counts(tmp_path, monkeypatch):
    """
    Pipeline-mode E2E-ish test (without running real Stanza / real cleaner).

    This test verifies:
      - main() detects pipeline mode when 'groups' is absent and 'cleaner_config' is present
      - cleaner is invoked
      - cleaned output directory is inferred from cleaner config's 'output'
      - groups are auto-generated as cleaned_dir/*.txt
      - default out_dir becomes cleaned_dir/vocab when out_dir is not specified
      - CSV and summary.txt are created
    """

    # Prepare fake cleaner config file
    script_dir = tmp_path / "runner_dir"
    script_dir.mkdir(parents=True, exist_ok=True)

    cleaner_cfg_path = script_dir / "cleaner.yml"
    cleaned_dir = script_dir / "cleaned"

    cleaner_cfg_path.write_text(
        "\n".join(
            [
                "kind: corpus_corporum",
                "input: input",
                "output: cleaned",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Fake groups.config.yml load (pipeline mode)
    cfg = {
        "cleaner_config": str(cleaner_cfg_path),
        "language": "la",
        "stanza_package": "perseus",
        "cpu_only": True,
        # NOTE: out_dir intentionally omitted to test default: cleaned_dir/vocab
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

    # Stub cleaner runner:
    # create cleaned_dir + one cleaned text file.

    cleaner_called = {"ok": False}

    def fake_cleaner_main(argv):
        # argv should include the cleaner config path
        assert argv and Path(argv[0]).resolve() == cleaner_cfg_path.resolve()
        cleaner_called["ok"] = True

        cleaned_dir.mkdir(parents=True, exist_ok=True)
        (cleaned_dir / "c1.txt").write_text(
            "Puella rosam amat.\n",
            encoding="utf-8",
        )
        (cleaned_dir / "c2.txt").write_text(
            "Rosa pulchra est.\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(mod.clean_mod, "main", fake_cleaner_main)


    # Stub NLP pipeline + counting to avoid Stanza downloads.
    # We only need deterministic output for CSV creation.
    def fake_build_pipeline(language, stanza_package, cpu_only):
        # return any opaque objects
        return object(), {"language": language, "package": stanza_package}

    def fake_count_group(text, nlp, gname):
        return Counter({"rosa": 2, "puella": 1})

    monkeypatch.setattr(mod, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(mod, "count_group", fake_count_group)

    monkeypatch.setattr(mod, "render_stanza_package_table", lambda nlp, pkg: ["[stanza stub]"])


    # Ensure mod.main() looks for groups.config.yml under our tmp script_dir.
    # main() does: script_dir = Path(__file__).resolve().parent
    # So we patch mod.__file__ to appear inside script_dir.
    monkeypatch.setattr(mod, "__file__", str(script_dir / "count_corpus_vocabula_local.py"))


    # Run main()
    rc = mod.main()
    assert rc == 0
    assert cleaner_called["ok"] is True

    # Verify output paths
    # default out_dir should be cleaned_dir / "vocab"
    out_dir = cleaned_dir / "vocab"

    csv_path = out_dir / "noun_frequency_text.csv"
    summary_path = out_dir / "summary.txt"

    assert csv_path.is_file(), f"Expected CSV not found: {csv_path}"
    assert summary_path.is_file(), f"Expected summary not found: {summary_path}"

    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    assert rows and rows[0] == ["word", "frequency"]
    assert len(rows) >= 2

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "text:" in summary_text
