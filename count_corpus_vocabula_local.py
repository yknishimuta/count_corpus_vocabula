#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict
from collections import Counter

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from count_corpus_vocabula.config import load_config
from count_corpus_vocabula.io_utils import expand_globs, read_concat, save_counter_csv, write_summary
from count_corpus_vocabula.nlp_utils import build_pipeline
from count_corpus_vocabula.counters import count_group, load_exclude_list
from count_corpus_vocabula.compose import compose_all
from nlpo_toolkit.nlp import render_stanza_package_table

from nlpo_toolkit.latin.cleaners.config_loader import load_clean_config
from nlpo_toolkit.latin.cleaners import run_clean_config as clean_mod

def _resolve_cleaner_output_dir(cleaner_config_path: Path) -> Path:
    """
    Load the cleaner config and resolve its 'output' path
    (relative to the cleaner config file directory).
    """

    cleaner_cfg = load_clean_config(cleaner_config_path)
    raw_output = cleaner_cfg["output"]

    base_dir = cleaner_config_path.parent
    out_path = Path(raw_output)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    return out_path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config" / "groups.config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    if "groups" in cfg:
        # Legacy mode: use groups as defined in YAML
        groups_cfg = cfg["groups"]
        out_dir = Path(cfg.get("out_dir", "output"))
    else:
        # Pipeline mode: run cleaner first, then auto-generate groups
        cleaner_config_raw = cfg.get("cleaner_config")
        if not cleaner_config_raw:
            raise ValueError("Either 'groups' or 'cleaner_config' must be defined in groups.config.yml")
        
        cleaner_config_path = Path(cleaner_config_raw)
        if not cleaner_config_path.is_absolute():
            cleaner_config_path = (script_dir / cleaner_config_path).resolve()

        if not cleaner_config_path.exists():
            raise FileNotFoundError(f"Cleaner config file not found: {cleaner_config_path}")

        print(f"[Cleaner] Running with config: {cleaner_config_path}")
        # Run the cleaner CLI (this will read its own YAML and write cleaned files)
        clean_mod.main([str(cleaner_config_path)])

        cleaned_dir = _resolve_cleaner_output_dir(cleaner_config_path)
        print(f"[Cleaner] Output directory (cleaned text): {cleaned_dir}")

        # Auto-generate a single group pointing to all cleaned .txt files
        groups_cfg = {
            "text": {
                "files": [str(cleaned_dir / "*.txt")],
            }
        }
        cfg["groups"] = groups_cfg

        out_dir = Path(cfg.get("out_dir", cleaned_dir / "vocab"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # language setting
    language = cfg.get("language", "la")
    stanza_package = cfg.get("stanza_package", "perseus")
    cpu_only = bool(cfg.get("cpu_only", True))

    nlp, package = build_pipeline(language=language, stanza_package=stanza_package, cpu_only=cpu_only)

    group_counts: Dict[str, Counter] = {}

    for gname, gdef in cfg["groups"].items():
        files = expand_globs(gdef["files"])
        if not files:
            print(f"[WARN] group '{gname}' matched no files; skipping")
            continue
        text = read_concat(files)
        print(f"[Processing] {gname}: {len(text):,} chars / {len(files)} files")
        exclude = load_exclude_list("config/exclude_lemmas.txt")
        total = count_group(text, nlp, label=gname, exclude_lemmas=exclude)
        group_counts[gname] = total
        save_counter_csv(out_dir / f"noun_frequency_{gname}.csv", total)

    if len(group_counts) >= 2:
        all_counts = compose_all(group_counts)
        group_counts["ALL"] = all_counts
        save_counter_csv(out_dir / "noun_frequency_ALL.csv", all_counts)

    lines = ["=== Summary ==="]
    for k in sorted(group_counts.keys()):
        cnt = group_counts[k]
        total_tokens = sum(cnt.values())
        unique_types = len(cnt)

        if total_tokens > 0:
            ttr = unique_types / total_tokens
        else:
            ttr = 0.0

        lines.append(f"{k}:")
        lines.append(f"  total_tokens      = {total_tokens}")
        lines.append(f"  unique_types      = {unique_types}")
        lines.append(f"  type_token_ratio  = {ttr:.4f}")
        lines.append("")
        lines.extend(render_stanza_package_table(nlp, package))
    write_summary(out_dir / "summary.txt", lines)

    print("[Done] Saved to", out_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())