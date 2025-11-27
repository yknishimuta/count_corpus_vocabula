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
from count_corpus_vocabula.counters import count_group
from count_corpus_vocabula.compose import compose_all
from nlpo_toolkit.nlp import render_stanza_package_table

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "groups.config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    out_dir = Path(cfg.get("out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # language setting
    language = cfg.get("language", "la")
    stanza_pkg = cfg.get("stanza_package", "perseus")
    cpu_only = bool(cfg.get("cpu_only", True))

    nlp, package = build_pipeline(language=language, stanza_package=stanza_pkg, cpu_only=cpu_only)

    group_counts: Dict[str, Counter] = {}

    for gname, gdef in cfg["groups"].items():
        files = expand_globs(gdef["files"])
        if not files:
            print(f"[WARN] group '{gname}' matched no files; skipping")
            continue
        text = read_concat(files)
        print(f"[Processing] {gname}: {len(text):,} chars / {len(files)} files")
        total = count_group(text, nlp, gname)
        group_counts[gname] = total
        save_counter_csv(out_dir / f"noun_frequency_{gname}.csv", total)

    if len(group_counts) >= 2:
        all_counts = compose_all(group_counts)
        group_counts["ALL"] = all_counts
        save_counter_csv(out_dir / "noun_frequency_ALL.csv", all_counts)

    lines = ["=== Summary ==="]
    for k in sorted(group_counts.keys()):
        lines.append(f"{k}: raw_unique={len(group_counts[k])}")
    lines.append("")
    lines.extend(render_stanza_package_table(nlp, package))
    write_summary(out_dir / "summary.txt", lines)

    print("[Done] Saved to", out_dir)
    return 0