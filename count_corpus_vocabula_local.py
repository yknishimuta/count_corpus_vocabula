#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, Mapping, List, Optional
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
from count_corpus_vocabula.text_prep import one_sentence_per_line

from nlpo_toolkit.nlp import render_stanza_package_table, build_sentence_splitter
from nlpo_toolkit.latin.cleaners import run_clean_config as clean_mod
from nlpo_toolkit.latin.cleaners.config_loader import load_clean_config

def _expand_cleaned_dir_placeholders(patterns: list[str], cleaned_dir: Path | None) -> list[str]:
    """
    Expand '{cleaned_dir}' placeholder in glob patterns.

    We use string replace (not str.format) to avoid KeyError when patterns contain
    other braces (e.g., '{something_else}').
    """
    if cleaned_dir is None:
        return list(patterns)

    cd = str(cleaned_dir)
    return [p.replace("{cleaned_dir}", cd) for p in patterns]

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

def run_dictcheck_if_enabled(
    *,
    cfg: Mapping,
    out_dir: Path,
    group_counts: Mapping[str, Counter],
    script_dir: Path,
) -> None:
    dc = cfg.get("dictcheck") or {}
    dc_enabled = bool(dc.get("enabled", False))
    if not dc_enabled:
        return

    from count_corpus_vocabula.dictcheck import split_frequency_csv

    wordlist_raw = dc.get("wordlist")
    if not wordlist_raw:
        raise ValueError("dictcheck.enabled=true requires dictcheck.wordlist")

    # resolve wordlist path (relative to repo/script_dir)
    wordlist_path = Path(wordlist_raw)
    if not wordlist_path.is_absolute():
        wordlist_path = (script_dir / wordlist_path).resolve()

    if not wordlist_path.exists():
        raise FileNotFoundError(f"Wordlist not found: {wordlist_path}")

    lemma_norm_raw = dc.get("lemma_normalize")
    lemma_norm_path = None
    if lemma_norm_raw:
        p = Path(lemma_norm_raw)
        if not p.is_absolute():
            p = (script_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"lemma_normalize file not found: {p}")
        lemma_norm_path = p

    # save_counter_csv writes header ["word","frequency"]
    lemma_col = dc.get("lemma_column") or "word"
    count_col = dc.get("count_column") or "frequency"

    for gname in group_counts.keys():
        freq_csv = out_dir / f"noun_frequency_{gname}.csv"
        if not freq_csv.exists():
            continue

        known_csv = out_dir / f"noun_frequency_{gname}.known.csv"
        unknown_csv = out_dir / f"noun_frequency_{gname}.unknown.csv"

        k, u = split_frequency_csv(
            freq_csv=freq_csv,
            wordlist_path=wordlist_path,
            out_known_csv=known_csv,
            out_unknown_csv=unknown_csv,
            lemma_col=lemma_col,
            count_col=count_col,
            normalize=True,
            normalize_map_path=lemma_norm_path,
        )
        print(f"[DictCheck] {gname}: known={k} unknown={u} wordlist={wordlist_path}")

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config" / "groups.config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    # preprocess (optional)
    cleaned_dir: Path | None = None
    pp = cfg.get("preprocess")
    if pp and pp.get("kind") == "cleaner":
        cleaner_config_raw = pp.get("config")
        if not cleaner_config_raw:
            raise ValueError("'preprocess.config' is required when preprocess.kind=cleaner")

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

    # ----------------------------
    # output directory
    # ----------------------------
    out_dir = Path(cfg.get("out_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # language settings
    # ----------------------------
    language = cfg.get("language", "la")
    stanza_package = cfg.get("stanza_package", "perseus")
    cpu_only = bool(cfg.get("cpu_only", True))

    nlp, package = build_pipeline(language, stanza_package, cpu_only)
    splitter_nlp = build_sentence_splitter(language, stanza_package=package, cpu_only=cpu_only)

    group_counts: Dict[str, Counter] = {}

    # ----------------------------
    # process groups (always present)
    # ----------------------------
    for gname, gdef in cfg["groups"].items():
        patterns = gdef["files"]
        patterns = _expand_cleaned_dir_placeholders(patterns, cleaned_dir)

        files = expand_globs(patterns)

        # Safety guard: if preprocess ran, forbid mixing non-cleaned files
        if cleaned_dir is not None and files:
            cleaned_root = cleaned_dir.resolve()
            bad = []
            for f in files:
                fp = Path(f).resolve()
                if not fp.is_relative_to(cleaned_root):
                    bad.append(str(fp))
            if bad:
                raise ValueError(
                    "preprocess is enabled, but group files include non-cleaned files. "
                    f"group={gname} cleaned_dir={cleaned_root} bad_sample={bad[:3]}"
                )

        if not files:
            print(f"[WARN] group '{gname}' matched no files; skipping")
            continue

        text = read_concat(files)
        text = one_sentence_per_line(text, splitter_nlp)
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
        ttr = (unique_types / total_tokens) if total_tokens > 0 else 0.0

        lines.append(f"{k}:")
        lines.append(f"  total_tokens      = {total_tokens}")
        lines.append(f"  unique_types      = {unique_types}")
        lines.append(f"  type_token_ratio  = {ttr:.4f}")
        lines.append("")
        lines.extend(render_stanza_package_table(nlp, package))

    write_summary(out_dir / "summary.txt", lines)

    # dictcheck (optional)
    run_dictcheck_if_enabled(
        cfg=cfg,
        out_dir=out_dir,
        group_counts=group_counts,
        script_dir=script_dir,
    )

    print("[Done] Saved to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())