#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, Mapping, List, Optional, Any
from collections import Counter
from datetime import datetime, timezone
import sys, re, json, hashlib, platform, subprocess

from count_corpus_vocabula.config import load_config
from count_corpus_vocabula.io_utils import expand_globs, read_concat, save_counter_csv, write_summary
from count_corpus_vocabula.nlp_utils import build_pipeline
from count_corpus_vocabula.counters import count_group, load_exclude_list
from count_corpus_vocabula.compose import compose_all
from count_corpus_vocabula.text_prep import one_sentence_per_line
from count_corpus_vocabula.ref_tags import load_ref_tag_set, build_ref_tag_detector

from nlpo_toolkit.nlp import render_stanza_package_table, build_sentence_splitter
from nlpo_toolkit.latin.cleaners import run_clean_config as clean_mod
from nlpo_toolkit.latin.cleaners.config_loader import load_clean_config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

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

def _safe_stem(s: str, default: str = "group") -> str:
    s = (s or "").strip()
    if not s:
        return default
    s = _SAFE_NAME_RE.sub("_", s)
    s = s.strip("._-")
    return s or default

def _get_trace_kwargs(cfg: dict, out_dir: Path, label: str) -> dict:
    tcfg = cfg.get("trace") or {}
    enabled = bool(tcfg.get("enabled", False))
    if not enabled:
        return {}

    trace_dir = Path(tcfg.get("dir", "trace"))
    if not trace_dir.is_absolute():
        trace_dir = out_dir / trace_dir

    stem = _safe_stem(label, default="group")
    tsv_path = trace_dir / f"{stem}.tsv"

    max_rows = int(tcfg.get("max_rows", 0))
    return {"trace_tsv": tsv_path, "trace_max_rows": max_rows}

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

def _sha256_head(path: Path, head_bytes: int = 1_048_576) -> str:
    """Fast-ish fingerprint: sha256 of first N bytes (default 1MiB)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()


def _safe_run(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def collect_run_meta(
    *,
    out_dir: Path,
    cfg: Dict[str, Any],
    groups_files: Dict[str, List[str]],
    stanza_nlp=None,
    splitter_nlp=None,
    hash_inputs: bool = False,
) -> Dict[str, Any]:
    """Collect reproducibility metadata for the current run."""
    out_dir = out_dir.resolve()
    repo_root = _safe_run(["git", "rev-parse", "--show-toplevel"])
    git_commit = _safe_run(["git", "rev-parse", "HEAD"], cwd=Path(repo_root) if repo_root else None)
    git_dirty = _safe_run(["git", "status", "--porcelain"], cwd=Path(repo_root) if repo_root else None)
    git_branch = _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=Path(repo_root) if repo_root else None)

    py_ver = platform.python_version()
    versions = {"python": py_ver}

    try:
        import stanza
        versions["stanza"] = getattr(stanza, "__version__", None)
    except Exception:
        versions["stanza"] = None

    try:
        import torch
        versions["torch"] = getattr(torch, "__version__", None)
    except Exception:
        versions["torch"] = None

    try:
        import nlpo_toolkit
        versions["nlpo_toolkit"] = getattr(nlpo_toolkit, "__version__", None)
    except Exception:
        versions["nlpo_toolkit"] = None

    # Pipeline config snapshots
    def _pipeline_cfg(nlp_obj):
        cfg_ = getattr(nlp_obj, "config", None)
        return cfg_ if isinstance(cfg_, dict) else None

    meta: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": {
            "repo_root": repo_root,
            "branch": git_branch,
            "commit": git_commit,
            "dirty": bool(git_dirty),
        },
        "versions": versions,
        "pipelines": {
            "main": _pipeline_cfg(stanza_nlp),
            "splitter": _pipeline_cfg(splitter_nlp),
        },
        "config": cfg,
        "inputs": {
            "hash_inputs": hash_inputs,
            "groups": {},
        },
    }

    # Input file listing (with size/mtime; optional hash)
    for gname, files in groups_files.items():
        entries = []
        for f in files:
            p = Path(f)
            try:
                st = p.stat()
                e = {
                    "path": str(p),
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
                if hash_inputs:
                    e["sha256_head_1MiB"] = _sha256_head(p)
                entries.append(e)
            except FileNotFoundError:
                entries.append({"path": str(p), "missing": True})
        meta["inputs"]["groups"][gname] = {
            "file_count": len(files),
            "files": entries,
        }

    return meta


def write_run_meta(meta: Dict[str, Any], out_dir: Path, filename: str = "run_meta.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


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
    # ref_tags (optional)
    # ----------------------------
    ref_cfg = cfg.get("ref_tags") or {}
    ref_detector = None
    if ref_cfg.get("enabled"):
        rt_path = Path(ref_cfg["patterns"])
        if not rt_path.is_absolute():
            rt_path = (script_dir / rt_path).resolve()
        if not rt_path.exists():
            raise FileNotFoundError(f"ref_tags patterns file not found: {rt_path}")
        ref_tag_set = load_ref_tag_set(rt_path)
        ref_detector = build_ref_tag_detector(ref_tag_set)
        print(f"[RefTags] Loaded {len(ref_tag_set)} patterns from {rt_path}")

    # process groups (always present)
    exclude = load_exclude_list("config/exclude_lemmas.txt")
    groups_files: dict[str, list[str]] = {}

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

        groups_files[gname] = [str(f) for f in files]

        text = read_concat(files)
        text = one_sentence_per_line(text, splitter_nlp)
        print(f"[Processing] {gname}: {len(text):,} chars / {len(files)} files")

        trace_kwargs = _get_trace_kwargs(cfg, Path(out_dir), label=gname)

        ref_counter: Counter | None = Counter() if ref_detector else None
        total = count_group(
            text, nlp, label=gname, exclude_lemmas=exclude,
            trace_kwargs=trace_kwargs,
            ref_tag_detector=ref_detector,
            ref_tag_counter=ref_counter,
        )

        group_counts[gname] = total
        safe = _safe_stem(gname)
        save_counter_csv(out_dir / f"noun_frequency_{safe}.csv", total)
        if ref_counter:
            save_counter_csv(out_dir / f"ref_tag_frequency_{safe}.csv", ref_counter)
            print(f"[RefTags] {gname}: {sum(ref_counter.values())} ref_tag tokens / {len(ref_counter)} types")

    meta = collect_run_meta(
        out_dir=Path(out_dir),
        cfg=cfg,
        groups_files=groups_files,
        stanza_nlp=nlp,
        splitter_nlp=splitter_nlp,
        hash_inputs=False,
    )
    write_run_meta(meta, Path(out_dir))

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