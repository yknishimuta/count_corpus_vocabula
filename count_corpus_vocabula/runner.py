from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .io_utils import expand_globs, read_concat
from .normalizer import normalize_text
from .outputs import (
    build_run_meta,
    collect_runtime_environment,
    write_frequency_csv,
    write_run_meta,
)
from .preprocess import expand_cleaned_dir_placeholders, run_preprocess_if_needed
from .ref_tags import load_ref_tag_patterns, strip_and_count_ref_tags


def _resolve_analysis_unit(cfg: Dict[str, Any]) -> tuple[str, bool, tuple[str, str]]:
    """
    Returns:
      - unit: "lemma" | "surface"
      - use_lemma: bool (to pass into count_group_fn)
      - csv_header: (col1, col2)
    """
    unit = str(cfg.get("analysis_unit", "lemma")).strip().lower()
    if unit not in {"lemma", "surface"}:
        raise ValueError("analysis_unit must be 'lemma' or 'surface'")

    use_lemma = (unit == "lemma")

    # default headers
    if unit == "lemma":
        header = ("lemma", "count")
    else:
        header = ("word", "frequency")

    # optional override: csv_header: ["...", "..."]
    hdr = cfg.get("csv_header")
    if hdr is not None:
        if (
            isinstance(hdr, list)
            and len(hdr) == 2
            and all(isinstance(x, str) and x.strip() for x in hdr)
        ):
            header = (hdr[0], hdr[1])
        else:
            raise ValueError("csv_header must be a list[str] of length 2")

    return unit, use_lemma, header

def _format_normalization_kv(norm: dict) -> str:
    if not isinstance(norm, dict) or not norm:
        return "(none)"

    keys_first = ["enabled", "casefold", "uv", "ij", "diacritics"]
    parts: list[str] = []

    for k in keys_first:
        if k in norm:
            parts.append(f"{k}={norm[k]}")

    lig = norm.get("ligatures")
    if isinstance(lig, dict) and lig:
        lig_s = ",".join(f"{a}→{b}" for a, b in sorted(lig.items(), key=lambda x: x[0]))
        parts.append(f"ligatures={lig_s}")

    for k in sorted(norm.keys()):
        if k in keys_first or k == "ligatures":
            continue
        parts.append(f"{k}={norm[k]}")

    return " ".join(parts)

def run(
    *,
    script_dir: Path,
    config_path: Path,
    load_config_fn: Callable[[Path], Dict[str, Any]],
    clean_mod: Any,
    build_pipeline_fn: Callable[[str, str, bool], Tuple[Any, str]],
    build_sentence_splitter_fn: Optional[Callable[..., Any]],
    count_group_fn: Callable[..., Counter],
    render_stanza_package_table_fn: Callable[..., List[str]],
) -> int:
    """
    Core runner. Dependencies are injectable so tests can monkeypatch:
      - load_config_fn
      - clean_mod.main
      - build_pipeline_fn
      - build_sentence_splitter_fn
      - count_group_fn
      - render_stanza_package_table_fn
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config_fn(config_path)

    # preprocess (optional)
    cleaned_dir = run_preprocess_if_needed(cfg=cfg, script_dir=script_dir, clean_mod=clean_mod)

    # output directory
    out_dir = Path(cfg.get("out_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # language settings
    language = cfg.get("language", "la")
    stanza_package = cfg.get("stanza_package") or "perseus"
    cpu_only = bool(cfg.get("cpu_only", True))

    # analysis unit (lemma / surface)
    unit, use_lemma, csv_header = _resolve_analysis_unit(cfg)

    # build NLP
    nlp, package = build_pipeline_fn(language, stanza_package, cpu_only)

    # sentence splitter is optional
    splitter_nlp = None
    if build_sentence_splitter_fn is not None:
        try:
            splitter_nlp = build_sentence_splitter_fn(
                language,
                stanza_package=package,
                cpu_only=cpu_only,
            )
        except Exception:
            splitter_nlp = None

    # ref_tags setting is global (summary/meta needs it)
    ref_cfg = cfg.get("ref_tags") or {}
    ref_enabled = bool(ref_cfg.get("enabled", False))

    filters_cfg = cfg.get("filters") or {}
    min_token_length = int(filters_cfg.get("min_token_length", 0))
    drop_roman_numerals = bool(filters_cfg.get("drop_roman_numerals", False))
    roman_exceptions_file = filters_cfg.get("roman_exceptions_file")
    if roman_exceptions_file:
        roman_exceptions_file = (script_dir / Path(roman_exceptions_file)).resolve()

    # groups
    groups = cfg.get("groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise ValueError("config.groups must be a non-empty mapping")

    group_counts: Dict[str, Counter] = {}
    group_ref_tags: Dict[str, Counter] = {}
    groups_files: Dict[str, List[str]] = {}

    for gname, gdef in groups.items():
        if not isinstance(gdef, dict):
            raise ValueError(f"groups.{gname} must be mapping")

        patterns = gdef.get("files") or []
        if not isinstance(patterns, list):
            raise ValueError(f"groups.{gname}.files must be list[str]")

        patterns = [str(p) for p in patterns]
        patterns = expand_cleaned_dir_placeholders(patterns, cleaned_dir)

        files = expand_globs(patterns)  # List[Path]
        groups_files[gname] = [str(p) for p in files]

        whole = read_concat(files)

        if splitter_nlp is not None:
            doc = splitter_nlp(whole)
            joined = "\n".join([s.text for s in getattr(doc, "sentences", [])])
            if not joined.strip():
                joined = whole
        else:
            joined = whole

        # normalization (config-driven)
        joined = normalize_text(joined, cfg)

        # ref_tags stripping/counting
        ref_counter = Counter()
        if ref_enabled:
            ref_file = ref_cfg.get("patterns") or ref_cfg.get("ref_tags_file")
            if not ref_file:
                raise ValueError(
                    "ref_tags.patterns (or ref_tags.ref_tags_file) is required when ref_tags.enabled=true"
                )

            ref_path = Path(str(ref_file))
            if not ref_path.is_absolute():
                ref_path = (script_dir / ref_path).resolve()

            ref_patterns = load_ref_tag_patterns(ref_path)
            joined, ref_counter = strip_and_count_ref_tags(joined, ref_patterns)
            group_ref_tags[gname] = ref_counter

            # ref_tags csv (per group)
            write_frequency_csv(
                out_dir / f"ref_tags_{gname}.csv",
                ref_counter,
                header=("tag", "count"),
            )
        
        # ---- trace (optional) ----
        trace_cfg = cfg.get("trace") or {}
        trace_kwargs: Dict[str, Any] = {}

        if bool(trace_cfg.get("enabled", False)):
            trace_path = Path(str(trace_cfg.get("path", out_dir / "trace.tsv")))
            if not trace_path.is_absolute():
                trace_path = (script_dir / trace_path).resolve()

            trace_kwargs = {
                "trace_tsv": trace_path,
                "trace_max_rows": int(trace_cfg.get("max_rows", 0)),
                "trace_only_keys": set(trace_cfg.get("only_keys", []) or []),
                "trace_write_truncation_marker": bool(
                    trace_cfg.get("write_truncation_marker", True)
                ),
            }

        c = count_group_fn(joined, nlp, use_lemma=use_lemma, min_token_length=min_token_length, drop_roman_numerals=drop_roman_numerals, roman_exceptions_file=roman_exceptions_file, **trace_kwargs)
        group_counts[gname] = c

        # base csv
        base = f"noun_frequency_{gname}"
        write_frequency_csv(out_dir / f"{base}.csv", c, header=csv_header)

        # dictcheck
        dc = cfg.get("dictcheck") or {}
        wordlist = dc.get("wordlist")

        if bool(dc.get("enabled", False)) and not wordlist:
            raise ValueError(
                f"dictcheck.wordlist is required when dictcheck.enabled=true (analysis_unit={unit})"
            )

        if bool(dc.get("enabled", False)):
            wl_path = Path(str(wordlist))
            if not wl_path.is_absolute():
                wl_path = (script_dir / wl_path).resolve()

            known = set(
                x.strip()
                for x in wl_path.read_text(encoding="utf-8").splitlines()
                if x.strip()
            )

            known_c = Counter({w: n for (w, n) in c.items() if w in known})
            unknown_c = Counter({w: n for (w, n) in c.items() if w not in known})

            write_frequency_csv(
                out_dir / f"noun_frequency_{gname}.known.csv",
                known_c,
                header=csv_header,
            )
            write_frequency_csv(
                out_dir / f"noun_frequency_{gname}.unknown.csv",
                unknown_c,
                header=csv_header,
            )

    # ---- summary.txt ----
    summary_lines: List[str] = []
    summary_lines.append("# Summary")
    summary_lines.append("")
    summary_lines.append(f"language: {language}")
    summary_lines.append(f"stanza_package: {stanza_package}")
    summary_lines.append(f"analysis_unit: {unit}")

    # normalization policy (human-readable, stable)
    norm = cfg.get("normalization", {}) or {}
    summary_lines.append(f"normalization: {_format_normalization_kv(norm)}")

    summary_lines.append("")
    summary_lines.extend(render_stanza_package_table_fn(nlp, stanza_package))
    summary_lines.append("")

    if ref_enabled:
        for gn, rc in group_ref_tags.items():
            summary_lines.append(
                f"- group={gn} ref_tag_types={len(rc)} ref_tag_tokens={sum(rc.values())}"
            )

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # ---- run_meta.json ----
    meta = build_run_meta(
        groups_files=groups_files,
        hash_inputs=False,
    )

    meta["analysis_unit"] = unit
    meta["environment"] = collect_runtime_environment(script_dir)

    norm_canon = json.dumps(norm, ensure_ascii=False, sort_keys=True)
    meta["normalization"] = norm
    meta["normalization_hash_sha256"] = hashlib.sha256(norm_canon.encode("utf-8")).hexdigest()

    write_run_meta(meta, out_dir)

    return 0