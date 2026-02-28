from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .io_utils import expand_globs, read_concat
from .outputs import write_frequency_csv, collect_runtime_environment, build_run_meta, write_run_meta
from .preprocess import expand_cleaned_dir_placeholders, run_preprocess_if_needed

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

    # groups
    groups = cfg.get("groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise ValueError("config.groups must be a non-empty mapping")

    group_counts: Dict[str, Counter] = {}

    for gname, gdef in groups.items():
        if not isinstance(gdef, dict):
            raise ValueError(f"groups.{gname} must be mapping")

        patterns = gdef.get("files") or []
        if not isinstance(patterns, list):
            raise ValueError(f"groups.{gname}.files must be list[str]")

        patterns = [str(p) for p in patterns]
        patterns = expand_cleaned_dir_placeholders(patterns, cleaned_dir)

        files = expand_globs(patterns)  # List[Path]
        whole = read_concat(files)

        if splitter_nlp is not None:
            doc = splitter_nlp(whole)
            joined = "\n".join([s.text for s in getattr(doc, "sentences", [])])
            if not joined.strip():
                joined = whole
        else:
            joined = whole

        # NOTE: pass use_lemma so we can switch surface/lemma without changing counter implementation
        c = count_group_fn(joined, nlp, use_lemma=use_lemma)
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
            wordlist = dc.get("wordlist")
            if not wordlist:
                raise ValueError(
                    f"dictcheck.wordlist is required when dictcheck.enabled=true (analysis_unit={unit})"
                )

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

    # summary.txt
    summary_lines: List[str] = []
    summary_lines.append("# Summary")
    summary_lines.append("")
    summary_lines.append(f"language: {language}")
    summary_lines.append(f"stanza_package: {stanza_package}")
    summary_lines.append(f"analysis_unit: {unit}")
    summary_lines.append("")
    summary_lines.extend(render_stanza_package_table_fn(nlp, stanza_package))
    summary_lines.append("")
    for gname, c in group_counts.items():
        summary_lines.append(f"- group={gname} types={len(c)}")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    meta = build_run_meta(
        groups_files={g: [] for g in group_counts.keys()},
        hash_inputs=False,
    )

    meta["analysis_unit"] = unit
    meta["environment"] = collect_runtime_environment(script_dir)

    write_run_meta(meta, out_dir)

    return 0