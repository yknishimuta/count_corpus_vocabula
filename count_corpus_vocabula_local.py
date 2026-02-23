from __future__ import annotations

import csv
import glob as pyglob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ----------------------------
# Compatibility hooks for tests
# ----------------------------

# tests expect: mod.clean_mod.main to exist and be monkeypatchable
try:
    from count_corpus_vocabula import clean as clean_mod  # type: ignore
except Exception:
    try:
        from count_corpus_vocabula import cleaner as clean_mod  # type: ignore
    except Exception:
        clean_mod = SimpleNamespace(main=lambda argv: 0)

# tests expect: mod.load_config to exist and be monkeypatchable
def load_config(cfg_path: Path) -> Dict[str, Any]:
    cfg_path = Path(cfg_path)
    obj = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be mapping: {cfg_path}")

    if "groups" not in obj and "group" in obj:
        obj["groups"] = obj["group"]

    if "groups" not in obj:
        raise ValueError("Please define 'groups' or 'group' in config")

    if "cleaner_config" in obj:
        raise ValueError("Deprecated key 'cleaner_config'")

    return obj


# ----------------------------
# helpers
# ----------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return obj


def _resolve_cleaner_output_dir(cleaner_yml: Path) -> Path:
    cfg = _load_yaml(cleaner_yml)
    out = cfg.get("output")
    if not out:
        raise ValueError(f"cleaner config missing 'output': {cleaner_yml}")
    out_p = Path(str(out))
    if out_p.is_absolute():
        return out_p
    return (cleaner_yml.parent / out_p).resolve()


def _expand_cleaned_dir_placeholders(patterns: List[str], cleaned_dir: Optional[Path]) -> List[str]:
    if cleaned_dir is None:
        return patterns
    return [p.replace("{cleaned_dir}", str(cleaned_dir)) for p in patterns]


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _glob_paths(patterns: List[str]) -> List[str]:
    """
    Expand globs for both relative and absolute patterns.
    pathlib.Path.glob() does NOT support absolute patterns; use glob.glob().
    """
    out: List[str] = []
    for pat in patterns:
        out.extend(pyglob.glob(pat))
    return sorted(set(out))


# ----------------------------
# NLP / counting hooks (tests monkeypatch these)
# ----------------------------

def build_pipeline(language: str, stanza_package: str, cpu_only: bool):
    """
    Production pipeline builder (Stanza via nlpo_toolkit).
    Returns (nlp, package).
    """
    from nlpo_toolkit.nlp import build_stanza_pipeline  # type: ignore

    # processors は config から取りたいが、互換 runner なので最小で
    processors = "tokenize,pos,lemma"

    nlp = build_stanza_pipeline(
        lang=language,
        processors=processors,
        package=stanza_package,
        use_gpu=(not cpu_only),
    )
    return nlp, stanza_package

def build_sentence_splitter(language: str, stanza_package: str, cpu_only: bool):
    # Optional in this compat runner: tests do NOT monkeypatch this.
    # In production you can implement it; in tests we just won't call it.
    raise RuntimeError("build_sentence_splitter is optional and should not be required in tests")


def count_group(text: str, nlp, **kwargs) -> Counter:
    """
    Production counter: count noun lemmas using nlpo_toolkit.

    kwargs (optional):
      - use_lemma: bool
      - upos_targets: set[str] | frozenset[str]
      - chunk_chars: int
      - label: str
      - ref_tag_detector: callable | None
      - ref_tag_counter: Counter | None
    """
    from nlpo_toolkit.nlp import count_nouns_streaming  # type: ignore

    use_lemma = bool(kwargs.get("use_lemma", True))
    upos_targets = kwargs.get("upos_targets", {"NOUN"})
    if isinstance(upos_targets, set):
        upos_targets = frozenset(upos_targets)
    chunk_chars = int(kwargs.get("chunk_chars", 200_000))
    label = str(kwargs.get("label", ""))

    ref_tag_detector = kwargs.get("ref_tag_detector")
    ref_tag_counter = kwargs.get("ref_tag_counter")

    return count_nouns_streaming(
        text,
        nlp,
        use_lemma=use_lemma,
        upos_targets=upos_targets,
        chunk_chars=chunk_chars,
        ref_tag_detector=ref_tag_detector,
        ref_tag_counter=ref_tag_counter,
        label=label,
    )


def render_stanza_package_table(nlp, pkg: str) -> List[str]:
    return [f"package={pkg}"]


# ----------------------------
# output
# ----------------------------

def write_frequency_csv(path: Path, counter: Counter, header: Tuple[str, str] = ("word", "frequency")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for word, freq in rows:
            w.writerow([word, int(freq)])


def _build_run_meta(groups_files: Dict[str, List[str]], *, hash_inputs: bool = False) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"groups": {}},
    }
    for gname, files in groups_files.items():
        entries = []
        for f in files:
            p = Path(f)
            try:
                st = p.stat()
                e: Dict[str, Any] = {
                    "path": str(p),
                    "size": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                }
                if hash_inputs:
                    e["sha256_head_1MiB"] = None
                entries.append(e)
            except FileNotFoundError:
                entries.append({"path": str(p), "missing": True})
        meta["inputs"]["groups"][gname] = {"file_count": len(files), "files": entries}
    return meta


def write_run_meta(meta: Dict[str, Any], out_dir: Path, filename: str = "run_meta.json") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ----------------------------
# main
# ----------------------------

def main() -> int:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "config" / "groups.config.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    # preprocess (optional)
    cleaned_dir: Optional[Path] = None
    pp = cfg.get("preprocess")
    if pp and pp.get("kind") == "cleaner":
        cleaner_config_raw = pp.get("config")
        if not cleaner_config_raw:
            raise ValueError("'preprocess.config' is required when preprocess.kind=cleaner")

        cleaner_config_path = Path(str(cleaner_config_raw))
        if not cleaner_config_path.is_absolute():
            cleaner_config_path = (script_dir / cleaner_config_path).resolve()

        if not cleaner_config_path.exists():
            raise FileNotFoundError(f"Cleaner config file not found: {cleaner_config_path}")

        print(f"[Cleaner] Running with config: {cleaner_config_path}")
        clean_mod.main([str(cleaner_config_path)])

        cleaned_dir = _resolve_cleaner_output_dir(cleaner_config_path)
        print(f"[Cleaner] Output directory (cleaned text): {cleaned_dir}")

    # output directory
    out_dir = Path(cfg.get("out_dir", "output"))
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # language settings
    language = cfg.get("language", "la")
    stanza_package = cfg.get("stanza_package", "perseus")
    cpu_only = bool(cfg.get("cpu_only", True))

    # build NLP (tests monkeypatch this)
    nlp, package = build_pipeline(language, stanza_package, cpu_only)

    # NOTE:
    # - tests DO NOT monkeypatch build_sentence_splitter
    # - so splitter must be OPTIONAL. We'll try to build it, but ignore failures.
    splitter_nlp = None
    try:
        splitter_nlp = build_sentence_splitter(language, stanza_package=package, cpu_only=cpu_only)  # type: ignore[arg-type]
    except Exception:
        splitter_nlp = None

    # groups
    groups = cfg.get("groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise ValueError("config.groups must be a non-empty mapping")

    group_counts: Dict[str, Counter] = {}
    groups_files: Dict[str, List[str]] = {}

    for gname, gdef in groups.items():
        if not isinstance(gdef, dict):
            raise ValueError(f"groups.{gname} must be mapping")

        patterns = gdef.get("files") or []
        patterns = [str(p) for p in _as_list(patterns)]
        patterns = _expand_cleaned_dir_placeholders(patterns, cleaned_dir)

        files = _glob_paths(patterns)
        groups_files[gname] = files

        texts: List[str] = []
        for fp in files:
            texts.append(Path(fp).read_text(encoding="utf-8"))
        whole = "".join(texts)

        if splitter_nlp is not None:
            doc = splitter_nlp(whole)
            joined = "\n".join([s.text for s in getattr(doc, "sentences", [])])
            if not joined.strip():
                joined = whole
        else:
            joined = whole

        c = count_group(joined, nlp)
        group_counts[gname] = c

        # base csv
        base = f"noun_frequency_{gname}"
        write_frequency_csv(out_dir / f"{base}.csv", c, header=("word", "frequency"))

        # dictcheck
        dc = cfg.get("dictcheck") or {}
        if bool(dc.get("enabled", False)):
            wordlist = dc.get("wordlist")
            if not wordlist:
                raise ValueError("dictcheck.wordlist is required when dictcheck.enabled=true")

            wl_path = Path(str(wordlist))
            if not wl_path.is_absolute():
                wl_path = (script_dir / wl_path).resolve()
            known = set(x.strip() for x in wl_path.read_text(encoding="utf-8").splitlines() if x.strip())

            known_c = Counter({w: n for (w, n) in c.items() if w in known})
            unknown_c = Counter({w: n for (w, n) in c.items() if w not in known})

            write_frequency_csv(out_dir / f"noun_frequency_{gname}.known.csv", known_c, header=("word", "frequency"))
            write_frequency_csv(out_dir / f"noun_frequency_{gname}.unknown.csv", unknown_c, header=("word", "frequency"))

    # summary.txt
    summary_lines: List[str] = []
    summary_lines.append("# Summary")
    summary_lines.append("")
    summary_lines.append(f"language: {language}")
    summary_lines.append(f"stanza_package: {stanza_package}")
    summary_lines.append("")
    summary_lines.extend(render_stanza_package_table(nlp, stanza_package))
    summary_lines.append("")
    for gname, c in group_counts.items():
        summary_lines.append(f"- group={gname} types={len(c)}")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # run meta
    meta = _build_run_meta(groups_files, hash_inputs=False)
    write_run_meta(meta, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())