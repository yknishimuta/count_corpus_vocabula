# count_corpus_vocabula/preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return obj


def resolve_cleaner_output_dir(cleaner_yml: Path) -> Path:
    cfg = _load_yaml(cleaner_yml)
    out = cfg.get("output")
    if not out:
        raise ValueError(f"cleaner config missing 'output': {cleaner_yml}")
    out_p = Path(str(out))
    if out_p.is_absolute():
        return out_p
    return (cleaner_yml.parent / out_p).resolve()


def expand_cleaned_dir_placeholders(patterns: list[str], cleaned_dir: Optional[Path]) -> list[str]:
    if cleaned_dir is None:
        return patterns
    return [p.replace("{cleaned_dir}", str(cleaned_dir)) for p in patterns]


def run_preprocess_if_needed(
    *,
    cfg: Dict[str, Any],
    script_dir: Path,
    clean_mod: Any,
) -> Optional[Path]:
    """
    If cfg.preprocess.kind == 'cleaner', run clean_mod.main([...]) and return cleaned_dir.
    Otherwise return None.
    """
    cleaned_dir: Optional[Path] = None
    pp = cfg.get("preprocess")
    if pp and isinstance(pp, dict) and pp.get("kind") == "cleaner":
        cleaner_config_raw = pp.get("config")
        if not cleaner_config_raw:
            raise ValueError("'preprocess.config' is required when preprocess.kind=cleaner")

        cleaner_config_path = Path(str(cleaner_config_raw))
        if not cleaner_config_path.is_absolute():
            cleaner_config_path = (script_dir / cleaner_config_path).resolve()

        if not cleaner_config_path.exists():
            raise FileNotFoundError(f"Cleaner config file not found: {cleaner_config_path}")

        clean_mod.main([str(cleaner_config_path)])
        cleaned_dir = resolve_cleaner_output_dir(cleaner_config_path)
    return cleaned_dir