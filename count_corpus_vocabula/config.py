from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
import yaml

class GroupDef(TypedDict):
    files: List[str]

class Config(TypedDict, total=False):
    groups: Dict[str, Dict[str, Any]]

    cleaner_config: str
    out_dir: str
    vocab_path: Optional[str]
    language: str
    stanza_package: Optional[str]
    cpu_only: bool

def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("Config file must be YAML (.yml / .yaml)")

    text = path.read_text(encoding="utf-8")
    config_data = yaml.safe_load(text) or {}

    # validation:
    # Allow either:
    #  (A) legacy mode: groups
    #  (B) pipeline mode: cleaner_config
    has_groups = "groups" in config_data
    has_cleaner = bool(config_data.get("cleaner_config"))

    if not has_groups and not has_cleaner:
        raise ValueError("Config must have either 'groups' mapping or 'cleaner_config'.")

    if has_groups:
        if not isinstance(config_data["groups"], dict):
            raise ValueError("Config 'groups' must be a mapping.")
        for k, v in config_data["groups"].items():
            if not isinstance(v, dict) or "files" not in v:
                raise ValueError(f"Group '{k}' must have 'files' list.")
            if not isinstance(v["files"], list) or not all(isinstance(x, str) for x in v["files"]):
                raise ValueError(f"Group '{k}' must have 'files' as list[str].")
    else:
        # pipeline mode: we only require cleaner_config here
        if not isinstance(config_data["cleaner_config"], str):
            raise ValueError("'cleaner_config' must be a string path.")

    return config_data
