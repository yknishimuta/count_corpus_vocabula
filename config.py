from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
import yaml

class GroupDef(TypedDict):
    files: List[str]

class Config(TypedDict, total=False):
    groups: Dict[str, GroupDef]
    out_dir: str
    vocab_path: str
    stanza_pkg: Optional[str]
    cpu_only: bool

def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("Config file must be YAML (.yml / .yaml)")

    text = path.read_text(encoding="utf-8")
    config_data = yaml.safe_load(text) or {}

    # validation
    if "groups" not in config_data or not isinstance(config_data["groups"], dict):
        raise ValueError("Config must have 'groups' mapping.")
    for k, v in config_data["groups"].items():
        if not isinstance(v, dict) or "files" not in v:
            raise ValueError(f"Group '{k}' must have 'files' list.")
    return config_data
