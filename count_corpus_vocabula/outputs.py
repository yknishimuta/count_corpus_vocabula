from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def write_frequency_csv(
    path: Path,
    freq: Mapping[str, int] | Counter[str],
    *,
    header: Sequence[str] = ("lemma", "count"),
) -> None:
    """
    Write frequency table CSV.

    Sorting:
      - frequency desc
      - key asc (stable for ties)

    Args:
      path: output csv path
      freq: mapping from token/lemma -> count
      header: two column names
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not (isinstance(header, (list, tuple)) and len(header) == 2):
        raise ValueError("header must be a sequence of length 2")

    items = sorted(freq.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([header[0], header[1]])
        for k, v in items:
            w.writerow([k, int(v)])


def _safe_check_output(cmd: Sequence[str], *, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(list(cmd), cwd=str(cwd) if cwd else None)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def collect_runtime_environment(script_dir: Path) -> Dict[str, Any]:
    """
    Collect runtime environment info for run metadata.
    Must not raise even if git is unavailable.

    Returns keys:
      - python_version
      - platform
      - executable
      - script_dir
      - git_commit (or None)
      - git_status (or None)
    """
    script_dir = Path(script_dir)

    env: Dict[str, Any] = {}
    env["python_version"] = sys.version
    env["platform"] = sys.platform
    env["executable"] = sys.executable
    env["script_dir"] = str(script_dir)

    # Best-effort git info (non-fatal)
    env["git_commit"] = _safe_check_output(["git", "rev-parse", "HEAD"], cwd=script_dir)
    env["git_status"] = _safe_check_output(["git", "status", "--porcelain"], cwd=script_dir)

    return env


def build_run_meta(
    *,
    groups_files: Dict[str, Iterable[Path] | Iterable[str]],
    hash_inputs: bool = False,
) -> Dict[str, Any]:
    """
    Build a minimal run metadata dict.

    Args:
      groups_files: mapping group -> iterable of paths or strings
      hash_inputs: reserved for future; kept for API compatibility
    """
    normalized: Dict[str, list[str]] = {}
    for g, files in (groups_files or {}).items():
        if files is None:
            normalized[g] = []
        else:
            normalized[g] = [str(p) for p in files]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_inputs": bool(hash_inputs),
        "groups_files": normalized,
    }


def write_run_meta(meta: Dict[str, Any], out_dir: Path) -> Path:
    """
    Write run metadata JSON to <out_dir>/run_meta.json

    Returns:
      Path to written file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path