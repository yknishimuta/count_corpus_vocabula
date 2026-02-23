from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from collections import Counter


CACHE_VERSION = 4  # bump when payload schema/behavior changes


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return _sha256_bytes(s.encode("utf-8", errors="strict"))


def hash_file_content(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute sha256 of file content (streaming).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic write to avoid corrupt cache on crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmp = Path(tf.name)
    os.replace(str(tmp), str(path))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Manifest (optional)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestEntry:
    size: int
    mtime_ns: int
    content_hash: str


class ContentHashManifest:
    """
    A small JSON manifest to avoid re-hashing large files every run.

    Supports absolute-path keys (default) and relative-path keys (project_root-based).

    Manifest JSON format:
      {
        "__meta__": {"version": 1, "key_mode": "...", "project_root": "..."},
        "<path_key>": {"size": ..., "mtime_ns": ..., "content_hash": "..."},
        ...
      }
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        key_mode: str = "absolute",  # "absolute" or "relative"
        project_root: Optional[Path] = None,
    ):
        self.manifest_path = manifest_path
        self.key_mode = (key_mode or "absolute").strip().lower()
        if self.key_mode not in ("absolute", "relative"):
            raise ValueError("manifest key_mode must be 'absolute' or 'relative'")
        self.project_root = project_root.resolve() if project_root is not None else None
        if self.key_mode == "relative" and self.project_root is None:
            raise ValueError("manifest key_mode='relative' requires project_root")
        self._data: dict[str, dict[str, Any]] = {}

    def _meta(self) -> dict[str, Any]:
        return {
            "version": 1,
            "key_mode": self.key_mode,
            "project_root": str(self.project_root) if self.project_root is not None else None,
        }

    def _path_key(self, path: Path) -> str:
        p = path.resolve()
        if self.key_mode == "absolute":
            return str(p)
        assert self.project_root is not None
        try:
            return str(p.relative_to(self.project_root))
        except Exception:
            # fallback if outside project_root
            return str(p)

    def load(self) -> None:
        if not self.manifest_path.exists():
            self._data = {"__meta__": self._meta()}
            return
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                self._data = {"__meta__": self._meta()}
                return
            raw["__meta__"] = self._meta()
            self._data = raw
        except Exception:
            self._data = {"__meta__": self._meta()}

    def save(self) -> None:
        atomic_write_text(
            self.manifest_path,
            json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True),
        )

    def get(self, path: Path) -> Optional[ManifestEntry]:
        key = self._path_key(path)
        raw = self._data.get(key)
        if not isinstance(raw, dict):
            return None
        try:
            return ManifestEntry(
                size=int(raw["size"]),
                mtime_ns=int(raw["mtime_ns"]),
                content_hash=str(raw["content_hash"]),
            )
        except Exception:
            return None

    def put(self, path: Path, entry: ManifestEntry) -> None:
        key = self._path_key(path)
        self._data[key] = {
            "size": int(entry.size),
            "mtime_ns": int(entry.mtime_ns),
            "content_hash": str(entry.content_hash),
        }


# ---------------------------------------------------------------------------
# Cache payload (JSON for portability)
# ---------------------------------------------------------------------------

@dataclass
class LemmaCachePayload:
    """
    Payload stored per file.

    - lemmas: Counter[str]
    - ref_tags: Counter[str]
    """
    lemmas: Counter[str]
    ref_tags: Counter[str]

    def to_json_obj(self) -> dict[str, Any]:
        # store as list of pairs (more compact than huge dict with sorted_keys)
        return {
            "lemmas": [[k, int(v)] for k, v in self.lemmas.items()],
            "ref_tags": [[k, int(v)] for k, v in self.ref_tags.items()],
        }

    @classmethod
    def from_json_obj(cls, d: Mapping[str, Any]) -> "LemmaCachePayload":
        lemmas = Counter()
        for row in (d.get("lemmas") or []):
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            lemmas[str(row[0])] += _safe_int(row[1], 0)

        ref_tags = Counter()
        for row in (d.get("ref_tags") or []):
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            ref_tags[str(row[0])] += _safe_int(row[1], 0)

        return cls(lemmas=lemmas, ref_tags=ref_tags)


def _make_cache_key(content_hash: str, config_hash: str) -> str:
    return _sha256_text(f"{content_hash}|{config_hash}|v{CACHE_VERSION}")


def _cache_file_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "objects" / cache_key[:2] / f"{cache_key}.json"


def _lock_file_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "locks" / cache_key[:2] / f"{cache_key}.lock"


# ---------------------------------------------------------------------------
# File lock (simple lockfile)
# ---------------------------------------------------------------------------

class CacheLockTimeout(RuntimeError):
    pass


def _acquire_lock(lock_path: Path, *, timeout_sec: float = 300.0, poll_sec: float = 0.1) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pid = os.getpid()

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={pid}\n".encode("utf-8"))
            finally:
                os.close(fd)
            return
        except FileExistsError:
            if (time.time() - start) >= timeout_sec:
                raise CacheLockTimeout(f"Timeout acquiring lock: {lock_path}")
            time.sleep(poll_sec)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------

def build_config_hash(
    *,
    stanza_model: str,
    lang: str,
    processors: str,
    use_lemma: bool,
    upos_targets: set[str],
    ref_tags_file: Optional[Path] = None,
    include_ref_tags_in_config_hash: bool = True,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Build stable settings hash.

    Put anything that changes tokenization/lemmatization/count results into here.
    """
    d: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "stanza_model": stanza_model,
        "lang": lang,
        "processors": processors,
        "use_lemma": bool(use_lemma),
        "upos_targets": sorted({x.strip().upper() for x in upos_targets if x and str(x).strip()}),
    }

    if include_ref_tags_in_config_hash:
        if ref_tags_file is None:
            d["ref_tags"] = None
        else:
            d["ref_tags_file"] = str(ref_tags_file.resolve())
            d["ref_tags_hash"] = hash_file_content(ref_tags_file) if ref_tags_file.exists() else "MISSING"

    if extra:
        # Ensure stable keys
        d["extra"] = {str(k): extra[k] for k in sorted(extra.keys(), key=lambda x: str(x))}

    s = json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_compute_cached(
    *,
    path: Path,
    cache_dir: Path,
    config_hash: str,
    compute_fn: Callable[[], LemmaCachePayload],
    use_manifest: bool = True,
    manifest_path: Optional[Path] = None,
    manifest_key_mode: str = "absolute",  # "absolute" or "relative"
    manifest_project_root: Optional[Path] = None,
    verbose: bool = False,
    lock_timeout_sec: float = 300.0,
) -> tuple[LemmaCachePayload, bool]:
    """
    Return (payload, cache_hit).

    - Cache key: sha256( content_hash + config_hash + CACHE_VERSION )
    - Concurrency safe: lock only on miss compute/write (double-check after lock)
    - Payload: JSON (portable)
    """
    path = path.resolve()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest: Optional[ContentHashManifest] = None
    if use_manifest:
        mp = manifest_path or (cache_dir / "manifest.json")
        manifest = ContentHashManifest(
            mp,
            key_mode=manifest_key_mode,
            project_root=manifest_project_root,
        )
        manifest.load()

    st = path.stat()

    # content hash (with manifest optimization)
    if manifest is not None:
        ent = manifest.get(path)
        if ent and ent.size == st.st_size and ent.mtime_ns == st.st_mtime_ns:
            content_hash = ent.content_hash
            if verbose:
                print(f"[CACHE] manifest hit: {path} -> {content_hash[:12]}…")
        else:
            content_hash = hash_file_content(path)
            manifest.put(path, ManifestEntry(size=st.st_size, mtime_ns=st.st_mtime_ns, content_hash=content_hash))
            manifest.save()
            if verbose:
                print(f"[CACHE] manifest miss: {path} -> computed {content_hash[:12]}…")
    else:
        content_hash = hash_file_content(path)

    cache_key = _make_cache_key(content_hash, config_hash)
    cpath = _cache_file_path(cache_dir, cache_key)

    # Fast path: load without lock
    if cpath.exists():
        try:
            payload = _load_payload_json(cpath)
            if verbose:
                print(f"[CACHE] hit: {path.name} -> {cpath.name}")
            return payload, True
        except Exception as e:
            if verbose:
                print(f"[CACHE] broken cache ignored: {cpath} ({e})")

    # Miss: lock compute/write
    lock_path = _lock_file_path(cache_dir, cache_key)
    _acquire_lock(lock_path, timeout_sec=lock_timeout_sec)
    try:
        # double-check after lock
        if cpath.exists():
            try:
                payload = _load_payload_json(cpath)
                if verbose:
                    print(f"[CACHE] hit-after-lock: {path.name} -> {cpath.name}")
                return payload, True
            except Exception as e:
                if verbose:
                    print(f"[CACHE] broken cache after lock ignored: {cpath} ({e})")

        payload = compute_fn()
        _save_payload_json(cpath, payload)
        if verbose:
            print(f"[CACHE] miss: computed -> {cpath.name}")
        return payload, False
    finally:
        _release_lock(lock_path)


# ---------------------------------------------------------------------------
# Payload I/O (JSON)
# ---------------------------------------------------------------------------

def _save_payload_json(path: Path, payload: LemmaCachePayload) -> None:
    obj = {
        "format": "count_corpus_vocabula.lemma_cache",
        "cache_version": CACHE_VERSION,
        "payload": payload.to_json_obj(),
    }
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_payload_json(path: Path) -> LemmaCachePayload:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("cache file is not a JSON object")
    ver = obj.get("cache_version")
    if ver != CACHE_VERSION:
        raise ValueError(f"cache version mismatch: {ver} != {CACHE_VERSION}")
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload missing")
    return LemmaCachePayload.from_json_obj(payload)


# ---------------------------------------------------------------------------
# Prune / cleanup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PruneReport:
    removed_objects: int
    removed_locks: int
    removed_empty_dirs: int
    bytes_freed: int


def prune_cache(
    cache_dir: Path,
    *,
    keep_days: int = 30,
    keep_files: int = 50_000,
    lock_ttl_sec: int = 3600,
    verbose: bool = False,
) -> PruneReport:
    """
    Best-effort cache cleanup.

    - objects: keep at most keep_files newest (mtime), and delete anything older than keep_days
    - locks: delete lockfiles older than lock_ttl_sec (stale locks)
    - remove empty subdirs under objects/locks (tidy)

    Note:
      This uses file mtime as a proxy for recency. It is simple and robust.
    """
    cache_dir = cache_dir.resolve()
    objects_dir = cache_dir / "objects"
    locks_dir = cache_dir / "locks"

    now = time.time()
    cutoff_ts = now - (keep_days * 86400)

    removed_objects = 0
    removed_locks = 0
    removed_empty_dirs = 0
    bytes_freed = 0

    # --- locks: remove stale ---
    if locks_dir.exists():
        for p in locks_dir.rglob("*.lock"):
            try:
                st = p.stat()
                if (now - st.st_mtime) > lock_ttl_sec:
                    bytes_freed += st.st_size
                    p.unlink(missing_ok=True)
                    removed_locks += 1
                    if verbose:
                        print(f"[PRUNE] removed stale lock: {p}")
            except FileNotFoundError:
                continue
            except OSError:
                continue

    # --- objects: prune by age + max files ---
    object_files: list[Path] = []
    if objects_dir.exists():
        for p in objects_dir.rglob("*.json"):
            if p.is_file():
                object_files.append(p)

    # sort newest first by mtime
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    object_files.sort(key=_mtime, reverse=True)

    # keep first keep_files always (even if old), then apply age cutoff to the rest
    survivors = set(object_files[: max(0, int(keep_files))])

    for p in object_files:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        except OSError:
            continue

        if p in survivors:
            continue

        # remove if older than keep_days
        if st.st_mtime < cutoff_ts:
            try:
                bytes_freed += st.st_size
                p.unlink(missing_ok=True)
                removed_objects += 1
                if verbose:
                    print(f"[PRUNE] removed old object: {p}")
            except OSError:
                pass

    # --- tidy empty dirs (bottom-up) ---
    for base in (objects_dir, locks_dir):
        if not base.exists():
            continue
        # walk deepest-first
        dirs = sorted([d for d in base.rglob("*") if d.is_dir()], key=lambda d: len(str(d)), reverse=True)
        for d in dirs:
            try:
                if not any(d.iterdir()):
                    d.rmdir()
                    removed_empty_dirs += 1
            except OSError:
                pass

    return PruneReport(
        removed_objects=removed_objects,
        removed_locks=removed_locks,
        removed_empty_dirs=removed_empty_dirs,
        bytes_freed=bytes_freed,
    )