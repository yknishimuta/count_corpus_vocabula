from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path

import pytest

import count_corpus_vocabula.lemma_cache as lc


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _touch_mtime(p: Path, *, seconds_ago: int) -> None:
    """
    Set file mtime/atime to (now - seconds_ago).
    """
    ts = time.time() - seconds_ago
    os.utime(p, (ts, ts))


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    # Arrange
    src = tmp_path / "input.txt"
    _write_text(src, "Puella rosam amat.\n")

    cache_dir = tmp_path / ".lemma_cache"
    calls = {"n": 0}

    payload0 = lc.LemmaCachePayload(
        lemmas=Counter({"puella": 1, "rosa": 1}),
        ref_tags=Counter({"ref:a": 2}),
    )

    def compute():
        calls["n"] += 1
        return payload0

    # Any stable config hash is fine for this test
    config_hash = lc.build_config_hash(
        stanza_model="perseus",
        lang="la",
        processors="tokenize,pos,lemma",
        use_lemma=True,
        upos_targets={"NOUN"},
        ref_tags_file=None,
        include_ref_tags_in_config_hash=True,
        extra={"stanza_version": "x", "nlpo_toolkit_version": "y"},
    )

    # Act: 1st run => miss
    got1, hit1 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_hash,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=cache_dir / "manifest.json",
        manifest_key_mode="absolute",
        manifest_project_root=None,
        verbose=False,
        lock_timeout_sec=5.0,
    )

    # Act: 2nd run => hit
    got2, hit2 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_hash,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=cache_dir / "manifest.json",
        manifest_key_mode="absolute",
        manifest_project_root=None,
        verbose=False,
        lock_timeout_sec=5.0,
    )

    # Assert
    assert hit1 is False
    assert hit2 is True
    assert calls["n"] == 1

    assert got1.lemmas == payload0.lemmas
    assert got1.ref_tags == payload0.ref_tags
    assert got2.lemmas == payload0.lemmas
    assert got2.ref_tags == payload0.ref_tags

    # cache object should exist somewhere under objects/
    assert (cache_dir / "objects").exists()
    assert any(p.suffix == ".json" for p in (cache_dir / "objects").rglob("*.json"))


def test_cache_invalidated_by_config_hash(tmp_path: Path) -> None:
    # Arrange
    src = tmp_path / "input.txt"
    _write_text(src, "A B C\n")

    cache_dir = tmp_path / ".lemma_cache"
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return lc.LemmaCachePayload(lemmas=Counter({"a": 1}), ref_tags=Counter())

    # two different config hashes (extra differs)
    config_a = lc.build_config_hash(
        stanza_model="perseus",
        lang="la",
        processors="tokenize,pos,lemma",
        use_lemma=True,
        upos_targets={"NOUN"},
        ref_tags_file=None,
        include_ref_tags_in_config_hash=True,
        extra={"stanza_version": "1", "nlpo_toolkit_version": "1"},
    )
    config_b = lc.build_config_hash(
        stanza_model="perseus",
        lang="la",
        processors="tokenize,pos,lemma",
        use_lemma=True,
        upos_targets={"NOUN"},
        ref_tags_file=None,
        include_ref_tags_in_config_hash=True,
        extra={"stanza_version": "2", "nlpo_toolkit_version": "1"},
    )
    assert config_a != config_b

    # Act
    _p1, hit1 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_a,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=cache_dir / "manifest.json",
        verbose=False,
        lock_timeout_sec=5.0,
    )
    _p2, hit2 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_b,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=cache_dir / "manifest.json",
        verbose=False,
        lock_timeout_sec=5.0,
    )

    # Assert: both are misses because config changed => different cache key
    assert hit1 is False
    assert hit2 is False
    assert calls["n"] == 2


def test_manifest_avoids_rehash_when_unchanged(tmp_path: Path, monkeypatch) -> None:
    # This test ensures that on 2nd run (unchanged file),
    # content hash comes from manifest and does NOT call hash_file_content again.

    src = tmp_path / "big.txt"
    _write_text(src, "x" * 10_000 + "\n")

    cache_dir = tmp_path / ".lemma_cache"
    manifest_path = cache_dir / "manifest.json"

    calls = {"hash": 0, "compute": 0}

    real_hash_file_content = lc.hash_file_content

    def wrapped_hash_file_content(path: Path, chunk_size: int = 1024 * 1024) -> str:
        calls["hash"] += 1
        return real_hash_file_content(path, chunk_size=chunk_size)

    monkeypatch.setattr(lc, "hash_file_content", wrapped_hash_file_content)

    def compute():
        calls["compute"] += 1
        return lc.LemmaCachePayload(lemmas=Counter({"x": 1}), ref_tags=Counter())

    config_hash = lc.build_config_hash(
        stanza_model="perseus",
        lang="la",
        processors="tokenize,pos,lemma",
        use_lemma=True,
        upos_targets={"NOUN"},
        ref_tags_file=None,
        include_ref_tags_in_config_hash=True,
        extra={"stanza_version": "1", "nlpo_toolkit_version": "1"},
    )

    # 1st run: must hash file content at least once
    _p1, hit1 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_hash,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=manifest_path,
        verbose=False,
        lock_timeout_sec=5.0,
    )
    assert hit1 is False
    assert calls["hash"] >= 1
    assert manifest_path.exists()

    # Now make hashing fail if called again — to prove manifest hit avoids rehash
    def fail_hash_file_content(*a, **k):
        raise AssertionError("hash_file_content should not be called on unchanged file when manifest hit")

    monkeypatch.setattr(lc, "hash_file_content", fail_hash_file_content)

    # 2nd run: should be cache hit, and should NOT call hash_file_content
    _p2, hit2 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_hash,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=manifest_path,
        verbose=False,
        lock_timeout_sec=5.0,
    )
    assert hit2 is True
    assert calls["compute"] == 1


def test_ref_tags_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "t.txt"
    _write_text(src, "abc\n")

    cache_dir = tmp_path / ".lemma_cache"
    calls = {"n": 0}

    payload0 = lc.LemmaCachePayload(
        lemmas=Counter({"abc": 3}),
        ref_tags=Counter({"REF:foo": 2, "REF:bar": 1}),
    )

    def compute():
        calls["n"] += 1
        return payload0

    config_hash = lc.build_config_hash(
        stanza_model="perseus",
        lang="la",
        processors="tokenize,pos,lemma",
        use_lemma=True,
        upos_targets={"NOUN"},
        ref_tags_file=None,
        include_ref_tags_in_config_hash=True,
        extra={"stanza_version": "1", "nlpo_toolkit_version": "1"},
    )

    got1, hit1 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_hash,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=cache_dir / "manifest.json",
        verbose=False,
        lock_timeout_sec=5.0,
    )
    got2, hit2 = lc.get_or_compute_cached(
        path=src,
        cache_dir=cache_dir,
        config_hash=config_hash,
        compute_fn=compute,
        use_manifest=True,
        manifest_path=cache_dir / "manifest.json",
        verbose=False,
        lock_timeout_sec=5.0,
    )

    assert hit1 is False
    assert hit2 is True
    assert calls["n"] == 1
    assert got1.ref_tags == payload0.ref_tags
    assert got2.ref_tags == payload0.ref_tags


def test_prune_cache_removes_old_objects_and_stale_locks(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".lemma_cache"
    obj_dir = cache_dir / "objects" / "aa"
    lock_dir = cache_dir / "locks" / "bb"
    obj_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    obj = obj_dir / "deadbeef.json"
    lock = lock_dir / "deadbeef.lock"
    _write_text(obj, '{"cache_version": 999, "payload": {}}')
    _write_text(lock, "pid=123\n")

    # make them old
    _touch_mtime(obj, seconds_ago=10_000)
    _touch_mtime(lock, seconds_ago=10_000)

    assert obj.exists()
    assert lock.exists()

    rep = lc.prune_cache(
        cache_dir,
        keep_days=0,         # delete anything older than "now"
        keep_files=0,        # keep nothing by count
        lock_ttl_sec=0,      # delete any lock older than "now"
        verbose=False,
    )

    assert rep.removed_objects >= 1
    assert rep.removed_locks >= 1
    assert not obj.exists()
    assert not lock.exists()


def test_lock_timeout(tmp_path: Path) -> None:
    # This is a minimal sanity check: if a lock file already exists,
    # acquiring should time out quickly.
    src = tmp_path / "x.txt"
    _write_text(src, "x\n")
    cache_dir = tmp_path / ".lemma_cache"

    config_hash = lc.build_config_hash(
        stanza_model="perseus",
        lang="la",
        processors="tokenize,pos,lemma",
        use_lemma=True,
        upos_targets={"NOUN"},
        ref_tags_file=None,
        include_ref_tags_in_config_hash=True,
        extra={"stanza_version": "1", "nlpo_toolkit_version": "1"},
    )

    # Compute key to know lock path: we just call once to get key material,
    # but we cannot access internal key directly. Instead:
    # create a "miss" run that will lock; we will pre-create a stale lock by
    # using lc._lock_file_path if present; if not present, skip this test.
    if not hasattr(lc, "_lock_file_path") or not hasattr(lc, "_make_cache_key") or not hasattr(lc, "hash_file_content"):
        pytest.skip("internal helpers not available; lock timeout test skipped")

    content_hash = lc.hash_file_content(src)
    cache_key = lc._make_cache_key(content_hash, config_hash)  # type: ignore[attr-defined]
    lock_path = lc._lock_file_path(cache_dir, cache_key)       # type: ignore[attr-defined]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(lock_path, "pid=99999\n")

    def compute():
        return lc.LemmaCachePayload(lemmas=Counter({"x": 1}), ref_tags=Counter())

    with pytest.raises(lc.CacheLockTimeout):
        lc.get_or_compute_cached(
            path=src,
            cache_dir=cache_dir,
            config_hash=config_hash,
            compute_fn=compute,
            use_manifest=False,
            verbose=False,
            lock_timeout_sec=0.2,
        )
