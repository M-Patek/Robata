from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from threading import Barrier
from typing import Any

from robata.frame_cache import LayeredMediaCache, raw_frame_cache_key


def _write_layered_media_cache_from_process(root: str, prefix: str, start: Any) -> None:
    start.wait()
    cache = LayeredMediaCache(root, max_entries=32, max_bytes=4_096)
    for ordinal in range(4):
        key = raw_frame_cache_key(
            source_identity="source-fixture",
            frame_identity=f"{prefix}:{ordinal}",
            decode_identity="target-h264-v1",
        )
        cache.put_raw_frame(key, f"{prefix}-payload-{ordinal}".encode("ascii"))


def test_layered_media_cache_serializes_independent_handles_for_one_root(
    tmp_path: Path,
) -> None:
    first = LayeredMediaCache(tmp_path, max_entries=32, max_bytes=4_096)
    second = LayeredMediaCache(tmp_path, max_entries=32, max_bytes=4_096)
    assert first._lock is second._lock
    start = Barrier(2)

    def write(cache: LayeredMediaCache, prefix: str) -> dict[str, bytes]:
        start.wait()
        written: dict[str, bytes] = {}
        for ordinal in range(4):
            key = raw_frame_cache_key(
                source_identity="source-fixture",
                frame_identity=f"{prefix}:{ordinal}",
                decode_identity="target-h264-v1",
            )
            payload = f"{prefix}-payload-{ordinal}".encode("ascii")
            cache.put_raw_frame(key, payload)
            written[key] = payload
        return written

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(write, first, "left")
        right = pool.submit(write, second, "right")
        expected = left.result() | right.result()

    reopened = LayeredMediaCache(tmp_path, max_entries=32, max_bytes=4_096)
    assert reopened.stats().entry_count == len(expected)
    assert {key: reopened.get_raw_frame(key) for key in expected} == expected
    assert tuple((tmp_path / first.namespace / ".staging").iterdir()) == ()


def test_layered_media_cache_reconciles_stale_writer_staging(tmp_path: Path) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=4, max_bytes=64)
    staging = tmp_path / cache.namespace / ".staging"
    stale = staging / "tmp-crashed-writer"
    stale.write_bytes(b"partial")

    detected = cache.reconcile()
    assert stale in detected.partial_paths
    cleaned = cache.reconcile(remove_partials=True)

    assert stale in cleaned.removed_paths
    assert not stale.exists()


def test_layered_media_cache_serializes_independent_processes_for_one_root(
    tmp_path: Path,
) -> None:
    context = get_context("spawn")
    start = context.Event()
    root = str(tmp_path)
    left = context.Process(
        target=_write_layered_media_cache_from_process,
        args=(root, "left", start),
    )
    right = context.Process(
        target=_write_layered_media_cache_from_process,
        args=(root, "right", start),
    )
    left.start()
    right.start()
    start.set()
    left.join(timeout=30)
    right.join(timeout=30)

    assert left.exitcode == 0
    assert right.exitcode == 0
    cache = LayeredMediaCache(tmp_path, max_entries=32, max_bytes=4_096)
    expected = {
        raw_frame_cache_key(
            source_identity="source-fixture",
            frame_identity=f"{prefix}:{ordinal}",
            decode_identity="target-h264-v1",
        ): f"{prefix}-payload-{ordinal}".encode("ascii")
        for prefix in ("left", "right")
        for ordinal in range(4)
    }
    assert {key: cache.get_raw_frame(key) for key in expected} == expected
    assert tuple((tmp_path / cache.namespace / ".staging").iterdir()) == ()
