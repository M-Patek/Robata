from __future__ import annotations

from pathlib import Path

import pytest

from robata.frame_cache import (
    LayeredMediaCache,
    encoded_artifact_cache_key,
    manifest_cache_key,
    raw_frame_cache_key,
)


def _raw_key(*, frame_identity: str = "cam-01:100") -> str:
    return raw_frame_cache_key(
        source_identity="source-sha256:fixture-a",
        frame_identity=frame_identity,
        decode_identity="fake-nvdec-h264-v1",
    )


def test_layered_media_cache_keys_keep_each_equivalence_boundary_independent() -> None:
    raw_key = _raw_key()
    changed_raw_key = raw_frame_cache_key(
        source_identity="source-sha256:fixture-a",
        frame_identity="cam-01:101",
        decode_identity="fake-nvdec-h264-v1",
    )
    encoded_key = encoded_artifact_cache_key(
        raw_frame_key=raw_key,
        encoding_identity="png-lossless-v1",
    )
    changed_encoded_key = encoded_artifact_cache_key(
        raw_frame_key=raw_key,
        encoding_identity="jpeg-quality-90-v2",
    )
    manifest_key = manifest_cache_key(
        ordered_artifact_keys=(encoded_key, changed_encoded_key),
        manifest_identity="package-bindings-v1",
    )
    reversed_manifest_key = manifest_cache_key(
        ordered_artifact_keys=(changed_encoded_key, encoded_key),
        manifest_identity="package-bindings-v1",
    )

    assert raw_key != changed_raw_key
    assert raw_key != encoded_key
    assert encoded_key != changed_encoded_key
    assert manifest_key != reversed_manifest_key


def test_layered_media_cache_persists_and_tracks_layer_hits_and_misses(tmp_path: Path) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=8, max_bytes=64)
    raw_key = _raw_key()
    encoded_key = encoded_artifact_cache_key(
        raw_frame_key=raw_key,
        encoding_identity="png-lossless-v1",
    )
    package_key = manifest_cache_key(
        ordered_artifact_keys=(encoded_key,),
        manifest_identity="package-bindings-v1",
    )

    cache.put_raw_frame(raw_key, b"raw")
    cache.put_encoded_artifact(encoded_key, b"encoded")
    cache.put_manifest(package_key, b'{"frames":["encoded"]}')

    assert cache.get_raw_frame(raw_key) == b"raw"
    assert cache.get_encoded_artifact(encoded_key) == b"encoded"
    assert cache.get_manifest(package_key) == b'{"frames":["encoded"]}'
    assert cache.get_raw_frame("0" * 64) is None

    stats = cache.stats()
    assert stats.entry_count == 3
    assert stats.raw_frame_count == 1
    assert stats.encoded_artifact_count == 1
    assert stats.manifest_count == 1
    assert stats.byte_count == len(b"rawencoded") + len(b'{"frames":["encoded"]}')
    assert stats.cache_hits == 3
    assert stats.cache_misses == 1

    reopened = LayeredMediaCache(tmp_path, max_entries=8, max_bytes=64)
    assert reopened.get_raw_frame(raw_key) == b"raw"
    assert reopened.get_encoded_artifact(encoded_key) == b"encoded"
    assert reopened.get_manifest(package_key) == b'{"frames":["encoded"]}'


def test_layered_media_cache_evicts_oldest_entry_deterministically_with_byte_bound(
    tmp_path: Path,
) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=2, max_bytes=4)
    raw_key = _raw_key(frame_identity="cam-01:100")
    encoded_key = encoded_artifact_cache_key(
        raw_frame_key=raw_key,
        encoding_identity="png-lossless-v1",
    )
    package_key = manifest_cache_key(
        ordered_artifact_keys=(encoded_key,),
        manifest_identity="package-bindings-v1",
    )

    cache.put_raw_frame(raw_key, b"aa")
    cache.put_encoded_artifact(encoded_key, b"bb")
    assert cache.get_raw_frame(raw_key) == b"aa"
    cache.put_manifest(package_key, b"cc")

    assert cache.get_raw_frame(raw_key) == b"aa"
    assert cache.get_encoded_artifact(encoded_key) is None
    assert cache.get_manifest(package_key) == b"cc"
    stats = cache.stats()
    assert stats.entry_count == 2
    assert stats.byte_count == 4
    assert stats.eviction_count == 1


def test_layered_media_cache_enforces_lower_retention_limits_after_restart(
    tmp_path: Path,
) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=4, max_bytes=16)
    raw_key = _raw_key()
    encoded_key = encoded_artifact_cache_key(
        raw_frame_key=raw_key,
        encoding_identity="png-lossless-v1",
    )
    package_key = manifest_cache_key(
        ordered_artifact_keys=(encoded_key,),
        manifest_identity="package-bindings-v1",
    )

    cache.put_raw_frame(raw_key, b"aa")
    cache.put_encoded_artifact(encoded_key, b"bb")
    cache.put_manifest(package_key, b"cc")

    reopened = LayeredMediaCache(tmp_path, max_entries=1, max_bytes=2)
    stats = reopened.stats()
    assert stats.entry_count == 1
    assert stats.byte_count == 2
    assert stats.eviction_count == 2
    assert reopened.get_raw_frame(raw_key) is None
    assert reopened.get_encoded_artifact(encoded_key) is None
    assert reopened.get_manifest(package_key) == b"cc"


def test_layered_media_cache_keeps_a_shared_blob_until_its_last_entry_is_evicted(
    tmp_path: Path,
) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=1, max_bytes=16)
    raw_key = _raw_key()
    encoded_key = encoded_artifact_cache_key(
        raw_frame_key=raw_key,
        encoding_identity="png-lossless-v1",
    )

    cache.put_raw_frame(raw_key, b"same-bytes")
    cache.put_encoded_artifact(encoded_key, b"same-bytes")

    assert cache.get_raw_frame(raw_key) is None
    assert cache.get_encoded_artifact(encoded_key) == b"same-bytes"
    assert cache.stats().byte_count == len(b"same-bytes")
    assert len(tuple((tmp_path / cache.namespace / "blobs").glob("*.blob"))) == 1


def test_layered_media_cache_invalidates_only_the_requested_shared_mapping(
    tmp_path: Path,
) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=4, max_bytes=64)
    raw_key = _raw_key()
    encoded_key = encoded_artifact_cache_key(
        raw_frame_key=raw_key,
        encoding_identity="png-lossless-v1",
    )

    raw_entry = cache.put_raw_frame(raw_key, b"same-bytes")
    cache.put_encoded_artifact(encoded_key, b"same-bytes")
    blob_path = tmp_path / cache.namespace / "blobs" / f"{raw_entry.content_sha256}.blob"

    assert cache.invalidate("raw", raw_key)
    assert cache.get_raw_frame(raw_key) is None
    assert cache.get_encoded_artifact(encoded_key) == b"same-bytes"
    assert blob_path.exists()

    assert cache.invalidate("encoded", encoded_key)
    assert not blob_path.exists()
    assert not cache.invalidate("encoded", encoded_key)


def test_layered_media_cache_detects_and_cleans_corruption(tmp_path: Path) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=4, max_bytes=64)
    raw_key = _raw_key()
    entry = cache.put_raw_frame(raw_key, b"verified-frame")
    blob_path = tmp_path / cache.namespace / "blobs" / f"{entry.content_sha256}.blob"
    blob_path.write_bytes(b"tampered")

    assert cache.get_raw_frame(raw_key) is None
    assert cache.stats().corruption_count == 1

    report = cache.reconcile()
    assert f"raw:{raw_key}" in report.corrupt_entry_keys
    assert blob_path in report.corrupt_blob_paths
    assert not report.reconciled

    cleaned = cache.reconcile(remove_corrupt=True, remove_orphans=True)
    assert blob_path in cleaned.removed_paths
    assert cleaned.reconciled
    assert cache.get_raw_frame(raw_key) is None


def test_layered_media_cache_removes_stale_staging_files_on_startup(tmp_path: Path) -> None:
    cache = LayeredMediaCache(tmp_path, max_entries=4, max_bytes=64)
    staging = tmp_path / cache.namespace / ".staging"
    stale = staging / "crashed-writer.tmp"
    stale.write_bytes(b"partial")

    reopened = LayeredMediaCache(tmp_path, max_entries=4, max_bytes=64)

    assert not stale.exists()
    assert tuple(staging.iterdir()) == ()
    assert reopened.reconcile().reconciled


def test_layered_media_cache_rejects_unsafe_configuration_and_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        LayeredMediaCache(tmp_path, namespace="../outside")

    cache = LayeredMediaCache(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        cache.get_raw_frame("not-a-cache-key")
