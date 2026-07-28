"""Shared local frame cache and feed-once coordination.

QA is the first stage that decodes footage.  This module provides a small provider-neutral cache
that can be replaced by R2/object storage in production while preserving the same manifest and
idempotency semantics.  Annotation consumes the manifest instead of decoding the source again.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from typing import Annotated, Any, ClassVar

from pydantic import Field, StringConstraints

from robata.contracts.common import Sha256Digest, StrictModel
from robata.tempfiles import make_temp_file

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


class FrameRef(StrictModel):
    """A content-addressed frame reference; image bytes remain outside the wire record."""

    frame_id: NonEmptyString
    video_id: NonEmptyString
    ordinal: NonNegativeInt
    timestamp_sec: NonNegativeFloat
    uri: NonEmptyString
    content_sha256: Sha256Digest
    size_bytes: NonNegativeInt


class FrameFeedManifest(StrictModel):
    """The immutable output of one decode/feed operation."""

    video_id: NonEmptyString
    source_uri: NonEmptyString
    frame_rate: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    frames: tuple[FrameRef, ...]
    cache_key: NonEmptyString
    decoded_once: bool = True

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def total_bytes(self) -> int:
        return sum(frame.size_bytes for frame in self.frames)

    def frame_at(self, ordinal: int) -> FrameRef:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("ordinal must be an integer")
        try:
            return self.frames[ordinal]
        except IndexError as exc:
            raise KeyError(ordinal) from exc


@dataclass(frozen=True, slots=True)
class FramePayload:
    """Decoder-neutral frame payload accepted by :meth:`SharedFrameCache.feed_once`."""

    timestamp_sec: float
    data: bytes
    frame_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_sec, bool) or not isinstance(self.timestamp_sec, (int, float)):
            raise TypeError("timestamp_sec must be numeric")
        if self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be non-negative")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("data must be non-empty bytes")
        if self.frame_id is not None and (
            not isinstance(self.frame_id, str) or not self.frame_id.strip()
        ):
            raise ValueError("frame_id must be a non-empty string when supplied")


@dataclass(frozen=True, slots=True)
class FrameCacheStats:
    frame_count: int
    byte_count: int
    cache_hits: int
    cache_misses: int
    decode_attempts: int


@dataclass(frozen=True, slots=True)
class FeedOnceResult:
    manifest: FrameFeedManifest
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class FrameCacheReconciliation:
    """A bounded visibility and crash-leftover report for one local frame cache."""

    manifest_count: int
    visible_manifest_count: int
    missing_frame_refs: tuple[str, ...] = ()
    corrupt_frame_refs: tuple[str, ...] = ()
    invalid_manifest_paths: tuple[Path, ...] = ()
    orphan_blob_paths: tuple[Path, ...] = ()
    partial_blob_paths: tuple[Path, ...] = ()
    removed_paths: tuple[Path, ...] = ()
    failed_removal_paths: tuple[Path, ...] = ()

    @property
    def missing_count(self) -> int:
        return len(self.missing_frame_refs)

    @property
    def corrupt_count(self) -> int:
        return len(self.corrupt_frame_refs)

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_blob_paths)

    @property
    def partial_count(self) -> int:
        return len(self.partial_blob_paths) + len(self.invalid_manifest_paths)

    @property
    def issue_count(self) -> int:
        return (
            self.missing_count
            + self.corrupt_count
            + self.orphan_count
            + self.partial_count
            + len(self.failed_removal_paths)
        )

    @property
    def reconciled(self) -> bool:
        return self.issue_count == 0

    @property
    def ok(self) -> bool:
        return self.reconciled


@dataclass(frozen=True, slots=True)
class FrameCacheCapacityEstimate:
    """Storage estimate for a retention window; it is an assumption, not a bill."""

    recording_hours_per_day: float = 500.0
    cameras: int = 6
    frame_rate: float = 2.0
    average_frame_bytes: int = 100_000
    retention_days: int = 3

    def __post_init__(self) -> None:
        if self.recording_hours_per_day <= 0 or self.cameras <= 0 or self.frame_rate <= 0:
            raise ValueError("recording hours, cameras, and frame rate must be positive")
        if self.average_frame_bytes <= 0 or self.retention_days <= 0:
            raise ValueError("average frame size and retention must be positive")

    @property
    def estimated_bytes(self) -> int:
        return round(
            self.recording_hours_per_day
            * 3600
            * self.cameras
            * self.frame_rate
            * self.average_frame_bytes
            * self.retention_days
        )

    @property
    def estimated_terabytes(self) -> float:
        return self.estimated_bytes / 1_000_000_000_000


_LAYERED_MEDIA_LAYERS = ("raw", "encoded", "manifest")
_LAYERED_MEDIA_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_LAYERED_MEDIA_ENTRY_SUFFIX = ".entry.json"
_LAYERED_MEDIA_PROCESS_LOCK_FILENAME = ".lock"
_LAYERED_MEDIA_PROCESS_LOCK_TIMEOUT_SECONDS = 30.0
_LAYERED_MEDIA_PROCESS_LOCK_RETRY_SECONDS = 0.05


def _validate_layered_media_layer(layer: str) -> str:
    if layer not in _LAYERED_MEDIA_LAYERS:
        raise ValueError(f"unsupported media cache layer: {layer!r}")
    return layer


def _validate_layered_media_key(cache_key: str) -> str:
    if not isinstance(cache_key, str) or _LAYERED_MEDIA_KEY_RE.fullmatch(cache_key) is None:
        raise ValueError("cache_key must be a lowercase SHA-256 digest")
    return cache_key


def _validate_layered_media_identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL bytes")
    return value


def _layered_media_equivalence_key(layer: str, attributes: Mapping[str, object]) -> str:
    canonical = json.dumps(
        {
            "layer": layer,
            "semantic_projection_version": "layered-media-equivalence-v1",
            "attributes": attributes,
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def raw_frame_cache_key(
    *,
    source_identity: str,
    frame_identity: str,
    decode_identity: str,
) -> str:
    """Return the raw-frame equivalence key without conflating it with encoding policy."""

    return _layered_media_equivalence_key(
        "raw",
        {
            "decode_identity": _validate_layered_media_identity(decode_identity, "decode_identity"),
            "frame_identity": _validate_layered_media_identity(frame_identity, "frame_identity"),
            "source_identity": _validate_layered_media_identity(source_identity, "source_identity"),
        },
    )


def encoded_artifact_cache_key(*, raw_frame_key: str, encoding_identity: str) -> str:
    """Return an encoded-artifact key bound to one raw frame and one encoding policy."""

    return _layered_media_equivalence_key(
        "encoded",
        {
            "encoding_identity": _validate_layered_media_identity(
                encoding_identity, "encoding_identity"
            ),
            "raw_frame_key": _validate_layered_media_key(raw_frame_key),
        },
    )


def manifest_cache_key(
    *,
    ordered_artifact_keys: Iterable[str],
    manifest_identity: str,
) -> str:
    """Return an ordered manifest key independent of raw and encoded cache keys."""

    if isinstance(ordered_artifact_keys, str):
        raise TypeError("ordered_artifact_keys must be an iterable of cache keys")
    keys = tuple(_validate_layered_media_key(key) for key in ordered_artifact_keys)
    return _layered_media_equivalence_key(
        "manifest",
        {
            "manifest_identity": _validate_layered_media_identity(
                manifest_identity, "manifest_identity"
            ),
            "ordered_artifact_keys": keys,
        },
    )


@dataclass(frozen=True, slots=True)
class LayeredMediaCacheEntry:
    """One semantic cache mapping backed by a content-addressed blob."""

    layer: str
    cache_key: str
    content_sha256: str
    size_bytes: int
    access_sequence: int

    def __post_init__(self) -> None:
        _validate_layered_media_layer(self.layer)
        _validate_layered_media_key(self.cache_key)
        _validate_layered_media_key(self.content_sha256)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if isinstance(self.access_sequence, bool) or not isinstance(self.access_sequence, int):
            raise TypeError("access_sequence must be an integer")
        if self.access_sequence <= 0:
            raise ValueError("access_sequence must be positive")


@dataclass(frozen=True, slots=True)
class LayeredMediaCacheStats:
    """Current bounded storage facts and process-local lookup counters."""

    entry_count: int
    byte_count: int
    raw_frame_count: int
    encoded_artifact_count: int
    manifest_count: int
    cache_hits: int
    cache_misses: int
    eviction_count: int
    corruption_count: int


@dataclass(frozen=True, slots=True)
class LayeredMediaCacheReconciliation:
    """Visibility, corruption, and cleanup report for a layered media cache."""

    entry_count: int
    visible_entry_count: int
    missing_entry_keys: tuple[str, ...] = ()
    corrupt_entry_keys: tuple[str, ...] = ()
    invalid_entry_paths: tuple[Path, ...] = ()
    corrupt_blob_paths: tuple[Path, ...] = ()
    orphan_blob_paths: tuple[Path, ...] = ()
    partial_paths: tuple[Path, ...] = ()
    removed_paths: tuple[Path, ...] = ()
    failed_removal_paths: tuple[Path, ...] = ()

    @property
    def missing_count(self) -> int:
        return len(self.missing_entry_keys)

    @property
    def corrupt_count(self) -> int:
        return (
            len(self.corrupt_entry_keys)
            + len(self.invalid_entry_paths)
            + len(self.corrupt_blob_paths)
        )

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_blob_paths)

    @property
    def partial_count(self) -> int:
        return len(self.partial_paths)

    @property
    def issue_count(self) -> int:
        return (
            self.missing_count
            + self.corrupt_count
            + self.orphan_count
            + self.partial_count
            + len(self.failed_removal_paths)
        )

    @property
    def reconciled(self) -> bool:
        return self.issue_count == 0

    @property
    def ok(self) -> bool:
        return self.reconciled


class _LayeredMediaCacheCorruption(ValueError):
    """A persisted entry or blob cannot safely serve a cache hit."""


@dataclass(slots=True)
class _LayeredMediaCacheScan:
    entry_paths: list[Path]
    entries: dict[tuple[str, str], LayeredMediaCacheEntry]
    invalid_entry_paths: list[Path]
    missing_entries: list[tuple[str, Path]]
    corrupt_entries: list[tuple[str, Path]]
    corrupt_blob_paths: list[Path]
    orphan_blob_paths: list[Path]
    partial_paths: list[Path]


def _lock_layered_media_cache_process(stream: Any) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        deadline = monotonic() + _LAYERED_MEDIA_PROCESS_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out acquiring layered media cache process lock"
                    ) from error
                sleep(_LAYERED_MEDIA_PROCESS_LOCK_RETRY_SECONDS)
                stream.seek(0)

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]


def _unlock_layered_media_cache_process(stream: Any) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def _with_layered_media_root_lock(operation: Callable[..., Any]) -> Callable[..., Any]:
    """Serialize cache internals across every local handle for the same root."""

    @wraps(operation)
    def locked(cache: Any, *args: Any, **kwargs: Any) -> Any:
        with cache._lock:
            return operation(cache, *args, **kwargs)

    return locked


class LayeredMediaCache:
    """Bounded local cache with separate raw, encoded, and manifest equivalence layers.

    Semantic keys answer whether a layer may be reused; content SHA-256 values address the
    stored bytes. The distinction prevents a change in encoding or manifest semantics from
    accidentally reusing a raw frame, and lets identical bytes share one local blob.

    Handles for the same root share a process-local lock and an on-disk process lock. This keeps
    startup cleanup from classifying another writer's atomic-write temporary file as stale.
    """

    _root_locks_guard: ClassVar[threading.Lock] = threading.Lock()
    _root_locks: ClassVar[dict[Path, threading.RLock]] = {}

    raw_frame_cache_key = staticmethod(raw_frame_cache_key)
    encoded_artifact_cache_key = staticmethod(encoded_artifact_cache_key)
    manifest_cache_key = staticmethod(manifest_cache_key)
    raw_frame_key = raw_frame_cache_key
    encoded_artifact_key = encoded_artifact_cache_key
    manifest_key = manifest_cache_key

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        namespace: str = "layered-media-v1",
        max_entries: int = 1_024,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self._validate_namespace(namespace)
        self._validate_capacity(max_entries, "max_entries")
        self._validate_capacity(max_bytes, "max_bytes")
        self.root = Path(root)
        self.namespace = namespace
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._root = self.root / namespace
        self._root.mkdir(parents=True, exist_ok=True)
        self._resolved_root = self._root.resolve(strict=True)
        for layer in _LAYERED_MEDIA_LAYERS:
            self._layer_directory(layer).mkdir(parents=True, exist_ok=True)
        self._blob_directory.mkdir(parents=True, exist_ok=True)
        self._staging_directory.mkdir(parents=True, exist_ok=True)
        self._lock = self._lock_for_root(self._resolved_root)
        self._cache_hits = 0
        self._cache_misses = 0
        self._eviction_count = 0
        self._corruption_count = 0
        with self._storage_transaction():
            self._remove_stale_staging_files()
            scan = self._scan_storage()
            self._access_sequence = max(
                (entry.access_sequence for entry in scan.entries.values()),
                default=0,
            )
            self._enforce_capacity()

    def put_raw_frame(self, cache_key: str, data: bytes) -> LayeredMediaCacheEntry:
        return self.put("raw", cache_key, data)

    def get_raw_frame(self, cache_key: str) -> bytes | None:
        return self.get("raw", cache_key)

    def put_encoded_artifact(self, cache_key: str, data: bytes) -> LayeredMediaCacheEntry:
        return self.put("encoded", cache_key, data)

    def get_encoded_artifact(self, cache_key: str) -> bytes | None:
        return self.get("encoded", cache_key)

    def put_manifest(self, cache_key: str, data: bytes) -> LayeredMediaCacheEntry:
        return self.put("manifest", cache_key, data)

    def get_manifest(self, cache_key: str) -> bytes | None:
        return self.get("manifest", cache_key)

    def invalidate(self, layer: str, cache_key: str) -> bool:
        """Remove one cache mapping so verified bytes can replace a semantic miss.

        Content blobs are shared across layers, so an unreferenced blob is removed only
        after its entry is gone and no visible cache mapping still references it.
        """

        layer = _validate_layered_media_layer(layer)
        cache_key = _validate_layered_media_key(cache_key)
        entry_path = self._entry_path(layer, cache_key)
        with self._storage_transaction():
            try:
                entry = self._read_entry(layer, cache_key)
            except _LayeredMediaCacheCorruption:
                self._corruption_count += 1
                entry = None
            try:
                entry_path.lstat()
            except FileNotFoundError:
                return False
            except OSError as error:
                raise OSError(
                    f"could not inspect layered media cache entry: {entry_path}"
                ) from error

            removed, failed = self._remove_paths([entry_path])
            if entry_path not in removed or failed:
                raise OSError(f"could not invalidate layered media cache entry: {entry_path}")

            if entry is not None:
                scan = self._scan_storage()
                if not any(
                    remaining.content_sha256 == entry.content_sha256
                    for remaining in scan.entries.values()
                ):
                    blob_path = self._blob_path(entry.content_sha256)
                    try:
                        blob_path.lstat()
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        raise OSError(
                            f"could not inspect layered media cache blob: {blob_path}"
                        ) from error
                    else:
                        blob_removed, blob_failed = self._remove_paths([blob_path])
                        if blob_path not in blob_removed or blob_failed:
                            raise OSError(
                                f"could not invalidate layered media cache blob: {blob_path}"
                            )
            return True

    def put(self, layer: str, cache_key: str, data: bytes) -> LayeredMediaCacheEntry:
        """Store bytes for an equivalence key or verify the existing immutable mapping."""

        layer = _validate_layered_media_layer(layer)
        cache_key = _validate_layered_media_key(cache_key)
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if len(data) > self.max_bytes:
            raise ValueError("cache value exceeds max_bytes")
        content_sha256 = hashlib.sha256(data).hexdigest()
        with self._storage_transaction():
            try:
                existing = self._read_entry(layer, cache_key)
            except _LayeredMediaCacheCorruption:
                self._corruption_count += 1
                existing = None
            if existing is not None:
                existing_data = self._entry_data(existing)
                if existing_data is None:
                    self._corruption_count += 1
                elif existing.content_sha256 != content_sha256:
                    raise ValueError(
                        "cache key is already bound to different content; "
                        "use a different equivalence key"
                    )
                else:
                    self._cache_hits += 1
                    return self._touch(existing)

            self._write_blob(content_sha256, data)
            entry = LayeredMediaCacheEntry(
                layer=layer,
                cache_key=cache_key,
                content_sha256=content_sha256,
                size_bytes=len(data),
                access_sequence=self._next_access_sequence(),
            )
            self._write_entry(entry)
            self._enforce_capacity()
            return entry

    def get(self, layer: str, cache_key: str) -> bytes | None:
        """Return verified bytes and advance deterministic LRU state on a cache hit."""

        layer = _validate_layered_media_layer(layer)
        cache_key = _validate_layered_media_key(cache_key)
        with self._storage_transaction():
            try:
                entry = self._read_entry(layer, cache_key)
            except _LayeredMediaCacheCorruption:
                self._corruption_count += 1
                self._cache_misses += 1
                return None
            if entry is None:
                self._cache_misses += 1
                return None
            data = self._entry_data(entry)
            if data is None:
                self._corruption_count += 1
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            self._touch(entry)
            return data

    def stats(self) -> LayeredMediaCacheStats:
        with self._storage_transaction():
            scan = self._scan_storage()
            entries = tuple(scan.entries.values())
            content_sizes = {entry.content_sha256: entry.size_bytes for entry in entries}
            return LayeredMediaCacheStats(
                entry_count=len(entries),
                byte_count=sum(content_sizes.values()),
                raw_frame_count=sum(entry.layer == "raw" for entry in entries),
                encoded_artifact_count=sum(entry.layer == "encoded" for entry in entries),
                manifest_count=sum(entry.layer == "manifest" for entry in entries),
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                eviction_count=self._eviction_count,
                corruption_count=self._corruption_count,
            )

    def reconcile(
        self,
        *,
        remove_corrupt: bool = False,
        remove_orphans: bool = False,
        remove_partials: bool = False,
        strict: bool = False,
    ) -> LayeredMediaCacheReconciliation:
        """Inspect persisted cache state and optionally remove only cache-owned leftovers."""

        for name, value in (
            ("remove_corrupt", remove_corrupt),
            ("remove_orphans", remove_orphans),
            ("remove_partials", remove_partials),
            ("strict", strict),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

        with self._storage_transaction():
            initial = self._scan_storage_with_staging_partials()
            removed: list[Path] = []
            failed: list[Path] = []
            current = initial
            for _ in range(2):
                cleanup = self._cleanup_paths(
                    current,
                    remove_corrupt=remove_corrupt,
                    remove_orphans=remove_orphans,
                    remove_partials=remove_partials,
                )
                if not cleanup:
                    break
                removed_now, failed_now = self._remove_paths(cleanup)
                removed.extend(removed_now)
                failed.extend(failed_now)
                if not removed_now:
                    break
                current = self._scan_storage_with_staging_partials()
            report = self._reconciliation_report(
                entry_count=len(initial.entry_paths),
                visible_entry_count=len(initial.entries),
                scan=current,
                removed=removed,
                failed=failed,
            )
        if strict and not report.reconciled:
            raise ValueError("layered media cache reconciliation found unresolved discrepancies")
        return report

    def reconcile_storage(self, **kwargs: object) -> LayeredMediaCacheReconciliation:
        """Compatibility spelling for callers that name the backing storage explicitly."""

        return self.reconcile(**kwargs)  # type: ignore[arg-type]

    @_with_layered_media_root_lock
    def _enforce_capacity(self) -> None:
        scan = self._scan_storage()
        cleanup = self._cleanup_paths(
            scan,
            remove_corrupt=True,
            remove_orphans=True,
            remove_partials=True,
        )
        if cleanup:
            self._remove_paths(cleanup)
            scan = self._scan_storage()
        entries = dict(scan.entries)
        content_sizes = {entry.content_sha256: entry.size_bytes for entry in entries.values()}
        total_bytes = sum(content_sizes.values())
        candidates = sorted(
            entries.values(),
            key=lambda entry: (
                entry.access_sequence,
                _LAYERED_MEDIA_LAYERS.index(entry.layer),
                entry.cache_key,
            ),
        )
        while len(entries) > self.max_entries or total_bytes > self.max_bytes:
            if not candidates:
                raise OSError("layered media cache capacity cannot be enforced")
            entry = candidates.pop(0)
            entry_path = self._entry_path(entry.layer, entry.cache_key)
            removed, failed = self._remove_paths([entry_path])
            if entry_path not in removed or failed:
                raise OSError("could not evict layered media cache entry")
            entries.pop((entry.layer, entry.cache_key), None)
            self._eviction_count += 1
            if not any(
                remaining.content_sha256 == entry.content_sha256 for remaining in entries.values()
            ):
                blob_path = self._blob_path(entry.content_sha256)
                blob_removed, blob_failed = self._remove_paths([blob_path])
                if blob_path not in blob_removed or blob_failed:
                    raise OSError("could not evict layered media cache blob")
                content_sizes.pop(entry.content_sha256, None)
            total_bytes = sum(content_sizes.values())

    def _cleanup_paths(
        self,
        scan: _LayeredMediaCacheScan,
        *,
        remove_corrupt: bool,
        remove_orphans: bool,
        remove_partials: bool,
    ) -> list[Path]:
        cleanup: list[Path] = []
        if remove_corrupt:
            cleanup.extend(scan.invalid_entry_paths)
            cleanup.extend(path for _, path in scan.missing_entries)
            cleanup.extend(path for _, path in scan.corrupt_entries)
            cleanup.extend(scan.corrupt_blob_paths)
        if remove_orphans:
            cleanup.extend(scan.orphan_blob_paths)
        if remove_partials:
            cleanup.extend(scan.partial_paths)
        return cleanup

    def _reconciliation_report(
        self,
        *,
        entry_count: int,
        visible_entry_count: int,
        scan: _LayeredMediaCacheScan,
        removed: Iterable[Path],
        failed: Iterable[Path],
    ) -> LayeredMediaCacheReconciliation:
        return LayeredMediaCacheReconciliation(
            entry_count=entry_count,
            visible_entry_count=visible_entry_count,
            missing_entry_keys=tuple(sorted({key for key, _ in scan.missing_entries})),
            corrupt_entry_keys=tuple(sorted({key for key, _ in scan.corrupt_entries})),
            invalid_entry_paths=tuple(
                sorted(set(scan.invalid_entry_paths), key=lambda path: path.as_posix())
            ),
            corrupt_blob_paths=tuple(
                sorted(set(scan.corrupt_blob_paths), key=lambda path: path.as_posix())
            ),
            orphan_blob_paths=tuple(
                sorted(set(scan.orphan_blob_paths), key=lambda path: path.as_posix())
            ),
            partial_paths=tuple(sorted(set(scan.partial_paths), key=lambda path: path.as_posix())),
            removed_paths=tuple(sorted(set(removed), key=lambda path: path.as_posix())),
            failed_removal_paths=tuple(sorted(set(failed), key=lambda path: path.as_posix())),
        )

    @_with_layered_media_root_lock
    def _scan_storage(self) -> _LayeredMediaCacheScan:
        scan = _LayeredMediaCacheScan(
            entry_paths=[],
            entries={},
            invalid_entry_paths=[],
            missing_entries=[],
            corrupt_entries=[],
            corrupt_blob_paths=[],
            orphan_blob_paths=[],
            partial_paths=[],
        )
        for layer in _LAYERED_MEDIA_LAYERS:
            directory = self._layer_directory(layer)
            for path in sorted(directory.iterdir(), key=lambda value: value.as_posix()):
                try:
                    file_stat = path.lstat()
                except OSError:
                    scan.partial_paths.append(path)
                    continue
                if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                    scan.partial_paths.append(path)
                    continue
                scan.entry_paths.append(path)
                if not path.name.endswith(_LAYERED_MEDIA_ENTRY_SUFFIX):
                    scan.partial_paths.append(path)
                    continue
                cache_key = path.name[: -len(_LAYERED_MEDIA_ENTRY_SUFFIX)]
                try:
                    _validate_layered_media_key(cache_key)
                    entry = self._read_entry(layer, cache_key)
                except _LayeredMediaCacheCorruption:
                    scan.invalid_entry_paths.append(path)
                    continue
                if entry is None:
                    continue
                state = self._blob_state(
                    self._blob_path(entry.content_sha256),
                    entry.content_sha256,
                    entry.size_bytes,
                )
                entry_key = f"{layer}:{cache_key}"
                if state == "MISSING":
                    scan.missing_entries.append((entry_key, path))
                elif state != "VISIBLE":
                    scan.corrupt_entries.append((entry_key, path))
                else:
                    scan.entries[(layer, cache_key)] = entry

        referenced_blobs = {
            self._blob_path(entry.content_sha256) for entry in scan.entries.values()
        }
        for path in sorted(self._blob_directory.iterdir(), key=lambda value: value.as_posix()):
            try:
                file_stat = path.lstat()
            except OSError:
                scan.partial_paths.append(path)
                continue
            if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                scan.partial_paths.append(path)
                continue
            if not path.name.endswith(".blob"):
                scan.partial_paths.append(path)
                continue
            content_sha256 = path.name[:-5]
            try:
                _validate_layered_media_key(content_sha256)
            except ValueError:
                scan.partial_paths.append(path)
                continue
            if self._blob_state(path, content_sha256) != "VISIBLE":
                scan.corrupt_blob_paths.append(path)
            elif path not in referenced_blobs:
                scan.orphan_blob_paths.append(path)
        return scan

    def _scan_storage_with_staging_partials(self) -> _LayeredMediaCacheScan:
        scan = self._scan_storage()
        scan.partial_paths.extend(self._staging_partial_paths())
        return scan

    def _remove_stale_staging_files(self) -> None:
        stale_paths = self._staging_partial_paths()
        if stale_paths:
            self._remove_paths(stale_paths)

    def _staging_partial_paths(self) -> list[Path]:
        """Report temporary writer files only during explicit reconciliation."""

        directory = self._staging_directory
        try:
            paths = sorted(directory.iterdir(), key=lambda path: path.as_posix())
        except FileNotFoundError:
            return []
        except OSError:
            return [directory]
        partial_paths: list[Path] = []
        for path in paths:
            try:
                self._assert_safe_path(path)
                path.lstat()
            except (OSError, ValueError):
                partial_paths.append(path)
                continue
            partial_paths.append(path)
        return partial_paths

    def _read_entry(self, layer: str, cache_key: str) -> LayeredMediaCacheEntry | None:
        path = self._entry_path(layer, cache_key)
        try:
            payload = self._read_regular_bytes(path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _LayeredMediaCacheCorruption(f"invalid cache entry: {path}") from exc
        try:
            decoded = json.loads(payload)
            if not isinstance(decoded, dict):
                raise TypeError("entry must be an object")
            expected_fields = {
                "access_sequence",
                "cache_key",
                "content_sha256",
                "layer",
                "metadata_sha256",
                "semantic_projection_version",
                "size_bytes",
            }
            if set(decoded) != expected_fields:
                raise ValueError("entry fields do not match the local cache format")
            core = {
                "access_sequence": decoded["access_sequence"],
                "cache_key": decoded["cache_key"],
                "content_sha256": decoded["content_sha256"],
                "layer": decoded["layer"],
                "semantic_projection_version": decoded["semantic_projection_version"],
                "size_bytes": decoded["size_bytes"],
            }
            if core["semantic_projection_version"] != "layered-media-entry-v1":
                raise ValueError("unsupported local cache entry version")
            expected_metadata_sha256 = self._metadata_sha256(core)
            if decoded["metadata_sha256"] != expected_metadata_sha256:
                raise ValueError("entry metadata hash mismatch")
            entry = LayeredMediaCacheEntry(
                layer=decoded["layer"],
                cache_key=decoded["cache_key"],
                content_sha256=decoded["content_sha256"],
                size_bytes=decoded["size_bytes"],
                access_sequence=decoded["access_sequence"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise _LayeredMediaCacheCorruption(f"invalid cache entry: {path}") from exc
        if entry.layer != layer or entry.cache_key != cache_key:
            raise _LayeredMediaCacheCorruption(f"entry identity does not match path: {path}")
        return entry

    def _entry_data(self, entry: LayeredMediaCacheEntry) -> bytes | None:
        path = self._blob_path(entry.content_sha256)
        try:
            data = self._read_regular_bytes(path)
        except (FileNotFoundError, OSError):
            return None
        if len(data) != entry.size_bytes:
            return None
        if hashlib.sha256(data).hexdigest() != entry.content_sha256:
            return None
        return data

    def _write_blob(self, content_sha256: str, data: bytes) -> None:
        path = self._blob_path(content_sha256)
        if self._blob_state(path, content_sha256, len(data)) == "VISIBLE":
            return
        self._atomic_write(path, data)

    def _write_entry(self, entry: LayeredMediaCacheEntry) -> None:
        core = {
            "access_sequence": entry.access_sequence,
            "cache_key": entry.cache_key,
            "content_sha256": entry.content_sha256,
            "layer": entry.layer,
            "semantic_projection_version": "layered-media-entry-v1",
            "size_bytes": entry.size_bytes,
        }
        payload = dict(core)
        payload["metadata_sha256"] = self._metadata_sha256(core)
        self._atomic_write(
            self._entry_path(entry.layer, entry.cache_key),
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
        )

    def _touch(self, entry: LayeredMediaCacheEntry) -> LayeredMediaCacheEntry:
        touched = LayeredMediaCacheEntry(
            layer=entry.layer,
            cache_key=entry.cache_key,
            content_sha256=entry.content_sha256,
            size_bytes=entry.size_bytes,
            access_sequence=self._next_access_sequence(),
        )
        self._write_entry(touched)
        return touched

    def _next_access_sequence(self) -> int:
        visible_sequence = max(
            (entry.access_sequence for entry in self._scan_storage().entries.values()),
            default=0,
        )
        self._access_sequence = max(self._access_sequence, visible_sequence)
        self._access_sequence += 1
        return self._access_sequence

    @classmethod
    def _lock_for_root(cls, root: Path) -> threading.RLock:
        """Return the process-local lock shared by every handle for ``root``."""

        with cls._root_locks_guard:
            existing = cls._root_locks.get(root)
            if existing is not None:
                return existing
            created = threading.RLock()
            cls._root_locks[root] = created
            return created

    @contextmanager
    def _storage_transaction(self) -> Iterator[None]:
        with self._lock, self._exclusive_process_lock():
            yield

    @contextmanager
    def _exclusive_process_lock(self) -> Iterator[None]:
        lock_path = self._root / _LAYERED_MEDIA_PROCESS_LOCK_FILENAME
        self._assert_safe_path(lock_path)
        try:
            metadata = lock_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise OSError(f"could not inspect layered media cache lock: {lock_path}") from error
        else:
            if lock_path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise OSError("layered media cache process lock must be a regular file")

        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        if os.name != "nt":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("layered media cache process lock must be a regular file")
            stream = os.fdopen(descriptor, "r+b")
        except BaseException:
            os.close(descriptor)
            raise

        locked = False
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            _lock_layered_media_cache_process(stream)
            locked = True
            yield
        finally:
            if locked:
                _unlock_layered_media_cache_process(stream)
            stream.close()

    def _layer_directory(self, layer: str) -> Path:
        _validate_layered_media_layer(layer)
        return self._root / layer

    @property
    def _blob_directory(self) -> Path:
        return self._root / "blobs"

    @property
    def _staging_directory(self) -> Path:
        return self._root / ".staging"

    def _entry_path(self, layer: str, cache_key: str) -> Path:
        filename = f"{_validate_layered_media_key(cache_key)}{_LAYERED_MEDIA_ENTRY_SUFFIX}"
        return self._layer_directory(layer) / filename

    def _blob_path(self, content_sha256: str) -> Path:
        return self._blob_directory / f"{_validate_layered_media_key(content_sha256)}.blob"

    def _blob_state(
        self,
        path: Path,
        expected_sha256: str,
        expected_size: int | None = None,
    ) -> str:
        try:
            data = self._read_regular_bytes(path)
        except FileNotFoundError:
            return "MISSING"
        except OSError:
            return "CORRUPT"
        if expected_size is not None and len(data) != expected_size:
            return "CORRUPT"
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            return "CORRUPT"
        return "VISIBLE"

    def _read_regular_bytes(self, path: Path) -> bytes:
        self._assert_safe_path(path)
        file_stat = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
            raise OSError(f"cache path is not a regular file: {path}")
        return path.read_bytes()

    def _atomic_write(self, path: Path, data: bytes) -> None:
        self._assert_safe_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            descriptor, temp_path = make_temp_file(
                self._staging_directory,
                prefix=".layered-media-",
                suffix=".tmp",
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self._assert_safe_path(temp_path)
            temp_path.replace(path)
            self._fsync_directory(path.parent)
        except Exception:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            raise

    def _remove_paths(self, paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
        removed: list[Path] = []
        failed: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                self._assert_safe_path(path)
                file_stat = path.lstat()
                if stat.S_ISDIR(file_stat.st_mode):
                    path.rmdir()
                else:
                    path.unlink()
                self._fsync_directory(path.parent)
            except (FileNotFoundError, OSError, ValueError):
                failed.append(path)
            else:
                removed.append(path)
        return removed, failed

    def _assert_safe_path(self, path: Path) -> None:
        try:
            path.resolve(strict=False).relative_to(self._resolved_root)
        except (OSError, ValueError) as exc:
            raise OSError(f"cache path escapes root: {path}") from exc

    @staticmethod
    def _metadata_sha256(core: Mapping[str, object]) -> str:
        canonical = json.dumps(
            core,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_capacity(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if namespace in {".", ".."} or "\x00" in namespace or "/" in namespace or "\\" in namespace:
            raise ValueError("namespace must be a single safe path component")
        if Path(namespace).is_absolute() or Path(namespace).name != namespace:
            raise ValueError("namespace must be a single safe path component")


class SharedFrameCache:
    """Thread-safe content-addressed cache with per-video feed-once locking.

    The implementation is intentionally filesystem-only and has no cloud SDK dependency.  A
    production adapter can persist the same ``FrameRef`` URIs in R2 while retaining this API.
    """

    def __init__(self, root: str | os.PathLike[str], *, namespace: str = "frames-v1") -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be non-empty")
        self.root = Path(root)
        self.namespace = namespace.strip()
        self._root = self.root / self.namespace
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._video_locks: dict[str, threading.Lock] = {}
        self._manifests: dict[str, FrameFeedManifest] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._decode_attempts = 0

    def feed_once(
        self,
        video_id: str,
        source_uri: str,
        decoder: Callable[
            [], Iterable[FramePayload | bytes | Mapping[str, Any] | tuple[float, bytes]]
        ],
        *,
        frame_rate: float = 2.0,
    ) -> FeedOnceResult:
        """Decode and persist one video's frames at most once.

        Concurrent callers for the same video serialize on a per-video lock.  If a manifest
        already exists, the decoder is not invoked and ``cache_hit`` is true.
        """
        self._validate_video_id(video_id)
        self._validate_id(source_uri, "source_uri")
        if (
            isinstance(frame_rate, bool)
            or not isinstance(frame_rate, (int, float))
            or frame_rate <= 0
        ):
            raise ValueError("frame_rate must be positive")
        with self._lock:
            existing = self._manifests.get(video_id) or self._load_manifest_for_reuse(video_id)
            if existing is not None and existing.cache_key != self._cache_key(
                video_id,
                source_uri,
            ):
                raise ValueError("video_id is already bound to a different source URI")
            if existing is not None and not self._manifest_visible(existing):
                existing = None
            if existing is not None:
                self._manifests[video_id] = existing
                self._cache_hits += 1
                return FeedOnceResult(existing, True)
            lock = self._video_locks.setdefault(video_id, threading.Lock())
        with lock:
            with self._lock:
                existing = self._manifests.get(video_id) or self._load_manifest_for_reuse(video_id)
                if existing is not None and existing.cache_key != self._cache_key(
                    video_id,
                    source_uri,
                ):
                    raise ValueError("video_id is already bound to a different source URI")
                if existing is not None and not self._manifest_visible(existing):
                    existing = None
                if existing is not None:
                    self._manifests[video_id] = existing
                    self._cache_hits += 1
                    return FeedOnceResult(existing, True)
                self._cache_misses += 1
                self._decode_attempts += 1
            payloads = decoder()
            if payloads is None:
                raise ValueError("decoder must return an iterable")
            refs: list[FrameRef] = []
            for ordinal, raw in enumerate(payloads):
                payload = _coerce_payload(raw, default_timestamp=ordinal / float(frame_rate))
                refs.append(self.put_frame(video_id, ordinal, payload))
            if not refs:
                raise ValueError("decoder produced no frames")
            manifest = FrameFeedManifest(
                video_id=video_id,
                source_uri=source_uri,
                frame_rate=float(frame_rate),
                frames=tuple(refs),
                cache_key=self._cache_key(video_id, source_uri),
            )
            self._write_manifest(manifest)
            with self._lock:
                self._manifests[video_id] = manifest
            return FeedOnceResult(manifest, False)

    def put_frame(self, video_id: str, ordinal: int, payload: FramePayload) -> FrameRef:
        self._validate_video_id(video_id)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        if not isinstance(payload, FramePayload):
            payload = _coerce_payload(payload)
        digest = hashlib.sha256(payload.data).hexdigest()
        directory = self._root / video_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}.bin"
        if (
            not path.exists()
            or self._frame_path_state(path, digest, len(payload.data)) != "VISIBLE"
        ):
            self._atomic_write(path, payload.data)
        frame_id = payload.frame_id or f"{video_id}:{ordinal}:{digest[:16]}"
        return FrameRef(
            frame_id=frame_id,
            video_id=video_id,
            ordinal=ordinal,
            timestamp_sec=float(payload.timestamp_sec),
            uri=path.as_posix(),
            content_sha256=digest,
            size_bytes=len(payload.data),
        )

    def read_frame(self, frame: FrameRef) -> bytes:
        path = Path(frame.uri)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != frame.content_sha256:
            raise ValueError(f"frame content hash mismatch: {frame.frame_id}")
        return data

    def get_manifest(self, video_id: str) -> FrameFeedManifest | None:
        self._validate_video_id(video_id)
        with self._lock:
            manifest = self._manifests.get(video_id) or self._load_manifest(video_id)
            if manifest is not None:
                self._manifests[video_id] = manifest
            return manifest

    def stats(self) -> FrameCacheStats:
        with self._lock:
            frames = 0
            bytes_total = 0
            manifests = list(self._manifests.values())
            for manifest in manifests:
                frames += manifest.frame_count
                bytes_total += manifest.total_bytes
            return FrameCacheStats(
                frames, bytes_total, self._cache_hits, self._cache_misses, self._decode_attempts
            )

    def clear_video(self, video_id: str) -> None:
        """Remove derived frame artifacts for a video; source recordings remain untouched."""
        self._validate_video_id(video_id)
        with self._lock:
            self._manifests.pop(video_id, None)
            manifest_path = self._manifest_path(video_id)
            if manifest_path.exists():
                manifest_path.unlink()
        directory = self._root / video_id
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
            directory.rmdir()

    def reconcile(
        self,
        *,
        remove_orphans: bool = False,
        remove_partials: bool = False,
        strict: bool = False,
    ) -> FrameCacheReconciliation:
        """Reconcile persisted manifests, frame bytes, and crash leftovers.

        A manifest is visible only when every referenced frame remains a regular file
        with the recorded exact hash and size. Unreferenced files are retained by
        default so an operator can repair or inspect a failed publication.
        """

        for name, value in (
            ("remove_orphans", remove_orphans),
            ("remove_partials", remove_partials),
            ("strict", strict),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")

        manifest_paths = sorted(
            self._root.glob("*.manifest.json"),
            key=lambda path: path.as_posix(),
        )
        missing: list[str] = []
        corrupt: list[str] = []
        invalid_manifests: list[Path] = []
        referenced_paths: set[Path] = set()
        valid_manifests: dict[str, FrameFeedManifest] = {}
        visible_manifests = 0
        for manifest_path in manifest_paths:
            try:
                manifest = FrameFeedManifest.model_validate_json(
                    manifest_path.read_bytes(),
                    strict=True,
                )
            except Exception:
                invalid_manifests.append(manifest_path)
                continue
            expected_name = f"{manifest.video_id}.manifest.json"
            if manifest_path.name != expected_name:
                invalid_manifests.append(manifest_path)
                continue
            visible = True
            seen_ordinals: set[int] = set()
            seen_frame_ids: set[str] = set()
            for expected_ordinal, frame in enumerate(manifest.frames):
                frame_key = f"{manifest.video_id}:{frame.ordinal}"
                if (
                    frame.video_id != manifest.video_id
                    or frame.ordinal != expected_ordinal
                    or frame.ordinal in seen_ordinals
                    or frame.frame_id in seen_frame_ids
                ):
                    corrupt.append(frame_key)
                    visible = False
                    continue
                seen_ordinals.add(frame.ordinal)
                seen_frame_ids.add(frame.frame_id)
                path = self._safe_frame_path(frame.uri)
                if path is None:
                    corrupt.append(frame_key)
                    visible = False
                    continue
                referenced_paths.add(path)
                state = self._frame_path_state(path, frame.content_sha256, frame.size_bytes)
                if state == "MISSING":
                    missing.append(frame_key)
                    visible = False
                elif state != "VISIBLE":
                    corrupt.append(frame_key)
                    visible = False
            if not manifest.frames:
                invalid_manifests.append(manifest_path)
                visible = False
            valid_manifests[manifest.video_id] = manifest
            if visible:
                visible_manifests += 1

        orphan: list[Path] = []
        partial: list[Path] = []
        for video_directory in sorted(self._root.iterdir(), key=lambda path: path.as_posix()):
            if video_directory.name.endswith(".manifest.json"):
                continue
            try:
                directory_stat = video_directory.lstat()
            except OSError:
                partial.append(video_directory)
                continue
            if video_directory.is_symlink() or not stat.S_ISDIR(directory_stat.st_mode):
                partial.append(video_directory)
                continue
            for path in sorted(video_directory.rglob("*"), key=lambda value: value.as_posix()):
                try:
                    file_stat = path.lstat()
                except OSError:
                    partial.append(path)
                    continue
                if stat.S_ISDIR(file_stat.st_mode):
                    continue
                if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                    partial.append(path)
                elif self._safe_frame_path(path.as_posix()) not in referenced_paths:
                    if path.name.startswith("tmp") or path.name.startswith(".tmp"):
                        partial.append(path)
                    else:
                        orphan.append(path)

        cleanup: list[Path] = []
        if remove_orphans:
            cleanup.extend(orphan)
        if remove_partials:
            cleanup.extend(partial)
            cleanup.extend(invalid_manifests)
        removed, failed = self._remove_paths(cleanup)
        removed_set = set(removed)
        orphan = [path for path in orphan if path not in removed_set]
        partial = [path for path in partial if path not in removed_set]
        invalid_manifests = [path for path in invalid_manifests if path not in removed_set]
        with self._lock:
            self._manifests.update(valid_manifests)
            for video_id, manifest in tuple(self._manifests.items()):
                if manifest.video_id not in valid_manifests:
                    self._manifests.pop(video_id, None)
        report = FrameCacheReconciliation(
            manifest_count=len(manifest_paths),
            visible_manifest_count=visible_manifests,
            missing_frame_refs=tuple(sorted(set(missing))),
            corrupt_frame_refs=tuple(sorted(set(corrupt))),
            invalid_manifest_paths=tuple(
                sorted(invalid_manifests, key=lambda path: path.as_posix())
            ),
            orphan_blob_paths=tuple(sorted(orphan, key=lambda path: path.as_posix())),
            partial_blob_paths=tuple(sorted(partial, key=lambda path: path.as_posix())),
            removed_paths=tuple(sorted(removed, key=lambda path: path.as_posix())),
            failed_removal_paths=tuple(sorted(failed, key=lambda path: path.as_posix())),
        )
        if strict and not report.reconciled:
            raise ValueError("frame cache reconciliation found unresolved storage discrepancies")
        return report

    def reconcile_storage(self, **kwargs: object) -> FrameCacheReconciliation:
        """Compatibility alias for callers naming the backing store explicitly."""

        return self.reconcile(**kwargs)  # type: ignore[arg-type]

    def _safe_frame_path(self, uri: str) -> Path | None:
        try:
            candidate = Path(uri)
            resolved = candidate.resolve(strict=False)
            root = self._root.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        return resolved

    def _manifest_visible(self, manifest: FrameFeedManifest) -> bool:
        if not manifest.frames:
            return False
        seen_ordinals: set[int] = set()
        seen_frame_ids: set[str] = set()
        for expected_ordinal, frame in enumerate(manifest.frames):
            if (
                frame.video_id != manifest.video_id
                or frame.ordinal != expected_ordinal
                or frame.ordinal in seen_ordinals
                or frame.frame_id in seen_frame_ids
            ):
                return False
            seen_ordinals.add(frame.ordinal)
            seen_frame_ids.add(frame.frame_id)
            path = self._safe_frame_path(frame.uri)
            if (
                path is None
                or self._frame_path_state(
                    path,
                    frame.content_sha256,
                    frame.size_bytes,
                )
                != "VISIBLE"
            ):
                return False
        return True

    @staticmethod
    def _frame_path_state(path: Path, expected_sha256: str, expected_bytes: int) -> str:
        try:
            file_stat = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                return "CORRUPT"
            if file_stat.st_size != expected_bytes:
                return "CORRUPT"
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            return "VISIBLE" if digest.hexdigest() == expected_sha256 else "CORRUPT"
        except FileNotFoundError:
            return "MISSING"
        except OSError:
            return "CORRUPT"

    def _remove_paths(self, paths: list[Path]) -> tuple[list[Path], list[Path]]:
        removed: list[Path] = []
        failed: list[Path] = []
        seen: set[Path] = set()
        root = self._root.resolve(strict=True)
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                path.resolve(strict=False).relative_to(root)
                path.unlink()
            except (OSError, ValueError):
                failed.append(path)
            else:
                removed.append(path)
        return removed, failed

    def _cache_key(self, video_id: str, source_uri: str) -> str:
        return hashlib.sha256(f"{self.namespace}:{video_id}:{source_uri}".encode()).hexdigest()

    def _manifest_path(self, video_id: str) -> Path:
        return self._root / f"{video_id}.manifest.json"

    def _write_manifest(self, manifest: FrameFeedManifest) -> None:
        path = self._manifest_path(manifest.video_id)
        payload = manifest.model_dump_json().encode("utf-8")
        self._atomic_write(path, payload)

    def _load_manifest_for_reuse(self, video_id: str) -> FrameFeedManifest | None:
        try:
            return self._load_manifest(video_id)
        except ValueError:
            return None

    def _load_manifest(self, video_id: str) -> FrameFeedManifest | None:
        path = self._manifest_path(video_id)
        if not path.exists():
            return None
        try:
            return FrameFeedManifest.model_validate_json(path.read_bytes())
        except Exception as exc:  # pragma: no cover - corrupted cache recovery path
            raise ValueError(f"invalid frame cache manifest: {path}") from exc

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        temp_path.replace(path)

    @staticmethod
    def _validate_id(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    @staticmethod
    def _validate_video_id(value: str) -> None:
        """Keep caller-controlled video IDs inside the cache namespace."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("video_id must be a non-empty string")
        if value in {".", ".."} or "\x00" in value or "/" in value or "\\" in value:
            raise ValueError("video_id must be a single safe path component")
        candidate = Path(value)
        if candidate.is_absolute() or candidate.name != value:
            raise ValueError("video_id must be a single safe path component")


# Compatibility method names used by orchestration code.
SharedFrameCache.get_or_create = SharedFrameCache.feed_once  # type: ignore[attr-defined]
SharedFrameCache.get_or_feed = SharedFrameCache.feed_once  # type: ignore[attr-defined]
SharedFrameCache.put = SharedFrameCache.put_frame  # type: ignore[attr-defined]
SharedFrameCache.manifest_for = SharedFrameCache.get_manifest  # type: ignore[attr-defined]

FrameCache = SharedFrameCache
FrameFeedCoordinator = SharedFrameCache


def _coerce_payload(
    raw: FramePayload | bytes | Mapping[str, Any] | tuple[float, bytes],
    *,
    default_timestamp: float = 0.0,
) -> FramePayload:
    if isinstance(raw, FramePayload):
        return raw
    if isinstance(raw, bytes):
        # Bytes-only decoders get a deterministic timestamp from the caller's ordinal/frame rate.
        return FramePayload(timestamp_sec=default_timestamp, data=raw)
    if isinstance(raw, Mapping):
        return FramePayload(
            timestamp_sec=float(raw["timestamp_sec"]),
            data=bytes(raw["data"]),
            frame_id=raw.get("frame_id"),
        )
    if isinstance(raw, tuple) and len(raw) == 2:
        return FramePayload(timestamp_sec=float(raw[0]), data=bytes(raw[1]))
    raise TypeError("decoder yielded unsupported frame payload")


__all__ = [
    "FeedOnceResult",
    "FrameCache",
    "FrameCacheCapacityEstimate",
    "FrameCacheReconciliation",
    "FrameCacheStats",
    "FrameFeedCoordinator",
    "FrameFeedManifest",
    "FramePayload",
    "FrameRef",
    "LayeredMediaCache",
    "LayeredMediaCacheEntry",
    "LayeredMediaCacheReconciliation",
    "LayeredMediaCacheStats",
    "SharedFrameCache",
    "encoded_artifact_cache_key",
    "manifest_cache_key",
    "raw_frame_cache_key",
]
