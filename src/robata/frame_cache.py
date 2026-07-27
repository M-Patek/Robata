"""Shared local frame cache and feed-once coordination.

QA is the first stage that decodes footage.  This module provides a small provider-neutral cache
that can be replaced by R2/object storage in production while preserving the same manifest and
idempotency semantics.  Annotation consumes the manifest instead of decoding the source again.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any

from pydantic import Field, StringConstraints

from robata.contracts.common import Sha256Digest, StrictModel

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
    '''A bounded visibility and crash-leftover report for one local frame cache.'''

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
                raise ValueError('video_id is already bound to a different source URI')
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
                    raise ValueError('video_id is already bound to a different source URI')
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
            or self._frame_path_state(path, digest, len(payload.data)) != 'VISIBLE'
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
        '''Reconcile persisted manifests, frame bytes, and crash leftovers.

        A manifest is visible only when every referenced frame remains a regular file
        with the recorded exact hash and size. Unreferenced files are retained by
        default so an operator can repair or inspect a failed publication.
        '''

        for name, value in (
            ('remove_orphans', remove_orphans),
            ('remove_partials', remove_partials),
            ('strict', strict),
        ):
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a boolean')

        manifest_paths = sorted(
            self._root.glob('*.manifest.json'),
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
            expected_name = f'{manifest.video_id}.manifest.json'
            if manifest_path.name != expected_name:
                invalid_manifests.append(manifest_path)
                continue
            visible = True
            seen_ordinals: set[int] = set()
            seen_frame_ids: set[str] = set()
            for expected_ordinal, frame in enumerate(manifest.frames):
                frame_key = f'{manifest.video_id}:{frame.ordinal}'
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
                if state == 'MISSING':
                    missing.append(frame_key)
                    visible = False
                elif state != 'VISIBLE':
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
            if video_directory.name.endswith('.manifest.json'):
                continue
            try:
                directory_stat = video_directory.lstat()
            except OSError:
                partial.append(video_directory)
                continue
            if video_directory.is_symlink() or not stat.S_ISDIR(directory_stat.st_mode):
                partial.append(video_directory)
                continue
            for path in sorted(video_directory.rglob('*'), key=lambda value: value.as_posix()):
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
                    if path.name.startswith('tmp') or path.name.startswith('.tmp'):
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
            raise ValueError('frame cache reconciliation found unresolved storage discrepancies')
        return report

    def reconcile_storage(self, **kwargs: object) -> FrameCacheReconciliation:
        '''Compatibility alias for callers naming the backing store explicitly.'''

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
            if path is None or self._frame_path_state(
                path,
                frame.content_sha256,
                frame.size_bytes,
            ) != 'VISIBLE':
                return False
        return True

    @staticmethod
    def _frame_path_state(path: Path, expected_sha256: str, expected_bytes: int) -> str:
        try:
            file_stat = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                return 'CORRUPT'
            if file_stat.st_size != expected_bytes:
                return 'CORRUPT'
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            return 'VISIBLE' if digest.hexdigest() == expected_sha256 else 'CORRUPT'
        except FileNotFoundError:
            return 'MISSING'
        except OSError:
            return 'CORRUPT'

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
    'FrameCacheReconciliation',
    "FrameCacheStats",
    "FrameFeedCoordinator",
    "FrameFeedManifest",
    "FramePayload",
    "FrameRef",
    "SharedFrameCache",
]
