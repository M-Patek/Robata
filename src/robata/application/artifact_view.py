"""Verified filesystem views materialized from immutable registry blobs."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import stat
import sys
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Never

from pydantic import ValidationError

from robata.contracts.artifacts import (
    ArtifactParent,
    ArtifactParentRelation,
    ArtifactRegistryEntry,
    ArtifactRegistrySnapshot,
    ArtifactType,
)
from robata.contracts.hashing import CanonicalizationError, canonical_json_bytes
from robata.contracts.video_export_v2 import CameraVideoExportManifestV2
from robata.ports.artifact_registry import (
    ArtifactRegistry,
    ArtifactRegistryError,
    ArtifactRegistryErrorCode,
)
from robata.tempfiles import make_staging_directory

_CHUNK_SIZE = 1024 * 1024
_MANIFEST_FILENAME = "camera-video-export-manifest.json"


class ArtifactViewErrorCode(StrEnum):
    """Stable machine-readable failures at the artifact-view boundary."""

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
    MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
    REGISTRY_BLOB_UNAVAILABLE = "REGISTRY_BLOB_UNAVAILABLE"
    REGISTRY_BLOB_INVALID = "REGISTRY_BLOB_INVALID"
    EXISTING_VIEW_INVALID = "EXISTING_VIEW_INVALID"
    OUTPUT_IO_ERROR = "OUTPUT_IO_ERROR"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    ATOMIC_PUBLISH_FAILED = "ATOMIC_PUBLISH_FAILED"


class ArtifactViewError(RuntimeError):
    """A view-materialization failure carrying a stable error code."""

    def __init__(self, code: ArtifactViewErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CameraVideoViewPublication:
    """Location of a complete verified view and whether it already existed."""

    output_directory: Path
    reused: bool


@dataclass(frozen=True, slots=True)
class _ViewArtifact:
    filename: str
    entry: ArtifactRegistryEntry
    exact_bytes: bytes | None = None


class _FileValidationError(RuntimeError):
    pass


def _raise(code: ArtifactViewErrorCode, message: str) -> Never:
    raise ArtifactViewError(code, message)


def _revalidate_snapshot(snapshot: object) -> ArtifactRegistrySnapshot:
    if not isinstance(snapshot, ArtifactRegistrySnapshot):
        _raise(ArtifactViewErrorCode.INVALID_SNAPSHOT, "snapshot has the wrong type")
    try:
        return ArtifactRegistrySnapshot.model_validate(
            snapshot.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as error:
        _raise(ArtifactViewErrorCode.INVALID_SNAPSHOT, f"snapshot is invalid: {error}")


def _revalidate_manifest(manifest: object) -> CameraVideoExportManifestV2:
    if not isinstance(manifest, CameraVideoExportManifestV2):
        _raise(ArtifactViewErrorCode.MANIFEST_MISMATCH, "manifest has the wrong type")
    try:
        return CameraVideoExportManifestV2.model_validate(
            manifest.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError as error:
        _raise(ArtifactViewErrorCode.MANIFEST_MISMATCH, f"manifest is invalid: {error}")


def _manifest_parents(manifest: CameraVideoExportManifestV2) -> tuple[ArtifactParent, ...]:
    parents = [
        ArtifactParent(
            artifact_id=manifest.export_config_artifact_id,
            relation=ArtifactParentRelation.EXPORT_CONFIG,
        ),
        ArtifactParent(
            artifact_id=manifest.mapping_profile_artifact_id,
            relation=ArtifactParentRelation.MAPPING_PROFILE,
        ),
        ArtifactParent(
            artifact_id=manifest.source_artifact_id,
            relation=ArtifactParentRelation.SOURCE_CONTENT,
        ),
    ]
    for camera in manifest.cameras:
        parents.extend(
            (
                ArtifactParent(
                    artifact_id=camera.timestamp_sidecar_artifact.artifact.artifact_id,
                    relation=ArtifactParentRelation.TIMESTAMP_OUTPUT,
                ),
                ArtifactParent(
                    artifact_id=camera.video_artifact.artifact_id,
                    relation=ArtifactParentRelation.VIDEO_OUTPUT,
                ),
            )
        )
    return tuple(sorted(parents, key=lambda item: (item.relation.value, item.artifact_id)))


def _build_view_artifacts(
    *,
    snapshot: ArtifactRegistrySnapshot,
    manifest_artifact_id: str,
    manifest: CameraVideoExportManifestV2,
) -> tuple[_ViewArtifact, ...]:
    entries_by_id = {entry.artifact_id: entry for entry in snapshot.entries}
    manifest_entry = entries_by_id.get(manifest_artifact_id)
    if manifest_entry is None:
        _raise(
            ArtifactViewErrorCode.MANIFEST_MISMATCH,
            f"manifest artifact {manifest_artifact_id} is absent from the snapshot",
        )
    if manifest_entry.artifact_type is not ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST:
        _raise(
            ArtifactViewErrorCode.MANIFEST_MISMATCH,
            "manifest_artifact_id does not identify a camera-video manifest",
        )

    manifest_entries = tuple(
        entry
        for entry in snapshot.entries
        if entry.artifact_type is ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST
    )
    if manifest_entries != (manifest_entry,):
        _raise(
            ArtifactViewErrorCode.MANIFEST_MISMATCH,
            "snapshot must contain exactly the selected camera-video manifest",
        )

    try:
        manifest_bytes = canonical_json_bytes(manifest)
    except CanonicalizationError as error:
        _raise(
            ArtifactViewErrorCode.MANIFEST_MISMATCH,
            f"manifest cannot be canonically serialized: {error}",
        )

    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        manifest_entry.sha256 != manifest_digest
        or manifest_entry.bytes != len(manifest_bytes)
        or manifest_entry.semantic_sha256 != manifest.semantic_content_sha256
        or manifest_entry.payload_schema_ref != manifest.schema_ref
        or manifest_entry.parents != _manifest_parents(manifest)
    ):
        _raise(
            ArtifactViewErrorCode.MANIFEST_MISMATCH,
            "manifest bytes, semantic identity, schema, or lineage disagree with the registry",
        )

    artifacts: list[_ViewArtifact] = [
        _ViewArtifact(
            filename=_MANIFEST_FILENAME,
            entry=manifest_entry,
            exact_bytes=manifest_bytes,
        )
    ]
    for camera in manifest.cameras:
        camera_id = camera.camera_id.value
        embedded_entries = (
            camera.video_artifact,
            camera.timestamp_sidecar_artifact.artifact,
        )
        for embedded in embedded_entries:
            if entries_by_id.get(embedded.artifact_id) != embedded:
                _raise(
                    ArtifactViewErrorCode.MANIFEST_MISMATCH,
                    f"embedded artifact {embedded.artifact_id} disagrees with the snapshot",
                )
        artifacts.extend(
            (
                _ViewArtifact(filename=f"{camera_id}.mp4", entry=camera.video_artifact),
                _ViewArtifact(
                    filename=f"{camera_id}.timestamps.jsonl",
                    entry=camera.timestamp_sidecar_artifact.artifact,
                ),
            )
        )

    if len(artifacts) != 13 or len({item.filename for item in artifacts}) != 13:
        _raise(
            ArtifactViewErrorCode.MANIFEST_MISMATCH,
            "camera-video view must contain exactly 13 uniquely named artifacts",
        )
    return tuple(artifacts)


def _open_source(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    source = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
        source.close()
        raise _FileValidationError(f"{path} is not a regular file")
    return source


def _file_facts(path: Path, *, capture_bytes: bool) -> tuple[str, int, bytes | None]:
    if path.is_symlink():
        raise _FileValidationError(f"{path} is a symbolic link")
    digest = hashlib.sha256()
    size = 0
    captured = bytearray() if capture_bytes else None
    with _open_source(path) as source:
        while chunk := source.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
            if captured is not None:
                captured.extend(chunk)
    return digest.hexdigest(), size, None if captured is None else bytes(captured)


def _validate_file(path: Path, artifact: _ViewArtifact) -> None:
    digest, size, exact_bytes = _file_facts(
        path,
        capture_bytes=artifact.exact_bytes is not None,
    )
    if size != artifact.entry.bytes:
        raise _FileValidationError(f"{path} has {size} bytes; expected {artifact.entry.bytes}")
    if digest != artifact.entry.sha256:
        raise _FileValidationError(f"{path} has digest {digest}; expected {artifact.entry.sha256}")
    if artifact.exact_bytes is not None and exact_bytes != artifact.exact_bytes:
        raise _FileValidationError(f"{path} does not contain the exact canonical manifest")


def _resolve_registry_sources(
    registry: ArtifactRegistry,
    artifacts: tuple[_ViewArtifact, ...],
) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    unavailable_codes = {
        ArtifactRegistryErrorCode.ARTIFACT_NOT_FOUND,
        ArtifactRegistryErrorCode.BLOB_SOURCE_MISSING,
        ArtifactRegistryErrorCode.DERIVATION_NOT_FOUND,
    }
    for artifact in artifacts:
        try:
            source = registry.resolve_blob(artifact.entry.artifact_id)
        except ArtifactRegistryError as error:
            code = (
                ArtifactViewErrorCode.REGISTRY_BLOB_UNAVAILABLE
                if error.code in unavailable_codes
                else ArtifactViewErrorCode.REGISTRY_BLOB_INVALID
            )
            _raise(code, f"cannot resolve registry blob {artifact.entry.artifact_id}: {error}")
        except (OSError, RuntimeError) as error:
            _raise(
                ArtifactViewErrorCode.REGISTRY_BLOB_UNAVAILABLE,
                f"cannot resolve registry blob {artifact.entry.artifact_id}: {error}",
            )
        if not isinstance(source, Path):
            _raise(
                ArtifactViewErrorCode.REGISTRY_BLOB_INVALID,
                f"registry returned a non-Path source for {artifact.entry.artifact_id}",
            )
        try:
            _validate_file(source, artifact)
        except FileNotFoundError as error:
            _raise(
                ArtifactViewErrorCode.REGISTRY_BLOB_UNAVAILABLE,
                f"registry blob {artifact.entry.artifact_id} is unavailable: {error}",
            )
        except (OSError, _FileValidationError) as error:
            _raise(
                ArtifactViewErrorCode.REGISTRY_BLOB_INVALID,
                f"registry blob {artifact.entry.artifact_id} is invalid: {error}",
            )
        sources[artifact.entry.artifact_id] = source
    return sources


def _validate_view(directory: Path, artifacts: tuple[_ViewArtifact, ...]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise _FileValidationError(f"{directory} is not a regular directory")
    expected_names = {artifact.filename for artifact in artifacts}
    actual_names = {child.name for child in directory.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise _FileValidationError(f"view layout mismatch; missing={missing}, extra={extra}")
    for artifact in artifacts:
        _validate_file(directory / artifact.filename, artifact)


def _copy_registry_blob(source: Path, destination: Path, artifact: _ViewArtifact) -> None:
    digest = hashlib.sha256()
    size = 0
    captured = bytearray() if artifact.exact_bytes is not None else None
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(destination, destination_flags, 0o600)
    try:
        with _open_source(source) as source_file, os.fdopen(descriptor, "wb") as output_file:
            descriptor = -1
            while chunk := source_file.read(_CHUNK_SIZE):
                output_file.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if captured is not None:
                    captured.extend(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if size != artifact.entry.bytes or digest.hexdigest() != artifact.entry.sha256:
        raise _FileValidationError(
            f"registry blob {artifact.entry.artifact_id} changed while being copied"
        )
    if captured is not None and bytes(captured) != artifact.exact_bytes:
        raise _FileValidationError("registry manifest changed while being copied")


def _materialize_registry_blob(
    source: Path,
    destination: Path,
    artifact: _ViewArtifact,
) -> None:
    """Prefer a same-filesystem hardlink and copy only when links are unsupported."""

    try:
        os.link(source, destination, follow_symlinks=False)
        return
    except OSError as error:
        unsupported = {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if error.errno not in unsupported:
            raise
    _copy_registry_blob(source, destination, artifact)


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any competing target."""

    if os.name == "nt":
        source.rename(target)
        return

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    result = -1
    if sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
        result = renameat2(-100, source_bytes, -100, target_bytes, 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOTSUP, "renamex_np is unavailable")
        result = renamex_np(source_bytes, target_bytes, 0x00000004)
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory rename is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), target)


def materialize_camera_video_view(
    *,
    registry: ArtifactRegistry,
    snapshot: ArtifactRegistrySnapshot,
    manifest_artifact_id: str,
    manifest: CameraVideoExportManifestV2,
    output_directory: Path,
) -> CameraVideoViewPublication:
    """Copy one complete V2 six-camera view from verified registry blobs."""

    if not callable(getattr(registry, "resolve_blob", None)):
        _raise(ArtifactViewErrorCode.INVALID_REQUEST, "registry must provide resolve_blob")
    if not isinstance(manifest_artifact_id, str) or not manifest_artifact_id:
        _raise(
            ArtifactViewErrorCode.INVALID_REQUEST,
            "manifest_artifact_id must be a nonempty string",
        )
    if not isinstance(output_directory, Path):
        _raise(ArtifactViewErrorCode.INVALID_REQUEST, "output_directory must be a Path")
    if output_directory.name in {"", ".", ".."}:
        _raise(
            ArtifactViewErrorCode.INVALID_REQUEST,
            "output_directory must name a child directory",
        )

    validated_snapshot = _revalidate_snapshot(snapshot)
    validated_manifest = _revalidate_manifest(manifest)
    artifacts = _build_view_artifacts(
        snapshot=validated_snapshot,
        manifest_artifact_id=manifest_artifact_id,
        manifest=validated_manifest,
    )
    sources = _resolve_registry_sources(registry, artifacts)
    try:
        target = Path(os.path.abspath(output_directory))
    except (OSError, ValueError) as error:
        _raise(
            ArtifactViewErrorCode.INVALID_REQUEST,
            f"output_directory is invalid: {error}",
        )

    if target.exists() or target.is_symlink():
        try:
            _validate_view(target, artifacts)
        except (OSError, _FileValidationError) as error:
            _raise(
                ArtifactViewErrorCode.EXISTING_VIEW_INVALID,
                f"existing output view is invalid: {error}",
            )
        return CameraVideoViewPublication(output_directory=target, reused=True)

    staging: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = make_staging_directory(target.parent, prefix=f".{target.name}.partial-")
        os.chmod(staging, 0o700)
    except OSError as error:
        if staging is not None:
            with suppress(OSError):
                shutil.rmtree(staging)
        _raise(
            ArtifactViewErrorCode.OUTPUT_IO_ERROR,
            f"cannot create private staging directory: {error}",
        )
    assert staging is not None

    published = False
    try:
        try:
            for artifact in artifacts:
                source = sources[artifact.entry.artifact_id]
                _materialize_registry_blob(source, staging / artifact.filename, artifact)
            _validate_view(staging, artifacts)
            _sync_directory(staging)
        except _FileValidationError as error:
            _raise(ArtifactViewErrorCode.REGISTRY_BLOB_INVALID, str(error))
        except OSError as error:
            _raise(ArtifactViewErrorCode.OUTPUT_IO_ERROR, f"cannot build output view: {error}")

        if target.exists() or target.is_symlink():
            _raise(
                ArtifactViewErrorCode.OUTPUT_EXISTS,
                "output appeared while the view was being materialized",
            )
        try:
            _rename_no_replace(staging, target)
        except OSError as error:
            if target.exists() or target.is_symlink():
                _raise(
                    ArtifactViewErrorCode.OUTPUT_EXISTS,
                    f"output appeared during atomic publication: {error}",
                )
            _raise(
                ArtifactViewErrorCode.ATOMIC_PUBLISH_FAILED,
                f"cannot atomically publish output view: {error}",
            )
        published = True
        try:
            _sync_directory(target.parent)
        except OSError as error:
            _raise(
                ArtifactViewErrorCode.OUTPUT_IO_ERROR,
                f"published output but could not sync its parent directory: {error}",
            )
    finally:
        if not published:
            with suppress(OSError):
                shutil.rmtree(staging)

    return CameraVideoViewPublication(output_directory=target, reused=False)


__all__ = [
    "ArtifactViewError",
    "ArtifactViewErrorCode",
    "CameraVideoViewPublication",
    "materialize_camera_video_view",
]
