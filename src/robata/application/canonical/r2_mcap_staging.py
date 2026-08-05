"""Exact-byte staging of a pinned R2 MCAP object for a local canonical run.

The canonical MCAP reader currently consumes a local ``Path``.  This module is
the deliberately small bridge between an immutable R2 object and that local
reader: callers first load a canonical source manifest, then explicitly invoke
``stage_r2_mcap_source`` to fetch and verify the object.  It is not a production
composition root and does not make source staging an output-admission authority.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.object_storage import ObjectLocator, ObjectVisibility
from robata.ports.object_storage import ObjectStore, ObjectStoreError

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

R2_MCAP_SOURCE_MANIFEST_VERSION: Literal["robata-r2-mcap-source-v1"] = "robata-r2-mcap-source-v1"


class R2McapSourceStagingError(ValueError):
    """A pinned R2 source cannot be loaded or staged safely."""


class R2McapSourceManifest(StrictModel):
    """Non-schema local launch input binding one R2 object to exact MCAP bytes."""

    format_version: Literal["robata-r2-mcap-source-v1"] = R2_MCAP_SOURCE_MANIFEST_VERSION
    locator: ObjectLocator
    expected_sha256: Sha256Digest
    expected_byte_count: PositiveInt
    expected_media_type: NonEmptyString

    @model_validator(mode="after")
    def validate_r2_source(self) -> R2McapSourceManifest:
        parsed = urlsplit(self.locator.uri)
        if parsed.scheme != "r2" or not parsed.netloc or not parsed.path.startswith("/"):
            raise ValueError("source locator must be an r2:// object locator")
        if parsed.query or parsed.fragment:
            raise ValueError("source locator cannot include a query or fragment")
        if self.locator.presigned_url is not None:
            raise ValueError("source locator cannot include a presigned URL")
        if self.expected_media_type != self.expected_media_type.strip():
            raise ValueError("expected_media_type must not start or end with whitespace")
        return self


@dataclass(frozen=True, slots=True)
class R2McapSourceStageReceipt:
    """Verified result of staging one R2 source into a caller-owned directory."""

    manifest: R2McapSourceManifest
    destination: Path
    byte_count: int
    content_sha256: str
    reused_existing_file: bool


def load_r2_mcap_source_manifest(path: str | Path) -> R2McapSourceManifest:
    """Load a strict, exact-canonical source manifest from a local file."""

    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or pathlib.Path")
    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise R2McapSourceStagingError("source manifest must be a regular file")
    try:
        raw = source_path.read_bytes()
    except OSError as error:
        raise R2McapSourceStagingError(f"cannot read source manifest: {error}") from error
    return parse_r2_mcap_source_manifest_bytes(raw)


def parse_r2_mcap_source_manifest_bytes(raw: bytes) -> R2McapSourceManifest:
    """Validate immutable source-manifest bytes without reading any external target."""

    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise R2McapSourceStagingError(f"invalid source manifest JSON: {error}") from error
    if not isinstance(document, dict):
        raise R2McapSourceStagingError("source manifest root must be an object")
    if canonical_json_bytes(document) != raw:
        raise R2McapSourceStagingError("source manifest must use exact canonical JSON bytes")
    try:
        manifest = R2McapSourceManifest.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise R2McapSourceStagingError(f"invalid source manifest: {error}") from error
    if canonical_json_bytes(manifest) != raw:
        raise R2McapSourceStagingError("source manifest bytes are inconsistent with its model")
    return manifest


def stage_r2_mcap_source(
    *,
    manifest: R2McapSourceManifest,
    object_store: ObjectStore,
    destination: str | Path,
) -> R2McapSourceStageReceipt:
    """Fetch one R2 MCAP only after its metadata and exact bytes are verified.

    The destination is immutable at this boundary.  A pre-existing regular file
    is accepted only when it has the exact source bytes; a differing file is not
    overwritten.  This keeps retries safe and prevents a source-path typo from
    destroying unrelated local state.
    """

    if not isinstance(manifest, R2McapSourceManifest):
        raise TypeError("manifest must be R2McapSourceManifest")
    _require_object_store(object_store)
    if not isinstance(destination, (str, Path)):
        raise TypeError("destination must be a str or pathlib.Path")
    destination_path = Path(destination)
    _ensure_destination_parent(destination_path)

    try:
        head = object_store.head(manifest.locator)
    except ObjectStoreError:
        raise
    except Exception as error:
        raise R2McapSourceStagingError(
            "source object HEAD raised an unexpected exception"
        ) from error
    if head.visibility is not ObjectVisibility.VISIBLE:
        raise R2McapSourceStagingError(
            f"source object is not visibly readable: {head.visibility.value}"
        )
    if (
        head.sha256 != manifest.expected_sha256
        or head.byte_count != manifest.expected_byte_count
        or head.media_type != manifest.expected_media_type
    ):
        raise R2McapSourceStagingError(
            "source object metadata differs from the immutable source manifest"
        )

    try:
        payload = object_store.get(manifest.locator)
    except ObjectStoreError:
        raise
    except Exception as error:
        raise R2McapSourceStagingError(
            "source object GET raised an unexpected exception"
        ) from error
    _require_exact_payload(payload, manifest)
    reused = _publish_exact_file(destination_path, payload, manifest.expected_sha256)
    return R2McapSourceStageReceipt(
        manifest=manifest,
        destination=destination_path,
        byte_count=len(payload),
        content_sha256=manifest.expected_sha256,
        reused_existing_file=reused,
    )


def r2_mcap_source_manifest_projection(
    manifest: R2McapSourceManifest,
) -> dict[str, object]:
    """Return a safe operator projection with no credential material."""

    if not isinstance(manifest, R2McapSourceManifest):
        raise TypeError("manifest must be R2McapSourceManifest")
    return {
        "format_version": manifest.format_version,
        "locator": manifest.locator.model_dump(mode="json"),
        "expected_sha256": manifest.expected_sha256,
        "expected_byte_count": manifest.expected_byte_count,
        "expected_media_type": manifest.expected_media_type,
    }


def _require_object_store(value: object) -> None:
    for method in ("head", "get"):
        if not callable(getattr(value, method, None)):
            raise TypeError("object_store must provide head() and get()")


def _require_exact_payload(payload: object, manifest: R2McapSourceManifest) -> None:
    if not isinstance(payload, bytes):
        raise R2McapSourceStagingError("source object GET must return bytes")
    if len(payload) != manifest.expected_byte_count:
        raise R2McapSourceStagingError("source object byte count differs from the source manifest")
    if exact_bytes_sha256(payload) != manifest.expected_sha256:
        raise R2McapSourceStagingError("source object SHA-256 differs from the source manifest")


def _ensure_destination_parent(destination: Path) -> None:
    if destination.name in {"", ".", ".."}:
        raise R2McapSourceStagingError("destination must name a regular file")
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise R2McapSourceStagingError(f"cannot create staging directory: {error}") from error
    for ancestor in (parent.absolute(), *parent.absolute().parents):
        if ancestor.is_symlink():
            raise R2McapSourceStagingError("staging directory cannot traverse a symbolic link")
    if not parent.is_dir():
        raise R2McapSourceStagingError("staging directory must be a regular directory")
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise R2McapSourceStagingError("destination must be a regular file when it already exists")


def _publish_exact_file(destination: Path, payload: bytes, expected_sha256: str) -> bool:
    """Atomically create an exact file, or accept an identical pre-existing one."""

    descriptor, temporary = mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            _require_identical_existing_file(destination, expected_sha256, len(payload))
            return True
        except OSError as error:
            raise R2McapSourceStagingError(f"cannot publish staged source: {error}") from error
        return False
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_identical_existing_file(
    destination: Path, expected_sha256: str, expected_size: int
) -> None:
    if destination.is_symlink() or not destination.is_file():
        raise R2McapSourceStagingError("existing destination must be a regular file")
    try:
        existing = destination.read_bytes()
    except OSError as error:
        raise R2McapSourceStagingError(f"cannot read existing staged source: {error}") from error
    if len(existing) != expected_size or exact_bytes_sha256(existing) != expected_sha256:
        raise R2McapSourceStagingError(
            "destination already contains different bytes and will not be overwritten"
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


__all__ = [
    "R2_MCAP_SOURCE_MANIFEST_VERSION",
    "R2McapSourceManifest",
    "R2McapSourceStageReceipt",
    "R2McapSourceStagingError",
    "load_r2_mcap_source_manifest",
    "parse_r2_mcap_source_manifest_bytes",
    "r2_mcap_source_manifest_projection",
    "stage_r2_mcap_source",
]
