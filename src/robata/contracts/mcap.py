"""MCAP ingestion and native six-camera data model contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_ID_VALUES, CameraId
from robata.contracts.common import (
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$",
    ),
]


class MCAPRecordingStatus(StrEnum):
    """Lifecycle states for an MCAP recording."""

    DISCOVERED = "DISCOVERED"
    HASHING = "HASHING"
    INSPECTING = "INSPECTING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    INVALID = "INVALID"
    ALIGNMENT_QUEUED = "ALIGNMENT_QUEUED"
    RETRY_WAIT = "RETRY_WAIT"
    FAILED = "FAILED"


class TerminalErrorCode(StrEnum):
    """Terminal error codes for MCAP validation."""

    INVALID_CAMERA_COUNT = "INVALID_CAMERA_COUNT"
    INVALID_CAMERA_MAPPING = "INVALID_CAMERA_MAPPING"
    CORRUPT_MCAP = "CORRUPT_MCAP"
    UNSUPPORTED_CODEC = "UNSUPPORTED_CODEC"
    MISSING_TIMESTAMPS = "MISSING_TIMESTAMPS"
    ZERO_DURATION = "ZERO_DURATION"


class CameraStream(StrictModel):
    """One raw video stream within an MCAP recording."""

    stream_id: NonEmptyString
    topic: NonEmptyString
    channel_id: NonNegativeInt
    codec: NonEmptyString
    width: PositiveInt
    height: PositiveInt
    nominal_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    source_start_ns: Nanoseconds
    source_end_ns: Nanoseconds
    frame_count: NonNegativeInt


class CameraMapping(StrictModel):
    """One camera-to-stream mapping within a mapping run."""

    camera_id: NonEmptyString
    role: NonEmptyString
    stream_id: NonEmptyString


class CameraMappingRun(StrictModel):
    """A versioned camera mapping for one MCAP."""

    mapping_run_id: NonEmptyString
    mcap_id: NonEmptyString
    mapping_policy_version: SchemaVersion
    status: Literal["PUBLISHED", "SUPERSEDED"]
    created_at: Rfc3339Timestamp
    cameras: tuple[CameraMapping, ...]


class MCAPRecording(StrictModel):
    """An immutable MCAP recording after ingestion."""

    mcap_id: NonEmptyString
    recording_identity: NonEmptyString
    source_artifact_id: NonEmptyString
    source_uri: NonEmptyString
    source_version: NonEmptyString
    content_sha256: NonEmptyString
    observed_size_bytes: NonNegativeInt
    start_utc: Rfc3339Timestamp | None = None
    end_utc: Rfc3339Timestamp | None = None
    duration_ns: Nanoseconds
    timebase: NonEmptyString
    camera_count: Annotated[int, Field(strict=True, ge=0, le=6)]
    raw_video_stream_count: NonNegativeInt = 0
    status: MCAPRecordingStatus
    error_code: TerminalErrorCode | None = None
    ingested_at: Rfc3339Timestamp


class MCAPReadySource(StrictModel):
    """Durable source facts embedded in a READY manifest."""

    uri: NonEmptyString
    version: NonEmptyString
    sha256: Sha256Digest
    bytes: NonNegativeInt


class MCAPValidationSource(StrictModel):
    """Verified source facts embedded in a validation report."""

    uri: NonEmptyString
    version: str | None
    sha256: Sha256Digest
    bytes: NonNegativeInt


class MCAPReadyRecording(StrictModel):
    """Recording-time facts admitted by a selected VALID report."""

    start_utc: Rfc3339Timestamp | None
    end_utc: Rfc3339Timestamp | None
    duration_ns: Nanoseconds
    timebase: NonEmptyString

    @model_validator(mode="after")
    def require_positive_duration(self) -> MCAPReadyRecording:
        if self.duration_ns <= 0:
            raise ValueError("READY recording duration_ns must be positive")
        return self


class MCAPReadyCamera(StrictModel):
    """One canonical camera row in the registered READY wire schema."""

    camera_id: CameraId
    role: NonEmptyString
    stream_id: OpaqueUuid
    topic: NonEmptyString
    channel_id: NonNegativeInt
    codec: NonEmptyString
    width: PositiveInt
    height: PositiveInt
    nominal_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    source_start_ns: Nanoseconds
    source_end_ns: Nanoseconds
    frame_count: NonNegativeInt

    @model_validator(mode="after")
    def require_nonempty_source_interval(self) -> MCAPReadyCamera:
        if self.source_start_ns >= self.source_end_ns:
            raise ValueError("camera source_start_ns must be less than source_end_ns")
        return self


class MCAPValidationVerdict(StrEnum):
    """Source-admission verdict carried by immutable validation evidence."""

    VALID = "VALID"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


class MCAPMappingPolicyReference(StrictModel):
    """Exact mapping-policy candidate used by a validation pass."""

    version: NonEmptyString
    digest: Sha256Digest


class MCAPValidationError(StrictModel):
    """Schema-compatible diagnostic in a validation report."""

    code: NonEmptyString
    message: NonEmptyString
    path: str | None
    camera_id: CameraId | None
    stream_id: OpaqueUuid | None


class MCAPValidationReport(StrictModel):
    """Immutable source-validation evidence, separate from READY publication."""

    schema_version: Literal["1.0"]
    validation_report_id: OpaqueUuid
    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    source: MCAPValidationSource
    mapping_policy: MCAPMappingPolicyReference
    verdict: MCAPValidationVerdict
    discovered_video_stream_count: NonNegativeInt
    mapped_camera_count: Annotated[int, Field(strict=True, ge=0, le=6)]
    errors: tuple[MCAPValidationError, ...]
    validated_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def require_verdict_evidence(self) -> MCAPValidationReport:
        if self.verdict is MCAPValidationVerdict.VALID:
            if self.mapped_camera_count != 6:
                raise ValueError("VALID validation report requires six mapped cameras")
            if self.errors:
                raise ValueError("VALID validation report cannot contain errors")
        elif not self.errors:
            raise ValueError("non-VALID validation report requires diagnostics")
        return self


class MCAPReadyManifest(StrictModel):
    """Published READY manifest selected from immutable VALID evidence."""

    schema_version: Literal["1.0"]
    mcap_id: OpaqueUuid
    validation_report_id: OpaqueUuid
    source: MCAPReadySource
    recording: MCAPReadyRecording
    camera_count: Literal[6]
    camera_mapping_run_id: OpaqueUuid
    camera_mapping_version: SchemaVersion
    cameras: Annotated[tuple[MCAPReadyCamera, ...], Field(min_length=6, max_length=6)]
    ingested_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def require_canonical_camera_order(self) -> MCAPReadyManifest:
        actual = tuple(camera.camera_id.value for camera in self.cameras)
        if actual != CAMERA_ID_VALUES:
            raise ValueError("READY cameras must be ordered cam_01 through cam_06")
        if len({camera.stream_id for camera in self.cameras}) != 6:
            raise ValueError("READY cameras must reference six distinct streams")
        return self


# Compatibility name for older callers. The wire semantics are READY-only.
MCAPManifest = MCAPReadyManifest


__all__ = [
    "CameraMapping",
    "CameraMappingRun",
    "CameraStream",
    "MCAPManifest",
    "MCAPMappingPolicyReference",
    "MCAPReadyCamera",
    "MCAPReadyManifest",
    "MCAPReadyRecording",
    "MCAPReadySource",
    "MCAPRecording",
    "MCAPRecordingStatus",
    "MCAPValidationError",
    "MCAPValidationReport",
    "MCAPValidationSource",
    "MCAPValidationVerdict",
    "TerminalErrorCode",
]
