"""MCAP ingestion and native six-camera data model contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, SchemaVersion, StrictModel

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


class MCAPManifest(StrictModel):
    """Published manifest for an MCAP recording."""

    schema_version: Literal["1.0"]
    mcap_id: NonEmptyString
    source: dict[str, object]
    recording: dict[str, object]
    camera_count: Literal[6]
    camera_mapping_run_id: NonEmptyString
    camera_mapping_version: SchemaVersion
    cameras: tuple[CameraStream, ...]
    ingested_at: Rfc3339Timestamp
    status: Literal["READY"]


__all__ = [
    "CameraMapping",
    "CameraMappingRun",
    "CameraStream",
    "MCAPManifest",
    "MCAPRecording",
    "MCAPRecordingStatus",
    "TerminalErrorCode",
]
