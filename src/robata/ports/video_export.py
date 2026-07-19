"""Camera-video export boundary and immutable observed export facts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from robata.contracts import CameraId, Sha256Digest
from robata.ports.ingestion import ChannelInspection


class VideoExportErrorCode(StrEnum):
    """Stable machine-readable failures at the camera-video export boundary."""

    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_IO_ERROR = "SOURCE_IO_ERROR"
    INVALID_CHANNEL = "INVALID_CHANNEL"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    UNSUPPORTED_CODEC = "UNSUPPORTED_CODEC"
    DESTINATION_EXISTS = "DESTINATION_EXISTS"
    INVALID_DESTINATION = "INVALID_DESTINATION"
    MCAP_READ_ERROR = "MCAP_READ_ERROR"
    NO_CAMERA_MESSAGES = "NO_CAMERA_MESSAGES"
    NONMONOTONIC_LOG_TIME = "NONMONOTONIC_LOG_TIME"
    INVALID_ACCESS_UNIT = "INVALID_ACCESS_UNIT"
    BOOTSTRAP_NOT_FOUND = "BOOTSTRAP_NOT_FOUND"
    INVALID_TIMESTAMP_METADATA = "INVALID_TIMESTAMP_METADATA"
    FRAME_REORDERING_UNSUPPORTED = "FRAME_REORDERING_UNSUPPORTED"
    REMUX_FAILED = "REMUX_FAILED"
    DECODE_VALIDATION_FAILED = "DECODE_VALIDATION_FAILED"
    ATOMIC_COMMIT_FAILED = "ATOMIC_COMMIT_FAILED"


class VideoExportError(RuntimeError):
    """A camera-video export failure with a stable error code."""

    def __init__(self, code: VideoExportErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ExportedCameraVideoFacts:
    """Observed facts for one committed MP4 and its packet timestamp ledger."""

    camera_id: CameraId
    channel_id: int
    topic: str
    video_path: Path
    sidecar_path: Path
    source_message_count: int
    leading_access_unit_count: int
    trailing_access_unit_count: int
    exported_packet_count: int
    decoded_frame_count: int
    keyframe_count: int
    width: int
    height: int
    source_first_log_time_ns: int
    source_last_log_time_ns: int
    leading_first_log_time_ns: int | None
    leading_last_log_time_ns: int | None
    trailing_first_log_time_ns: int | None
    trailing_last_log_time_ns: int | None
    export_first_source_log_time_ns: int
    export_last_source_log_time_ns: int
    first_pts_ns: int
    last_pts_ns: int
    duration_ns: int
    time_base_numerator: int
    time_base_denominator: int
    tail_duration_ns: int
    tail_duration_policy: str
    max_timestamp_mapping_error_ns: int
    video_size_bytes: int
    video_sha256: Sha256Digest
    sidecar_row_count: int
    sidecar_size_bytes: int
    sidecar_sha256: Sha256Digest


class CameraVideoExporter(Protocol):
    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> ExportedCameraVideoFacts:
        """Export one exact mapped camera channel to MP4 plus a timestamp ledger."""
