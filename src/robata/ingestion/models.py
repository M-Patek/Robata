"""Ingestion domain models for stream indexing and camera mapping."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, StrictModel
from robata.contracts.logical_nodes import Rfc3339Timestamp
from robata.contracts.mcap import CameraMapping as McapCameraMapping
from robata.contracts.mcap import CameraMappingRun as McapCameraMappingRun

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class StreamIndex(StrictModel):
    """Immutable index of frames for one raw video stream."""

    stream_id: NonEmptyString
    mcap_id: NonEmptyString
    topic: NonEmptyString
    channel_id: NonNegativeInt
    codec: NonEmptyString
    width: PositiveInt
    height: PositiveInt
    nominal_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    source_start_ns: Nanoseconds
    source_end_ns: Nanoseconds
    frame_count: NonNegativeInt


class CameraMapping(McapCameraMapping):
    """One camera-to-stream mapping within a mapping run.

    Exactly six rows per published ``CameraMappingRun``.
    """


class CameraMappingRun(McapCameraMappingRun):
    """A versioned camera mapping for one MCAP.

    Published runs are immutable; exactly one is selected by a versioned decision.
    """


class SourceFrameIndex(StrictModel):
    """Immutable source identity for one frame within a video stream.

    ``frame_number`` is seek/order metadata only and must not be used for
    cross-camera temporal alignment.
    """

    frame_id: NonEmptyString
    stream_id: NonEmptyString
    source_timestamp_ns: Nanoseconds
    message_offset: NonNegativeInt
    message_sequence: NonNegativeInt
    frame_number: NonNegativeInt
    artifact_id: NonEmptyString | None = None


class IngestionResult(StrictModel):
    """Result of indexing and validating one MCAP recording."""

    mcap_id: NonEmptyString
    stream_index: tuple[StreamIndex, ...]
    camera_mapping_run: CameraMappingRun
    frame_indexes: tuple[SourceFrameIndex, ...]
    status: Literal["INDEXED", "FAILED"]
    error_code: NonEmptyString | None = None
    error_message: NonEmptyString | None = None
    indexed_at: Rfc3339Timestamp


__all__ = [
    "CameraMapping",
    "CameraMappingRun",
    "IngestionResult",
    "SourceFrameIndex",
    "StreamIndex",
]
