"""Source-inspection and decoder-probe boundaries.

These values intentionally describe observed source facts. They do not assert that a
recording is admitted or READY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from robata.contracts import Sha256Digest, SixCameraMap

COMPRESSED_IMAGE_SCHEMA = "foxglove.CompressedImage"


class IngestionErrorCode(StrEnum):
    """Stable error codes at the ingestion adapter boundary."""

    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_IO_ERROR = "SOURCE_IO_ERROR"
    CORRUPT_MCAP = "CORRUPT_MCAP"
    INVALID_CAMERA_MAPPING = "INVALID_CAMERA_MAPPING"
    UNSUPPORTED_CODEC = "UNSUPPORTED_CODEC"
    MISSING_TIMESTAMPS = "MISSING_TIMESTAMPS"
    DECODER_PROBE_FAILED = "DECODER_PROBE_FAILED"


class IngestionError(RuntimeError):
    """An ingestion failure carrying a stable machine-readable code."""

    def __init__(self, code: IngestionErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ChannelInspection:
    """Facts observed by scanning one MCAP channel."""

    channel_id: int
    topic: str
    schema_name: str | None
    message_encoding: str
    message_count: int
    first_message_time_ns: int | None
    last_message_time_ns: int | None
    monotonic: bool
    codec: str | None
    frame_id: str | None

    @property
    def schema(self) -> str | None:
        """Short alias useful in serialized inspection views."""

        return self.schema_name

    @property
    def count(self) -> int:
        return self.message_count


@dataclass(frozen=True, slots=True)
class McapInspection:
    """Container/header facts plus the complete observed channel inventory."""

    source: Path
    source_size_bytes: int
    source_sha256: Sha256Digest
    header_profile: str
    header_library: str
    summary_available: bool
    channel_count: int
    message_count: int
    first_message_time_ns: int | None
    last_message_time_ns: int | None
    channels: tuple[ChannelInspection, ...]

    @property
    def profile(self) -> str:
        return self.header_profile

    @property
    def library(self) -> str:
        return self.header_library

    def channels_for_topic(self, topic: str) -> tuple[ChannelInspection, ...]:
        return tuple(channel for channel in self.channels if channel.topic == topic)


@dataclass(frozen=True, slots=True)
class DecodeFailure:
    """One payload that an implemented decoder path could not decode."""

    code: str
    timestamp_ns: int | None
    message: str


@dataclass(frozen=True, slots=True)
class DecoderProbeResult:
    """Evidence produced by decoding source payload bytes, not metadata."""

    topic: str
    codec: str
    success: bool
    width: int | None
    height: int | None
    first_decoded_timestamp_ns: int | None
    messages_examined: int
    decoded_frames: int
    failures: tuple[DecodeFailure, ...]

    @property
    def failure_count(self) -> int:
        return len(self.failures)


class McapInspector(Protocol):
    def inspect(self, source: Path) -> McapInspection:
        """Read and scan a local MCAP source."""


class CameraMappingPolicy(Protocol):
    def resolve(self, inspection: McapInspection) -> SixCameraMap[ChannelInspection]:
        """Resolve observed channels to the six canonical camera slots."""


class DecoderProbe(Protocol):
    def probe(self, source: Path, channel: ChannelInspection) -> DecoderProbeResult:
        """Exercise a real decoder implementation against one observed channel."""
