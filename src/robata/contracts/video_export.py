"""Immutable provider-neutral contracts for six-camera MP4 export evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import (
    INT64_MAX,
    INT64_MIN,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
ArtifactUri = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=4,
        pattern=r"^[A-Za-z][A-Za-z0-9+.-]*://\S+$",
    ),
]
MediaType = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$",
    ),
]
UuidString = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
    ),
]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
Int64 = Annotated[int, Field(strict=True, ge=INT64_MIN, le=INT64_MAX)]


class VideoExportExecutionMode(StrEnum):
    """Admission context under which an export manifest may be published."""

    GOVERNED_READY = "GOVERNED_READY"
    LOCAL_DEVELOPMENT_OVERRIDE = "LOCAL_DEVELOPMENT_OVERRIDE"


class VideoExportAlignmentStatus(StrEnum):
    """The only alignment states relevant to an export derivation."""

    UNVERIFIED = "UNVERIFIED"
    VALID = "VALID"


class VideoExporterMode(StrEnum):
    """Whether compressed packets were remuxed or pixels were transcoded."""

    REMUX = "REMUX"
    TRANSCODE = "TRANSCODE"


class TailDurationPolicy(StrEnum):
    """How the final exported packet or frame duration was derived."""

    OBSERVED_NEXT_TIMESTAMP = "OBSERVED_NEXT_TIMESTAMP"
    MEDIAN_POSITIVE_INTERVAL = "MEDIAN_POSITIVE_INTERVAL"


class DroppedMessageReasonCode(StrEnum):
    """Stable reason codes for contiguous leading or trailing source drops."""

    NONE = "NONE"
    BEFORE_FIRST_DECODABLE_KEYFRAME = "BEFORE_FIRST_DECODABLE_KEYFRAME"
    AFTER_LAST_COMPLETE_SAMPLE = "AFTER_LAST_COMPLETE_SAMPLE"
    EXPORT_POLICY_EXCLUSION = "EXPORT_POLICY_EXCLUSION"


class MappingProfileReference(StrictModel):
    """Immutable identity and approval state of the selected camera mapping profile."""

    version: SchemaVersion
    digest: Sha256Digest
    approved: bool


class VideoExporterIdentity(StrictModel):
    """Exporter implementation and materialization behavior."""

    name: NonEmptyString
    version: SchemaVersion
    mode: VideoExporterMode
    export_profile_id: NonEmptyString
    profile_version: SchemaVersion
    canonical_config_sha256: Sha256Digest


class DerivedArtifact(StrictModel):
    """Content-addressed derived bytes; the URI is a locator, never identity."""

    uri: ArtifactUri
    sha256: Sha256Digest
    bytes: PositiveInt
    media_type: MediaType


class TimestampSidecarArtifact(DerivedArtifact):
    """Packet-to-source-time lineage stored alongside an exported MP4."""

    row_count: PositiveInt


class SourceVideoStream(StrictModel):
    """The immutable MCAP channel selected for one logical camera slot."""

    topic: NonEmptyString
    channel_id: NonNegativeInt
    schema_name: NonEmptyString
    codec: NonEmptyString


class DroppedMessageProvenance(StrictModel):
    """Count, stable reason, and source-time range for contiguous dropped messages."""

    count: NonNegativeInt
    reason_code: DroppedMessageReasonCode
    first_source_ns: Nanoseconds | None
    last_source_ns: Nanoseconds | None

    @model_validator(mode="after")
    def validate_drop_provenance(self) -> Self:
        if self.reason_code is DroppedMessageReasonCode.NONE:
            if (
                self.count != 0
                or self.first_source_ns is not None
                or self.last_source_ns is not None
            ):
                raise ValueError("NONE drops require zero count and null source timestamps")
            return self

        if self.count == 0:
            raise ValueError("non-NONE drops require a positive count")
        if self.first_source_ns is None or self.last_source_ns is None:
            raise ValueError("non-NONE drops require first and last source timestamps")
        if self.first_source_ns > self.last_source_ns:
            raise ValueError("drop first_source_ns must not exceed last_source_ns")
        return self


class MediaTimeMapping(StrictModel):
    """Exact mapping inputs between source nanoseconds and integer media PTS."""

    zero_source_ns: Nanoseconds
    time_base_numerator: PositiveInt
    time_base_denominator: PositiveInt
    first_pts: Int64
    last_pts: Int64
    last_duration: PositiveInt
    tail_duration_policy: TailDurationPolicy
    rounding: Literal["HALF_EVEN"]
    max_rounding_error_ns: Nanoseconds

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        if self.first_pts > self.last_pts:
            raise ValueError("first_pts must be less than or equal to last_pts")
        if self.last_pts + self.last_duration > INT64_MAX:
            raise ValueError("last_pts plus last_duration must fit in signed int64")
        if self.max_rounding_error_ns < 0:
            raise ValueError("max_rounding_error_ns must be nonnegative")
        return self


class CameraVideoTimestampRow(StrictModel):
    """One canonical NDJSON row mapping an MP4 packet to its source timestamps."""

    schema_version: Literal["1.0"]
    export_profile_id: NonEmptyString
    export_profile_version: SchemaVersion
    camera_id: CameraId
    packet_index: NonNegativeInt
    source_sequence: NonNegativeInt
    source_log_time_ns: Nanoseconds
    source_publish_time_ns: Nanoseconds
    embedded_header_time_ns: Nanoseconds
    relative_pts_ns: Nanoseconds
    relative_dts_ns: Nanoseconds
    duration_ns: Nanoseconds
    time_base_numerator: PositiveInt
    time_base_denominator: PositiveInt
    is_keyframe: bool
    duration_is_estimated: bool

    @model_validator(mode="after")
    def validate_timestamp_row(self) -> Self:
        if self.relative_pts_ns < 0 or self.relative_dts_ns < 0:
            raise ValueError("relative PTS and DTS nanoseconds must be nonnegative")
        if self.duration_ns <= 0:
            raise ValueError("duration_ns must be positive")
        if self.relative_pts_ns + self.duration_ns > INT64_MAX:
            raise ValueError("relative_pts_ns plus duration_ns must fit in signed int64")
        if self.time_base_numerator != 1 or self.time_base_denominator != 1_000_000_000:
            raise ValueError("nanosecond timestamp rows require a 1/1000000000 time base")
        return self


class CameraVideoExportRecord(StrictModel):
    """One camera's source observations, export accounting, and derived artifacts."""

    camera_id: CameraId
    source: SourceVideoStream
    input_message_count: PositiveInt
    source_first_observed_message_ns: Nanoseconds
    source_last_observed_message_ns: Nanoseconds
    export_first_observed_source_message_ns: Nanoseconds
    export_last_observed_source_message_ns: Nanoseconds
    leading_drops: DroppedMessageProvenance
    trailing_drops: DroppedMessageProvenance
    exported_packet_count: PositiveInt
    exported_frame_count: PositiveInt
    keyframe_count: NonNegativeInt
    width: PositiveInt
    height: PositiveInt
    video_artifact: DerivedArtifact
    timestamp_sidecar_artifact: TimestampSidecarArtifact
    media_time_mapping: MediaTimeMapping

    @model_validator(mode="after")
    def validate_export_accounting(self) -> Self:
        expected_input_count = (
            self.leading_drops.count + self.exported_packet_count + self.trailing_drops.count
        )
        if self.input_message_count != expected_input_count:
            raise ValueError(
                "input_message_count must equal leading drops, exported packets, and trailing drops"
            )
        if self.timestamp_sidecar_artifact.row_count != self.exported_packet_count:
            raise ValueError("timestamp sidecar row_count must equal exported_packet_count")
        if self.keyframe_count > self.exported_packet_count:
            raise ValueError("keyframe_count must not exceed exported_packet_count")
        if self.keyframe_count > self.exported_frame_count:
            raise ValueError("keyframe_count must not exceed exported_frame_count")

        source_first = self.source_first_observed_message_ns
        source_last = self.source_last_observed_message_ns
        export_first = self.export_first_observed_source_message_ns
        export_last = self.export_last_observed_source_message_ns
        if source_first > source_last:
            raise ValueError(
                "source_first_observed_message_ns must not exceed source_last_observed_message_ns"
            )
        if not source_first <= export_first <= export_last <= source_last:
            raise ValueError("export observations must be ordered within source observations")
        if not source_first <= self.media_time_mapping.zero_source_ns <= source_last:
            raise ValueError("media zero_source_ns must lie within source observations")
        if self.leading_drops.first_source_ns is not None:
            assert self.leading_drops.last_source_ns is not None
            if not (
                source_first
                <= self.leading_drops.first_source_ns
                <= self.leading_drops.last_source_ns
                < export_first
            ):
                raise ValueError("leading drop range must precede the export range")
        if self.trailing_drops.first_source_ns is not None:
            assert self.trailing_drops.last_source_ns is not None
            if not (
                export_last
                < self.trailing_drops.first_source_ns
                <= self.trailing_drops.last_source_ns
                <= source_last
            ):
                raise ValueError("trailing drop range must follow the export range")
        return self


class CameraVideoExportManifest(StrictModel):
    """Immutable six-camera MP4 export manifest, independent of any model provider."""

    schema_version: Literal["1.0"]
    execution_mode: VideoExportExecutionMode
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    source_size_bytes: PositiveInt
    mapping_profile: MappingProfileReference
    ready_manifest_id: UuidString | None
    alignment_id: UuidString | None
    alignment_status: VideoExportAlignmentStatus
    exporter: VideoExporterIdentity
    cameras: tuple[CameraVideoExportRecord, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        actual_camera_ids = tuple(camera.camera_id for camera in self.cameras)
        if actual_camera_ids != CAMERA_IDS:
            raise ValueError("cameras must contain cam_01 through cam_06 in canonical order")

        if self.execution_mode is VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE:
            if self.ready_manifest_id is not None:
                raise ValueError("local development override cannot reference a READY manifest")
            if self.mapping_profile.approved:
                raise ValueError("local development override cannot use an approved mapping")
            if self.alignment_status is not VideoExportAlignmentStatus.UNVERIFIED:
                raise ValueError("local development override alignment must be UNVERIFIED")
        else:
            if self.ready_manifest_id is None:
                raise ValueError("governed export requires a READY manifest")
            if not self.mapping_profile.approved:
                raise ValueError("governed export requires an approved mapping")

        if self.alignment_status is VideoExportAlignmentStatus.VALID and self.alignment_id is None:
            raise ValueError("VALID alignment requires alignment_id")
        return self


__all__ = [
    "CameraVideoExportManifest",
    "CameraVideoExportRecord",
    "CameraVideoTimestampRow",
    "DerivedArtifact",
    "DroppedMessageProvenance",
    "DroppedMessageReasonCode",
    "MappingProfileReference",
    "MediaTimeMapping",
    "SourceVideoStream",
    "TailDurationPolicy",
    "TimestampSidecarArtifact",
    "VideoExportAlignmentStatus",
    "VideoExportExecutionMode",
    "VideoExporterIdentity",
    "VideoExporterMode",
]
