"""Artifact-registry-aware V2 contract for six-camera MP4 export evidence."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.artifacts import (
    ArtifactParent,
    ArtifactParentRelation,
    ArtifactRegistryEntry,
    ArtifactType,
    SchemaArtifactReference,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.video_export import (
    DroppedMessageProvenance,
    MappingProfileReference,
    MediaTimeMapping,
    SourceVideoStream,
    UuidString,
    VideoExportAlignmentStatus,
    VideoExporterIdentity,
    VideoExportExecutionMode,
)

type NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
type PositiveInt = Annotated[int, Field(strict=True, ge=1)]
type NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class TimestampSidecarArtifactV2(StrictModel):
    """Registered timestamp-map artifact plus its packet-row cardinality."""

    artifact: ArtifactRegistryEntry
    row_count: PositiveInt

    @model_validator(mode="after")
    def validate_artifact_type(self) -> Self:
        if self.artifact.artifact_type is not ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP:
            raise ValueError("timestamp sidecar artifact must be CAMERA_VIDEO_TIMESTAMP_MAP")
        return self


class MappingProfileArtifactPayloadV2(StrictModel):
    """Canonical bytes registered for the selected exact topic-mapping profile."""

    schema_version: Literal["2.0"]
    profile_id: NonEmptyString
    version: NonEmptyString
    profile_kind: NonEmptyString
    approval_status: Literal["APPROVED", "UNAPPROVED"]
    approved: bool
    mapping_policy: Literal["EXACT_TOPIC"]
    required_schema: Literal["foxglove.CompressedImage"]
    topics: SixCameraMap[str]

    @model_validator(mode="after")
    def validate_mapping_profile(self) -> Self:
        if self.approved != (self.approval_status == "APPROVED"):
            raise ValueError("approved and approval_status are contradictory")
        topic_values = tuple(self.topics.values())
        if len(topic_values) != len(set(topic_values)):
            raise ValueError("mapping topics must be unique across camera slots")
        if any(not topic for topic in topic_values):
            raise ValueError("mapping topics must be nonempty")
        return self


class ExportConfigArtifactPayloadV2(StrictModel):
    """Canonical effective exporter identity and configuration artifact."""

    schema_version: Literal["2.0"]
    exporter: VideoExporterIdentity


class CameraVideoExportRecordV2(StrictModel):
    """One camera's V2 source facts, accounting, and registered artifacts."""

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
    video_artifact: ArtifactRegistryEntry
    timestamp_sidecar_artifact: TimestampSidecarArtifactV2
    media_time_mapping: MediaTimeMapping

    @model_validator(mode="after")
    def validate_export_accounting(self) -> Self:
        if self.video_artifact.artifact_type is not ArtifactType.CAMERA_VIDEO_MP4:
            raise ValueError("video artifact must be CAMERA_VIDEO_MP4")

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


class CameraVideoExportManifestV2(StrictModel):
    """V2 manifest body; its own artifact identity remains in the external registry."""

    schema_version: Literal["2.0"]
    schema_ref: SchemaArtifactReference
    semantic_content_sha256: Sha256Digest
    execution_mode: VideoExportExecutionMode
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    source_size_bytes: PositiveInt
    source_artifact_id: UuidString
    mapping_profile_artifact_id: UuidString
    export_config_artifact_id: UuidString
    mapping_profile: MappingProfileReference
    ready_manifest_id: UuidString | None
    ready_manifest_semantic_sha256: Sha256Digest | None
    alignment_id: UuidString | None
    alignment_semantic_sha256: Sha256Digest | None
    alignment_status: VideoExportAlignmentStatus
    exporter: VideoExporterIdentity
    cameras: tuple[CameraVideoExportRecordV2, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        actual_camera_ids = tuple(camera.camera_id for camera in self.cameras)
        if actual_camera_ids != CAMERA_IDS:
            raise ValueError("cameras must contain cam_01 through cam_06 in canonical order")

        input_ids = (
            self.source_artifact_id,
            self.mapping_profile_artifact_id,
            self.export_config_artifact_id,
        )
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("source, mapping, and export-config artifact IDs must be unique")

        expected_parents = tuple(
            sorted(
                (
                    ArtifactParent(
                        artifact_id=self.export_config_artifact_id,
                        relation=ArtifactParentRelation.EXPORT_CONFIG,
                    ),
                    ArtifactParent(
                        artifact_id=self.mapping_profile_artifact_id,
                        relation=ArtifactParentRelation.MAPPING_PROFILE,
                    ),
                    ArtifactParent(
                        artifact_id=self.source_artifact_id,
                        relation=ArtifactParentRelation.SOURCE_CONTENT,
                    ),
                ),
                key=lambda parent: (parent.relation.value, parent.artifact_id),
            )
        )
        output_ids: list[str] = []
        for camera in self.cameras:
            video = camera.video_artifact
            timestamp_map = camera.timestamp_sidecar_artifact.artifact
            if video.parents != expected_parents or timestamp_map.parents != expected_parents:
                raise ValueError(
                    f"{camera.camera_id.value} artifacts must reference exact source, "
                    "mapping, and export-config parents"
                )
            output_ids.extend((video.artifact_id, timestamp_map.artifact_id))
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("camera output artifact IDs must be unique")
        if set(input_ids).intersection(output_ids):
            raise ValueError("input and output artifact IDs must be disjoint")

        if self.execution_mode is VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE:
            if (
                self.ready_manifest_id is not None
                or self.ready_manifest_semantic_sha256 is not None
            ):
                raise ValueError("local development override cannot reference READY evidence")
            if self.mapping_profile.approved:
                raise ValueError("local development override cannot use an approved mapping")
            if self.alignment_status is not VideoExportAlignmentStatus.UNVERIFIED:
                raise ValueError("local development override alignment must be UNVERIFIED")
            if self.alignment_id is not None or self.alignment_semantic_sha256 is not None:
                raise ValueError("local development override cannot reference alignment evidence")
        else:
            if self.ready_manifest_id is None or self.ready_manifest_semantic_sha256 is None:
                raise ValueError(
                    "governed export requires READY artifact identity and semantic digest"
                )
            if not self.mapping_profile.approved:
                raise ValueError("governed export requires an approved mapping")

        if self.alignment_status is VideoExportAlignmentStatus.VALID:
            if self.alignment_id is None or self.alignment_semantic_sha256 is None:
                raise ValueError("VALID alignment requires artifact identity and semantic digest")
        elif self.alignment_id is not None or self.alignment_semantic_sha256 is not None:
            raise ValueError("UNVERIFIED alignment cannot reference alignment evidence")

        expected_digest = semantic_sha256(camera_video_manifest_v2_semantic_projection(self))
        if self.semantic_content_sha256 != expected_digest:
            raise ValueError("semantic_content_sha256 does not match the canonical V2 projection")
        return self


def artifact_entry_semantic_projection(entry: ArtifactRegistryEntry) -> dict[str, Any]:
    """Project stable artifact content facts without opaque publication metadata."""

    schema_ref = entry.payload_schema_ref
    return {
        "artifact_type": entry.artifact_type.value,
        "semantic_sha256": entry.semantic_sha256,
        "sha256": entry.sha256,
        "bytes": entry.bytes,
        "media_type": entry.media_type,
        "producer": entry.producer.model_dump(mode="json"),
        "payload_schema_ref": (
            None
            if schema_ref is None
            else {
                "schema_id": schema_ref.schema_id,
                "version": schema_ref.version,
                "sha256": schema_ref.sha256,
            }
        ),
    }


def camera_video_manifest_v2_semantic_projection(
    manifest: CameraVideoExportManifestV2,
) -> dict[str, Any]:
    """Return the run- and locator-independent V2 manifest semantic projection."""

    cameras: list[dict[str, Any]] = []
    for camera in manifest.cameras:
        record = camera.model_dump(
            mode="json",
            exclude={"video_artifact", "timestamp_sidecar_artifact"},
        )
        record["video_artifact"] = artifact_entry_semantic_projection(camera.video_artifact)
        record["timestamp_sidecar_artifact"] = {
            "artifact": artifact_entry_semantic_projection(
                camera.timestamp_sidecar_artifact.artifact
            ),
            "row_count": camera.timestamp_sidecar_artifact.row_count,
        }
        cameras.append(record)

    return {
        "schema_version": manifest.schema_version,
        "schema_ref": {
            "schema_id": manifest.schema_ref.schema_id,
            "version": manifest.schema_ref.version,
            "sha256": manifest.schema_ref.sha256,
        },
        "execution_mode": manifest.execution_mode.value,
        "recording_identity": manifest.recording_identity,
        "source_content_sha256": manifest.source_content_sha256,
        "source_size_bytes": manifest.source_size_bytes,
        "mapping_profile": manifest.mapping_profile.model_dump(mode="json"),
        "ready_manifest_semantic_sha256": manifest.ready_manifest_semantic_sha256,
        "alignment_semantic_sha256": manifest.alignment_semantic_sha256,
        "alignment_status": manifest.alignment_status.value,
        "exporter": manifest.exporter.model_dump(mode="json"),
        "cameras": cameras,
    }


__all__ = [
    "CameraVideoExportManifestV2",
    "CameraVideoExportRecordV2",
    "ExportConfigArtifactPayloadV2",
    "MappingProfileArtifactPayloadV2",
    "TimestampSidecarArtifactV2",
    "artifact_entry_semantic_projection",
    "camera_video_manifest_v2_semantic_projection",
]
