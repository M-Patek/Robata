from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from robata.contracts.artifacts import (
    ArtifactLifecycle,
    ArtifactLocator,
    ArtifactParent,
    ArtifactParentRelation,
    ArtifactProducer,
    ArtifactRegistryEntry,
    ArtifactType,
    SchemaArtifactReference,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.hashing import semantic_sha256
from robata.contracts.video_export import (
    DroppedMessageProvenance,
    DroppedMessageReasonCode,
    MappingProfileReference,
    MediaTimeMapping,
    SourceVideoStream,
    TailDurationPolicy,
    VideoExportAlignmentStatus,
    VideoExporterIdentity,
    VideoExporterMode,
    VideoExportExecutionMode,
)
from robata.contracts.video_export_v2 import (
    CameraVideoExportManifestV2,
    CameraVideoExportRecordV2,
    ExportConfigArtifactPayloadV2,
    MappingProfileArtifactPayloadV2,
    TimestampSidecarArtifactV2,
    camera_video_manifest_v2_semantic_projection,
)


def _uuid(number: int) -> str:
    return f"00000000-0000-5000-8000-{number:012x}"


def _digest(number: int) -> str:
    return f"{number:064x}"


MANIFEST_SCHEMA_REF = SchemaArtifactReference(
    schema_id="https://schemas.robata.dev/camera-video-export-manifest",
    version="2.0.0",
    artifact_id=_uuid(1),
    sha256=_digest(1),
)
TIMESTAMP_SCHEMA_REF = SchemaArtifactReference(
    schema_id="https://schemas.robata.dev/camera-video-timestamp-row",
    version="1.0.0",
    artifact_id=_uuid(2),
    sha256=_digest(2),
)
SOURCE_ARTIFACT_ID = _uuid(10)
MAPPING_ARTIFACT_ID = _uuid(11)
CONFIG_ARTIFACT_ID = _uuid(12)
PRODUCER = ArtifactProducer(
    name="robata.test-exporter",
    version="1.0.0",
    canonical_config_sha256=_digest(20),
)
LIFECYCLE = ArtifactLifecycle(state="ACTIVE", policy_version="local-evidence-v1")


def _parents() -> tuple[ArtifactParent, ...]:
    return tuple(
        sorted(
            (
                ArtifactParent(
                    artifact_id=CONFIG_ARTIFACT_ID,
                    relation=ArtifactParentRelation.EXPORT_CONFIG,
                ),
                ArtifactParent(
                    artifact_id=MAPPING_ARTIFACT_ID,
                    relation=ArtifactParentRelation.MAPPING_PROFILE,
                ),
                ArtifactParent(
                    artifact_id=SOURCE_ARTIFACT_ID,
                    relation=ArtifactParentRelation.SOURCE_CONTENT,
                ),
            ),
            key=lambda parent: (parent.relation.value, parent.artifact_id),
        )
    )


def _artifact(
    artifact_type: ArtifactType,
    number: int,
    *,
    payload_schema_ref: SchemaArtifactReference | None,
) -> ArtifactRegistryEntry:
    media_types = {
        ArtifactType.CAMERA_VIDEO_MP4: "video/mp4",
        ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP: "application/x-ndjson",
    }
    return ArtifactRegistryEntry(
        schema_version="2.0",
        artifact_id=_uuid(number),
        artifact_type=artifact_type,
        semantic_sha256=_digest(100 + number),
        locator=ArtifactLocator(
            uri=f"robata-artifact://sha256/{_digest(200 + number)}",
            object_version=_digest(200 + number),
        ),
        sha256=_digest(200 + number),
        bytes=1_000 + number,
        media_type=media_types[artifact_type],
        producer=PRODUCER,
        lifecycle=LIFECYCLE,
        parents=_parents(),
        payload_schema_ref=payload_schema_ref,
        created_at="2026-07-18T12:00:00Z",
    )


def _record(number: int, camera_id: CameraId) -> CameraVideoExportRecordV2:
    return CameraVideoExportRecordV2(
        camera_id=camera_id,
        source=SourceVideoStream(
            topic=f"/camera/{number}",
            channel_id=number,
            schema_name="foxglove.CompressedImage",
            codec="h264",
        ),
        input_message_count=3,
        source_first_observed_message_ns=100,
        source_last_observed_message_ns=300,
        export_first_observed_source_message_ns=200,
        export_last_observed_source_message_ns=300,
        leading_drops=DroppedMessageProvenance(
            count=1,
            reason_code=DroppedMessageReasonCode.BEFORE_FIRST_DECODABLE_KEYFRAME,
            first_source_ns=100,
            last_source_ns=100,
        ),
        trailing_drops=DroppedMessageProvenance(
            count=0,
            reason_code=DroppedMessageReasonCode.NONE,
            first_source_ns=None,
            last_source_ns=None,
        ),
        exported_packet_count=2,
        exported_frame_count=2,
        keyframe_count=1,
        width=1600,
        height=1300,
        video_artifact=_artifact(
            ArtifactType.CAMERA_VIDEO_MP4,
            20 + number,
            payload_schema_ref=None,
        ),
        timestamp_sidecar_artifact=TimestampSidecarArtifactV2(
            artifact=_artifact(
                ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
                30 + number,
                payload_schema_ref=TIMESTAMP_SCHEMA_REF,
            ),
            row_count=2,
        ),
        media_time_mapping=MediaTimeMapping(
            zero_source_ns=200,
            time_base_numerator=1,
            time_base_denominator=1_000_000_000,
            first_pts=0,
            last_pts=100,
            last_duration=100,
            tail_duration_policy=TailDurationPolicy.MEDIAN_POSITIVE_INTERVAL,
            rounding="HALF_EVEN",
            max_rounding_error_ns=0,
        ),
    )


def _manifest_fields() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "schema_ref": MANIFEST_SCHEMA_REF,
        "semantic_content_sha256": "0" * 64,
        "execution_mode": VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE,
        "recording_identity": _digest(50),
        "source_content_sha256": _digest(51),
        "source_size_bytes": 999,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "mapping_profile_artifact_id": MAPPING_ARTIFACT_ID,
        "export_config_artifact_id": CONFIG_ARTIFACT_ID,
        "mapping_profile": MappingProfileReference(
            version="mapping-v1",
            digest=_digest(52),
            approved=False,
        ),
        "ready_manifest_id": None,
        "ready_manifest_semantic_sha256": None,
        "alignment_id": None,
        "alignment_semantic_sha256": None,
        "alignment_status": VideoExportAlignmentStatus.UNVERIFIED,
        "exporter": VideoExporterIdentity(
            name=PRODUCER.name,
            version=PRODUCER.version,
            mode=VideoExporterMode.REMUX,
            export_profile_id="direct-h264-remux",
            profile_version="1.0",
            canonical_config_sha256=PRODUCER.canonical_config_sha256,
        ),
        "cameras": tuple(
            _record(number, camera_id) for number, camera_id in enumerate(CAMERA_IDS, start=1)
        ),
    }


def _manifest(**updates: Any) -> CameraVideoExportManifestV2:
    fields = _manifest_fields()
    fields.update(updates)
    draft = CameraVideoExportManifestV2.model_construct(**fields)
    fields["semantic_content_sha256"] = semantic_sha256(
        camera_video_manifest_v2_semantic_projection(draft)
    )
    return CameraVideoExportManifestV2.model_validate(fields, strict=True)


def test_v2_manifest_has_exact_registered_six_camera_lineage() -> None:
    manifest = _manifest()

    assert manifest.schema_version == "2.0"
    assert tuple(record.camera_id for record in manifest.cameras) == CAMERA_IDS
    assert (
        len(
            {
                artifact_id
                for record in manifest.cameras
                for artifact_id in (
                    record.video_artifact.artifact_id,
                    record.timestamp_sidecar_artifact.artifact.artifact_id,
                )
            }
        )
        == 12
    )
    assert manifest.semantic_content_sha256 == semantic_sha256(
        camera_video_manifest_v2_semantic_projection(manifest)
    )


def test_semantic_projection_excludes_locator_and_publication_metadata() -> None:
    manifest = _manifest()
    first = manifest.cameras[0]
    relocated = first.video_artifact.model_copy(
        update={
            "locator": ArtifactLocator(
                uri="robata-artifact://relocated/cam-01",
                object_version="relocation-v2",
            ),
            "created_at": "2026-07-19T00:00:00Z",
        }
    )
    changed_record = first.model_copy(update={"video_artifact": relocated})
    changed = manifest.model_copy(update={"cameras": (changed_record, *manifest.cameras[1:])})

    assert camera_video_manifest_v2_semantic_projection(changed) == (
        camera_video_manifest_v2_semantic_projection(manifest)
    )


def test_semantic_projection_includes_exact_output_bytes() -> None:
    manifest = _manifest()
    first = manifest.cameras[0]
    changed_artifact = first.video_artifact.model_copy(update={"sha256": _digest(999)})
    changed_record = first.model_copy(update={"video_artifact": changed_artifact})
    changed = manifest.model_copy(update={"cameras": (changed_record, *manifest.cameras[1:])})

    assert semantic_sha256(camera_video_manifest_v2_semantic_projection(changed)) != (
        manifest.semantic_content_sha256
    )


def test_manifest_rejects_wrong_semantic_digest() -> None:
    fields = _manifest_fields()
    fields["semantic_content_sha256"] = _digest(999)
    with pytest.raises(ValidationError, match="semantic_content_sha256"):
        CameraVideoExportManifestV2.model_validate(fields, strict=True)


def test_manifest_rejects_substituted_parent() -> None:
    fields = _manifest_fields()
    first = fields["cameras"][0]
    bad_parents = tuple(
        parent.model_copy(update={"artifact_id": _uuid(999)})
        if parent.relation is ArtifactParentRelation.SOURCE_CONTENT
        else parent
        for parent in first.video_artifact.parents
    )
    bad_video = first.video_artifact.model_copy(update={"parents": bad_parents})
    fields["cameras"] = (
        first.model_copy(update={"video_artifact": bad_video}),
        *fields["cameras"][1:],
    )
    draft = CameraVideoExportManifestV2.model_construct(**fields)
    fields["semantic_content_sha256"] = semantic_sha256(
        camera_video_manifest_v2_semantic_projection(draft)
    )

    with pytest.raises(ValidationError, match="exact source"):
        CameraVideoExportManifestV2.model_validate(fields, strict=True)


def test_manifest_rejects_duplicate_output_artifact_id() -> None:
    fields = _manifest_fields()
    first = fields["cameras"][0]
    second = fields["cameras"][1]
    duplicate = second.video_artifact.model_copy(
        update={"artifact_id": first.video_artifact.artifact_id}
    )
    fields["cameras"] = (
        first,
        second.model_copy(update={"video_artifact": duplicate}),
        *fields["cameras"][2:],
    )
    draft = CameraVideoExportManifestV2.model_construct(**fields)
    fields["semantic_content_sha256"] = semantic_sha256(
        camera_video_manifest_v2_semantic_projection(draft)
    )

    with pytest.raises(ValidationError, match="output artifact IDs must be unique"):
        CameraVideoExportManifestV2.model_validate(fields, strict=True)


def test_local_mode_rejects_ready_or_alignment_evidence() -> None:
    with pytest.raises(ValidationError, match="READY evidence"):
        _manifest(
            ready_manifest_id=_uuid(90),
            ready_manifest_semantic_sha256=_digest(90),
        )

    with pytest.raises(ValidationError, match="alignment must be UNVERIFIED"):
        _manifest(
            alignment_id=_uuid(91),
            alignment_semantic_sha256=_digest(91),
            alignment_status=VideoExportAlignmentStatus.VALID,
        )


def test_registered_mapping_and_export_config_payloads_are_closed() -> None:
    mapping = MappingProfileArtifactPayloadV2(
        schema_version="2.0",
        profile_id="observed",
        version="observed-v1",
        profile_kind="OBSERVED",
        approval_status="UNAPPROVED",
        approved=False,
        mapping_policy="EXACT_TOPIC",
        required_schema="foxglove.CompressedImage",
        topics=SixCameraMap[str](
            {camera_id: f"/camera/{number}" for number, camera_id in enumerate(CAMERA_IDS, start=1)}
        ),
    )
    config = ExportConfigArtifactPayloadV2(
        schema_version="2.0",
        exporter=_manifest().exporter,
    )

    assert mapping.model_dump(mode="json")["topics"]["cam_06"] == "/camera/6"
    assert config.exporter.mode is VideoExporterMode.REMUX
    with pytest.raises(ValidationError):
        MappingProfileArtifactPayloadV2.model_validate(
            {**mapping.model_dump(mode="json"), "unexpected": True},
            strict=True,
        )
