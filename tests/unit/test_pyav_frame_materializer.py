from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import pytest

from robata.adapters.fake_vision_model import DeterministicFakeVisionModelAdapter
from robata.adapters.pyav_frame_materializer import PyAvFrameMaterializer
from robata.application.mainline import LocalMainlinePipeline
from robata.application.registered_video_export import PublishedRegisteredVideoExport
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
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.mainline import (
    ActionEventStatus,
    CameraPackageStatus,
    RunStatus,
    SamplingPurpose,
    SamplingStrategy,
    TemporalVisualPackage,
    TemporalWindow,
    VisionInferenceOutcome,
    VisionInferenceRequest,
    VisionTask,
)
from robata.contracts.video_export import (
    CameraVideoTimestampRow,
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
    TimestampSidecarArtifactV2,
    camera_video_manifest_v2_semantic_projection,
)
from robata.ports.mainline import (
    FrameMaterializationError,
    FrameMaterializationErrorCode,
    FrameMaterializationRequest,
)

SECOND = 1_000_000_000
SOURCE_ORIGIN_NS = 1_710_000_000_000_000_000
FIXED_NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)
FRAME_COUNT = 4
WIDTH = 640
HEIGHT = 360
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_MEDIA_TYPES = {
    ArtifactType.CAMERA_VIDEO_MP4: "video/mp4",
    ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP: "application/x-ndjson",
}


def _uuid(number: int) -> str:
    return f"00000000-0000-5000-8000-{number:012x}"


def _sha256(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


_PRODUCER = ArtifactProducer(
    name="robata.test-pyav-fixture",
    version="1.0.0",
    canonical_config_sha256=_sha256("pyav-fixture-config"),
)
_LIFECYCLE = ArtifactLifecycle(state="ACTIVE", policy_version="local-evidence-v1")
_SCHEMA_REF = SchemaArtifactReference(
    schema_id="https://schemas.robata.dev/test-camera-video",
    version="1.0.0",
    artifact_id=_uuid(1),
    sha256=_sha256("test-camera-video-schema"),
)
_SOURCE_ARTIFACT_ID = _uuid(2)
_MAPPING_ARTIFACT_ID = _uuid(3)
_CONFIG_ARTIFACT_ID = _uuid(4)
_SHARED_PARENTS = tuple(
    sorted(
        (
            ArtifactParent(
                artifact_id=_CONFIG_ARTIFACT_ID,
                relation=ArtifactParentRelation.EXPORT_CONFIG,
            ),
            ArtifactParent(
                artifact_id=_MAPPING_ARTIFACT_ID,
                relation=ArtifactParentRelation.MAPPING_PROFILE,
            ),
            ArtifactParent(
                artifact_id=_SOURCE_ARTIFACT_ID,
                relation=ArtifactParentRelation.SOURCE_CONTENT,
            ),
        ),
        key=lambda parent: (parent.relation.value, parent.artifact_id),
    )
)
_EXPORTER = VideoExporterIdentity(
    name=_PRODUCER.name,
    version=_PRODUCER.version,
    mode=VideoExporterMode.TRANSCODE,
    export_profile_id="test-libx264-no-bframes",
    profile_version="1.0.0",
    canonical_config_sha256=_PRODUCER.canonical_config_sha256,
)


class _CountingFakeAdapter(DeterministicFakeVisionModelAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[VisionInferenceRequest] = []

    def infer(
        self,
        request: VisionInferenceRequest,
        package: TemporalVisualPackage | None = None,
        artifact_root: Path | None = None,
    ) -> VisionInferenceOutcome:
        self.requests.append(request)
        return super().infer(request, package, artifact_root)


def _entry(
    *,
    number: int,
    artifact_type: ArtifactType,
    data: bytes,
    payload_schema_ref: SchemaArtifactReference | None,
) -> ArtifactRegistryEntry:
    digest = exact_bytes_sha256(data)
    return ArtifactRegistryEntry(
        schema_version="2.0",
        artifact_id=_uuid(number),
        artifact_type=artifact_type,
        semantic_sha256=_sha256(f"semantic:{number}:{digest}"),
        locator=ArtifactLocator(
            uri=f"artifact://test-pyav-fixture/{number}",
            object_version=digest,
        ),
        sha256=digest,
        bytes=len(data),
        media_type=_MEDIA_TYPES[artifact_type],
        producer=_PRODUCER,
        lifecycle=_LIFECYCLE,
        parents=_SHARED_PARENTS,
        payload_schema_ref=payload_schema_ref,
        created_at="2026-07-19T12:00:00Z",
    )


def _write_h264_mp4(path: Path, *, camera_index: int, pts_offset_ns: int = 0) -> None:
    try:
        av.codec.Codec("libx264", "w")
    except av.FFmpegError as error:
        pytest.skip(f"PyAV build has no libx264 encoder: {error}")

    with av.open(str(path), mode="w", format="mp4") as output:
        stream = output.add_stream("libx264", rate=1)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, SECOND)
        stream.codec_context.time_base = Fraction(1, 1)
        stream.codec_context.options = {
            "bf": "0",
            "g": "1",
            "preset": "ultrafast",
            "sc_threshold": "0",
            "tune": "zerolatency",
        }

        for frame_index in range(FRAME_COUNT):
            frame = av.VideoFrame(WIDTH, HEIGHT, "yuv420p")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 1)
            luma = 24 + camera_index * 12 + frame_index * 8
            for plane_index, plane in enumerate(frame.planes):
                value = luma if plane_index == 0 else 128
                plane.update(bytes([value]) * plane.buffer_size)

            packets = tuple(stream.encode(frame))
            assert len(packets) == 1
            (packet,) = packets
            timestamp_ns = frame_index * SECOND
            if frame_index == 1:
                timestamp_ns += pts_offset_ns
            packet.pts = timestamp_ns
            packet.dts = timestamp_ns
            packet.duration = SECOND
            packet.time_base = Fraction(1, SECOND)
            packet.stream = stream
            output.mux(packet)

        assert tuple(stream.encode(None)) == ()


def _timestamp_bytes(camera_id: CameraId) -> bytes:
    rows = tuple(
        CameraVideoTimestampRow(
            schema_version="1.0",
            export_profile_id=_EXPORTER.export_profile_id,
            export_profile_version=_EXPORTER.profile_version,
            camera_id=camera_id,
            packet_index=frame_index,
            source_sequence=frame_index,
            source_log_time_ns=SOURCE_ORIGIN_NS + frame_index * SECOND,
            source_publish_time_ns=SOURCE_ORIGIN_NS + frame_index * SECOND,
            embedded_header_time_ns=SOURCE_ORIGIN_NS + frame_index * SECOND,
            relative_pts_ns=frame_index * SECOND,
            relative_dts_ns=frame_index * SECOND,
            duration_ns=SECOND,
            time_base_numerator=1,
            time_base_denominator=SECOND,
            is_keyframe=True,
            duration_is_estimated=frame_index == FRAME_COUNT - 1,
        )
        for frame_index in range(FRAME_COUNT)
    )
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _publication(
    tmp_path: Path,
    *,
    mismatched_pts_camera: CameraId | None = None,
) -> PublishedRegisteredVideoExport:
    view = tmp_path / "video-view"
    view.mkdir()
    records: list[CameraVideoExportRecordV2] = []
    no_drops = DroppedMessageProvenance(
        count=0,
        reason_code=DroppedMessageReasonCode.NONE,
        first_source_ns=None,
        last_source_ns=None,
    )

    for camera_index, camera_id in enumerate(CAMERA_IDS, start=1):
        video_path = view / f"{camera_id.value}.mp4"
        _write_h264_mp4(
            video_path,
            camera_index=camera_index,
            pts_offset_ns=1 if camera_id is mismatched_pts_camera else 0,
        )
        video_bytes = video_path.read_bytes()
        sidecar_bytes = _timestamp_bytes(camera_id)
        (view / f"{camera_id.value}.timestamps.jsonl").write_bytes(sidecar_bytes)

        records.append(
            CameraVideoExportRecordV2(
                camera_id=camera_id,
                source=SourceVideoStream(
                    topic=f"/camera/{camera_index}",
                    channel_id=camera_index,
                    schema_name="foxglove.CompressedImage",
                    codec="h264",
                ),
                input_message_count=FRAME_COUNT,
                source_first_observed_message_ns=SOURCE_ORIGIN_NS,
                source_last_observed_message_ns=(SOURCE_ORIGIN_NS + (FRAME_COUNT - 1) * SECOND),
                export_first_observed_source_message_ns=SOURCE_ORIGIN_NS,
                export_last_observed_source_message_ns=(
                    SOURCE_ORIGIN_NS + (FRAME_COUNT - 1) * SECOND
                ),
                leading_drops=no_drops,
                trailing_drops=no_drops,
                exported_packet_count=FRAME_COUNT,
                exported_frame_count=FRAME_COUNT,
                keyframe_count=FRAME_COUNT,
                width=WIDTH,
                height=HEIGHT,
                video_artifact=_entry(
                    number=100 + camera_index,
                    artifact_type=ArtifactType.CAMERA_VIDEO_MP4,
                    data=video_bytes,
                    payload_schema_ref=None,
                ),
                timestamp_sidecar_artifact=TimestampSidecarArtifactV2(
                    artifact=_entry(
                        number=200 + camera_index,
                        artifact_type=ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
                        data=sidecar_bytes,
                        payload_schema_ref=_SCHEMA_REF,
                    ),
                    row_count=FRAME_COUNT,
                ),
                media_time_mapping=MediaTimeMapping(
                    zero_source_ns=SOURCE_ORIGIN_NS,
                    time_base_numerator=1,
                    time_base_denominator=SECOND,
                    first_pts=0,
                    last_pts=(FRAME_COUNT - 1) * SECOND,
                    last_duration=SECOND,
                    tail_duration_policy=TailDurationPolicy.MEDIAN_POSITIVE_INTERVAL,
                    rounding="HALF_EVEN",
                    max_rounding_error_ns=0,
                ),
            )
        )

    manifest_fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": _SCHEMA_REF,
        "semantic_content_sha256": "0" * 64,
        "execution_mode": VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE,
        "recording_identity": _sha256("pyav-materializer-recording"),
        "source_content_sha256": _sha256("synthetic-source"),
        "source_size_bytes": 1,
        "source_artifact_id": _SOURCE_ARTIFACT_ID,
        "mapping_profile_artifact_id": _MAPPING_ARTIFACT_ID,
        "export_config_artifact_id": _CONFIG_ARTIFACT_ID,
        "mapping_profile": MappingProfileReference(
            version="test-v1",
            digest=_sha256("test-mapping-profile"),
            approved=False,
        ),
        "ready_manifest_id": None,
        "ready_manifest_semantic_sha256": None,
        "alignment_id": None,
        "alignment_semantic_sha256": None,
        "alignment_status": VideoExportAlignmentStatus.UNVERIFIED,
        "exporter": _EXPORTER,
        "cameras": tuple(records),
    }
    draft = CameraVideoExportManifestV2.model_construct(**manifest_fields)
    manifest_fields["semantic_content_sha256"] = semantic_sha256(
        camera_video_manifest_v2_semantic_projection(draft)
    )
    manifest = CameraVideoExportManifestV2.model_validate(manifest_fields, strict=True)
    manifest_bytes = canonical_json_bytes(manifest)
    (view / "camera-video-export-manifest.json").write_bytes(manifest_bytes)
    return PublishedRegisteredVideoExport(
        output_directory=view,
        manifest=manifest,
        manifest_sha256=exact_bytes_sha256(manifest_bytes),
        manifest_artifact_id=_uuid(300),
        logical_key="real-pyav-materializer-fixture",
        derivation_reused=False,
        materialized_view_reused=False,
    )


def _window(publication: PublishedRegisteredVideoExport) -> TemporalWindow:
    interval = NanosecondInterval(start_ns=0, end_ns=FRAME_COUNT * SECOND)
    return TemporalWindow(
        schema_version="1.0",
        window_id=_uuid(400),
        mcap_id=_uuid(401),
        camera_mapping_run_id=None,
        alignment_id=None,
        requested_interval=interval,
        interval=interval,
        purpose=SamplingPurpose.QA_COARSE,
        parent_window_id=None,
        source_candidate_id=None,
        source_event_id=None,
        generation=0,
    )


def _request(
    publication: PublishedRegisteredVideoExport,
    output_directory: Path,
) -> FrameMaterializationRequest:
    return FrameMaterializationRequest(
        video_export=publication,
        output_directory=output_directory,
        window=_window(publication),
        purpose=SamplingPurpose.QA_COARSE,
        rate_num=1,
        rate_den=1,
        selection_tolerance_ns=0,
    )


def test_real_pyav_sampling_verifies_pts_and_publishes_stable_six_view_pngs(
    tmp_path: Path,
) -> None:
    publication = _publication(tmp_path)
    materializer = PyAvFrameMaterializer(max_width=320, clock=lambda: FIXED_NOW)
    first_root = tmp_path / "materialized-first"
    first = materializer.materialize(_request(publication, first_root))
    second = materializer.materialize(_request(publication, tmp_path / "materialized-second"))

    assert first == second
    assert first.package_id == second.package_id
    assert first.content_sha256 == second.content_sha256
    assert tuple(first.cameras.keys()) == CAMERA_IDS
    assert first.frame_count_total == len(CAMERA_IDS) * FRAME_COUNT

    for camera_id, camera in first.cameras.items():
        assert camera.status is CameraPackageStatus.AVAILABLE
        assert camera.sampling.strategy is SamplingStrategy.UNIFORM
        assert camera.sampling.target_fps == camera.sampling.actual_fps == 1.0
        assert (
            camera.sampling.target_count,
            camera.sampling.actual_count,
            camera.sampling.missed_targets,
        ) == (FRAME_COUNT, FRAME_COUNT, 0)
        assert tuple(frame.source_frame_index for frame in camera.frames) == tuple(
            range(FRAME_COUNT)
        )
        assert tuple(frame.target_timestamp_ns for frame in camera.frames) == tuple(
            frame_index * SECOND for frame_index in range(FRAME_COUNT)
        )
        assert tuple(frame.aligned_timestamp_ns for frame in camera.frames) == tuple(
            frame_index * SECOND for frame_index in range(FRAME_COUNT)
        )
        assert all(frame.delta_to_target_ns == 0 for frame in camera.frames)

        for frame in camera.frames:
            filename = frame.artifact_uri.rsplit("/", 1)[-1]
            png_path = first_root / "frames" / first.package_id / camera_id.value / filename
            png_bytes = png_path.read_bytes()
            assert png_bytes.startswith(PNG_SIGNATURE)
            assert exact_bytes_sha256(png_bytes) == frame.artifact_sha256
            assert (frame.width, frame.height) == (320, 180)


def test_decoded_pts_that_disagrees_with_canonical_sidecar_fails(tmp_path: Path) -> None:
    publication = _publication(tmp_path, mismatched_pts_camera=CameraId.CAM_01)
    materializer = PyAvFrameMaterializer(clock=lambda: FIXED_NOW)

    with pytest.raises(FrameMaterializationError) as caught:
        materializer.materialize(_request(publication, tmp_path / "materialized"))

    assert caught.value.code is FrameMaterializationErrorCode.TIMESTAMP_MISMATCH
    assert "cam_01 frame 1 PTS" in str(caught.value)
    assert "does not match sidecar PTS" in str(caught.value)


def test_real_publication_runs_complete_local_mainline_without_provider_requests(
    tmp_path: Path,
) -> None:
    publication = _publication(tmp_path)
    adapter = _CountingFakeAdapter()
    pipeline = LocalMainlinePipeline(
        PyAvFrameMaterializer(max_width=320, clock=lambda: FIXED_NOW),
        adapter,
        clock=lambda: FIXED_NOW,
        monotonic=lambda: 100.0,
    )

    published = pipeline.run(publication, tmp_path / "analysis")
    bundle = published.bundle
    report = bundle.report

    assert tuple(request.task for request in adapter.requests) == (
        VisionTask.QA_COARSE,
        VisionTask.EVENT_PROPOSAL,
        VisionTask.QA_DENSE,
        VisionTask.ACTION_EVIDENCE,
        VisionTask.BOUNDARY_REFINEMENT,
    )
    assert report.status is RunStatus.PRIMARY_COMPLETE
    assert report.inference_attempt_count == report.fake_inference_attempt_count == 5
    assert report.inference_success_count == 5
    assert report.real_provider_request_count == adapter.external_provider_requests == 0
    assert report.event_count == len(bundle.events) == 1
    assert len(bundle.packages) == 2
    assert len(bundle.qa_aggregates) == 2
    assert report.source_recording_identity == publication.manifest.recording_identity
    assert report.source_content_sha256 == publication.manifest.source_content_sha256
    assert report.video_manifest_artifact_id == publication.manifest_artifact_id
    assert report.video_manifest_sha256 == publication.manifest_sha256
    assert all(outcome.provider == "fake" for outcome in bundle.inference_outcomes)

    (event,) = bundle.events
    assert event.status is ActionEventStatus.FINAL
    assert event.production_eligible is False
    assert tuple(event.camera_evidence.keys()) == CAMERA_IDS
    assert all(evidence.frame_ids for evidence in event.camera_evidence.values())
    assert len(tuple((published.output_directory / "frames").glob("*/cam_*/*.png"))) == (
        len(CAMERA_IDS) * FRAME_COUNT * 2
    )
