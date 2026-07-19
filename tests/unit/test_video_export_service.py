from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from robata.application import (
    LocalVideoExportRequest,
    SixCameraVideoExportService,
    VideoExporterDescriptor,
    VideoExportRunError,
    VideoExportRunErrorCode,
)
from robata.contracts import (
    CAMERA_IDS,
    CameraId,
    SixCameraMap,
    canonical_json_bytes,
    exact_bytes_sha256,
)
from robata.contracts.video_export import CameraVideoTimestampRow, VideoExporterMode
from robata.ingestion import TopicMappingProfile
from robata.ports import (
    ChannelInspection,
    ExportedCameraVideoFacts,
    McapInspection,
    VideoExportError,
    VideoExportErrorCode,
)


@dataclass
class _FakeExporter:
    fail_on: CameraId | None = None
    invalid_sidecar_on: CameraId | None = None
    mutate_source_on: CameraId | None = None

    def __post_init__(self) -> None:
        self.calls: list[CameraId] = []

    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> ExportedCameraVideoFacts:
        self.calls.append(camera_id)
        if camera_id is self.fail_on:
            raise VideoExportError(VideoExportErrorCode.REMUX_FAILED, "injected failure")

        video_bytes = f"mp4-{camera_id.value}".encode()
        sidecar_bytes = _sidecar_bytes(camera_id)
        if camera_id is self.invalid_sidecar_on:
            sidecar_bytes = b"{}\n{}\n"
        video_path.write_bytes(video_bytes)
        sidecar_path.write_bytes(sidecar_bytes)
        if camera_id is self.mutate_source_on:
            source.write_bytes(b"changed-source")

        return ExportedCameraVideoFacts(
            camera_id=camera_id,
            channel_id=channel.channel_id,
            topic=channel.topic,
            video_path=video_path,
            sidecar_path=sidecar_path,
            source_message_count=3,
            leading_access_unit_count=1,
            trailing_access_unit_count=0,
            exported_packet_count=2,
            decoded_frame_count=2,
            keyframe_count=1,
            width=1600,
            height=1300,
            source_first_log_time_ns=100,
            source_last_log_time_ns=300,
            leading_first_log_time_ns=100,
            leading_last_log_time_ns=100,
            trailing_first_log_time_ns=None,
            trailing_last_log_time_ns=None,
            export_first_source_log_time_ns=200,
            export_last_source_log_time_ns=300,
            first_pts_ns=0,
            last_pts_ns=100,
            duration_ns=200,
            time_base_numerator=1,
            time_base_denominator=1_000_000_000,
            tail_duration_ns=100,
            tail_duration_policy="MEDIAN_POSITIVE_INTERVAL",
            max_timestamp_mapping_error_ns=0,
            video_size_bytes=len(video_bytes),
            video_sha256=exact_bytes_sha256(video_bytes),
            sidecar_row_count=2,
            sidecar_size_bytes=len(sidecar_bytes),
            sidecar_sha256=exact_bytes_sha256(sidecar_bytes),
        )


def _sidecar_bytes(camera_id: CameraId) -> bytes:
    rows = []
    for index, source_ns in enumerate((200, 300)):
        row = CameraVideoTimestampRow(
            schema_version="1.0",
            export_profile_id="test-remux",
            export_profile_version="1.0",
            camera_id=camera_id,
            packet_index=index,
            source_sequence=index,
            source_log_time_ns=source_ns,
            source_publish_time_ns=source_ns,
            embedded_header_time_ns=source_ns,
            relative_pts_ns=index * 100,
            relative_dts_ns=index * 100,
            duration_ns=100,
            time_base_numerator=1,
            time_base_denominator=1_000_000_000,
            is_keyframe=index == 0,
            duration_is_estimated=index == 1,
        )
        rows.append(canonical_json_bytes(row) + b"\n")
    return b"".join(rows)


def _request(tmp_path: Path, *, namespace: str = "robata") -> LocalVideoExportRequest:
    source = tmp_path / "source.mcap"
    if not source.exists():
        source.write_bytes(b"stable-source")
    channels = SixCameraMap[ChannelInspection](
        {
            camera_id: ChannelInspection(
                channel_id=index,
                topic=f"/camera/{index}",
                schema_name="foxglove.CompressedImage",
                message_encoding="protobuf",
                message_count=3,
                first_message_time_ns=100,
                last_message_time_ns=300,
                monotonic=True,
                codec="h264",
                frame_id=f"camera-{index}",
            )
            for index, camera_id in enumerate(CAMERA_IDS, start=1)
        }
    )
    source_bytes = source.read_bytes()
    inspection = McapInspection(
        source=source,
        source_size_bytes=len(source_bytes),
        source_sha256=exact_bytes_sha256(source_bytes),
        header_profile="synthetic",
        header_library="test",
        summary_available=True,
        channel_count=6,
        message_count=18,
        first_message_time_ns=100,
        last_message_time_ns=300,
        channels=tuple(channels.values()),
    )
    profile = TopicMappingProfile(
        profile_id="test-observed",
        version="test-observed-v1",
        profile_kind="OBSERVED",
        approval_status="UNAPPROVED",
        approved=False,
        mapping_policy="EXACT_TOPIC",
        required_schema="foxglove.CompressedImage",
        topics=SixCameraMap[str](
            {camera_id: channels[camera_id].topic for camera_id in CAMERA_IDS}
        ),
    )
    return LocalVideoExportRequest(
        source=source,
        output_directory=tmp_path / "export",
        namespace=namespace,
        inspection=inspection,
        channels=channels,
        mapping_profile=profile,
        mapping_profile_digest=profile.semantic_digest,
        exporter=VideoExporterDescriptor(
            name="test.exporter",
            version="0.1.0",
            mode=VideoExporterMode.REMUX,
            export_profile_id="test-remux",
            profile_version="1.0",
            canonical_config_sha256="a" * 64,
        ),
    )


def test_complete_export_is_atomic_schema_valid_and_reusable(tmp_path: Path) -> None:
    request = _request(tmp_path)
    first_exporter = _FakeExporter()

    first = SixCameraVideoExportService(first_exporter).export_local(request)

    assert first.reused is False
    assert first.output_directory.is_dir()
    assert [camera.camera_id for camera in first.manifest.cameras] == list(CAMERA_IDS)
    assert first.manifest.mapping_profile.approved is False
    assert first.manifest.ready_manifest_id is None
    assert len(first_exporter.calls) == 6
    assert len(tuple(first.output_directory.iterdir())) == 13

    second_exporter = _FakeExporter()
    second = SixCameraVideoExportService(second_exporter).export_local(request)

    assert second.reused is True
    assert second.manifest_sha256 == first.manifest_sha256
    assert second_exporter.calls == []


def test_partial_export_failure_publishes_nothing(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(VideoExportError, match="injected failure"):
        SixCameraVideoExportService(_FakeExporter(fail_on=CameraId.CAM_03)).export_local(request)

    assert not request.output_directory.exists()
    assert list(tmp_path.glob(".export.partial-*")) == []


def test_invalid_sidecar_fails_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(VideoExportRunError) as caught:
        SixCameraVideoExportService(_FakeExporter(invalid_sidecar_on=CameraId.CAM_02)).export_local(
            request
        )

    assert caught.value.code is VideoExportRunErrorCode.DERIVED_ARTIFACT_INVALID
    assert not request.output_directory.exists()


def test_source_mutation_before_publish_fails_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(VideoExportRunError) as caught:
        SixCameraVideoExportService(_FakeExporter(mutate_source_on=CameraId.CAM_06)).export_local(
            request
        )

    assert caught.value.code is VideoExportRunErrorCode.SOURCE_CHANGED
    assert not request.output_directory.exists()


def test_existing_export_with_other_identity_is_not_reused(tmp_path: Path) -> None:
    request = _request(tmp_path)
    SixCameraVideoExportService(_FakeExporter()).export_local(request)
    different_request = _request(tmp_path, namespace="different")

    with pytest.raises(VideoExportRunError) as caught:
        SixCameraVideoExportService(_FakeExporter()).export_local(different_request)

    assert caught.value.code is VideoExportRunErrorCode.OUTPUT_EXISTS


def test_request_rejects_mapping_digest_not_derived_from_profile(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), mapping_profile_digest="f" * 64)
    exporter = _FakeExporter()

    with pytest.raises(VideoExportRunError) as caught:
        SixCameraVideoExportService(exporter).export_local(request)

    assert caught.value.code is VideoExportRunErrorCode.INVALID_REQUEST
    assert exporter.calls == []
    assert not request.output_directory.exists()


def test_existing_manifest_cannot_substitute_source_channel_facts(tmp_path: Path) -> None:
    request = _request(tmp_path)
    SixCameraVideoExportService(_FakeExporter()).export_local(request)
    manifest_path = request.output_directory / "camera-video-export-manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["cameras"][0]["source"]["topic"] = "/substituted/topic"
    manifest_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(VideoExportRunError) as caught:
        SixCameraVideoExportService(_FakeExporter()).export_local(request)

    assert caught.value.code is VideoExportRunErrorCode.OUTPUT_EXISTS


def test_parallel_export_preserves_manifest_and_artifact_order(tmp_path: Path) -> None:
    request = _request(tmp_path)
    serial_request = replace(request, output_directory=tmp_path / "serial")
    parallel_request = replace(request, output_directory=tmp_path / "parallel")

    serial = SixCameraVideoExportService(_FakeExporter()).export_local(serial_request)
    parallel = SixCameraVideoExportService(
        _FakeExporter(),
        max_parallel_exports=6,
    ).export_local(parallel_request)

    assert parallel.manifest == serial.manifest
    assert parallel.manifest_sha256 == serial.manifest_sha256
    assert tuple(record.camera_id for record in parallel.manifest.cameras) == CAMERA_IDS
    for camera_id in CAMERA_IDS:
        assert (parallel.output_directory / f"{camera_id.value}.mp4").read_bytes() == (
            serial.output_directory / f"{camera_id.value}.mp4"
        ).read_bytes()
        assert (parallel.output_directory / f"{camera_id.value}.timestamps.jsonl").read_bytes() == (
            serial.output_directory / f"{camera_id.value}.timestamps.jsonl"
        ).read_bytes()


def test_parallel_export_worker_count_is_bounded() -> None:
    with pytest.raises(ValueError, match="six camera"):
        SixCameraVideoExportService(_FakeExporter(), max_parallel_exports=7)
    with pytest.raises(ValueError, match="positive"):
        SixCameraVideoExportService(_FakeExporter(), max_parallel_exports=0)
