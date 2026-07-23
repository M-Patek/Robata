from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from robata.adapters.local_artifact_registry import LocalArtifactRegistry
from robata.application.registered_video_export import RegisteredSixCameraVideoExportService
from robata.application.video_export import (
    LocalVideoExportRequest,
    SixCameraVideoExportService,
    StagedSixCameraVideoExport,
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
    semantic_sha256,
)
from robata.contracts.video_export import (
    CameraVideoTimestampRow,
    VideoExporterMode,
)
from robata.ingestion.mapping import TopicMappingProfile
from robata.ports import COMPRESSED_IMAGE_SCHEMA, ChannelInspection, McapInspection
from robata.ports.video_export import ExportedCameraVideoFacts

_TAIL_NS = 500_000_000


class _NoCallExporter:
    def __init__(self) -> None:
        self.calls = 0

    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> ExportedCameraVideoFacts:
        del source, camera_id, channel, video_path, sidecar_path
        self.calls += 1
        raise AssertionError("staged publication must not call the camera exporter")


class _FixtureExporter:
    def __init__(self, request: LocalVideoExportRequest) -> None:
        self._request = request
        self.calls = 0

    def export(
        self,
        source: Path,
        camera_id: CameraId,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> ExportedCameraVideoFacts:
        assert source == self._request.source
        assert channel == self._request.channels[camera_id]
        self.calls += 1
        return _write_camera_artifacts(self._request, camera_id, video_path.parent)


def test_staged_publication_calls_single_traversal_once_and_skips_exporter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, output_name="staged-view")
    exporter = _NoCallExporter()
    service = _service(tmp_path, exporter)
    calls = 0
    observed_staging: Path | None = None
    monkeypatch.setattr(
        SixCameraVideoExportService,
        "_verify_source_unchanged",
        _unexpected_source_rehash,
    )

    def produce(staging_directory: Path) -> StagedSixCameraVideoExport:
        nonlocal calls, observed_staging
        calls += 1
        observed_staging = staging_directory
        assert staging_directory.is_dir()
        return _staged_result(request, staging_directory)

    published = service.export_staged_local(request, produce)

    assert calls == 1
    assert exporter.calls == 0
    assert observed_staging is not None
    assert not observed_staging.exists()
    assert not published.derivation_reused
    assert tuple(record.camera_id for record in published.manifest.cameras) == CAMERA_IDS
    assert tuple(path.name for path in published.output_directory.glob("*.mp4")) == tuple(
        f"{camera_id.value}.mp4" for camera_id in CAMERA_IDS
    )
    for record in published.manifest.cameras:
        view_path = published.output_directory / f"{record.camera_id.value}.mp4"
        blob_path = (
            tmp_path
            / "registry"
            / "blobs"
            / "sha256"
            / record.video_artifact.sha256[:2]
            / record.video_artifact.sha256
        )
        assert os.path.samefile(view_path, blob_path)


def test_staged_publication_rejects_source_identity_mismatch_without_commit(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, output_name="mismatch-view")
    exporter = _NoCallExporter()
    service = _service(tmp_path, exporter)
    calls = 0

    def produce(staging_directory: Path) -> StagedSixCameraVideoExport:
        nonlocal calls
        calls += 1
        result = _staged_result(request, staging_directory)
        return StagedSixCameraVideoExport(
            source_size_bytes=result.source_size_bytes,
            source_sha256="0" * 64,
            camera_facts=result.camera_facts,
        )

    with pytest.raises(VideoExportRunError) as raised:
        service.export_staged_local(request, produce)

    assert raised.value.code is VideoExportRunErrorCode.SOURCE_CHANGED
    assert calls == 1
    assert exporter.calls == 0
    assert not request.output_directory.exists()
    assert not tuple((tmp_path / "registry").glob("blobs/sha256/*/*"))


def test_existing_export_local_still_calls_exporter_for_each_camera(tmp_path: Path) -> None:
    request = _request(tmp_path, output_name="legacy-view")
    exporter = _FixtureExporter(request)
    service = _service(tmp_path, exporter)

    published = service.export_local(request)

    assert exporter.calls == len(CAMERA_IDS)
    assert not published.derivation_reused
    assert tuple(record.camera_id for record in published.manifest.cameras) == CAMERA_IDS


def _service(
    tmp_path: Path,
    exporter: _NoCallExporter | _FixtureExporter,
) -> RegisteredSixCameraVideoExportService:
    return RegisteredSixCameraVideoExportService(
        exporter,
        LocalArtifactRegistry(tmp_path / "registry"),
        clock=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )


def _unexpected_source_rehash(request: LocalVideoExportRequest) -> None:
    del request
    raise AssertionError("staged publication must use the traversal source identity")


def _request(tmp_path: Path, *, output_name: str) -> LocalVideoExportRequest:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"deterministic source fixture")
    source_bytes = source.read_bytes()
    source_sha256 = exact_bytes_sha256(source_bytes)
    channels = tuple(
        ChannelInspection(
            channel_id=index,
            topic=f"/fixture/{camera_id.value}",
            schema_name=COMPRESSED_IMAGE_SCHEMA,
            message_encoding="protobuf",
            message_count=1,
            first_message_time_ns=1_000_000_000 + index,
            last_message_time_ns=1_000_000_000 + index,
            monotonic=True,
            codec="h264",
            frame_id=f"fixture_{camera_id.value}",
        )
        for index, camera_id in enumerate(CAMERA_IDS, start=1)
    )
    channel_map = SixCameraMap[ChannelInspection].model_validate(
        dict(zip(CAMERA_IDS, channels, strict=True)),
        strict=True,
    )
    topics = SixCameraMap[str].model_validate(
        {camera_id: channel_map[camera_id].topic for camera_id in CAMERA_IDS},
        strict=True,
    )
    profile = TopicMappingProfile(
        profile_id="fixture-six-camera",
        version="1.0.0",
        profile_kind="OBSERVED",
        approval_status="UNAPPROVED",
        approved=False,
        mapping_policy="EXACT_TOPIC",
        required_schema=COMPRESSED_IMAGE_SCHEMA,
        topics=topics,
    )
    inspection = McapInspection(
        source=source,
        source_size_bytes=len(source_bytes),
        source_sha256=source_sha256,
        header_profile="fixture",
        header_library="fixture",
        summary_available=True,
        channel_count=len(CAMERA_IDS),
        message_count=len(CAMERA_IDS),
        first_message_time_ns=channels[0].first_message_time_ns,
        last_message_time_ns=channels[-1].last_message_time_ns,
        channels=channels,
    )
    return LocalVideoExportRequest(
        source=source,
        output_directory=tmp_path / output_name,
        namespace="robata-test",
        inspection=inspection,
        channels=channel_map,
        mapping_profile=profile,
        mapping_profile_digest=profile.semantic_digest,
        exporter=VideoExporterDescriptor(
            name="fixture-exporter",
            version="1.0.0",
            mode=VideoExporterMode.REMUX,
            export_profile_id="fixture-remux",
            profile_version="1.0.0",
            canonical_config_sha256=semantic_sha256({"fixture": "staged-publication"}),
        ),
    )


def _staged_result(
    request: LocalVideoExportRequest,
    staging_directory: Path,
) -> StagedSixCameraVideoExport:
    return StagedSixCameraVideoExport(
        source_size_bytes=request.inspection.source_size_bytes,
        source_sha256=request.inspection.source_sha256,
        camera_facts=tuple(
            _write_camera_artifacts(request, camera_id, staging_directory)
            for camera_id in CAMERA_IDS
        ),
    )


def _write_camera_artifacts(
    request: LocalVideoExportRequest,
    camera_id: CameraId,
    staging_directory: Path,
) -> ExportedCameraVideoFacts:
    channel = request.channels[camera_id]
    assert channel.first_message_time_ns is not None
    source_ns = channel.first_message_time_ns
    video_path = staging_directory / f"{camera_id.value}.mp4"
    sidecar_path = staging_directory / f"{camera_id.value}.timestamps.jsonl"
    video_bytes = f"fixture-mp4:{camera_id.value}".encode()
    sidecar_bytes = canonical_json_bytes(
        CameraVideoTimestampRow(
            schema_version="1.0",
            export_profile_id=request.exporter.export_profile_id,
            export_profile_version=request.exporter.profile_version,
            camera_id=camera_id,
            packet_index=0,
            source_sequence=0,
            source_log_time_ns=source_ns,
            source_publish_time_ns=source_ns,
            embedded_header_time_ns=source_ns,
            relative_pts_ns=0,
            relative_dts_ns=0,
            duration_ns=_TAIL_NS,
            time_base_numerator=1,
            time_base_denominator=1_000_000_000,
            is_keyframe=True,
            duration_is_estimated=True,
        )
    ) + b"\n"
    video_path.write_bytes(video_bytes)
    sidecar_path.write_bytes(sidecar_bytes)
    return ExportedCameraVideoFacts(
        camera_id=camera_id,
        channel_id=channel.channel_id,
        topic=channel.topic,
        video_path=video_path,
        sidecar_path=sidecar_path,
        source_message_count=1,
        leading_access_unit_count=0,
        trailing_access_unit_count=0,
        exported_packet_count=1,
        decoded_frame_count=1,
        keyframe_count=1,
        width=16,
        height=16,
        source_first_log_time_ns=source_ns,
        source_last_log_time_ns=source_ns,
        leading_first_log_time_ns=None,
        leading_last_log_time_ns=None,
        trailing_first_log_time_ns=None,
        trailing_last_log_time_ns=None,
        export_first_source_log_time_ns=source_ns,
        export_last_source_log_time_ns=source_ns,
        first_pts_ns=0,
        last_pts_ns=0,
        duration_ns=_TAIL_NS,
        time_base_numerator=1,
        time_base_denominator=1_000_000_000,
        tail_duration_ns=_TAIL_NS,
        tail_duration_policy="MEDIAN_POSITIVE_INTERVAL",
        max_timestamp_mapping_error_ns=0,
        video_size_bytes=len(video_bytes),
        video_sha256=exact_bytes_sha256(video_bytes),
        sidecar_row_count=1,
        sidecar_size_bytes=len(sidecar_bytes),
        sidecar_sha256=exact_bytes_sha256(sidecar_bytes),
    )
