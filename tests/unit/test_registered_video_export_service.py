from __future__ import annotations

import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from robata.adapters import LocalArtifactRegistry
from robata.application import (
    LocalVideoExportRequest,
    PublishedRegisteredVideoExport,
    RegisteredSixCameraVideoExportService,
    VideoExporterDescriptor,
    VideoExportRunError,
    VideoExportRunErrorCode,
)
from robata.contracts import (
    CAMERA_IDS,
    CameraId,
    CameraVideoExportManifestV2,
    SixCameraMap,
    camera_video_manifest_v2_semantic_projection,
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.contracts.artifacts import (
    ArtifactLocator,
    ArtifactProducer,
    ArtifactRegistryEntry,
    ArtifactRegistrySnapshot,
    ArtifactType,
    SchemaArtifactReference,
)
from robata.contracts.video_export import CameraVideoTimestampRow, VideoExporterMode
from robata.ingestion import TopicMappingProfile
from robata.ports import (
    ArtifactRegistryError,
    ArtifactRegistryErrorCode,
    ChannelInspection,
    ExportedCameraVideoFacts,
    McapInspection,
    VideoExportError,
    VideoExportErrorCode,
)

_FIXED_NOW = datetime(2026, 7, 18, 4, 5, 6, 789000, tzinfo=UTC)
_FIXED_CREATED_AT = "2026-07-18T04:05:06.789000Z"


@dataclass
class _FakeExporter:
    fail_on: CameraId | None = None

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
        video_path.write_bytes(video_bytes)
        sidecar_path.write_bytes(sidecar_bytes)

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


def _clock() -> datetime:
    return _FIXED_NOW


def _service(
    exporter: _FakeExporter,
    registry: LocalArtifactRegistry,
) -> RegisteredSixCameraVideoExportService:
    return RegisteredSixCameraVideoExportService(exporter, registry, clock=_clock)


def _expected_view_names() -> set[str]:
    names = {"camera-video-export-manifest.json"}
    for camera_id in CAMERA_IDS:
        names.add(f"{camera_id.value}.mp4")
        names.add(f"{camera_id.value}.timestamps.jsonl")
    return names


def _view_names(directory: Path) -> set[str]:
    return {child.name for child in directory.iterdir()}


def _registry_row_counts(registry: LocalArtifactRegistry) -> tuple[int, int]:
    with sqlite3.connect(registry.database_path) as connection:
        artifacts = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()
        derivations = connection.execute("SELECT COUNT(*) FROM derivations").fetchone()
    assert artifacts is not None
    assert derivations is not None
    return int(artifacts[0]), int(derivations[0])


def _publish_coherent_foreign_derivation(
    *,
    source_registry: LocalArtifactRegistry,
    target_registry: LocalArtifactRegistry,
    published: PublishedRegisteredVideoExport,
    mutation: str,
) -> None:
    source_snapshot = source_registry.verify_derivation(published.logical_key)
    entries_by_id = {entry.artifact_id: entry for entry in source_snapshot.entries}
    records = list(published.manifest.cameras)
    first_record = records[0]

    if mutation == "producer":
        original_output = first_record.video_artifact
        changed_output = ArtifactRegistryEntry.model_validate(
            original_output.model_copy(
                update={
                    "producer": ArtifactProducer(
                        name="foreign.exporter",
                        version="9.9.9",
                        canonical_config_sha256="b" * 64,
                    )
                }
            ).model_dump(mode="python"),
            strict=True,
        )
        records[0] = first_record.model_copy(update={"video_artifact": changed_output})
    elif mutation == "timestamp-schema":
        original_output = first_record.timestamp_sidecar_artifact.artifact
        current_schema = original_output.payload_schema_ref
        assert current_schema is not None
        wrong_schema = next(
            entry
            for entry in source_snapshot.entries
            if entry.artifact_type is ArtifactType.JSON_SCHEMA
            and entry.artifact_id != current_schema.artifact_id
        )
        wrong_schema_ref = SchemaArtifactReference(
            schema_id=wrong_schema.locator.uri,
            version=wrong_schema.locator.object_version,
            artifact_id=wrong_schema.artifact_id,
            sha256=wrong_schema.sha256,
        )
        parent_semantics = {
            entries_by_id[parent.artifact_id].artifact_type.value: entries_by_id[
                parent.artifact_id
            ].semantic_sha256
            for parent in original_output.parents
        }
        changed_semantic_sha256 = semantic_sha256(
            {
                "schema_version": "2.0",
                "camera_id": first_record.camera_id.value,
                "artifact_type": original_output.artifact_type.value,
                "sha256": original_output.sha256,
                "bytes": original_output.bytes,
                "parent_semantic_sha256": parent_semantics,
                "payload_schema_sha256": wrong_schema_ref.sha256,
            }
        )
        changed_artifact_id = target_registry.allocate_artifact_id(
            original_output.artifact_type,
            changed_semantic_sha256,
        )
        changed_output = ArtifactRegistryEntry.model_validate(
            original_output.model_copy(
                update={
                    "artifact_id": changed_artifact_id,
                    "semantic_sha256": changed_semantic_sha256,
                    "locator": ArtifactLocator(
                        uri=f"robata-artifact://local/{changed_artifact_id}",
                        object_version=original_output.sha256,
                    ),
                    "payload_schema_ref": wrong_schema_ref,
                }
            ).model_dump(mode="python"),
            strict=True,
        )
        changed_sidecar = first_record.timestamp_sidecar_artifact.model_copy(
            update={"artifact": changed_output}
        )
        records[0] = first_record.model_copy(update={"timestamp_sidecar_artifact": changed_sidecar})
    else:
        raise AssertionError(f"unsupported foreign derivation mutation: {mutation}")

    draft_manifest = published.manifest.model_copy(
        update={
            "semantic_content_sha256": "0" * 64,
            "cameras": tuple(records),
        }
    )
    manifest_semantic_sha256 = semantic_sha256(
        camera_video_manifest_v2_semantic_projection(draft_manifest)
    )
    foreign_manifest = CameraVideoExportManifestV2.model_validate(
        draft_manifest.model_copy(
            update={"semantic_content_sha256": manifest_semantic_sha256}
        ).model_dump(mode="python"),
        strict=True,
    )
    manifest_bytes = canonical_json_bytes(foreign_manifest)
    manifest_exact_sha256 = exact_bytes_sha256(manifest_bytes)
    original_manifest_entry = entries_by_id[published.manifest_artifact_id]
    foreign_manifest_artifact_id = target_registry.allocate_artifact_id(
        ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST,
        manifest_semantic_sha256,
    )
    manifest_parents = tuple(
        sorted(
            (
                parent.model_copy(update={"artifact_id": changed_output.artifact_id})
                if parent.artifact_id == original_output.artifact_id
                else parent
                for parent in original_manifest_entry.parents
            ),
            key=lambda parent: (parent.relation.value, parent.artifact_id),
        )
    )
    foreign_manifest_entry = ArtifactRegistryEntry.model_validate(
        original_manifest_entry.model_copy(
            update={
                "artifact_id": foreign_manifest_artifact_id,
                "semantic_sha256": manifest_semantic_sha256,
                "locator": ArtifactLocator(
                    uri=f"robata-artifact://local/{foreign_manifest_artifact_id}",
                    object_version=manifest_exact_sha256,
                ),
                "sha256": manifest_exact_sha256,
                "bytes": len(manifest_bytes),
                "parents": manifest_parents,
            }
        ).model_dump(mode="python"),
        strict=True,
    )

    entries = [
        changed_output if entry.artifact_id == original_output.artifact_id else entry
        for entry in source_snapshot.entries
        if entry.artifact_id != published.manifest_artifact_id
    ]
    entries.append(foreign_manifest_entry)
    foreign_snapshot = ArtifactRegistrySnapshot(
        schema_version="2.0",
        entries=tuple(sorted(entries, key=lambda entry: entry.artifact_id)),
    )
    blob_sources = {
        entry.artifact_id: (
            manifest_bytes
            if entry.artifact_id == foreign_manifest_artifact_id
            else source_registry.resolve_blob(
                original_output.artifact_id
                if entry.artifact_id == changed_output.artifact_id
                else entry.artifact_id
            ).read_bytes()
        )
        for entry in foreign_snapshot.entries
    }
    target_registry.publish_derivation(
        snapshot=foreign_snapshot,
        logical_key=published.logical_key,
        manifest_artifact_id=foreign_manifest_artifact_id,
        blob_sources=blob_sources,
    )


def test_first_export_registers_twenty_entries_and_materializes_thirteen_file_view(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    exporter = _FakeExporter()
    registry = LocalArtifactRegistry(tmp_path / "registry")

    result = _service(exporter, registry).export_local(request)

    assert result.derivation_reused is False
    assert result.materialized_view_reused is False
    assert exporter.calls == list(CAMERA_IDS)
    assert _view_names(result.output_directory) == _expected_view_names()

    snapshot = registry.verify_derivation(result.logical_key)
    assert len(snapshot.entries) == 20
    assert Counter(entry.artifact_type for entry in snapshot.entries) == Counter(
        {
            ArtifactType.JSON_SCHEMA: 4,
            ArtifactType.RAW_MCAP: 1,
            ArtifactType.MAPPING_PROFILE: 1,
            ArtifactType.EXPORT_CONFIG: 1,
            ArtifactType.CAMERA_VIDEO_MP4: 6,
            ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP: 6,
            ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST: 1,
        }
    )
    assert {entry.created_at for entry in snapshot.entries} == {_FIXED_CREATED_AT}

    manifest_entries = tuple(
        entry
        for entry in snapshot.entries
        if entry.artifact_type is ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST
    )
    assert len(manifest_entries) == 1
    manifest_entry = manifest_entries[0]
    manifest_bytes = canonical_json_bytes(result.manifest)
    manifest_body = result.manifest.model_dump(mode="json")
    assert "artifact_id" not in manifest_body
    assert "manifest_artifact_id" not in manifest_body
    assert "manifest_bytes_sha256" not in manifest_body
    assert result.manifest_artifact_id.encode() not in manifest_bytes
    assert result.manifest_sha256.encode() not in manifest_bytes
    assert manifest_entry.artifact_id == result.manifest_artifact_id
    assert manifest_entry.semantic_sha256 == result.manifest.semantic_content_sha256
    assert manifest_entry.sha256 == result.manifest_sha256
    assert manifest_entry.bytes == len(manifest_bytes)
    assert exact_bytes_sha256(manifest_bytes) == result.manifest_sha256
    assert registry.resolve_blob(result.manifest_artifact_id).read_bytes() == manifest_bytes
    assert (
        result.output_directory / "camera-video-export-manifest.json"
    ).read_bytes() == manifest_bytes

    embedded_artifact_ids = {
        result.manifest.source_artifact_id,
        result.manifest.mapping_profile_artifact_id,
        result.manifest.export_config_artifact_id,
        *(camera.video_artifact.artifact_id for camera in result.manifest.cameras),
        *(
            camera.timestamp_sidecar_artifact.artifact.artifact_id
            for camera in result.manifest.cameras
        ),
    }
    assert result.manifest_artifact_id not in embedded_artifact_ids


def test_same_directory_reuses_registry_derivation_and_verified_view(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    first_exporter = _FakeExporter()
    first = _service(first_exporter, registry).export_local(request)
    committed = registry.verify_derivation(first.logical_key)
    second_exporter = _FakeExporter()

    second = _service(second_exporter, registry).export_local(request)

    assert first_exporter.calls == list(CAMERA_IDS)
    assert second_exporter.calls == []
    assert second.derivation_reused is True
    assert second.materialized_view_reused is True
    assert second.logical_key == first.logical_key
    assert second.manifest_artifact_id == first.manifest_artifact_id
    assert second.manifest_sha256 == first.manifest_sha256
    assert second.manifest == first.manifest
    assert registry.verify_derivation(second.logical_key) == committed
    assert _registry_row_counts(registry) == (20, 1)


def test_new_output_path_reuses_derivation_and_materializes_identical_view(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    first = _service(_FakeExporter(), registry).export_local(request)
    new_request = replace(request, output_directory=tmp_path / "export-copy")
    replay_exporter = _FakeExporter()

    replay = _service(replay_exporter, registry).export_local(new_request)

    assert replay_exporter.calls == []
    assert replay.derivation_reused is True
    assert replay.materialized_view_reused is False
    assert replay.output_directory == new_request.output_directory.resolve()
    assert replay.logical_key == first.logical_key
    assert replay.manifest_artifact_id == first.manifest_artifact_id
    assert replay.manifest_sha256 == first.manifest_sha256
    assert replay.manifest == first.manifest
    assert _view_names(replay.output_directory) == _expected_view_names()
    assert {
        name: (replay.output_directory / name).read_bytes() for name in _expected_view_names()
    } == {name: (first.output_directory / name).read_bytes() for name in _expected_view_names()}
    assert _registry_row_counts(registry) == (20, 1)


@pytest.mark.parametrize("mutation", ["producer", "timestamp-schema"])
def test_reuse_rejects_coherent_foreign_output_provenance(
    tmp_path: Path,
    mutation: str,
) -> None:
    request = _request(tmp_path)
    source_registry = LocalArtifactRegistry(tmp_path / "source-registry")
    published = _service(_FakeExporter(), source_registry).export_local(request)
    target_registry = LocalArtifactRegistry(tmp_path / f"target-registry-{mutation}")
    _publish_coherent_foreign_derivation(
        source_registry=source_registry,
        target_registry=target_registry,
        published=published,
        mutation=mutation,
    )
    committed = target_registry.verify_derivation(published.logical_key)
    replay_exporter = _FakeExporter()
    replay_request = replace(
        request,
        output_directory=tmp_path / f"foreign-view-{mutation}",
    )

    with pytest.raises(VideoExportRunError) as caught:
        _service(replay_exporter, target_registry).export_local(replay_request)

    assert caught.value.code is VideoExportRunErrorCode.MANIFEST_INVALID
    assert replay_exporter.calls == []
    assert not replay_request.output_directory.exists()
    assert target_registry.verify_derivation(published.logical_key) == committed


def test_tampered_existing_view_fails_closed_while_registry_stays_verified(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    first = _service(_FakeExporter(), registry).export_local(request)
    committed = registry.verify_derivation(first.logical_key)
    video_entry = first.manifest.cameras[0].video_artifact
    view_video = first.output_directory / "cam_01.mp4"
    original_bytes = view_video.read_bytes()
    view_video.write_bytes(bytes([original_bytes[0] ^ 1]) + original_bytes[1:])
    replay_exporter = _FakeExporter()

    with pytest.raises(VideoExportRunError) as caught:
        _service(replay_exporter, registry).export_local(request)

    assert caught.value.code is VideoExportRunErrorCode.MATERIALIZED_VIEW_FAILED
    assert replay_exporter.calls == []
    assert view_video.read_bytes() != original_bytes
    assert registry.verify_derivation(first.logical_key) == committed
    assert registry.resolve_blob(video_entry.artifact_id).read_bytes() == original_bytes
    assert _registry_row_counts(registry) == (20, 1)


def test_registry_miss_rejects_preexisting_unregistered_output_directory(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    request.output_directory.mkdir()
    (request.output_directory / "unregistered.txt").write_text("unregistered")
    registry = LocalArtifactRegistry(tmp_path / "registry")
    exporter = _FakeExporter()

    with pytest.raises(VideoExportRunError) as caught:
        _service(exporter, registry).export_local(request)

    assert caught.value.code is VideoExportRunErrorCode.OUTPUT_EXISTS
    assert exporter.calls == []
    assert (request.output_directory / "unregistered.txt").read_text() == "unregistered"
    assert _registry_row_counts(registry) == (0, 0)
    assert list(tmp_path.glob(".export.artifacts-*")) == []


def test_exporter_failure_commits_no_derivation_and_exposes_no_view(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    exporter = _FakeExporter(fail_on=CameraId.CAM_03)

    with pytest.raises(VideoExportError) as caught:
        _service(exporter, registry).export_local(request)

    assert caught.value.code is VideoExportErrorCode.REMUX_FAILED
    assert exporter.calls == list(CAMERA_IDS[:3])
    assert not request.output_directory.exists()
    assert _registry_row_counts(registry) == (0, 0)
    assert list(tmp_path.glob(".export.artifacts-*")) == []
    assert list(tmp_path.glob(".export.partial-*")) == []


def test_view_rename_failure_leaves_committed_derivation_for_export_free_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    exporter = _FakeExporter()
    target = request.output_directory.resolve()
    original_rename = Path.rename
    rename_failed = False

    def fail_first_view_publish(staging: Path, destination: Path) -> Path:
        nonlocal rename_failed
        if (
            not rename_failed
            and Path(destination) == target
            and staging.name.startswith(f".{target.name}.partial-")
        ):
            rename_failed = True
            raise OSError("injected view rename failure")
        return original_rename(staging, destination)

    monkeypatch.setattr(Path, "rename", fail_first_view_publish)

    with pytest.raises(VideoExportRunError) as caught:
        _service(exporter, registry).export_local(request)

    assert caught.value.code is VideoExportRunErrorCode.MATERIALIZED_VIEW_FAILED
    assert rename_failed is True
    assert exporter.calls == list(CAMERA_IDS)
    assert not target.exists()
    assert list(tmp_path.glob(".export.partial-*")) == []
    assert list(tmp_path.glob(".export.artifacts-*")) == []

    with sqlite3.connect(registry.database_path) as connection:
        derivation_rows = connection.execute(
            "SELECT logical_key, manifest_artifact_id FROM derivations"
        ).fetchall()
    assert len(derivation_rows) == 1
    logical_key, manifest_artifact_id = derivation_rows[0]
    committed = registry.verify_derivation(logical_key)
    assert len(committed.entries) == 20
    assert _registry_row_counts(registry) == (20, 1)

    recovery_request = replace(request, output_directory=tmp_path / "recovered-export")
    recovery_exporter = _FakeExporter()
    recovered = _service(recovery_exporter, registry).export_local(recovery_request)

    assert recovery_exporter.calls == []
    assert recovered.derivation_reused is True
    assert recovered.materialized_view_reused is False
    assert recovered.logical_key == logical_key
    assert recovered.manifest_artifact_id == manifest_artifact_id
    assert registry.verify_derivation(recovered.logical_key) == committed
    assert _view_names(recovered.output_directory) == _expected_view_names()
    assert list(tmp_path.glob(".recovered-export.partial-*")) == []


def test_commit_uncertainty_recovers_verified_derivation_without_reexport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    exporter = _FakeExporter()
    original_publish = registry.publish_derivation
    injected = False

    def publish_then_raise(**kwargs: object) -> object:
        nonlocal injected
        published = original_publish(**kwargs)  # type: ignore[arg-type]
        if not injected:
            injected = True
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.TRANSACTION_FAILED,
                "injected commit uncertainty",
            )
        return published

    monkeypatch.setattr(registry, "publish_derivation", publish_then_raise)

    recovered = _service(exporter, registry).export_local(request)

    assert injected is True
    assert exporter.calls == list(CAMERA_IDS)
    assert recovered.derivation_reused is True
    assert recovered.materialized_view_reused is False
    assert _view_names(recovered.output_directory) == _expected_view_names()
    assert registry.verify_derivation(recovered.logical_key).entries
    assert _registry_row_counts(registry) == (20, 1)
    assert list(tmp_path.glob(".export.artifacts-*")) == []

    replay_exporter = _FakeExporter()
    replay_request = replace(request, output_directory=tmp_path / "replayed-export")
    replayed = _service(replay_exporter, registry).export_local(replay_request)

    assert replay_exporter.calls == []
    assert replayed.derivation_reused is True
    assert replayed.manifest_artifact_id == recovered.manifest_artifact_id
    assert replayed.manifest_sha256 == recovered.manifest_sha256


def test_concurrent_registry_misses_converge_on_one_verified_derivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    requests = (
        replace(request, output_directory=tmp_path / "concurrent-a"),
        replace(request, output_directory=tmp_path / "concurrent-b"),
    )
    registry_root = tmp_path / "concurrent-registry"
    registries = (
        LocalArtifactRegistry(registry_root),
        LocalArtifactRegistry(registry_root),
    )
    initial_lookup_barrier = Barrier(2)

    def gate_first_lookup(registry: LocalArtifactRegistry) -> object:
        original_lookup = registry.lookup_derivation
        initial_lookup_pending = True

        def gated_lookup(logical_key: str) -> object:
            nonlocal initial_lookup_pending
            result = original_lookup(logical_key)
            if initial_lookup_pending:
                initial_lookup_pending = False
                initial_lookup_barrier.wait(timeout=30)
            return result

        return gated_lookup

    for registry in registries:
        monkeypatch.setattr(registry, "lookup_derivation", gate_first_lookup(registry))

    exporters = (_FakeExporter(), _FakeExporter())
    services = (
        RegisteredSixCameraVideoExportService(
            exporters[0],
            registries[0],
            clock=lambda: _FIXED_NOW,
        ),
        RegisteredSixCameraVideoExportService(
            exporters[1],
            registries[1],
            clock=lambda: _FIXED_NOW + timedelta(seconds=1),
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(service.export_local, export_request)
            for service, export_request in zip(services, requests, strict=True)
        )
        results = tuple(future.result(timeout=60) for future in futures)

    assert exporters[0].calls == list(CAMERA_IDS)
    assert exporters[1].calls == list(CAMERA_IDS)
    assert sorted(result.derivation_reused for result in results) == [False, True]
    assert results[0].logical_key == results[1].logical_key
    assert results[0].manifest_artifact_id == results[1].manifest_artifact_id
    assert results[0].manifest_sha256 == results[1].manifest_sha256
    assert _registry_row_counts(registries[0]) == (20, 1)
    assert registries[0].verify_derivation(results[0].logical_key) == registries[
        1
    ].verify_derivation(results[1].logical_key)
    assert {
        name: (results[0].output_directory / name).read_bytes() for name in _expected_view_names()
    } == {
        name: (results[1].output_directory / name).read_bytes() for name in _expected_view_names()
    }
