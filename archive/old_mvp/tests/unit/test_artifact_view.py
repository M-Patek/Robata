from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import robata.application.artifact_view as artifact_view
from robata.application.artifact_view import (
    ArtifactViewError,
    ArtifactViewErrorCode,
    CameraVideoViewPublication,
    materialize_camera_video_view,
)
from robata.contracts.artifacts import (
    ArtifactLifecycle,
    ArtifactLocator,
    ArtifactParent,
    ArtifactParentRelation,
    ArtifactProducer,
    ArtifactRegistryEntry,
    ArtifactRegistrySnapshot,
    ArtifactType,
    SchemaArtifactReference,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
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
    TimestampSidecarArtifactV2,
    camera_video_manifest_v2_semantic_projection,
)
from robata.ports.artifact_registry import ArtifactRegistryError, ArtifactRegistryErrorCode

_CREATED_AT = "2026-07-18T12:00:00+08:00"
_MEDIA_TYPES = {
    ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST: "application/json",
    ArtifactType.CAMERA_VIDEO_MP4: "video/mp4",
    ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP: "application/x-ndjson",
    ArtifactType.EXPORT_CONFIG: "application/json",
    ArtifactType.JSON_SCHEMA: "application/schema+json",
    ArtifactType.MAPPING_PROFILE: "application/json",
    ArtifactType.RAW_MCAP: "application/x-mcap",
}


def _uuid(number: int) -> str:
    return f"00000000-0000-5000-8000-{number:012x}"


def _sha256(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _Fixture:
    snapshot: ArtifactRegistrySnapshot
    manifest_artifact_id: str
    manifest: CameraVideoExportManifestV2
    blob_paths: dict[str, Path]
    blob_bytes: dict[str, bytes]
    filename_bytes: dict[str, bytes]


@dataclass
class _FakeRegistry:
    blob_paths: dict[str, Path]
    calls: list[str] = field(default_factory=list)

    def resolve_blob(self, artifact_id: str) -> Path:
        self.calls.append(artifact_id)
        try:
            return self.blob_paths[artifact_id]
        except KeyError as error:
            raise ArtifactRegistryError(
                ArtifactRegistryErrorCode.ARTIFACT_NOT_FOUND,
                f"missing test blob {artifact_id}",
            ) from error


def _fixture(tmp_path: Path) -> _Fixture:
    producer = ArtifactProducer(
        name="robata.test-exporter",
        version="1.0.0",
        canonical_config_sha256=_sha256("producer-config"),
    )
    lifecycle = ArtifactLifecycle(state="ACTIVE", policy_version="local-evidence-v1")
    entries: dict[str, ArtifactRegistryEntry] = {}
    blob_bytes: dict[str, bytes] = {}

    def make_entry(
        name: str,
        artifact_type: ArtifactType,
        data: bytes,
        number: int,
        *,
        parents: tuple[ArtifactParent, ...] = (),
        payload_schema_ref: SchemaArtifactReference | None = None,
        semantic_digest: str | None = None,
        locator: tuple[str, str] | None = None,
    ) -> ArtifactRegistryEntry:
        digest = _sha256(data)
        uri, object_version = locator or (f"artifact://test/{name}", digest)
        entry = ArtifactRegistryEntry(
            schema_version="2.0",
            artifact_id=_uuid(number),
            artifact_type=artifact_type,
            semantic_sha256=semantic_digest or _sha256(f"semantic:{name}"),
            locator=ArtifactLocator(uri=uri, object_version=object_version),
            sha256=digest,
            bytes=len(data),
            media_type=_MEDIA_TYPES[artifact_type],
            producer=producer,
            lifecycle=lifecycle,
            parents=tuple(
                sorted(parents, key=lambda parent: (parent.relation.value, parent.artifact_id))
            ),
            payload_schema_ref=payload_schema_ref,
            created_at=_CREATED_AT,
        )
        entries[name] = entry
        blob_bytes[entry.artifact_id] = data
        return entry

    schema = make_entry(
        "schema",
        ArtifactType.JSON_SCHEMA,
        b'{"type":"object"}',
        1,
        locator=("https://schemas.robata.dev/test-artifact", "1.0.0"),
    )
    schema_ref = SchemaArtifactReference(
        schema_id=schema.locator.uri,
        version=schema.locator.object_version,
        artifact_id=schema.artifact_id,
        sha256=schema.sha256,
    )
    source = make_entry("source", ArtifactType.RAW_MCAP, b"synthetic-mcap", 10)
    mapping = make_entry(
        "mapping",
        ArtifactType.MAPPING_PROFILE,
        b'{"profile":"test"}',
        11,
        payload_schema_ref=schema_ref,
    )
    config = make_entry(
        "config",
        ArtifactType.EXPORT_CONFIG,
        b'{"exporter":"test"}',
        12,
        payload_schema_ref=schema_ref,
    )
    shared_parents = tuple(
        sorted(
            (
                ArtifactParent(
                    artifact_id=config.artifact_id,
                    relation=ArtifactParentRelation.EXPORT_CONFIG,
                ),
                ArtifactParent(
                    artifact_id=mapping.artifact_id,
                    relation=ArtifactParentRelation.MAPPING_PROFILE,
                ),
                ArtifactParent(
                    artifact_id=source.artifact_id,
                    relation=ArtifactParentRelation.SOURCE_CONTENT,
                ),
            ),
            key=lambda parent: (parent.relation.value, parent.artifact_id),
        )
    )

    records: list[CameraVideoExportRecordV2] = []
    filename_bytes: dict[str, bytes] = {}
    for index, camera_id in enumerate(CAMERA_IDS, start=1):
        video_bytes = f"mp4:{camera_id.value}".encode()
        timestamp_bytes = f'{{"camera":"{camera_id.value}","packet":0}}\n'.encode()
        video = make_entry(
            f"video-{camera_id.value}",
            ArtifactType.CAMERA_VIDEO_MP4,
            video_bytes,
            20 + index,
            parents=shared_parents,
        )
        timestamp = make_entry(
            f"timestamp-{camera_id.value}",
            ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
            timestamp_bytes,
            30 + index,
            parents=shared_parents,
            payload_schema_ref=schema_ref,
        )
        filename_bytes[f"{camera_id.value}.mp4"] = video_bytes
        filename_bytes[f"{camera_id.value}.timestamps.jsonl"] = timestamp_bytes
        records.append(
            CameraVideoExportRecordV2(
                camera_id=camera_id,
                source=SourceVideoStream(
                    topic=f"/camera/{index}",
                    channel_id=index,
                    schema_name="foxglove.CompressedImage",
                    codec="h264",
                ),
                input_message_count=2,
                source_first_observed_message_ns=100,
                source_last_observed_message_ns=200,
                export_first_observed_source_message_ns=200,
                export_last_observed_source_message_ns=200,
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
                exported_packet_count=1,
                exported_frame_count=1,
                keyframe_count=1,
                width=1600,
                height=1300,
                video_artifact=video,
                timestamp_sidecar_artifact=TimestampSidecarArtifactV2(
                    artifact=timestamp,
                    row_count=1,
                ),
                media_time_mapping=MediaTimeMapping(
                    zero_source_ns=200,
                    time_base_numerator=1,
                    time_base_denominator=1_000_000_000,
                    first_pts=0,
                    last_pts=0,
                    last_duration=100,
                    tail_duration_policy=TailDurationPolicy.MEDIAN_POSITIVE_INTERVAL,
                    rounding="HALF_EVEN",
                    max_rounding_error_ns=0,
                ),
            )
        )

    manifest_fields: dict[str, Any] = {
        "schema_version": "2.0",
        "schema_ref": schema_ref,
        "semantic_content_sha256": "0" * 64,
        "execution_mode": VideoExportExecutionMode.LOCAL_DEVELOPMENT_OVERRIDE,
        "recording_identity": _sha256("recording"),
        "source_content_sha256": source.sha256,
        "source_size_bytes": source.bytes,
        "source_artifact_id": source.artifact_id,
        "mapping_profile_artifact_id": mapping.artifact_id,
        "export_config_artifact_id": config.artifact_id,
        "mapping_profile": MappingProfileReference(
            version="test-v1",
            digest=mapping.semantic_sha256,
            approved=False,
        ),
        "ready_manifest_id": None,
        "ready_manifest_semantic_sha256": None,
        "alignment_id": None,
        "alignment_semantic_sha256": None,
        "alignment_status": VideoExportAlignmentStatus.UNVERIFIED,
        "exporter": VideoExporterIdentity(
            name=producer.name,
            version=producer.version,
            mode=VideoExporterMode.REMUX,
            export_profile_id="test-remux",
            profile_version="1.0.0",
            canonical_config_sha256=producer.canonical_config_sha256,
        ),
        "cameras": tuple(records),
    }
    draft = CameraVideoExportManifestV2.model_construct(**manifest_fields)
    manifest_fields["semantic_content_sha256"] = semantic_sha256(
        camera_video_manifest_v2_semantic_projection(draft)
    )
    manifest = CameraVideoExportManifestV2.model_validate(manifest_fields, strict=True)
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_parents = (
        shared_parents
        + tuple(
            ArtifactParent(
                artifact_id=record.timestamp_sidecar_artifact.artifact.artifact_id,
                relation=ArtifactParentRelation.TIMESTAMP_OUTPUT,
            )
            for record in records
        )
        + tuple(
            ArtifactParent(
                artifact_id=record.video_artifact.artifact_id,
                relation=ArtifactParentRelation.VIDEO_OUTPUT,
            )
            for record in records
        )
    )
    manifest_entry = make_entry(
        "manifest",
        ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST,
        manifest_bytes,
        40,
        parents=manifest_parents,
        payload_schema_ref=schema_ref,
        semantic_digest=manifest.semantic_content_sha256,
    )
    filename_bytes["camera-video-export-manifest.json"] = manifest_bytes
    snapshot = ArtifactRegistrySnapshot(
        schema_version="2.0",
        entries=tuple(sorted(entries.values(), key=lambda entry: entry.artifact_id)),
    )

    blob_directory = tmp_path / "registry-blobs"
    blob_directory.mkdir()
    blob_paths: dict[str, Path] = {}
    for artifact_id, data in blob_bytes.items():
        path = blob_directory / artifact_id
        path.write_bytes(data)
        blob_paths[artifact_id] = path
    return _Fixture(
        snapshot=snapshot,
        manifest_artifact_id=manifest_entry.artifact_id,
        manifest=manifest,
        blob_paths=blob_paths,
        blob_bytes=blob_bytes,
        filename_bytes=filename_bytes,
    )


def _materialize(
    fixture: _Fixture,
    registry: _FakeRegistry,
    output: Path,
) -> CameraVideoViewPublication:
    return materialize_camera_video_view(
        registry=registry,  # type: ignore[arg-type]
        snapshot=fixture.snapshot,
        manifest_artifact_id=fixture.manifest_artifact_id,
        manifest=fixture.manifest,
        output_directory=output,
    )


def test_materializes_exact_13_file_view_from_registry_blobs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"

    publication = _materialize(fixture, registry, output)

    assert publication == CameraVideoViewPublication(output_directory=output, reused=False)
    assert {path.name for path in output.iterdir()} == set(fixture.filename_bytes)
    assert len(fixture.filename_bytes) == 13
    for filename, expected_bytes in fixture.filename_bytes.items():
        assert (output / filename).read_bytes() == expected_bytes
    assert len(registry.calls) == 13
    assert list(tmp_path.glob(".view.partial-*")) == []


def test_materialized_files_are_copies_not_hardlinks(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"
    _materialize(fixture, registry, output)
    artifact_id = fixture.manifest.cameras[0].video_artifact.artifact_id
    registry_bytes_before = fixture.blob_paths[artifact_id].read_bytes()

    (output / "cam_01.mp4").write_bytes(b"changed-view")

    assert fixture.blob_paths[artifact_id].read_bytes() == registry_bytes_before


def test_existing_complete_view_is_verified_and_reused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"
    _materialize(fixture, registry, output)
    manifest_stat = (output / "camera-video-export-manifest.json").stat()

    publication = _materialize(fixture, registry, output)

    assert publication.reused is True
    assert publication.output_directory == output
    assert (output / "camera-video-export-manifest.json").stat() == manifest_stat
    assert len(registry.calls) == 26


@pytest.mark.parametrize("damage", ["tampered", "missing", "extra"])
def test_invalid_existing_view_fails_closed(tmp_path: Path, damage: str) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"
    _materialize(fixture, registry, output)
    if damage == "tampered":
        (output / "cam_02.mp4").write_bytes(b"tampered")
    elif damage == "missing":
        (output / "cam_02.mp4").unlink()
    else:
        (output / "unexpected.txt").write_text("extra")

    with pytest.raises(ArtifactViewError) as raised:
        _materialize(fixture, registry, output)

    assert raised.value.code is ArtifactViewErrorCode.EXISTING_VIEW_INVALID
    assert output.exists()
    if damage == "extra":
        assert (output / "unexpected.txt").read_text() == "extra"


def test_symbolic_link_in_existing_view_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"
    _materialize(fixture, registry, output)
    video = output / "cam_03.mp4"
    actual = tmp_path / "actual.mp4"
    actual.write_bytes(video.read_bytes())
    video.unlink()
    try:
        video.symlink_to(actual)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(ArtifactViewError) as raised:
        _materialize(fixture, registry, output)

    assert raised.value.code is ArtifactViewErrorCode.EXISTING_VIEW_INVALID
    assert video.is_symlink()


def test_missing_registry_blob_publishes_nothing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    missing_id = fixture.manifest.cameras[4].video_artifact.artifact_id
    registry_paths = dict(fixture.blob_paths)
    registry_paths.pop(missing_id)
    registry = _FakeRegistry(registry_paths)
    output = tmp_path / "view"

    with pytest.raises(ArtifactViewError) as raised:
        _materialize(fixture, registry, output)

    assert raised.value.code is ArtifactViewErrorCode.REGISTRY_BLOB_UNAVAILABLE
    assert not output.exists()
    assert list(tmp_path.glob(".view.partial-*")) == []


def test_tampered_registry_blob_is_rejected_even_when_view_exists(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"
    _materialize(fixture, registry, output)
    artifact_id = fixture.manifest.cameras[1].video_artifact.artifact_id
    registry_path = fixture.blob_paths[artifact_id]
    registry_path.write_bytes(b"x" * len(registry_path.read_bytes()))

    with pytest.raises(ArtifactViewError) as raised:
        _materialize(fixture, registry, output)

    assert raised.value.code is ArtifactViewErrorCode.REGISTRY_BLOB_INVALID
    assert (output / "cam_02.mp4").read_bytes() == fixture.filename_bytes["cam_02.mp4"]


def test_rename_race_preserves_competing_target_registry_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"
    registry_before = {
        artifact_id: path.read_bytes() for artifact_id, path in fixture.blob_paths.items()
    }

    def competing_rename(_source: Path, target: str | Path) -> Path:
        competing_target = Path(target)
        competing_target.mkdir()
        (competing_target / "winner.txt").write_text("winner")
        raise FileExistsError("injected publication race")

    monkeypatch.setattr(artifact_view, "_rename_no_replace", competing_rename)

    with pytest.raises(ArtifactViewError) as raised:
        _materialize(fixture, registry, output)

    assert raised.value.code is ArtifactViewErrorCode.OUTPUT_EXISTS
    assert (output / "winner.txt").read_text() == "winner"
    assert list(tmp_path.glob(".view.partial-*")) == []
    assert {
        artifact_id: path.read_bytes() for artifact_id, path in fixture.blob_paths.items()
    } == registry_before


def test_rename_failure_preserves_registry_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    registry = _FakeRegistry(fixture.blob_paths)
    output = tmp_path / "view"
    registry_before = dict(fixture.blob_bytes)

    def failed_rename(_source: Path, _target: str | Path) -> Path:
        raise OSError("injected rename failure")

    monkeypatch.setattr(artifact_view, "_rename_no_replace", failed_rename)

    with pytest.raises(ArtifactViewError) as raised:
        _materialize(fixture, registry, output)

    assert raised.value.code is ArtifactViewErrorCode.ATOMIC_PUBLISH_FAILED
    assert not output.exists()
    assert list(tmp_path.glob(".view.partial-*")) == []
    assert {
        artifact_id: path.read_bytes() for artifact_id, path in fixture.blob_paths.items()
    } == registry_before
