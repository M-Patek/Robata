from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from robata.adapters.local_artifact_registry import (
    LocalArtifactRegistry,
    deterministic_local_artifact_id,
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
from robata.contracts.hashing import canonical_json_bytes
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


@dataclass(frozen=True, slots=True)
class _RegistryFixture:
    snapshot: ArtifactRegistrySnapshot
    sources: dict[str, bytes]
    by_name: dict[str, ArtifactRegistryEntry]
    manifest_artifact_id: str


def _sha256(value: str | bytes) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _fixture(
    seed: str = "a",
    *,
    schema_locator: tuple[str, str] | None = None,
) -> _RegistryFixture:
    producer = ArtifactProducer(
        name="test-video-exporter",
        version="1.0",
        canonical_config_sha256=_sha256(f"producer-config:{seed}"),
    )
    lifecycle = ArtifactLifecycle(state="ACTIVE", policy_version="1.0")
    entries: dict[str, ArtifactRegistryEntry] = {}
    sources: dict[str, bytes] = {}

    def make_entry(
        name: str,
        artifact_type: ArtifactType,
        *,
        parents: tuple[ArtifactParent, ...] = (),
        payload_schema_ref: SchemaArtifactReference | None = None,
        locator: tuple[str, str] | None = None,
    ) -> ArtifactRegistryEntry:
        data = f"exact-bytes:{seed}:{name}".encode()
        semantic_digest = _sha256(f"semantic:{seed}:{name}")
        artifact_id = deterministic_local_artifact_id(artifact_type, semantic_digest)
        uri, object_version = locator or (f"artifact://local/{seed}/{name}", "1.0")
        entry = ArtifactRegistryEntry(
            schema_version="2.0",
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            semantic_sha256=semantic_digest,
            locator=ArtifactLocator(uri=uri, object_version=object_version),
            sha256=_sha256(data),
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
        sources[artifact_id] = data
        return entry

    schema = make_entry(
        "schema",
        ArtifactType.JSON_SCHEMA,
        locator=schema_locator,
    )
    schema_ref = SchemaArtifactReference(
        schema_id=schema.locator.uri,
        version=schema.locator.object_version,
        artifact_id=schema.artifact_id,
        sha256=schema.sha256,
    )
    source = make_entry("source", ArtifactType.RAW_MCAP)
    export_config = make_entry(
        "export-config",
        ArtifactType.EXPORT_CONFIG,
        payload_schema_ref=schema_ref,
    )
    mapping_profile = make_entry(
        "mapping-profile",
        ArtifactType.MAPPING_PROFILE,
        payload_schema_ref=schema_ref,
    )

    shared_parents = (
        ArtifactParent(
            artifact_id=export_config.artifact_id,
            relation=ArtifactParentRelation.EXPORT_CONFIG,
        ),
        ArtifactParent(
            artifact_id=mapping_profile.artifact_id,
            relation=ArtifactParentRelation.MAPPING_PROFILE,
        ),
        ArtifactParent(
            artifact_id=source.artifact_id,
            relation=ArtifactParentRelation.SOURCE_CONTENT,
        ),
    )
    videos = tuple(
        make_entry(
            f"video-{index}",
            ArtifactType.CAMERA_VIDEO_MP4,
            parents=shared_parents,
        )
        for index in range(6)
    )
    timestamp_maps = tuple(
        make_entry(
            f"timestamp-{index}",
            ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
            parents=shared_parents,
            payload_schema_ref=schema_ref,
        )
        for index in range(6)
    )
    manifest_parents = (
        shared_parents
        + tuple(
            ArtifactParent(
                artifact_id=timestamp_map.artifact_id,
                relation=ArtifactParentRelation.TIMESTAMP_OUTPUT,
            )
            for timestamp_map in timestamp_maps
        )
        + tuple(
            ArtifactParent(
                artifact_id=video.artifact_id,
                relation=ArtifactParentRelation.VIDEO_OUTPUT,
            )
            for video in videos
        )
    )
    manifest = make_entry(
        "manifest",
        ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST,
        parents=manifest_parents,
        payload_schema_ref=schema_ref,
    )
    snapshot = ArtifactRegistrySnapshot(
        schema_version="2.0",
        entries=tuple(sorted(entries.values(), key=lambda entry: entry.artifact_id)),
    )
    return _RegistryFixture(
        snapshot=snapshot,
        sources=sources,
        by_name=entries,
        manifest_artifact_id=manifest.artifact_id,
    )


def _publish(
    registry: LocalArtifactRegistry,
    fixture: _RegistryFixture,
    *,
    logical_key: str = "video-export:test-a",
) -> None:
    registry.publish_derivation(
        snapshot=fixture.snapshot,
        logical_key=logical_key,
        manifest_artifact_id=fixture.manifest_artifact_id,
        blob_sources=fixture.sources,
    )


def test_publish_resolve_load_lookup_and_verify_complete_derivation(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")

    source_entry = fixture.by_name["source"]
    assert (
        registry.allocate_artifact_id(
            source_entry.artifact_type,
            source_entry.semantic_sha256,
        )
        == source_entry.artifact_id
    )
    source_path = tmp_path / "source.mcap"
    source_path.write_bytes(fixture.sources[source_entry.artifact_id])
    blob_sources: dict[str, Path | bytes] = dict(fixture.sources)
    blob_sources[source_entry.artifact_id] = source_path

    published = registry.publish_derivation(
        snapshot=fixture.snapshot,
        logical_key="video-export:test-a",
        manifest_artifact_id=fixture.manifest_artifact_id,
        blob_sources=blob_sources,
    )

    assert published.reused is False
    assert published.snapshot == fixture.snapshot
    assert registry.database_path == registry.root / "registry.sqlite3"
    assert registry.database_path.is_file()
    for entry in fixture.snapshot.entries:
        blob = registry.resolve_blob(entry.artifact_id)
        assert blob == registry.blob_root / entry.sha256[:2] / entry.sha256
        assert blob.read_bytes() == fixture.sources[entry.artifact_id]

    assert registry.load_snapshot("video-export:test-a") == fixture.snapshot
    assert registry.verify_derivation("video-export:test-a") == fixture.snapshot
    reusable = registry.lookup_derivation("video-export:test-a")
    assert reusable is not None
    assert reusable.reused is True
    assert reusable.manifest_artifact_id == fixture.manifest_artifact_id
    source = fixture.by_name["source"]
    assert registry.lookup_artifact(source.artifact_type, source.semantic_sha256) == source
    assert registry.lookup_artifact(source.artifact_type, _sha256("absent")) is None

    with sqlite3.connect(registry.database_path) as connection:
        created_at = connection.execute(
            "SELECT created_at FROM derivations WHERE logical_key = ?",
            ("video-export:test-a",),
        ).fetchone()
    assert created_at == (_CREATED_AT,)


def test_replay_reuses_exact_request_without_duplicate_rows(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, fixture)

    replay = registry.publish_derivation(
        snapshot=fixture.snapshot,
        logical_key="video-export:test-a",
        manifest_artifact_id=fixture.manifest_artifact_id,
        blob_sources=fixture.sources,
    )

    assert replay.reused is True
    with sqlite3.connect(registry.database_path) as connection:
        counts = tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "artifacts",
                "artifact_locations",
                "artifact_edges",
                "derivations",
                "derivation_artifacts",
            )
        )
    assert counts == (17, 17, 51, 1, 17)


def test_replay_rejects_a_source_that_disagrees_with_existing_blob(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, fixture)
    sources = dict(fixture.sources)
    manifest_bytes = sources[fixture.manifest_artifact_id]
    sources[fixture.manifest_artifact_id] = bytes([manifest_bytes[0] ^ 1]) + manifest_bytes[1:]

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.publish_derivation(
            snapshot=fixture.snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=sources,
        )
    assert caught.value.code is ArtifactRegistryErrorCode.BLOB_DIGEST_MISMATCH


@pytest.mark.parametrize(
    ("replacement", "expected_code"),
    [
        (
            lambda value: bytes([value[0] ^ 1]) + value[1:],
            ArtifactRegistryErrorCode.BLOB_DIGEST_MISMATCH,
        ),
        (lambda value: value + b"x", ArtifactRegistryErrorCode.BLOB_SIZE_MISMATCH),
    ],
)
def test_publish_rejects_blob_not_matching_exact_entry(
    tmp_path: Path,
    replacement: object,
    expected_code: ArtifactRegistryErrorCode,
) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    sources = dict(fixture.sources)
    target = fixture.by_name["manifest"]
    assert callable(replacement)
    sources[target.artifact_id] = replacement(sources[target.artifact_id])

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.publish_derivation(
            snapshot=fixture.snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=sources,
        )

    assert caught.value.code is expected_code
    assert registry.lookup_derivation("video-export:test-a") is None


def test_coherent_blob_replacement_invalidates_lookup_and_replay(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, fixture)
    video = fixture.by_name["video-0"]
    blob = registry.resolve_blob(video.artifact_id)
    blob.write_bytes(b"x" * video.bytes)

    with pytest.raises(ArtifactRegistryError) as lookup_error:
        registry.lookup_derivation("video-export:test-a")
    assert lookup_error.value.code is ArtifactRegistryErrorCode.INTEGRITY_ERROR

    with pytest.raises(ArtifactRegistryError) as artifact_lookup_error:
        registry.lookup_artifact(video.artifact_type, video.semantic_sha256)
    assert artifact_lookup_error.value.code is ArtifactRegistryErrorCode.INTEGRITY_ERROR

    with pytest.raises(ArtifactRegistryError) as replay_error:
        registry.publish_derivation(
            snapshot=fixture.snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=fixture.sources,
        )
    assert replay_error.value.code is ArtifactRegistryErrorCode.BLOB_CONFLICT


def test_publish_requires_the_exact_blob_source_key_set(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    missing_sources = dict(fixture.sources)
    missing_sources.pop(fixture.manifest_artifact_id)

    with pytest.raises(ArtifactRegistryError) as missing_error:
        registry.publish_derivation(
            snapshot=fixture.snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=missing_sources,
        )
    assert missing_error.value.code is ArtifactRegistryErrorCode.BLOB_SOURCE_MISSING

    unexpected_sources = dict(fixture.sources)
    unexpected_sources["unused"] = b"unused"
    with pytest.raises(ArtifactRegistryError) as unexpected_error:
        registry.publish_derivation(
            snapshot=fixture.snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=unexpected_sources,
        )
    assert unexpected_error.value.code is ArtifactRegistryErrorCode.BLOB_SOURCE_UNEXPECTED


def test_same_semantic_identity_with_different_exact_bytes_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, fixture)
    source = fixture.by_name["source"]
    replacement_bytes = b"different-exact-source-bytes"
    replacement_entry = source.model_copy(
        update={
            "sha256": _sha256(replacement_bytes),
            "bytes": len(replacement_bytes),
        }
    )
    entries = tuple(
        sorted(
            (
                replacement_entry if entry.artifact_id == source.artifact_id else entry
                for entry in fixture.snapshot.entries
            ),
            key=lambda entry: entry.artifact_id,
        )
    )
    conflicting_snapshot = ArtifactRegistrySnapshot(schema_version="2.0", entries=entries)
    sources = dict(fixture.sources)
    sources[source.artifact_id] = replacement_bytes

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.publish_derivation(
            snapshot=conflicting_snapshot,
            logical_key="video-export:test-conflict",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=sources,
        )

    assert caught.value.code is ArtifactRegistryErrorCode.ARTIFACT_CONFLICT
    assert registry.lookup_derivation("video-export:test-conflict") is None


def test_locator_object_version_cannot_be_rebound(tmp_path: Path) -> None:
    first = _fixture("a")
    shared_locator = (
        first.by_name["schema"].locator.uri,
        first.by_name["schema"].locator.object_version,
    )
    second = _fixture("b", schema_locator=shared_locator)
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, first)

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.publish_derivation(
            snapshot=second.snapshot,
            logical_key="video-export:test-b",
            manifest_artifact_id=second.manifest_artifact_id,
            blob_sources=second.sources,
        )

    assert caught.value.code is ArtifactRegistryErrorCode.LOCATION_CONFLICT
    assert registry.lookup_derivation("video-export:test-b") is None


def test_missing_parent_is_rejected_before_any_derivation_commit(tmp_path: Path) -> None:
    fixture = _fixture()
    video = fixture.by_name["video-0"]
    missing_id = "00000000-0000-0000-0000-000000000001"
    invalid_video = video.model_copy(
        update={
            "parents": tuple(
                ArtifactParent(artifact_id=missing_id, relation=parent.relation)
                if parent.relation is ArtifactParentRelation.SOURCE_CONTENT
                else parent
                for parent in video.parents
            )
        }
    )
    invalid_snapshot = fixture.snapshot.model_copy(
        update={
            "entries": tuple(
                invalid_video if entry.artifact_id == video.artifact_id else entry
                for entry in fixture.snapshot.entries
            )
        }
    )
    registry = LocalArtifactRegistry(tmp_path / "registry")

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.publish_derivation(
            snapshot=invalid_snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=fixture.sources,
        )

    assert caught.value.code is ArtifactRegistryErrorCode.MISSING_PARENT
    assert registry.lookup_derivation("video-export:test-a") is None


def test_cycle_is_rejected_before_any_derivation_commit(tmp_path: Path) -> None:
    fixture = _fixture()
    export_config = fixture.by_name["export-config"]
    cyclic_config = export_config.model_copy(
        update={
            "parents": (
                ArtifactParent(
                    artifact_id=fixture.manifest_artifact_id,
                    relation=ArtifactParentRelation.SOURCE_CONTENT,
                ),
            )
        }
    )
    cyclic_snapshot = fixture.snapshot.model_copy(
        update={
            "entries": tuple(
                cyclic_config if entry.artifact_id == export_config.artifact_id else entry
                for entry in fixture.snapshot.entries
            )
        }
    )
    registry = LocalArtifactRegistry(tmp_path / "registry")

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.publish_derivation(
            snapshot=cyclic_snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=fixture.sources,
        )

    assert caught.value.code is ArtifactRegistryErrorCode.GRAPH_CYCLE
    assert registry.lookup_derivation("video-export:test-a") is None


def test_disconnected_artifact_is_rejected_as_incomplete_dag(tmp_path: Path) -> None:
    fixture = _fixture()
    template = fixture.by_name["schema"]
    data = b"unreferenced-schema"
    semantic_digest = _sha256("semantic:unreferenced-schema")
    extra = template.model_copy(
        update={
            "artifact_id": deterministic_local_artifact_id(
                ArtifactType.JSON_SCHEMA,
                semantic_digest,
            ),
            "semantic_sha256": semantic_digest,
            "locator": ArtifactLocator(
                uri="artifact://local/a/unreferenced-schema",
                object_version="1.0",
            ),
            "sha256": _sha256(data),
            "bytes": len(data),
        }
    )
    incomplete_snapshot = ArtifactRegistrySnapshot(
        schema_version="2.0",
        entries=tuple(
            sorted((*fixture.snapshot.entries, extra), key=lambda entry: entry.artifact_id)
        ),
    )
    sources = dict(fixture.sources)
    sources[extra.artifact_id] = data
    registry = LocalArtifactRegistry(tmp_path / "registry")

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.publish_derivation(
            snapshot=incomplete_snapshot,
            logical_key="video-export:test-a",
            manifest_artifact_id=fixture.manifest_artifact_id,
            blob_sources=sources,
        )

    assert caught.value.code is ArtifactRegistryErrorCode.INCOMPLETE_DAG


def test_commit_failure_rolls_back_metadata_and_retry_can_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    root = tmp_path / "registry"
    registry = LocalArtifactRegistry(root)

    def fail_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected commit failure")

    monkeypatch.setattr(registry, "_commit", fail_commit)
    with pytest.raises(ArtifactRegistryError) as caught:
        _publish(registry, fixture)

    assert caught.value.code is ArtifactRegistryErrorCode.TRANSACTION_FAILED
    assert registry.lookup_derivation("video-export:test-a") is None

    retry_registry = LocalArtifactRegistry(root)
    result = retry_registry.publish_derivation(
        snapshot=fixture.snapshot,
        logical_key="video-export:test-a",
        manifest_artifact_id=fixture.manifest_artifact_id,
        blob_sources=fixture.sources,
    )
    assert result.reused is False


def test_typed_edge_tamper_invalidates_verified_lookup(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, fixture)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            """
            DELETE FROM artifact_edges
            WHERE child_artifact_id = ? AND relation = ?
            """,
            (
                fixture.manifest_artifact_id,
                ArtifactParentRelation.VIDEO_OUTPUT.value,
            ),
        )

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.lookup_derivation("video-export:test-a")
    assert caught.value.code is ArtifactRegistryErrorCode.INTEGRITY_ERROR


def test_coherent_entry_json_tamper_invalidates_semantic_lookup(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, fixture)
    source = fixture.by_name["source"]
    tampered = source.model_copy(update={"created_at": "2026-07-19T12:00:00+08:00"})
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute(
            "UPDATE artifacts SET entry_json = ? WHERE artifact_id = ?",
            (canonical_json_bytes(tampered), source.artifact_id),
        )

    with pytest.raises(ArtifactRegistryError) as caught:
        registry.lookup_artifact(source.artifact_type, source.semantic_sha256)
    assert caught.value.code is ArtifactRegistryErrorCode.INTEGRITY_ERROR


def test_foreign_keys_restrict_deleting_committed_artifacts(tmp_path: Path) -> None:
    fixture = _fixture()
    registry = LocalArtifactRegistry(tmp_path / "registry")
    _publish(registry, fixture)

    connection = sqlite3.connect(registry.database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM artifacts WHERE artifact_id = ?",
                (fixture.manifest_artifact_id,),
            )
        connection.rollback()
    finally:
        connection.close()
    assert registry.verify_derivation("video-export:test-a") == fixture.snapshot
