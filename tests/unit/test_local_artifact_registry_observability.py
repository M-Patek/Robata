from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from robata.adapters.local_artifact_registry import LocalArtifactRegistry
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
from robata.contracts.hashing import exact_bytes_sha256
from robata.ports.artifact_registry import ArtifactRegistryError, ArtifactRegistryErrorCode
from robata.runtime.observability import RuntimeProfileRecorder

_CREATED_AT = "2026-07-22T12:00:00Z"
_MEDIA_TYPES = {
    ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST: "application/json",
    ArtifactType.CAMERA_VIDEO_MP4: "video/mp4",
    ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP: "application/x-ndjson",
    ArtifactType.EXPORT_CONFIG: "application/json",
    ArtifactType.JSON_SCHEMA: "application/schema+json",
    ArtifactType.MAPPING_PROFILE: "application/json",
    ArtifactType.RAW_MCAP: "application/x-mcap",
}


def _publication_fixture(
    registry: LocalArtifactRegistry,
) -> tuple[ArtifactRegistrySnapshot, str, dict[str, bytes], ArtifactRegistryEntry]:
    blobs: dict[str, bytes] = {}
    producer = ArtifactProducer(
        name="robata-observability-fixture",
        version="1.0.0",
        canonical_config_sha256=exact_bytes_sha256(b"fixture-config"),
    )
    lifecycle = ArtifactLifecycle(state="ACTIVE", policy_version="retention-v1")

    def entry(
        name: str,
        artifact_type: ArtifactType,
        *,
        parents: tuple[ArtifactParent, ...] = (),
        payload_schema_ref: SchemaArtifactReference | None = None,
    ) -> ArtifactRegistryEntry:
        blob = f"{name}-exact-bytes".encode()
        semantic_sha256 = exact_bytes_sha256(f"{name}-semantic".encode())
        artifact_id = registry.allocate_artifact_id(artifact_type, semantic_sha256)
        value = ArtifactRegistryEntry(
            schema_version="2.0",
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            semantic_sha256=semantic_sha256,
            locator=ArtifactLocator(
                uri=f"fixture://artifacts/{name}",
                object_version="1.0",
            ),
            sha256=exact_bytes_sha256(blob),
            bytes=len(blob),
            media_type=_MEDIA_TYPES[artifact_type],
            producer=producer,
            lifecycle=lifecycle,
            parents=parents,
            payload_schema_ref=payload_schema_ref,
            created_at=_CREATED_AT,
        )
        blobs[value.artifact_id] = blob
        return value

    schema = entry("schema", ArtifactType.JSON_SCHEMA)
    schema_reference = SchemaArtifactReference(
        schema_id=schema.locator.uri,
        version=schema.locator.object_version,
        artifact_id=schema.artifact_id,
        sha256=schema.sha256,
    )
    source = entry("source", ArtifactType.RAW_MCAP)
    mapping = entry(
        "mapping",
        ArtifactType.MAPPING_PROFILE,
        payload_schema_ref=schema_reference,
    )
    config = entry(
        "config",
        ArtifactType.EXPORT_CONFIG,
        payload_schema_ref=schema_reference,
    )
    input_parents = tuple(
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
    videos = tuple(
        entry(
            f"camera-{ordinal}-video",
            ArtifactType.CAMERA_VIDEO_MP4,
            parents=input_parents,
        )
        for ordinal in range(6)
    )
    timestamp_maps = tuple(
        entry(
            f"camera-{ordinal}-timestamps",
            ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
            parents=input_parents,
            payload_schema_ref=schema_reference,
        )
        for ordinal in range(6)
    )
    manifest_parents = tuple(
        sorted(
            (
                *input_parents,
                *(
                    ArtifactParent(
                        artifact_id=value.artifact_id,
                        relation=ArtifactParentRelation.TIMESTAMP_OUTPUT,
                    )
                    for value in timestamp_maps
                ),
                *(
                    ArtifactParent(
                        artifact_id=value.artifact_id,
                        relation=ArtifactParentRelation.VIDEO_OUTPUT,
                    )
                    for value in videos
                ),
            ),
            key=lambda parent: (parent.relation.value, parent.artifact_id),
        )
    )
    manifest = entry(
        "manifest",
        ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST,
        parents=manifest_parents,
        payload_schema_ref=schema_reference,
    )
    snapshot = ArtifactRegistrySnapshot(
        schema_version="2.0",
        entries=tuple(
            sorted(
                (
                    schema,
                    source,
                    mapping,
                    config,
                    *videos,
                    *timestamp_maps,
                    manifest,
                ),
                key=lambda value: value.artifact_id,
            )
        ),
    )
    return snapshot, manifest.artifact_id, blobs, source


def _counter_map(
    recorder: RuntimeProfileRecorder,
    name: str,
) -> dict[tuple[tuple[str, object], ...], int]:
    return {
        tuple((attribute.name, attribute.value) for attribute in counter.attributes): counter.value
        for counter in recorder.snapshot().counters
        if counter.name == name
    }


def test_observes_publication_reads_and_actual_commit_failure_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    root = tmp_path / "artifact-registry"
    registry = LocalArtifactRegistry(root, runtime_observer=recorder)
    snapshot, manifest_artifact_id, blobs, source = _publication_fixture(registry)

    published = registry.publish_derivation(
        snapshot=snapshot,
        logical_key="observed-success",
        manifest_artifact_id=manifest_artifact_id,
        blob_sources=blobs,
    )
    assert published.reused is False
    assert registry.lookup_artifact(source.artifact_type, source.semantic_sha256) == source

    def fail_commit(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected commit failure")

    monkeypatch.setattr(registry, "_commit", fail_commit)
    with pytest.raises(ArtifactRegistryError) as raised:
        registry.publish_derivation(
            snapshot=snapshot,
            logical_key="observed-rollback",
            manifest_artifact_id=manifest_artifact_id,
            blob_sources=blobs,
        )
    assert raised.value.code is ArtifactRegistryErrorCode.TRANSACTION_FAILED
    assert LocalArtifactRegistry(root).lookup_derivation("observed-rollback") is None

    runtime = recorder.snapshot()
    assert (
        sum(span.name == "sqlite.artifact_registry.initialization" for span in runtime.spans) == 1
    )
    assert _counter_map(recorder, "sqlite.artifact_registry.transactions") == {
        (("operation", "initialize_schema"), ("write", True)): 1,
        (("operation", "lookup_artifact"), ("write", False)): 1,
        (("operation", "publish_derivation"), ("write", True)): 2,
        (("operation", "verify_derivation"), ("write", False)): 1,
    }
    assert _counter_map(recorder, "sqlite.artifact_registry.commits") == {
        (("operation", "initialize_schema"), ("write", True)): 1,
        (("operation", "lookup_artifact"), ("write", False)): 1,
        (("operation", "publish_derivation"), ("write", True)): 1,
        (("operation", "verify_derivation"), ("write", False)): 1,
    }
    assert _counter_map(recorder, "sqlite.artifact_registry.rollbacks") == {
        (("operation", "publish_derivation"), ("write", True)): 1,
    }
    assert _counter_map(recorder, "sqlite.artifact_registry.commit_failures") == {
        (("operation", "publish_derivation"), ("write", True)): 1,
    }
    assert (
        _counter_map(
            recorder,
            "sqlite.artifact_registry.transaction_outcomes_unknown",
        )
        == {}
    )
    assert sum(span.name == "sqlite.artifact_registry.transaction" for span in runtime.spans) == 4


def test_business_failure_is_observed_as_rollback_not_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    registry = LocalArtifactRegistry(
        tmp_path / "artifact-registry",
        runtime_observer=recorder,
    )
    snapshot, manifest_artifact_id, blobs, _source = _publication_fixture(registry)

    def fail_graph_verification(_connection: sqlite3.Connection) -> None:
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.INTEGRITY_ERROR,
            "injected business failure",
        )

    monkeypatch.setattr(registry, "_verify_database_graph", fail_graph_verification)
    with pytest.raises(ArtifactRegistryError) as raised:
        registry.publish_derivation(
            snapshot=snapshot,
            logical_key="business-failure",
            manifest_artifact_id=manifest_artifact_id,
            blob_sources=blobs,
        )
    assert raised.value.code is ArtifactRegistryErrorCode.INTEGRITY_ERROR

    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM derivations").fetchone() == (0,)
    assert _counter_map(recorder, "sqlite.artifact_registry.rollbacks") == {
        (("operation", "publish_derivation"), ("write", True)): 1,
    }
    assert _counter_map(recorder, "sqlite.artifact_registry.commit_failures") == {}
    assert (
        _counter_map(
            recorder,
            "sqlite.artifact_registry.transaction_outcomes_unknown",
        )
        == {}
    )


def test_commit_that_raises_after_commit_is_observed_as_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    root = tmp_path / "artifact-registry"
    registry = LocalArtifactRegistry(root, runtime_observer=recorder)
    snapshot, manifest_artifact_id, blobs, _source = _publication_fixture(registry)

    def commit_then_raise(connection: sqlite3.Connection) -> None:
        connection.commit()
        raise sqlite3.OperationalError("injected post-commit failure")

    monkeypatch.setattr(registry, "_commit", commit_then_raise)
    with pytest.raises(ArtifactRegistryError) as raised:
        registry.publish_derivation(
            snapshot=snapshot,
            logical_key="uncertain-commit",
            manifest_artifact_id=manifest_artifact_id,
            blob_sources=blobs,
        )
    assert raised.value.code is ArtifactRegistryErrorCode.TRANSACTION_FAILED
    assert LocalArtifactRegistry(root).lookup_derivation("uncertain-commit") is not None

    expected = {(("operation", "publish_derivation"), ("write", True)): 1}
    assert _counter_map(recorder, "sqlite.artifact_registry.commit_failures") == expected
    assert (
        _counter_map(
            recorder,
            "sqlite.artifact_registry.transaction_outcomes_unknown",
        )
        == expected
    )
    assert _counter_map(recorder, "sqlite.artifact_registry.rollbacks") == {}


def test_rollback_failure_is_observed_as_unknown_and_preserves_business_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RuntimeProfileRecorder()
    registry = LocalArtifactRegistry(
        tmp_path / "artifact-registry",
        runtime_observer=recorder,
    )
    snapshot, manifest_artifact_id, blobs, _source = _publication_fixture(registry)

    def fail_graph_verification(_connection: sqlite3.Connection) -> None:
        raise ArtifactRegistryError(
            ArtifactRegistryErrorCode.INTEGRITY_ERROR,
            "injected business failure",
        )

    def fail_rollback(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("injected rollback failure")

    monkeypatch.setattr(registry, "_verify_database_graph", fail_graph_verification)
    monkeypatch.setattr(registry, "_rollback", fail_rollback)
    with pytest.raises(ArtifactRegistryError) as raised:
        registry.publish_derivation(
            snapshot=snapshot,
            logical_key="rollback-failure",
            manifest_artifact_id=manifest_artifact_id,
            blob_sources=blobs,
        )
    assert raised.value.code is ArtifactRegistryErrorCode.INTEGRITY_ERROR

    with sqlite3.connect(registry.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM derivations").fetchone() == (0,)
    expected = {(("operation", "publish_derivation"), ("write", True)): 1}
    assert _counter_map(recorder, "sqlite.artifact_registry.rollback_failures") == expected
    assert (
        _counter_map(
            recorder,
            "sqlite.artifact_registry.transaction_outcomes_unknown",
        )
        == expected
    )
    assert _counter_map(recorder, "sqlite.artifact_registry.commit_failures") == {}
