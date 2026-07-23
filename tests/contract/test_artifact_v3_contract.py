from __future__ import annotations

import json
from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from robata.contracts.artifacts import (
    ArtifactLifecycle,
    ArtifactLocator,
    ArtifactProducer,
    ArtifactType,
    SchemaArtifactReference,
)
from robata.contracts.artifacts_v3 import (
    ArtifactParentRelationV3,
    ArtifactParentV3,
    ArtifactRegistryEntryV3,
    ArtifactRegistrySnapshotV3,
    ArtifactTypeV3,
)


def _uuid(number: int) -> str:
    return f"00000000-0000-4000-8000-{number:012x}"


def _digest(number: int) -> str:
    return f"{number:064x}"


def _parent(number: int, relation: ArtifactParentRelationV3) -> ArtifactParentV3:
    return ArtifactParentV3(artifact_id=_uuid(number), relation=relation)


def _entry(
    number: int,
    artifact_type: ArtifactTypeV3,
    *,
    parents: Iterable[ArtifactParentV3] = (),
    schema_ref: SchemaArtifactReference | None = None,
) -> ArtifactRegistryEntryV3:
    media_types = {
        ArtifactTypeV3.JSON_SCHEMA: "application/schema+json",
        ArtifactTypeV3.STREAM_ENCODED_SPOOL: "application/octet-stream",
    }
    return ArtifactRegistryEntryV3(
        schema_version="3.0",
        artifact_id=_uuid(number),
        artifact_type=artifact_type,
        semantic_sha256=_digest(1000 + number),
        locator=ArtifactLocator(
            uri=f"object://stream-artifacts/{number}",
            object_version="1.0.0",
        ),
        sha256=_digest(2000 + number),
        bytes=1000 + number,
        media_type=media_types.get(artifact_type, "application/json"),
        producer=ArtifactProducer(
            name="stream-contract-test",
            version="1.0.0",
            canonical_config_sha256=_digest(3000),
        ),
        lifecycle=ArtifactLifecycle(state="ACTIVE", policy_version="retention-v1"),
        parents=tuple(sorted(parents, key=lambda item: (item.relation.value, item.artifact_id))),
        payload_schema_ref=schema_ref,
        created_at="2026-07-22T12:00:00Z",
    )


def _snapshot() -> ArtifactRegistrySnapshotV3:
    schema = _entry(1, ArtifactTypeV3.JSON_SCHEMA)
    schema = schema.model_copy(
        update={
            "locator": ArtifactLocator(
                uri="https://schemas.robata.dev/stream-artifact-payload",
                object_version="1.0.0",
            )
        }
    )
    schema_ref = SchemaArtifactReference(
        schema_id=schema.locator.uri,
        version=schema.locator.object_version,
        artifact_id=schema.artifact_id,
        sha256=schema.sha256,
    )
    capture = _entry(2, ArtifactTypeV3.PRE_EOS_CAPTURE, schema_ref=schema_ref)
    spool = _entry(
        3,
        ArtifactTypeV3.STREAM_ENCODED_SPOOL,
        parents=(_parent(2, ArtifactParentRelationV3.CAPTURE_SCOPE),),
    )
    segment = _entry(
        4,
        ArtifactTypeV3.STREAM_SEGMENT_MANIFEST,
        parents=(
            _parent(2, ArtifactParentRelationV3.CAPTURE_SCOPE),
            _parent(3, ArtifactParentRelationV3.ENCODED_SPOOL),
        ),
        schema_ref=schema_ref,
    )
    window = _entry(
        5,
        ArtifactTypeV3.INCREMENTAL_WINDOW,
        parents=(
            _parent(2, ArtifactParentRelationV3.CAPTURE_SCOPE),
            _parent(4, ArtifactParentRelationV3.SEGMENT_INPUT),
        ),
        schema_ref=schema_ref,
    )
    plan = _entry(
        6,
        ArtifactTypeV3.INFERENCE_INPUT_PLAN,
        parents=(_parent(5, ArtifactParentRelationV3.WINDOW_INPUT),),
        schema_ref=schema_ref,
    )
    intent = _entry(
        7,
        ArtifactTypeV3.STREAM_INFERENCE_INTENT,
        parents=(
            _parent(5, ArtifactParentRelationV3.WINDOW_INPUT),
            _parent(6, ArtifactParentRelationV3.INPUT_PLAN),
        ),
        schema_ref=schema_ref,
    )
    accepted = _entry(
        8,
        ArtifactTypeV3.STREAM_ACCEPTED_CALL_EVIDENCE,
        parents=(_parent(7, ArtifactParentRelationV3.INFERENCE_INTENT),),
        schema_ref=schema_ref,
    )
    terminal = _entry(
        9,
        ArtifactTypeV3.STREAM_INFERENCE_TERMINAL,
        parents=(
            _parent(7, ArtifactParentRelationV3.INFERENCE_INTENT),
            _parent(8, ArtifactParentRelationV3.ACCEPTED_CALL),
        ),
        schema_ref=schema_ref,
    )
    result = _entry(
        10,
        ArtifactTypeV3.STREAM_WINDOW_RESULT,
        parents=(
            _parent(5, ArtifactParentRelationV3.WINDOW_INPUT),
            _parent(9, ArtifactParentRelationV3.INFERENCE_TERMINAL),
        ),
        schema_ref=schema_ref,
    )
    return ArtifactRegistrySnapshotV3(
        schema_version="3.0",
        entries=(schema, capture, spool, segment, window, plan, intent, accepted, terminal, result),
    )


def test_v3_registers_stream_inference_lineage_without_opening_v2() -> None:
    snapshot = _snapshot()

    assert snapshot.entries[-1].artifact_type is ArtifactTypeV3.STREAM_WINDOW_RESULT
    with pytest.raises(ValueError):
        ArtifactType("STREAM_WINDOW_RESULT")


def test_v3_snapshot_rejects_relation_to_wrong_parent_type() -> None:
    snapshot = _snapshot()
    payload = snapshot.model_dump(mode="json")
    terminal = payload["entries"][8]
    terminal["parents"][0]["artifact_id"] = _uuid(6)

    with pytest.raises(ValidationError, match="incompatible artifact type"):
        ArtifactRegistrySnapshotV3.model_validate_json(json.dumps(payload))


def test_v3_entry_rejects_missing_accepted_call_lineage() -> None:
    snapshot = _snapshot()
    terminal = snapshot.entries[8]
    payload = terminal.model_dump(mode="json")
    payload["parents"] = [
        parent
        for parent in payload["parents"]
        if parent["relation"] != ArtifactParentRelationV3.ACCEPTED_CALL.value
    ]

    with pytest.raises(ValidationError, match="missing required lineage"):
        ArtifactRegistryEntryV3.model_validate_json(json.dumps(payload))
