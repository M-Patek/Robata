"""Additive Artifact Registry V3 contracts for pre-EOS stream artifacts.

V2 remains a closed, published contract. This module repeats its seven
artifact kinds and adds the stream lineage vocabulary without changing the V2
enums or models in place.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import field_validator, model_validator

from robata.contracts.artifacts import (
    ArtifactId,
    ArtifactLifecycle,
    ArtifactLocator,
    ArtifactProducer,
    MediaType,
    PositiveInt,
    Rfc3339Timestamp,
    SchemaArtifactReference,
)
from robata.contracts.common import Sha256Digest, StrictModel


class ArtifactTypeV3(StrEnum):
    """Closed V3 vocabulary, including the unchanged V2 artifact kinds."""

    CAMERA_VIDEO_EXPORT_MANIFEST = "CAMERA_VIDEO_EXPORT_MANIFEST"
    CAMERA_VIDEO_MP4 = "CAMERA_VIDEO_MP4"
    CAMERA_VIDEO_TIMESTAMP_MAP = "CAMERA_VIDEO_TIMESTAMP_MAP"
    EXPORT_CONFIG = "EXPORT_CONFIG"
    JSON_SCHEMA = "JSON_SCHEMA"
    MAPPING_PROFILE = "MAPPING_PROFILE"
    RAW_MCAP = "RAW_MCAP"
    PRE_EOS_CAPTURE = "PRE_EOS_CAPTURE"
    STREAM_ENCODED_SPOOL = "STREAM_ENCODED_SPOOL"
    STREAM_SEGMENT_MANIFEST = "STREAM_SEGMENT_MANIFEST"
    INCREMENTAL_WINDOW = "INCREMENTAL_WINDOW"
    INFERENCE_INPUT_PLAN = "INFERENCE_INPUT_PLAN"
    STREAM_INFERENCE_INTENT = "STREAM_INFERENCE_INTENT"
    STREAM_ACCEPTED_CALL_EVIDENCE = "STREAM_ACCEPTED_CALL_EVIDENCE"
    STREAM_INFERENCE_TERMINAL = "STREAM_INFERENCE_TERMINAL"
    STREAM_WINDOW_RESULT = "STREAM_WINDOW_RESULT"
    EXPECTED_WINDOW_PLAN = "EXPECTED_WINDOW_PLAN"
    EXPECTED_WINDOW_DECLARATION = "EXPECTED_WINDOW_DECLARATION"
    EXPECTED_WINDOW_PLAN_SEAL = "EXPECTED_WINDOW_PLAN_SEAL"
    WINDOW_TERMINAL_CLOSURE = "WINDOW_TERMINAL_CLOSURE"
    RECORDING_FINALIZATION_MAP = "RECORDING_FINALIZATION_MAP"


class ArtifactParentRelationV3(StrEnum):
    """Typed direct lineage edges understood by the V3 registry."""

    EXPORT_CONFIG = "EXPORT_CONFIG"
    MAPPING_PROFILE = "MAPPING_PROFILE"
    SOURCE_CONTENT = "SOURCE_CONTENT"
    TIMESTAMP_OUTPUT = "TIMESTAMP_OUTPUT"
    VIDEO_OUTPUT = "VIDEO_OUTPUT"
    CAPTURE_SCOPE = "CAPTURE_SCOPE"
    ENCODED_SPOOL = "ENCODED_SPOOL"
    SEGMENT_INPUT = "SEGMENT_INPUT"
    WINDOW_INPUT = "WINDOW_INPUT"
    INPUT_PLAN = "INPUT_PLAN"
    INFERENCE_INTENT = "INFERENCE_INTENT"
    ACCEPTED_CALL = "ACCEPTED_CALL"
    INFERENCE_TERMINAL = "INFERENCE_TERMINAL"
    EXPECTED_PLAN = "EXPECTED_PLAN"
    EXPECTED_DECLARATION = "EXPECTED_DECLARATION"
    PLAN_SEAL = "PLAN_SEAL"
    WINDOW_RESULT = "WINDOW_RESULT"
    TERMINAL_CLOSURE = "TERMINAL_CLOSURE"
    EXPORT_MANIFEST = "EXPORT_MANIFEST"


class ArtifactParentV3(StrictModel):
    """One typed direct V3 lineage edge."""

    artifact_id: ArtifactId
    relation: ArtifactParentRelationV3


_MEDIA_TYPE_BY_ARTIFACT_TYPE: dict[ArtifactTypeV3, str] = {
    ArtifactTypeV3.CAMERA_VIDEO_EXPORT_MANIFEST: "application/json",
    ArtifactTypeV3.CAMERA_VIDEO_MP4: "video/mp4",
    ArtifactTypeV3.CAMERA_VIDEO_TIMESTAMP_MAP: "application/x-ndjson",
    ArtifactTypeV3.EXPORT_CONFIG: "application/json",
    ArtifactTypeV3.JSON_SCHEMA: "application/schema+json",
    ArtifactTypeV3.MAPPING_PROFILE: "application/json",
    ArtifactTypeV3.RAW_MCAP: "application/x-mcap",
    ArtifactTypeV3.PRE_EOS_CAPTURE: "application/json",
    ArtifactTypeV3.STREAM_ENCODED_SPOOL: "application/octet-stream",
    ArtifactTypeV3.STREAM_SEGMENT_MANIFEST: "application/json",
    ArtifactTypeV3.INCREMENTAL_WINDOW: "application/json",
    ArtifactTypeV3.INFERENCE_INPUT_PLAN: "application/json",
    ArtifactTypeV3.STREAM_INFERENCE_INTENT: "application/json",
    ArtifactTypeV3.STREAM_ACCEPTED_CALL_EVIDENCE: "application/json",
    ArtifactTypeV3.STREAM_INFERENCE_TERMINAL: "application/json",
    ArtifactTypeV3.STREAM_WINDOW_RESULT: "application/json",
    ArtifactTypeV3.EXPECTED_WINDOW_PLAN: "application/json",
    ArtifactTypeV3.EXPECTED_WINDOW_DECLARATION: "application/json",
    ArtifactTypeV3.EXPECTED_WINDOW_PLAN_SEAL: "application/json",
    ArtifactTypeV3.WINDOW_TERMINAL_CLOSURE: "application/json",
    ArtifactTypeV3.RECORDING_FINALIZATION_MAP: "application/json",
}

_LEGACY_PARENT_COUNTS: dict[
    ArtifactTypeV3,
    dict[ArtifactParentRelationV3, int],
] = {
    ArtifactTypeV3.CAMERA_VIDEO_EXPORT_MANIFEST: {
        ArtifactParentRelationV3.EXPORT_CONFIG: 1,
        ArtifactParentRelationV3.MAPPING_PROFILE: 1,
        ArtifactParentRelationV3.SOURCE_CONTENT: 1,
        ArtifactParentRelationV3.TIMESTAMP_OUTPUT: 6,
        ArtifactParentRelationV3.VIDEO_OUTPUT: 6,
    },
    ArtifactTypeV3.CAMERA_VIDEO_MP4: {
        ArtifactParentRelationV3.EXPORT_CONFIG: 1,
        ArtifactParentRelationV3.MAPPING_PROFILE: 1,
        ArtifactParentRelationV3.SOURCE_CONTENT: 1,
    },
    ArtifactTypeV3.CAMERA_VIDEO_TIMESTAMP_MAP: {
        ArtifactParentRelationV3.EXPORT_CONFIG: 1,
        ArtifactParentRelationV3.MAPPING_PROFILE: 1,
        ArtifactParentRelationV3.SOURCE_CONTENT: 1,
    },
    ArtifactTypeV3.EXPORT_CONFIG: {},
    ArtifactTypeV3.JSON_SCHEMA: {},
    ArtifactTypeV3.MAPPING_PROFILE: {},
    ArtifactTypeV3.RAW_MCAP: {},
}

_REQUIRED_STREAM_RELATIONS: dict[
    ArtifactTypeV3,
    frozenset[ArtifactParentRelationV3],
] = {
    ArtifactTypeV3.PRE_EOS_CAPTURE: frozenset(),
    ArtifactTypeV3.STREAM_ENCODED_SPOOL: frozenset({ArtifactParentRelationV3.CAPTURE_SCOPE}),
    ArtifactTypeV3.STREAM_SEGMENT_MANIFEST: frozenset(
        {
            ArtifactParentRelationV3.CAPTURE_SCOPE,
            ArtifactParentRelationV3.ENCODED_SPOOL,
        }
    ),
    ArtifactTypeV3.INCREMENTAL_WINDOW: frozenset(
        {
            ArtifactParentRelationV3.CAPTURE_SCOPE,
            ArtifactParentRelationV3.SEGMENT_INPUT,
        }
    ),
    ArtifactTypeV3.INFERENCE_INPUT_PLAN: frozenset({ArtifactParentRelationV3.WINDOW_INPUT}),
    ArtifactTypeV3.STREAM_INFERENCE_INTENT: frozenset(
        {
            ArtifactParentRelationV3.WINDOW_INPUT,
            ArtifactParentRelationV3.INPUT_PLAN,
        }
    ),
    ArtifactTypeV3.STREAM_ACCEPTED_CALL_EVIDENCE: frozenset(
        {ArtifactParentRelationV3.INFERENCE_INTENT}
    ),
    ArtifactTypeV3.STREAM_INFERENCE_TERMINAL: frozenset(
        {
            ArtifactParentRelationV3.INFERENCE_INTENT,
            ArtifactParentRelationV3.ACCEPTED_CALL,
        }
    ),
    ArtifactTypeV3.STREAM_WINDOW_RESULT: frozenset({ArtifactParentRelationV3.WINDOW_INPUT}),
    ArtifactTypeV3.EXPECTED_WINDOW_PLAN: frozenset({ArtifactParentRelationV3.CAPTURE_SCOPE}),
    ArtifactTypeV3.EXPECTED_WINDOW_DECLARATION: frozenset(
        {
            ArtifactParentRelationV3.EXPECTED_PLAN,
            ArtifactParentRelationV3.WINDOW_INPUT,
        }
    ),
    ArtifactTypeV3.EXPECTED_WINDOW_PLAN_SEAL: frozenset({ArtifactParentRelationV3.EXPECTED_PLAN}),
    ArtifactTypeV3.WINDOW_TERMINAL_CLOSURE: frozenset({ArtifactParentRelationV3.PLAN_SEAL}),
    ArtifactTypeV3.RECORDING_FINALIZATION_MAP: frozenset(
        {
            ArtifactParentRelationV3.TERMINAL_CLOSURE,
            ArtifactParentRelationV3.EXPORT_MANIFEST,
        }
    ),
}

_PAYLOAD_SCHEMA_ARTIFACT_TYPES = frozenset(
    {
        artifact_type
        for artifact_type, media_type in _MEDIA_TYPE_BY_ARTIFACT_TYPE.items()
        if media_type in {"application/json", "application/x-ndjson"}
    }
    - {ArtifactTypeV3.JSON_SCHEMA}
)


class ArtifactRegistryEntryV3(StrictModel):
    """One immutable V3 artifact, preserving semantic and exact identities."""

    schema_version: Literal["3.0"]
    artifact_id: ArtifactId
    artifact_type: ArtifactTypeV3
    semantic_sha256: Sha256Digest
    locator: ArtifactLocator
    sha256: Sha256Digest
    bytes: PositiveInt
    media_type: MediaType
    producer: ArtifactProducer
    lifecycle: ArtifactLifecycle
    parents: tuple[ArtifactParentV3, ...]
    payload_schema_ref: SchemaArtifactReference | None
    created_at: Rfc3339Timestamp

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError("created_at must be a valid RFC3339 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must include an RFC3339 timezone")
        return value

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        parent_ids = tuple(parent.artifact_id for parent in self.parents)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("parents must not repeat an artifact_id")
        if self.artifact_id in parent_ids:
            raise ValueError("an artifact must not be its own parent")
        if self.parents != tuple(
            sorted(self.parents, key=lambda parent: (parent.relation.value, parent.artifact_id))
        ):
            raise ValueError("parents must use canonical relation and artifact_id order")

        expected_media_type = _MEDIA_TYPE_BY_ARTIFACT_TYPE[self.artifact_type]
        if self.media_type != expected_media_type:
            raise ValueError(f"{self.artifact_type.value} media_type must be {expected_media_type}")

        legacy_counts = _LEGACY_PARENT_COUNTS.get(self.artifact_type)
        if legacy_counts is not None:
            if Counter(parent.relation for parent in self.parents) != Counter(legacy_counts):
                raise ValueError(f"{self.artifact_type.value} has invalid typed parent cardinality")
        else:
            required = _REQUIRED_STREAM_RELATIONS[self.artifact_type]
            actual = frozenset(parent.relation for parent in self.parents)
            if not required.issubset(actual):
                raise ValueError(f"{self.artifact_type.value} is missing required lineage")

        schema_required = self.artifact_type in _PAYLOAD_SCHEMA_ARTIFACT_TYPES
        if schema_required != (self.payload_schema_ref is not None):
            expectation = "requires" if schema_required else "must not carry"
            raise ValueError(f"{self.artifact_type.value} {expectation} a payload_schema_ref")
        return self


_PARENT_TYPES_BY_RELATION: dict[
    ArtifactParentRelationV3,
    frozenset[ArtifactTypeV3],
] = {
    ArtifactParentRelationV3.EXPORT_CONFIG: frozenset({ArtifactTypeV3.EXPORT_CONFIG}),
    ArtifactParentRelationV3.MAPPING_PROFILE: frozenset({ArtifactTypeV3.MAPPING_PROFILE}),
    ArtifactParentRelationV3.SOURCE_CONTENT: frozenset({ArtifactTypeV3.RAW_MCAP}),
    ArtifactParentRelationV3.TIMESTAMP_OUTPUT: frozenset(
        {ArtifactTypeV3.CAMERA_VIDEO_TIMESTAMP_MAP}
    ),
    ArtifactParentRelationV3.VIDEO_OUTPUT: frozenset({ArtifactTypeV3.CAMERA_VIDEO_MP4}),
    ArtifactParentRelationV3.CAPTURE_SCOPE: frozenset({ArtifactTypeV3.PRE_EOS_CAPTURE}),
    ArtifactParentRelationV3.ENCODED_SPOOL: frozenset({ArtifactTypeV3.STREAM_ENCODED_SPOOL}),
    ArtifactParentRelationV3.SEGMENT_INPUT: frozenset({ArtifactTypeV3.STREAM_SEGMENT_MANIFEST}),
    ArtifactParentRelationV3.WINDOW_INPUT: frozenset({ArtifactTypeV3.INCREMENTAL_WINDOW}),
    ArtifactParentRelationV3.INPUT_PLAN: frozenset({ArtifactTypeV3.INFERENCE_INPUT_PLAN}),
    ArtifactParentRelationV3.INFERENCE_INTENT: frozenset({ArtifactTypeV3.STREAM_INFERENCE_INTENT}),
    ArtifactParentRelationV3.ACCEPTED_CALL: frozenset(
        {ArtifactTypeV3.STREAM_ACCEPTED_CALL_EVIDENCE}
    ),
    ArtifactParentRelationV3.INFERENCE_TERMINAL: frozenset(
        {ArtifactTypeV3.STREAM_INFERENCE_TERMINAL}
    ),
    ArtifactParentRelationV3.EXPECTED_PLAN: frozenset({ArtifactTypeV3.EXPECTED_WINDOW_PLAN}),
    ArtifactParentRelationV3.EXPECTED_DECLARATION: frozenset(
        {ArtifactTypeV3.EXPECTED_WINDOW_DECLARATION}
    ),
    ArtifactParentRelationV3.PLAN_SEAL: frozenset({ArtifactTypeV3.EXPECTED_WINDOW_PLAN_SEAL}),
    ArtifactParentRelationV3.WINDOW_RESULT: frozenset({ArtifactTypeV3.STREAM_WINDOW_RESULT}),
    ArtifactParentRelationV3.TERMINAL_CLOSURE: frozenset({ArtifactTypeV3.WINDOW_TERMINAL_CLOSURE}),
    ArtifactParentRelationV3.EXPORT_MANIFEST: frozenset(
        {ArtifactTypeV3.CAMERA_VIDEO_EXPORT_MANIFEST}
    ),
}


class ArtifactRegistrySnapshotV3(StrictModel):
    """Canonical, self-contained V3 artifact lineage DAG."""

    schema_version: Literal["3.0"]
    entries: tuple[ArtifactRegistryEntryV3, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if not self.entries:
            raise ValueError("entries must not be empty")
        if self.entries != tuple(sorted(self.entries, key=lambda entry: entry.artifact_id)):
            raise ValueError("entries must use canonical artifact_id order")

        artifact_ids = tuple(entry.artifact_id for entry in self.entries)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("entries must have unique artifact_id values")
        locator_versions = tuple(
            (entry.locator.uri, entry.locator.object_version) for entry in self.entries
        )
        if len(locator_versions) != len(set(locator_versions)):
            raise ValueError("entries must have unique (locator URI, object version) values")
        semantic_ids = tuple((entry.artifact_type, entry.semantic_sha256) for entry in self.entries)
        if len(semantic_ids) != len(set(semantic_ids)):
            raise ValueError("entries must have unique (artifact type, semantic digest) values")

        entries_by_id = {entry.artifact_id: entry for entry in self.entries}
        for entry in self.entries:
            for parent in entry.parents:
                parent_entry = entries_by_id.get(parent.artifact_id)
                if parent_entry is None:
                    raise ValueError(f"parent artifact {parent.artifact_id} is absent")
                if parent_entry.artifact_type not in _PARENT_TYPES_BY_RELATION[parent.relation]:
                    raise ValueError(
                        f"{parent.relation.value} references an incompatible artifact type"
                    )
            reference = entry.payload_schema_ref
            if reference is None:
                continue
            schema_entry = entries_by_id.get(reference.artifact_id)
            if schema_entry is None or schema_entry.artifact_type is not ArtifactTypeV3.JSON_SCHEMA:
                raise ValueError("payload_schema_ref must resolve to a JSON_SCHEMA artifact")
            if (
                reference.schema_id != schema_entry.locator.uri
                or reference.version != schema_entry.locator.object_version
                or reference.sha256 != schema_entry.sha256
            ):
                raise ValueError("payload schema reference does not match exact schema artifact")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(artifact_id: str) -> None:
            if artifact_id in visiting:
                raise ValueError("artifact parent graph must be acyclic")
            if artifact_id in visited:
                return
            visiting.add(artifact_id)
            for parent in entries_by_id[artifact_id].parents:
                visit(parent.artifact_id)
            visiting.remove(artifact_id)
            visited.add(artifact_id)

        for artifact_id in artifact_ids:
            visit(artifact_id)
        return self


__all__ = [
    "ArtifactParentRelationV3",
    "ArtifactParentV3",
    "ArtifactRegistryEntryV3",
    "ArtifactRegistrySnapshotV3",
    "ArtifactTypeV3",
]
