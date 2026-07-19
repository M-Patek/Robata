"""Immutable contracts for the versioned artifact registry and its lineage DAG."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel

type NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
type ArtifactUri = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=4,
        pattern=r"^[A-Za-z][A-Za-z0-9+.-]*://\S+$",
    ),
]
type ArtifactId = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
    ),
]
type MediaType = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$",
    ),
]
type PositiveInt = Annotated[int, Field(strict=True, ge=1)]
type Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
        ),
    ),
]


class ArtifactType(StrEnum):
    """Closed artifact kinds understood by the version 2 registry."""

    CAMERA_VIDEO_EXPORT_MANIFEST = "CAMERA_VIDEO_EXPORT_MANIFEST"
    CAMERA_VIDEO_MP4 = "CAMERA_VIDEO_MP4"
    CAMERA_VIDEO_TIMESTAMP_MAP = "CAMERA_VIDEO_TIMESTAMP_MAP"
    EXPORT_CONFIG = "EXPORT_CONFIG"
    JSON_SCHEMA = "JSON_SCHEMA"
    MAPPING_PROFILE = "MAPPING_PROFILE"
    RAW_MCAP = "RAW_MCAP"


class ArtifactParentRelation(StrEnum):
    """Typed lineage edge from a derived artifact to one direct parent."""

    EXPORT_CONFIG = "EXPORT_CONFIG"
    MAPPING_PROFILE = "MAPPING_PROFILE"
    SOURCE_CONTENT = "SOURCE_CONTENT"
    TIMESTAMP_OUTPUT = "TIMESTAMP_OUTPUT"
    VIDEO_OUTPUT = "VIDEO_OUTPUT"


class SchemaArtifactReference(StrictModel):
    """Exact registry identity of the schema governing an artifact payload."""

    schema_id: ArtifactUri
    version: SchemaVersion
    artifact_id: ArtifactId
    sha256: Sha256Digest


class ArtifactProducer(StrictModel):
    """Producer implementation and its canonicalized effective configuration."""

    name: NonEmptyString
    version: SchemaVersion
    canonical_config_sha256: Sha256Digest


class ArtifactLifecycle(StrictModel):
    """Publication state and immutable retention-policy identity."""

    state: Literal["ACTIVE"]
    policy_version: SchemaVersion


class ArtifactLocator(StrictModel):
    """Versioned storage locator; neither field is an artifact identity."""

    uri: ArtifactUri
    object_version: SchemaVersion


class ArtifactParent(StrictModel):
    """One typed, direct lineage edge."""

    artifact_id: ArtifactId
    relation: ArtifactParentRelation


_MEDIA_TYPE_BY_ARTIFACT_TYPE: dict[ArtifactType, str] = {
    ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST: "application/json",
    ArtifactType.CAMERA_VIDEO_MP4: "video/mp4",
    ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP: "application/x-ndjson",
    ArtifactType.EXPORT_CONFIG: "application/json",
    ArtifactType.JSON_SCHEMA: "application/schema+json",
    ArtifactType.MAPPING_PROFILE: "application/json",
    ArtifactType.RAW_MCAP: "application/x-mcap",
}

_PARENT_COUNTS_BY_ARTIFACT_TYPE: dict[
    ArtifactType,
    dict[ArtifactParentRelation, int],
] = {
    ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST: {
        ArtifactParentRelation.EXPORT_CONFIG: 1,
        ArtifactParentRelation.MAPPING_PROFILE: 1,
        ArtifactParentRelation.SOURCE_CONTENT: 1,
        ArtifactParentRelation.TIMESTAMP_OUTPUT: 6,
        ArtifactParentRelation.VIDEO_OUTPUT: 6,
    },
    ArtifactType.CAMERA_VIDEO_MP4: {
        ArtifactParentRelation.EXPORT_CONFIG: 1,
        ArtifactParentRelation.MAPPING_PROFILE: 1,
        ArtifactParentRelation.SOURCE_CONTENT: 1,
    },
    ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP: {
        ArtifactParentRelation.EXPORT_CONFIG: 1,
        ArtifactParentRelation.MAPPING_PROFILE: 1,
        ArtifactParentRelation.SOURCE_CONTENT: 1,
    },
    ArtifactType.EXPORT_CONFIG: {},
    ArtifactType.JSON_SCHEMA: {},
    ArtifactType.MAPPING_PROFILE: {},
    ArtifactType.RAW_MCAP: {},
}

_PAYLOAD_SCHEMA_ARTIFACT_TYPES = frozenset(
    {
        ArtifactType.CAMERA_VIDEO_EXPORT_MANIFEST,
        ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
        ArtifactType.EXPORT_CONFIG,
        ArtifactType.MAPPING_PROFILE,
    }
)


class ArtifactRegistryEntry(StrictModel):
    """One immutable registry record with exact bytes and semantic identity."""

    schema_version: Literal["2.0"]
    artifact_id: ArtifactId
    artifact_type: ArtifactType
    semantic_sha256: Sha256Digest
    locator: ArtifactLocator
    sha256: Sha256Digest
    bytes: PositiveInt
    media_type: MediaType
    producer: ArtifactProducer
    lifecycle: ArtifactLifecycle
    parents: tuple[ArtifactParent, ...]
    payload_schema_ref: SchemaArtifactReference | None
    created_at: Rfc3339Timestamp

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        """Reject calendar-invalid strings that still match the RFC 3339 shape."""

        normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError("created_at must be a valid RFC3339 timestamp") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must include an RFC3339 timezone")
        return value

    @model_validator(mode="after")
    def validate_entry_semantics(self) -> Self:
        parent_ids = tuple(parent.artifact_id for parent in self.parents)
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("parents must not repeat an artifact_id")
        if self.artifact_id in parent_ids:
            raise ValueError("an artifact must not be its own parent")

        canonical_parents = tuple(
            sorted(
                self.parents,
                key=lambda parent: (parent.relation.value, parent.artifact_id),
            )
        )
        if self.parents != canonical_parents:
            raise ValueError("parents must use canonical relation and artifact_id order")

        actual_parent_counts = Counter(parent.relation for parent in self.parents)
        expected_parent_counts = Counter(_PARENT_COUNTS_BY_ARTIFACT_TYPE[self.artifact_type])
        if actual_parent_counts != expected_parent_counts:
            raise ValueError(f"{self.artifact_type.value} has invalid typed parent cardinality")

        expected_media_type = _MEDIA_TYPE_BY_ARTIFACT_TYPE[self.artifact_type]
        if self.media_type != expected_media_type:
            raise ValueError(f"{self.artifact_type.value} media_type must be {expected_media_type}")

        schema_reference_required = self.artifact_type in _PAYLOAD_SCHEMA_ARTIFACT_TYPES
        if schema_reference_required != (self.payload_schema_ref is not None):
            expectation = "requires" if schema_reference_required else "must not carry"
            raise ValueError(f"{self.artifact_type.value} {expectation} a payload_schema_ref")
        return self


_PARENT_ARTIFACT_TYPE_BY_RELATION: dict[ArtifactParentRelation, ArtifactType] = {
    ArtifactParentRelation.EXPORT_CONFIG: ArtifactType.EXPORT_CONFIG,
    ArtifactParentRelation.MAPPING_PROFILE: ArtifactType.MAPPING_PROFILE,
    ArtifactParentRelation.SOURCE_CONTENT: ArtifactType.RAW_MCAP,
    ArtifactParentRelation.TIMESTAMP_OUTPUT: ArtifactType.CAMERA_VIDEO_TIMESTAMP_MAP,
    ArtifactParentRelation.VIDEO_OUTPUT: ArtifactType.CAMERA_VIDEO_MP4,
}


class ArtifactRegistrySnapshot(StrictModel):
    """Canonical, self-contained snapshot of an artifact lineage DAG."""

    schema_version: Literal["2.0"]
    entries: tuple[ArtifactRegistryEntry, ...]

    @model_validator(mode="after")
    def validate_snapshot_semantics(self) -> Self:
        if not self.entries:
            raise ValueError("entries must not be empty")

        canonical_entries = tuple(sorted(self.entries, key=lambda entry: entry.artifact_id))
        if self.entries != canonical_entries:
            raise ValueError("entries must use canonical artifact_id order")

        artifact_ids = tuple(entry.artifact_id for entry in self.entries)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("entries must have unique artifact_id values")

        locator_versions = tuple(
            (entry.locator.uri, entry.locator.object_version) for entry in self.entries
        )
        if len(locator_versions) != len(set(locator_versions)):
            raise ValueError("entries must have unique (locator URI, object version) values")

        semantic_identities = tuple(
            (entry.artifact_type, entry.semantic_sha256) for entry in self.entries
        )
        if len(semantic_identities) != len(set(semantic_identities)):
            raise ValueError("entries must have unique (artifact type, semantic digest) values")

        entries_by_id = {entry.artifact_id: entry for entry in self.entries}
        for entry in self.entries:
            for parent in entry.parents:
                parent_entry = entries_by_id.get(parent.artifact_id)
                if parent_entry is None:
                    raise ValueError(
                        f"parent artifact {parent.artifact_id} is absent from the snapshot"
                    )
                expected_parent_type = _PARENT_ARTIFACT_TYPE_BY_RELATION[parent.relation]
                if parent_entry.artifact_type is not expected_parent_type:
                    raise ValueError(
                        f"{parent.relation.value} must reference {expected_parent_type.value}"
                    )

            reference = entry.payload_schema_ref
            if reference is None:
                continue
            schema_entry = entries_by_id.get(reference.artifact_id)
            if schema_entry is None:
                raise ValueError(
                    f"payload schema artifact {reference.artifact_id} is absent from the snapshot"
                )
            if schema_entry.artifact_type is not ArtifactType.JSON_SCHEMA:
                raise ValueError("payload_schema_ref must reference a JSON_SCHEMA artifact")
            if reference.schema_id != schema_entry.locator.uri:
                raise ValueError("payload schema_id must match the schema artifact locator URI")
            if reference.version != schema_entry.locator.object_version:
                raise ValueError("payload schema version must match the schema object version")
            if reference.sha256 != schema_entry.sha256:
                raise ValueError("payload schema digest must match the schema artifact bytes")

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
    "ArtifactId",
    "ArtifactLifecycle",
    "ArtifactLocator",
    "ArtifactParent",
    "ArtifactParentRelation",
    "ArtifactProducer",
    "ArtifactRegistryEntry",
    "ArtifactRegistrySnapshot",
    "ArtifactType",
    "ArtifactUri",
    "MediaType",
    "SchemaArtifactReference",
]
