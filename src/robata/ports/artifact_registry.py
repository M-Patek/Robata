"""Artifact-registry boundary for durable, content-addressed derivations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from robata.contracts.artifacts import (
    ArtifactRegistryEntry,
    ArtifactRegistrySnapshot,
    ArtifactType,
)

type ArtifactBlobSource = Path | bytes


class ArtifactRegistryErrorCode(StrEnum):
    """Stable machine-readable failures at the artifact-registry boundary."""

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
    ARTIFACT_ID_MISMATCH = "ARTIFACT_ID_MISMATCH"
    BLOB_SOURCE_MISSING = "BLOB_SOURCE_MISSING"
    BLOB_SOURCE_UNEXPECTED = "BLOB_SOURCE_UNEXPECTED"
    BLOB_SOURCE_INVALID = "BLOB_SOURCE_INVALID"
    BLOB_DIGEST_MISMATCH = "BLOB_DIGEST_MISMATCH"
    BLOB_SIZE_MISMATCH = "BLOB_SIZE_MISMATCH"
    BLOB_CONFLICT = "BLOB_CONFLICT"
    ARTIFACT_CONFLICT = "ARTIFACT_CONFLICT"
    LOCATION_CONFLICT = "LOCATION_CONFLICT"
    MISSING_PARENT = "MISSING_PARENT"
    GRAPH_CYCLE = "GRAPH_CYCLE"
    INCOMPLETE_DAG = "INCOMPLETE_DAG"
    DERIVATION_CONFLICT = "DERIVATION_CONFLICT"
    DERIVATION_NOT_FOUND = "DERIVATION_NOT_FOUND"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    STORAGE_IO_ERROR = "STORAGE_IO_ERROR"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


class ArtifactRegistryError(RuntimeError):
    """An artifact-registry failure carrying a stable error code."""

    def __init__(self, code: ArtifactRegistryErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PublishedArtifactDerivation:
    """One verified logical derivation and whether publication reused it."""

    logical_key: str
    manifest_artifact_id: str
    snapshot: ArtifactRegistrySnapshot
    reused: bool


class ArtifactRegistry(Protocol):
    """Durable publication and verified-reuse boundary."""

    def allocate_artifact_id(
        self,
        artifact_type: ArtifactType,
        semantic_sha256: str,
    ) -> str:
        """Allocate the artifact ID before constructing an immutable entry."""

    def publish_derivation(
        self,
        *,
        snapshot: ArtifactRegistrySnapshot,
        logical_key: str,
        manifest_artifact_id: str,
        blob_sources: Mapping[str, ArtifactBlobSource],
    ) -> PublishedArtifactDerivation:
        """Put exact blobs first, then atomically publish the complete derivation."""

    def lookup_artifact(
        self,
        artifact_type: ArtifactType,
        semantic_sha256: str,
    ) -> ArtifactRegistryEntry | None:
        """Resolve a verified immutable entry by its semantic identity."""

    def lookup_derivation(self, logical_key: str) -> PublishedArtifactDerivation | None:
        """Return a fully verified reusable derivation, or ``None`` when absent."""

    def load_snapshot(self, logical_key: str) -> ArtifactRegistrySnapshot:
        """Load and validate the committed canonical snapshot for a logical key."""

    def resolve_blob(self, artifact_id: str) -> Path:
        """Resolve an artifact to a content-addressed blob after rehashing it."""

    def verify_derivation(self, logical_key: str) -> ArtifactRegistrySnapshot:
        """Verify registry metadata, full lineage, and every exact blob."""


__all__ = [
    "ArtifactBlobSource",
    "ArtifactRegistry",
    "ArtifactRegistryError",
    "ArtifactRegistryErrorCode",
    "PublishedArtifactDerivation",
]
