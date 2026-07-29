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


class ArtifactBlobReconciliationState(StrEnum):
    """States observed while reconciling the blob store with registry metadata."""

    REGISTERED = "REGISTERED"
    MISSING = "MISSING"
    CORRUPT = "CORRUPT"
    ORPHAN = "ORPHAN"
    PARTIAL = "PARTIAL"
    DUPLICATE = "DUPLICATE"


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


@dataclass(frozen=True, slots=True)
class ArtifactRegistryReconciliation:
    """A bounded observation of registry metadata and content-addressed blobs."""

    registered_artifact_count: int
    visible_artifact_count: int
    missing_artifact_ids: tuple[str, ...] = ()
    corrupt_artifact_ids: tuple[str, ...] = ()
    orphan_blob_paths: tuple[Path, ...] = ()
    partial_blob_paths: tuple[Path, ...] = ()
    duplicate_blob_paths: tuple[Path, ...] = ()
    removed_blob_paths: tuple[Path, ...] = ()
    failed_removal_paths: tuple[Path, ...] = ()

    @property
    def missing_count(self) -> int:
        return len(self.missing_artifact_ids)

    @property
    def corrupt_count(self) -> int:
        return len(self.corrupt_artifact_ids)

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_blob_paths)

    @property
    def partial_count(self) -> int:
        return len(self.partial_blob_paths)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_blob_paths)

    @property
    def issue_count(self) -> int:
        return (
            self.missing_count
            + self.corrupt_count
            + self.orphan_count
            + self.partial_count
            + self.duplicate_count
            + len(self.failed_removal_paths)
        )

    @property
    def reconciled(self) -> bool:
        return self.issue_count == 0

    @property
    def ok(self) -> bool:
        return self.reconciled


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

    def reconcile(
        self,
        *,
        remove_orphans: bool = False,
        remove_partials: bool = False,
        remove_duplicates: bool = False,
        strict: bool = False,
    ) -> ArtifactRegistryReconciliation:
        """Reconcile metadata, exact blobs, and crash leftovers in backing storage."""

    def verify_derivation(self, logical_key: str) -> ArtifactRegistrySnapshot:
        """Verify registry metadata, full lineage, and every exact blob."""


__all__ = [
    "ArtifactBlobReconciliationState",
    "ArtifactBlobSource",
    "ArtifactRegistry",
    "ArtifactRegistryError",
    "ArtifactRegistryErrorCode",
    "ArtifactRegistryReconciliation",
    "PublishedArtifactDerivation",
]
