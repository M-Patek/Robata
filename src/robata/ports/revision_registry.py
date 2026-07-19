"""Boundary for immutable node revisions and deterministic current selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from robata.contracts.common import SchemaVersion
from robata.contracts.logical_nodes import (
    KeyNamespace,
    LogicalNode,
    NodeLogicalKey,
    NodeType,
    OpaqueUuid,
    Rfc3339Timestamp,
)
from robata.contracts.revisions import (
    CurrentSelection,
    ImmutableNodeRevision,
    SelectionDecision,
)


class RevisionSelectionRegistryErrorCode(StrEnum):
    """Stable machine-readable revision and selection registry failures."""

    INVALID_REQUEST = "INVALID_REQUEST"
    SUBJECT_NOT_FOUND = "SUBJECT_NOT_FOUND"
    REVISION_NOT_FOUND = "REVISION_NOT_FOUND"
    REVISION_CONFLICT = "REVISION_CONFLICT"
    REVISION_INELIGIBLE = "REVISION_INELIGIBLE"
    SELECTION_CONFLICT = "SELECTION_CONFLICT"
    STALE_SELECTION = "STALE_SELECTION"
    TRANSACTION_FAILED = "TRANSACTION_FAILED"
    STORAGE_IO_ERROR = "STORAGE_IO_ERROR"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


class RevisionSelectionRegistryError(RuntimeError):
    """A revision registry failure carrying a stable error code."""

    def __init__(self, code: RevisionSelectionRegistryErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PublishedRevision:
    """Verified immutable revision publication and insertion attribution.

    The inserted flag is None when commit recovery proves that the revision exists but
    cannot attribute its insertion to a particular concurrent transaction.
    """

    revision: ImmutableNodeRevision
    inserted: bool | None


@dataclass(frozen=True, slots=True)
class PublishedSelection:
    """One committed decision and the verified projection after the operation.

    The current field is the actual current projection when the call returns. An exact
    retry returns its original decision without rolling a projection back from a later
    decision. Attribution flags are None when commit recovery cannot prove which
    concurrent transaction performed the corresponding write.
    """

    decision: SelectionDecision
    current: CurrentSelection
    decision_inserted: bool | None
    projection_advanced: bool | None


@dataclass(frozen=True, slots=True)
class VerifiedRevisionSubject:
    """A verified subject with canonical revision, decision, and current state."""

    node: LogicalNode
    revisions: tuple[ImmutableNodeRevision, ...]
    decisions: tuple[SelectionDecision, ...]
    current: CurrentSelection | None


class RevisionSelectionRegistry(Protocol):
    """Durable publication, selection-CAS, and projection-rebuild boundary."""

    def publish_revision(
        self,
        revision: ImmutableNodeRevision,
    ) -> PublishedRevision:
        """Publish or resolve one immutable revision under its existing subject."""

    def select_revision(
        self,
        *,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        selected_revision_id: OpaqueUuid,
        selection_decision_id: OpaqueUuid,
        selection_key_namespace: KeyNamespace,
        expected_previous_selection_decision_id: OpaqueUuid | None,
        selection_policy_version: SchemaVersion,
        selected_at: Rfc3339Timestamp,
    ) -> PublishedSelection:
        """Append a decision and compare-and-swap its current projection atomically.

        The adapter derives the decision sequence, predecessor logical key, selected
        revision logical key, semantic digest, and decision logical key. Retrying an
        exact committed decision never rewinds a projection advanced by a later one.
        """

    def lookup_revision(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        revision_id: OpaqueUuid,
    ) -> ImmutableNodeRevision | None:
        """Return one verified revision by subject and opaque revision ID."""

    def lookup_selection_decision(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
        selection_decision_id: OpaqueUuid,
    ) -> SelectionDecision | None:
        """Return one verified append-only decision by subject and decision ID."""

    def lookup_current_selection(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> CurrentSelection | None:
        """Return the verified current projection, or None before first selection."""

    def list_revisions(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> tuple[ImmutableNodeRevision, ...]:
        """List a subject's revisions in canonical logical-key and ID order."""

    def list_selection_decisions(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> tuple[SelectionDecision, ...]:
        """List a subject's decisions in canonical sequence and ID order."""

    def verify_subject(
        self,
        subject_type: NodeType,
        subject_id: NodeLogicalKey,
    ) -> VerifiedRevisionSubject:
        """Verify ownership, immutable rows, decision chain, and current projection."""

    def rebuild_current_projection(self) -> tuple[CurrentSelection, ...]:
        """Atomically rebuild and return all current rows in canonical subject order."""


__all__ = [
    "PublishedRevision",
    "PublishedSelection",
    "RevisionSelectionRegistry",
    "RevisionSelectionRegistryError",
    "RevisionSelectionRegistryErrorCode",
    "VerifiedRevisionSubject",
]
