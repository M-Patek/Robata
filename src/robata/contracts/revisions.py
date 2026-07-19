"""Immutable node revisions, append-only selections, and current projection contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator

from robata.contracts.common import INT64_MAX, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256 as compute_semantic_sha256
from robata.contracts.logical_nodes import (
    KeyNamespace,
    NodeLogicalKey,
    NodeType,
    OpaqueUuid,
    Rfc3339Timestamp,
)

RevisionPublicationStatus = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$",
    ),
]
SelectionSequence = Annotated[int, Field(strict=True, ge=1, le=INT64_MAX)]


def _validate_rfc3339_calendar(value: str) -> str:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("timestamp must be a valid RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an RFC3339 timezone")
    return value


_ValidatedRfc3339Timestamp = Annotated[
    Rfc3339Timestamp,
    AfterValidator(_validate_rfc3339_calendar),
]


class RevisionEligibility(StrEnum):
    """Whether a published immutable revision may be selected."""

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


def _revision_semantic_projection(
    *,
    subject_type: NodeType,
    subject_id: NodeLogicalKey,
    payload_sha256: Sha256Digest,
    lineage_sha256: Sha256Digest,
    status_at_publication: RevisionPublicationStatus,
    eligibility_at_publication: RevisionEligibility,
    revision_policy_version: SchemaVersion,
    supersedes_revision_logical_key: NodeLogicalKey | None,
) -> dict[str, str | None]:
    return {
        "semantic_projection_version": "immutable-node-revision-semantic-v1",
        "subject_type": subject_type,
        "subject_id": subject_id,
        "payload_sha256": payload_sha256,
        "lineage_sha256": lineage_sha256,
        "status_at_publication": status_at_publication,
        "eligibility_at_publication": eligibility_at_publication.value,
        "revision_policy_version": revision_policy_version,
        "supersedes_revision_logical_key": supersedes_revision_logical_key,
    }


def _selection_semantic_projection(
    *,
    subject_type: NodeType,
    subject_id: NodeLogicalKey,
    selected_revision_logical_key: NodeLogicalKey,
    previous_selection_decision_logical_key: NodeLogicalKey | None,
    selection_policy_version: SchemaVersion,
    projection_version: SchemaVersion,
) -> dict[str, str | None]:
    return {
        "semantic_projection_version": "selection-decision-semantic-v1",
        "subject_type": subject_type,
        "subject_id": subject_id,
        "selected_revision_logical_key": selected_revision_logical_key,
        "previous_selection_decision_logical_key": (previous_selection_decision_logical_key),
        "selection_policy_version": selection_policy_version,
        "projection_version": projection_version,
    }


class ImmutableNodeRevision(StrictModel):
    """One immutable revision owned by a run-independent logical node."""

    schema_version: Literal["1.0"]
    revision_id: OpaqueUuid
    subject_type: NodeType
    subject_id: NodeLogicalKey
    revision_key_namespace: KeyNamespace
    revision_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    payload_sha256: Sha256Digest
    lineage_sha256: Sha256Digest
    status_at_publication: RevisionPublicationStatus
    eligibility_at_publication: RevisionEligibility
    revision_policy_version: SchemaVersion
    supersedes_revision_id: OpaqueUuid | None
    supersedes_revision_logical_key: NodeLogicalKey | None
    published_at: _ValidatedRfc3339Timestamp

    @model_validator(mode="after")
    def validate_revision_semantics(self) -> Self:
        if (self.supersedes_revision_id is None) != (self.supersedes_revision_logical_key is None):
            raise ValueError(
                "supersedes_revision_id and supersedes_revision_logical_key must be "
                "both null or both non-null"
            )
        expected_digest = compute_semantic_sha256(
            _revision_semantic_projection(
                subject_type=self.subject_type,
                subject_id=self.subject_id,
                payload_sha256=self.payload_sha256,
                lineage_sha256=self.lineage_sha256,
                status_at_publication=self.status_at_publication,
                eligibility_at_publication=self.eligibility_at_publication,
                revision_policy_version=self.revision_policy_version,
                supersedes_revision_logical_key=self.supersedes_revision_logical_key,
            )
        )
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match the immutable revision projection")
        expected_key = f"{self.revision_key_namespace}:{expected_digest}"
        if self.revision_logical_key != expected_key:
            raise ValueError(
                "revision_logical_key must equal revision_key_namespace:semantic_sha256"
            )
        return self


class SelectionDecision(StrictModel):
    """One append-only decision in a subject's linear current-selection chain."""

    schema_version: Literal["1.0"]
    selection_decision_id: OpaqueUuid
    selection_key_namespace: KeyNamespace
    selection_decision_logical_key: NodeLogicalKey
    semantic_sha256: Sha256Digest
    subject_type: NodeType
    subject_id: NodeLogicalKey
    selected_revision_id: OpaqueUuid
    selected_revision_logical_key: NodeLogicalKey
    previous_selection_decision_id: OpaqueUuid | None
    previous_selection_decision_logical_key: NodeLogicalKey | None
    selection_sequence: SelectionSequence
    selection_policy_version: SchemaVersion
    projection_version: SchemaVersion
    selected_at: _ValidatedRfc3339Timestamp

    @model_validator(mode="after")
    def validate_selection_semantics(self) -> Self:
        if (self.previous_selection_decision_id is None) != (
            self.previous_selection_decision_logical_key is None
        ):
            raise ValueError(
                "previous_selection_decision_id and "
                "previous_selection_decision_logical_key must be both null or both non-null"
            )
        expected_digest = compute_semantic_sha256(
            _selection_semantic_projection(
                subject_type=self.subject_type,
                subject_id=self.subject_id,
                selected_revision_logical_key=self.selected_revision_logical_key,
                previous_selection_decision_logical_key=(
                    self.previous_selection_decision_logical_key
                ),
                selection_policy_version=self.selection_policy_version,
                projection_version=self.projection_version,
            )
        )
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match the selection-decision projection")
        expected_key = f"{self.selection_key_namespace}:{expected_digest}"
        if self.selection_decision_logical_key != expected_key:
            raise ValueError(
                "selection_decision_logical_key must equal selection_key_namespace:semantic_sha256"
            )
        return self


class CurrentSelection(StrictModel):
    """Replaceable projection of the verified tail selection decision."""

    schema_version: Literal["1.0"]
    subject_type: NodeType
    subject_id: NodeLogicalKey
    selected_revision_id: OpaqueUuid
    selection_decision_id: OpaqueUuid
    selection_policy_version: SchemaVersion
    projection_version: SchemaVersion
    selected_at: _ValidatedRfc3339Timestamp


def create_immutable_node_revision(
    *,
    revision_id: OpaqueUuid,
    subject_type: NodeType,
    subject_id: NodeLogicalKey,
    revision_key_namespace: KeyNamespace,
    payload_sha256: Sha256Digest,
    lineage_sha256: Sha256Digest,
    status_at_publication: RevisionPublicationStatus,
    eligibility_at_publication: RevisionEligibility,
    revision_policy_version: SchemaVersion,
    supersedes_revision_id: OpaqueUuid | None,
    supersedes_revision_logical_key: NodeLogicalKey | None,
    published_at: Rfc3339Timestamp,
) -> ImmutableNodeRevision:
    """Derive and validate an immutable revision envelope from typed inputs."""

    digest = compute_semantic_sha256(
        _revision_semantic_projection(
            subject_type=subject_type,
            subject_id=subject_id,
            payload_sha256=payload_sha256,
            lineage_sha256=lineage_sha256,
            status_at_publication=status_at_publication,
            eligibility_at_publication=eligibility_at_publication,
            revision_policy_version=revision_policy_version,
            supersedes_revision_logical_key=supersedes_revision_logical_key,
        )
    )
    return ImmutableNodeRevision(
        schema_version="1.0",
        revision_id=revision_id,
        subject_type=subject_type,
        subject_id=subject_id,
        revision_key_namespace=revision_key_namespace,
        revision_logical_key=f"{revision_key_namespace}:{digest}",
        semantic_sha256=digest,
        payload_sha256=payload_sha256,
        lineage_sha256=lineage_sha256,
        status_at_publication=status_at_publication,
        eligibility_at_publication=eligibility_at_publication,
        revision_policy_version=revision_policy_version,
        supersedes_revision_id=supersedes_revision_id,
        supersedes_revision_logical_key=supersedes_revision_logical_key,
        published_at=published_at,
    )


def create_selection_decision(
    *,
    selection_decision_id: OpaqueUuid,
    selection_key_namespace: KeyNamespace,
    subject_type: NodeType,
    subject_id: NodeLogicalKey,
    selected_revision_id: OpaqueUuid,
    selected_revision_logical_key: NodeLogicalKey,
    previous_selection_decision_id: OpaqueUuid | None,
    previous_selection_decision_logical_key: NodeLogicalKey | None,
    selection_sequence: SelectionSequence,
    selection_policy_version: SchemaVersion,
    projection_version: SchemaVersion,
    selected_at: Rfc3339Timestamp,
) -> SelectionDecision:
    """Derive and validate an append-only selection decision from resolved keys."""

    digest = compute_semantic_sha256(
        _selection_semantic_projection(
            subject_type=subject_type,
            subject_id=subject_id,
            selected_revision_logical_key=selected_revision_logical_key,
            previous_selection_decision_logical_key=previous_selection_decision_logical_key,
            selection_policy_version=selection_policy_version,
            projection_version=projection_version,
        )
    )
    return SelectionDecision(
        schema_version="1.0",
        selection_decision_id=selection_decision_id,
        selection_key_namespace=selection_key_namespace,
        selection_decision_logical_key=f"{selection_key_namespace}:{digest}",
        semantic_sha256=digest,
        subject_type=subject_type,
        subject_id=subject_id,
        selected_revision_id=selected_revision_id,
        selected_revision_logical_key=selected_revision_logical_key,
        previous_selection_decision_id=previous_selection_decision_id,
        previous_selection_decision_logical_key=previous_selection_decision_logical_key,
        selection_sequence=selection_sequence,
        selection_policy_version=selection_policy_version,
        projection_version=projection_version,
        selected_at=selected_at,
    )


__all__ = [
    "CurrentSelection",
    "ImmutableNodeRevision",
    "RevisionEligibility",
    "RevisionPublicationStatus",
    "SelectionDecision",
    "SelectionSequence",
    "create_immutable_node_revision",
    "create_selection_decision",
]
