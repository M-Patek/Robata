"""Registered compact evidence for one authoritative primary completion.

The detailed run result remains an immutable external JSON artifact.  This
module does not register that detailed-result schema or verify its referenced
bytes; both remain mandatory before a governed completion can be committed.
The compact contract only binds the transaction facts without embedding those
potentially large collections in the metadata row.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from robata.contracts.artifacts import ArtifactId
from robata.contracts.common import INT64_MAX, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry, default_schema_registry

PRIMARY_COMPLETION_RECORD_SCHEMA_ID = "https://schemas.robata.dev/primary-completion-record"
PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION = "2.0.0"
PRIMARY_COMPLETION_RECORD_WIRE_VERSION = "2.0"
PRIMARY_COMPLETION_RECORD_SEMANTIC_PROJECTION_VERSION = "primary-completion-record-semantic-v2"

NonNegativeInt = Annotated[int, Field(strict=True, ge=0, le=INT64_MAX)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=INT64_MAX)]
ValidatedRfc3339Timestamp = Annotated[
    Rfc3339Timestamp,
    Field(json_schema_extra={"format": "date-time"}),
]


def _parse_rfc3339_timestamp(value: str, *, field_name: str) -> datetime:
    if not value.endswith("Z"):
        offset_hour = int(value[-5:-3])
        offset_minute = int(value[-2:])
        if offset_hour > 23 or offset_minute > 59:
            raise ValueError(f"{field_name} must be a valid RFC3339 timestamp")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an RFC3339 timezone")
    return parsed


class PrimaryCompletionOutcome(StrEnum):
    """Terminal outcomes that constitute authoritative primary completion."""

    PRIMARY_COMPLETE = "PRIMARY_COMPLETE"
    PRIMARY_COMPLETE_NO_EVENTS = "PRIMARY_COMPLETE_NO_EVENTS"
    PRIMARY_COMPLETE_WITH_SKIPS = "PRIMARY_COMPLETE_WITH_SKIPS"


class DetailedResultArtifactReference(StrictModel):
    """Exact immutable reference to the schema-governed detailed run result."""

    artifact_id: ArtifactId
    exact_bytes_sha256: Sha256Digest
    byte_count: PositiveInt
    media_type: Literal["application/json"]
    schema_ref: SchemaRef


class PrimaryCompletionRecord(StrictModel):
    """Compact registered record committed with all primary publication facts."""

    schema_version: Literal["2.0"]
    schema_ref: SchemaRef
    semantic_projection_version: Literal["primary-completion-record-semantic-v2"]
    semantic_sha256: Sha256Digest

    run_id: OpaqueUuid
    recording_identity: Sha256Digest
    mcap_id: OpaqueUuid
    pipeline_version: SchemaVersion
    config_sha256: Sha256Digest
    started_at: ValidatedRfc3339Timestamp
    outcome: PrimaryCompletionOutcome

    barrier_definition_semantic_sha256: Sha256Digest
    barrier_reduction_semantic_sha256: Sha256Digest
    output_decision_semantic_sha256: Sha256Digest
    output_admission_policy_version: SchemaVersion
    output_admission_policy_sha256: Sha256Digest

    run_membership_count: NonNegativeInt
    run_membership_digest_root: Sha256Digest
    barrier_member_count: NonNegativeInt
    barrier_member_digest_root: Sha256Digest
    hypothesis_count: NonNegativeInt
    hypothesis_digest_root: Sha256Digest
    identity_assignment_count: NonNegativeInt
    identity_assignment_digest_root: Sha256Digest
    new_identity_count: NonNegativeInt
    new_identity_digest_root: Sha256Digest
    identity_relation_count: NonNegativeInt
    identity_relation_digest_root: Sha256Digest
    revision_count: NonNegativeInt
    revision_digest_root: Sha256Digest
    selection_decision_count: NonNegativeInt
    selection_decision_digest_root: Sha256Digest
    current_selection_count: NonNegativeInt
    current_selection_digest_root: Sha256Digest
    successor_outbox_count: NonNegativeInt
    successor_outbox_digest_root: Sha256Digest
    skipped_work_item_count: NonNegativeInt
    skipped_work_item_digest_root: Sha256Digest

    detailed_result: DetailedResultArtifactReference
    completed_at: ValidatedRfc3339Timestamp

    @model_validator(mode="after")
    def validate_completion_facts(self) -> Self:
        if (
            self.schema_ref.schema_id != PRIMARY_COMPLETION_RECORD_SCHEMA_ID
            or self.schema_ref.version != PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_ref must identify "
                f"{PRIMARY_COMPLETION_RECORD_SCHEMA_ID}"
                f"@{PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION}"
            )

        started_at = _parse_rfc3339_timestamp(self.started_at, field_name="started_at")
        completed_at = _parse_rfc3339_timestamp(self.completed_at, field_name="completed_at")
        if completed_at < started_at:
            raise ValueError("completed_at must be at or after started_at")

        if self.outcome is PrimaryCompletionOutcome.PRIMARY_COMPLETE_NO_EVENTS:
            event_collection_counts = {
                "hypothesis_count": self.hypothesis_count,
                "identity_assignment_count": self.identity_assignment_count,
                "new_identity_count": self.new_identity_count,
                "identity_relation_count": self.identity_relation_count,
                "revision_count": self.revision_count,
                "selection_decision_count": self.selection_decision_count,
                "current_selection_count": self.current_selection_count,
            }
            nonempty = [name for name, count in event_collection_counts.items() if count != 0]
            if nonempty:
                raise ValueError(
                    "PRIMARY_COMPLETE_NO_EVENTS requires zero counts for " + ", ".join(nonempty)
                )

        if self.outcome is PrimaryCompletionOutcome.PRIMARY_COMPLETE_WITH_SKIPS:
            if self.skipped_work_item_count == 0:
                raise ValueError("PRIMARY_COMPLETE_WITH_SKIPS requires skipped_work_item_count")
        elif self.skipped_work_item_count != 0:
            raise ValueError("skipped_work_item_count requires PRIMARY_COMPLETE_WITH_SKIPS")

        expected_digest = semantic_sha256(primary_completion_record_semantic_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match the primary completion projection")
        return self


def primary_completion_record_semantic_projection(
    record: PrimaryCompletionRecord,
) -> dict[str, Any]:
    """Return the V2 completion-clock- and row-identity-independent projection."""

    return {
        "semantic_projection_version": record.semantic_projection_version,
        "schema_version": record.schema_version,
        "schema_ref": record.schema_ref.model_dump(mode="json"),
        "run_id": record.run_id,
        "recording_identity": record.recording_identity,
        "mcap_id": record.mcap_id,
        "pipeline_version": record.pipeline_version,
        "config_sha256": record.config_sha256,
        "started_at": record.started_at,
        "outcome": record.outcome.value,
        "barrier_definition_semantic_sha256": (record.barrier_definition_semantic_sha256),
        "barrier_reduction_semantic_sha256": (record.barrier_reduction_semantic_sha256),
        "output_decision_semantic_sha256": record.output_decision_semantic_sha256,
        "output_admission_policy_version": record.output_admission_policy_version,
        "output_admission_policy_sha256": record.output_admission_policy_sha256,
        "run_membership_count": record.run_membership_count,
        "run_membership_digest_root": record.run_membership_digest_root,
        "barrier_member_count": record.barrier_member_count,
        "barrier_member_digest_root": record.barrier_member_digest_root,
        "hypothesis_count": record.hypothesis_count,
        "hypothesis_digest_root": record.hypothesis_digest_root,
        "identity_assignment_count": record.identity_assignment_count,
        "identity_assignment_digest_root": record.identity_assignment_digest_root,
        "new_identity_count": record.new_identity_count,
        "new_identity_digest_root": record.new_identity_digest_root,
        "identity_relation_count": record.identity_relation_count,
        "identity_relation_digest_root": record.identity_relation_digest_root,
        "revision_count": record.revision_count,
        "revision_digest_root": record.revision_digest_root,
        "selection_decision_count": record.selection_decision_count,
        "selection_decision_digest_root": record.selection_decision_digest_root,
        "current_selection_count": record.current_selection_count,
        "current_selection_digest_root": record.current_selection_digest_root,
        "successor_outbox_count": record.successor_outbox_count,
        "successor_outbox_digest_root": record.successor_outbox_digest_root,
        "skipped_work_item_count": record.skipped_work_item_count,
        "skipped_work_item_digest_root": record.skipped_work_item_digest_root,
        "detailed_result": {
            "exact_bytes_sha256": record.detailed_result.exact_bytes_sha256,
            "byte_count": record.detailed_result.byte_count,
            "media_type": record.detailed_result.media_type,
            "schema_ref": record.detailed_result.schema_ref.model_dump(mode="json"),
        },
    }


def create_primary_completion_record(
    *,
    schema_ref: SchemaRef,
    run_id: OpaqueUuid,
    recording_identity: Sha256Digest,
    mcap_id: OpaqueUuid,
    pipeline_version: SchemaVersion,
    config_sha256: Sha256Digest,
    started_at: Rfc3339Timestamp,
    outcome: PrimaryCompletionOutcome,
    barrier_definition_semantic_sha256: Sha256Digest,
    barrier_reduction_semantic_sha256: Sha256Digest,
    output_decision_semantic_sha256: Sha256Digest,
    output_admission_policy_version: SchemaVersion,
    output_admission_policy_sha256: Sha256Digest,
    run_membership_count: NonNegativeInt,
    run_membership_digest_root: Sha256Digest,
    barrier_member_count: NonNegativeInt,
    barrier_member_digest_root: Sha256Digest,
    hypothesis_count: NonNegativeInt,
    hypothesis_digest_root: Sha256Digest,
    identity_assignment_count: NonNegativeInt,
    identity_assignment_digest_root: Sha256Digest,
    new_identity_count: NonNegativeInt,
    new_identity_digest_root: Sha256Digest,
    identity_relation_count: NonNegativeInt,
    identity_relation_digest_root: Sha256Digest,
    revision_count: NonNegativeInt,
    revision_digest_root: Sha256Digest,
    selection_decision_count: NonNegativeInt,
    selection_decision_digest_root: Sha256Digest,
    current_selection_count: NonNegativeInt,
    current_selection_digest_root: Sha256Digest,
    successor_outbox_count: NonNegativeInt,
    successor_outbox_digest_root: Sha256Digest,
    skipped_work_item_count: NonNegativeInt,
    skipped_work_item_digest_root: Sha256Digest,
    detailed_result: DetailedResultArtifactReference,
    completed_at: Rfc3339Timestamp,
) -> PrimaryCompletionRecord:
    """Derive and validate a compact completion record from committed facts."""

    fields: dict[str, Any] = {
        "schema_version": PRIMARY_COMPLETION_RECORD_WIRE_VERSION,
        "schema_ref": schema_ref,
        "semantic_projection_version": (PRIMARY_COMPLETION_RECORD_SEMANTIC_PROJECTION_VERSION),
        "semantic_sha256": "0" * 64,
        "run_id": run_id,
        "recording_identity": recording_identity,
        "mcap_id": mcap_id,
        "pipeline_version": pipeline_version,
        "config_sha256": config_sha256,
        "started_at": started_at,
        "outcome": outcome,
        "barrier_definition_semantic_sha256": barrier_definition_semantic_sha256,
        "barrier_reduction_semantic_sha256": barrier_reduction_semantic_sha256,
        "output_decision_semantic_sha256": output_decision_semantic_sha256,
        "output_admission_policy_version": output_admission_policy_version,
        "output_admission_policy_sha256": output_admission_policy_sha256,
        "run_membership_count": run_membership_count,
        "run_membership_digest_root": run_membership_digest_root,
        "barrier_member_count": barrier_member_count,
        "barrier_member_digest_root": barrier_member_digest_root,
        "hypothesis_count": hypothesis_count,
        "hypothesis_digest_root": hypothesis_digest_root,
        "identity_assignment_count": identity_assignment_count,
        "identity_assignment_digest_root": identity_assignment_digest_root,
        "new_identity_count": new_identity_count,
        "new_identity_digest_root": new_identity_digest_root,
        "identity_relation_count": identity_relation_count,
        "identity_relation_digest_root": identity_relation_digest_root,
        "revision_count": revision_count,
        "revision_digest_root": revision_digest_root,
        "selection_decision_count": selection_decision_count,
        "selection_decision_digest_root": selection_decision_digest_root,
        "current_selection_count": current_selection_count,
        "current_selection_digest_root": current_selection_digest_root,
        "successor_outbox_count": successor_outbox_count,
        "successor_outbox_digest_root": successor_outbox_digest_root,
        "skipped_work_item_count": skipped_work_item_count,
        "skipped_work_item_digest_root": skipped_work_item_digest_root,
        "detailed_result": detailed_result,
        "completed_at": completed_at,
    }
    draft = PrimaryCompletionRecord.model_construct(**fields)
    fields["semantic_sha256"] = semantic_sha256(
        primary_completion_record_semantic_projection(draft)
    )
    return PrimaryCompletionRecord.model_validate(fields, strict=True)


def validate_registered_primary_completion_record(
    record: PrimaryCompletionRecord,
    registry: SchemaRegistry | None = None,
) -> PrimaryCompletionRecord:
    """Resolve both exact schema quartets and validate the registered wire payload."""

    active_registry = registry or default_schema_registry()
    active_registry.resolve_exact(record.schema_ref)
    active_registry.resolve_exact(record.detailed_result.schema_ref)
    validated = PrimaryCompletionRecord.model_validate(
        record.model_dump(mode="python"), strict=True
    )
    active_registry.validate_pinned(
        validated.schema_ref,
        validated.model_dump(mode="json"),
    )
    return validated


__all__ = [
    "PRIMARY_COMPLETION_RECORD_SCHEMA_ID",
    "PRIMARY_COMPLETION_RECORD_SCHEMA_VERSION",
    "PRIMARY_COMPLETION_RECORD_SEMANTIC_PROJECTION_VERSION",
    "PRIMARY_COMPLETION_RECORD_WIRE_VERSION",
    "DetailedResultArtifactReference",
    "PrimaryCompletionOutcome",
    "PrimaryCompletionRecord",
    "create_primary_completion_record",
    "primary_completion_record_semantic_projection",
    "validate_registered_primary_completion_record",
]
