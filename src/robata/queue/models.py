"""Authoritative durable-work ledger and barrier contracts.

``WorkItem`` is the persisted source of truth. Broker task values are only
delivery projections and must never be used to reconstruct ledger state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.queue.stage import DependencyCriticality, Stage

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class WorkItemSubjectType(StrEnum):
    """Classification of the subject referenced by a work item."""

    MCAP = "MCAP"
    WINDOW = "WINDOW"
    PACKAGE_SET = "PACKAGE_SET"
    PACKAGE = "PACKAGE"
    SPLIT_REDUCTION = "SPLIT_REDUCTION"
    CANDIDATE = "CANDIDATE"
    EVENT = "EVENT"
    INFERENCE = "INFERENCE"


class WorkItemState(StrEnum):
    """Canonical durable-work lifecycle from Architecture V1.1 section 25."""

    PLANNED = "PLANNED"
    READY = "READY"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCEEDED = "SUCCEEDED"
    FAILED_PERMANENT = "FAILED_PERMANENT"
    SKIPPED_POLICY = "SKIPPED_POLICY"
    SKIPPED_NOT_NEEDED = "SKIPPED_NOT_NEEDED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class WorkAttemptOutcome(StrEnum):
    """Persisted outcome of one fenced execution attempt."""

    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_PERMANENT = "FAILED_PERMANENT"
    ABANDONED = "ABANDONED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    SKIPPED = "SKIPPED"


TERMINAL_WORK_STATES = frozenset(
    {
        WorkItemState.SUCCEEDED,
        WorkItemState.FAILED_PERMANENT,
        WorkItemState.SKIPPED_POLICY,
        WorkItemState.SKIPPED_NOT_NEEDED,
        WorkItemState.CANCELLED,
        WorkItemState.EXPIRED,
        WorkItemState.INVALIDATED,
    }
)
SUCCESSFUL_DEPENDENCY_STATES = frozenset(
    {
        WorkItemState.SUCCEEDED,
        WorkItemState.SKIPPED_POLICY,
        WorkItemState.SKIPPED_NOT_NEEDED,
    }
)


def _parse_timestamp(value: str, field_name: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an RFC3339 timezone")
    return parsed.astimezone(UTC)


class WorkItemPlan(StrictModel):
    """Immutable identity and scheduling policy used to create a ledger row."""

    schema_version: Literal["1.0"] = "1.0"
    work_item_id: OpaqueUuid
    work_logical_key: NonEmptyString
    run_id: OpaqueUuid
    mcap_id: OpaqueUuid
    stage: Stage
    subject_type: WorkItemSubjectType
    subject_id: OpaqueUuid
    input_digest: Sha256Digest
    config_digest: Sha256Digest
    priority: NonNegativeInt = 0
    sla_deadline_at: Rfc3339Timestamp | None = None
    execution_expiry_at: Rfc3339Timestamp | None = None
    max_attempts: PositiveInt = 3
    trace_id: NonEmptyString | None = None
    created_at: Rfc3339Timestamp

    @field_validator("created_at", "sla_deadline_at", "execution_expiry_at")
    @classmethod
    def validate_timestamp(cls, value: str | None, info: object) -> str | None:
        if value is not None:
            field_name = getattr(info, "field_name", "timestamp")
            _parse_timestamp(value, field_name)
        return value


class WorkDependency(StrictModel):
    """Normalized directed edge from downstream work to existing upstream work."""

    dependency_id: OpaqueUuid
    downstream_work_item_id: OpaqueUuid
    upstream_work_item_id: OpaqueUuid
    criticality: DependencyCriticality = DependencyCriticality.REQUIRED

    @model_validator(mode="after")
    def validate_distinct_nodes(self) -> Self:
        if self.downstream_work_item_id == self.upstream_work_item_id:
            raise ValueError("a work item cannot depend on itself")
        return self


class WorkItem(WorkItemPlan):
    """Complete authoritative ledger snapshot for one unit of work."""

    state: WorkItemState
    cancel_requested: bool = False
    lease_epoch: NonNegativeInt = 0
    fencing_token: NonEmptyString | None = None
    leased_by: NonEmptyString | None = None
    lease_expires_at: Rfc3339Timestamp | None = None
    attempt: NonNegativeInt = 0
    retry_not_before_at: Rfc3339Timestamp | None = None
    terminal_reason_code: NonEmptyString | None = None
    terminal_reason_detail: NonEmptyString | None = None
    result_reference: NonEmptyString | None = None
    result_sha256: Sha256Digest | None = None
    completed_at: Rfc3339Timestamp | None = None
    updated_at: Rfc3339Timestamp
    row_version: NonNegativeInt = 0

    @field_validator(
        "lease_expires_at",
        "retry_not_before_at",
        "completed_at",
        "updated_at",
    )
    @classmethod
    def validate_mutable_timestamp(cls, value: str | None, info: object) -> str | None:
        if value is not None:
            field_name = getattr(info, "field_name", "timestamp")
            _parse_timestamp(value, field_name)
        return value

    @model_validator(mode="after")
    def validate_state_shape(self) -> Self:
        active_lease = self.state in {WorkItemState.LEASED, WorkItemState.RUNNING}
        lease_values = (self.fencing_token, self.leased_by, self.lease_expires_at)
        if active_lease and any(value is None for value in lease_values):
            raise ValueError("leased or running work requires complete lease metadata")
        if not active_lease and any(value is not None for value in lease_values):
            raise ValueError("only leased or running work may retain lease metadata")
        if active_lease and (self.lease_epoch == 0 or self.attempt == 0):
            raise ValueError("leased or running work requires positive epoch and attempt")

        if self.state is WorkItemState.RETRY_WAIT:
            if self.retry_not_before_at is None:
                raise ValueError("retry-wait work requires retry_not_before_at")
        elif self.retry_not_before_at is not None:
            raise ValueError("only retry-wait work may have retry_not_before_at")

        terminal = self.state in TERMINAL_WORK_STATES
        if terminal != (self.completed_at is not None):
            raise ValueError("terminal work and completed_at must be present together")
        if terminal and self.state is not WorkItemState.SUCCEEDED:
            if self.terminal_reason_code is None:
                raise ValueError("non-success terminal work requires a reason code")
        elif not terminal and self.terminal_reason_code is not None:
            raise ValueError("nonterminal work cannot have a terminal reason")
        if self.terminal_reason_detail is not None and self.terminal_reason_code is None:
            raise ValueError("terminal reason detail requires a reason code")

        result_values = (self.result_reference, self.result_sha256)
        if any(value is None for value in result_values) and any(
            value is not None for value in result_values
        ):
            raise ValueError("result reference and digest must be present together")
        if self.state is not WorkItemState.SUCCEEDED and self.result_reference is not None:
            raise ValueError("only succeeded work may publish a result")
        if _parse_timestamp(self.updated_at, "updated_at") < _parse_timestamp(
            self.created_at, "created_at"
        ):
            raise ValueError("updated_at cannot precede created_at")
        return self


class WorkLease(StrictModel):
    """Opaque capability required for every mutation of a claimed work item."""

    work_item_id: OpaqueUuid
    worker_id: NonEmptyString
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString
    lease_expires_at: Rfc3339Timestamp

    @field_validator("lease_expires_at")
    @classmethod
    def validate_expiry(cls, value: str) -> str:
        _parse_timestamp(value, "lease_expires_at")
        return value


class WorkLeaseClaim(StrictModel):
    """Atomic claim result containing the ledger snapshot and its capability."""

    work_item: WorkItem
    lease: WorkLease

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        item = self.work_item
        lease = self.lease
        if item.state is not WorkItemState.LEASED:
            raise ValueError("a claim must contain a leased work item")
        if (
            item.work_item_id != lease.work_item_id
            or item.leased_by != lease.worker_id
            or item.lease_epoch != lease.lease_epoch
            or item.fencing_token != lease.fencing_token
            or item.lease_expires_at != lease.lease_expires_at
        ):
            raise ValueError("claim lease does not match its work-item snapshot")
        return self


class WorkAttempt(StrictModel):
    """Append-oriented audit record for one claim epoch."""

    work_item_id: OpaqueUuid
    attempt_number: PositiveInt
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString
    worker_id: NonEmptyString
    claimed_at: Rfc3339Timestamp
    started_at: Rfc3339Timestamp | None = None
    completed_at: Rfc3339Timestamp | None = None
    outcome: WorkAttemptOutcome = WorkAttemptOutcome.ACTIVE
    error_code: NonEmptyString | None = None
    error_detail: NonEmptyString | None = None

    @field_validator("claimed_at", "started_at", "completed_at")
    @classmethod
    def validate_attempt_timestamp(cls, value: str | None, info: object) -> str | None:
        if value is not None:
            field_name = getattr(info, "field_name", "timestamp")
            _parse_timestamp(value, field_name)
        return value

    @model_validator(mode="after")
    def validate_outcome_shape(self) -> Self:
        if (self.outcome is WorkAttemptOutcome.ACTIVE) == (self.completed_at is not None):
            raise ValueError("active attempts must be open and terminal attempts must be completed")
        if self.error_detail is not None and self.error_code is None:
            raise ValueError("attempt error detail requires an error code")
        return self


class WorkBarrier(StrictModel):
    """A synchronization barrier that groups related work items."""

    barrier_id: OpaqueUuid
    logical_key: NonEmptyString
    subject: NonEmptyString
    expected_member_count: NonNegativeInt
    empty_semantics: NonEmptyString
    reduction_policy: NonEmptyString
    status: NonEmptyString


class WorkBarrierMember(StrictModel):
    """Membership of a single work item within a barrier."""

    member_id: OpaqueUuid
    barrier_id: OpaqueUuid
    work_item_id: OpaqueUuid
    ordinal: NonNegativeInt
    criticality: DependencyCriticality = DependencyCriticality.REQUIRED
    terminal_outcome: NonEmptyString | None = None


__all__ = [
    "SUCCESSFUL_DEPENDENCY_STATES",
    "TERMINAL_WORK_STATES",
    "DependencyCriticality",
    "NonEmptyString",
    "NonNegativeInt",
    "OpaqueUuid",
    "PositiveInt",
    "Rfc3339Timestamp",
    "WorkAttempt",
    "WorkAttemptOutcome",
    "WorkBarrier",
    "WorkBarrierMember",
    "WorkDependency",
    "WorkItem",
    "WorkItemPlan",
    "WorkItemState",
    "WorkItemSubjectType",
    "WorkLease",
    "WorkLeaseClaim",
]
