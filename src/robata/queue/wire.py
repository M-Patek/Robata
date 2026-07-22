"""Registered wire projections for durable work and barrier delivery.

The scheduler and barrier stores remain authoritative. These models are
immutable transport/read projections; a consumer must present the exact
schema reference and the originating store rebuilds a fresh projection after
each state change.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self, cast

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry, default_schema_registry
from robata.queue.barrier import Barrier, BarrierMember, BarrierState
from robata.queue.models import WorkDependency, WorkItem, WorkItemState, WorkItemSubjectType
from robata.queue.stage import DependencyCriticality, Stage

WORK_MESSAGE_SCHEMA_ID = "https://schemas.robata.dev/work-message"
WORK_MESSAGE_SCHEMA_VERSION = "1.0.0"
WORK_MESSAGE_WIRE_VERSION: Literal["1.0"] = "1.0"
WORK_MESSAGE_PROJECTION_VERSION = "work-message-v1"

PERSISTED_BARRIER_SCHEMA_ID = "https://schemas.robata.dev/persisted-barrier"
PERSISTED_BARRIER_SCHEMA_VERSION = "1.0.0"
PERSISTED_BARRIER_WIRE_VERSION: Literal["1.0"] = "1.0"
PERSISTED_BARRIER_PROJECTION_VERSION = "persisted-barrier-v1"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
TerminalOutcome = Literal[
    "SUCCEEDED",
    "SKIPPED_POLICY",
    "SKIPPED_NOT_NEEDED",
    "FAILED",
    "CANCELLED",
    "EXPIRED",
    "QUARANTINED",
    "INCOMPLETE",
]
_SUCCESS_OUTCOMES = frozenset({"SUCCEEDED", "SKIPPED_POLICY", "SKIPPED_NOT_NEEDED"})
_FAILURE_OUTCOMES = frozenset({"FAILED", "CANCELLED", "EXPIRED", "QUARANTINED", "INCOMPLETE"})


def _parse_timestamp(value: str, field_name: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an RFC3339 timezone")
    return parsed.astimezone(UTC)


class WorkMessageDependency(StrictModel):
    """One dependency edge in the broker delivery projection."""

    work_item_id: NonEmptyString
    criticality: DependencyCriticality


class WorkMessage(StrictModel):
    """Lease-bound delivery projection of one authoritative work item."""

    schema_version: Literal["1.0"]
    schema_ref: SchemaRef
    work_item_id: OpaqueUuid
    work_logical_key: NonEmptyString
    run_id: OpaqueUuid
    mcap_id: OpaqueUuid
    stage: Stage
    subject_type: WorkItemSubjectType
    subject_id: OpaqueUuid
    dependencies: tuple[WorkMessageDependency, ...]
    input_digest: Sha256Digest
    config_digest: Sha256Digest
    priority: NonNegativeInt
    sla_deadline_at: Rfc3339Timestamp | None
    execution_expiry_at: Rfc3339Timestamp | None
    cancel_requested: bool
    lease_epoch: PositiveInt
    fencing_token: NonEmptyString
    attempt: PositiveInt
    trace_id: NonEmptyString | None

    @model_validator(mode="after")
    def validate_wire_identity(self) -> Self:
        if (
            self.schema_ref.schema_id != WORK_MESSAGE_SCHEMA_ID
            or self.schema_ref.version != WORK_MESSAGE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"schema_ref must identify {WORK_MESSAGE_SCHEMA_ID}@{WORK_MESSAGE_SCHEMA_VERSION}"
            )
        if tuple(self.dependencies) != tuple(
            sorted(self.dependencies, key=lambda item: item.work_item_id)
        ):
            raise ValueError("work-message dependencies must be canonically ordered")
        if len({item.work_item_id for item in self.dependencies}) != len(self.dependencies):
            raise ValueError("work-message dependency ids must be unique")
        for field_name, value in (
            ("sla_deadline_at", self.sla_deadline_at),
            ("execution_expiry_at", self.execution_expiry_at),
        ):
            if value is not None:
                _parse_timestamp(value, field_name)
        return self

    @classmethod
    def from_ledger(
        cls,
        item: WorkItem,
        dependencies: tuple[WorkDependency, ...],
        *,
        schema_ref: SchemaRef,
    ) -> WorkMessage:
        """Build a message only from a complete active scheduler lease."""

        if item.state not in {WorkItemState.LEASED, WorkItemState.RUNNING}:
            raise ValueError("work-message requires a leased or running work item")
        if item.fencing_token is None or item.lease_epoch < 1 or item.attempt < 1:
            raise ValueError("work-message requires complete active lease metadata")
        if item.lease_expires_at is None or item.leased_by is None:
            raise ValueError("work-message requires lease owner and expiry in the ledger")
        if any(dep.downstream_work_item_id != item.work_item_id for dep in dependencies):
            raise ValueError("work-message dependency does not bind to its downstream item")
        projected = tuple(
            WorkMessageDependency(
                work_item_id=dependency.upstream_work_item_id,
                criticality=dependency.criticality,
            )
            for dependency in sorted(dependencies, key=lambda value: value.upstream_work_item_id)
        )
        return cls(
            schema_version=WORK_MESSAGE_WIRE_VERSION,
            schema_ref=schema_ref,
            work_item_id=item.work_item_id,
            work_logical_key=item.work_logical_key,
            run_id=item.run_id,
            mcap_id=item.mcap_id,
            stage=item.stage,
            subject_type=item.subject_type,
            subject_id=item.subject_id,
            dependencies=projected,
            input_digest=item.input_digest,
            config_digest=item.config_digest,
            priority=item.priority,
            sla_deadline_at=item.sla_deadline_at,
            execution_expiry_at=item.execution_expiry_at,
            cancel_requested=item.cancel_requested,
            lease_epoch=item.lease_epoch,
            fencing_token=item.fencing_token,
            attempt=item.attempt,
            trace_id=item.trace_id,
        )


class PersistedBarrierStatus(StrEnum):
    """Durable generic-barrier status, separate from member outcomes."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class PersistedBarrierMember(StrictModel):
    """Terminal member fact included in a barrier snapshot."""

    work_item_id: NonEmptyString
    criticality: DependencyCriticality
    outcome: TerminalOutcome


class PersistedBarrier(StrictModel):
    """Atomic snapshot of a generic SQLite barrier and its terminal members."""

    schema_version: Literal["1.0"]
    schema_ref: SchemaRef
    barrier_id: OpaqueUuid
    logical_key: NonEmptyString
    expected_member_count: NonNegativeInt
    empty_semantics: TerminalOutcome
    reduction_policy: NonEmptyString
    required_success_count: NonNegativeInt
    max_degraded_failures: NonNegativeInt
    state_version: NonNegativeInt
    completed_members: NonNegativeInt
    pending_members: NonNegativeInt
    failed_members: NonNegativeInt
    status: PersistedBarrierStatus
    members: tuple[PersistedBarrierMember, ...]

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if (
            self.schema_ref.schema_id != PERSISTED_BARRIER_SCHEMA_ID
            or self.schema_ref.version != PERSISTED_BARRIER_SCHEMA_VERSION
        ):
            raise ValueError(
                "schema_ref must identify "
                f"{PERSISTED_BARRIER_SCHEMA_ID}@{PERSISTED_BARRIER_SCHEMA_VERSION}"
            )
        if self.required_success_count > self.expected_member_count:
            raise ValueError("required_success_count cannot exceed expected_member_count")
        if self.max_degraded_failures > self.expected_member_count:
            raise ValueError("max_degraded_failures cannot exceed expected_member_count")
        if tuple(self.members) != tuple(sorted(self.members, key=lambda item: item.work_item_id)):
            raise ValueError("persisted-barrier members must be canonically ordered")
        if len({item.work_item_id for item in self.members}) != len(self.members):
            raise ValueError("persisted-barrier member ids must be unique")
        if len(self.members) > self.expected_member_count:
            raise ValueError("persisted-barrier member capacity exceeded")
        if self.completed_members != len(self.members):
            raise ValueError("completed_members must equal the persisted member count")
        if self.pending_members != self.expected_member_count - len(self.members):
            raise ValueError("pending_members must reconcile with expected_member_count")
        failed = sum(item.outcome in _FAILURE_OUTCOMES for item in self.members)
        if self.failed_members != failed:
            raise ValueError("failed_members must reconcile with terminal member outcomes")

        if self.expected_member_count == 0:
            expected_status = (
                PersistedBarrierStatus.FAILED
                if self.empty_semantics in _FAILURE_OUTCOMES
                else PersistedBarrierStatus.CLOSED
            )
        elif len(self.members) < self.expected_member_count:
            expected_status = PersistedBarrierStatus.OPEN
        else:
            successful = sum(item.outcome in _SUCCESS_OUTCOMES for item in self.members)
            required_failed = any(
                item.criticality is DependencyCriticality.REQUIRED
                and item.outcome not in _SUCCESS_OUTCOMES
                for item in self.members
            )
            degraded = sum(
                item.criticality is DependencyCriticality.DEGRADABLE
                and item.outcome in _FAILURE_OUTCOMES
                for item in self.members
            )
            expected_status = (
                PersistedBarrierStatus.CLOSED
                if not required_failed
                and successful >= self.required_success_count
                and degraded <= self.max_degraded_failures
                else PersistedBarrierStatus.FAILED
            )
        if self.status is not expected_status:
            raise ValueError("persisted-barrier status does not reconcile with members")
        return self

    @classmethod
    def from_snapshot(
        cls,
        barrier: Barrier,
        state: BarrierState,
        members: tuple[BarrierMember, ...],
        state_version: int,
        *,
        schema_ref: SchemaRef,
    ) -> PersistedBarrier:
        if state.barrier_id != barrier.barrier_id:
            raise ValueError("barrier definition and state ids do not match")
        return cls(
            schema_version=PERSISTED_BARRIER_WIRE_VERSION,
            schema_ref=schema_ref,
            barrier_id=barrier.barrier_id,
            logical_key=barrier.logical_key,
            expected_member_count=barrier.expected_member_count,
            empty_semantics=cast(TerminalOutcome, barrier.empty_semantics),
            reduction_policy=barrier.reduction_policy,
            required_success_count=barrier.required_success_count,
            max_degraded_failures=barrier.max_degraded_failures,
            state_version=state_version,
            completed_members=state.completed_members,
            pending_members=state.pending_members,
            failed_members=state.failed_members,
            status=PersistedBarrierStatus(state.status),
            members=tuple(
                PersistedBarrierMember(
                    work_item_id=member.work_item_id,
                    criticality=member.criticality,
                    outcome=cast(TerminalOutcome, member.outcome.value),
                )
                for member in sorted(members, key=lambda value: value.work_item_id)
            ),
        )


def validate_registered_work_message(
    message: WorkMessage,
    registry: SchemaRegistry | None = None,
) -> WorkMessage:
    active_registry = registry or default_schema_registry()
    active_registry.resolve_exact(message.schema_ref)
    checked = WorkMessage.model_validate(message.model_dump(mode="python"), strict=True)
    active_registry.validate_pinned(checked.schema_ref, checked.model_dump(mode="json"))
    return checked


def validate_registered_persisted_barrier(
    barrier: PersistedBarrier,
    registry: SchemaRegistry | None = None,
) -> PersistedBarrier:
    active_registry = registry or default_schema_registry()
    active_registry.resolve_exact(barrier.schema_ref)
    checked = PersistedBarrier.model_validate(barrier.model_dump(mode="python"), strict=True)
    active_registry.validate_pinned(checked.schema_ref, checked.model_dump(mode="json"))
    return checked


__all__ = [
    "PERSISTED_BARRIER_PROJECTION_VERSION",
    "PERSISTED_BARRIER_SCHEMA_ID",
    "PERSISTED_BARRIER_SCHEMA_VERSION",
    "PERSISTED_BARRIER_WIRE_VERSION",
    "WORK_MESSAGE_PROJECTION_VERSION",
    "WORK_MESSAGE_SCHEMA_ID",
    "WORK_MESSAGE_SCHEMA_VERSION",
    "WORK_MESSAGE_WIRE_VERSION",
    "PersistedBarrier",
    "PersistedBarrierMember",
    "PersistedBarrierStatus",
    "WorkMessage",
    "WorkMessageDependency",
    "validate_registered_persisted_barrier",
    "validate_registered_work_message",
]
