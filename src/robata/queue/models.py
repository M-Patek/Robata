"""Core queue models for work-item lifecycle, dependencies, barriers, and outbox events."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp


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


class DependencyCriticality(StrEnum):
    """How strongly a downstream work item depends on an upstream one."""

    REQUIRED = "REQUIRED"
    DEGRADABLE = "DEGRADABLE"
    OPTIONAL = "OPTIONAL"


class OutboxEventStatus(StrEnum):
    """Lifecycle of an outbox event awaiting delivery."""

    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"


class WorkItem(StrictModel):
    """A single unit of pipeline work with identity, lineage, and scheduling metadata.

    ``work_item_id`` is the canonical UUID.  ``work_logical_key`` ties the item
    to a logical node.  ``subject_type`` and ``subject_id`` describe what entity
    the work operates on.  ``dependencies`` is a flat list of upstream
    ``work_item_id`` values.  ``lease_epoch`` and ``fencing_token`` are used by
    the queue adapter for optimistic concurrency control.
    """

    work_item_id: OpaqueUuid
    work_logical_key: NonEmptyString
    run_id: OpaqueUuid
    mcap_id: OpaqueUuid
    stage: NonEmptyString
    subject_type: WorkItemSubjectType
    subject_id: OpaqueUuid
    dependencies: tuple[OpaqueUuid, ...] = ()
    input_digest: NonEmptyString
    config_digest: NonEmptyString
    priority: NonNegativeInt = 0
    sla_deadline_at: Rfc3339Timestamp | None = None
    execution_expiry_at: Rfc3339Timestamp | None = None
    cancel_requested: bool = False
    lease_epoch: NonNegativeInt = 0
    fencing_token: NonNegativeInt = 0
    attempt: NonNegativeInt = 0
    trace_id: NonEmptyString | None = None


class WorkDependency(StrictModel):
    """Directed edge from a downstream work item to an upstream work item."""

    dependency_id: OpaqueUuid
    downstream_work_item_id: OpaqueUuid
    upstream_work_item_id: OpaqueUuid
    criticality: DependencyCriticality = DependencyCriticality.REQUIRED


class WorkBarrier(StrictModel):
    """A synchronization barrier that groups related work items.

    Barriers are used for fan-out/reduction patterns where multiple work items
    must complete before a downstream stage can proceed.
    """

    barrier_id: OpaqueUuid
    logical_key: NonEmptyString
    subject: NonEmptyString
    expected_member_count: PositiveInt
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


class OutboxEvent(StrictModel):
    """Transactional outbox row for at-least-once event delivery.

    Events are written to the outbox table in the same transaction as the
    business mutation, then asynchronously relayed to the message broker.
    """

    event_id: OpaqueUuid
    topic: NonEmptyString
    key: NonEmptyString
    payload_reference: NonEmptyString
    status: OutboxEventStatus = OutboxEventStatus.PENDING
    attempts: NonNegativeInt = 0


__all__ = [
    "DependencyCriticality",
    "NonEmptyString",
    "NonNegativeInt",
    "OpaqueUuid",
    "OutboxEvent",
    "OutboxEventStatus",
    "PositiveInt",
    "Rfc3339Timestamp",
    "WorkBarrier",
    "WorkBarrierMember",
    "WorkDependency",
    "WorkItem",
    "WorkItemSubjectType",
]
