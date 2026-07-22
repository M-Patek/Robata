"""Provider-neutral broker-delivery contracts.

This port transports payload bytes and offers broker-local retry/lease
mechanics. It is not the authoritative durable-work ledger: dependency state,
execution deadlines, lease epochs, fencing tokens, and terminal outcomes belong
to the work scheduler. A broker acknowledgement alone can never commit work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol


class TaskStatus(StrEnum):
    """Lifecycle states exposed by a queue implementation."""

    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


class LeaseStatus(StrEnum):
    """State returned by optional queue inspection helpers."""

    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    RELEASED = "RELEASED"
    UNKNOWN = "UNKNOWN"


class TaskQueueErrorCode(StrEnum):
    """Stable machine-readable failures at the task queue boundary."""

    ADAPTER_UNAVAILABLE = "ADAPTER_UNAVAILABLE"
    INVALID_REQUEST = "INVALID_REQUEST"
    DUPLICATE_TASK = "DUPLICATE_TASK"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    LEASE_NOT_FOUND = "LEASE_NOT_FOUND"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    TASK_NOT_CLAIMED = "TASK_NOT_CLAIMED"
    TASK_ALREADY_COMPLETED = "TASK_ALREADY_COMPLETED"
    QUEUE_FULL = "QUEUE_FULL"


class TaskQueueError(RuntimeError):
    """A queue operation failed with a stable error code."""

    def __init__(self, code: TaskQueueErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class TaskId:
    """Opaque, caller-assigned task identity."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("task id must be a non-empty string")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class LeaseId:
    """Opaque queue-issued lease identity."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("lease id must be a non-empty string")

    def __str__(self) -> str:
        return self.value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class PipelineTask:
    """A provider-neutral unit of pipeline work.

    ``lease_id``, ``leased_by`` and ``lease_expires_at`` are populated only on
    the value returned by :meth:`TaskQueue.claim`; callers must leave them empty
    when enqueueing a new task.  Keeping the lease metadata on the immutable
    value preserves the historical ``claim(...) -> PipelineTask`` shape while
    still making the lease required for heartbeat/complete/fail operations.
    """

    task_id: TaskId
    recording_id: str
    stage: str
    payload: bytes
    priority: int = 0
    created_at: datetime = field(default_factory=_utc_now)
    retry_count: int = 0
    max_retries: int = 3
    lease_id: LeaseId | None = None
    leased_by: str | None = None
    lease_expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, TaskId):
            raise TypeError("task_id must be TaskId")
        if not isinstance(self.recording_id, str) or not self.recording_id.strip():
            raise ValueError("recording_id must be a non-empty string")
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(self.payload, bytes):
            raise TypeError("payload must be bytes")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an integer")
        _validate_datetime(self.created_at, "created_at")
        for field_name, value in (
            ("retry_count", self.retry_count),
            ("max_retries", self.max_retries),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.lease_id is None:
            if self.leased_by is not None or self.lease_expires_at is not None:
                raise ValueError("lease metadata requires lease_id")
        else:
            if not isinstance(self.lease_id, LeaseId):
                raise TypeError("lease_id must be LeaseId")
            if not isinstance(self.leased_by, str) or not self.leased_by.strip():
                raise ValueError("leased_by must be a non-empty string when leased")
            if self.lease_expires_at is None:
                raise ValueError("lease_expires_at is required when leased")
            _validate_datetime(self.lease_expires_at, "lease_expires_at")


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """Read-only operational state for one task.

    Queue adapters may expose richer fields, but these fields are intentionally
    provider-neutral and sufficient for retries, dead-letter inspection, and
    deterministic tests.
    """

    task_id: TaskId
    status: TaskStatus
    retry_count: int
    max_retries: int
    available_at: datetime | None
    lease_id: LeaseId | None
    leased_by: str | None
    lease_expires_at: datetime | None
    failure_reason: str | None
    result: bytes | None

    def __post_init__(self) -> None:
        _validate_datetime(self.available_at, "available_at") if self.available_at else None
        if self.lease_expires_at is not None:
            _validate_datetime(self.lease_expires_at, "lease_expires_at")
        if self.retry_count < 0 or self.max_retries < 0:
            raise ValueError("retry counters must be non-negative")


class TaskQueue(Protocol):
    """Optional broker-delivery port, never the scheduling source of truth.

    Implementations make their own delivery claim and acknowledgement operations
    atomic. Durable workers must still present the scheduler-issued epoch and
    fencing token when committing an authoritative work outcome.
    """

    def enqueue(self, task: PipelineTask) -> TaskId:
        """Add a task to the queue, returning its caller-assigned ID."""
        ...

    def claim(self, worker_id: str, lease_duration_seconds: int) -> PipelineTask | None:
        """Claim the next eligible task and return it with lease metadata."""
        ...

    def heartbeat(
        self,
        lease_id: LeaseId,
        lease_duration_seconds: int | None = None,
    ) -> bool:
        """Renew a lease; return ``False`` when the lease is unknown/expired."""
        ...

    def complete(self, lease_id: LeaseId, result: bytes) -> None:
        """Mark the leased task complete and persist its result bytes."""
        ...

    def fail(self, lease_id: LeaseId, reason: str) -> None:
        """Fail a lease and schedule retry or dead-letter according to policy."""
        ...

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """Return the current lifecycle state for a task."""
        ...


class InspectableTaskQueue(TaskQueue, Protocol):
    """Optional read-side extensions implemented by the local fake."""

    @property
    def depth(self) -> int:
        """Number of pending or claimed tasks."""
        ...

    def inspect(self, task_id: TaskId) -> TaskSnapshot:
        """Return immutable operational state for one task."""
        ...

    def get_result(self, task_id: TaskId) -> bytes | None:
        """Return a completed result, or ``None`` when no result exists."""
        ...

    def sweep_expired(self) -> int:
        """Requeue/dead-letter expired leases and return the number processed."""
        ...

    def list_dead_letters(self) -> tuple[TaskSnapshot, ...]:
        """Return dead-letter tasks in deterministic insertion order."""
        ...


__all__ = [
    "InspectableTaskQueue",
    "LeaseId",
    "LeaseStatus",
    "PipelineTask",
    "TaskId",
    "TaskQueue",
    "TaskQueueError",
    "TaskQueueErrorCode",
    "TaskSnapshot",
    "TaskStatus",
]
