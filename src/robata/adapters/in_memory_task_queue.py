"""Deterministic provider-neutral in-memory task queue.

This adapter intentionally has no network, Redis, Celery, or credential
requirements.  It is a T2 scaffold and a contract-test implementation for
local development; a production adapter must provide equivalent atomicity and
durability guarantees behind :class:`robata.ports.task_queue.TaskQueue`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from threading import RLock

from robata.ports.task_queue import (
    InspectableTaskQueue,
    LeaseId,
    PipelineTask,
    TaskId,
    TaskQueueError,
    TaskQueueErrorCode,
    TaskSnapshot,
    TaskStatus,
)

Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TaskQueueError(
            TaskQueueErrorCode.INVALID_REQUEST,
            f"{field_name} must be an integer",
        )
    if value <= 0:
        raise TaskQueueError(
            TaskQueueErrorCode.INVALID_REQUEST,
            f"{field_name} must be positive",
        )
    return value


def _require_nonnegative_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskQueueError(
            TaskQueueErrorCode.INVALID_REQUEST,
            f"{field_name} must be a number",
        )
    if value < 0:
        raise TaskQueueError(
            TaskQueueErrorCode.INVALID_REQUEST,
            f"{field_name} must be non-negative",
        )
    return float(value)


def _require_now(clock: Clock) -> datetime:
    now = clock()
    if not isinstance(now, datetime):
        raise TypeError("clock must return datetime")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return now


@dataclass(slots=True)
class _Lease:
    lease_id: LeaseId
    worker_id: str
    expires_at: datetime
    duration_seconds: int


@dataclass(slots=True)
class _Record:
    task: PipelineTask
    status: TaskStatus
    available_at: datetime
    sequence: int
    lease: _Lease | None = None
    failure_reason: str | None = None
    result: bytes | None = None


class InMemoryTaskQueue(InspectableTaskQueue):
    """Single-process deterministic implementation of the task queue port.

    Ordering is stable: higher ``priority`` first, then earliest retry
    availability, then ``created_at``, then enqueue order.  Lease expiry is
    processed on every mutating/read operation and follows the same retry and
    dead-letter policy as explicit :meth:`fail` calls.

    ``max_size`` bounds pending + claimed work and provides a deterministic
    backpressure signal.  This adapter is intentionally not a durability
    implementation; process restart discards all state.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        retry_backoff_seconds: float = 1.0,
        max_size: int | None = None,
    ) -> None:
        self._clock = clock or _default_clock
        self._retry_backoff_seconds = _require_nonnegative_number(
            retry_backoff_seconds,
            "retry_backoff_seconds",
        )
        if max_size is not None:
            _require_positive_int(max_size, "max_size")
        self._max_size = max_size
        self._records: dict[TaskId, _Record] = {}
        self._leases: dict[LeaseId, _Lease] = {}
        self._retired_leases: dict[LeaseId, TaskStatus] = {}
        self._dead_letter_order: list[TaskId] = []
        self._next_sequence = 0
        self._next_lease_number = 0
        self._lock = RLock()

    @property
    def depth(self) -> int:
        """Return the count of pending or currently claimed tasks."""

        with self._lock:
            self._expire_due(_require_now(self._clock))
            return sum(
                record.status in {TaskStatus.PENDING, TaskStatus.CLAIMED}
                for record in self._records.values()
            )

    def enqueue(self, task: PipelineTask) -> TaskId:
        """Add a task, rejecting duplicate IDs and bounded-queue overflow."""

        if not isinstance(task, PipelineTask):
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "task must be PipelineTask",
            )
        if task.lease_id is not None:
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "a task with lease metadata cannot be enqueued",
            )
        with self._lock:
            now = _require_now(self._clock)
            self._expire_due(now)
            if task.task_id in self._records:
                raise TaskQueueError(
                    TaskQueueErrorCode.DUPLICATE_TASK,
                    f"task already exists: {task.task_id}",
                )
            active_count = sum(
                record.status in {TaskStatus.PENDING, TaskStatus.CLAIMED}
                for record in self._records.values()
            )
            if self._max_size is not None and active_count >= self._max_size:
                raise TaskQueueError(
                    TaskQueueErrorCode.QUEUE_FULL,
                    f"queue capacity exceeded ({self._max_size})",
                )
            # Preserve caller timestamps for deterministic ordering but ensure
            # the queue's availability is never in the past for a new task.
            available_at = max(task.created_at, now)
            self._records[task.task_id] = _Record(
                task=task,
                status=TaskStatus.PENDING,
                available_at=available_at,
                sequence=self._next_sequence,
            )
            self._next_sequence += 1
            return task.task_id

    def claim(self, worker_id: str, lease_duration_seconds: int) -> PipelineTask | None:
        """Claim the highest-priority eligible task for ``worker_id``."""

        if not isinstance(worker_id, str) or not worker_id.strip():
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "worker_id must be a non-empty string",
            )
        duration = _require_positive_int(lease_duration_seconds, "lease_duration_seconds")
        with self._lock:
            now = _require_now(self._clock)
            self._expire_due(now)
            candidates = [
                record
                for record in self._records.values()
                if record.status is TaskStatus.PENDING and record.available_at <= now
            ]
            if not candidates:
                return None
            record = min(
                candidates,
                key=lambda item: (
                    -item.task.priority,
                    item.available_at,
                    item.task.created_at,
                    item.sequence,
                ),
            )
            lease_id = LeaseId(f"lease-{self._next_lease_number:08d}")
            self._next_lease_number += 1
            lease = _Lease(
                lease_id=lease_id,
                worker_id=worker_id,
                expires_at=now + timedelta(seconds=duration),
                duration_seconds=duration,
            )
            record.lease = lease
            record.status = TaskStatus.CLAIMED
            self._leases[lease_id] = lease
            return replace(
                record.task,
                lease_id=lease.lease_id,
                leased_by=lease.worker_id,
                lease_expires_at=lease.expires_at,
            )

    def heartbeat(
        self,
        lease_id: LeaseId,
        lease_duration_seconds: int | None = None,
    ) -> bool:
        """Renew an active lease; return ``False`` for unknown/expired leases."""

        if not isinstance(lease_id, LeaseId):
            return False
        if lease_duration_seconds is not None:
            try:
                duration = _require_positive_int(
                    lease_duration_seconds,
                    "lease_duration_seconds",
                )
            except TaskQueueError:
                return False
        else:
            duration = None
        with self._lock:
            now = _require_now(self._clock)
            self._expire_due(now)
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            if duration is None:
                duration = lease.duration_seconds
            lease.duration_seconds = duration
            lease.expires_at = now + timedelta(seconds=duration)
            record = self._find_record_for_lease(lease)
            if record is None:
                self._leases.pop(lease_id, None)
                self._retired_leases[lease_id] = TaskStatus.FAILED
                return False
            return True

    def complete(self, lease_id: LeaseId, result: bytes) -> None:
        """Complete an active lease and retain exact result bytes."""

        if not isinstance(result, bytes):
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "result must be bytes",
            )
        with self._lock:
            now = _require_now(self._clock)
            record, lease = self._active_lease(lease_id, now)
            record.status = TaskStatus.COMPLETED
            record.result = result
            record.failure_reason = None
            record.lease = None
            self._leases.pop(lease.lease_id, None)
            self._retired_leases[lease.lease_id] = TaskStatus.COMPLETED

    def fail(self, lease_id: LeaseId, reason: str) -> None:
        """Fail an active lease and apply deterministic retry/DLQ policy."""

        if not isinstance(reason, str) or not reason.strip():
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "reason must be a non-empty string",
            )
        with self._lock:
            now = _require_now(self._clock)
            record, lease = self._active_lease(lease_id, now)
            self._leases.pop(lease.lease_id, None)
            self._retired_leases[lease.lease_id] = TaskStatus.FAILED
            record.lease = None
            self._schedule_after_failure(record, reason.strip(), now)

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """Return a task status, applying lease expiry first."""

        with self._lock:
            self._expire_due(_require_now(self._clock))
            return self._record_for_task(task_id).status

    def inspect(self, task_id: TaskId) -> TaskSnapshot:
        """Return immutable state for one task."""

        with self._lock:
            self._expire_due(_require_now(self._clock))
            record = self._record_for_task(task_id)
            lease = record.lease
            return TaskSnapshot(
                task_id=record.task.task_id,
                status=record.status,
                retry_count=record.task.retry_count,
                max_retries=record.task.max_retries,
                available_at=record.available_at if record.status is TaskStatus.PENDING else None,
                lease_id=lease.lease_id if lease else None,
                leased_by=lease.worker_id if lease else None,
                lease_expires_at=lease.expires_at if lease else None,
                failure_reason=record.failure_reason,
                result=record.result,
            )

    def get_result(self, task_id: TaskId) -> bytes | None:
        """Return a completed result, or ``None`` when not completed."""

        with self._lock:
            self._expire_due(_require_now(self._clock))
            return self._record_for_task(task_id).result

    def sweep_expired(self) -> int:
        """Requeue/dead-letter every lease that has expired."""

        with self._lock:
            return self._expire_due(_require_now(self._clock))

    def list_dead_letters(self) -> tuple[TaskSnapshot, ...]:
        """Return dead-letter snapshots in deterministic insertion order."""

        with self._lock:
            self._expire_due(_require_now(self._clock))
            return tuple(
                self.inspect(task_id)
                for task_id in self._dead_letter_order
                if task_id in self._records
            )

    def _record_for_task(self, task_id: TaskId) -> _Record:
        if not isinstance(task_id, TaskId):
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "task_id must be TaskId",
            )
        record = self._records.get(task_id)
        if record is None:
            raise TaskQueueError(
                TaskQueueErrorCode.TASK_NOT_FOUND,
                f"task not found: {task_id}",
            )
        return record

    def _active_lease(self, lease_id: LeaseId, now: datetime) -> tuple[_Record, _Lease]:
        if not isinstance(lease_id, LeaseId):
            raise TaskQueueError(
                TaskQueueErrorCode.INVALID_REQUEST,
                "lease_id must be LeaseId",
            )
        lease = self._leases.get(lease_id)
        if lease is None:
            if self._retired_leases.get(lease_id) is TaskStatus.FAILED:
                raise TaskQueueError(
                    TaskQueueErrorCode.LEASE_EXPIRED,
                    f"lease is no longer active: {lease_id}",
                )
            raise TaskQueueError(
                TaskQueueErrorCode.LEASE_NOT_FOUND,
                f"lease not found: {lease_id}",
            )
        if now >= lease.expires_at:
            record = self._find_record_for_lease(lease)
            if record is not None:
                self._expire_lease(record, now)
            raise TaskQueueError(
                TaskQueueErrorCode.LEASE_EXPIRED,
                f"lease expired: {lease_id}",
            )
        record = self._find_record_for_lease(lease)
        if record is None:
            raise TaskQueueError(
                TaskQueueErrorCode.LEASE_NOT_FOUND,
                f"lease record not found: {lease_id}",
            )
        if record.status is not TaskStatus.CLAIMED:
            raise TaskQueueError(
                TaskQueueErrorCode.TASK_NOT_CLAIMED,
                f"task is not claimed by lease: {lease_id}",
            )
        return record, lease

    def _find_record_for_lease(self, lease: _Lease) -> _Record | None:
        return next(
            (item for item in self._records.values() if item.lease is lease),
            None,
        )

    def _expire_due(self, now: datetime) -> int:
        expired = [
            record
            for record in self._records.values()
            if record.status is TaskStatus.CLAIMED
            and record.lease is not None
            and now >= record.lease.expires_at
        ]
        for record in expired:
            self._expire_lease(record, now)
        return len(expired)

    def _expire_lease(self, record: _Record, now: datetime) -> None:
        lease = record.lease
        if lease is None:
            return
        self._leases.pop(lease.lease_id, None)
        self._retired_leases[lease.lease_id] = TaskStatus.FAILED
        record.lease = None
        self._schedule_after_failure(record, "lease expired", now)

    def _schedule_after_failure(self, record: _Record, reason: str, now: datetime) -> None:
        next_retry_count = record.task.retry_count + 1
        updated_task = replace(record.task, retry_count=next_retry_count)
        record.task = updated_task
        record.failure_reason = reason
        record.result = None
        if next_retry_count > updated_task.max_retries:
            record.status = TaskStatus.DEAD_LETTER
            record.available_at = now
            if record.task.task_id not in self._dead_letter_order:
                self._dead_letter_order.append(record.task.task_id)
            return
        delay_seconds = self._retry_backoff_seconds * (2 ** (next_retry_count - 1))
        record.available_at = now + timedelta(seconds=delay_seconds)
        record.status = TaskStatus.PENDING


__all__ = ["InMemoryTaskQueue"]
