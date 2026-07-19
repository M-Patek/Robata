"""Contract tests for the provider-neutral task queue scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from robata.adapters.in_memory_task_queue import InMemoryTaskQueue
from robata.ports.task_queue import (
    LeaseId,
    PipelineTask,
    TaskId,
    TaskQueueError,
    TaskQueueErrorCode,
    TaskStatus,
)


@dataclass
class _FakeClock:
    now: datetime

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _task(
    task_id: str,
    *,
    priority: int = 0,
    created_at: datetime | None = None,
    max_retries: int = 3,
) -> PipelineTask:
    return PipelineTask(
        task_id=TaskId(task_id),
        recording_id="recording-1",
        stage="QA_INFERENCE",
        payload=task_id.encode(),
        priority=priority,
        created_at=created_at or datetime(2026, 7, 19, tzinfo=UTC),
        max_retries=max_retries,
    )


def test_claim_orders_priority_and_returns_lease_metadata() -> None:
    clock = _FakeClock(datetime(2026, 7, 19, tzinfo=UTC))
    queue = InMemoryTaskQueue(clock=clock)
    queue.enqueue(_task("low", priority=1))
    queue.enqueue(_task("high", priority=10))

    claimed = queue.claim("worker-a", lease_duration_seconds=30)

    assert claimed is not None
    assert claimed.task_id == TaskId("high")
    assert claimed.lease_id == LeaseId("lease-00000000")
    assert claimed.leased_by == "worker-a"
    assert claimed.lease_expires_at == clock.now + timedelta(seconds=30)
    assert queue.get_status(TaskId("high")) is TaskStatus.CLAIMED
    assert queue.depth == 2


def test_heartbeat_and_complete_persist_result() -> None:
    clock = _FakeClock(datetime(2026, 7, 19, tzinfo=UTC))
    queue = InMemoryTaskQueue(clock=clock)
    queue.enqueue(_task("one"))
    claimed = queue.claim("worker-a", lease_duration_seconds=5)
    assert claimed is not None and claimed.lease_id is not None

    clock.advance(4)
    assert queue.heartbeat(claimed.lease_id, lease_duration_seconds=10)
    clock.advance(9)
    queue.complete(claimed.lease_id, b"result")

    assert queue.get_status(TaskId("one")) is TaskStatus.COMPLETED
    assert queue.get_result(TaskId("one")) == b"result"
    snapshot = queue.inspect(TaskId("one"))
    assert snapshot.lease_id is None
    assert snapshot.failure_reason is None


def test_failure_retries_with_exponential_backoff_then_dead_letters() -> None:
    clock = _FakeClock(datetime(2026, 7, 19, tzinfo=UTC))
    queue = InMemoryTaskQueue(clock=clock, retry_backoff_seconds=2)
    queue.enqueue(_task("retry", max_retries=2))

    first = queue.claim("worker-a", lease_duration_seconds=10)
    assert first is not None and first.lease_id is not None
    queue.fail(first.lease_id, "provider timeout")
    snapshot = queue.inspect(TaskId("retry"))
    assert snapshot.status is TaskStatus.PENDING
    assert snapshot.retry_count == 1
    assert snapshot.available_at == clock.now + timedelta(seconds=2)
    assert queue.claim("worker-b", lease_duration_seconds=10) is None

    clock.advance(2)
    second = queue.claim("worker-b", lease_duration_seconds=10)
    assert second is not None and second.lease_id is not None
    queue.fail(second.lease_id, "provider timeout")
    snapshot = queue.inspect(TaskId("retry"))
    assert snapshot.status is TaskStatus.PENDING
    assert snapshot.retry_count == 2
    assert snapshot.available_at == clock.now + timedelta(seconds=4)

    clock.advance(4)
    third = queue.claim("worker-c", lease_duration_seconds=10)
    assert third is not None and third.lease_id is not None
    queue.fail(third.lease_id, "permanent failure")
    assert queue.get_status(TaskId("retry")) is TaskStatus.DEAD_LETTER
    dead_letters = queue.list_dead_letters()
    assert len(dead_letters) == 1
    assert dead_letters[0].failure_reason == "permanent failure"
    assert dead_letters[0].retry_count == 3


def test_lease_expiry_requeues_and_stale_lease_is_rejected() -> None:
    clock = _FakeClock(datetime(2026, 7, 19, tzinfo=UTC))
    queue = InMemoryTaskQueue(clock=clock, retry_backoff_seconds=0)
    queue.enqueue(_task("expired", max_retries=1))
    claimed = queue.claim("worker-a", lease_duration_seconds=5)
    assert claimed is not None and claimed.lease_id is not None

    clock.advance(5)
    assert queue.sweep_expired() == 1
    assert queue.heartbeat(claimed.lease_id) is False
    assert queue.get_status(TaskId("expired")) is TaskStatus.PENDING
    with pytest.raises(TaskQueueError) as exc_info:
        queue.complete(claimed.lease_id, b"late")
    assert exc_info.value.code is TaskQueueErrorCode.LEASE_EXPIRED

    reassigned = queue.claim("worker-b", lease_duration_seconds=5)
    assert reassigned is not None
    assert reassigned.lease_id != claimed.lease_id


def test_duplicate_capacity_and_invalid_requests_are_machine_readable() -> None:
    clock = _FakeClock(datetime(2026, 7, 19, tzinfo=UTC))
    queue = InMemoryTaskQueue(clock=clock, max_size=1)
    task = _task("one")
    queue.enqueue(task)
    with pytest.raises(TaskQueueError) as duplicate:
        queue.enqueue(task)
    assert duplicate.value.code is TaskQueueErrorCode.DUPLICATE_TASK
    with pytest.raises(TaskQueueError) as full:
        queue.enqueue(_task("two"))
    assert full.value.code is TaskQueueErrorCode.QUEUE_FULL
    with pytest.raises(TaskQueueError) as invalid:
        queue.claim("", lease_duration_seconds=1)
    assert invalid.value.code is TaskQueueErrorCode.INVALID_REQUEST
