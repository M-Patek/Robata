from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

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
from robata.runtime.observability import MetricsRegistry
from robata.worker import PipelineWorker, WorkerConfig, WorkerRunStatus


def _task(task_id: str = "task-1", *, max_retries: int = 1) -> PipelineTask:
    return PipelineTask(
        task_id=TaskId(task_id),
        recording_id="recording-1",
        stage="LOCAL_STAGE",
        payload=b"payload",
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        max_retries=max_retries,
    )


def _worker(
    queue: InMemoryTaskQueue,
    handler: object,
    *,
    sleep: object = time.sleep,
) -> PipelineWorker:
    return PipelineWorker(
        queue,
        handler,  # type: ignore[arg-type]
        config=WorkerConfig(
            worker_id="worker-1",
            lease_duration_seconds=1,
            heartbeat_interval_seconds=0.05,
            poll_interval_seconds=0.01,
        ),
        sleep=sleep,  # type: ignore[arg-type]
    )


def test_worker_completes_task_and_persists_exact_result() -> None:
    queue = InMemoryTaskQueue()
    queue.enqueue(_task())
    worker = _worker(queue, lambda task: task.payload + b"-result")

    result = worker.run_once()

    assert result.status is WorkerRunStatus.COMPLETED
    assert result.task_id == "task-1"
    assert queue.get_status(TaskId("task-1")) is TaskStatus.COMPLETED
    assert queue.get_result(TaskId("task-1")) == b"payload-result"


def test_worker_failure_routes_through_queue_retry_contract() -> None:
    queue = InMemoryTaskQueue(retry_backoff_seconds=0)
    queue.enqueue(_task(max_retries=1))
    worker = _worker(queue, lambda _task: (_ for _ in ()).throw(RuntimeError("boom")))

    result = worker.run_once()

    assert result.status is WorkerRunStatus.FAILED
    assert result.error == "boom"
    assert queue.inspect(TaskId("task-1")).status is TaskStatus.PENDING
    assert queue.inspect(TaskId("task-1")).retry_count == 1


def test_worker_rejects_non_bytes_handler_result() -> None:
    queue = InMemoryTaskQueue(retry_backoff_seconds=0)
    queue.enqueue(_task(max_retries=0))
    worker = _worker(queue, lambda _task: "not-bytes")

    result = worker.run_once()

    assert result.status is WorkerRunStatus.FAILED
    assert "must return bytes" in (result.error or "")
    assert queue.inspect(TaskId("task-1")).status is TaskStatus.DEAD_LETTER


def test_worker_heartbeat_keeps_long_handler_lease_alive() -> None:
    queue = InMemoryTaskQueue()
    queue.enqueue(_task())

    def slow_handler(_task: PipelineTask) -> bytes:
        time.sleep(0.15)
        return b"done"

    result = _worker(queue, slow_handler).run_once()

    assert result.status is WorkerRunStatus.COMPLETED
    assert queue.get_result(TaskId("task-1")) == b"done"


def test_worker_run_stops_gracefully() -> None:
    queue = InMemoryTaskQueue()
    event = threading.Event()
    event.set()
    runs = _worker(queue, lambda _task: b"unused").run(event, max_iterations=2)

    assert runs == (runs[0],)
    assert runs[0].status is WorkerRunStatus.STOPPED


def test_worker_config_rejects_unsafe_heartbeat_interval() -> None:
    with pytest.raises(ValueError, match="shorter"):
        WorkerConfig(worker_id="worker", lease_duration_seconds=1, heartbeat_interval_seconds=1)


class _HeartbeatFailureQueue(InMemoryTaskQueue):
    def heartbeat(
        self,
        lease_id: LeaseId,
        lease_duration_seconds: int | None = None,
    ) -> bool:
        raise TaskQueueError(TaskQueueErrorCode.LEASE_EXPIRED, "heartbeat rejected")


class _FailFailureQueue(InMemoryTaskQueue):
    def fail(self, lease_id: LeaseId, reason: str) -> None:
        raise TaskQueueError(TaskQueueErrorCode.LEASE_EXPIRED, "fail rejected")


def test_worker_reports_lost_lease_when_heartbeat_is_rejected() -> None:
    queue = _HeartbeatFailureQueue()
    queue.enqueue(_task())

    def slow_handler(_task: PipelineTask) -> bytes:
        time.sleep(0.15)
        return b"done"

    result = _worker(queue, slow_handler).run_once()

    assert result.status is WorkerRunStatus.LOST_LEASE
    assert "heartbeat" in (result.error or "")
    assert queue.get_result(TaskId("task-1")) is None


def test_worker_reports_lost_lease_when_failure_acknowledgement_is_rejected() -> None:
    queue = _FailFailureQueue()
    queue.enqueue(_task(max_retries=1))

    result = _worker(queue, lambda _task: (_ for _ in ()).throw(RuntimeError("boom"))).run_once()

    assert result.status is WorkerRunStatus.LOST_LEASE
    assert result.error == "fail rejected"


def test_worker_uses_injected_sleep_between_idle_polls() -> None:
    queue = InMemoryTaskQueue()
    sleeps: list[float] = []
    worker = _worker(queue, lambda _task: b"unused", sleep=sleeps.append)

    runs = worker.run(max_iterations=2)

    assert tuple(run.status for run in runs) == (WorkerRunStatus.IDLE, WorkerRunStatus.IDLE)
    assert sleeps == [0.01, 0.01]


def test_worker_metrics_hook_records_completion() -> None:
    queue = InMemoryTaskQueue()
    queue.enqueue(_task())
    metrics = MetricsRegistry()
    worker = PipelineWorker(
        queue,
        lambda _task: b"done",
        config=WorkerConfig(
            worker_id="worker-1",
            lease_duration_seconds=1,
            heartbeat_interval_seconds=0.05,
        ),
        metrics=metrics,
    )

    assert worker.run_once().status is WorkerRunStatus.COMPLETED
    points = metrics.as_dict()
    assert {point["name"] for point in points} == {"worker_tasks_claimed", "worker_tasks_completed"}
    assert all(point["labels"] == {"worker_id": "worker-1"} for point in points)
