"""Provider-neutral local pipeline worker over the TaskQueue port.

The worker intentionally knows nothing about Redis, Celery, model SDKs, or network clients. It
executes opaque task payloads through an injected handler and keeps lease/heartbeat/retry/DLQ
semantics at the queue boundary.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from robata.ports.task_queue import (
    LeaseId,
    PipelineTask,
    TaskQueue,
    TaskQueueError,
)
from robata.runtime.observability import MetricsRegistry, StructuredLogger, new_correlation_id


class TaskHandler(Protocol):
    """Opaque provider-neutral task handler."""

    def __call__(self, task: PipelineTask) -> bytes:
        """Return result bytes or raise to trigger retry/dead-letter handling."""


class WorkerRunStatus(StrEnum):
    """Outcome of one polling attempt."""

    IDLE = "IDLE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    LOST_LEASE = "LOST_LEASE"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class WorkerRun:
    """Immutable accounting for one worker polling attempt."""

    status: WorkerRunStatus
    task_id: str | None = None
    lease_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Bounded worker timing and polling configuration."""

    worker_id: str
    lease_duration_seconds: int = 30
    heartbeat_interval_seconds: float = 5.0
    poll_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if isinstance(self.lease_duration_seconds, bool) or not isinstance(
            self.lease_duration_seconds, int
        ):
            raise TypeError("lease_duration_seconds must be an integer")
        if self.lease_duration_seconds <= 0:
            raise ValueError("lease_duration_seconds must be positive")
        for field_name, value in (
            ("heartbeat_interval_seconds", self.heartbeat_interval_seconds),
            ("poll_interval_seconds", self.poll_interval_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.heartbeat_interval_seconds >= self.lease_duration_seconds:
            raise ValueError("heartbeat interval must be shorter than lease duration")


class PipelineWorker:
    """Execute provider-neutral tasks with lease heartbeat and graceful shutdown."""

    def __init__(
        self,
        queue: TaskQueue,
        handler: TaskHandler,
        *,
        config: WorkerConfig,
        sleep: Callable[[float], None] = time.sleep,
        metrics: MetricsRegistry | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._queue = queue
        self._handler = handler
        self._config = config
        self._sleep = sleep
        self._metrics = metrics
        self._logger = logger

    @property
    def worker_id(self) -> str:
        return self._config.worker_id

    def run_once(self, *, stop_event: threading.Event | None = None) -> WorkerRun:
        """Claim and execute at most one task."""

        if stop_event is not None and stop_event.is_set():
            return WorkerRun(WorkerRunStatus.STOPPED)
        task = self._queue.claim(
            self._config.worker_id,
            self._config.lease_duration_seconds,
        )
        if task is None:
            self._increment("worker_tasks_idle")
            return WorkerRun(WorkerRunStatus.IDLE)
        self._increment("worker_tasks_claimed")
        lease_id = task.lease_id
        if lease_id is None:
            self._increment("worker_tasks_lost_lease")
            return WorkerRun(
                WorkerRunStatus.LOST_LEASE,
                task_id=str(task.task_id),
                error="queue returned claimed task without lease",
            )

        heartbeat_stop = threading.Event()
        heartbeat_lost = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(lease_id, heartbeat_stop, heartbeat_lost),
            name=f"robata-heartbeat-{self._config.worker_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            result = self._handler(task)
            if not isinstance(result, bytes):
                raise TypeError("task handler must return bytes")
            if heartbeat_lost.is_set():
                self._increment("worker_tasks_lost_lease")
                self._emit(
                    "worker.lease_lost",
                    task,
                    lease_id,
                    "lease heartbeat was lost before completion",
                )
                return WorkerRun(
                    WorkerRunStatus.LOST_LEASE,
                    task_id=str(task.task_id),
                    lease_id=str(lease_id),
                    error="lease heartbeat was lost before completion",
                )
            self._queue.complete(lease_id, result)
            self._increment("worker_tasks_completed")
            self._emit("worker.task_completed", task, lease_id)
            return WorkerRun(
                WorkerRunStatus.COMPLETED,
                task_id=str(task.task_id),
                lease_id=str(lease_id),
            )
        except Exception as error:
            if heartbeat_lost.is_set():
                self._increment("worker_tasks_lost_lease")
                self._emit("worker.lease_lost", task, lease_id, str(error))
                return WorkerRun(
                    WorkerRunStatus.LOST_LEASE,
                    task_id=str(task.task_id),
                    lease_id=str(lease_id),
                    error=str(error),
                )
            try:
                self._queue.fail(lease_id, str(error))
            except TaskQueueError as queue_error:
                self._increment("worker_tasks_lost_lease")
                self._emit("worker.lease_lost", task, lease_id, str(queue_error))
                return WorkerRun(
                    WorkerRunStatus.LOST_LEASE,
                    task_id=str(task.task_id),
                    lease_id=str(lease_id),
                    error=str(queue_error),
                )
            self._increment("worker_tasks_failed")
            self._emit("worker.task_failed", task, lease_id, str(error))
            return WorkerRun(
                WorkerRunStatus.FAILED,
                task_id=str(task.task_id),
                lease_id=str(lease_id),
                error=str(error),
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=max(0.1, self._config.heartbeat_interval_seconds * 2))

    def run(
        self, stop_event: threading.Event | None = None, *, max_iterations: int | None = None
    ) -> tuple[WorkerRun, ...]:
        """Poll until stopped or an optional iteration bound is reached."""

        event = stop_event or threading.Event()
        if max_iterations is not None:
            if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
                raise TypeError("max_iterations must be an integer or None")
            if max_iterations < 0:
                raise ValueError("max_iterations must be non-negative")
        runs: list[WorkerRun] = []
        while not event.is_set() and (max_iterations is None or len(runs) < max_iterations):
            result = self.run_once(stop_event=event)
            runs.append(result)
            if result.status is WorkerRunStatus.IDLE:
                # Keep the timing primitive injectable for deterministic tests and local
                # harnesses. The loop checks ``event`` again after sleeping, so a stop
                # request is observed before the next claim even when the injected sleeper
                # cannot be interrupted.
                self._sleep(self._config.poll_interval_seconds)
        if event.is_set() and (not runs or runs[-1].status is not WorkerRunStatus.STOPPED):
            runs.append(WorkerRun(WorkerRunStatus.STOPPED))
        return tuple(runs)

    def _increment(self, name: str) -> None:
        if self._metrics is not None:
            self._metrics.increment(name, labels={"worker_id": self._config.worker_id})

    def _emit(
        self,
        event: str,
        task: PipelineTask,
        lease_id: LeaseId,
        error: str | None = None,
    ) -> None:
        if self._logger is None:
            return
        fields: dict[str, object] = {
            "task_id": str(task.task_id),
            "stage": task.stage,
            "worker_id": self._config.worker_id,
        }
        if error is not None:
            fields["error"] = error
        self._logger.emit(
            logging.INFO,
            event,
            correlation_id=new_correlation_id(f"{self._config.worker_id}:{task.task_id}"),
            fields=fields,
        )

    def _heartbeat_loop(
        self,
        lease_id: LeaseId,
        stop_event: threading.Event,
        lost_event: threading.Event,
    ) -> None:
        while not stop_event.wait(self._config.heartbeat_interval_seconds):
            try:
                renewed = self._queue.heartbeat(
                    lease_id,
                    self._config.lease_duration_seconds,
                )
            except TaskQueueError:
                renewed = False
            except Exception:
                # A broken adapter must not terminate the daemon heartbeat thread silently;
                # surface it through the same lost-lease path as an explicit ``False``.
                renewed = False
            if not renewed:
                lost_event.set()
                return


__all__ = ["PipelineWorker", "TaskHandler", "WorkerConfig", "WorkerRun", "WorkerRunStatus"]
