"""Fail-closed placeholder for the future Redis task queue adapter.

The class remains importable so composition code has a stable target, but no
operation reports success until atomic Redis behavior is implemented.
"""

from __future__ import annotations

from typing import NoReturn

from robata.ports.task_queue import (
    LeaseId,
    PipelineTask,
    TaskId,
    TaskQueueError,
    TaskQueueErrorCode,
    TaskStatus,
)

_UNAVAILABLE_MESSAGE = "RedisTaskQueue is a non-runnable architecture skeleton"


def _raise_unavailable() -> NoReturn:
    raise TaskQueueError(
        TaskQueueErrorCode.ADAPTER_UNAVAILABLE,
        _UNAVAILABLE_MESSAGE,
    )


class RedisTaskQueue:
    """Redis-backed durable task queue.

    All operations fail with ``ADAPTER_UNAVAILABLE``.  This prevents a
    mistakenly injected placeholder from acknowledging work that was never
    persisted.
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url

    def enqueue(self, task: PipelineTask) -> TaskId:
        """Add a task to the queue, returning its caller-assigned ID.

        The task is serialized and pushed onto a Redis list or stream.
        """
        _raise_unavailable()

    def claim(self, worker_id: str, lease_duration_seconds: int) -> PipelineTask | None:
        """Claim the next eligible task and return it with lease metadata.

        Uses a Redis Lua script or transaction to atomically pop the next
        task and record the lease.
        """
        _raise_unavailable()

    def heartbeat(
        self,
        lease_id: LeaseId,
        lease_duration_seconds: int | None = None,
    ) -> bool:
        """Renew a lease; return ``False`` when the lease is unknown/expired.

        Updates the lease TTL in Redis.
        """
        _raise_unavailable()

    def complete(self, lease_id: LeaseId, result: bytes) -> None:
        """Mark the leased task complete and persist its result bytes.

        Atomically removes the lease and stores the result.
        """
        _raise_unavailable()

    def fail(self, lease_id: LeaseId, reason: str) -> None:
        """Fail a lease and schedule retry or dead-letter according to policy.

        Atomically releases the lease and updates retry counters.
        """
        _raise_unavailable()

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """Return the current lifecycle state for a task."""
        _raise_unavailable()


__all__ = [
    "RedisTaskQueue",
]
