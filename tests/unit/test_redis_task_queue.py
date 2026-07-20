from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from robata.ports.task_queue import (
    LeaseId,
    PipelineTask,
    TaskId,
    TaskQueueError,
    TaskQueueErrorCode,
)
from robata.queue.redis_adapter import RedisTaskQueue


def _task() -> PipelineTask:
    return PipelineTask(
        task_id=TaskId("task-1"),
        recording_id="recording-1",
        stage="QA_COARSE",
        payload=b"payload",
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda queue: queue.enqueue(_task()),
        lambda queue: queue.claim("worker-1", 30),
        lambda queue: queue.heartbeat(LeaseId("lease-1")),
        lambda queue: queue.complete(LeaseId("lease-1"), b"result"),
        lambda queue: queue.fail(LeaseId("lease-1"), "failure"),
        lambda queue: queue.get_status(TaskId("task-1")),
    ],
)
def test_redis_task_queue_operations_fail_closed(
    operation: Callable[[RedisTaskQueue], Any],
) -> None:
    queue = RedisTaskQueue("redis://127.0.0.1:6379/0")

    with pytest.raises(TaskQueueError) as captured:
        operation(queue)

    assert captured.value.code is TaskQueueErrorCode.ADAPTER_UNAVAILABLE
    assert str(captured.value) == "RedisTaskQueue is a non-runnable architecture skeleton"
