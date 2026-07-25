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
    assert str(captured.value).startswith("Redis task queue adapter is unavailable:")


class _ScriptedRedis:
    def __init__(self, *responses: list[bytes]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, int, tuple[object, ...]]] = []

    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> list[bytes]:
        self.calls.append((script, numkeys, keys_and_args))
        return self._responses.pop(0)


def test_redis_task_queue_uses_injected_atomic_client_for_claim_and_completion() -> None:
    client = _ScriptedRedis(
        [b"ok", b"task-1"],
        [
            b"ok",
            b"task-1",
            b"recording-1",
            b"QA_COARSE",
            b"cGF5bG9hZA==",
            b"0",
            b"2026-07-25T00:00:00.000000Z",
            b"0",
            b"3",
            b"lease-00000000000000000001",
            b"worker-1",
            b"1784937630000000",
        ],
        [b"ok"],
    )
    queue = RedisTaskQueue(client=client, key_prefix="test:queue")

    assert queue.enqueue(_task()) == TaskId("task-1")
    claimed = queue.claim("worker-1", 30)
    assert claimed is not None
    assert claimed.lease_id == LeaseId("lease-00000000000000000001")
    assert claimed.payload == b"payload"
    queue.complete(claimed.lease_id, b"result")

    assert len(client.calls) == 3
    script, numkeys, arguments = client.calls[0]
    assert "redis.call('TIME')" in script
    assert numkeys == 1
    assert arguments[0:4] == ("test:queue", "enqueue", "1000000", "86400")
    assert client.calls[1][2][1] == "claim"
    assert client.calls[2][2][1] == "complete"


def test_redis_task_queue_maps_atomic_script_errors_to_port_errors() -> None:
    queue = RedisTaskQueue(client=_ScriptedRedis([b"error", b"DUPLICATE_TASK"]))

    with pytest.raises(TaskQueueError) as captured:
        queue.enqueue(_task())

    assert captured.value.code is TaskQueueErrorCode.DUPLICATE_TASK
