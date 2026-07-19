# Task queue T2 scaffold

`robata.ports.task_queue` freezes the provider-neutral boundary for queued pipeline
work without selecting Redis, Celery, Kafka, or another infrastructure vendor.
`robata.adapters.in_memory_task_queue.InMemoryTaskQueue` is the deterministic
local implementation used by contract tests and development-only orchestration.
It deliberately provides **no durability and no network/credential traffic**.

## Contract

- `PipelineTask` carries caller-assigned `TaskId`, recording/stage metadata,
  opaque bytes payload, priority, creation time, retry budget, and (on a claim)
  queue-issued lease metadata.
- `TaskQueue.enqueue` rejects duplicate IDs and optional bounded-queue overflow.
- `TaskQueue.claim(worker_id, lease_duration_seconds)` selects highest priority,
  then earliest availability/creation, and returns the task with `lease_id`.
- `heartbeat` renews an active lease and returns `False` for unknown/expired
  leases.
- `complete` stores exact result bytes and transitions to `COMPLETED`.
- `fail` (and lease expiry) increments `retry_count`; retries use deterministic
  exponential backoff (`base * 2**(retry_count - 1)`) until the task exceeds its
  `max_retries`, then transitions to `DEAD_LETTER`.
- `inspect`, `get_result`, `sweep_expired`, and `list_dead_letters` are optional
  read-side helpers implemented by the in-memory adapter for tests and local
  diagnostics.

The adapter accepts an injectable timezone-aware clock, making lease expiry and
backoff tests deterministic. All state is process-local and is discarded on
restart. A production Redis/PostgreSQL adapter must preserve the same state
transitions atomically and add durability, worker fencing, observability,
backpressure metrics, and governed failure/retry policies before promotion.

## Example

```python
from robata.adapters import InMemoryTaskQueue
from robata.ports import PipelineTask, TaskId

queue = InMemoryTaskQueue()
task_id = queue.enqueue(
    PipelineTask(
        task_id=TaskId("window-001"),
        recording_id="recording-001",
        stage="QA_INFERENCE",
        payload=b"provider-neutral payload",
    )
)
claimed = queue.claim("worker-1", lease_duration_seconds=30)
if claimed is not None and claimed.lease_id is not None:
    queue.complete(claimed.lease_id, b"result")
assert queue.get_status(task_id).value == "COMPLETED"
```

## Scope boundary

This scaffold closes only the port/adapter contract and deterministic unit
coverage for T2 planning. It does **not** claim production readiness, durable
queue semantics, distributed worker execution, provider SDK integration, or
network access. Those remain separate governed tasks (`2.1.2`, `2.3.x`, and the
production gates listed in `ARCHITECTURE_GOVERNANCE_TASKS.md`).
