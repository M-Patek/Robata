# Local worker, metrics, and benchmark operations

This runbook documents the provider-neutral execution scaffolds that complete the local
preparation work without connecting a real model or distributed service. All examples are
in-process and deterministic; they do not require Redis, PostgreSQL, S3, a Prometheus server,
credentials, or network access.

## Pipeline worker contract

`robata.worker.PipelineWorker` consumes the `robata.ports.task_queue.TaskQueue` port and an
injected handler:

```python
from robata.worker import PipelineWorker, WorkerConfig

worker = PipelineWorker(
    queue,
    handler=lambda task: task.payload,
    config=WorkerConfig(
        worker_id="local-worker-1",
        lease_duration_seconds=30,
        heartbeat_interval_seconds=5,
    ),
)
result = worker.run_once()
```

The handler receives an opaque `PipelineTask` and must return `bytes`. The worker:

- claims at most one task per `run_once()` call;
- renews the queue lease on a daemon heartbeat thread;
- treats a rejected/failed heartbeat as `LOST_LEASE` and never acknowledges a result after
  the lease is lost;
- delegates retry and dead-letter policy to the queue's `fail()` implementation;
- supports bounded polling via `run(max_iterations=...)` and a stop event; and
- emits optional local metrics and structured log events when the corresponding hooks are
  injected.

The checked-in `InMemoryTaskQueue` is a test/local scaffold. It is not durable across process
restart and is not a distributed atomicity implementation. Redis/PostgreSQL/S3 adapters remain
separate ADR/PoC work.

## Dependency-free observability

`robata.runtime.observability.MetricsRegistry` provides thread-safe counters, gauges, and
histogram observations. `render_prometheus()` returns deterministic text exposition for local
inspection or test assertions; it does not open an HTTP endpoint.

```python
metrics.increment("worker_tasks_completed", labels={"worker_id": "local-worker-1"})
metrics.observe("pipeline_latency_ms", 42.5, labels={"stage": "QA_DENSE"})
print(metrics.render_prometheus())
```

`new_correlation_id(seed)` is deterministic for a stable seed. Use `correlation_scope()` for
context-local propagation and `StructuredLogger.emit()` for JSON records through the standard
`logging` module. Correlation IDs are operational metadata only; raw frames and credentials are
never added by these helpers.

## Resource observations

`measure_callable_with_resources()` augments a wall-clock `ThroughputSample` with portable
process CPU time and peak `tracemalloc` bytes. The helper restores the caller's tracing state
even when the workload or timing clock raises. These observations are engineering evidence, not
capacity certification. The benchmark CLI remains `measurement_status=NOT_MEASURED` until a
governed corpus, workload definition, and normative O-01 evidence are approved.

## Failure handling checklist

1. Keep serial defaults enabled for reproducible local runs.
2. Use the in-memory queue only for contract tests or single-process development.
3. Inspect `WorkerRun.status` before treating a result as committed; `LOST_LEASE` is not a
   successful completion.
4. Preserve `provider_requests=0`, `execution_mode=LOCAL_DEVELOPMENT_FAKE_MODEL`, and
   `production_eligible=false` for the fake-model mainline.
5. Do not add provider SDKs, credentials, or network-backed adapters without a new ADR and
   explicit infrastructure approval.
