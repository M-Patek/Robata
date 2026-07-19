# Architecture Governance Implementation Plan

**Status:** Draft
**Version:** 1.0
**Date:** 2026-07-19
**Scope:** Concrete implementation steps for non-normative throughput tracks T1/T2
**Authority:** ARCHITECTURE_GOVERNANCE.md, Architecture V1.1, ADRs 0001-0008
**Phase note:** T1/T2 are optimization tracks. They do not redefine normative Architecture V1.1 Phase 1B/2 gates or permit production promotion.

---

## 1. Overview

This document translates the throughput roadmap into concrete, actionable implementation steps. Each track has deliverables, evidence requirements, and estimated effort. Timing and throughput values below are hypotheses until a governed benchmark corpus exists; current repository reports mark capacity as `NOT_MEASURED`.

### Track Timeline

```
Week 1-2:  Track T1 parallel-local experiment
Week 3-4:  Track T1 optimization & benchmarking
Week 5-8:  Track T2 distributed-infrastructure design/PoC
Week 9-12: Track T2 worker-pool & scaling PoC
Week 13+: Track T2+ production hardening after normative gates
```

---

## 2. Track T1: Parallel Local Execution (not normative Phase 1B)

### 2.1 Epic: Parallelize 6-Camera Video Export

**Story:** As a pipeline operator, I want independent camera exports to overlap while preserving the
serial publication contract.

**Implemented local path (2026-07-19):**

- `src/robata/application/video_export.py::SixCameraVideoExportService` accepts
  `max_parallel_exports` (default `1`). Values greater than one use a bounded local
  `ThreadPoolExecutor`; each camera writes unique files into the existing private staging tree.
- `src/robata/adapters/parallel_video_export.py::ParallelSixCameraVideoExportService` is a small
  opt-in adapter that selects the bounded path (default six workers).
- `RegisteredSixCameraVideoExportService` forwards the worker bound, and
  `scripts/run_local_mainline.py` exposes `--parallel-video-export` plus
  `--max-video-export-workers`.
- Results are collected in canonical `CAMERA_IDS` order regardless of completion order. Source
  revalidation, sidecar/MP4 verification, manifest construction, registry publication, and atomic
  cleanup are unchanged.

**Acceptance Criteria:**

- [x] Serial remains the default compatibility path.
- [x] Parallel manifest and all six MP4/sidecar bytes match serial unit evidence.
- [x] Worker count is bounded to the six camera slots and invalid values fail closed.
- [ ] Capture serial and parallel wall time on a governed workload; no speedup claim is made yet.
- [ ] Evaluate a ProcessPool design only after a Windows-spawn PoC and resource measurements.

**Evidence:** ADR 0008 and `tests/unit/test_video_export_service.py`.

### 2.2 Epic: Parallelize Frame Materialization

**Story:** As a pipeline operator, I want independent camera decode/render plans to overlap while
preserving immutable package identity.

**Implemented local path (2026-07-19):**

- `PyAvFrameMaterializer` accepts `max_parallel_cameras` (default `1`). Values greater than one
  use a bounded local `ThreadPoolExecutor` for `_decode_and_render` calls.
- `ParallelPyAvFrameMaterializer` in `src/robata/adapters/parallel_frame_materializer.py` selects
  the opt-in six-worker path.
- Every camera writes to its own staging subdirectory. Results are resolved in canonical
  `CAMERA_IDS` order before content projection, package ID, and publication.
- `scripts/run_local_mainline.py` exposes `--parallel-frame-materialization` and
  `--max-frame-materialization-workers` while keeping serial behavior as the default.

**Acceptance Criteria:**

- [x] Parallel scheduling and deterministic merge are implemented without provider dependencies.
- [x] Existing PTS/sidecar verification and atomic cleanup remain on the shared path.
- [ ] Capture serial and parallel wall time on a governed workload; no speedup claim is made yet.
- [ ] Add Windows-spawn ProcessPool PoC if thread scheduling is insufficient.
- [ ] Optimize PNG encoder reuse only after bit-exact PyAV evidence is available.

**Evidence:** `src/robata/adapters/parallel_frame_materializer.py`, ADR 0008, and existing
materializer contract tests.

### 2.3 Epic: Parallelize Inference Pipeline

**Story:** As a pipeline operator, I want independent inference stages to run in parallel so that overall pipeline latency is reduced.

**Acceptance Criteria:**
- [x] Only the proven independent pair (`QA_DENSE` and `ACTION_EVIDENCE`) runs in parallel after event proposal
- [ ] Pipeline latency is measured and reported; `2-3x` remains a candidate hypothesis
- [x] Deterministic ordering of results preserved

**Tasks:**

#### Task 1.3.1: Identify Parallelizable Stages

From `src/robata/application/mainline.py`, the inference pipeline is:

```
QA Coarse → Event Proposal → Dense QA → Action Evidence → Boundary Refinement
```

**Analysis:**
- QA Coarse → Event Proposal: **Sequential dependency** (Event Proposal needs QA results)
- Event Proposal → Dense QA: **Sequential dependency** (Dense QA needs event proposals)
- Dense QA → Action Evidence: **Can be parallel** (both need event proposals but not each other)
- Action Evidence → Boundary Refinement: **Sequential dependency**
- Dense QA + Action Evidence → Fusion: **Fusion needs both**

**Optimized flow:**

```
QA Coarse → Event Proposal ──┬──► Dense QA ────┐
                              │                 ├──► Fusion ──► Output
                              └──► Action Evidence ──► Boundary Refinement
```

#### Task 1.3.2: Implement Parallel Stage Execution

**File:** `src/robata/application/mainline.py` (modify)

```python
from concurrent.futures import ThreadPoolExecutor

# In LocalMainlinePipeline._execute():

# Stage 1: QA Coarse (must run first)
qa_request, qa_outcome = self._infer(...)

# Stage 2: Event Proposal (depends on QA)
proposal_request, proposal_outcome = self._infer(...)

# Stage 3-4: Dense QA and Action Evidence (can run in parallel)
with ThreadPoolExecutor(max_workers=2) as executor:
    dense_future = executor.submit(self._infer, ...)
    action_future = executor.submit(self._infer, ...)

    dense_qa_request, dense_qa_outcome = dense_future.result()
    action_request, action_outcome = action_future.result()

# Stage 5: Boundary Refinement (depends on Action Evidence)
boundary_request, boundary_outcome = self._infer(...)

# Stage 6: Fusion (needs all previous results)
fused = self._fuse(...)
```

**Effort:** 1 day
**Dependencies:** None

### 2.4 Epic: Benchmarking Infrastructure

The repository now contains dependency-free accounting primitives in `src/robata/runtime/benchmark.py`
and a repeatable local CLI at `scripts/benchmark_local_mainline.py`. Both throughput units are
reported. Reports remain `NOT_MEASURED` until an approved corpus, resource instrumentation, and
normative O-01 evidence exist.

**Implemented local path (2026-07-19):**

- `run_repeated` records warmups and timed iterations without interpreting results as capacity.
- `benchmark_local_mainline.py` executes the fake-model CLI in isolated output directories, records
  cache mode and parallel flags, derives recording duration from the V2 manifest, and enforces
  `provider_requests=0` / `production_eligible=false`.

**Acceptance Criteria:**

- [x] Dependency-free benchmark report emits recording-hours and camera-video-hours.
- [x] Warmups are excluded from sample summaries; cache mode is explicit.
- [ ] Optional pytest benchmark plugins and CI publication remain pending dependency approval.
- [ ] Governed corpus/resource instrumentation and certifying capacity report remain pending.

**Evidence:** `tests/unit/test_benchmark.py`, ADR 0008, and the generated
`benchmark-report.json` artifact.

#### Task 1.4.1: Optional Benchmark Plugins (pending approval)

The repository intentionally does not add `pytest-benchmark` or `pytest-timeout` yet. Adding
those dependencies requires a separate dependency review and CI policy decision. The current
runner is dependency-free.

#### Task 1.4.2: Dependency-Free Benchmark Runner (implemented)

**Files:** `src/robata/runtime/benchmark.py`, `scripts/benchmark_local_mainline.py`

The runner executes isolated local fake-model runs, records warmups/iterations, derives recording
duration from the V2 manifest, and emits both throughput units. It never asserts a capacity target.
The emitted summary remains `measurement_status=NOT_MEASURED` until a governed corpus and O-01
evidence are approved.

**Evidence:** `tests/unit/test_benchmark.py` and
`reports/local-mainline-t1-benchmark-smoke-2026-07-19.md`.

---

## 3. Track T2: Distributed Infrastructure (not normative Phase 2)

All T2 components below are design/PoC backlog items. Do not add Redis, PostgreSQL, S3,
Kafka, Celery, or other external dependencies until the corresponding port contract and
infrastructure ADR are accepted. The current local path remains offline and dependency-free.

The provider-neutral `TaskQueue` port and deterministic `InMemoryTaskQueue` scaffold are now
implemented for local contract tests (`src/robata/ports/task_queue.py`,
`src/robata/adapters/in_memory_task_queue.py`, ADR 0007). Durable queue/worker integration is
still open.

### 3.1 Epic: Task Queue and Scheduler

**Story:** As a pipeline operator, I want tasks to be queued and scheduled across multiple workers so that throughput scales horizontally.

**Acceptance Criteria:**
- [ ] Tasks are queued durably
- [ ] Workers claim tasks via lease mechanism
- [ ] Failed tasks are retried with exponential backoff
- [ ] Dead letter queue captures permanently failed tasks

**Tasks:**

#### Task 2.1.1: Implement Task Queue Port

**File:** `src/robata/ports/task_queue.py` (new)

```python
"""Task queue port for distributed pipeline execution."""

from __future__ import annotations

from typing import Protocol
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


@dataclass(frozen=True)
class TaskId:
    value: str


@dataclass(frozen=True)
class LeaseId:
    value: str


@dataclass(frozen=True)
class PipelineTask:
    """A unit of work in the pipeline."""
    task_id: TaskId
    recording_id: str
    stage: PipelineStage
    payload: bytes
    priority: int = 0
    created_at: datetime = datetime.utcnow()
    retry_count: int = 0


class TaskQueue(Protocol):
    """Port for durable task queue operations."""

    def enqueue(self, task: PipelineTask) -> TaskId:
        """Add task to queue, return task ID."""
        ...

    def claim(self, worker_id: str, lease_duration_seconds: int) -> PipelineTask | None:
        """Claim next available task with lease."""
        ...

    def heartbeat(self, lease_id: LeaseId) -> bool:
        """Renew lease, return False if expired."""
        ...

    def complete(self, lease_id: LeaseId, result: bytes) -> None:
        """Mark task as completed."""
        ...

    def fail(self, lease_id: LeaseId, reason: str) -> None:
        """Mark task as failed, trigger retry or dead-letter."""
        ...

    def get_status(self, task_id: TaskId) -> TaskStatus:
        """Get current task status."""
        ...
```

**Effort:** 2 days
**Dependencies:** None

**Local scaffold status (2026-07-19):** `src/robata/ports/task_queue.py` now
contains the frozen provider-neutral contract and stable error/status values.
`src/robata/adapters/in_memory_task_queue.py` provides a deterministic,
clock-injectable fake covering priority ordering, lease heartbeat/expiry,
exponential retry, bounded backpressure, completion, and dead-letter capture.
The claimed task carries queue-issued lease metadata so callers can safely pass
`lease_id` to heartbeat/complete/fail. This implementation is process-local
and intentionally does not provide Redis/network/credential traffic; Task 2.1.2
remains the governed production-adapter task.

#### Task 2.1.2: Implement Redis Task Queue Adapter

**File:** `src/robata/adapters/redis_task_queue.py` (new)

```python
"""Redis-backed task queue adapter."""

import redis
from robata.ports.task_queue import TaskQueue, TaskId, LeaseId, PipelineTask


class RedisTaskQueue(TaskQueue):
    """Redis-backed implementation of task queue port."""

    def __init__(self, redis_client: redis.Redis, queue_key: str = "robata:tasks") -> None:
        self._redis = redis_client
        self._queue_key = queue_key

    def enqueue(self, task: PipelineTask) -> TaskId:
        """Push task to priority-sorted queue."""
        task_json = task.to_json()
        # Use Redis sorted set with priority as score
        self._redis.zadd(self._queue_key, {task_json: task.priority})
        return task.task_id

    def claim(self, worker_id: str, lease_duration_seconds: int) -> PipelineTask | None:
        """Atomically pop highest-priority task and set lease."""
        # Use Redis Lua script for atomic pop-and-lease
        ...
```

**Effort:** 2 days
**Dependencies:** Task 2.1.1

### 3.2 Epic: Distributed Artifact Registry

**Story:** As a pipeline operator, I want the artifact registry to support distributed storage so that artifacts are durable and accessible from any worker.

**Acceptance Criteria:**
- [ ] PostgreSQL adapter implements ArtifactRegistry port
- [ ] S3/MinIO adapter implements blob storage
- [ ] Content-addressed deduplication works across workers
- [ ] Registry operations are atomic and consistent

**Tasks:**

#### Task 2.2.1: Implement PostgreSQL Artifact Registry

**File:** `src/robata/adapters/postgres_artifact_registry.py` (new)

```python
"""PostgreSQL-backed artifact registry adapter."""

from __future__ import annotations

import asyncpg
from robata.ports.artifact_registry import ArtifactRegistry


class PostgresArtifactRegistry(ArtifactRegistry):
    """PostgreSQL implementation of artifact registry.

    Uses content-addressed storage with semantic and exact-byte
    SHA-256 for deduplication.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def register(self, artifact: Artifact) -> ArtifactId:
        """Register artifact, return existing ID if duplicate."""
        async with self._pool.acquire() as conn:
            # Check for existing semantic match
            existing = await conn.fetchrow(
                "SELECT artifact_id FROM artifacts WHERE semantic_sha256 = $1",
                artifact.semantic_sha256,
            )
            if existing:
                return ArtifactId(existing["artifact_id"])

            # Insert new artifact
            ...
```

**Effort:** 3 days
**Dependencies:** None

#### Task 2.2.2: Implement S3 Blob Storage

**File:** `src/robata/adapters/s3_blob_storage.py` (new)

```python
"""S3/MinIO-backed blob storage adapter."""

import boto3
from botocore.exceptions import ClientError

from robata.ports.blob_storage import BlobStorage


class S3BlobStorage(BlobStorage):
    """S3-compatible blob storage with content-addressed keys."""

    def __init__(self, bucket: str, endpoint_url: str | None = None) -> None:
        self._s3 = boto3.client("s3", endpoint_url=endpoint_url)
        self._bucket = bucket

    def store(self, data: bytes) -> str:
        """Store blob, return content-addressed key."""
        key = f"blobs/{exact_bytes_sha256(data)}"
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return key  # Already exists
        except ClientError:
            self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)
            return key
```

**Effort:** 2 days
**Dependencies:** None

### 3.3 Epic: Worker Pool

**Story:** As a pipeline operator, I want workers to process tasks from the queue so that throughput scales with worker count.

**Acceptance Criteria:**
- [ ] Workers claim tasks, process them, and mark complete
- [ ] Workers heartbeat to prevent lease expiration
- [ ] Worker failures trigger task reassignment
- [ ] Worker count can be scaled up/down dynamically

**Tasks:**

#### Task 2.3.1: Implement Worker

**File:** `src/robata/worker.py` (new)

```python
"""Pipeline worker for distributed execution."""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass

from robata.ports.task_queue import TaskQueue, LeaseId


@dataclass
class WorkerConfig:
    """Configuration for a pipeline worker."""
    worker_id: str
    lease_duration_seconds: int = 30
    heartbeat_interval_seconds: int = 10
    max_retries: int = 3


class PipelineWorker:
    """Worker that claims and processes pipeline tasks."""

    def __init__(self, config: WorkerConfig, queue: TaskQueue) -> None:
        self._config = config
        self._queue = queue
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        """Main worker loop: claim tasks, process, repeat."""
        while not self._shutdown.is_set():
            task = self._queue.claim(
                self._config.worker_id,
                self._config.lease_duration_seconds,
            )
            if task is None:
                await asyncio.sleep(1)
                continue

            lease_id = LeaseId(f"{task.task_id.value}:{self._config.worker_id}")

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(lease_id)
            )

            try:
                result = await self._process_task(task)
                self._queue.complete(lease_id, result)
            except Exception as e:
                self._queue.fail(lease_id, str(e))
            finally:
                heartbeat_task.cancel()

    async def _heartbeat_loop(self, lease_id: LeaseId) -> None:
        """Send periodic heartbeats to keep lease alive."""
        while True:
            await asyncio.sleep(self._config.heartbeat_interval_seconds)
            if not self._queue.heartbeat(lease_id):
                raise LeaseExpiredError(lease_id)

    async def _process_task(self, task: PipelineTask) -> bytes:
        """Process a single pipeline task."""
        # Delegate to appropriate stage handler
        handler = self._get_handler(task.stage)
        return await handler(task.payload)

    def shutdown(self) -> None:
        """Signal worker to shut down gracefully."""
        self._shutdown.set()
```

**Effort:** 2 days
**Dependencies:** Task 2.1.1

---

## 4. Testing Strategy for T1/T2

### 4.1 Unit Tests

| Component | Coverage Target | Key Tests |
|-----------|----------------|-----------|
| Parallel export | 90% | Determinism, ordering, error handling |
| Parallel materialization | 90% | Bit-exact output, resource cleanup |
| Task queue | 90% | Lease expiry, retry, dead letter |
| Worker | 85% | Heartbeat, graceful shutdown, failure |

### 4.2 Integration Tests

| Scenario | Setup | Expected |
|----------|-------|----------|
| End-to-end parallel pipeline | 6-camera MCAP | <15s total, bit-exact output |
| Worker failure recovery | Kill worker mid-task | Task retried, no data loss |
| Scale-up | Add workers mid-run | Throughput increases linearly |
| Backpressure | Flood queue | Queue bounded, no OOM |

### 4.3 Benchmark Tests

```python
# tests/benchmark/test_capacity.py

@pytest.mark.benchmark(group="capacity")
@pytest.mark.parametrize("worker_count", [1, 3, 6, 12])
def test_capacity_scaling(benchmark, worker_count):
    """Verify throughput scales linearly with worker count."""
    ...
```

---

## 5. Monitoring and Observability (future T2 work)

### 5.1 Metrics

| Metric | Type | Alert Threshold |
|--------|------|----------------|
| `pipeline.throughput` | Gauge | report both units; candidate T1/T2 thresholds remain NOT_MEASURED |
| `pipeline.latency` | Histogram | p99 > 60s |
| `worker.tasks_completed` | Counter | - |
| `worker.tasks_failed` | Counter | > 1% of completed |
| `queue.depth` | Gauge | > 1000 |
| `queue.oldest_task_age` | Gauge | > 5 minutes |

### 5.2 Logging

```python
# Structured logging with correlation IDs
{
    "timestamp": "2026-07-19T10:30:00Z",
    "level": "INFO",
    "correlation_id": "abc123",
    "task_id": "task-456",
    "stage": "video_export",
    "camera_id": "cam_03",
    "message": "Export completed",
    "duration_ms": 1200,
    "output_sha256": "aabbcc...",
}
```

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| PyAV process safety | High | Medium | ProcessPool, isolated contexts | TBD |
| PostgreSQL performance | Medium | High | Connection pooling, read replicas | TBD |
| S3 eventual consistency | Medium | Medium | Content-addressed, retry reads | TBD |
| Worker lease expiry | Medium | Medium | Heartbeat, lease extension | TBD |
| Cost overrun | Medium | Medium | Resource quotas, spot instances | TBD |
| Schema migration failure | Low | High | Compatibility tests, rollback | TBD |

---

## 7. Appendix

### A. Dependency Versions

| Component | Current | Track T1 | Track T2 |
|-----------|---------|----------|---------|
| Python | 3.12 | 3.12 | 3.12+ |
| PyAV | 14.x | 14.x | 14.x |
| Pydantic | 2.10 | 2.10 | 2.10 |
| PostgreSQL | - | - | 15+ |
| Redis | - | - | 7+ |
| Kafka | - | - | 3.5+ |
| Celery | - | - | 5.3+ |

### B. Environment Variables

| Variable | Track T1 | Track T2 | Description |
|----------|----------|---------|-------------|
| `ROBATA_WORKERS` | 6 | 12+ | Number of parallel workers |
| `ROBATA_REDIS_URL` | - | Required | Redis connection string |
| `ROBATA_POSTGRES_URL` | - | Required | PostgreSQL connection string |
| `ROBATA_S3_ENDPOINT` | - | Required | S3/MinIO endpoint |
| `ROBATA_QUEUE_TYPE` | memory | redis | Task queue backend |
