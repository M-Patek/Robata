# Architecture Governance Implementation Plan

**Status:** Draft  
**Version:** 1.0  
**Date:** 2026-07-19  
**Scope:** Concrete implementation steps for throughput optimization  
**Authority:** ARCHITECTURE_GOVERNANCE.md

---

## 1. Overview

This document translates the architecture governance framework into concrete, actionable implementation steps. Each phase has defined deliverables, acceptance criteria, and estimated effort.

### Phase Timeline

```
Week 1-2:  Phase 1B Parallel Local
Week 3-4:  Phase 1B Optimization & Benchmarking
Week 5-8:  Phase 2 Distributed Infrastructure
Week 9-12: Phase 2 Worker Pool & Scaling
Week 13+: Phase 2+ Production Hardening
```

---

## 2. Phase 1B: Parallel Local Execution

### 2.1 Epic: Parallelize 6-Camera Video Export

**Story:** As a pipeline operator, I want 6-camera video export to run in parallel so that export time is reduced by 5-6x.

**Acceptance Criteria:**
- [ ] Export time for 6-camera recording < 12 seconds (was ~66s)
- [ ] Output is bit-exact identical to serial execution
- [ ] All existing tests pass
- [ ] New contract tests cover parallel execution

**Tasks:**

#### Task 1.1.1: Create Parallel Export Service

**File:** `src/robata/adapters/parallel_video_export.py` (new)

```python
"""Parallel video export adapter using ProcessPoolExecutor."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from robata.contracts.video_export import CameraSlot, LocalVideoExportRequest
    from robata.contracts.artifacts import PublishedVideoExport


def _export_camera_worker(
    request: LocalVideoExportRequest,
    camera_id: CameraSlot,
    staging_directory: Path,
) -> tuple[CameraSlot, CameraVideoExportFacts]:
    """Worker function for parallel camera export.

    Must be a module-level function for ProcessPoolExecutor pickling.
    """
    from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter

    exporter = PyAvH264Mp4Exporter()
    facts = exporter.export(request, camera_id, staging_directory)
    return camera_id, facts


class ParallelSixCameraVideoExportService:
    """Parallel video export service with deterministic ordering."""

    def __init__(self, max_workers: int = 6) -> None:
        self._max_workers = max_workers

    def export_local(
        self,
        request: LocalVideoExportRequest,
        staging_directory: Path,
    ) -> PublishedVideoExport:
        """Export 6 cameras in parallel, return deterministically ordered results."""
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    _export_camera_worker,
                    request,
                    camera_id,
                    staging_directory,
                ): camera_id
                for camera_id in CAMERA_IDS
            }

            facts: dict[CameraSlot, CameraVideoExportFacts] = {}
            for future in as_completed(futures):
                camera_id, camera_facts = future.result()
                facts[camera_id] = camera_facts

        # Deterministic ordering by canonical camera slot
        ordered_facts = tuple(facts[cid] for cid in sorted(CAMERA_IDS))
        # ... rest of publish logic
```

**Effort:** 1 day  
**Dependencies:** None

#### Task 1.1.2: Add Parallel Export Tests

**File:** `tests/unit/test_parallel_video_export.py` (new)

```python
"""Tests for parallel video export."""

import pytest
from robata.adapters.parallel_video_export import ParallelSixCameraVideoExportService


class TestParallelVideoExport:
    """Parallel export must match serial export output exactly."""

    def test_parallel_export_matches_serial_output(self, sample_mcap: Path) -> None:
        """Parallel and serial export produce identical artifacts."""
        # ... setup request

        serial_service = SixCameraVideoExportService()
        parallel_service = ParallelSixCameraVideoExportService()

        serial_result = serial_service.export_local(request)
        parallel_result = parallel_service.export_local(request)

        # Bit-exact comparison
        assert serial_result.manifest == parallel_result.manifest
        for cid in CAMERA_IDS:
            assert serial_result.camera_sha256[cid] == parallel_result.camera_sha256[cid]

    def test_parallel_export_faster_than_serial(self, sample_mcap: Path) -> None:
        """Parallel export is at least 4x faster than serial."""
        import time

        serial_service = SixCameraVideoExportService()
        parallel_service = ParallelSixCameraVideoExportService()

        serial_start = time.perf_counter()
        serial_service.export_local(request)
        serial_time = time.perf_counter() - serial_start

        parallel_start = time.perf_counter()
        parallel_service.export_local(request)
        parallel_time = time.perf_counter() - parallel_start

        assert parallel_time < serial_time / 4
```

**Effort:** 1 day  
**Dependencies:** Task 1.1.1

### 2.2 Epic: Parallelize Frame Materialization

**Story:** As a pipeline operator, I want frame materialization to run in parallel across 6 cameras so that materialization time is reduced by 5-6x.

**Acceptance Criteria:**
- [ ] Materialization time for 6-camera recording < 5 seconds (was ~30s)
- [ ] Output PNGs are bit-exact identical to serial execution
- [ ] All existing tests pass

**Tasks:**

#### Task 1.2.1: Create Parallel Frame Materializer

**File:** `src/robata/adapters/parallel_frame_materializer.py` (new)

```python
"""Parallel frame materialization adapter."""

from concurrent.futures import ProcessPoolExecutor, as_completed


def _decode_and_render_worker(
    plan: FrameMaterializationPlan,
    staging: Path,
    max_width: int,
) -> tuple[CameraSlot, RenderedCameraFrames]:
    """Worker for parallel frame decoding and rendering."""
    from robata.adapters.pyav_frame_materializer import _decode_and_render

    rendered = _decode_and_render(plan, staging, max_width=max_width)
    return plan.ledger.record.camera_id, rendered


class ParallelPyAvFrameMaterializer(FrameMaterializer):
    """Parallel frame materializer with deterministic ordering."""

    def __init__(self, max_workers: int = 6) -> None:
        self._max_workers = max_workers

    def materialize(self, request: FrameMaterializationRequest) -> TemporalVisualPackage:
        """Materialize frames for all 6 cameras in parallel."""
        plans = self._build_plans(request)

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    _decode_and_render_worker,
                    plan,
                    staging,
                    self._max_width,
                ): plan.ledger.record.camera_id
                for plan in plans
            }

            rendered: dict[CameraSlot, RenderedCameraFrames] = {}
            for future in as_completed(futures):
                camera_id, frames = future.result()
                rendered[camera_id] = frames

        # Deterministic ordering
        ordered = tuple(rendered[cid] for cid in sorted(CAMERA_IDS))
        return TemporalVisualPackage(frames=ordered)
```

**Effort:** 1 day  
**Dependencies:** None

#### Task 1.2.2: Optimize PNG Encoder Reuse

**File:** `src/robata/adapters/pyav_frame_materializer.py` (modify)

```python
@contextmanager
def _png_encoder() -> Iterator[av.CodecContext]:
    """Reusable PNG encoder context.

    Creating a CodecContext per frame is expensive. Reuse across
    all frames in a single camera's materialization.
    """
    codec = av.CodecContext.create("png", "w")
    try:
        yield codec
    finally:
        codec.close()


def _encode_frame(
    frame: av.VideoFrame,
    codec: av.CodecContext,
) -> bytes:
    """Encode a single frame using a reusable codec context."""
    packets = codec.encode(frame)
    return b"".join(bytes(packet) for packet in packets)
```

**Effort:** 0.5 days  
**Dependencies:** None

### 2.3 Epic: Parallelize Inference Pipeline

**Story:** As a pipeline operator, I want independent inference stages to run in parallel so that overall pipeline latency is reduced.

**Acceptance Criteria:**
- [ ] Independent inference stages (QA Coarse, Event Proposal, etc.) run in parallel
- [ ] Pipeline latency reduced by 2-3x
- [ ] Deterministic ordering of results preserved

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

**Story:** As a pipeline operator, I want automated benchmarks so that throughput improvements are measured and tracked.

**Acceptance Criteria:**
- [ ] pytest-benchmark integrated
- [ ] Benchmarks run in CI
- [ ] Results published to reports/

**Tasks:**

#### Task 1.4.1: Add Benchmark Dependencies

**File:** `pyproject.toml` (modify)

```toml
[dependency-groups]
dev = [
    # ... existing deps ...
    "pytest-benchmark>=4.0,<5",
    "pytest-timeout>=2.3,<3",
]
```

#### Task 1.4.2: Create Benchmark Suite

**File:** `tests/benchmark/test_throughput.py` (new)

```python
"""Throughput benchmarks for the Robata pipeline."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.benchmark(group="video_export")
@pytest.mark.parametrize("camera_count", [1, 3, 6])
def test_video_export_throughput(
    benchmark: BenchmarkFixture,
    camera_count: int,
    sample_mcap: Path,
) -> None:
    """Benchmark video export throughput for varying camera counts."""
    service = ParallelSixCameraVideoExportService(max_workers=camera_count)

    result = benchmark(service.export_local, request)

    # Assert throughput target
    recording_seconds = 120  # 2-minute recording
    wall_seconds = result.stats.mean
    recording_hours_per_wall_hour = (recording_seconds / wall_seconds) * 3600

    assert recording_hours_per_wall_hour > 20  # Phase 1B target
```

**Effort:** 1 day  
**Dependencies:** Task 1.1.1, Task 1.2.1

---

## 3. Phase 2: Distributed Infrastructure

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

## 4. Testing Strategy

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

## 5. Monitoring and Observability

### 5.1 Metrics

| Metric | Type | Alert Threshold |
|--------|------|----------------|
| `pipeline.throughput` | Gauge | < 20 rec-hrs/wall-hr (1B), < 500 (2) |
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

| Component | Current | Phase 1B | Phase 2 |
|-----------|---------|----------|---------|
| Python | 3.12 | 3.12 | 3.12+ |
| PyAV | 14.x | 14.x | 14.x |
| Pydantic | 2.10 | 2.10 | 2.10 |
| PostgreSQL | - | - | 15+ |
| Redis | - | - | 7+ |
| Kafka | - | - | 3.5+ |
| Celery | - | - | 5.3+ |

### B. Environment Variables

| Variable | Phase 1B | Phase 2 | Description |
|----------|----------|---------|-------------|
| `ROBATA_WORKERS` | 6 | 12+ | Number of parallel workers |
| `ROBATA_REDIS_URL` | - | Required | Redis connection string |
| `ROBATA_POSTGRES_URL` | - | Required | PostgreSQL connection string |
| `ROBATA_S3_ENDPOINT` | - | Required | S3/MinIO endpoint |
| `ROBATA_QUEUE_TYPE` | memory | redis | Task queue backend |
