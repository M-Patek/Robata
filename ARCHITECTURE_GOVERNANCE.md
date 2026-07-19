# Architecture Governance: Throughput Optimization Roadmap

**Status:** Proposed  
**Version:** 1.0  
**Date:** 2026-07-19  
**Scope:** Architecture governance for achieving 500 hours/day throughput target  
**Authority:** Architecture V1.1, ADRs 0001-0005, Execution Spec V1

---

## 1. Executive Summary

This document defines the architecture governance framework for transforming Robata from its current single-threaded local execution into a high-throughput distributed pipeline capable of processing **500 recording hours per day** (3,000 camera-video hours).

### Current State vs. Target

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Recording hours/wall-clock hour | ~0.62 | 20.83 | **33x** |
| Camera-video hours/wall-clock hour | ~3.69 | 125.0 | **33x** |
| Architecture | Modular monolith | Distributed pipeline | Event-driven |
| Concurrency | None | Parallel per-camera | Worker pools |
| Storage | Local SQLite | Distributed | PostgreSQL + S3 |

### Governance Principles

1. **Determinism over speed** — Parallelism must not compromise reproducibility
2. **Content-addressed caching** — Leverage existing content-addressing for deduplication
3. **Phase-gated evolution** — Each phase has measurable exit criteria
4. **Backward compatibility** — Schema and contract evolution must not break existing pipelines
5. **Fail-closed by default** — Safety and correctness take precedence over throughput

---

## 2. Architecture Bottleneck Analysis

### 2.1 Data Flow Bottlenecks

```
Current (Serial)                                    Target (Parallel)
─────────────────────────────────────────────────    ─────────────────────────────────────────────────
MCAP → Export cam_01 → Export cam_02 → ...          MCAP → [Export cam_01..06] ─┐
       ↓                                               ↓                           │
       Decode cam_01 → Decode cam_02 → ...           [Decode cam_01..06] ──────┤
       ↓                                               ↓                         │
       QA Coarse → Event Proposal → ...               [QA + Event + Action] ────┼→ Fusion → Output
       ↓                                               ↓                         │
       Publish                                          Publish                  │
```

### 2.2 Resource Bottleneck Matrix

| Operation | CPU | I/O | Memory | Parallel Potential | Current Pattern |
|-----------|-----|-----|--------|-------------------|-----------------|
| H.264 remux (6 cameras) | High | Medium | Low | **Very High** | Serial loop |
| Frame decode (6 cameras) | Very High | Low | Medium | **Very High** | Serial loop |
| PNG encoding | High | Low | Medium | **High** | Per-frame create |
| SHA-256 hashing | Medium | High | Low | **Medium** | Sequential |
| Model inference | High | Network | Medium | **High** | Sequential calls |
| JSON serialization | Low | Medium | Low | **Low** | Inline |

### 2.3 Critical Path Analysis

For a typical 2-minute recording:

```
Phase                    Current Time    Optimized Time    Speedup
─────────────────────────────────────────────────────────────────
Video Export             ~66s            ~11s (parallel)   6x
Frame Materialization    ~30s            ~5s (parallel)    6x
QA Coarse Inference      ~2s             ~2s               1x
Event Proposal           ~2s             ~2s               1x
Dense Sampling           ~15s            ~5s (seek)        3x
Dense QA + Action        ~4s             ~2s (parallel)    2x
Fusion + Publish         ~1s             ~1s               1x
─────────────────────────────────────────────────────────────────
Total                    ~120s           ~28s              **4.3x**
```

> Note: These are estimates. Actual benchmarks must be measured per Section 6.

---

## 3. Target Architecture (Phase 2)

### 3.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
│         (FastAPI/gRPC API, CLI, Message Queue Consumers)       │
├─────────────────────────────────────────────────────────────────┤
│                    Scheduling Layer                             │
│    (Workflow orchestration, priority queues, backpressure,      │
│     resource quotas, lease management)                          │
├─────────────────────────────────────────────────────────────────┤
│                    Application Layer                            │
│         (MainlinePipeline, VideoExportService,                  │
│          ArtifactView, RegisteredVideoExport)                  │
├─────────────────────────────────────────────────────────────────┤
│                    Processing Layer                             │
│    (FrameDecoderPool, InferenceWorkerPool, StreamAggregator)   │
├─────────────────────────────────────────────────────────────────┤
│                    Adapter Layer                                │
│    (PyAV, PostgreSQL Registry, S3 Storage,                      │
│     Qwen/GPT Adapter, Fake Model Adapter)                      │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                         │
│    (Redis Cache, Kafka/RabbitMQ, Prometheus/Grafana,            │
│     Distributed Tracing, Object Storage)                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Specifications

#### 3.2.1 Scheduling Layer

**Responsibilities:**
- Task queue management with priority
- Worker lease and heartbeat tracking
- Backpressure signaling
- Resource quota enforcement
- Retry and dead-letter handling

**Interface:**

```python
from typing import Protocol, AsyncIterator
from collections.abc import Awaitable

class WorkScheduler(Protocol):
    """High-throughput workflow scheduler boundary."""

    async def enqueue(self, task: PipelineTask) -> TaskId:
        """Enqueue a task, return immediately with task ID."""
        ...

    async def claim_work(self, worker_id: WorkerId) -> PipelineTask | None:
        """Claim next available work item for this worker."""
        ...

    async def heartbeat(self, lease_id: LeaseId) -> LeaseStatus:
        """Renew work lease, fail if expired."""
        ...

    async def complete_work(self, lease_id: LeaseId, result: StageResult) -> None:
        """Mark work as complete with result."""
        ...

    async def stream_results(self, task_id: TaskId) -> AsyncIterator[StageResult]:
        """Stream incremental results back to caller."""
        ...

    async def backpressure(self) -> BackpressureSignal:
        """Return current system load for upstream throttling."""
        ...
```

#### 3.2.2 Processing Layer

**Frame Decoder Pool:**

```python
class FrameDecoderPool(Protocol):
    """Parallel frame decoding worker pool."""

    async def decode_camera_stream(
        self,
        source: VideoSource,
        window: TemporalWindow,
        sampling_plan: SamplingPlan,
    ) -> AsyncIterator[DecodedFrame]:
        """Decode frames for one camera stream."""
        ...

    async def decode_all_cameras(
        self,
        sources: dict[CameraSlot, VideoSource],
        window: TemporalWindow,
        sampling_plan: SamplingPlan,
    ) -> dict[CameraSlot, AsyncIterator[DecodedFrame]]:
        """Decode all 6 camera streams in parallel."""
        ...
```

**Inference Worker Pool:**

```python
class InferenceWorkerPool(Protocol):
    """Parallel inference worker pool."""

    async def submit_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResult]:
        """Submit batch of inference requests, return when all complete."""
        ...

    async def submit_parallel_stages(
        self,
        stage_requests: dict[PipelineStage, InferenceRequest],
    ) -> dict[PipelineStage, InferenceResult]:
        """Submit independent pipeline stages in parallel."""
        ...
```

### 3.3 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│              MCAP Recording (6 camera streams)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: INGESTION                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Validate    │  │ Extract     │  │ Register    │             │
│  │ MCAP        │  │ 6 streams   │  │ source      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: EXPORT (Parallel ×6)                 │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│  │cam_01  │ │cam_02  │ │cam_03  │ │cam_04  │ │cam_05  │...   │
│  │Export  │ │Export  │ │Export  │ │Export  │ │Export  │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
│  ProcessPoolExecutor(max_workers=6)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 3: FRAME MATERIALIZATION (Parallel ×6)  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│  │cam_01  │ │cam_02  │ │cam_03  │ │cam_04  │ │cam_05  │...   │
│  │Decode  │ │Decode  │ │Decode  │ │Decode  │ │Decode  │     │
│  │+ PNG   │ │+ PNG   │ │+ PNG   │ │+ PNG   │ │+ PNG   │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
│  ProcessPoolExecutor(max_workers=6)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 4: INFERENCE PIPELINE                   │
│                                                                  │
│   ┌─────────────┐                                               │
│   │ QA Coarse   │ ─────────────────┐                             │
│   └─────────────┘                 │                             │
│                                   ▼                             │
│   ┌─────────────┐          ┌─────────────┐                     │
│   │ Event       │◄─────────│ Dense QA    │                     │
│   │ Proposal    │          │ (parallel)  │                     │
│   └─────────────┘          └─────────────┘                     │
│                                   │                             │
│   ┌─────────────┐          ┌─────────────┐                     │
│   │ Boundary    │◄─────────│ Action      │                     │
│   │ Refinement  │          │ Evidence    │                     │
│   └─────────────┘          └─────────────┘                     │
│                                   │                             │
│                                   ▼                             │
│   ┌─────────────────────────────────────┐                     │
│   │ Fusion (requires all 6 cameras)      │                     │
│   └─────────────────────────────────────┘                     │
│                                                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 5: PUBLISH                              │
│              Atomic commit to Artifact Registry                  │
│              + Materialized view to Object Storage               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase-Gated Evolution

### 4.1 Phase 1A → Phase 1B (Current → Near-term)

**Duration:** 1-2 weeks  
**Goal:** Achieve 5-10x throughput improvement through local parallelism  
**Exit Criteria:**
- [ ] 6-camera export parallelized (ProcessPoolExecutor)
- [ ] 6-camera frame materialization parallelized
- [ ] Inference pipeline stages parallelized where independent
- [ ] Benchmarks show 5-10x improvement on representative MCAPs
- [ ] All existing tests pass (determinism preserved)
- [ ] New contract tests for parallel execution paths

**Technical Decisions:**

| Decision | Rationale |
|----------|-----------|
| Use `ProcessPoolExecutor` not `ThreadPoolExecutor` | PyAV/FFmpeg are CPU-bound; need to bypass GIL |
| Keep synchronous ports, add async wrappers | Minimize contract changes; ports remain testable |
| No shared memory between workers | Each camera processes independent files; IPC via filesystem |
| Preserve deterministic output ordering | Sort results by camera slot after parallel execution |

**Code Changes:**

```python
# src/robata/application/video_export.py
from concurrent.futures import ProcessPoolExecutor, as_completed

class ParallelSixCameraVideoExportService:
    """Parallel video export with deterministic ordering."""

    def export_local(self, request: LocalVideoExportRequest) -> PublishedVideoExport:
        # ... validation and setup ...

        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self._export_camera, request, cid, staging): cid
                for cid in CAMERA_IDS
            }

            facts = {}
            for future in as_completed(futures):
                cid = futures[future]
                facts[cid] = future.result()

        # Deterministic ordering by camera slot
        ordered_facts = tuple(facts[cid] for cid in sorted(CAMERA_IDS))
        # ...
```

### 4.2 Phase 1B → Phase 2 (Near-term → Medium-term)

**Duration:** 1-2 months  
**Goal:** Achieve 500 hours/day through distributed execution  
**Exit Criteria:**
- [ ] Task queue and scheduler implemented
- [ ] Worker pool with lease/heartbeat management
- [ ] PostgreSQL artifact registry (parallel to SQLite)
- [ ] S3/MinIO object storage for blobs
- [ ] Redis cache for intermediate results
- [ ] Kafka/RabbitMQ for inter-stage messaging
- [ ] Horizontal scaling: multiple worker nodes
- [ ] Benchmarks show 500 hours/day sustained throughput

**Technical Decisions:**

| Decision | Rationale |
|----------|-----------|
| PostgreSQL for metadata | ACID, JSON support, mature, team familiarity |
| S3/MinIO for blobs | Content-addressed, infinite scale, cost-effective |
| Redis for caching | Fast lookups, TTL support, session storage |
| Kafka for messaging | Durability, replay, backpressure, partitioning by mcap_id |
| Celery for task scheduling | Mature, Python-native, retry/dead-letter support |

### 4.3 Phase 2+ (Future)

**Duration:** 3-6 months  
**Goal:** Production-grade distributed pipeline  
**Potential Enhancements:**
- GPU-accelerated decoding (NVENC/VAAPI)
- Hardware-accelerated inference (TensorRT, ONNX Runtime)
- Multi-region deployment
- Automatic scaling based on queue depth
- Cost optimization (spot instances, batching)

---

## 5. Quality Preservation in Parallel/Distributed Context

### 5.1 Determinism Guarantees

**Challenge:** Parallel execution must produce identical results to serial execution.

**Solution:**

```python
@dataclass(frozen=True)
class DeterministicExecutionContext:
    """Context that guarantees deterministic output regardless of execution order."""

    pipeline_version: str
    config_sha256: Sha256Digest
    sampling_plan_digest: Sha256Digest
    alignment_id: str
    random_seed: int

    def seeded_rng(self) -> random.Random:
        return random.Random(self.random_seed)
```

**Rules:**
1. All randomness must use seeded RNG from context
2. Time-based operations use logical clock, not wall clock
3. Parallel results are merged by deterministic key (e.g., camera slot)
4. Content-addressed artifacts ensure idempotency

### 5.2 Time Stamp Precision

**Challenge:** Distributed workers may have clock skew.

**Solution:**

```python
@dataclass(frozen=True)
class LogicalClock:
    """Hybrid logical clock for distributed timestamp ordering."""

    physical_ns: int          # NTP-synchronized physical time
    logical_counter: int      # Monotonic counter for same-nanosecond ordering
    node_id: str              # Worker node identifier

    def __lt__(self, other: LogicalClock) -> bool:
        if self.physical_ns != other.physical_ns:
            return self.physical_ns < other.physical_ns
        if self.logical_counter != other.logical_counter:
            return self.logical_counter < other.logical_counter
        return self.node_id < other.node_id
```

### 5.3 Schema Compatibility

**Challenge:** Schema evolution in distributed, multi-version environment.

**Solution:**

| Compatibility Mode | Behavior | When to Use |
|-------------------|----------|-------------|
| `BACKWARD` | New reader reads old writer | Adding optional fields |
| `FORWARD` | Old reader reads new writer | Adding fields with defaults |
| `FULL` | Bidirectional compatible | Documentation changes only |
| `NONE` | No compatibility guaranteed | Breaking changes (requires migration) |

**Process:**
1. Register new schema version in `SchemaRegistry`
2. Deploy producer with new schema
3. Wait for all consumers to upgrade
4. Deprecate old schema version

---

## 6. Benchmarking and Capacity Planning

### 6.1 Benchmark Requirements

Per Architecture V1.1 Section 16, all performance claims must be backed by measured benchmarks.

**Required Benchmarks:**

| Benchmark | Metric | Target | Phase |
|-----------|--------|--------|-------|
| Single-recording throughput | Recording hours / wall-clock hour | >20 | 1B |
| Sustained throughput | Recording hours / wall-clock hour | >500 | 2 |
| Latency (p50) | Seconds per 2-minute recording | <30 | 1B |
| Latency (p99) | Seconds per 2-minute recording | <60 | 1B |
| Resource utilization | CPU % during processing | >70% | 1B |
| Resource utilization | Memory GB per recording | <4 | 1B |
| Error rate | Failed recordings / total | <0.1% | 2 |
| Determinism | Bit-exact output on replay | 100% | All |

### 6.2 Benchmark Infrastructure

```python
# tests/benchmark/test_throughput.py
import pytest

@pytest.mark.benchmark
@pytest.mark.parametrize("recording_duration_minutes", [2, 5, 10])
def test_recording_throughput(benchmark, recording_duration_minutes):
    """Benchmark throughput for recordings of varying duration."""
    result = benchmark(
        process_recording,
        duration_minutes=recording_duration_minutes,
    )
    assert result.recording_hours_per_wall_clock_hour > 20
```

### 6.3 Capacity Planning Model

```
Target: 500 recording hours/day
       = 500 / 24 = 20.83 recording hours/hour
       = 20.83 / 60 = 0.347 recording hours/minute

For 2-minute recordings:
  Recordings per minute = 0.347 * 60 / 2 = 10.4 recordings/minute
  Recordings per hour = 625 recordings/hour

Worker Requirements (Phase 2):
  Assume 30 seconds per recording with parallelization
  Throughput per worker = 120 recordings/hour
  Workers needed = 625 / 120 ≈ 6 workers

GPU Requirements (for model inference):
  Assume 5 inference calls per recording, 2 seconds each
  Inference time per recording = 10 seconds (parallelized)
  GPU throughput = 360 inferences/hour
  Recordings per GPU = 360 / 5 = 72 recordings/hour
  GPUs needed = 625 / 72 ≈ 9 GPUs
```

---

## 7. Backward Compatibility

### 7.1 Data Format Compatibility

| Layer | Strategy | Implementation |
|-------|----------|----------------|
| Wire format | Schema versioning | `SchemaRegistry` with version negotiation |
| Storage format | Immutable versions | Append-only, never update in place |
| API format | Semantic versioning | `v1`, `v2` in URL path |
| Configuration | Semantic projection | `LocalMainlineConfig` with projection hash |

### 7.2 Migration Path

```
Phase 1A (Current)
    │
    ├── SQLite Registry ──► PostgreSQL Registry (parallel operation)
    │
    ├── Local Filesystem ──► S3/MinIO (content-addressed, transparent)
    │
    ├── Sync Execution ──► Async Wrappers (same contracts)
    │
    └── Single Process ──► ProcessPool (local parallel)
            │
            ▼
    Phase 1B (Parallel Local)
            │
            ├── Add Task Queue (Celery/RQ)
            │
            ├── Add Worker Pool
            │
            ├── Add Distributed Registry
            │
            └── Add Object Storage
                    │
                    ▼
            Phase 2 (Distributed)
```

### 7.3 Compatibility Testing

**Golden Vector Tests:**
- Save known input/output pairs
- Run on every code change
- Verify bit-exact output

**Replay Tests:**
- Re-run historical MCAPs
- Compare output to stored baseline
- Flag any divergence

**Schema Compatibility Tests:**
- Test all schema version combinations
- Verify upgrade/downgrade paths
- Fail CI on incompatible changes

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Parallel execution non-determinism | Medium | High | Seeded RNG, logical clock, content-addressing |
| PyAV thread safety issues | High | Medium | ProcessPool (not ThreadPool), isolated contexts |
| Memory exhaustion | Medium | High | Bounded queues, streaming, backpressure |
| Schema version conflicts | Low | High | Registry pinning, compatibility tests |
| Worker node failures | Medium | Medium | Lease timeout, retry, dead-letter queue |
| Network partition | Low | High | Idempotent operations, at-least-once delivery |

### 8.2 Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cost overruns | Medium | Medium | Resource quotas, spot instances, batching |
| Data residency violations | Low | High | Phase 0 governance, region pinning |
| Provider rate limiting | Medium | Medium | Token bucket, exponential backoff |
| Storage growth | High | Medium | Lifecycle policies, archival, compression |

---

## 9. Governance Checklist

### Before Phase 1B Exit

- [ ] All Phase 1A tests pass
- [ ] Parallel benchmarks show 5-10x improvement
- [ ] Determinism tests pass (replay produces identical output)
- [ ] Contract tests cover parallel execution paths
- [ ] ADR documenting parallelization decision
- [ ] Updated IMPLEMENTATION_PLAN.md

### Before Phase 2 Exit

- [ ] Distributed benchmarks show 500 hours/day sustained
- [ ] Fault tolerance tests (worker failure, network partition)
- [ ] Security review (Phase 0 controls applied to distributed system)
- [ ] Cost analysis and optimization
- [ ] Operational runbooks (monitoring, alerting, incident response)
- [ ] ADR documenting distributed architecture decisions

---

## 10. References

- [Architecture Design V1](ARCHITECTURE_DESIGN_V1.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)
- [Execution Spec V1 Overlay](docs/architecture/execution-spec-v1-overlay.md)
- ADR 0001: Executable Baseline
- ADR 0002: Execution Spec Integration
- ADR 0003: Artifact Registry and Schema Evolution
- ADR 0004: Run-Independent Logical Nodes
- ADR 0005: Immutable Revisions and Current Selection

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Recording hour** | One hour on the physical recording timeline, regardless of camera count |
| **Camera-video hour** | One hour from one camera stream. A complete one-hour recording contributes six camera-video hours |
| **Content-addressed** | Identifying data by its hash rather than location |
| **Deterministic** | Same input always produces same output, regardless of execution order |
| **Fail-closed** | Default to rejecting/safe behavior when uncertain |
| **Semantic projection** | Canonical representation of data for hashing/identity |
| **Wall-clock hour** | One hour of real time |
