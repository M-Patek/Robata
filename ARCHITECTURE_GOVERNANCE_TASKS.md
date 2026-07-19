# Architecture Governance Implementation Tasks (Tracks T1/T2)

**Status:** Active - T1/T2 track; normative phase gates remain separate
**Version:** 1.0
**Date:** 2026-07-19
**Scope:** Track non-normative throughput optimization work without redefining Architecture V1.1 phases
**Phase note:** `T1`/`T2` below are throughput tracks. Normative Phase 0/1A/1B/2 gates and O-01/O-03/O-04/O-10 remain authoritative.

---

## Track T1: Parallel Local Execution (Week 1-2; not normative Phase 1B)

### Epic 1: Parallelize 6-Camera Video Export
- [x] **Task 1.1.1**: Add `ParallelSixCameraVideoExportService` in `src/robata/adapters/parallel_video_export.py`
  - Uses a bounded local `ThreadPoolExecutor` (serial remains the default).
  - Ensures deterministic ordering by camera slot and preserves atomic publication/source revalidation.
  - `ProcessPoolExecutor` remains a separate Windows-spawn PoC; no production claim is made.
  - Completed: 2026-07-19

- [x] **Task 1.1.2**: Add parallel export contract tests in `tests/unit/test_video_export_service.py`
  - Verifies manifest identity, artifact ordering, and per-camera bytes match serial execution.
  - Tests worker-count bounds; no ungoverned speedup threshold is asserted.
  - Completed: 2026-07-19

### Epic 2: Parallelize Frame Materialization
- [x] **Task 1.2.1**: Add opt-in `ParallelPyAvFrameMaterializer` in `src/robata/adapters/parallel_frame_materializer.py`
  - Bounded local `ThreadPoolExecutor` across independent camera decode/render plans
  - Per-camera staging directories and canonical merge order
  - ProcessPool/Windows-spawn validation remains a follow-up PoC
  - Completed: 2026-07-19

- [x] **Task 1.2.2**: Optimize PNG encoder reuse PoC in `src/robata/runtime/process_pool_poc.py`
  - Added a compatibility PoC that reuses a same-shape PyAV codec without flushing between PNG packets.
  - The PoC proves byte stability against isolated encoding; production materialization remains isolated until governed replay approval.

### Epic 3: Parallelize Inference Pipeline
- [x] **Task 1.3.1**: Identify parallelizable stages in `src/robata/application/mainline.py`
  - `QA_DENSE` and `ACTION_EVIDENCE` are independent after event proposal; `BOUNDARY_REFINEMENT` remains downstream/serial.
  - Evidence: ADR 0006 and dependency analysis in the implementation plan.

- [x] **Task 1.3.2**: Implement opt-in parallel stage execution
  - `LocalMainlineConfig.parallel_independent_inference` uses a bounded `ThreadPoolExecutor`.
  - Requires `supports_parallel_inference`; results merge in canonical task order.
  - Unit test proves serial/parallel bundle bytes and event semantics match.
  - Failure/timeout behavior remains covered by the application fail-closed contract; a governed latency benchmark is still pending.

### Epic 4: Benchmarking Infrastructure
- [ ] **Task 1.4.1**: Add optional benchmark plugins only after dependency approval
  - The current implementation intentionally remains dependency-free
  - Estimated effort: 0.5 days

- [x] **Task 1.4.2**: Add dependency-free benchmark runner
  - `src/robata/runtime/benchmark.py::run_repeated` provides warmup/iteration accounting
  - `scripts/benchmark_local_mainline.py` emits both throughput units and explicit `NOT_MEASURED` status
  - Completed: 2026-07-19

---

## Track T2: Distributed Infrastructure (Week 3-8; not normative Phase 2)

### Epic 5: Task Queue and Scheduler
- [x] **Task 2.1.1**: Implement provider-neutral `TaskQueue` port in `src/robata/ports/task_queue.py`
  - Defines `enqueue`, `claim`, `heartbeat`, `complete`, and `fail` plus stable IDs/status/error contracts
  - Added deterministic `InMemoryTaskQueue` adapter and lease/retry/dead-letter unit coverage
  - Scope is a T2 local scaffold only; durability and distributed atomicity remain pending
  - Estimated effort: 2 days

- [ ] **Task 2.1.2**: Implement `RedisTaskQueue` adapter in `src/robata/adapters/redis_task_queue.py`
  - Use Redis sorted sets for priority queue
  - Estimated effort: 2 days

### Epic 6: Distributed Artifact Registry
- [ ] **Task 2.2.1**: Implement `PostgresArtifactRegistry` in `src/robata/adapters/postgres_artifact_registry.py`
  - PostgreSQL-backed registry with content-addressed storage
  - Estimated effort: 3 days

- [ ] **Task 2.2.2**: Implement `S3BlobStorage` in `src/robata/adapters/s3_blob_storage.py`
  - S3/MinIO-backed blob storage
  - Estimated effort: 2 days

### Epic 7: Worker Pool
- [x] **Task 2.3.1**: Implement provider-neutral `PipelineWorker` in `src/robata/worker.py`
  - Lease/heartbeat management, lost-lease handling, queue-owned retry/DLQ semantics, and graceful shutdown.
  - Handler payload/result types are opaque `bytes`; no provider SDK or network dependency is imported.
  - Scope is a deterministic local worker contract, not a durable worker pool.
  - Completed: 2026-07-19

- [x] **Task 2.3.2**: Add worker tests in `tests/unit/test_worker.py`
  - Covers completion, retry/dead-letter routing, heartbeat rejection, failure acknowledgement rejection, non-byte results, and graceful shutdown.
  - Completed: 2026-07-19

---

## Track T2+: Production Hardening (Week 9+; after normative gates)

### Epic 8: Monitoring and Observability
- [x] **Task 3.1.1**: Add dependency-free metrics primitives
  - `src/robata/runtime/observability.py` provides thread-safe counters, gauges, histograms, deterministic snapshots, and Prometheus text exposition.
  - No network endpoint or Prometheus client dependency is introduced; deployment wiring remains future work.
  - Completed: 2026-07-19

- [x] **Task 3.1.2**: Add structured logging with correlation IDs
  - Stdlib logging emits deterministic JSON events and context-scoped correlation IDs.
  - Completed: 2026-07-19

### Epic 9: Cost Optimization
- [ ] **Task 3.2.1**: Implement spot instance support
  - Estimated effort: 2 days

- [ ] **Task 3.2.2**: Implement batching for model inference
  - Estimated effort: 2 days

---

## Epic 10: Local validation and calibration (completed 2026-07-19)

- [x] ProcessPool Windows-spawn probe and PNG byte-stability PoC (`runtime/process_pool_poc.py`).
- [x] Optional local model adapter boundary with lazy transformers loading (`adapters/local_vision_model.py`).
- [x] Sample-MCAP QA validation plus complete 21-issue policy matrix.
- [x] Zero-GPU search MVP validation with verb-family and faceted queries.
- [x] SharedFrameCache concurrency stress and worker QA -> annotation -> search integration.
- [x] Synthetic serial/parallel benchmark fixture with output-hash guardrail.
- [x] CapacityPlanner 1/2/4 H100 x 7B/32B assumption calibration.

## Backlog

- [ ] GPU-accelerated decoding (NVENC/VAAPI)
- [ ] Hardware-accelerated inference (TensorRT, ONNX Runtime)
- [ ] Multi-region deployment
- [ ] Automatic scaling based on queue depth

---

## Completed

### Local fake-model vertical slice (complete; separate from T1/T2)

- [x] MCAP -> six-camera V2 export -> temporal packages -> fake QA/proposal/action/boundary/fusion.
- [x] Atomic publication, execution manifest/audit, preflight, and offline verification.
- [x] Event and no-event smoke outputs with `provider_requests = 0`.
- [x] Full local verification recorded in `reports/local-mainline-readiness-checklist-2026-07-19.md`.

### Governance artifacts

- [x] Create `ARCHITECTURE_GOVERNANCE.md`
- [x] Create `ARCHITECTURE_GOVERNANCE_IMPLEMENTATION.md`
- [x] Create `ARCHITECTURE_GOVERNANCE_TASKS.md`
- [x] Evaluate phase/unit/interface conflicts (`reports/architecture-governance-evaluation-2026-07-19.md`)
- [x] Record opt-in parallel inference decision (`docs/adr/0006-throughput-track-local-parallel-inference.md`)
- [x] Add provider-neutral in-memory TaskQueue scaffold and ADR 0007 (`src/robata/ports/task_queue.py`, `src/robata/adapters/in_memory_task_queue.py`).

## Status summary

| Area | Status | Notes |
|---|---|---|
| Local fake vertical slice | COMPLETE | Development-only; provider traffic is impossible. |
| T1 export parallelism | OPT-IN IMPLEMENTED | V1/V2 publication paths use bounded local threads; serial remains default. ProcessPool requires a separate Windows PoC/benchmark. |
| T1 frame materialization parallelism | OPT-IN IMPLEMENTED | Per-camera decode/render uses bounded local threads; serial remains default. PNG byte-stability PoC is complete; production reuse remains opt-in pending governed replay. |
| T1 inference parallelism | OPT-IN IMPLEMENTED | Dense QA + Action Evidence only; serial/parallel semantic replay passes; no benchmark/default enablement. |
| T1 benchmark suite | LOCAL FIXTURE READY | Synthetic serial/parallel fixture, hash guardrail, and explicit `NOT_MEASURED` status are implemented; governed corpus certification remains pending. |
| T2 queue/registry/worker | LOCAL CONTRACT READY | Provider-neutral TaskQueue, deterministic in-memory adapter, and PipelineWorker are tested; durable Redis/PostgreSQL/S3 adapters remain open. |
| T2+ observability/API | LOCAL PRIMITIVES READY | Dependency-free metrics/logging contracts are tested; network telemetry/export wiring remains future work. |
| Normative Phase 0/1B/2 promotion | BLOCKED BY DESIGN | Requires existing governance decisions and evidence; this tracker cannot close them. |

## Traceability matrix

| Task | Code boundary | Contract/evidence | Test/benchmark | Gate status |
|---|---|---|---|---|
| 1.1 export parallelism | `SixCameraVideoExportService`, `RegisteredSixCameraVideoExportService`, `ParallelSixCameraVideoExportService` | V1/V2 manifest, sidecar/PTS/source invariants | serial/parallel manifest + six artifact byte test; governed speedup/ProcessPool evidence pending | opt-in only |
| 1.2 materialization parallelism | `PyAvFrameMaterializer`, `ParallelPyAvFrameMaterializer` | `TemporalVisualPackage`, frame hashes, package lineage | all-parallel end-to-end smoke passed; replay/ProcessPool evidence pending | opt-in only |
| 1.3 inference parallelism | `LocalMainlinePipeline._infer_dense_and_action` | `VisionInferenceRequest/Outcome`, ADR 0006 | `tests/unit/test_mainline_pipeline.py`; benchmark not started | opt-in only |
| 1.4 benchmark harness | `src/robata/runtime/benchmark.py`, `scripts/benchmark_local_mainline.py` | non-certifying timing/hash report with both units | accounting + one local smoke report; governed corpus/CI suite pending | blocked on governed corpus/O-01 |
| 2.x distributed execution | `robata.ports.task_queue`, `InMemoryTaskQueue`, `PipelineWorker`; durable adapters remain planned | task/lease/retry/DLQ contract, ADR 0007 | `tests/unit/test_task_queue.py`, `tests/unit/test_worker.py`; durable adapters pending | local scaffold only |
| 3.x observability | `robata.runtime.observability` | correlation/audit/metric contract | `tests/unit/test_observability.py` | local primitives only |
