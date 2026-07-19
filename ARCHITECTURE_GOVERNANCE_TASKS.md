# Architecture Governance Implementation Tasks

**Status:** Active  
**Version:** 1.0  
**Date:** 2026-07-19  
**Scope:** Track implementation of architecture governance for throughput optimization

---

## Phase 1B: Parallel Local Execution (Week 1-2)

### Epic 1: Parallelize 6-Camera Video Export
- [ ] **Task 1.1.1**: Create `ParallelSixCameraVideoExportService` in `src/robata/adapters/parallel_video_export.py`
  - Use `ProcessPoolExecutor` with 6 workers
  - Ensure deterministic ordering by camera slot
  - Estimated effort: 1 day

- [ ] **Task 1.1.2**: Add parallel export tests in `tests/unit/test_parallel_video_export.py`
  - Verify bit-exact output matches serial execution
  - Benchmark: parallel export ≥4x faster than serial
  - Estimated effort: 1 day

### Epic 2: Parallelize Frame Materialization
- [ ] **Task 1.2.1**: Create `ParallelPyAvFrameMaterializer` in `src/robata/adapters/parallel_frame_materializer.py`
  - Use `ProcessPoolExecutor` with 6 workers
  - Ensure deterministic ordering
  - Estimated effort: 1 day

- [ ] **Task 1.2.2**: Optimize PNG encoder reuse in `src/robata/adapters/pyav_frame_materializer.py`
  - Reuse `av.CodecContext` across frames
  - Estimated effort: 0.5 days

### Epic 3: Parallelize Inference Pipeline
- [ ] **Task 1.3.1**: Identify parallelizable stages in `src/robata/application/mainline.py`
  - Dense QA and Action Evidence can run in parallel
  - Estimated effort: 0.5 days

- [ ] **Task 1.3.2**: Implement parallel stage execution
  - Use `ThreadPoolExecutor` for independent stages
  - Estimated effort: 1 day

### Epic 4: Benchmarking Infrastructure
- [ ] **Task 1.4.1**: Add `pytest-benchmark` and `pytest-timeout` to `pyproject.toml`
  - Estimated effort: 0.5 days

- [ ] **Task 1.4.2**: Create benchmark suite in `tests/benchmark/test_throughput.py`
  - Benchmark video export, frame materialization, and full pipeline
  - Estimated effort: 1 day

---

## Phase 2: Distributed Infrastructure (Week 3-8)

### Epic 5: Task Queue and Scheduler
- [ ] **Task 2.1.1**: Implement `TaskQueue` port in `src/robata/ports/task_queue.py`
  - Define `enqueue`, `claim`, `heartbeat`, `complete`, `fail` methods
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
- [ ] **Task 2.3.1**: Implement `PipelineWorker` in `src/robata/worker.py`
  - Lease/heartbeat management
  - Graceful shutdown
  - Estimated effort: 2 days

- [ ] **Task 2.3.2**: Add worker tests in `tests/unit/test_worker.py`
  - Heartbeat expiry, task retry, graceful shutdown
  - Estimated effort: 1 day

---

## Phase 2+: Production Hardening (Week 9+)

### Epic 8: Monitoring and Observability
- [ ] **Task 3.1.1**: Add Prometheus metrics
  - `pipeline.throughput`, `pipeline.latency`, `worker.tasks_completed`
  - Estimated effort: 2 days

- [ ] **Task 3.1.2**: Add structured logging with correlation IDs
  - Estimated effort: 1 day

### Epic 9: Cost Optimization
- [ ] **Task 3.2.1**: Implement spot instance support
  - Estimated effort: 2 days

- [ ] **Task 3.2.2**: Implement batching for model inference
  - Estimated effort: 2 days

---

## Backlog

- [ ] GPU-accelerated decoding (NVENC/VAAPI)
- [ ] Hardware-accelerated inference (TensorRT, ONNX Runtime)
- [ ] Multi-region deployment
- [ ] Automatic scaling based on queue depth

---

## Completed

- [x] Create `ARCHITECTURE_GOVERNANCE.md`
- [x] Create `ARCHITECTURE_GOVERNANCE_IMPLEMENTATION.md`
- [x] Initial commit to git
