# Architecture Governance Evaluation ? 2026-07-19

## Scope

This review compares:

- `ARCHITECTURE_GOVERNANCE.md`
- `ARCHITECTURE_GOVERNANCE_IMPLEMENTATION.md`
- `ARCHITECTURE_GOVERNANCE_TASKS.md`
- the current local fake-model implementation and its existing Architecture V1.1/ADR gates

The review is deliberately limited to preparation that does **not** connect a real model,
provider SDK, credentials, or network route.

## Executive decision

The documents are a useful throughput **blueprint**, but they are not yet a promotion-ready
implementation specification. The throughput roadmap must be treated as a non-normative
optimization track until the governing Architecture V1.1 Phase 0/1A/1B/2 gates and O-01/O-03/O-04/O-10
decisions are closed. To avoid redefining the normative phase numbering, this report names the
roadmap tracks **T1 (parallel local)** and **T2 (distributed execution)**.

The local fake-model vertical slice is complete and verified. The throughput track is not
complete and must not be reported as production readiness.

## Findings

| Priority | Finding | Decision |
|---|---|---|
| P0 | The roadmap calls parallel local work ?Phase 1B/Phase 2?, while ADR 0001 and Architecture V1.1 use Phase 1B for source/time admission and Phase 2 for the provider-neutral temporal data plane. | Rename roadmap work to T1/T2 and keep normative phase gates authoritative. |
| P0 | The target is expressed as 500 recording-hours/day in one place, but benchmark tables also use `>500` recording-hours per wall-clock hour. Architecture V1.1 leaves the 500/day interpretation open. | Track both recording-hours and camera-video-hours; do not close O-01 by assumption. |
| P0 | Baseline and optimized timings are presented as measured values although the readiness reports mark throughput/resource measurements `NOT_MEASURED`. | Label all current numbers as estimates until a governed corpus and benchmark report exist. |
| P1 | The implementation snippets reference symbols and ports that do not exist in the repository (for example, the proposed exporter/materializer worker signatures and a separate BlobStorage port). | Use the existing `CameraVideoExporter`, `FrameMaterializer`, `ArtifactRegistry`, and V2 publication contracts as the integration boundaries. |
| P1 | ProcessPool, Redis, PostgreSQL, S3, Kafka, and Celery are presented as decisions without a throughput ADR or measured PoC. | Keep infrastructure choices as candidates; add an ADR before T1/T2 exit. |
| P1 | The task tracker does not distinguish the completed local fake vertical slice from the unstarted throughput backlog. | Add an explicit status split and traceability fields. |
| P2 | Benchmark, coverage, Windows-spawn, replay, fault, and backpressure jobs are not wired into the project checks. | Add benchmark/negative-test infrastructure incrementally; keep results non-certifying until gates close. |

## Implemented in this change

1. Added an opt-in deterministic local inference path:
   - `LocalMainlineConfig.parallel_independent_inference`
   - `LocalMainlineConfig.max_parallel_inference_workers`
   - `DeterministicFakeVisionModelAdapter.supports_parallel_inference = True`
   - `LocalMainlinePipeline` runs `QA_DENSE` and `ACTION_EVIDENCE` concurrently only when the
     adapter declares the capability, then merges requests/outcomes in canonical task order.
   - `BOUNDARY_REFINEMENT` remains serial because it is downstream of action evidence.
   - Operational parallelism settings are excluded from the semantic projection so serial and
     parallel execution preserve run-independent output identity.
   - CLI opt-in: `--parallel-independent-inference`.
2. Added a unit test proving the opt-in path produces the same canonical inference sequence,
   outcomes, event semantics, and bundle bytes as the serial path.
3. Added dependency-free benchmark accounting primitives (`ThroughputSample`, `BenchmarkSummary`,
   `measure_callable`, and `summarize_samples`). They report both throughput units and default to
   `NOT_MEASURED`; they do not add a certifying corpus or performance claim.
4. Added a provider-neutral in-memory TaskQueue scaffold (`robata.ports.task_queue`,
   `robata.adapters.in_memory_task_queue`) with lease/retry/dead-letter/backpressure tests and ADR 0007.
5. Added this evaluation report and the throughput-track ADRs.

## Not implemented by design

- Parallel six-camera export and parallel frame materialization (require ProcessPool/Windows-spawn
  design, source-change/error cleanup, and V1/V2 integration tests).
- Durable Redis/PostgreSQL/S3 adapters, worker leases, and distributed deployment; only the
  provider-neutral in-memory queue scaffold is prepared.
- Prometheus/remote logging, provider retries/rate limits/cost accounting, or any provider SDK.
- Production promotion, throughput claims, quality claims, or closure of Phase 0/O-03/O-04/O-10.

## Required next implementation order

1. T1 baseline harness: capture serial local timings and exact artifact/bundle hashes on approved
   synthetic fixtures; report recording-hours and camera-video-hours separately.
2. T1 export/materialization experiments behind feature flags, with deterministic replay,
   Windows spawn, failure cleanup, and V1/V2 contract tests.
3. T1 benchmark report and ADR only after measurements; no target promotion from estimates.
4. T2 ports first: the task/lease/retry/dead-letter contract and in-memory failure-tested adapter
   are now prepared; next freeze worker integration and infrastructure selection before selecting
   Redis/PostgreSQL/S3 or a broker.
5. Keep the real model/provider adapter as a separate governed handoff with `provider_requests = 0`
   until approval and credentials exist.

## Evidence

- Full local test/static baseline after this change: `488 passed, 3 skipped` (recorded in the local readiness report).
- Fake-model event and no-event smoke outputs pass `scripts/verify_local_mainline.py`.
- The new parallel inference unit test passes with the existing mainline unit suite.
- End-to-end T1 fake-model smoke output: `reports/local-mainline-t1-parallel-inference-2026-07-19.md`.
- T2 queue contract details: `docs/operations/task-queue-scaffold.md`.
- Current outputs remain `LOCAL_DEVELOPMENT_FAKE_MODEL`, `provider_requests = 0`, and
  `production_eligible = false`.
