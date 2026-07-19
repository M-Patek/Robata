# ADR 0008: Opt-In Local Camera Export Parallelism and Benchmark Accounting

- Status: Accepted for local fake-model experimentation only
- Date: 2026-07-19
- Governing authority: Architecture V1.1, ADR 0001, and the local mainline contracts
- Scope: non-normative throughput track T1; does not redefine Architecture V1.1 Phase 1B/2

## Context

The six camera exports are independent after mapping authorization and source inspection. The
serial service already verifies every generated MP4 and timestamp sidecar before constructing one
canonical manifest. Throughput work needs an opt-in path that preserves those checks and does not
silently change the default execution mode.

## Decision

1. `SixCameraVideoExportService` accepts `max_parallel_exports`, defaulting to `1` (serial).
2. Values above one use a bounded local `ThreadPoolExecutor`; each camera writes unique files in
a private staging directory. The service waits for results in canonical `CAMERA_IDS` order, so
manifest and artifact ordering are independent of completion order.
3. `RegisteredSixCameraVideoExportService` forwards the same bound. The CLI exposes
`--parallel-video-export` and `--max-video-export-workers`; the flags are opt-in and local-only.
4. No broker, durable queue, provider SDK, credential, or network dependency is introduced.
5. `runtime.benchmark.run_repeated` and `scripts/benchmark_local_mainline.py` report both
recording-hours/wall-hour and camera-video-hours/wall-hour. Reports are explicitly
`NOT_MEASURED`, set `provider_requests=0`, and set `production_eligible=false` until a governed
corpus and normative benchmark evidence are approved.

## Consequences

- Serial behavior and artifact identity remain the compatibility baseline.
- Parallel export can overlap independent PyAV work on local hosts, but this ADR makes no speedup
claim and does not authorize ProcessPool or distributed deployment.
- A failed worker propagates through the atomic service; the private staging tree is removed and
no incomplete manifest is published.
- Benchmark output is engineering evidence only; cache mode and worker settings are recorded.

## Verification

- Unit coverage compares serial and parallel manifest bytes and all six artifact bytes.
- CLI validation rejects worker counts outside 1..6.
- Full local verification must continue to pass with `provider_requests=0` and
`production_eligible=false`.
