# ADR 0008: Opt-In Local Camera Export/Materialization Parallelism and Benchmark Accounting

- Status: Accepted for generic local media-service behavior; legacy runner surfaces retired
  on 2026-07-20
- Date: 2026-07-19
- Governing authority: Architecture V1.1, ADR 0001, and the local mainline contracts
- Scope: non-normative throughput track T1; does not redefine Architecture V1.1 Phase 1B/2

The bounded export/materialization service behavior remains live. References below to the removed
fake-model analysis CLI and its benchmark command are retained as historical decision context, not
as supported entry points.

## Context

The six camera exports are independent after mapping authorization and source inspection. The
serial service already verifies every generated MP4 and timestamp sidecar before constructing one
canonical manifest. Frame decode/render plans are likewise independent after sampling. Throughput
work needs opt-in paths that preserve those checks and do not silently change the default execution
mode.

## Decision

1. `SixCameraVideoExportService` accepts `max_parallel_exports`, defaulting to `1` (serial).
2. Values above one use a bounded local `ThreadPoolExecutor`; each camera writes unique files in
a private staging directory. The service waits for results in canonical `CAMERA_IDS` order, so
manifest and artifact ordering are independent of completion order.
3. `RegisteredSixCameraVideoExportService` forwards the same bound. The retired fake-mainline CLI
   exposed `--parallel-video-export` and `--max-video-export-workers`; those flags are historical
   and are not part of the live media CLI.
4. `PyAvFrameMaterializer` accepts `max_parallel_cameras`, defaulting to `1`; the opt-in
`ParallelPyAvFrameMaterializer` uses isolated per-camera staging directories and canonical camera
merge order before content hashing/publication. The retired fake-mainline CLI exposed
`--parallel-frame-materialization` and `--max-frame-materialization-workers`; no live CLI exposes
those controls.
5. No broker, durable queue, provider SDK, credential, or network dependency is introduced.
6. `runtime.benchmark.run_repeated` remains available for local measurement primitives. The
   retired `scripts/benchmark_local_mainline.py` historically reported both
   recording-hours/wall-hour and camera-video-hours/wall-hour, with results explicitly marked
   `NOT_MEASURED`, `provider_requests=0`, and `production_eligible=false`.

## Consequences

- Serial behavior and artifact identity remain the compatibility baseline.
- Parallel export/materialization can overlap independent PyAV work on local hosts, but this ADR
makes no speedup claim and does not authorize ProcessPool or distributed deployment. A separate
`runtime.process_pool_poc` records Windows-spawn support and compares reusable PNG bytes with
isolated encoding; codec reuse remains opt-in until governed replay approval.
- A failed worker propagates through the atomic service; the private staging tree is removed and
no incomplete manifest is published.
- Benchmark output is engineering evidence only; cache mode and worker settings are recorded.

## Verification

- Unit coverage compares serial and parallel manifest bytes and all six artifact bytes; parallel
export failure cleanup is covered.
- Service-level validation rejects worker counts outside 1..6 for both camera controls. The
  corresponding fake-mainline CLI checks are retired historical evidence.
- Any future benchmark composition must keep local ungoverned results `NOT_MEASURED` and
  `production_eligible=false`; no live benchmark CLI or provider route is claimed here.
