# Local Mainline T1 Benchmark Smoke Report

- Date: 2026-07-19
- Source: `data/source/sample-medium.mcap`
- Workload: one all-parallel fake-model run (`SHARED_REGISTRY` cache mode)
- Status: engineering evidence only (`NOT_MEASURED`)

## Invariants

```text
execution_mode = LOCAL_DEVELOPMENT_FAKE_MODEL
provider_requests = 0
production_eligible = false
```

## Observed sample

| Metric | Value |
|---|---:|
| wall time | 95,433 ms |
| recording duration | 40.83349 s |
| recording-hours/wall-hour | 0.4278759968 |
| camera-video-hours/wall-hour | 2.5672559806 |
| fake events | 1 |
| fake inference attempts | 5 |

These values are not a capacity claim. The sample uses the local fake model and a single local
machine; no governed corpus, CPU/RSS/disk instrumentation, or normative O-01 evidence is attached.
