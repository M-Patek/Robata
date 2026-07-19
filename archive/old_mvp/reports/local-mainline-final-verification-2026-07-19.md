# Local Mainline Final Verification ? 2026-07-19

## Scope

This report records the final non-provider preparation pass. The real model/provider adapter is
intentionally not implemented. All commands ran against the local fake-model path and the
repository `.venv`.

## Static and test verification

```text
pytest: 515 passed, 3 skipped
ruff check .: passed
mypy src/robata: passed (73 source files)
compileall src scripts: passed (Windows ACL warnings only for pre-existing protected temp trees)
scripts/verify_schema_registry.py: passed (16 pinned schemas)
git diff --check: passed
```

## Parallel end-to-end smoke

Command:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_local_mainline.py data\source\sample-medium.mcap tmp\final-parallel-smoke-20260719 --allow-unapproved
.\.venv\Scripts\python.exe scripts\run_local_mainline.py data\source\sample-medium.mcap tmp\final-parallel-smoke-20260719 --allow-unapproved --parallel-video-export --parallel-frame-materialization --parallel-independent-inference
.\.venv\Scripts\python.exe scripts\verify_local_mainline.py tmp\final-parallel-smoke-20260719
```

Observed invariants:

```text
run_status                  = PRIMARY_COMPLETE
execution_mode              = LOCAL_DEVELOPMENT_FAKE_MODEL
fake_inference_attempt_count = 5
event_count                 = 1
provider_requests           = 0
production_eligible         = false
execution_manifest_semantic_sha256 = b08d26dd2644388db487d194311b54945b20a374edd013531ceb56ea107fd702
bundle_sha256               = c7a1f9e5bcc5d5e86debd690e77c16fcc86148fb8f272c48de128e1fa908c2a4
```

The offline verifier passed and confirmed the published artifact count (540), execution
manifest/audit integrity, event lineage, and V2 video manifest hash. A serial replay using the
same local registry produced the same execution semantic hash
(`b08d26dd2644388db487d194311b54945b20a374edd013531ceb56ea107fd702`) and the same V2 video
semantic content hash; wall-clock report fields remain observational.

## Benchmark smoke

A one-iteration all-parallel benchmark completed successfully with
`measurement_status=NOT_MEASURED`:

```text
wall time                         = 89,866 ms
recording-hours/wall-hour          = 0.4543819687
camera-video-hours/wall-hour       = 2.7262918123
provider_requests                  = 0
production_eligible                = false
```

The values are engineering observations from a local fake-model run, not capacity claims. A
governed corpus, workload definition, resource baseline, and O-01 evidence are still required
before certification.

## Requirements acceptance

The provider-free requirements chain also passed:

```text
QA pass/warning/fail       = pass / warning / fail
annotation fail exclusion = confirmed
feed-once decode attempts  = 1
zero-GPU search            = confirmed
SLA/capacity accounting    = confirmed (NOT_MEASURED)
provider_requests          = 0
execution_mode             = LOCAL_DEVELOPMENT_FAKE_MODEL
production_eligible       = false
```

The machine-readable output is [`reports/requirements-acceptance-2026-07-19.json`](requirements-acceptance-2026-07-19.json); the implementation/runbook is [`docs/operations/requirements-workstreams.md`](../docs/operations/requirements-workstreams.md).
