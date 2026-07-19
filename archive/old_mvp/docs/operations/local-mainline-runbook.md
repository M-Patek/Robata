# Local Mainline Runbook

This runbook operates the **local development** six-camera MCAP-to-action path. It is a
repeatable evidence-producing slice, not a production admission procedure. The only model
implementation currently wired into the mainline is the deterministic fake adapter.

## 1. Preflight (offline)

Run preflight before allocating a registry or output directory:

```powershell
uv run --locked python scripts/preflight_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-mainline-next `
  --allow-unapproved
```

Preflight performs no MCAP decode, model invocation, provider import, network call, or output
creation. It checks Python `>=3.12,<3.14`, required imports (`av`, `mcap`, `mcap_protobuf`,
`pydantic`, `jsonschema`), mapping syntax/authorization, source readability, output absence,
registry placement, and the pinned execution-spec digest. Every result contains
`provider_requests: 0`.

The checked-in `genrobot-observed-v0` mapping is intentionally unapproved. Passing
`--allow-unapproved` is a documented development override and must not be interpreted as
mapping approval.

## 2. Execute the local mainline

```powershell
uv run --locked python scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-mainline-next `
  --allow-unapproved `
  --registry-root tmp/local-mainline-next-registry
```

Optional controls:

- `--no-event` exercises the no-event path. Candidate-dependent stages are recorded as
  `SKIPPED`, and the terminal status is `PRIMARY_COMPLETE_NO_EVENTS`.
- `--coarse-rate NUM[/DEN]` and `--dense-rate NUM[/DEN]` change only local sampling config.
- `--namespace NAME` changes the run-independent recording identity namespace.
- `--parallel-independent-inference` opts into Track T1 local concurrency for `QA_DENSE` and
  `ACTION_EVIDENCE`. It requires an adapter capability declaration, preserves canonical request
  ordering, and leaves `BOUNDARY_REFINEMENT` serial.
- `--parallel-video-export` and `--max-video-export-workers N` opt into bounded local camera export.
- `--parallel-frame-materialization` and `--max-frame-materialization-workers N` opt into bounded
  per-camera decode/render. Both defaults remain serial and neither flag is a throughput or
  production-readiness claim.

The command first authorizes the mapping, then inspects the source, exports six registered
videos, materializes frames, runs the fake QA/proposal/action/boundary path, fuses evidence,
and atomically publishes one top-level output root. A failed component removes its complete
staging tree; a failed run never publishes `execution-manifest.json` or
`execution-audit.ndjson`.

### Dependency-free benchmark evidence

Run the benchmark CLI into a fresh output root:

```powershell
uv run --locked python scripts/benchmark_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-mainline-benchmark `
  --allow-unapproved `
  --iterations 1
```

The generated `benchmark-report.json` records warmups, cache mode, both throughput units, and
`provider_requests=0`. It is engineering evidence only and remains `NOT_MEASURED` until a governed
corpus and O-01 benchmark evidence are approved.

## 3. Published output layout

```text
<output>/
  video/
    cam_*.mp4
    cam_*.timestamps.jsonl
    camera-video-export-manifest.json
  analysis/
    frames/
    packages/
    inferences/
    qa-aggregates.json
    candidates.json
    action-events.json
    run-report.json
    mainline-bundle.json
  execution-manifest.json
  execution-audit.ndjson
```

`run-report.json` and `mainline-bundle.json` are the typed primary contracts. The execution
manifest inventories every other published regular file with exact byte length and SHA-256.
The manifest and audit file intentionally do not hash themselves to avoid a self-reference.

## 4. Verify evidence and replay

Verify the complete published root offline (execution evidence plus typed run, bundle, and V2
video-manifest lineage):

```powershell
uv run --locked python scripts/verify_local_mainline.py `
  tmp/local-mainline-next
```

For a lower-level manifest-only check:

```powershell
uv run --locked python -c "from pathlib import Path; from robata.runtime.execution import verify_execution_evidence; import json; print(json.dumps(verify_execution_evidence(Path('tmp/local-mainline-next')), indent=2, sort_keys=True))"
```

The T1 parallel flag is an execution strategy, not semantic configuration; serial and opt-in
parallel fake-model runs must preserve the same semantic bundle identity. Replay is deterministic
at the semantic level: use the same source bytes, mapping profile,
namespace, and rates in a new output directory. The execution semantic hash excludes wall-clock
measurements and exact hashes of volatile report files, while exact artifact hashes remain
available for integrity verification. Do not copy an old output directory over a new one; the
publisher requires an absent, non-symlink target.

## 5. Recovery and stale partials

- If the command exits before publication, inspect the JSON error's `stage` and `code`.
- A top-level partial directory is named `.<output-name>.partial-<unique>` and is removed by
  the CLI on failure. If a process is externally terminated, remove only a stale partial whose
  resolved path is a direct child of the intended output parent.
- The shared local registry may retain content-addressed artifacts after a failed run. This is
  safe to reuse; do not treat registry presence as proof that a run was published.
- Re-run into a fresh output root. Do not retry by mutating a partially published directory.
- For a no-event result, inspect the QA aggregates and the `SKIPPED` action/boundary/fusion
  stage reports before deciding whether the source should be replayed with a different sampling
  configuration.

## 6. Governance-track references

The throughput roadmap is tracked as non-normative Tracks T1/T2 to avoid colliding with the
Architecture V1.1 Phase 1B source/time and Phase 2 temporal-data-plane gates. See
`reports/architecture-governance-evaluation-2026-07-19.md`,
`ARCHITECTURE_GOVERNANCE_TASKS.md`, and ADR 0006 for the current implementation boundary.

## 7. Operational boundaries

This runbook does **not** authorize a real provider, credentials, internet access, production
capacity, quality claims, alignment approval, or event promotion. `production_eligible` remains
`false` for every fake-model event and `provider_requests` must remain zero.

The provider-neutral worker, local metrics/logging primitives, and resource benchmark observations
are documented in [`worker-and-observability.md`](worker-and-observability.md). They are local
contracts only: no Redis/PostgreSQL/S3 adapter, Prometheus endpoint, or model SDK is installed by
this project.
