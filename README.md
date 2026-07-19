# Robata

Robata is the executable contract foundation for the native six-camera video QA and
physical-action event pipeline described in
[Architecture Design V1.1](ARCHITECTURE_DESIGN_V1.md).

Project and architecture inputs are intentionally split by authority:

- [Large-Scale Agent Execution Specification](large_scale_6camera_video_agent_execution_spec.md)
  supplies product goals, fixed functional direction, experiments, and reporting intent.
- [Execution Specification V1 Overlay](docs/architecture/execution-spec-v1-overlay.md)
  normalizes those goals into requirement IDs, units, open decisions, phase mapping, and
  executable acceptance without changing the source instruction.
- [ADR 0002](docs/adr/0002-execution-spec-integration.md) defines the authority split and
  immutable provider-neutral MP4 derived-artifact boundary.
- [ADR 0003](docs/adr/0003-artifact-registry-and-schema-evolution.md) freezes the V1
  wires and defines exact schema pinning, V2 artifact lineage, registry publication,
  and registry-backed materialized views.
- [ADR 0004](docs/adr/0004-run-independent-logical-nodes.md) defines run-independent
  logical-node identity, immutable processing-run memberships, and atomic local attach
  semantics.
- [ADR 0005](docs/adr/0005-immutable-revisions-and-current-selection.md) defines
  node-scoped immutable revisions, append-only selection decisions, atomic current
  selection, and deterministic projection rebuild.
- [ADR 0006](docs/adr/0006-throughput-track-local-parallel-inference.md) records the opt-in
  deterministic T1 inference experiment, [ADR 0007](docs/adr/0007-provider-neutral-task-queue-scaffold.md)
  records the provider-neutral T2 queue scaffold, and [ADR 0008](docs/adr/0008-throughput-track-local-camera-parallelism.md)
  records opt-in camera export/materialization parallelism and benchmark accounting.
- [ADR 0001](docs/adr/0001-executable-baseline.md), Architecture V1.1 Section 25, and
  registered schemas govern the executable security, identity, time, status, and wire
  contracts.
- [Architecture Governance](ARCHITECTURE_GOVERNANCE.md) defines non-normative throughput
  optimization tracks T1/T2; it does not redefine Architecture V1.1 phases or close capacity gates.
- [Architecture Governance Implementation](ARCHITECTURE_GOVERNANCE_IMPLEMENTATION.md) provides
  concrete T1/T2 implementation steps and evidence requirements.
- [Architecture Governance Tasks](ARCHITECTURE_GOVERNANCE_TASKS.md) tracks the local fake slice
  separately from the unpromoted T1/T2 backlog.
- [ADR 0006](docs/adr/0006-throughput-track-local-parallel-inference.md) records the opt-in
  deterministic parallel inference decision.
- [Governance evaluation](reports/architecture-governance-evaluation-2026-07-19.md) records
  the phase taxonomy, unit, interface, and gate review.
- [Task queue scaffold runbook](docs/operations/task-queue-scaffold.md) documents the local T2
  contract and its no-network scope.
- [Worker and observability runbook](docs/operations/worker-and-observability.md) documents the
  provider-neutral worker, local metrics/logging, and resource-observation contracts.

The current baseline is deliberately contract-first. It provides strict domain values,
canonical six-camera collections, a 16-document exact schema catalog, authoritative JSON
Schema validation, deterministic RFC 8785/SHA-256 helpers, a local immutable artifact
 registry, and local logical-node, run-membership, immutable-revision, selection-decision,
 and current-selection primitives. The local development mainline now also runs the
 complete MCAP -> six-camera video -> materialized PNG package -> QA/proposal/action/boundary
 fusion path with a deterministic fake model. It does not claim production ingestion,
 real-model quality, provider integration, or measured capacity.

The V1 video-export wire remains frozen and readable. New local publication uses the V2
`CameraVideoExportManifest`, exact schema references, registered raw/mapping/config/video/
timestamp/manifest artifacts, typed lineage, and a registry-backed 13-file materialized
view. The media adapter performs direct H.264-to-MP4 `REMUX`; a local result uses
`execution_mode = LOCAL_DEVELOPMENT_OVERRIDE`, `ready_manifest_id = null`,
`mapping_profile.approved = false`, and `alignment_status = UNVERIFIED`; it publishes no
source READY manifest and makes no Qwen/GPT request. Exact local evidence is recorded in
[the V2 artifact-registry report](reports/local-artifact-registry-v2-2026-07-18.md). The
[earlier six-camera export report](reports/local-six-camera-video-export-2026-07-18.md)
records the frozen V1 exercise.

The local V2 export closes its artifact-registry gap, and the generic logical-node slice
proves replay under distinct run IDs while preserving both memberships. Exact
logical-node evidence is recorded in
[the run-membership report](reports/local-logical-node-membership-2026-07-18.md). ADR 0005
adds the generic immutable-revision, append-only selection-decision, atomic current-
selection, and deterministic rebuild primitive; its implementation boundary and final
local verification are recorded in
[the revision/selection report](reports/local-immutable-revision-selection-2026-07-18.md).
Concrete producer identity/revision admission, business eligibility and selection policy,
Phase 0, and Phase 1B source/time admission remain open.

## Requirements

- Python `>=3.12,<3.14`
- [uv](https://docs.astral.sh/uv/)

## Local six-camera export

The checked-in mapping is observed and unapproved, so the local override must be explicit:

```powershell
uv run python scripts/export_camera_videos.py data/source/sample-medium.mcap tmp/local-six-camera-export --allow-unapproved-profile --registry-root tmp/local-artifact-registry
```

The registry is publication authority; the destination is a reconstructible materialized
view. Repeating the same logical derivation with another absent output directory and the
same `--registry-root` verifies the committed DAG and blobs, skips media export, and
materializes the same 13 exact files. An existing exact view may be reused only after
strict verification. Any registry, lineage, blob, or view mismatch fails closed. Without
`--registry-root`, the CLI uses `.robata-artifacts` beside the output directory.

## Local end-to-end fake-model mainline

The complete local vertical slice is executable without a real vision provider. It performs
mapping authorization, real MCAP inspection, registry-backed six-camera H.264 remux, strict
timestamp/PTS verification, selected-frame PNG materialization, coarse and dense QA,
proposal, action evidence, boundary refinement, deterministic fusion, and atomic publication
of one development-only action event:

```powershell
uv run --locked python scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-full-mainline `
  --allow-unapproved `
  --registry-root tmp/local-mainline-registry
```

The command writes `video/` (the exact 13-file V2 view), `analysis/` (frames, two QA
aggregates, five request/outcome pairs, candidate, event, report, and bundle), and the
root-level `execution-manifest.json` / `execution-audit.ndjson` evidence only after both
branches succeed. The manifest records exact hashes for every other published regular file;
the audit is canonical NDJSON and contains no source paths, credentials, or raw frames.
`production_eligible` is always `false` in this mode and `provider_requests` must remain `0`.
Run the offline checks first with:

```powershell
uv run --locked python scripts/preflight_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-full-mainline `
  --allow-unapproved
```

After publication, verify the complete output root without model/provider access:

```powershell
uv run --locked python scripts/verify_local_mainline.py `
  tmp/local-full-mainline
```

For a local T1 experiment only, opt into concurrent `QA_DENSE` and `ACTION_EVIDENCE` calls:

```powershell
uv run --locked python scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-mainline-t1-parallel `
  --allow-unapproved `
  --parallel-independent-inference `
  --registry-root tmp/local-mainline-t1-registry
```

The flag is disabled by default, requires an adapter capability declaration, preserves canonical
request ordering, and does not constitute a throughput or production-readiness claim.

A real model can replace the `VisionModelAdapter` at the application port; no real provider
adapter or model-quality claim is included in this slice. The completed local sample evidence
is recorded in [`reports/local-full-mainline-fake-model-2026-07-19.md`](reports/local-full-mainline-fake-model-2026-07-19.md),
with the final non-provider verification pass summarized in [`reports/local-mainline-final-verification-2026-07-19.md`](reports/local-mainline-final-verification-2026-07-19.md). The operating procedures are in [`docs/operations/local-mainline-runbook.md`](docs/operations/local-mainline-runbook.md).

## Requirements workstreams (provider-free completion)

The production requirements in [`REQUIREMENTS.md`](REQUIREMENTS.md) are now represented by
provider-neutral local components; real model/provider wiring remains intentionally out of scope:

- `robata.qa`: canonical 21-issue taxonomy, immutable `ClipMark` intervals, and deterministic
  `pass`/`warning`/`fail` policy. Only `fail` requests deletion; all warning clips are retained.
- `robata.frame_cache`: filesystem content-addressed frame cache with per-video `feed_once`
  coordination. Annotation consumes the manifest produced by QA instead of decoding again.
- `robata.annotation`: annotation-principal port, deterministic fake principal, structured labels
  (`verb`, `noun`, `attributes`, `location`, `hand`), and fail exclusion/warning propagation.
- `robata.search`: zero-GPU structured-label clip index, 48+ verb-family normalization aliases,
  natural-language/facet parsing, and direct `start`/`end` playback targets.
- `robata.capacity`: T+1/T+3 SLA deadlines, 500 recording-hours/day target accounting, and
  explicit 2×H100/7B assumption reporting. Local fake measurements never set
  `production_eligible=true`.

Run the offline acceptance check (no network, credentials, or provider SDKs):

```powershell
.\.venv\Scripts\python.exe scripts\verify_requirements.py --output reports\requirements-acceptance-2026-07-19.json
```

The JSON report deliberately includes `provider_requests=0`,
`execution_mode=LOCAL_DEVELOPMENT_FAKE_MODEL`, and `production_eligible=false`.

## Full development

The development group includes the optional MCAP adapter toolchain so every source module
and acceptance test can be checked:

```powershell
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
uv run python scripts/verify_schema_registry.py
```

For a minimal runtime without development or MCAP adapter packages:

```powershell
uv sync --no-dev
```

Install the MCAP adapter extra explicitly in a non-development environment only when
source inspection or decoder probing is required:

```powershell
uv sync --extra mcap --no-dev
```

Nanosecond fields are signed 64-bit Python integers in domain models. On JSON boundaries
they accept and emit only canonical base-10 strings, preserving values beyond the
IEEE-754 safe integer range.

Real MCAP samples and extracted source corpora are local-only and are not committed under
`data/source/`. Small derived documentation or explicitly curated synthetic fixtures may be
kept separately when they contain no source media.

### T1 local parallel controls

The local CLI keeps serial execution as the default. The following flags are opt-in engineering
experiments only; they do not enable a real provider or change `production_eligible=false`:

```powershell
uv run --locked python scripts/run_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-mainline-t1-all-parallel `
  --allow-unapproved `
  --parallel-video-export `
  --parallel-frame-materialization `
  --parallel-independent-inference `
  --registry-root tmp/local-mainline-t1-all-parallel-registry
```

Use `scripts/benchmark_local_mainline.py` for dependency-free local timing evidence. Its report
contains both recording-hours/wall-hour and camera-video-hours/wall-hour, but remains
`measurement_status=NOT_MEASURED` until a governed corpus and normative benchmark evidence are
approved:

```powershell
uv run --locked python scripts/benchmark_local_mainline.py `
  data/source/sample-medium.mcap `
  tmp/local-mainline-benchmark `
  --allow-unapproved `
  --iterations 1
```

### Local workstream validation commands

The engineering table is executable offline with the following checks:

```powershell
.\.venv\Scripts\python.exe scripts\validate_qa_sample.py
.\.venv\Scripts\python.exe scripts\validate_search_mvp.py
.\.venv\Scripts\python.exe scripts\stress_frame_cache.py
.\.venv\Scripts\python.exe scripts\validate_worker_integration.py
.\.venv\Scripts\python.exe scripts\benchmark_synthetic.py
.\.venv\Scripts\python.exe scripts\calibrate_capacity.py
.\.venv\Scripts\python.exe scripts\probe_process_pool.py
```

`robata.adapters.local_vision_model.TransformersVisionModelAdapter` is a lazy, local-only
boundary.  It requires a caller-supplied runner that emits a validated
`VisionInferenceOutcome`; `load_local(..., local_files_only=True)` never downloads a checkpoint.
`robata.runtime.process_pool_poc` reports ProcessPool support and PNG byte stability as
non-certifying evidence.  See [`docs/operations/local-validation-workstreams.md`](docs/operations/local-validation-workstreams.md)
for the guardrails and expected flags.

For one aggregate local evidence run (including the sample MCAP inspection), use:

```powershell
.\.venv\Scripts\python.exe scripts\run_local_workstreams.py
```
