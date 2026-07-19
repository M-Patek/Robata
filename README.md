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
- [ADR 0001](docs/adr/0001-executable-baseline.md), Architecture V1.1 Section 25, and
  registered schemas govern the executable security, identity, time, status, and wire
  contracts.
- [Architecture Governance](ARCHITECTURE_GOVERNANCE.md) defines the throughput optimization
  roadmap and governance framework for achieving 500 hours/day.
- [Architecture Governance Implementation](ARCHITECTURE_GOVERNANCE_IMPLEMENTATION.md) provides
  concrete implementation steps for the governance framework.
- [Architecture Governance Tasks](ARCHITECTURE_GOVERNANCE_TASKS.md) tracks the implementation
  progress of Phase 1B and Phase 2.

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

A real model can replace the `VisionModelAdapter` at the application port; no real provider
adapter or model-quality claim is included in this slice. The completed local sample evidence
is recorded in [`reports/local-full-mainline-fake-model-2026-07-19.md`](reports/local-full-mainline-fake-model-2026-07-19.md),
and the operating procedures are in [`docs/operations/local-mainline-runbook.md`](docs/operations/local-mainline-runbook.md).

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
