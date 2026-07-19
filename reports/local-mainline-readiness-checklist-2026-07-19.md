# Local Mainline Readiness Checklist — 2026-07-19

## Scope

This checklist records readiness work completed **without** a real vision model/provider. It
covers the executable local path and the evidence needed to hand off the final adapter task.

## Completed

- [x] MCAP inspection and strict six-camera mapping resolution.
- [x] Registered six-video export with timestamp sidecars and V2 manifest.
- [x] Content-addressed local artifact registry and reuse accounting.
- [x] Coarse/dense package construction and strict frame materialization.
- [x] Fake QA, event proposal, dense action evidence, boundary refinement, and deterministic
      fusion.
- [x] Connected run report/mainline bundle with complete camera evidence lineage.
- [x] Atomic top-level publication and cleanup of failed staging trees.
- [x] Explicit no-event path with candidate-dependent stages marked `SKIPPED`.
- [x] Deterministic execution manifest with exact hashes for every published regular artifact.
- [x] Canonical append-only local audit NDJSON without source paths, credentials, or raw frames.
- [x] Offline preflight for runtime, imports, mapping, source, output, registry, and spec hash.
- [x] Replay/verification commands documented in the local runbook.
- [x] Full locked pytest/static/schema verification performed for the local slice.

## Not completed by design

- [ ] Real model/provider adapter and provider SDK integration.
- [ ] Provider credentials, network policy, retries, rate limits, and cost accounting.
- [ ] Real-model quality/latency/capacity evaluation.
- [ ] Production mapping/alignment approval and governed promotion.
- [ ] Phase 0, O-03, O-04, O-10, and Phase 1B production gates.

## Handoff invariant

Until every unchecked item above has an approved record, local outputs remain development
artifacts: `provider_requests = 0`, `production_eligible = false`, and no claim of production
readiness is valid.
