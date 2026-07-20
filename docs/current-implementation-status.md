# Current Implementation Status

- Status date: 2026-07-19
- Scope: live repository outside `archive/old_mvp`
- Evidence class: local development unless a row explicitly says otherwise

## Authority Boundary

The product execution specification owns intent and experiment priorities. Architecture V1.1
Section 25 owns normative security, identity, time, status, provider-trust, and dependency
rules. Accepted ADRs and registered schemas refine those rules. `IMPLEMENTATION_PLAN.md`
orders the work but cannot promote a phase without its required evidence.

The archive is intentionally outside this authority chain. Archived reports may explain prior
work, but their commands, paths, test totals, and conclusions do not describe the live tree
unless independently reproduced.

## Live Slices

| Slice | Live state | Promotion boundary |
|---|---|---|
| Contract and schema kernel | Strict values, canonical JSON/SHA-256, six-camera maps, schema registry, logical nodes, immutable revisions, and current-selection primitives are executable locally. | Phase 1A remains open until the complete gate is reviewed and evidenced. |
| Ingestion and alignment | Deterministic six-stream indexing, immutable validation/READY state transitions, exact rational transforms, reset segments, residuals, and admissibility gates are implemented behind injected ports. | No governed corpus, O-03, O-04, or Phase 0 approval; local results cannot claim Phase 1B. |
| Media and local mainline | Current CLIs expose mapping authorization, inspection/export, a legacy fake-model smoke path, and offline verification. The fake path is isolated from the canonical ingestion/alignment and package-to-plan chain. | Source access must be explicitly authorized. Local override is non-promotional and sends no provider traffic. |
| Sampling and package planning | Exact rational grid selection, deterministic frame tie/dedup behavior, and materialized provider-neutral frame-budget package sets are implemented and locally verified. | O-13 still owns promoted rates/budgets. Provider-limit splitting belongs to `InferenceInputPlan`, not package mutation. |
| Primary inference boundary | Provider-neutral request, capability, intent, terminal-attempt, output-validation, selection, exact-pinned provider-claim/enriched schemas, raw-before-parse artifacts, and strict `InferenceInputPlan` contracts are executable with injected adapters and local ledgers. | The legacy fake mainline is not rewired to this boundary; there is no real Qwen adapter, governed capability evidence, credential path, or production attempt/artifact store. |
| Canonical offline vertical slice | A resolved V2 admission context is connected through root window, materialized package set, exact input plan/catalog, single-part barrier and selected attempt, raw bytes, strict claims, enrichment, local output admission, and recording-scoped fenced identity/outbox assignment. Exact replay is idempotent and the adapter is network-incapable. | This starts after admission, is fixture-backed and in-memory, supports only single-part `FUSION_ADJUDICATION`, and is conformance evidence rather than a durable execution path. `PRODUCTION_ADMITTED` names a local versioned output-policy result, not production readiness. |
| Shadow and evaluation | Deterministic random/hard-case routing, explicit budget outcomes, primary isolation, paired comparison, and append-only disagreement evidence are executable locally. | GPT/provider governance, quota, retention, and Phase 8 predecessors remain blocked. |
| QA | Cross-camera suspicion reduction and six-camera recording aggregation preserve provenance and reject promotional policy claims. | Coarse/dense model execution and O-10 thresholds remain unresolved. |
| Event processing | Local boundary refinement, eight-stage fusion evidence, ambiguity handling, versioned adjudication, and recording-scoped generation/fence identity assignment are executable. Canonical ordering, replay, cross-recording isolation, and a local outbox are tested. | Durable transactional identity/outbox storage, production fusion/resolver policy, merge/split lineage, and full revision publication require O-11/O-12 decisions and real task evidence. |
| Queueing | In-memory barrier aggregation and weighted-fair local admission/reservation are deterministic and tested. | Redis durability, atomic leases, DLQ/outbox, and production concurrency depend on O-14. |
| Retrieval | Append-only event revisions/current selections, structured filters, lexical ranking, optional fail-closed reranking, and clip/provenance registration are local. | Production index/storage, clip service, ontology, and retrieval SLO remain open. |
| Benchmarking | Offline QA/event/boundary/calibration metrics, grouped leakage-safe splits, clustered local statistics, and fail-closed evidence-context promotion gates are executable. | Governed grouped splits, frozen ground truth, O-16 thresholds/power, and capacity evidence remain open; unbound local inputs stay `NOT_MEASURED`. |

## Explicit Blockers

The following are deliberately skipped rather than replaced with convenient defaults:

- Phase 0 security, privacy, retention, audit, residency, and provider-data-governance approval.
- O-03 source topics/schemas/camera roles/codecs and O-04 clock/reset/alignment policy.
- O-06/O-07 real Qwen/GPT deployment, limits, cost, data handling, and shadow approval.
- O-10/O-11/O-12 QA taxonomy thresholds, action ontology, candidate identity, merge/split,
  boundary tolerance, calibration, and review policy.
- O-13 promoted sampling rates, padding, adaptive triggers, and frame budgets.
- O-14 production database, broker, object store, vector index, isolation, and recovery.
- O-16 governed ground truth, annotator agreement, statistical power, and numeric promotion
  thresholds.
- Complete Section 25.7 schema quartets and durable ledgers/stores for raw responses, parsed
  claims, enrichment, output decisions, and run results. The provider/enriched schema pair is
  registered, but the full persistence surface is not.
- Multi-part inference dispatch/reduction and a durable barrier/recovery implementation.
- Rewiring the legacy fake mainline to the canonical ingestion -> alignment -> package set ->
  input-plan -> inference -> QA/event chain. The separate post-admission offline slice does not
  make that CLI canonical.
- Durable raw-MCAP admission and publication through the registered V2 validation/READY/alignment
  evidence. Frozen V1 evidence remains insufficient and cannot be promoted as V2.

SQLAlchemy/Redis/provider adapters are not silently emulated as production infrastructure.
Missing optional dependencies fail closed. No implementation may turn a local fake score,
empty cohort, synthetic replay, or archived report into `MEASURED` or promotional evidence.

## Current Interpretation

The repository now has a connected post-admission, single-part offline skeleton for one fusion
task, in addition to its other executable contract slices. That is sufficient to continue deep
development along the authoritative blueprint without waiting for a real model. It is not the
raw-MCAP-to-production durable skeleton: the legacy fake-model mainline remains isolated, and
work behind an open decision or governance gate must stop at an explicit port, policy input, or
fail-closed state.

No Architecture V1.1 phase is declared complete by this document. Real-model integration is a
later gated adapter task, not a prerequisite for continuing contract, orchestration, replay,
benchmark, and local state-machine development.

## Verification Commands

The live baseline is checked with the commands in the repository README. Results are meaningful
only for the exact worktree and environment in which they were run. Quality, capacity, and SLO
measurements remain `NOT_MEASURED` until governed corpora and registered benchmark inputs exist.
