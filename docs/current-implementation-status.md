# Current Implementation Status

- Status date: 2026-07-20
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
| Primary inference boundary | Provider-neutral request, capability, intent, terminal-attempt, output-validation, selection-decision logical keys, and strict `InferenceInputPlan` contracts are executable with injected adapters. Exact-pinned v1 schemas now cover intent, terminal, selection, typed raw artifact, parsed claim, and selected output; enriched-output v2 carries exact selection proof while frozen v1 remains readable. One append-only local SQLite ledger preserves pre-parse bytes and this typed evidence graph across fresh instances. | The legacy fake mainline is not rewired to this boundary; there is no real Qwen adapter, governed capability evidence, credential path, production attempt/artifact store, or production recovery decision. |
| Canonical offline vertical slice | A resolved V2 admission context is connected through an explicit fresh/resumed processing-run context, root window, materialized package set, exact input plan/catalog, independently retried call parts, an all-terminal barrier, selected raw/parsed/enriched lineage per part, deterministic reduction, explicit abstained/incomplete outcomes, local output admission, and recording-scoped fenced identity/outbox assignment. Every admitted stage is attached to the run through the SQLite-backed logical-node membership registry. Fresh runs retain distinct membership histories while locator-only package/prompt/schema changes converge on the same reusable nodes and selected terminal evidence; a fresh ledger/adapter/pipeline instance can reopen the SQLite inference graph without provider redispatch. Membership failure stops before event identity/outbox publication. | This starts after admission and supports only fixture-backed `FUSION_ADJUDICATION`. Processing-run/work lifecycle records, barriers, output decisions, and run results remain in-process, so this is conformance evidence rather than a complete durable execution path. `PRODUCTION_ADMITTED` names a local versioned output-policy result, not production readiness. |
| Shadow and evaluation | Deterministic random/hard-case routing, explicit budget outcomes, primary isolation, paired comparison, and append-only disagreement evidence are executable locally. | GPT/provider governance, quota, retention, and Phase 8 predecessors remain blocked. |
| QA | Cross-camera suspicion reduction and six-camera recording aggregation preserve provenance and reject promotional policy claims. | Coarse/dense model execution and O-10 thresholds remain unresolved. |
| Event processing | Local boundary refinement, eight-stage fusion evidence, ambiguity handling, versioned adjudication, and recording-scoped generation/fence identity assignment are executable. Canonical ordering, replay, cross-recording isolation, and an atomic local outbox are tested in-memory and with a restart-safe SQLite repository. | The SQLite adapter is local evidence, not the O-14 production database/outbox publisher. Production fusion/resolver policy, full merge/split behavior, and revision publication require O-11/O-12 decisions and real task evidence. |
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
- Remaining Section 25.7 persistence for work messages, output decisions, and run results, plus a
  production artifact store. The local inference ledger now carries exact schema quartets for
  intent, terminal, selection, typed raw metadata, parsed claims, selected outputs, and enriched
  outputs; its pre-parse byte blob is subordinate storage rather than a separately claimed wire
  contract.
- Durable inference barrier/recovery storage; local multi-part dispatch and deterministic
  reduction are implemented only with in-memory orchestration state.
- Durable processing-run/work ledgers, deadline/fence/recovery state, and registered run-result
  persistence. The canonical runner now accepts a strict fresh/resumed processing-run context and
  durably attaches its logical derivations through `ProcessingRunNodeMembership`, but its run
  lifecycle record itself remains an in-process, unregistered conformance contract.
- ActionEvent immutable revision/current-selection publication and an outbox consumer/publisher;
  the canonical slice stops at stable event identity assignment and committed local outbox rows.
- Rewiring the legacy fake mainline to the canonical ingestion -> alignment -> package set ->
  input-plan -> inference -> QA/event chain. The separate post-admission offline slice does not
  make that CLI canonical.
- Durable raw-MCAP admission and publication through the registered V2 validation/READY/alignment
  evidence. Frozen V1 evidence remains insufficient and cannot be promoted as V2.

SQLAlchemy/Redis/provider adapters are not silently emulated as production infrastructure.
Missing optional dependencies fail closed. No implementation may turn a local fake score,
empty cohort, synthetic replay, or archived report into `MEASURED` or promotional evidence.

## Current Interpretation

The repository now has a connected post-admission, single- and multi-part offline skeleton for one
fusion task, including explicit processing runs and run-to-node membership composition. That is
sufficient to continue deep development along the authoritative blueprint without waiting for a
real model. It is not the raw-MCAP-to-production durable skeleton: the legacy fake-model mainline
remains isolated, and durable run/work/barrier/output-decision state, revision publication, and
outbox delivery are not yet composed. The local restartable inference ledger closes one evidence
slice but does not select the O-14 production database or recovery topology. Work behind an open
decision or governance gate must stop at an explicit port, policy input, or fail-closed state.

The reusable package-set, input-plan, and inference identities now exclude exact artifact locators
and manifests while preserving them in audit validation. Locally persisted derived hashes created
by the former locator-polluted formula are incompatible evidence and must be rebuilt; no registered
wire shape or version changed.

No Architecture V1.1 phase is declared complete by this document. Real-model integration is a
later gated adapter task, not a prerequisite for continuing contract, orchestration, replay,
benchmark, and local state-machine development.

## Verification Commands

The live baseline is checked with the commands in the repository README. Results are meaningful
only for the exact worktree and environment in which they were run. Quality, capacity, and SLO
measurements remain `NOT_MEASURED` until governed corpora and registered benchmark inputs exist.
