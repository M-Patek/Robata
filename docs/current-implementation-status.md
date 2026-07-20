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
| Contract and schema kernel | Strict values, canonical JSON/SHA-256, six-camera maps, atomic schema registration and immutability checks, logical nodes, immutable revisions, current-selection primitives, and the registered compact primary-completion V2 contract are executable locally. Its V1 bytes remain frozen and exact-readable without an implicit upcast. | The live catalog still declares `upcasters=[]`. The compact completion contract is not a completion repository, durable transaction, or detailed-result artifact implementation. Phase 1A remains open until the complete gate is reviewed and evidenced. |
| Ingestion and alignment | Deterministic six-stream indexing, immutable validation/READY state transitions, exact rational transforms, reset segments, residuals, and admissibility gates are implemented behind injected ports. | No governed corpus, O-03, O-04, or Phase 0 approval; local results cannot claim Phase 1B. |
| Media inspection and export | Live CLIs expose explicitly authorized inspection and local video export. The former fake-model analysis runner and its verifier have been removed from the live package and remain only as archived history. | Source access must be explicitly authorized. Local override is non-promotional and sends no provider traffic. There is not yet a canonical operator CLI. |
| Sampling and package planning | Exact rational grid selection, deterministic frame tie/dedup behavior, materialized provider-neutral frame-budget package sets, and an exact frozen-adaptive resolver from immutable trigger evidence to rational grid segments or explicit integer-nanosecond targets are implemented and locally verified. | `AdaptiveSampler.sample()` still fails closed; signal detection and trigger-to-rate policy are not implemented. O-13 still owns promoted rates, padding, adaptive triggers, and budgets. Provider-limit splitting belongs to `InferenceInputPlan`, not package mutation. |
| Primary inference boundary | Provider-neutral request, capability, intent, terminal-attempt, output-validation, selection-decision logical keys, and strict `InferenceInputPlan` contracts are executable with injected adapters. Exact-pinned v1 schemas now cover intent, terminal, selection, typed raw artifact, parsed claim, and selected output; enriched-output v2 carries exact selection proof while frozen v1 remains readable. One append-only local SQLite ledger preserves pre-parse bytes and this typed evidence graph across fresh instances. | There is no canonical operator composition root, real Qwen adapter, governed capability evidence, credential path, production attempt/artifact store, or production recovery decision. |
| Canonical offline vertical slice | A resolved V2 admission context is connected through an explicit fresh/resumed processing-run context, root window, materialized package set, exact input plan/catalog, independently retried call parts, an all-terminal barrier, selected raw/parsed/enriched lineage per part, deterministic reduction, explicit abstained/incomplete outcomes, a local output decision, and immutable local event hypotheses. Every admitted stage is attached to the run through the SQLite-backed logical-node membership registry. Fresh runs retain distinct membership histories while locator-only package/prompt/schema changes converge on the same reusable nodes and selected terminal evidence; a fresh ledger/adapter/pipeline instance can reopen the SQLite inference graph without provider redispatch. The composition does not inject or call the identity registry and leaves `identity_result=None`, so it publishes no stable identity or outbox row. | This starts after admission and supports only fixture-backed `FUSION_ADJUDICATION`. Processing-run/work lifecycle records, barriers, output decisions, and run results remain in-process, so this is conformance evidence rather than a complete durable execution path. Its output decision is `ADMITTED` with `evidence_class=LOCAL_CONFORMANCE` and `production_eligible=false`; `PRODUCTION_QUALIFIED` fails closed until a governed qualification gateway exists. |
| Shadow and evaluation | Deterministic random/hard-case routing, explicit budget outcomes, primary isolation, paired comparison, and append-only disagreement evidence are executable locally. | GPT/provider governance, quota, retention, and Phase 8 predecessors remain blocked. |
| QA | Cross-camera suspicion reduction and six-camera recording aggregation preserve provenance and reject promotional policy claims. | `qa_pipeline/coarse.py` and `qa_pipeline/dense.py` remain explicit non-runnable skeletons. Coarse/dense model execution and O-10 thresholds remain unresolved. |
| Event processing | Local boundary refinement, eight-stage fusion evidence, ambiguity handling, versioned adjudication, and recording-scoped generation/fence identity assignment are executable. Canonical ordering, replay, cross-recording isolation, and an atomic local outbox are tested in-memory and with a restart-safe SQLite repository. | `event_pipeline/evidence.py` remains an explicit non-runnable extraction skeleton. The SQLite adapter is local evidence, not the O-14 production database/outbox publisher. Production fusion/resolver policy, full merge/split behavior, and revision publication require O-11/O-12 decisions and real task evidence. |
| Queueing | `queue/barrier.py` provides deterministic in-memory barrier aggregation, and weighted-fair local admission/reservation is tested. | The barrier has no durable adapter or recovery integration. Redis durability, atomic leases, DLQ/outbox, and production concurrency depend on O-14. |
| Retrieval | Append-only event revisions/current selections, structured filters, lexical ranking, optional fail-closed reranking, and clip/provenance registration are local. | Production index/storage, clip service, ontology, and retrieval SLO remain open. |
| Benchmarking | Offline QA/event/boundary/calibration metrics, grouped leakage-safe splits, clustered local statistics, and fail-closed evidence-context promotion gates are executable. | Governed grouped splits, frozen ground truth, O-16 thresholds/power, and capacity evidence remain open; unbound local inputs stay `NOT_MEASURED`. |

## Architecture Section 25 Conformance Matrix

These statuses assess every normative subsection as a whole: `COMPLETE` means all requirements in
that subsection have live implementation and local evidence; `PARTIAL` means only a stated subset
is executable; `BLOCKED` means an authority or policy gate prevents the required implementation or
promotion; and `ABSENT` means there is no runnable implementation. None of these labels is a phase
exit or production-readiness claim.

| Section | Status | Live evidence | Remaining boundary or blocker |
|---|---|---|---|
| 25.1 VLM trust boundary | PARTIAL | Provider outputs are preserved as typed raw bytes, parsed as untrusted claims, enriched against the exact input-plan catalog, and reduced only after validation in the local inference/canonical path. | Only fixture-backed inference is composed. Real Qwen capability, provider governance, credentials, artifact storage, and governed replay evidence are absent. |
| 25.2 provider-neutral packages and durable call barrier | PARTIAL | Temporal packages remain provider-neutral; provider limits and render details live in `InferenceInputPlan`; independently retried call parts and deterministic reduction are executable. `inference/call_barrier.py` binds the exact input-plan member set over the deterministic in-memory `queue/barrier.py` reference. | Neither barrier layer has a durable adapter. Lease/fence recovery, a production provider adapter, and registered persisted barrier contracts are not connected. |
| 25.3 exact and adaptive sampling | PARTIAL | Exact rational-grid selection and materialization are tested. `sampling/adaptive.py` deterministically resolves a frozen trigger artifact into reduced rational grid segments or explicit ordered integer-nanosecond targets, with a fixed semantic-projection version and target budget. | `AdaptiveSampler.sample()` is fail-closed. Signal detection and trigger-to-rate policy remain absent; O-13 owns promoted rates, padding, trigger policy, and budgets, and cross-language grid evidence is still required. |
| 25.4 logical identities and run membership | PARTIAL | Generic run-independent logical nodes, append-only memberships, typed canonical node producers, deterministic replay, and SQLite-backed membership recovery are executable. | Durable work planning/invalidation and complete producer coverage across later QA/event stages are not composed. A reusable node is not evidence that its processing run completed. |
| 25.5 immutable revisions and current selection | PARTIAL | Generic immutable node revisions, append-only selection decisions, current-selection compare-and-swap, rebuild, and local verification are implemented. | The canonical primary path does not yet publish concrete `ActionEvent` revisions or current selections inside the authoritative completion boundary. |
| 25.6 validation, READY, and separate ledgers | PARTIAL | V2 validation/READY/alignment contracts and local source/alignment ledger reconciliation exist behind injected ports and reject insufficient V1 evidence. | Governed raw-source admission, O-03/O-04 policy, production stores, and Phase 0 approval are absent. |
| 25.7 exact schema references and evolution | PARTIAL | Atomic schema registration, exact catalog/digest validation, published-schema immutability checks, synthetic upcaster fixtures, and exact-pinned V1/V2 `primary-completion-record` contracts are locally verified. The default V2 model versions its stricter RFC3339/order semantics; V1 remains frozen and readable without automatic reinterpretation. | The live catalog still has `upcasters=[]`. Detailed run-result, work-message, output-decision, and persisted barrier contracts remain unregistered; no compact completion transaction or durable repository exists. |
| 25.8 nonblocking human review | ABSENT | Generic immutable revision primitives can support later adjudication, but there is no runnable review-routing path. | Nonblocking human-review routing, annotation contracts, backlog/latency evidence, and the governing O-10/O-11/O-12 policy are absent. |
| 25.9 recording-scoped serialized identity | PARTIAL | Standalone in-memory and restart-safe SQLite registries enforce recording scope, generation/fence checks, deterministic replay, cross-recording isolation, and an atomic local identity/outbox commit. | Identity preparation/application is not integrated with canonical completion; the local adapter is not a production-qualified authority or aggregate transaction. |
| 25.10 security and phase dependencies | BLOCKED | Later-phase components are deliberately exercised only as local conformance slices. `qa_pipeline/coarse.py`, `qa_pipeline/dense.py`, and `event_pipeline/evidence.py` remain non-runnable skeletons rather than fabricated adapters. | Phase 0 is a hard gate. Required predecessor evidence and open O-decisions are unresolved, so later local code cannot promote any phase. |
| 25.11 promotion evidence | PARTIAL | Local tests cover exact schema pins, rational-grid cases, replay/idempotency, inference ordering, barrier behavior, identity concurrency, compact-completion RFC3339/order and V1-to-V2 policy migration, and adaptive projection invariants. | Governed real-source/provider replay, cross-language grid vectors, human-review backlog, production non-bypass controls, failure injection for aggregate completion, capacity/SLO evidence, and approved thresholds remain missing. |

### Named-file readiness

| File | Status | Exact executable boundary |
|---|---|---|
| `sampling/adaptive.py` | PARTIAL | The frozen-artifact resolver is implemented and tested; `AdaptiveSampler.sample()` still raises `NotImplementedError`, so adaptive signal/policy execution is absent. |
| `qa_pipeline/coarse.py` | ABSENT | The architecture type exists, but `CoarseQAPipeline.run_coarse()` raises `NotImplementedError`. |
| `qa_pipeline/dense.py` | ABSENT | The architecture type exists, but `DenseQAPipeline.run_dense()` raises `NotImplementedError`. |
| `event_pipeline/evidence.py` | ABSENT | The architecture type exists, but `ActionEvidenceExtractor.extract_evidence()` raises `NotImplementedError`. |
| `queue/barrier.py` | PARTIAL | The in-memory coordinator/storage and terminal criticality semantics are executable and tested; durable storage, recovery, and lease/fence integration are absent. |

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
- Remaining Section 25.7 contracts and persistence for work messages, output decisions, detailed
  run results, and barriers, plus a production artifact store. The compact
  `primary-completion-record` is registered and exact-pinned, but it is only a contract: no
  authoritative completion transaction or durable repository persists it. The local inference
  ledger carries exact schema quartets for intent, terminal, selection, typed raw metadata, parsed
  claims, selected outputs, and enriched outputs; its pre-parse byte blob is subordinate storage
  rather than a separately claimed wire contract. The live catalog still declares `upcasters=[]`.
- Durable inference barrier/recovery storage; local multi-part dispatch and deterministic
  reduction are implemented only with in-memory orchestration state.
- Durable processing-run/work ledgers, deadline/fence/recovery state, and detailed run-result
  schema/artifact persistence. The canonical runner now accepts a strict fresh/resumed
  processing-run context and durably attaches its logical derivations through
  `ProcessingRunNodeMembership`, but its run lifecycle and result remain in-process conformance
  contracts. The registered compact completion record does not close this persistence gap.
- ActionEvent immutable revision/current-selection publication and an outbox consumer/publisher;
  the canonical slice stops at immutable local event hypotheses before identity assignment or
  outbox publication. The restart-safe SQLite identity/outbox adapter is exercised separately.
- Durable raw-MCAP admission and publication through the registered V2 validation/READY/alignment
  evidence. Frozen V1 evidence remains insufficient and cannot be promoted as V2.

SQLAlchemy/Redis/provider adapters are not silently emulated as production infrastructure.
Missing optional dependencies fail closed. No implementation may turn a local fake score,
empty cohort, synthetic replay, or archived report into `MEASURED` or promotional evidence.

## Current Interpretation

The repository now has a connected post-admission, single- and multi-part offline skeleton for one
fusion task, including explicit processing runs and run-to-node membership composition. That is
sufficient to continue deep development along the authoritative blueprint without waiting for a
real model. It is not the raw-MCAP-to-production durable skeleton: the former fake-model runner has
been removed rather than presented as a mainline, and durable run/work/barrier/output-decision
state, revision publication, and outbox delivery are not yet composed. The local restartable
inference ledger closes one evidence slice but does not select the O-14 production database or
recovery topology. Work behind an open decision or governance gate must stop at an explicit port,
policy input, or fail-closed state.

Two recently closed contract slices do not change that boundary. Frozen adaptive trigger evidence
can now be resolved deterministically before package identity, but adaptive signal/policy execution
remains blocked by O-13. A compact primary-completion record now has an exact registered schema and
versioned semantic projection that defaults to V2 with strict RFC3339 and
completion-order semantics. Its exact V1 schema is frozen/readable, but no
implicit upcast exists. The authoritative transaction, detailed-result artifact,
and canonical completion repository described by ADR 0012 remain unimplemented.

The removed surface includes `robata.contracts.mainline`, `robata.ports.mainline`, their
smoke-only bundle/report/stage contracts, and the duplicate legacy model-adapter port. There are no
compatibility shims. Shared contracts and frame-materialization boundaries that still serve the
canonical, QA, event, and sampling code live under `robata.contracts.pipeline` and
`robata.ports.frame_materialization`.

The local output proof, output decision, and event-hypothesis payloads are unregistered V2
contracts; V1 payloads fail closed. The canonical run binding is
`canonical-offline-v2`, and an old `canonical-offline-v1` record, including one still marked
`RUNNING`, cannot resume in the V2 composition.

The reusable package-set, input-plan, and inference identities now exclude exact artifact locators
and manifests while preserving them in audit validation. Locally persisted derived hashes created
by the former locator-polluted formula are incompatible evidence and must be rebuilt; no registered
wire shape or version changed. The local unregistered output contracts changed wire version and
must also be rebuilt rather than relabeled.

No Architecture V1.1 phase is declared complete by this document. Real-model integration is a
later gated adapter task, not a prerequisite for continuing contract, orchestration, replay,
benchmark, and local state-machine development.

## Verification Commands

The live baseline is checked with the commands in the repository README. Results are meaningful
only for the exact worktree and environment in which they were run. Quality, capacity, and SLO
measurements remain `NOT_MEASURED` until governed corpora and registered benchmark inputs exist.
