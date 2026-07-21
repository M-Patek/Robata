# Current Implementation Status

- Status date: 2026-07-21
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
| Contract and schema kernel | Strict values, canonical JSON/SHA-256, six-camera maps, atomic schema registration and immutability checks, logical nodes, immutable revisions, current-selection primitives, registered compact/detailed primary-completion contracts, deterministic collection roots, and a local aggregate repository are executable. Compact V1 bytes remain frozen and exact-readable without an implicit upcast. | The live catalog still declares `upcasters=[]`. Detailed-result bytes are embedded in the local SQLite authority rather than a production artifact store. This is not an O-14 choice or a Phase 1A promotion. |
| Ingestion and alignment | Deterministic six-stream indexing, immutable validation/READY state transitions, exact rational transforms, reset segments, residuals, and admissibility gates are implemented behind injected ports. | No governed corpus, O-03, O-04, or Phase 0 approval; local results cannot claim Phase 1B. |
| Media inspection and export | Live CLIs expose explicitly authorized inspection and local video export. The canonical raw-MCAP command composes the same inspector, six H.264 decoder probes, registered MP4/sidecar export, canonical frame index, and selected PNG publication before entering the shared canonical runner. The former fake-model analysis runner remains archived history only. | Source access must be explicitly authorized. The current mapping profile requires an explicit local-development override; its V2 evidence is local only, non-promotional, and sends no provider traffic. |
| Sampling and package planning | Exact rational grid selection, deterministic frame tie/dedup behavior, materialized provider-neutral frame-budget package sets, and an exact frozen-adaptive resolver from immutable trigger evidence to rational grid segments or explicit integer-nanosecond targets are implemented and locally verified. | `AdaptiveSampler.sample()` still fails closed; signal detection and trigger-to-rate policy are not implemented. O-13 still owns promoted rates, padding, adaptive triggers, and budgets. Provider-limit splitting belongs to `InferenceInputPlan`, not package mutation. |
| Primary inference boundary | Provider-neutral request, capability, intent, terminal-attempt, output-validation, selection-decision logical keys, and strict `InferenceInputPlan` contracts are executable. The canonical runner injects `VisionModelAdapter`, the exact raw-byte store, and the strict claim parser independently; a protocol-only adapter now covers normal execution, exact replay without redispatch, and invalid raw-reference failure. Exact-pinned schemas and one append-only local SQLite ledger preserve the typed evidence graph across fresh instances. | The local composition root still selects only the offline fixture adapter. There is no real Qwen adapter, governed capability evidence, credential path, production attempt/artifact store, or production recovery decision. |
| Canonical offline vertical slice | The fixture and explicitly authorized raw-MCAP commands share one runner through coarse/dense QA, proposal, candidate reduction, per-candidate action evidence, deterministic provisional fusion, separate padded ONSET/OFFSET boundary passes, and final fusion. A versioned final-fusion context binds the exact ordered refined-action closure into the input-plan dependency identity and every adapter request; final reduction accepts explicit zero output or requires exact 1:1 coverage before local admission. Zero proposals/actions terminate as `NO_EVENTS`; indeterminate evidence fails closed; replay and restart-safe inference/barrier recovery avoid redispatch. Detailed completion V4 retains the complete stage chain, while compact completion V3 records the exact terminal stage. The local aggregate atomically commits identity, ActionEvent genesis, completion, and pending outbox for event-producing and early `NO_EVENTS` outcomes. | The path is bound to `canonical-offline-v5`, execution-policy semantic projection v3, fusion projector policy v2, local composition v13, runtime-policy projection v8, compact completion v3, detailed completion v4, and the offline fixture. Outbox delivery, durable work lease/fence lifecycle, real providers, governed policies, and production infrastructure remain open. Every local result remains `evidence_class=LOCAL_CONFORMANCE` and `production_eligible=false`. |
| Shadow and evaluation | Deterministic random/hard-case routing, explicit budget outcomes, primary isolation, paired comparison, and append-only disagreement evidence are executable locally. | GPT/provider governance, quota, retention, and Phase 8 predecessors remain blocked. |
| QA | The runner executes fixture-backed `QA_COARSE` and, for explicit degraded/unusable coordinates, `QA_DENSE` through the same raw/parsed/enriched/barrier chain. The completion projector produces `QA_COMPLETE` or `QA_INCOMPLETE` from exact six-camera evidence; all results remain non-promotable. | Real model quality, adaptive suspicion policy, governed O-10 thresholds, and external corpus validation remain unresolved. |
| Event processing | `EventProposalProjector` normalizes authoritative EVENT_PROPOSAL outputs into six-camera facts, `CandidateReducer` deterministically merges compatible intervals, `ActionEvidenceProjector` emits exact six-camera evidence, and `ProvisionalPhysicalActionFuser` produces ordered 0/1/N coarse actions. `BoundaryRefinementProjector` validates role-bound ONSET/OFFSET closure and emits one refined result per action. A versioned final-fusion context then binds all refined actions to provider requests and requires explicit zero output or exact 1:1 final hypotheses before admission. Run/attempt locators do not enter logical identity and all outputs remain non-production. | Governed O-11/O-12 association/ontology/tolerance policies and real-model quality remain absent. |
| Queueing | `queue/barrier.py` provides deterministic barrier aggregation with an in-memory reference store, and weighted-fair local admission/reservation is tested. `adapters/sqlite_barrier.py` durably stores generic and inference-call barrier facts for each local canonical run. Exact replay/reopen, completion-to-member crash repair, conflicting concurrent completion, corrupted-row failure, and same-run recovery are tested. | There is no durable work ledger, deadline/lease/fence recovery, Redis/broker/DLQ integration, registered persisted-barrier wire contract, or production concurrency topology; SQLite remains local conformance evidence. |
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
| 25.1 VLM trust boundary | PARTIAL | Provider outputs are preserved as typed raw bytes, parsed as untrusted claims, enriched against the exact input-plan catalog, and reduced only after validation across QA, proposal, action-evidence, ONSET/OFFSET boundary-refinement, and final local-inference stages. | Only fixture-backed inference is composed. Real Qwen capability, provider governance, credentials, artifact storage, and governed replay evidence are absent. |
| 25.2 provider-neutral packages and durable call barrier | PARTIAL | Temporal packages remain provider-neutral; provider limits and render details live in `InferenceInputPlan`; independently retried call parts and deterministic reduction are executable. `inference/call_barrier.py` binds the exact input-plan member set over `queue/barrier.py`. The local canonical composition injects one run-scoped SQLite adapter for both storage ports, persisting the generic definition/state/members and call definition/completions/reduction. A fresh pipeline instance can recover the interrupted same run after durable reduction/evidence with zero provider redispatch. | Work-ledger and lease/fence recovery, Redis/broker production topology, a real provider adapter, and registered persisted-barrier wire contracts are not connected. |
| 25.3 exact and adaptive sampling | PARTIAL | Exact rational-grid selection and materialization are tested. `sampling/adaptive.py` deterministically resolves a frozen trigger artifact into reduced rational grid segments or explicit ordered integer-nanosecond targets, with a fixed semantic-projection version and target budget. | `AdaptiveSampler.sample()` is fail-closed. Signal detection and trigger-to-rate policy remain absent; O-13 owns promoted rates, padding, trigger policy, and budgets, and cross-language grid evidence is still required. |
| 25.4 logical identities and run membership | PARTIAL | Generic logical nodes plus coarse QA, dense QA, QA completion, proposal, candidate, candidate-dense window, action-evidence, provisional-fusion result/action, boundary window, boundary role result, and combined boundary-refinement result producers use run-independent projections and append-only memberships. Final fusion retains a versioned run-independent refined-action context and binds its digest to the input plan and calls. Retry/run changes preserve logical keys. | The full event-chain lineage is retained in the exact-pinned terminal detail. Durable work invalidation remains absent. |
| 25.5 immutable revisions and current selection | PARTIAL | Generic immutable node revisions, append-only selection decisions, current-selection compare-and-swap, rebuild, and local verification are implemented. Deterministic local ActionEvent genesis revisions, selections, current projections, and identity current-revision references are applied with identity/completion/outbox facts in one local aggregate transaction. | The governed production ActionEvent contract and REUSED successor-selection policy remain absent; generic revision `ELIGIBLE` means locally selectable, not production-qualified. |
| 25.6 validation, READY, and separate ledgers | PARTIAL | V2 validation/READY/alignment contracts and local source/alignment ledger reconciliation exist behind injected ports and reject insufficient V1 evidence. The raw canonical bridge derives exact source/schema, six-stream decoder-probe, registered-media, frame-index, and alignment facts from an explicitly mapped real MCAP before resolving V2 admission. | The exercised mapping and development admission/alignment policies are unapproved. Governed O-03/O-04 decisions, production stores, independent promoted ledger reconciliation, and Phase 0 approval are absent. |
| 25.7 exact schema references and evolution | PARTIAL | Atomic registration, exact digest verification, and published-schema immutability checks are executable. Compact completion V3 records terminal-stage evidence; detailed completion V4 embeds the complete proposal/candidate/action/provisional/boundary/final-fusion lineage while older versions remain immutable. | Work-message and persisted-barrier contracts remain unregistered. |
| 25.8 nonblocking human review | ABSENT | Generic immutable revision primitives can support later adjudication, but there is no runnable review-routing path. | Nonblocking human-review routing, annotation contracts, backlog/latency evidence, and the governing O-10/O-11/O-12 policy are absent. |
| 25.9 recording-scoped serialized identity | PARTIAL | The one-command local path obtains a recording snapshot, performs side-effect-free canonical identity preparation, and applies it with run completion, ActionEvent genesis publication, and outbox under one generation/fence CAS transaction. Exact same-run replay and cross-run replay-only reuse create no duplicate business result or outbox. | The aggregate is local conformance evidence, not the O-14 production authority; outbox delivery and governed merge/split/REUSED successor behavior remain absent. |
| 25.10 security and phase dependencies | BLOCKED | Later phases run only as local conformance slices with `production_eligible=false`. Coarse/dense QA, proposal, candidate reduction, per-candidate action evidence, provisional fusion, per-action ONSET/OFFSET boundary refinement, and exact 0/1/N final-fusion handling are runnable without provider traffic. | Phase 0 and open O-decisions still block promotion. Real providers, governed policies, and later production stages remain fail-closed rather than fabricated. |
| 25.11 promotion evidence | PARTIAL | Focused local tests cover the canonical normal path, exact replay, proposal/action 0/1/N behavior, per-action ONSET/OFFSET refinement, multi-action final fusion, exact final-closure rejection, multi-part reduction, SQLite inference/barrier recovery, and atomic local completion/outbox replay. | Governed source/provider replay, outbox delivery, cross-language vectors, capacity/SLO evidence, and approved thresholds remain missing. |

### Named-file readiness

| File | Status | Exact executable boundary |
|---|---|---|
| `sampling/adaptive.py` | PARTIAL | The frozen-artifact resolver is implemented and tested; `AdaptiveSampler.sample()` still raises `NotImplementedError`, so adaptive signal/policy execution is absent. |
| `qa_pipeline/coarse.py` | PARTIAL | `CoarseQAProjector` validates authoritative `QA_COARSE` enriched coverage and emits a local result consumed by the separate completion gate. It does not itself authorize fusion or execute dense QA; every result is `production_eligible=false`. Model-quality calibration remains absent. |
| `qa_pipeline/completion.py` | PARTIAL | The deterministic three-state completion gate, explicit zero-child dense outcome, exact dense-work manifest, and six-camera all-GOOD aggregate are implemented and tested. It does not execute dense inference, and every result is `production_eligible=false`. |
| `qa_pipeline/dense.py` | PARTIAL | `DenseQAProjector` and the canonical runner execute exact planned dense work through the shared inference evidence chain. Real-model quality and governed dense policy remain external. |
| `event_pipeline/proposer.py` | PARTIAL | `EventProposalProjector` validates plan-bound enriched outputs, preserves attempt provenance separately, and emits stable `CLAIMS` or `NO_EVENTS` identities. The legacy `EventProposer.propose()` remains fail-closed. |
| `event_pipeline/candidate.py` | PARTIAL | `CandidateReducer` deterministically merges connected proposal intervals, binds policy and proposal-result digests, emits source-bound empty reductions, and keeps all candidates non-production. Legacy mutation APIs remain fail-closed; governed merge/split lifecycle is absent. |
| `event_pipeline/evidence.py` | PARTIAL | `ActionEvidenceProjector` validates candidate-scoped ACTION_DENSE package/input-plan lineage, authoritative enriched claims, exact part coverage, time/evidence bounds, and all six camera slots, then emits stable non-production `SUPPORTED`, `NO_ACTION`, or `INDETERMINATE` evidence. The incompatible legacy `ActionEvidenceExtractor.extract_evidence()` entry remains fail-closed. Governed ontology/calibration and real-model quality remain absent. |
| `event_pipeline/provisional_fusion.py` | PARTIAL | `ProvisionalPhysicalActionFuser` validates an exact candidate/ACTION_EVIDENCE closure and deterministically emits ordered 0/1/N coarse physical actions with six explicit camera slots, source closure, ambiguity, versioned semantic identities, and `production_eligible=false`. Compatible candidates may merge and one candidate may split across disconnected or incompatible positive evidence. Missing, duplicate, foreign, or indeterminate evidence fails closed. | Boundary refinement, governed O-11/O-12 association and ontology, stable event identity, revisions, and publication remain downstream and are intentionally not performed here. |
| `event_pipeline/boundary_refinement.py` | PARTIAL | `BoundaryRefinementProjector` validates separate orchestrator-owned ONSET/OFFSET windows, exact package/input-plan/alignment/enriched-output closure, uncertainty, context truncation, and all six camera slots, then deterministically reduces exactly one non-production result per provisional action without an `event_id`. | Real-model quality and governed O-12 boundary tolerance remain downstream. |
| `queue/barrier.py` | PARTIAL | The coordinator, in-memory reference storage, and terminal criticality semantics are executable and tested. The injected `adapters/sqlite_barrier.py` persists generic and inference-call barrier facts for a local canonical run and proves exact replay, reopen, crash repair, and fail-closed recovery. Durable work scheduling, deadline/lease/fence recovery, Redis/broker integration, registered persisted-barrier wire contracts, and production concurrency remain absent. |

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
- Remaining Section 25.7 registered wire contracts for work messages, standalone output decisions,
  and persisted barriers, plus a production artifact store. The compact completion and
  detailed-result contracts are exact-pinned and one local aggregate persists both, but detailed
  bytes are embedded in SQLite rather than crossing a production artifact boundary. The local
  inference ledger carries exact schema quartets for intent, terminal, selection, typed raw
  metadata, parsed claims, selected outputs, and enriched outputs; its pre-parse byte blob remains
  subordinate storage. The live catalog still declares `upcasters=[]`.
- Production barrier/work recovery. The local canonical composition now resumes same-run generic
  and inference-call barrier facts from run-scoped SQLite, including fresh-instance recovery after
  durable reduction/evidence without provider redispatch. A durable work-message ledger, scheduler
  state, deadlines, leases/fences, Redis/broker topology, and production recovery policy remain
  absent.
- An outbox consumer/publisher. The one-command composition applies prepared identity and
  deterministic `LOCAL_CONFORMANCE`, `production_eligible=false` ActionEvent genesis facts with
  completion and pending outbox rows; delivery, REUSED successor policy, and production storage
  remain absent.
- Governed raw-MCAP admission and publication under approved O-03/O-04 policy and production
  ledgers. The local command now derives registered V2 validation/READY/alignment evidence from
  exact raw facts, but its explicitly unapproved development policy cannot be promoted. Frozen V1
  evidence remains insufficient and cannot be promoted as V2.

SQLAlchemy/Redis/provider adapters are not silently emulated as production infrastructure.
Missing optional dependencies fail closed. No implementation may turn a local fake score,
empty cohort, synthetic replay, or archived report into `MEASURED` or promotional evidence.

## Current Interpretation

The repository now has one connected source-to-completion path for either an immutable fixture or
an explicitly mapped raw MCAP, including explicit processing runs, run-to-node membership,
identity preparation, deterministic genesis revision/current selection, atomic completion, and
pending outbox. Exact rerun is the recovery operation. On 2026-07-20 the live raw command processed
`sample-medium.mcap` into five local events, revisions, and outbox rows; exact replay returned
the same command, completion, event, revision, and outbox identities with zero inference
redispatch. An interrupted same-run execution also reopens its run-scoped SQLite generic/call
barrier and durable inference evidence after reduction commit, then completes with zero provider
redispatch. That is sufficient to continue deep development without a real model. It is not a
production durable skeleton: durable work scheduling and deadline/lease/fence state, registered
persisted-barrier wire contracts, external artifacts, Redis/broker integration, outbox delivery,
governed source/provider adapters, and the O-14 recovery topology remain open. Work behind an
authority gate must stop at an explicit port, policy input, or fail-closed state.

Frozen adaptive trigger evidence can be resolved deterministically before package identity, but
adaptive signal/policy execution remains blocked by O-13. Compact and detailed primary-completion
records now have exact registered schemas and versioned semantic projections; compact V1 remains
frozen/readable without implicit upcast. Side-effect-free identity and ActionEvent genesis
preparation feeds the local aggregate for CREATED/AMBIGUOUS assignments and exact replay. The
local composition invokes this sequence after the runner; REUSED successor lineage, governed
production ActionEvent contracts, external artifact storage, delivery, and production recovery
remain open.

The removed surface includes `robata.contracts.mainline`, `robata.ports.mainline`, their
smoke-only bundle/report/stage contracts, and the duplicate legacy model-adapter port. There are no
compatibility shims. Shared contracts and frame-materialization boundaries that still serve the
canonical, QA, event, and sampling code live under `robata.contracts.pipeline` and
`robata.ports.frame_materialization`.

The local output proof, output decision, and event-hypothesis payloads are unregistered V2
contracts; V1 payloads fail closed. The canonical run binding remains `canonical-offline-v5`;
execution-policy semantic projection v3, fusion projector policy v2, local composition v13,
and runtime-policy projection v8 prevent recovery under the pre-final-fusion policy namespace.

The canonical runner executes coarse QA and any exact planned dense QA before deterministic QA
completion. `QA_COMPLETE` enters EVENT_PROPOSAL and candidate reduction, then executes a
candidate-scoped ACTION_DENSE to ACTION_EVIDENCE chain for every candidate. The deterministic
provisional fuser validates that exact closure and emits ordered 0/1/N coarse physical actions.
Every action then executes separate padded ONSET/OFFSET boundary windows through the shared
raw/parsed/enriched/barrier chain and reduces to one exact six-camera refined result. The runner
builds one versioned final-fusion context from the complete ordered refined-action set, binds it to
the input plan and adapter request, and rejects any nonempty result that does not cover that set
exactly once. Zero proposals/actions and explicit empty final fusion remain distinct source-bound
`NO_EVENTS` outcomes; proposal/action/boundary failure and indeterminate evidence stop explicitly;
retry/run identity remains provenance. No QA or downstream local result is production eligible.

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
