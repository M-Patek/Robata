# Robata Evidence-First Throughput and Quality Optimization Blueprint

**Cycle:** measured throughput, quality, and qualification optimization
**Planning date:** 2026-07-27
**Source guide:** `ARCHITECTURE_OPTIMIZATION_GUIDE.md`
**Dispatch unit:** one numbered phase per main-agent window

## Authority and Evidence Boundary

This file is a local execution map, not a product contract or production certificate.

- Published schemas and `schemas/schema-catalog.json` govern wire contracts.
- Tracked source, tests, and conformance fixtures govern executable behavior.
- `governance/` explains navigation and execution order only.
- Released schema bytes are immutable. Identity, hash, logical-key, idempotency-key,
  fence, semantic-projection, or wire changes require an explicit version or migration
  decision before implementation.
- The optimization guide supplies audited hypotheses. A phase may promote a claim only
  through a workload-bound measurement artifact.

## Current Observed Baseline

These observations are investigation inputs, not production capacity evidence. Parent and
child profile spans overlap and must not be added together.

### Fresh source profile

The current local profile observation uses a 40.8335-second, six-camera recording fixture,
PyAV CPU media processing, and a deterministic offline provider. The source report date,
commit, and digest are not carried in this Blueprint, so these values are unbound historical
planning observations. The runtime profile family is `LOCAL_CONFORMANCE`,
`NOT_MEASURED`, and `NOT_PRODUCTION_QUALIFIED`; P0 must rerun it before
any before/after claim.

| Metric | Local observation | Status |
| --- | ---: | --- |
| Elapsed | 118.750 s | local profile only |
| Source-time ratio | 2.908 wall-sec/source-sec | local profile only |

### Actual local-mock smoke

The separate 30-minute fixture-backed smoke uses provider traffic `MOCKED`, execution mode
`ACTUAL_LOCAL_MOCK`, authority `AUTHORITATIVE_LOCAL_MOCK_SMOKE`, measurement status
`NOT_MEASURED`, and qualification status `NOT_PRODUCTION_QUALIFIED`.

| Metric | Local observation | Status |
| --- | ---: | --- |
| Throughput | 2.162 rec-sec/wall-sec | local regression evidence |
| Eligible-to-terminal p95 | 3.908 s | local regression evidence |
| Active backlog at end | 0 | local regression evidence |
| No-growing-backlog gate | met | local regression evidence |
| Windows | 1,800 | workload descriptor |
| Work items | 9,001 | workload descriptor |

### Profile attribution

| Scope | Wall time | Interpretation |
| --- | ---: | --- |
| `source.prepare` | about 77.38 s | parent envelope; do not add to children |
| `source.stream.capture_publish` | about 53.05 s | append, publication, drain, and related work |
| `source.materialize` | about 22.33 s | media materialization path |

The observations do not establish H100, NVDEC, vLLM, R2, representative quality, long-run,
or production capacity. Those environments and gates remain `NOT_MEASURED`.

## Product Outcome

**Outcome:** Robata processes six-camera recordings with measurably better throughput and
quality while preserving complete evidence, deterministic replay, durable recovery, and
transport-independent identity.

**Why now:** The current unbound local profile observation concentrates work in source preparation, stream
capture/publication, and materialization, while the production provider, target media,
representative quality, and long-run capacity boundaries remain unqualified. Optimization
must therefore begin with measurement and correctness rather than projected multipliers.

**Success measures:**

- Fresh source-time ratio and actual-local-mock throughput remain separate, correctly
  named report surfaces.
- Every performance phase records a like-for-like before/after profile and semantic parity.
- Every quality phase reports per-class, calibration, abstention/incomplete, temporal, and
  boundary effects applicable to that phase.
- No optimization weakens artifact durability, pending-terminal recovery, lease fencing,
  completion immutability, or inference evidence closure.
- Target NVDEC, real provider, external database/storage, representative labels, soak, and
  production capacity remain `NOT_MEASURED` until their named gate runs.
- An executed gate that misses a threshold retains measured evidence, records failure, and
  remains `NOT_PRODUCTION_QUALIFIED`.

**Non-goals for this cycle:**

- No universal 10x, 30x, FPS, latency, or accuracy promise is inferred from local smoke.
- No released schema, identity formula, semantic projection, or completion root is edited
  in place.
- No QA-cleanliness signal is used as proof of `NO_EVENTS`.
- No cross-invocation inference terminal is reused through the current selection contract.
- No lossy encoding or target backend becomes the default before parity qualification.
- No web-product work is included unless a later phase creates a versioned client contract.

## Non-Negotiable Invariants

| Boundary | Invariant |
| --- | --- |
| Source | Original MCAP bytes, timestamps, mapping, decode facts, and provenance are never rewritten. |
| Media | Derived artifacts bind source/time, exact bytes, and all semantics-affecting policies. |
| QA | Every eligible clip retains a complete 21-class projection with explicit unresolved states. |
| Events | Window and recording evidence retains source ordinals, digests, and deterministic replay. |
| Inference | Intent, raw, parsed, selection, accepted evidence, retry, and replay lineage closes. |
| Stream | Plans, readiness, leases, fences, pending terminals, acceptance, EOS, and recovery stay durable. |
| Completion | Published completion is complete, immutable, schema-valid, and semantically hashed. |
| Retrieval | Structured retrieval remains authoritative; vector rows are optional derived projections. |
| Evidence | Local, synthetic, measured, failed-gate, external, and production states are not conflated. |

## Overall Roadmap

| Phase | Outcome | Main module(s) | Depends on | Local proof | External follow-up |
| --- | --- | --- | --- | --- | --- |
| P0 - freeze measurement truth | Bound baseline, units, scope, and identity-impact register | `qualification-ops`, `contract-governance` | None | profile and benchmark tests | None |
| P1 - profile SQLite boundaries | Operation-level transaction/lock profile and safe batch candidates | `stream-control`, `qualification-ops` | P0 | scheduler transaction tests | Multi-process load later |
| P2 - artifact durability barrier | Batched publication without volatile committed references | `source-media`, `canonical-integration` | P0 | crash/reconciliation tests | Target filesystem/R2 |
| P3 - completion root cost | Preserve v3 ordered-root semantics while removing proven overhead | `identity-delivery`, `canonical-integration` | P0 | completion contract/integration tests | None |
| P4 - target decode and media reuse | Real backend qualification plus layer-specific reuse profile | `source-media`, `qualification-ops` | P0,P2 | NVDEC/media parity tests | Target GPU/codec matrix |
| P5 - alternate encoding experiment | Versioned PNG/JPEG policy and representative parity report | `source-media`, `sampling-qa`, `inference-evidence` | P0,P2 | media/input-plan tests | Representative labels/provider |
| P6 - stream-safe provider microbatching | Qualified single/concurrent/native envelopes without recording-wide accumulation or cross-invocation result, cache, or evidence reuse | `inference-evidence`, `qualification-ops` | P0 | RunPod/orchestrator/replay tests | Real provider/H100 |
| P7 - durable executor concurrency | Concurrent ready-work execution with unchanged terminal truth | `stream-control`, `canonical-integration` | P1 | ordering/recovery tests | Saturation run |
| P8 - measured adaptive backpressure | Observable rate inputs and restart-safe bounded controller | `stream-control`, `qualification-ops` | P7 | queue/backlog tests | Production arrivals/quotas |
| P9 - Product QA calibration | Auditable calibration bridge with explicit schema decision | `sampling-qa`, `inference-evidence`, `contract-governance` | P0 | QA/metrics/schema tests | Governed labels |
| P10 - causal adaptive sampling | Persisted upgrade decisions and only proven no-work paths | `sampling-qa`, `event-semantics`, `canonical-integration` | P9 | adaptive/QA/event tests | Recall qualification |
| P11 - event association and temporal signal | Extend existing recording merge without hidden identity change | `event-semantics`, `canonical-integration`, `identity-delivery` | P0 | reduction/event tests | Representative events |
| P12 - boundary estimator qualification | Robust estimator with raw observations and explicit fallback | `event-semantics`, `sampling-qa` | P9,P11 | event/boundary tests | Boundary labels |
| P13 - active-learning selector | Immutable pool-level review selection and dataset lineage | `identity-delivery`, `sampling-qa`, `qualification-ops` | P9 | review/split tests | Annotation cycles |
| P14 - vector adapter qualification | Existing structured-first vector path qualified on real backend | `canonical-integration`, `contract-governance`, `qualification-ops` | P0 | vector/retrieval tests | Encoder/pgvector/RLS |
| P15 - representative Pareto gate | Frozen throughput, quality, reliability, cost, and E0-E6 report | `qualification-ops`, all changed modules | P2-P14 | local qualification package | Labels, hardware, soak, review |

## Module Phases

Every phase in this cycle crosses a contract or evidence boundary, so the implementable
details are collected under `Cross-Module Phases`. This map identifies the module card to
read before dispatch.

| Module | Phases | Primary responsibility |
| --- | --- | --- |
| `contract-governance` | P0,P9,P14,P15 | version/migration decisions, schema/catalog proof, adapter contracts |
| `source-media` | P2,P4,P5,P15 | decode, artifact durability, encoding, media parity |
| `sampling-qa` | P5,P9,P10,P12,P13,P15 | complete QA, calibration consumption, sampling, quality metrics |
| `event-semantics` | P10,P11,P12,P15 | event merge/association, temporal and boundary semantics |
| `inference-evidence` | P5,P6,P9,P15 | input identity, provider envelopes, ledger/replay, calibration lineage |
| `stream-control` | P1,P7,P8,P15 | transactions, durable work concurrency, backpressure, recovery |
| `identity-delivery` | P3,P11,P13,P15 | immutable completion, event identity, review selection and delivery |
| `canonical-integration` | P2,P3,P7,P10,P11,P14,P15 | cross-module composition, replay, validation, end-to-end proof |
| `qualification-ops` | P0,P1,P4,P6,P8-P15 | profiles, benchmark context, promotion and qualification evidence |

## Cross-Module Phases

### P0 - Freeze Measurement and Contract Truth

**Participating modules:** `qualification-ops`, `contract-governance`

**End-to-end result:** Every later optimization compares against a reproducible, correctly
named baseline and declares its identity/schema impact before implementation.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `qualification-ops` | Separate fresh profile from actual-local-mock smoke and bind workload context | `src/robata/runtime/local_streaming_benchmark.py`, `local_streaming_smoke.py`, `canonical_profile.py` | runtime/profile tests |
| `contract-governance` | Record affected schema, identity, hash, and migration decisions | `schemas/schema-catalog.json`, `src/robata/contracts/**` | schema immutability tests |

**Implementation outline**

1. Freeze commit, catalog digest, fixtures, policies, provider mode, hardware, warm/cold
   state, and repeated-run method.
2. Report fresh source-time ratio in wall-sec/source-sec and throughput in
   rec-sec/wall-sec as separate surfaces.
3. Capture stage wall time, transactions by operation, I/O, queue/backlog, provider calls,
   artifacts, and completion time without summing parent/child spans.
4. Create an identity-impact register for P1-P15; unresolved changes block their phase.

**Keep intact**

Existing typed report modes and statuses. `ACTUAL_LOCAL_MOCK` is not virtual simulation or
production capacity.

**Done when**

- [ ] Fresh and smoke artifacts reproduce from documented commands.
- [ ] Units, workload, provider mode, authority, and qualification status are explicit.
- [ ] P1-P15 each has a recorded contract decision or `no contract change` result.
- [ ] No projected multiplier appears as measured evidence.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_local_streaming_benchmark.py tests/unit/test_local_streaming_smoke.py
python -m pytest tests/unit/test_canonical_profile.py tests/unit/test_runtime_capacity.py tests/unit/test_benchmark_qualification.py
python -m pytest tests/unit/test_schema_immutability.py tests/contract/test_schema_release_policy.py
~~~

**Next boundary:** P1-P14 consume the frozen measurement and identity context.

### P1 - Profile and Bound SQLite Transactions

**Participating modules:** `stream-control`, `qualification-ops`, `canonical-integration`

**End-to-end result:** Remaining SQLite cost is attributed by authority and operation;
only measured, same-authority batch or connection changes proceed.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `stream-control` | Transaction, lock, retry, and rollback observability | `src/robata/adapters/sqlite_work_scheduler.py`, `sqlite_stream_work_ledger.py`, `sqlite_stream_delivery.py` | scheduler transaction tests |
| `canonical-integration` | Preserve pending/succeed/accept recovery boundaries | `src/robata/application/canonical/stream_scheduler.py` | composition/recovery tests |
| `qualification-ops` | Compare operation counts and wall time on frozen workload | `src/robata/runtime/canonical_profile.py` | profile tests |

**Implementation outline**

1. Baseline existing `append_windows`, `plan_many`, `mark_published_many`, and atomic
   window-reduction delivery rather than reimplementing them.
2. Measure connection setup, lock wait, transaction duration, fsync, row count, retry, and
   rollback per stable operation name.
3. Select at most one proven bottleneck for connection residency, prepared statements, or
   a writer queue; document thread/process ownership and acknowledgement timing.
4. Test in-process and multi-process contention before changing `busy_timeout`.

**Keep intact**

Pending-terminal crash intent, exact conflict detection, leases, fences, row-version CAS,
and same-database atomic publication.

**Done when**

- [ ] Existing batch operations have explicit baseline counts.
- [ ] The chosen change reduces measured cost on the frozen workload.
- [ ] Crash injection passes before and after every commit boundary.
- [ ] Provider and media work never runs inside a SQLite transaction.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_sqlite_work_scheduler.py tests/unit/test_sqlite_stream_work_ledger.py
python -m pytest tests/unit/test_stream_scheduler_composition.py tests/integration/test_canonical_local_command.py
~~~

**Next boundary:** P7 uses the proven write/claim envelope for concurrent execution.

### P2 - Publish Artifacts Behind a Durability Barrier

**Participating modules:** `source-media`, `canonical-integration`, `qualification-ops`

**End-to-end result:** Batching writes or fsync cannot leave a committed manifest or
terminal referencing volatile or missing bytes.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `source-media` | Staging, hash verification, flush, directory sync, atomic exposure | `src/robata/adapters/pyav_frame_materializer.py`, `src/robata/adapters/local_artifact_registry.py` | artifact/media tests |
| `canonical-integration` | Commit authority only after durable publication | `src/robata/application/canonical/runner.py`, `mcap_source.py` | canonical MCAP tests |
| `qualification-ops` | Fault points and fsync/write profile | `src/robata/benchmark/media_qualification.py` | media qualification tests |

**Implementation outline**

1. Define the publication unit and promised durability model for each supported filesystem.
2. Stage files, verify byte count and exact hash, flush and `fsync` each file, then sync the
   staging directory.
3. Atomically rename the complete publication, sync the destination parent directory, and
   only then commit database authority that references it.
4. Reconcile abandoned staging, missing final files, partial publication, and corruption.
5. Introduce bounded batching only after the barrier and fault model are tested.

**Keep intact**

Exact-byte identity and artifact lineage. A hash proves content, not persistence; the stream
pending-terminal protocol is not substituted for media durability.

**Done when**

- [ ] Every crash point produces either no visible publication or a complete durable one.
- [ ] Restart reconciliation is idempotent and detects corruption.
- [ ] No committed authority references transient tmpfs-only bytes.
- [ ] Before/after write, fsync, latency, and end-to-end measurements are recorded.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_local_artifact_registry_observability.py tests/unit/test_staged_registered_video_export.py
python -m pytest tests/integration/test_canonical_mcap_source.py tests/unit/test_media_qualification.py
~~~

**Next boundary:** P4 and P5 publish new backend/encoding artifacts through this barrier.

### P3 - Profile Completion Root Construction

**Participating modules:** `identity-delivery`, `canonical-integration`, `qualification-ops`

**End-to-end result:** Proven completion overhead is reduced without mutable v3 records or
a changed ordered-collection digest formula.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `identity-delivery` | Preserve immutable v3 persistence | `src/robata/adapters/sqlite_primary_completion.py`, `src/robata/contracts/primary_completion.py` | completion contract tests |
| `canonical-integration` | Profile sort/serialize/hash/artifact/commit scopes | `src/robata/application/canonical/primary_completion.py` | completion integration tests |
| `qualification-ops` | Size-stratified completion report | `src/robata/runtime/canonical_profile.py` | profile tests |

**Implementation outline**

1. Attribute sorting, serialization, collection-root hashing, schema validation, artifact
   persistence, and database commit separately.
2. If leaf preparation is material, spool digests in each collection's exact existing
   canonical order. External sorting is allowed only by the existing canonical collection
   key, never by digest bytes.
3. Continue passing the complete ordered digest list to
   `canonical_collection_digest_root` for v3.
4. Treat any subtree-root algorithm or placeholder/backfill model as a new versioned design.

**Keep intact**

All eleven required count/root pairs, synchronous semantic validation, immutable table
triggers, and complete publication before outbox use.

**Done when**

- [ ] A profile proves which completion sub-scope is material.
- [ ] Zero, large, duplicate, and ordering cases reproduce current v3 roots exactly.
- [ ] Restart and artifact-write failures cannot publish partial completion.
- [ ] Any new root formula has a separate approved schema/projection migration phase.

**Run locally**

~~~powershell
python -m pytest tests/contract/test_primary_completion_contract.py
python -m pytest tests/integration/test_sqlite_primary_completion.py tests/unit/test_canonical_profile.py
~~~

**Next boundary:** P15 consumes unchanged completion semantics and measured cost.

### P4 - Qualify Target Decode and Layered Media Reuse

**Participating modules:** `source-media`, `canonical-integration`, `qualification-ops`

**End-to-end result:** A real target decode implementation is selectable behind the
existing port, and reuse removes only duplication proven in the canonical path.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `source-media` | Target backend, explicit fallback, layer-specific cache keys | `src/robata/adapters/nvdec_backend.py`, `nvdec_frame_materializer.py`, `nvdec_video_export.py`, `src/robata/frame_cache.py` | NVDEC/media unit tests |
| `canonical-integration` | Correct backend selection and publication ordering | `src/robata/application/canonical/mcap_source.py` | MCAP integration tests |
| `qualification-ops` | Codec/timestamp/artifact/resource parity matrix | `src/robata/benchmark/media_qualification.py` | media qualification tests |

**Implementation outline**

1. Profile decode, selection, encode, cache lookup, and package binding separately; do not
   assume overlapping windows cause repeated decode.
2. Implement the real backend behind the existing target interface and define supported
   codec/profile/resolution combinations.
3. Record media runtime provenance separately from inference `CapabilitySnapshot`.
4. Define raw-frame, encoded-artifact, and manifest cache equivalence independently.
5. Permit fallback only before dependent publication and record its reason.

**Keep intact**

Whole-MCAP identity, derived spool identity, frame/package/input-plan identities, source
timestamps, and the P2 durability barrier. Changed derived bytes require parity or versioning.

**Done when**

- [ ] Local fake-target and fallback paths pass malformed-input and restart tests.
- [ ] Target matrix reports timestamp, selected-frame, dimension, exact-byte, and semantic parity.
- [ ] Cache hit/miss, memory, eviction, cleanup, and corruption behavior are bounded.
- [ ] Target throughput/resource claims stay `NOT_MEASURED` until real hardware runs.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_nvdec_media_adapters.py tests/unit/test_media_qualification.py
python -m pytest tests/unit/test_mcap_single_pass.py tests/integration/test_canonical_mcap_source.py
~~~

**Next boundary:** P5 consumes the media policy boundary; P15 consumes target evidence.

### P5 - Evaluate a Versioned Alternate Encoding Policy

**Participating modules:** `source-media`, `sampling-qa`, `inference-evidence`, `contract-governance`

**End-to-end result:** PNG and experimental JPEG paths can be compared without silently
changing package, provider-input, or Product QA semantics.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `source-media` | Opt-in encoder policy and exact provenance | `src/robata/adapters/pyav_frame_materializer.py`, `src/robata/application/canonical/mcap_source.py` | MCAP/media tests |
| `sampling-qa` | Update PNG-only validation only under approved policy | `src/robata/qa_pipeline/supplemental.py` | QA pipeline tests |
| `inference-evidence` | Bind media type/bytes/config into input plan and replay | `src/robata/inference/input_plan.py` | input-plan tests |
| `contract-governance` | Decide policy-only change versus new published version | `schemas/schema-catalog.json`, `src/robata/contracts/**` | schema tests |

**Implementation outline**

1. Record a P0 contract decision before adding JPEG to any canonical policy.
2. Pin the encoder implementation and version, quality, chroma, resize, color conversion,
   and metadata.
3. Keep PNG as default while the alternate path is an explicit experiment.
4. Verify provider acceptance and exact request recording/replay.
5. Compare actual selected-frame sizes plus per-class QA/event/boundary outcomes.

**Keep intact**

Source identity does not authorize reuse of changed derived bytes. Artifact, package, and
input-plan identities must reflect encoding differences.

**Done when**

- [ ] Every PNG-only producer, validator, consumer, fixture, and error path is enumerated.
- [ ] Opt-in encoding has complete provenance and deterministic replay behavior.
- [ ] Representative parity criteria are signed before any default change.
- [ ] Speed, size, quality, and end-to-end deltas are reported independently.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_inference_input_plan.py tests/unit/test_qa_pipeline_core.py
python -m pytest tests/integration/test_canonical_mcap_source.py tests/integration/test_canonical_offline.py
python -m pytest tests/unit/test_schema_immutability.py
~~~

**Next boundary:** P6 consumes the exact provider media envelope; P15 decides promotion.

### P6 - Qualify Stream-Safe Provider Concurrency, Client Microbatching, and Native Batch Envelopes

**Participating modules:** `inference-evidence`, `canonical-integration`, `qualification-ops`

**End-to-end result:** Single, concurrent-single, bounded client-microbatch, and native-batch
modes have explicit wire/replay semantics and a measured provider capacity/latency/cost envelope
without turning the recording stream into a recording-wide batch job. Compatible ready invocations
may share one bounded dispatch envelope while their identities, terminal outcomes, and evidence stay separate.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `inference-evidence` | Envelope, item association, failure isolation, evidence closure | `src/robata/inference/orchestrator.py`, `runpod.py`, `sqlite_inference_evidence.py` | RunPod/orchestrator tests |
| `canonical-integration` | Bounded call-part concurrency and replay | `src/robata/application/canonical/parallel_service.py`, `runner.py` | call-part concurrency tests |
| `qualification-ops` | Concurrency/batch/latency/error/token/cost matrix | `src/robata/benchmark/**` | benchmark tests |

**Implementation outline**

1. Baseline the local canonical composition defaults: 8-item/5 ms client microbatching and
   6-call-part concurrency, with explicit single-request and concurrency controls. A microbatch
   contains only independently ready, compatibility-key-matched requests and flushes on its item
   limit or configured delay from its first queued item. Queued cancellation removes only that
   member; cancellation after provider dispatch settles against the resulting item outcome. It
   never waits for recording EOS or a recording-wide batch.
2. Enable native batches only for an endpoint handler that declares exact support and passes
   the streaming wait/deadline gate; otherwise retain single or concurrent-single dispatch.
3. Record/replay the exact HTTP request form and associate each item with its own evidence.
4. Test partial, out-of-order, duplicate, malformed, timeout, and retry outcomes.
5. Measure prefix caching only under the deployed server/model/prompt configuration.

**Keep intact**

Logical invocation identity, per-request evidence, selection/terminal ownership, accepted-call
closure, and streaming progress. Cross-invocation result, cache, and evidence reuse, plus
recording-wide accumulation, are outside the current contract; sharing a bounded HTTP envelope
never merges logical invocation identity or item evidence.

**Done when**

- [ ] All three modes pass exact replay and item-association tests.
- [ ] Native batch remains opt-in until representative endpoint evidence passes.
- [ ] Batch formation is bounded by count and queue-delay gates, never recording completion.
- [ ] Failed items cannot adopt successful sibling evidence.
- [ ] The selected operating point states throughput, p95/p99, errors, and cost.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_runpod_adapter.py tests/unit/test_inference_orchestrator.py
python -m pytest tests/unit/test_sqlite_inference_evidence.py tests/unit/test_inference_input_plan.py
python -m pytest tests/unit/test_inference_microbatch.py
python -m pytest tests/unit/test_canonical_offline_call_part_concurrency.py
~~~

**Next boundary:** P7 uses the measured provider concurrency limit; P15 reruns on real hardware.

### P7 - Execute Ready Durable Work Concurrently

**Participating modules:** `stream-control`, `canonical-integration`, `qualification-ops`

**End-to-end result:** Stage-affine workers increase overlap by claiming durable ready work,
without replacing authority with in-memory queues or inventing cross-window dependencies.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `stream-control` | Concurrent claims, leases, fences, terminal acceptance | `src/robata/application/canonical/stream_scheduler.py`, `src/robata/adapters/sqlite_work_scheduler.py` | scheduler tests |
| `canonical-integration` | Bounded executor pools and deterministic finalization | `src/robata/application/canonical/local_stream_finalization.py`, `parallel_service.py` | integration tests |
| `qualification-ops` | Stage queue-wait/execution/backlog profile | `src/robata/runtime/canonical_profile.py` | profile tests |

**Implementation outline**

1. Instrument actual executor work rather than assigning guessed CPU/GPU roles to stage names.
2. Add one bounded worker per measured stage/executor, then tune only bottleneck pools.
3. Use in-memory queues only after durable claim and retain durable terminal acceptance.
4. Preserve the existing per-window DAG; add no window-N to window-N+1 edge.
5. Exercise randomized completion, expiry, duplicates, crash/restart, and EOS races.

**Keep intact**

Plans, readiness, dependency semantics, leases/fences, pending terminals, closure, and one
finalization after EOS.

**Done when**

- [ ] Concurrency one exactly reproduces the prior result.
- [ ] Higher concurrency reproduces semantic results under randomized ordering.
- [ ] Memory and backlog remain bounded under overload.
- [ ] Before/after stage and end-to-end profiles use the P0 workload.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_stream_scheduler_composition.py tests/unit/test_bounded_stream_work_queues.py
python -m pytest tests/unit/test_sqlite_work_scheduler.py tests/integration/test_canonical_local_command.py
~~~

**Next boundary:** P8 controls the proven concurrent executor; P15 measures integrated capacity.

### P8 - Feed Measured Signals into Restart-Safe Backpressure

**Participating modules:** `stream-control`, `canonical-integration`, `qualification-ops`

**End-to-end result:** Admission and concurrency limits respond to measured queue/provider
conditions through one observable, bounded controller whose state survives restart.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `stream-control` | Rate observations, decisions, ownership, persisted controller state | `src/robata/queue/backpressure.py`, `src/robata/queue/stream_runtime.py`, `src/robata/adapters/sqlite_work_scheduler.py`, `src/robata/adapters/sqlite_stream_work_ledger.py` | queue/scheduler tests |
| `canonical-integration` | Supply real scheduler metrics and enforce timing-only actions | `src/robata/application/canonical/stream_scheduler.py` | stream composition tests |
| `qualification-ops` | Arrival/service/backlog/utilization stability report | `src/robata/runtime/observability.py`, `src/robata/runtime/benchmark.py`, `src/robata/runtime/capacity.py`, `src/robata/runtime/canonical_profile.py` | capacity/profile tests |

**Implementation outline**

1. Baseline the fixed P7 limits under steady, burst, overload, provider-quota, and drain
   workloads before introducing an adaptive limit.
2. Define observation windows and clock semantics for arrivals, accepted terminals, queue
   age/depth, backlog slope, provider quota, and worker utilization. Unknown rates must not
   be silently supplied as zero. The current `QueueMetrics` fields are non-negative;
   a signed drain slope requires a versioned model rather than overloading zero.
3. Persist policy version, owner/fence, current limit, last decision, and sufficient sample
   state for deterministic restart; use one controller owner per declared provider/partition.
4. Start with a measured fixed limit. If AIMD is selected, specify minimum/maximum, increase
   and decrease rules, cooldown/hysteresis, restart value, quota response, and recording
   fairness before enabling it.
5. Keep shedding actions operational. Any action that changes sampling or result semantics
   is dispatched through P10 with a new policy/identity decision.

**Keep intact**

Durable ledger authority, leases, fences, terminal acceptance, per-recording fairness, and
logical identities. Runtime timing and transient utilization do not enter result identity.

**Done when**

- [ ] Default snapshots no longer present unknown rate/quota signals as measured zeros.
- [ ] Identical persisted state and observations reproduce the same decision after restart.
- [ ] Multi-process ownership, quota changes, overload, oscillation, fairness, and drain pass.
- [ ] The adaptive controller beats or matches the fixed baseline without semantic drift.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_bounded_stream_work_queues.py tests/unit/test_stream_scheduler_composition.py
python -m pytest tests/unit/test_sqlite_work_scheduler.py tests/unit/test_runtime_capacity.py
python -m pytest tests/integration/test_canonical_local_command.py
~~~

**Next boundary:** P15 uses the selected stable controller and its backlog evidence.

### P9 - Establish an Auditable Product QA Calibration Bridge

**Participating modules:** `sampling-qa`, `inference-evidence`, `contract-governance`,
`qualification-ops`

**End-to-end result:** Calibrated scores, when applicable, are traceable to a frozen
calibration artifact and cannot silently change Product QA decisions or published meaning.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `sampling-qa` | Consume applicable calibration without conflating raw/model/policy scores | `src/robata/qa_pipeline/aggregate.py`, `src/robata/qa_pipeline/product.py`, `src/robata/application/canonical/product_qa.py`, `src/robata/contracts/qa.py` | QA pipeline/product tests |
| `inference-evidence` | Calibration artifact lineage on provider-neutral evidence | `src/robata/contracts/confidence.py`, `src/robata/inference/enrichment.py`, `src/robata/inference/models.py`, `src/robata/inference/orchestrator.py` | inference evidence tests |
| `contract-governance` | Decide inference-only reuse versus a new Product QA/detail version | `schemas/v1/model-inference.schema.json`, `schemas/schema-catalog.json`, `src/robata/contracts/**` | schema release tests |
| `qualification-ops` | Leakage-safe fit/evaluation and per-class reports | `src/robata/benchmark/metrics.py`, `src/robata/benchmark/qualification.py`, `src/robata/benchmark/splits.py` | metrics/split tests |

**Implementation outline**

1. Record the contract branch first. Existing calibrated-confidence types and ECE/Brier
   calculators do not mean that Product QA currently emits calibrated values or lineage.
2. Freeze score family, model/runtime/preprocess revisions, fitting method, parameters,
   training population, grouped split, applicability, and artifact digest.
3. Populate existing inference calibration lineage only when its applicability matches.
   Keep Product QA wire values uncalibrated unless a registered Product QA/detail
   version adds an explicit calibrated kind/field and semantic projection.
4. Fit on development/calibration data and evaluate once on frozen grouped test data. Keep
   raw score, calibrated probability, deterministic features, and policy decision distinct.
5. Gate any threshold/policy change separately from calibration quality so improved ECE
   cannot be reported as improved classification by implication.

**Keep intact**

The complete 21-class Product QA projection and its `OBSERVED`, `NO_ISSUE`,
`ABSTAINED`, and `INCOMPLETE_INPUT` semantics. `InferenceAttemptSelection` remains a
single accepted attempt, not an ensemble contract.

**Done when**

- [ ] The chosen contract branch and any required schema/upcaster decision are recorded.
- [ ] Calibration artifacts are immutable, content-addressed, applicable, and replayable.
- [ ] Per-class reliability, ECE/Brier, abstention, subgroup, temporal, and drift reports run.
- [ ] Missing, stale, or inapplicable calibration fails closed to the declared raw-score path.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_inference_enrichment.py tests/unit/test_qa_pipeline_core.py tests/unit/test_local_qa_product.py tests/unit/test_qa_completion.py
python -m pytest tests/unit/test_benchmark_metrics_promotion.py tests/unit/test_benchmark_splits.py tests/unit/test_representative_production_qualification.py
python -m pytest tests/unit/test_schema_immutability.py tests/contract/test_schema_catalog.py tests/contract/test_schema_release_policy.py tests/contract/test_schema_upcasting.py
~~~

**Next boundary:** P10 may consume accepted calibrated evidence; P12 may evaluate it as a
camera-quality input; P13 may use it as one selection term.

### P10 - Persist Causal Adaptive Sampling and Proven No-Work Decisions

**Participating modules:** `sampling-qa`, `event-semantics`, `canonical-integration`,
`contract-governance`, `qualification-ops`

**End-to-end result:** Additional sampling is caused by accepted upstream evidence and
replays from a frozen decision; work is skipped only when a domain result already proves it
unnecessary.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `sampling-qa` | Versioned upgrade decision, budgets, exact target provenance | `src/robata/sampling/adaptive.py`, `src/robata/sampling/materializer.py`, `src/robata/qa_pipeline/completion.py`, `src/robata/qa_pipeline/dense.py` | adaptive/QA tests |
| `event-semantics` | Preserve event evidence; define any future event-presence gate separately | `src/robata/event_pipeline/evidence.py`, `src/robata/event_pipeline/proposer.py` | event tests |
| `canonical-integration` | Add the bridge, seal the decision before extra work, and replay it on restart | `src/robata/application/canonical/mcap_source.py`, `src/robata/application/canonical/runner.py`, `src/robata/application/canonical/result_validation.py`, `src/robata/application/canonical/local_supplemental_qa.py` | canonical tests |
| `contract-governance` | Version changed reason vocabulary, plans, skip semantics, or projections | `schemas/schema-catalog.json`, `src/robata/contracts/**` | schema tests |
| `qualification-ops` | Measure quality/compute tradeoffs on grouped evidence | `src/robata/benchmark/metrics.py`, `src/robata/benchmark/splits.py` | metric/split tests |

**Implementation outline**

1. Freeze the base coverage plan before model feedback and retain its safety coverage.
2. The current `AdaptiveCoveragePlan` and `ResolvedAdaptivePlan` are pure
   models, not a decision store or complete base-plan binding. Define the internal durable
   decision schema and storage boundary, or register a published version if the decision
   crosses a wire. Bind accepted evidence, base-plan digest, policy, per-camera/total budget,
   trigger provenance, and selected additional timestamps.
3. Persist or publish that decision before extra targets execute. Replay consumes the stored
   decision and accepted evidence rather than asking a stochastic provider again.
4. Preserve existing explicit dense-QA not-needed outcomes only where coarse-complete
   semantics already prove them; every skipped item still receives an unambiguous terminal.
5. Keep event proposal work unless a separately versioned event-presence gate is trained and
   passes a representative recall bound. QA-clean video is never proof of `NO_EVENTS`.

**Keep intact**

Exact rational grid semantics, plan sealing, bounded targets, accepted evidence lineage,
domain-specific terminal meanings, event recall, and completion validation. A policy version
is immutable during a run.

**Done when**

- [ ] Duplicate/out-of-order triggers reduce to one bounded canonical decision.
- [ ] Restart and replay reproduce the same target coordinates and identities.
- [ ] Abstention, incomplete input, exhausted budgets, and late feedback are explicit.
- [ ] No-work tests include clear-video events and cannot map a scheduler skip to `NO_EVENTS`.
- [ ] Local tests prove deterministic decisions; representative quality/compute deltas are
      an explicit P15 qualification input before promotion.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_sampling_adaptive.py tests/unit/test_adaptive_sampler_runtime.py tests/unit/test_supplemental_temporal_package.py
python -m pytest tests/unit/test_qa_pipeline_core.py tests/unit/test_qa_completion.py tests/unit/test_local_qa_product.py
python -m pytest tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py tests/integration/test_canonical_offline.py
~~~

**Next boundary:** P11 consumes accepted event evidence; P12 consumes exact adaptive boundary
packages; P15 evaluates recall and compute jointly.

### P11 - Add Recording Association and a Nonblocking Temporal Signal

**Participating modules:** `event-semantics`, `canonical-integration`,
`identity-delivery`, `contract-governance`, `qualification-ops`

**End-to-end result:** Recording-level association adds explainable track/continuity evidence
beyond the existing same-label interval merge, without silently changing primary completion.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `event-semantics` | Versioned association rules, confidence, merge/split explanations | `src/robata/event_pipeline/candidate.py`, `src/robata/event_pipeline/evidence.py`, `src/robata/event_pipeline/provisional_fusion.py`, `src/robata/event_pipeline/proposer.py`, `src/robata/event_pipeline/fusion.py`, `src/robata/event_pipeline/boundary_refinement.py` | event/temporal tests |
| `canonical-integration` | Project exact accepted window evidence into recording association | `src/robata/application/canonical/stream_recording_reduction.py` | reduction tests |
| `identity-delivery` | Publish an asynchronous derived report linked to run/revisions | `src/robata/contracts/revisions.py`, `src/robata/application/canonical/logical_nodes.py`, `src/robata/application/canonical/local_outbox_delivery.py`, `src/robata/adapters/sqlite_outbox.py` | revision/outbox tests |
| `contract-governance` | Version any track, event identity, stage, wire, or completion change | `schemas/schema-catalog.json`, `src/robata/contracts/**` | schema/projection tests |
| `qualification-ops` | Produce fixture metrics and defer representative claims to P15 | `src/robata/benchmark/metrics.py` | benchmark metric tests |

**Implementation outline**

1. Freeze V3/V4 baseline behavior: `LocalStreamMergedHypothesis` already merges same-label
   touching/overlapping intervals and retains source ordinals and proposal digests.
2. Define only the added semantics: association across justified gaps or label transitions,
   using time, camera/evidence overlap, explicit confidence, and deterministic tie-breaking.
3. Retain every input hypothesis plus the policy term that explains each merge, split,
   ambiguity, or unassociated result.
4. V4 does not contain camera-overlap or confidence fields; load those facts from accepted
   evidence or add a new versioned input contract rather than inferring them from a digest.
5. Initially publish association and temporal consistency as an asynchronous derived report
   linked to the completed run and exact source evidence. Do not add `EVENT_TRACKING` to
   released `StreamStage` or fields to v4 completion in place.
6. If association must govern event identity or primary completion, stop and register a new
   event/detail/projection version with migration and replay rules.

**Keep intact**

Existing recording reduction, source ordinals/digests, deterministic replay, immutable
event revisions, and nonblocking primary completion.

**Done when**

- [ ] Existing V3/V4 fixtures reproduce byte-for-byte when association is disabled.
- [ ] Gaps, label changes, long events, overlaps, duplicates, ambiguity, and replay are tested.
- [ ] Derived reports cite exact source evidence and cannot mutate historical events.
- [ ] Local fixture metrics are produced; representative precision/recall, association quality,
      boundary effects, and review yield are explicit P15 inputs.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_stream_recording_reduction.py tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py
python -m pytest tests/unit/test_supplemental_temporal_package.py tests/unit/test_event_projection_guards.py
python -m pytest tests/unit/test_event_identity_registry.py tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py tests/contract/test_primary_completion_contract.py
python -m pytest tests/integration/test_canonical_action_event_revision.py
~~~

**Next boundary:** P12 evaluates boundaries on associated events; P15 evaluates temporal
quality without making the derived report a hidden completion dependency.

### P12 - Qualify a Quality-Aware Boundary Estimator

**Participating modules:** `event-semantics`, `sampling-qa`, `canonical-integration`,
`qualification-ops`, `contract-governance`

**End-to-end result:** A representative benchmark selects a robust boundary policy that
retains raw camera observations, exclusions, uncertainty, and deterministic replay.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `event-semantics` | Baseline and candidate reducers with explicit missing-data behavior | `src/robata/event_pipeline/boundary.py`, `src/robata/event_pipeline/boundary_refinement.py` | event core/projection tests |
| `sampling-qa` | Applicable camera-quality evidence without treating QA as geometry | `src/robata/qa_pipeline/aggregate.py` | QA pipeline tests |
| `canonical-integration` | Exact role windows, package lineage, and revision projection | `src/robata/application/canonical/boundary_windows.py`, `src/robata/application/canonical/action_event_revision.py` | action revision tests |
| `qualification-ops` | Onset/offset error and uncertainty coverage by class/condition | `src/robata/benchmark/metrics.py` | benchmark metrics tests |
| `contract-governance` | Reducer/policy/projection version decision | `schemas/schema-catalog.json`, `src/robata/contracts/**` | schema tests |

**Implementation outline**

1. Inventory both current paths before editing: legacy `BoundaryRefiner` permits an
   explicit candidate fallback, while canonical dual-role
   `median-low-max-envelope-v1` has `fallback_enabled=False`.
2. Freeze their current median-low center, maximum-uncertainty envelope, minimum-camera,
   missing-observation, alignment, and identity behavior.
3. Compare robust estimators and an explicit camera-exclusion rule on governed boundary
   labels. Persist every raw observation, quality input, exclusion reason, estimate, and
   uncertainty interval used by the chosen reducer.
4. When quality evidence is missing or inapplicable, retain the selected path's current
   estimator and record that quality weighting was not applied; do not invent a coarse-event
   fallback for the no-fallback canonical path.
5. Register a new reducer/policy and any required wire/projection version before changed
   estimates can become authoritative. Reject weighting that narrows uncertainty without
   a calibrated coverage argument.

**Keep intact**

Exact package-camera coordinates, verified alignment, raw evidence, role separation,
determinism, and action-event revision lineage.

**Done when**

- [ ] Baseline fixtures reproduce unchanged outputs under the current reducer version.
- [ ] Missing/degraded cameras, outliers, ties, contradictions, and no-fallback cases pass.
- [ ] Onset/offset error and uncertainty coverage are reported by class and camera condition.
- [ ] Any authoritative semantic change has an approved version and migration/replay plan.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py
python -m pytest tests/unit/test_qa_pipeline_core.py tests/unit/test_benchmark_metrics_promotion.py
python -m pytest tests/integration/test_canonical_action_event_revision.py
~~~

**Next boundary:** P15 compares the selected estimator with the frozen P0 quality baseline.

### P13 - Select Active-Learning Work from an Immutable Review Pool

**Participating modules:** `identity-delivery`, `sampling-qa`, `qualification-ops`,
`contract-governance`

**End-to-end result:** A versioned pool-level selector chooses bounded annotation work while
retaining existing trigger priority, immutable selection evidence, and leakage-safe lineage.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `identity-delivery` | Eligible pool snapshot, deterministic ranking, append-only decision | `src/robata/review/models.py`, `src/robata/review/routing.py`, `src/robata/adapters/sqlite_review_queue.py` | review/queue tests |
| `sampling-qa` | Auditable uncertainty/disagreement/coverage terms | `src/robata/application/canonical/local_review_routing.py` | local routing tests |
| `qualification-ops` | Grouped splits, selection-bias and held-out-yield reports | `src/robata/benchmark/splits.py`, `src/robata/benchmark/metrics.py` | split/metrics tests |
| `contract-governance` | Persistence/identity decision for selection and annotation lineage | `schemas/schema-catalog.json`, `src/robata/contracts/**` | schema tests |

**Implementation outline**

1. Preserve committed per-completion routing and its QA degradation, low-confidence, and
   sampling priorities as eligibility/priority inputs; do not replace them with one score.
2. Freeze a pool digest and budget, then rank with versioned uncertainty, disagreement,
   coverage, diversity, recency, and existing-priority terms plus deterministic tie-breaks.
3. Append an immutable decision binding the pool, policy/model revisions, term values,
   selected task IDs, budget, and reason for every selected item. Historical routes do not
   change when annotations or models arrive.
4. Link annotation artifacts and adjudication to selection and dataset lineage without
   making review block primary completion.
5. Keep development, calibration, and frozen evaluation groups disjoint by recording,
   connected capture/camera/time groups, and declared leakage policy.

**Keep intact**

Review remains nonblocking, task idempotency/conflict rules remain exact, and annotation
arrival never auto-promotes a model or rewrites an earlier decision.

**Done when**

- [ ] The same pool/policy/budget produces the same ordered selection.
- [ ] Duplicate tasks, concurrent selection, restart, exhausted budget, and late labels pass.
- [ ] Split validation prevents connected recordings from crossing train/calibration/test.
- [ ] Yield, coverage, subgroup balance, agreement, and frozen held-out quality are reported.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_review_routing.py tests/integration/test_sqlite_review_queue.py
python -m pytest tests/integration/test_canonical_local_review_routing.py
python -m pytest tests/unit/test_benchmark_splits.py tests/unit/test_benchmark_metrics_promotion.py
~~~

**Next boundary:** Governed P13 labels feed future training cycles; P15 consumes only frozen
evaluation evidence, never the selection pool itself.

### P14 - Qualify the Existing Structured-First Vector Boundary

**Participating modules:** `canonical-integration`, `contract-governance`,
`qualification-ops`, `identity-delivery`

**End-to-end result:** A real encoder and pgvector-compatible adapter can asynchronously
rerank bounded structured candidates without becoming source of truth or crossing tenants.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `canonical-integration` | Preserve structured candidate authority and revision linkage | `src/robata/retrieval/service.py`, `src/robata/retrieval/index.py` | retrieval tests |
| `contract-governance` | Physical adapter, access policy, versioned vector identity | `src/robata/ports/vector_projection.py`, `src/robata/contracts/retrieval.py` | projection contract tests |
| `qualification-ops` | Backfill/search/recall/latency/selectivity/cost evidence | `src/robata/benchmark/retrieval.py` | retrieval profile tests |
| `identity-delivery` | Async revision projection and idempotent backfill/retry | `src/robata/application/canonical/projections.py` | projection/replay tests |

**Implementation outline**

1. Baseline the existing in-memory contract: structured filters produce a bounded candidate
   set first; vector scoring is optional and non-authoritative.
2. Implement the encoder separately from storage and pin model/revision, dimension,
   preprocessing, normalization, distance metric, and embedding identity.
3. Implement a physical adapter behind `VectorProjectionStore` with immutable event
   revision linkage, existing idempotency/conflict semantics, explicit FAILED/retry state,
   and restartable cursor backfill.
4. Enforce tenant access in both structured and vector paths. An unbound query is not an
   administrative cross-tenant shortcut; verify database RLS in addition to application checks.
5. Qualify exact index type/parameters, build/rebuild, stale revisions, duplicate writes,
   missing vectors, adapter failure, pagination, and deterministic tie-breaking.

**Keep intact**

`EventIndex` and structured filtering remain authoritative. Vector projection is
asynchronous, revision-bound, nonblocking, and unable to self-promote production eligibility.

**Done when**

- [ ] In-memory and physical adapters pass the same projection/idempotency contract.
- [ ] Projection failure falls back to structured results without corrupting authority.
- [ ] Tenant isolation is proven at application and database layers.
- [ ] Representative recall at k, latency, selectivity, build/backfill, and cost are recorded.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_vector_projection_contract.py tests/unit/test_retrieval.py
python -m pytest tests/unit/test_retrieval_profile.py
python -m pytest tests/integration/test_canonical_action_event_revision.py
~~~

**Next boundary:** P15 records the real adapter as optional product evidence, not as a
primary completion dependency.

### P15 - Run the Representative Pareto and External Qualification Gates

**Participating modules:** `qualification-ops` and every module changed by P1-P14

**End-to-end result:** One content-addressed evidence package selects an operating point
from throughput, quality, reliability, deadline, resource, and cost tradeoffs; unresolved
external gates remain explicit.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `qualification-ops` | Frozen scope, Pareto report, representative gate aggregate | `src/robata/benchmark/pareto.py`, `src/robata/benchmark/qualification.py`, `src/robata/runtime/capacity.py` | Pareto/qualification tests |
| changed modules | Export exact policy, identity, recovery, quality, and resource evidence | phase paths above | phase-local tests |
| `contract-governance` | Catalog digest and release/version proof | `schemas/schema-catalog.json` | schema/release tests |

**External gates**

| Gate | Required evidence | Promotion condition |
| --- | --- | --- |
| E0 - evidence freeze | Commit, schema catalog, workload/benchmark manifests, governed corpus, grouped splits, policies, hardware/provider/storage and cost assumptions | Scope is immutable and reproducible before tuning |
| E1 - quality sign-off | QA/event/boundary/calibration metrics, abstention/incomplete states, subgroup/temporal coverage and leakage audit | Registered per-class and aggregate gates pass on frozen labels |
| E2 - media and storage | Target codec/camera matrix, NVDEC/fallback parity, artifact durability, target filesystem/object-store faults and reconciliation | Exact/semantic parity and promised durability pass |
| E3 - provider | Real model/runtime/hardware, single/concurrent/native envelopes, replay, saturation, partial failure, latency and cost | Selected topology passes correctness and operating envelope |
| E4 - reliability and soak | Representative arrivals, restart/retry/fence, provider/storage fault injection, backlog/drain and at least the declared soak duration | No lost/duplicate authority; error and backlog gates pass |
| E5 - capacity, deadlines, cost | 500 recording-hours per 24 hours equivalent, six-camera accounting, QA T+1, annotation T+3, p95/p99, utilization and unit cost | Preferred point sustains the declared service envelope |
| E6 - independent go/no-go | Complete E0-E5 artifacts, unresolved-risk register, security/retention/incident evidence and independent review | Reviewer records the release decision; Robata never self-promotes it |

**Implementation outline**

1. Freeze E0 before selecting thresholds or an operating point. Keep recording-hours and
   camera-hours separate and preserve both source-time ratio and throughput units.
2. Run phase-local tests, grouped quality evaluation, recovery/fault scenarios, repeated
   capacity points, and cost/resource collection under the exact frozen scope.
3. Build the Pareto set without collapsing throughput, quality, latency, reliability, and
   cost into an unsupported scalar. Record dominated points and the reason for preference.
4. Attach E1-E5 supporting artifacts by digest. An unexecuted gate remains
   `NOT_MEASURED`; an executed failure retains its measurement artifact and unresolved
   threshold failure, and never becomes `PRODUCTION_QUALIFIED`.
5. Leave E6 `PENDING_INDEPENDENT_REVIEW` until an independent reviewer makes the decision.

**Keep intact**

Typed evidence status, immutable qualification scope, external gate order, schema/catalog
authority, and the rule that technical evidence cannot self-authorize production release.

**Done when**

- [ ] The local package validates and names every external item still `NOT_MEASURED`.
- [ ] Every selected point cites exact workload, code, policies, hardware, provider, and runs.
- [ ] Quality, capacity, deadlines, recovery, resource, and cost populations cross-check.
- [ ] E0-E5 each has a valid artifact or explicit unresolved state; E6 remains independent.
- [ ] No missed threshold is hidden by relabeling measured evidence as unmeasured.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_benchmark_pareto.py tests/unit/test_benchmark_qualification.py
python -m pytest tests/unit/test_representative_production_qualification.py
python -m pytest tests/unit/test_runtime_capacity.py tests/integration/test_canonical_recovery_qualification_evidence.py
python -m pytest tests/unit/test_schema_immutability.py tests/contract/test_schema_release_policy.py
~~~

**Next boundary:** Independent E6 review and an explicit release decision.

## Blockers and External Dependencies

| Condition | What can still be completed locally | Temporary substitute | Later external proof |
| --- | --- | --- | --- |
| Target NVIDIA codec/GPU matrix | Backend selection, fake-target behavior, fallback, provenance and parity harness | PyAV plus deterministic fake target | E2 codec/profile/resolution/driver matrix on target hardware |
| Real provider and H100 topology | Envelope, replay, timeout/retry, partial-failure and evidence closure | Deterministic offline/mock provider | E3 saturation, latency, error and cost report |
| Production filesystem/object store | Staging barrier, crash points, reconciliation and local fault tests | Local durable filesystem | E2 durability/fault run on deployment storage and R2 when used |
| Governed representative recordings and labels | Deterministic fixtures, split validation and metric calculators | Versioned local/conformance fixtures | E1 blinded QA/event/boundary/calibration review |
| Real encoder, pgvector and tenant policy | Ports, in-memory contract, backfill/retry and isolation fixtures | In-memory vector store/fake encoder | Physical adapter, index and database RLS qualification |
| Representative arrivals and long-run capacity | Capacity harness, bounded overload and recovery scenarios | Short local smoke/profile | E4 soak and E5 500 recording-hours per 24 hours service run |
| Independent release authority | Complete content-addressed E0-E5 package and risk register | None | E6 signed go/no-go decision |

## Acceptance and Verification

- [ ] Every dispatched phase reads its module cards and changes only its named paths/boundary.
- [ ] Focused unit tests cover changed decisions and deterministic replay.
- [ ] A focused integration, crash-recovery, smoke, or profile proves each changed boundary.
- [ ] Published schema, wire, logical identity, exact-byte, semantic projection, and root
      changes use an explicit version/upcaster/migration decision.
- [ ] Performance claims include a P0-like before/after measurement with units and context.
- [ ] Quality claims use governed grouped splits and report abstention/incomplete outcomes.
- [ ] External limits remain `NOT_MEASURED` or unresolved until their named gate runs.
- [ ] Failed measured gates retain evidence and cannot be presented as qualified.
- [ ] The guide, Blueprint, tests, and resulting qualification artifacts agree on terminology.

## Suggested Dispatch Prompt

~~~text
Work on <module-id> / P<n> - <phase name>.

Read AGENTS.md, governance/BLUEPRINT.md, and every governance/modules/<module-id>.md named
by the phase. Implement only that phase's end-to-end result and named contract decision.
Primary paths: <copy the phase change-map paths>.
Preserve: <copy the phase Keep intact boundary>.
Run locally: <copy the phase commands>.
Report: changed files, command results, before/after measurement when claimed, schema or
identity decision, and any external blocker. Do not expand beyond this phase.
~~~
