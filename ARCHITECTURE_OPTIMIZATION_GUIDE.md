# Robata Architecture Optimization Guide

> This guide was reviewed against the current source, tests, published schemas, and
> `schemas/schema-catalog.json`. It is a construction guide, not a product contract,
> benchmark result, or production certificate. Where this guide conflicts with those
> sources, the contract and executable behavior win.

The `P0` / `P1` / `P2` labels express recommended investigation order. They are not
Blueprint phase identifiers, delivery commitments, or evidence grades.

## How to Read This Guide

Each proposal separates four things that the previous draft sometimes conflated:

- **Current implementation**: a fact observable in tracked source or tests.
- **Local observation**: a fixture-backed result useful for regression work.
- **Proposed change**: an implementation direction, still subject to profiling.
- **Contract decision required**: a change to a published wire shape, identity, hash,
  logical key, idempotency key, fence, or semantic projection. Such a change requires
  an explicit version or migration decision before implementation.

Performance improvements and quality targets remain `NOT_MEASURED` until the relevant
benchmark has been run. A projected multiplier is not evidence.

---

## 0. Current Baseline and Measurement Rules

### 0.1 Fresh source profile

The current local profile observation uses a 40.8335-second, six-camera recording fixture,
PyAV CPU media processing, and a deterministic offline provider. The source report date,
commit, and digest are not carried in this guide, so these values are unbound historical
planning observations. The runtime profile family is `LOCAL_CONFORMANCE`,
`NOT_MEASURED`, and `NOT_PRODUCTION_QUALIFIED`; rerun P0 before using
it for a before/after claim.

| Metric | Local observation | Direction |
| --- | ---: | --- |
| Fresh elapsed | 118.750 s | lower is better |
| Fresh source-time ratio | 2.908 wall-sec/source-sec | lower is better |

### 0.2 Actual local-mock smoke

The separate 30-minute fixture-backed smoke uses provider traffic `MOCKED`, execution mode
`ACTUAL_LOCAL_MOCK`, and authority status `AUTHORITATIVE_LOCAL_MOCK_SMOKE`. Its report still
sets `measurement_status=NOT_MEASURED` and `qualification_status=NOT_PRODUCTION_QUALIFIED`;
it is regression evidence, not production capacity evidence.

| Metric | Local observation | Direction |
| --- | ---: | --- |
| Throughput result | 2.162 rec-sec/wall-sec | higher is better |
| Eligible-to-terminal p95 latency | 3.908 s | lower is better |
| Active backlog at end | 0 | must remain bounded |
| No-growing-backlog gate | met | must remain met |
| Work items | 9,001 | workload descriptor |
| Windows | 1,800 | workload descriptor |

Fresh source-time ratio and throughput are different report surfaces and units. They must
not be given the same name or compared as if they were one metric.

### 0.3 Profile attribution

The current Blueprint records the following local profile observations:

| Scope | Wall time | Interpretation |
| --- | ---: | --- |
| `source.prepare` | about 77.38 s | parent envelope; do not add it to its children |
| `source.stream.capture_publish` | about 53.05 s | append, publication, drain, and related work |
| `source.materialize` | about 22.33 s | media materialization path |

These numbers identify investigation areas; they do not prove the causal contribution of
any single function. The profile does not establish an H100, NVDEC, vLLM, R2, long-run,
or production ceiling. Those environments remain `NOT_MEASURED`.

### 0.4 Rules for optimization claims

Compare the same workload before and after every change. Record the commit, dataset,
hardware, provider, configuration, warm/cold state, and repetition count. Report elapsed
time, throughput, latency, errors, abstentions, resource use, and correctness parity.

Target-hardware and real-model gains remain `NOT_MEASURED` until an artifact proves them.
There is no defensible physical ceiling in current evidence because codec, resolution,
sampling, model, batching, topology, and storage all change the achievable rate.

---

## 1. [P0 / Media] Qualify an NVDEC Decode Path

### Current implementation

`src/robata/adapters/nvdec_backend.py`, `nvdec_frame_materializer.py`, and
`nvdec_video_export.py` provide target-backend wrappers and explicit PyAV fallbacks. They
do not implement or qualify an NVIDIA backend. The canonical MCAP source path already
performs recording-scoped work and decodes selected evidence once per camera.
`PyAvFrameMaterializer` is a separate package path, while `pyav_decoder.py` is a probe.

### Recommended path

Implement a real backend behind the existing interface. Define codec, profile, resolution,
and concurrency support from the target SKU instead of assuming a fixed engine count. Keep
fallback explicit, observable, and before dependent publication. Put media provenance in a
media-specific record rather than silently repurposing inference `CapabilitySnapshot`.

### Constraints and proof

`H264SpoolFacts.sha256` identifies derived spool serialization, not the whole MCAP.
Derived frame hashes enter package and inference input projections, so changed decode bytes
can change downstream identity. Either prove exact CPU/GPU derived-byte parity or version
affected policies and semantic projections. Require codec, timestamp, frame, malformed
input, fallback/restart, and target-hardware evidence. Performance remains `NOT_MEASURED`.

---

## 2. [P0 / Inference] Qualify Provider Batching

### Current implementation

The generic `InferenceOrchestrator` defaults to `max_batch_size=1`, but the canonical
offline pipeline defaults client microbatching to 8 items with a 5 ms queue delay. Local
canonical composition also pins 6 concurrent call parts. `RunPodEndpointConfig` still
defaults native batching off with maximum size one. Client microbatching, a RunPod native
envelope, and server-side continuous scheduling are different mechanisms. Recorded
transport replays exact HTTP request bytes; it does not automatically split a native batch
into per-item recordings.

### Recommended path

First benchmark the existing canonical 8-item/5 ms client microbatch and 6-call-part
concurrency baseline, including an explicit single-request control. Enable native batches
only for handlers that support the exact envelope. Benchmark queue delay, batch size,
endpoint concurrency, provider limits, latency, failure isolation, and cost. Evaluate
prefix caching only for the actual server, prompt layout, model, and generation
configuration.

### Constraints and proof

`InferenceInputPlan.input_plan_id` is opaque, not the semantic formula previously stated
here. Transport grouping is operational only when each logical invocation retains its own
request, result, failure, selection, and accepted evidence. Test exact replay, partial
failure, ordering, duplicates, retries, and parity with batching disabled. Real-model
throughput and cache gains remain `NOT_MEASURED`.

---

## 3. [P0 / Architecture] Pipeline Ready Work from Durable State

### Current implementation

The durable per-window DAG has five stages:

```text
WINDOW -> QA_COARSE -> QA_DENSE -> EVENT_PROPOSAL -> WINDOW_REDUCTION
```

`EVENT_PROPOSAL` also depends on coarse and dense QA; reduction depends on all earlier
window stages. There is no declared dependency from window N reduction to window N+1.
The local-conformance finalizer can drain synchronously and write deterministic stage
receipts. Those receipts are not provider-neutral `ModelInference` evidence and must not
stand in for production-provider profiling. When the provider-neutral executor is
configured, QA and event stages persist immutable `ModelInference` bytes. `WINDOW` is not
the canonical MCAP decode stage.

### Recommended path

Instrument queue wait, execution time, and transaction time for actual executors. Allow
stage-affine pools to claim ready durable work concurrently. Bounded in-memory channels may
transport claimed work, but durable plans, leases, fences, pending terminals, accepted
terminals, and closure remain authoritative.

### Constraints and proof

Preserve the per-window graph, exact replay, EOS sealing, and finalization. Do not invent
cross-window order or hold SQLite transactions across provider or media work. Verify
equivalence at concurrency one and above, randomized completion, lease expiry,
crash/restart, duplicate delivery, EOS races, and bounded backlog. No pipeline speedup is
currently measured.

---

## 4. [P0 / Correctness] Profile SQLite Ownership Boundaries

### Current implementation

`SQLiteWorkScheduler._transaction` opens a connection per transaction.
`SQLiteInferenceEvidenceLedger` instead owns a lifetime connection guarded by an `RLock`.
The adapters therefore should not receive one generic pooling prescription. General stream
completion deliberately persists pending intent, execution terminal, and acceptance across
authority boundaries; it is not one SQLite transaction. Specialized window reduction
already has a same-database atomic publication path.

### Recommended path

Measure connection setup, lock wait, transaction time, fsync, rows, and retries per
operation. Consider residency only with explicit thread ownership or a writer queue. A
single writer may reduce churn but still serializes writes. Batch only within one database
authority and acknowledge only after commit. Test in-process and multi-process contention
before changing `busy_timeout`.

### Constraints and proof

Preserve pending/succeed/accept recovery, row-version compare-and-swap, leases, fences, and
exact replay. Never span provider or media work with a database transaction. Use crash
injection at each commit boundary and measure any new strategy. Benefits remain
`NOT_MEASURED`.

---

## 5. [P1 / Media] Reuse Recording-Scoped Media Work

### Current implementation

`BoundedMediaPolicy` uses a 2-second window and 1-second hop, so adjacent time windows
overlap. That does not prove duplicate decoding. The canonical MCAP path already decodes
selected evidence once per camera at recording scope, and package materialization can bind
existing artifacts. The previous per-window duplicate-decode model was incorrect.

`src/robata/frame_cache.py` stores content-addressed encoded bytes and feed manifests. It
is not a decoded GPU-tensor LRU.

### Recommended path

Profile decode, selection, encode, lookup, and package binding separately. Remove any
proven duplication at its earliest boundary; prefer a single-pass selector over caching the
result of repeated full-video decodes. Keep raw frame, encoded artifact, and manifest
reuse as separate designs with explicit lifetime and eviction.

### Constraints and proof

Define equivalence per layer. A raw decoded-frame cache binds source bytes, timestamp,
decoder, pixel format, and decode/transform policy; it does not require an encoder. An
encoded-artifact cache additionally binds the encoder policy and exact bytes. Manifest
reuse binds member identities, ordering, and package policy. Source identity may stay fixed
while package and inference identity change. Measure duplication, hit rate, memory,
cleanup, parity, restart, and corruption. Temporal overlap alone does not prove a 2x gain.

---

## 6. [P1 / Media] Evaluate Alternate Frame Encodings

### Current implementation

`PyAvFrameMaterializer._encode_png` opens the PyAV PNG encoder for writing. Canonical MCAP
and supplemental QA policies are PNG-specific, emit `image/png`, and validate PNG bytes.
The current 320-pixel maximum evidence width also makes unrelated 1080p figures
inapplicable. JPEG is lossy;
a quality setting of 90 is not lossless, and its effect on Robata outputs is unmeasured.

### Recommended path

Add an experimental encoding policy rather than replacing the default. Pin the encoder
implementation and version, quality, chroma subsampling, resize, color conversion, and
metadata. Verify each provider and exact request replay. Benchmark actual selected-frame
dimensions and labels.

### Constraints and proof

Generic schemas accept MIME-shaped media types, so `image/jpeg` alone may not require a
new generic artifact schema. However, PNG-specific policies and consumers must change,
and bytes, media type, package content, and input plans affect downstream identity. Require
an explicit policy/version decision, consumer updates, provider conformance, and per-class
QA/event/boundary parity. Speed, size, and quality gains remain `NOT_MEASURED`.

---

## 7. [P1 / I/O] Batch Artifact Durability Safely

### Current implementation

`PyAvFrameMaterializer._write_new_file` writes bytes, flushes, and calls `os.fsync()`
before returning. Exact-byte hashing already uses in-memory encoded bytes; there is no
read-after-write hash problem. Frame files do not map one-to-one to pending stream work,
so the stream terminal protocol cannot prove that an unflushed frame will be regenerated.

### Recommended path

Use this publication protocol: write under staging names; verify byte count and hash;
flush and `fsync` every file in the publication unit; sync the staging directory; publish
with an atomic rename; sync the destination parent directory; and only then commit database
authority that references the publication. Bound batches by memory and recovery cost. Use
tmpfs only for explicitly transient staging.

### Constraints and proof

A committed terminal or manifest must not reference bytes that can disappear under the
promised durability model. Hash equality proves content, not persistence. Recovery must
distinguish unpublished staging, durable publication, and corruption. Test crashes at each
step, restart reconciliation, filesystem rename/directory semantics, and storage faults.
I/O savings remain `NOT_MEASURED`.

---

## 8. [P1 / Quality] Qualify Confidence Calibration

### Current implementation

The repository defines a `CalibratedConfidence` contract and ECE/Brier calculators.
`RecordingQAResult` currently sets `calibrated_probability=None` as a downstream hook;
there is no production construction of `CalibratedConfidence` or calibration-artifact
lineage. The published Product QA branch uses `ProductQAIssueEvidence` and
`ProductQAConfidenceKind`, whose current values explicitly make no calibration claim.
Product QA is multi-label, not one 21-way softmax. `InferenceAttemptSelection` chooses
one attempt and is not an ensemble contract.

### Recommended path

For each score family, determine whether logits or only scalar scores are available. Fit
an appropriate per-class or shared method on a held-out, leakage-safe calibration split.
Persist method, parameters, training-data lineage, artifact digest, model/runtime pins, and
applicability. Evaluate conformal methods only with their exchangeability and coverage
assumptions stated for the multi-label problem. Decide whether Product QA only references
existing inference calibration lineage or exposes a new calibrated kind/field. The latter
requires a new published Product QA/detail schema and semantic projection version.

### Constraints and proof

Calibration can change QA decisions and therefore policy and semantic outputs. Use the
existing product states such as `OBSERVED`, `NO_ISSUE`, `ABSTAINED`, and
`INCOMPLETE_INPUT`; do not invent enum values. Temperature scaling does not guarantee an
ECE threshold, and conformal coverage does not prohibit false positives. Require per-class
reliability, Brier/ECE, abstention, subgroup, temporal, and drift reports. All quality
improvement claims remain `NOT_MEASURED`.

---

## 9. [P1 / Quality] Design Cross-Window Event Association

### Current implementation

Candidate fusion and boundary refinement are primarily window-scoped. V3/V4 recording
reduction already merges same-label touching or overlapping intervals into
`LocalStreamMergedHypothesis`, retaining source ordinals and proposal digests. It does not
perform confidence-bearing trajectory association across label changes or gaps. Adding an
`EVENT_TRACKING` value to published `StreamStage` would be a wire and DAG contract change.

### Recommended path

Start with a recording-level derived projection over accepted window results. Associate by
label, time, camera/evidence overlap, and explicit confidence; preserve source candidates
and explain every merge or split. Decide separately whether association belongs before
authoritative finalization or after completion as an asynchronous projection.

### Constraints and proof

Do not preselect `candidate-event-v2`: that namespace is already used internally. Any new
published track wire or stage needs a catalog/version decision; any identity or semantic
projection change needs an explicit version/migration decision. Purely internal types still
need local contract and replay tests but do not automatically enter the catalog. Test
overlapping windows, long actions, label changes, gaps, duplicate candidates, replay, and
association ambiguity. Measure event-level precision/recall and boundary quality on
representative labels. Quality gains remain `NOT_MEASURED`.

---

## 10. [P1 / Quality] Add an Active-Learning Selection Boundary

### Current implementation

`src/robata/application/canonical/local_review_routing.py` does not treat every clip
equally. It already assigns different triggers and priorities for QA degradation, low
confidence, and sampling. It routes one committed completion at a time; it is not a
pool-level top-k ranker. Review contracts and benchmark ground-truth structures also serve
different ownership boundaries.

### Recommended path

Add a separate batch selector over eligible review tasks. Use versioned uncertainty,
disagreement, coverage, diversity, and sampling terms, while retaining existing trigger
priority. Store immutable selection decisions and annotation lineage. Train candidates
offline and require explicit promotion; never mutate historical routes.

### Constraints and proof

Keep review nonblocking for primary truth. Separate training, calibration, and frozen
evaluation splits; prevent recording/camera/time leakage and monitor selection bias. A new
model must not become active merely because annotations arrived. Measure label yield,
coverage, subgroup balance, agreement, and held-out quality. A feedback-loop benefit is
not assumed and remains `NOT_MEASURED`.

---

## 11. [P2 / Architecture] Make Backpressure Adaptive Only After Measurement

### Current implementation

`BackpressureController` classifies supplied `QueueMetrics`. When
`backpressure_snapshot()` receives no metrics, the scheduler sets arrival rate, service
rate, backlog slope, and provider quota to zero. No monitoring path automatically fills
them, and no current `max_active_windows` control state exists.

### Recommended path

First supply measured rates, backlog age/slope, quota, utilization, and admission decisions
through an explicit observable boundary. Then evaluate fixed limits against an adaptive
controller. If AIMD is selected, define owner, persistence, restart value, clock behavior,
fairness across recordings, provider quotas, minimum/maximum, and anti-oscillation rules.

### Constraints and proof

Backpressure is operational only while it changes timing/concurrency. If it changes
sampling or semantics, it enters policy and identity. Record decisions without treating
runtime timing as logical identity. Test recovery, multiple processes, overload, quota
changes, fairness, oscillation, and drain. Constants must come from stability and capacity
profiles, not arbitrary increments. Impact remains `NOT_MEASURED`.

---

## 12. [P2 / Retrieval] Qualify Existing Vector Retrieval

### Current implementation

`RetrievalService` already supports an optional `VectorProjectionStore`: it obtains a
bounded structured candidate set and then applies vector scoring. Versioned vector
projection/search contracts, ports, and deterministic local implementations already
exist. Lexical search remains the offline default. External encoder/database behavior is
optional and not production-qualified.

### Recommended path

Qualify a real encoder and pgvector adapter rather than redesigning retrieval from zero.
Pin model, revision, dimension, normalization, distance metric, index configuration, and
tenant policy. Keep `EventIndex` and structured filters authoritative; vector rows are
asynchronous derived projections keyed to immutable event revisions.

### Constraints and proof

Projection failure must not corrupt structured retrieval. Backfill and replay must be
idempotent under the existing versioned keys, and stale revisions must remain distinguishable.
Test tenant isolation, missing vectors, adapter failure, duplicate/conflict behavior,
pagination, deterministic tie-breaking, backfill, and revision changes. Benchmark recall,
latency, selectivity, index build, and resource use on representative queries. Previous
fixed recall and millisecond projections are removed; results remain `NOT_MEASURED`.

---

## 13. [P0 / Architecture] Profile Existing Transaction Batches

### Current implementation

The earlier transaction count was stale. Current code already batches window and stage-plan
append, uses `plan_many` for execution projection, and uses `mark_published_many` for
publication. Window reduction also has a specialized atomic same-database delivery path.
Replay uses exact conflict checks rather than a blanket `INSERT OR IGNORE` rule.

### Recommended path

Use operation telemetry to find the remaining high-count or long transactions. Batch only
adjacent operations with the same authority, recovery semantics, and failure outcome.
Prefer targeted reads and prepared statements where profiles justify them. Sections 4 and
13 should be implemented as one measured SQLite workstream.

### Constraints and proof

Do not remove pending terminal intent or merge authority boundaries to reduce a counter.
Never execute provider calls inside a transaction. Preserve exact replay conflicts,
readiness derivation, crash recovery, and atomic window-reduction publication. Update
transaction-count tests and compare the same fixture before and after. The former per-window
counts and elapsed-time savings are invalid; new impact remains `NOT_MEASURED`.

---

## 14. [P0 / Quality] Make Adaptive Sampling Causally Explicit

### Current implementation

`AdaptiveUpgradeReason` already includes coarse uncertainty, cross-camera disagreement,
event candidate, and boundary refinement. `AdaptiveCoveragePolicy` already carries a
version and bounded limits. The missing question is whether model feedback is available
before the relevant plan is sealed and how that decision is replayed.

### Recommended path

Define a derived upgrade decision that binds the accepted upstream evidence, base plan,
policy, budget, and selected additional timestamps. Persist it before executing additional
work. On replay, consume accepted evidence and the stored decision rather than relying on a
new stochastic provider response. Retain base safety coverage because late feedback cannot
recover every fast event.

### Constraints and proof

Do not mutate a policy version at runtime. If feedback changes an input plan, its causal
evidence must enter the new plan identity. Any new wire fields, reason vocabulary, or
semantic projection need a version decision. Test budgets, duplicate decisions, restart,
out-of-order evidence, provider abstention, deterministic replay, and no-loss base coverage.
Quality/compute improvement remains `NOT_MEASURED`.

---

## 15. [P1 / Quality] Qualify Quality-Aware Boundary Estimation

### Current implementation

Canonical boundary refinement currently uses a deterministic median-low center, a maximum
uncertainty envelope, and a minimum observed-camera rule. Camera QA scores are not
automatically calibrated estimates of geometric boundary reliability. The previous
weighted-maximum formula could reduce an uncertainty bound after down-weighting a poor
camera and was therefore not conservative.

### Recommended path

Evaluate robust estimators and an explicit camera-exclusion policy using representative
boundary annotations. Preserve every raw per-camera observation, exclusion reason, quality
input, and aggregate result. If quality evidence is missing or inapplicable, retain the
current estimator and record that weighting was not applied.

### Constraints and proof

Changing only the formula still changes result semantics and identities. Bind the estimator
and quality inputs in a versioned policy; make a schema decision for any new wire fields.
Test missing/degraded cameras, outliers, ties, contradictory observations, uncertainty
coverage, replay, and monotonic safety properties. Report onset/offset error and coverage
by class and camera condition. Arbitrary fixed weights and quality gains are removed.

---

## 16. [P1 / Correctness] Optimize Proven No-Work Paths

### Current implementation

Video quality and event presence are different semantics. All cameras being `GOOD` does
not imply that no action occurred. `SKIPPED_POLICY` being acceptable to the generic work
dependency scheduler does not make it domain evidence for `NO_EVENTS`. Dense QA can already
produce explicit not-needed behavior in qualified coarse-complete cases, but downstream
event and completion contracts still require their domain results.

### Recommended path

Eliminate only orchestration that is already proven to be a semantic no-op, such as a
provider call replaced by an existing explicit no-work result. Keep event proposal unless
a separately trained and qualified event-presence gate meets a stated recall bound and has
versioned incomplete/skip semantics. Never use QA cleanliness as that gate.

### Constraints and proof

Every skipped work item needs explicit terminal evidence and a downstream mapping that
cannot be confused with evaluated `NO_EVENTS`. A lossy gate requires policy, identity,
wire, completion, and migration decisions. Test false-negative events in clear video,
degraded input, dependency reduction, replay, and completion validation. Measure recall
before compute savings. The earlier skip-rate and call-count estimates are removed.

---

## 17. [P1 / Inference] Keep Reuse Within Valid Invocation Identity

### Current implementation

`InferenceInputPlan.input_plan_id` is caller-supplied opaque identity; the plan has a
separate semantic digest. A selection and terminal must share the same
`logical_invocation_id`, policy, and full semantic binding. A new window cannot point its
selection at an old invocation terminal. Overlapping frames also do not imply identical
plans because timestamps, artifacts, request catalog, capability, prompt, and policy matter.

### Recommended path

First use existing idempotency and replay within one logical invocation. Measure exact
semantic duplicate frequency before proposing broader reuse. Cross-invocation reuse would
need a new immutable cache-reference or derivation record that states which prior response
was reused, why equivalence holds, and how current evidence closes without pretending the
old terminal belongs to the new invocation.

### Constraints and proof

Any cache key must bind full provider/model/runtime/generation, prompt, schema, media, and
policy semantics, including stochastic behavior. Storage location must not become identity.
Test changed timestamps, one-frame differences, model revisions, seeds, policies, expiry,
revocation, corrupt entries, and concurrent population. The prior 30-50 percent hit-rate
claim and simple-index design are invalid. Impact remains `NOT_MEASURED`.

---

## 18. [P2 / Completion] Optimize Root Construction Without Mutable Completion

### Current implementation

`PrimaryCompletionRecord` v3 contains eleven required count/root pairs, not nine. Its
semantic digest is validated synchronously. The `primary_completions` table has no generic
`row_version` backfill mechanism and forbids update/delete. Roots are built from the
in-memory completion detail, not by rereading every row from the database.

### Recommended path

Profile ordering, serialization, hashing, schema validation, artifact write, and database
commit independently. If root construction is material, spool leaf digests while
authoritative collections are built, preserving each collection's exact existing canonical
order. External sorting is valid only by that collection's existing canonical key, never by
digest bytes. The final v3 call must still pass the complete ordered digest list to
`canonical_collection_digest_root`. A tree/subtree-root algorithm would require a new
semantic projection and completion version.

### Constraints and proof

The final v3 record must remain fully populated, schema-valid, semantically hashed, and
atomically published. Placeholder roots followed by update are prohibited. A pre-completion
or revision model would require a new wire/identity/migration design and must not be called
v3 completion. Test zero/large collections, ordering, duplicate digests, restart, artifact
failure, and schema rejection. The previous O(1) and latency claims are removed.

---

## 19. [P2 / Quality] Add Temporal Consistency as a Derived Signal

### Current implementation

Per-window evidence can be compared after reduction, but no authoritative trajectory-level
consistency report exists. `CanonicalPrimaryCompletionDetail` is a published v4 wire shape,
and its semantic projection includes its detail fields. Adding a report there changes wire
and semantic identity and makes completion wait for the report.

### Recommended path

For a nonblocking feature, publish a separate asynchronous derived report linked to the
completed run and exact source window evidence. Include policy, association confidence,
quality transitions, gaps, and raw comparisons. If the report must govern primary
completion, design a new detail version and accept that it is blocking and identity-bearing.

### Constraints and proof

Do not convert low continuity into `INCOMPLETE_INPUT` or add ambiguity codes without a
versioned policy and contract decision. Cross-window event comparison also depends on an
association model from section 9. Test overlap, missing windows, real quality transitions,
long events, replay, late arrival, and policy revisions. Measure review yield and false
alerts on representative recordings. Benefit remains `NOT_MEASURED`.

---

## Appendix A: Contract and Correctness Checklist

Every implementation phase must answer these questions before code changes:

| Area | Required decision |
| --- | --- |
| Published schema | Is a new version/upcaster/migration required rather than an in-place edit? |
| Identity and hashes | Which source, derived, logical, semantic, and exact-byte identities change? |
| Idempotency and fences | Are replay, duplicate conflict, lease, and partition ownership preserved? |
| Artifacts | Can a committed reference ever point to missing or non-durable bytes? |
| Inference | Does intent-to-raw-to-parsed-to-selection-to-accepted lineage still close? |
| Stream work | Are plans, readiness, pending terminals, acceptance, EOS, and recovery durable? |
| Completion | Is the published record complete, immutable, schema-valid, and hashed before release? |
| Evidence | Does the report state workload, hardware, provider, status, and unresolved gaps? |

Additional rules:

- Original source bytes and provenance are never rewritten.
- Derived media must bind source/time, exact bytes, and every semantics-affecting policy.
- Transport grouping, raw credentials, storage locators, and runtime timing do not become
  logical identity. If account or capability changes semantics, bind a stable non-secret
  provider/account/capability identifier instead.
- Current qualification gateways must continue to reject production claims until the
  required evidence path is implemented and externally satisfied. `production_eligible`
  is not universally hardcoded false and must not be described that way.

---

## Appendix B: Evidence and Status Vocabulary

Use the typed vocabulary of the report being produced; not every report shares one global
enum. In particular, `NOT_MEASURED` is a measurement status, not proof of a benchmark.

| Term | Interpretation in this guide |
| --- | --- |
| `LOCAL_CONFORMANCE` | Local-only conformance scope; not representative or production proof |
| `SYNTHETIC_LOCAL` | A simulated/local result; useful for regressions only |
| `VIRTUAL_MODEL_DIAGNOSTIC` | A virtual estimate; non-authoritative and not hardware capacity |
| `NOT_MEASURED` | Required representative measurement has not been performed |
| `MEASURED` | A result is bound to a declared workload context; scope still matters |
| `NOT_PRODUCTION_QUALIFIED` | Production promotion gates have not passed |
| `PRODUCTION_QUALIFIED` | Allowed only by the governed external promotion path |

Do not relabel a `VIRTUAL_MODEL_DIAGNOSTIC` estimate as local hardware capacity. A measured
microbenchmark also cannot promote end-to-end production eligibility by itself.

---

## Appendix C: Dependency-Ordered Roadmap

This sequence has no calendar or guaranteed throughput multiplier.

1. **Freeze measurement semantics.** Preserve the fresh source-time ratio
   (wall-sec/source-sec) and throughput (rec-sec/wall-sec) as separate units; bind profiles
   to code, workload, hardware, provider, and configuration.
2. **Profile current hot paths.** Re-measure media, stream transactions, inference evidence,
   completion, artifact durability, and retrieval before selecting implementations.
3. **Protect correctness boundaries.** Resolve sections 4, 7, 13, 16, 17, and 18 before
   accepting performance changes that depend on weakened recovery or identity.
4. **Qualify physical adapters.** Exercise NVDEC, alternate encoding, provider batching,
   target storage, and vector adapters independently, then in the canonical path.
5. **Add concurrency from durable readiness.** Tune stage/executor pools and backpressure
   only after target-provider and SQLite limits are measured.
6. **Develop quality changes with governed labels.** Calibrate, associate events, adapt
   sampling, estimate boundaries, select review work, and score temporal consistency with
   leakage-safe representative data.
7. **Run external gates E0-E6.** Freeze evidence; sign off quality; qualify media/storage
   and provider topology; run reliability/soak; prove the 500 recording-hours per 24 hours
   SLA/cost envelope; then obtain independent go/no-go review.

An unexecuted gate remains `NOT_MEASURED`. An executed gate that misses a threshold keeps
its measured evidence and records the gate/threshold failure. In both cases the result
remains `NOT_PRODUCTION_QUALIFIED`; failure never justifies relaxing the invariant.

---

## Appendix D: Qualification Measurement Matrix

The previous industry-comparison table mixed unrelated workloads and unsupported targets.
Use this workload-bound matrix instead:

| Area | Minimum measured outputs | Required context |
| --- | --- | --- |
| Media | source fps, selected fps, latency, CPU/GPU/memory, parity and fallback | codec, profile, resolution, GOP, camera count, driver/SDK, policy |
| Inference | calls/images/tokens, queue time, p50/p95/p99, error/abstention, cost | model revision, endpoint, prompt, generation, batch/concurrency |
| Stream/SQLite | transactions by operation, lock wait, backlog, lease/retry/recovery | database layout, process/thread count, storage, arrival pattern |
| Artifacts | encode/write/fsync time, bytes, reconciliation and loss/corruption | filesystem/object store, durability promise, publication unit |
| QA/events | per-class metrics, calibration, abstention, boundary and temporal metrics | governed labels, splits, prevalence, subgroup and camera coverage |
| Retrieval | structured selectivity, recall at k, latency, build/backfill, isolation | query set, embedding revision, index parameters, tenant policy |
| End-to-end | elapsed, fresh source-time ratio (wall-sec/source-sec), throughput (rec-sec/wall-sec), p50/p95/p99, backlog, cost | frozen E0 envelope and representative arrival/failure scenarios |

Current H100, NVDEC, vLLM, pgvector, R2, 24-hour soak, quality, and production capacity
results are `NOT_MEASURED`. This guide intentionally states no external product comparison,
single-GPU upper bound, or guaranteed quality threshold.
