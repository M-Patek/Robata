# Streaming Throughput Next-Iteration Guide V1

- Status: non-normative local engineering plan
- Date: 2026-07-22
- Governing authority: Architecture V1.1 Sections 17, 19, 20, and normative Section 25;
  the execution-spec overlay; accepted ADRs; and registered schemas
- Evidence boundary: current facts are `LOCAL_CONFORMANCE`; timings are candidate-local
  observations; projections and vendor figures are non-certifying; `production_eligible=false`
- Supersedes: none

## 1. Authority and purpose

This guide translates the next throughput iteration into executable work. It is subordinate to:

1. the [product execution specification](../../large_scale_6camera_video_agent_execution_spec.md);
2. normative [Architecture V1.1](../../ARCHITECTURE_DESIGN_V1.md), especially Section 25;
3. the [execution-spec overlay](execution-spec-v1-overlay.md);
4. accepted ADRs and exact-pinned registered schemas; and
5. the repository [implementation plan](../../IMPLEMENTATION_PLAN.md).

It does not change a phase gate, approve a policy, select production infrastructure, or make a
model-quality or production-capacity claim. A conflict with an authority above must be resolved by
an architecture revision or ADR before implementation. New Wire contracts and semantic identity
formulas require versioning and publication through the existing atomic Schema registration path.

The goal is to replace the current whole-recording execution shape with a continuous, bounded,
replayable local execution shape while preserving the existing canonical final result. The target
experience is continuous input and seconds-scale incremental non-final evidence, followed by an
end-of-stream finalizer that produces recording-scoped canonical truth.

## 2. Evidence labels used here

Quantitative statements in this guide use four descriptive categories. They do not add new
repository status enums:

| Label | Meaning |
|---|---|
| Candidate-local observation | Observed on the 2026-07-22 local candidate and authorized sample |
| Derived projection | Arithmetic extrapolation from stated observations and assumptions |
| External vendor benchmark | Official result for a vendor-specific model and configuration |
| Unvalidated target | Proposed acceptance threshold that has not passed qualification |

Formal results continue to use existing evidence classes such as `LOCAL_CONFORMANCE`,
`SYNTHETIC_LOCAL`, and `NOT_MEASURED`.

## 3. Starting point and measured baseline

The current raw-MCAP command is a connected local-conformance batch chain:

```text
whole MCAP
  -> full inspection and source identity
  -> six registered MP4 exports and validation
  -> full frame index and selected PNGs
  -> QA, proposal, candidate, action, boundary, and fusion calls
  -> primary completion, identity, revision, and outbox
  -> relay, reconciliation, and nonblocking review
```

The chain is complete for its stated batch scope, but it is not a continuous stream processor.
The composition obtains whole-source facts before creating one root window, the media bridge scans
and exports the complete source, and the canonical runner advances one recording-level stage chain.
The durable scheduler currently owns `ACTION_PUBLISH`, not the full inference DAG. The live facts
and Section 25 status remain owned by the
[current implementation status](../current-implementation-status.md).

The candidate-local baseline used one 40.890455-second six-camera MCAP with 7,350 observations and
130,303,923 input bytes. The documented fresh run was 853.8 seconds with 16 deterministic
fixture-backed inference calls and no network calls. The documented exact replay was 26 seconds
with no redispatch. The current state tree is approximately 452.9 MB; the state tree plus the
original source is approximately 583.2 MB.

| Baseline quantity | Value | Interpretation |
|---|---:|---|
| Aggregate input frame rate | 179.75 frames/s | Approximately six cameras at 30 frames/s |
| Aggregate compressed input | 25.49 Mbit/s | 10.68 GiB per recording hour |
| Fresh end-to-end effective rate | 8.61 observations/s | 20.88 times slower than source time |
| Fresh capacity | 0.04789 recording-h/wall-h | Local fixture-backed conformance only |
| Current equivalent workers at 70% utilization | 29.83 per live six-camera group | Linear projection, not deployment sizing |
| Materialized state footprint | 37.13 GiB per recording hour | About 3.48 times the external source bytes; not measured disk write I/O |
| Source plus materialized state | 47.82 GiB per recording hour | About 1.12 TiB for one continuously active day |

File timestamps provide only a preliminary decomposition: source preparation was about 151
seconds, the 16-call inference/evidence/persistence/validation interval was about 672 seconds, and
finalization was about 17 seconds. These are not instrumented spans. In particular, the 672-second
interval must not be described as pure SQLite time: it includes mock inference orchestration,
canonical serialization, hashing, model validation, Schema validation, and durable evidence work.

Even deleting that entire 672-second interval would leave about 181.8 seconds, or 4.45 times source
time. The problem is structural as well as hardware-dependent. The next iteration must measure and
change both the media path and the evidence/inference path.

## 4. Enterprise pattern findings

The transferable enterprise pattern is a long-lived, bounded data plane, not a specific vendor
product:

- decode each compressed input once and retain frames in device or pipeline memory;
- batch compatible work across synchronized streams and independent recordings;
- separate latency-critical media/inference flow from durable evidence and audit work;
- put small identities, manifests, and object references on the message bus, not video bytes;
- use event time, watermarks, explicit allowed lateness, and idle-source handling;
- bound every queue and expose queue age, pressure, drops, and policy-driven degradation;
- commit business facts and delivery rows in one applicable metadata transaction; and
- use deterministic IDs and unique constraints so at-least-once delivery is harmless.

NVIDIA DeepStream implements a long-lived GStreamer decode, batching, inference, and metadata graph.
`nvstreammux` forms multi-source batches and sends a full or timed-out partial batch downstream.
GStreamer `queue` and `tee` isolate branches and provide bounded buffer, byte, and time limits.
Triton dynamic batching trades a bounded queue delay for higher model throughput.

Kafka partitioning provides ordered parallel metadata streams, but video belongs in immutable
object storage. Kafka's default producer request limit is 1,048,576 bytes; this sample's average
two-second compressed chunk for one camera is already about 1.06 MB before keyframe peaks. Flink's
event-time and watermark model provides the relevant semantics for multi-camera late and idle
inputs. Debezium's outbox pattern provides the target delivery shape, but consumers still require
idempotency because CDC delivery is not a substitute for the primary database transaction.

These mechanisms can be implemented behind Robata ports. DeepStream, Triton, Kafka, Flink,
PostgreSQL, and an external object store are candidate production adapters, not dependencies that
must be installed to complete the first local iteration.

The execution-spec overlay still fixes MCAP-to-six-MP4 delivery as FX-002. Throughput work moves
that derived-artifact branch out of the incremental analysis critical path; it does not delete the
six-record `CameraVideoExportManifest` product requirement.

## 5. Invariant constraints

1. Existing recording-scoped canonical identity remains final truth. It is not weakened to make
   early streaming output easier.
2. Window output is explicitly incremental and non-final. It uses a new versioned semantic
   projection and never reuses or impersonates an existing primary-completion or ActionEvent Wire
   contract.
3. Execution strategy fields such as worker count, queue depth, and GPU assignment remain outside
   semantic identity unless they change the logical evidence or policy result.
4. Large immutable bytes are committed to content-addressed storage before a metadata transaction.
   The transaction stores exact digest, byte count, media type, artifact ID, and Schema pins.
5. A database row cannot reference an object that has not completed digest verification. A crash
   after object publication but before metadata commit creates an unreferenced object for delayed
   garbage collection, not partial business truth.
6. A broker is a delivery projection. It is never the authority for primary completion.
7. Required truth is never silently dropped. Optional sampling and enhancement work can be reduced
   only by a versioned, observable policy that preserves the configured quality floor.
8. The entire current 16-call graph is not run for every one-second hop.
9. Full MP4 export, a complete PNG set, and a second decode are not unconditional prerequisites for
   incremental analysis. The fixed six-MP4 derived artifact remains an asynchronous recording-level
   deliverable and a final product-completion gate.
10. Local and mocked results remain `LOCAL_CONFORMANCE`, `NOT_MEASURED` where applicable, and
    `production_eligible=false`.

## 6. Target two-plane architecture

```text
MCAP, file tail, or six live compressed sources
  -> StreamingMediaPlane
       single-pass demux
       incremental exact hash/index
       per-camera bounded encoded ring
       low-cost quality observations
       |
       +-> AsyncSixCameraExport
       |    one pass or encoded spool, no second source traversal
       |    six immutable MP4 records -> export-manifest barrier
       |
       +-> immutable segment manifests and verified content-addressed bytes
            -> WindowCoordinator
                 event time + watermark + idle-source handling
                 W=2s, H=1s, initial allowed lateness=300ms
                 explicit complete/degraded/incomplete closure
            -> DurableWindowPlanner
                 quality -> coarse -> conditional dense/action/boundary -> reduction
            -> bounded media and inference worker pools
            -> IncrementalWindowEvidenceRepository
                 verified evidence refs
                 non-final window evidence
                 durable work terminalization
                 incremental delivery row
                 one local metadata transaction
            -> asynchronous relay and nonblocking review routing
            -> ExpectedWindowPlan
                 append each declaration before child publication
                 seal at EOS from source facts and policies, never execution outcomes
            -> WindowTerminalClosure and complete expected-set barrier
                 reconcile every sealed member to one terminal outcome and evidence reference
                 -> recording QA aggregation
                 -> cross-window proposal merge and overlap deduplication
                 -> recording candidate/action/boundary closure
                 -> ordered hypotheses, membership proof, and output decision
                 -> RecordingFinalizer at end of stream
                      waits for the terminal-closure and export-manifest barriers
                      final source digest and alignment closure
                      existing recording-level canonical completion and primary outbox semantics
                      reconciliation of incremental subjects to final identities
```

The names in this diagram describe ownership boundaries, not pre-approved Python or Wire names.
Implementation should extend existing ports and models where their semantics already match rather
than introduce parallel copies of scheduler, artifact, inference, or completion concepts.

### 6.1 Replaceable media and inference plane

The application layer should depend on a provider-neutral streaming media port. Two adapters can
then share the same contracts:

| Concern | First local adapter | Optional accelerated adapter |
|---|---|---|
| Ingest/demux | PyAV and current MCAP readers | GStreamer/DeepStream |
| Decode/materialize | bounded CPU workers | NVDEC and device-memory surfaces |
| Batch inference | fixture/mock batch adapter | Triton or native TensorRT |
| Immutable bytes | local content-addressed files | object store |
| Metadata transaction | SQLite | production relational database |
| Work delivery | existing SQLite scheduler | selected broker/worker topology |
| Final primary outbox | existing local relay/sink | CDC or production relay |

The first implementation does not install the accelerated stack merely to prove the port shape.
The local PyAV path is the deterministic correctness baseline. An NVIDIA adapter becomes useful
after stage instrumentation identifies a remaining critical path that can benefit from hardware
decode, device-memory processing, or dynamic batching.

### 6.2 Single-pass ingest and encoded ring

One source read should produce all of the following in a single ordered traversal:

- incremental source and chunk digests;
- per-camera packet/access-unit indexes and timestamp facts;
- mapping and alignment observations;
- a bounded compressed ring for neighbor and boundary context;
- low-resolution quality observations; and
- immutable segment manifests when the segmentation policy closes a chunk.

The initial logical chunk interval is one second. Physical compressed chunks may be keyframe-aligned
or represented as ranges in an append-only spool; the exact choice is a versioned segmentation
policy. It does not trigger transcoding solely to satisfy a one-second logical boundary.

At the observed 25.49 Mbit/s aggregate bitrate, a 10-second compressed ring is about 31.9 MB and a
30-second ring is about 95.6 MB per six-camera group. A two-second raw 1080p NV12 ring for six
cameras at 30 frames/s is roughly 1.04 GiB. Long retention therefore stays compressed; decoded
surfaces are short-lived and bounded.

The same traversal tees encoded packets or an append-only encoded spool to durable asynchronous
export work. That branch produces and independently validates the six ordered MP4 records and their
`CameraVideoExportManifest` without forcing incremental analysis to wait. Its work IDs, lease,
retry, failure, and barrier state are durable. EOS product finalization waits for its complete
manifest barrier; an export failure remains an explicit terminal product failure and can be
recovered without rescanning an otherwise retained source.

### 6.3 Incremental identity and EOS finalization

The current final root requires the complete source digest, duration, mapping, and alignment.
Those facts do not exist before end of stream. The streaming layer therefore needs separately
governed pre-EOS subjects. It does not reinterpret the current canonical `TemporalWindow`.

For a complete offline file, the exact source digest may be computed before window execution and
used as the stable source scope. A true live source instead requires an accepted architecture
decision or ADR for an immutable capture-session subject assigned by the source authority. That
subject binds the capture authority, acquisition identity/epoch, six source-channel bindings, and
initial clock/mapping authority. It is stable across worker, process, reconnect, and retry identity,
and it maps immutably to the final source digest at EOS. A local worker does not manufacture this
scope from a path, run ID, or random execution session.

Subject identities then use the stable source scope and actual semantic lineage, for example:

```text
segment_id = digest(
  pre_eos_source_scope_digest,
  camera_slot,
  requested_interval,
  effective_interval,
  ordered_source_packet_or_sequence_closure,
  exact_content_digest,
  mapping_semantic_digest,
  alignment_semantic_digest,
  segmentation_policy_version,
  segment_identity_policy_version
)

window_id = digest(
  pre_eos_source_scope_digest,
  purpose,
  requested_interval,
  effective_interval,
  ordered six-slot segment-or-explicit-absence closure,
  mapping_semantic_digest,
  alignment_semantic_digest,
  parent_subject_key_or_none,
  refinement_role_or_none,
  refinement_generation,
  window_identity_policy_version
)

inference_id = digest(
  window_id,
  purpose,
  input_plan_semantic_digest,
  inference_identity_policy_version
)
```

Execution-local session IDs, paths, locators, worker IDs, attempts, leases, processing timestamps,
and broker offsets do not enter these semantic preimages. They remain provenance.

The exact `InferenceInputPlan` semantic digest already commits the ordered provider-neutral package
inputs plus provider/model/capability, rendering, prompt/schema, limits, and call-plan semantics
required by Section 25.2. It cannot be replaced by a vague package or model label. Retrying the exact
plan preserves the logical `inference_id` but creates a distinct attempt identity; any plan change
creates a different logical inference.

Candidate new contracts include a pre-EOS capture subject, stream-segment manifest, incremental
window, window work result, expected-window plan/sealed manifest, window-terminal closure/barrier,
and recording-finalization map. Final names and shapes require an accepted identity and planning
decision before Schema publication. Every contract carries an explicit semantic-projection or
identity-policy version, and every published schema uses the atomic registration command. Existing
published schemas remain immutable.

The current `WorkItemPlan@1.0` requires an `mcap_id`. A live pre-EOS subject is not an MCAP and
does not get placed into that field. WP1 must either publish a new work-plan version with a typed
source-subject reference or publish a distinct pre-EOS work contract, including migration/replay
and mixed-version scheduler behavior. WP3 cannot start true live scheduling until that decision is
closed.

At end of stream, the finalizer computes the complete recording identity and an ordered segment or
window root. It records an immutable mapping from the pre-EOS source and incremental subjects to
recording-scoped final subjects. This mapping alone is not sufficient for primary completion: the
recording aggregation closure in Section 6.6 must also complete. Incremental facts remain auditable
history and are not overwritten.

### 6.4 Watermarks, lateness, and idle cameras

The initial engineering configuration is:

| Parameter | Starting value | Status |
|---|---:|---|
| Window width `W` | 2 seconds | unvalidated target |
| Hop `H` | 1 second | unvalidated target |
| Allowed lateness `L` | 300 milliseconds | unvalidated target |
| Encoded context ring | 10-30 seconds | bounded candidate configuration |
| Base sampling | 2 frames/s/camera | capacity assumption, not governed O-13 policy |

The aggregate watermark is limited by the slowest non-idle required camera. A camera is not
declared idle merely because it is late; idleness needs an explicit timeout, source-health fact,
and policy version. Once incremental evidence is published, later data follows a registered
late-data action: reject, append a correction/revision, or route review. It does not mutate an
already published record in place.

Offline MCAP execution uses the same event-time rules while reading faster than wall time. Live
execution advances watermarks from source timestamps and declared source health, not worker
completion time.

### 6.5 Durable window DAG

The existing scheduler models already provide dependency progression, leases, fencing, deadlines,
retry wait, cancellation, skip, invalidation, and recovery. The next iteration should extend
canonical composition so it owns a window-level DAG rather than adding a second scheduler.

The proposed logical roles are:

```text
EXPECTED_WINDOW_PLAN_APPEND
  -> SEGMENT_ADMISSION
  -> WINDOW_QUALITY
  -> BASE_PACKAGE
  -> QA_COARSE
  -> conditional QA_DENSE / EVENT_PROPOSAL
  -> conditional ACTION_EVIDENCE
  -> conditional ONSET and OFFSET boundary work
  -> WINDOW_REDUCTION
  -> INCREMENTAL_EVIDENCE_COMMIT
  -> DELIVERY_RECONCILIATION

At EOS:
EXPECTED_WINDOW_PLAN_SEAL
  + WINDOW_TERMINAL_CLOSURE
  + EXPORT_MANIFEST_BARRIER
  -> RECORDING_WINDOW_BARRIER
  -> RECORDING_QA_AGGREGATION
  -> CROSS_WINDOW_PROPOSAL_RECONCILIATION
  -> RECORDING_CANDIDATE_AND_ACTION_CLOSURE
  -> RECORDING_BOUNDARY_AND_OUTPUT_CLOSURE
  -> RECORDING_PRIMARY_COMPLETION
```

These are design labels, not approved enum values. Exact role and subject enums reuse or version
existing scheduler contracts. Independent work may execute concurrently, but dependency ordering
and canonical reduction order remain stable. Claimed work is fenced; an expired worker cannot
commit. Restart reconstructs runnable work from the durable ledger and exact evidence cache without
redispatching already accepted model calls.

### 6.6 Recording-scoped aggregation barrier

Window-local correctness is insufficient for recording completion. The expected set and execution
closure are separate immutable facts. For a complete offline source, the planner derives the full
`ExpectedWindowPlan` from the immutable source timeline plus registered segmentation/window policies
before publishing its child work, following Architecture Section 17.3. For an open-ended live
source, the planner atomically appends each window declaration before publishing that window's
children, then seals the plan at EOS from source duration, source-health closure, and the same
registered policies. It never derives the plan from scheduler rows or successful results.

Because a live source cannot know its complete recording-wide set before EOS, this append/seal
protocol requires the accepted architecture decision or ADR in WP1 before implementation. The
sealed expected-window manifest contains every required ordinal, interval, source subject, and
planning-policy digest, but no terminal status or evidence reference.

A distinct `WindowTerminalClosure` reconciles every sealed expected member to exactly one terminal
status and exact incremental evidence reference. It cannot add an unplanned member or omit a planned
member. The recording barrier completes only when the plan is sealed, every expected member has a
terminal reconciliation, and the fixed six-MP4 export-manifest barrier is complete. Failed, skipped,
late, or absent members remain explicit and are evaluated by the applicable finalization policy.

Recording aggregation then:

1. aggregates camera-level and window-level QA into complete six-camera recording QA evidence;
2. merges compatible proposals across adjacent and overlapping windows under a versioned policy;
3. suppresses duplicate cross-window candidates and action hypotheses by semantic lineage, not
   arrival order;
4. schedules any recording-level action evidence and ONSET/OFFSET work around the merged candidate
   closure, including required pre/post context;
5. reduces one ordered 0/1/N recording hypothesis closure;
6. constructs the ordered run-membership proof, terminal barrier proof, output decision, detailed
   result artifact reference, and final hypothesis references required by ADR 0012; and
7. invokes the existing recording-scoped primary completion transaction only after all required
   facts validate.

An empty recording result is explicit. An unsealed plan, omitted expected member, incomplete terminal
closure, or incomplete export barrier cannot be converted to a successful recording completion.
Tests cover crash points before/after plan append and child publication, an intentionally omitted
window, one event spanning two or more windows, overlapping duplicate proposals, events at the
first/last window, missing terminal members, and exact replay with a different work/run history.

### 6.7 Inference frequency and microbatching

With a two-second window and one-second hop, each six-camera group observes about 360 input frames
per window, but at 2 frames/s/camera only about 12 new sampled images arrive per hop. Re-running the
current 16-call recording graph each second would create 16 requests/s/group, about 41 times the
current average density of 0.391 requests/s/group.

The initial capacity hypothesis is instead:

- run local quality observations every hop;
- run coarse model work every five seconds or on a registered trigger, about 0.2 requests/s/group;
- send only an assumed 5% of suspicious windows to four additional dense/boundary calls, about
  0.2 requests/s/group; and
- keep the total near 0.4 requests/s/group until quality evidence justifies a different policy.

These rates are arithmetic assumptions, not an approved quality policy. O-13 still owns promoted
sampling, triggers, padding, thresholds, and budgets. Capacity tests sweep trigger rates and do not
improve a throughput result by silently reducing the registered quality floor.

Compatible requests enter a bounded microbatcher keyed by model, purpose, input shape, and policy
version. `max_batch_size` and `max_queue_delay` are measured execution settings. A batch does not
change package, request, or inference semantic identity. Results split back into their original
deterministic request identities before evidence persistence and reduction.

### 6.8 Evidence and commit path

The evidence ledger should persist one complete accepted call closure in one transaction where
possible, rather than reopening a database and repeating database-wide checks for each component.
Per-call validation remains strict, but expensive DDL fingerprint, `quick_check`, and complete
registry audits move to defined boundaries such as startup, checkpoint, recovery, and final
completion. Any reduction in audit frequency requires explicit corruption tests and an ADR when it
changes an accepted invariant.

Large intents, catalogs, responses, enriched documents, frame packages, and detailed results move
to immutable content-addressed storage. The database keeps typed exact references and compact
query fields. This follows the artifact boundary accepted in
[ADR 0012](../adr/0012-primary-completion-transaction.md).

The incremental window transaction is durable authority only for its own non-final evidence class.
It does not replace the recording primary-completion transaction. It atomically:

1. verifies the current scheduler lease epoch and fencing token;
2. binds the exact window and evidence closure;
3. appends incremental evidence or an explicit no-result/abstention fact;
4. terminalizes the corresponding window work;
5. appends its separately versioned incremental delivery row; and
6. records the expected EOS finalization lineage.

The incremental relay and final primary outbox relay are at-least-once paths with distinct Wire
subjects. IDs have database unique constraints, and consumers perform idempotent writes. Relay
failure leaves committed local evidence and visible backlog; it does not create or revoke final
recording truth. Final primary delivery continues to follow
[ADR 0014](../adr/0014-local-outbox-relay.md).

### 6.9 Quality response path

The throughput design preserves an explicit response to degraded video:

| Condition | Local evidence | Required response shape |
|---|---|---|
| Black or near-black frames | luminance distribution over a duration | mark degradation; use neighbors/other cameras; abstain if required closure is unavailable |
| Suspected frozen stream | temporal repetition, sequence, and cadence facts | mark a QA/media-degradation interval and avoid treating repeats as new evidence; record source failure only when an independent source-health policy proves it |
| Blur proxy | low edge energy plus temporal context | increase exact neighbors or dense review; never label semantic blur from the proxy alone |
| Suspected occlusion | coverage change and cross-camera disagreement | request semantic model/review evidence; local edge metrics cannot prove occlusion |
| Missing/late camera | watermark, sequence, and source-health evidence | wait within allowed lateness, then append explicit incomplete/degraded evidence |
| Boundary uncertainty | event-triggered pre/post context from encoded ring | run separate ONSET/OFFSET work; abstain if complete context cannot be obtained |

Quality facts are computed before optional model work so a black, frozen, or missing stream cannot
be mistaken for model certainty. Cross-camera alternatives and neighbor expansion can recover
evidence, but no policy promises that every obstruction is solvable. Insufficient evidence ends in
an explicit abstention or review route rather than a fabricated action.

### 6.10 Backpressure classes

Every queue has item, byte, and age limits. Pressure is handled by class:

| Class | Examples | Pressure behavior |
|---|---|---|
| A: authoritative input | segment manifest and required EOS finalization facts | never silently drop; pause admission or durably spool |
| B: primary analysis | registered quality-floor inference and evidence | scale workers, bound retries, or explicitly fail/abstain |
| C: enhancement | extra dense context, shadow evaluation, optional research | versioned downgrade or skip with accounted outcome |
| D: disposable | preview frames and redundant telemetry samples | leaky drop-old is allowed and measured |

The existing backpressure controller should be composed into the window scheduler. Required work
does not share an unbounded in-memory queue with preview or research work. Oldest-item age, not just
queue length, is the primary saturation signal.

A live source may not support pausing. Class A spooling therefore has configured byte/time soft and
hard high-water marks plus reserved recovery space. At the soft mark the service rejects new
optional work and raises pressure; at the hard mark it stops controllable admission. If the source
continues and bytes cannot be retained, the system appends an explicit source-loss, incomplete, or
quarantine-shaped fact under the selected policy, pages the operator, and never reports the missing
interval as successfully analyzed. Recovery resumes from the last verified segment and reconciles
the recorded gap.

### 6.11 Existing-module reuse map

The iteration extends the live canonical path. It does not create a parallel mainline:

| Responsibility | Reuse or extend |
|---|---|
| Composition and recovery order | [local_composition.py](../../src/robata/application/canonical/local_composition.py) |
| Source traversal and media ownership | [mcap_source.py](../../src/robata/application/canonical/mcap_source.py) |
| Incremental quality state | [media_quality.py](../../src/robata/application/canonical/media_quality.py) and its existing source-binding modules |
| Canonical stage semantics and reduction order | [runner.py](../../src/robata/application/canonical/runner.py) |
| Work contracts and durable scheduling | [queue models](../../src/robata/queue/models.py) and [SQLite scheduler](../../src/robata/adapters/sqlite_work_scheduler.py) |
| Pressure policy mechanics | [backpressure.py](../../src/robata/queue/backpressure.py) |
| Inference evidence | [sqlite_inference_evidence.py](../../src/robata/adapters/sqlite_inference_evidence.py) |
| Final completion command and local aggregate | [primary_completion.py](../../src/robata/application/canonical/primary_completion.py) and [sqlite_primary_completion.py](../../src/robata/adapters/sqlite_primary_completion.py) |
| Current scheduler/completion bridge | [durable_work.py](../../src/robata/application/canonical/durable_work.py) |

The retired fake-model throughput runner in
[ADR 0006](../adr/0006-throughput-track-local-parallel-inference.md) remains historical. The
bounded generic media-service behavior retained by
[ADR 0008](../adr/0008-throughput-track-local-camera-parallelism.md) may be reused where it fits the
single-pass design, but its historical flags and runner are not restored.

## 7. Preliminary capacity model

Throughput and latency are different. Pipelining may produce a low-latency first window while a
stage remains underprovisioned and backlog grows forever. Each stage therefore needs a steady-state
capacity proof.

Use the Architecture Section 19.6 planning bound for each stage `j`:

```text
required_concurrency_j >= ceil(h * m_j * lambda_peak_j * L_j / u)
```

Where `lambda_peak_j` is the registered pre-retry peak logical-work arrival rate, `L_j` is
measured service seconds per logical unit, `m_j = 1 + retry_rate_j`, `h` is the approved headroom
multiplier, and `u` is target utilization below saturation. `lambda_peak_j` includes actual
camera, package, candidate, boundary-role, window-overlap, and trigger fan-out. If the measured
arrival rate already includes retries, `m_j=1` avoids double counting.

A simplified `N * S / (H * u)` calculation is valid only when each group creates exactly one
logical unit per hop with no burst, retry, camera, candidate, or boundary fan-out. It is useful for
sanity checking, not stage sizing. Accepted sizing uses representative 15-minute and one-hour peaks,
measured saturation/batch curves, provider quota, and backlog drain after a registered burst.

At the current `R=0.391 requests/s/group`, 100 groups, assumed `M=1.5 seconds`, and `u=0.70`,
the mean-load illustration with `h=1` and `m=1` is 84 concurrent inference slots. A provisional
`h=1.3` raises it to 109 before any peak/fan-out correction. A slot is not a GPU: batching, model
memory, resolution, and accelerator throughput determine how many slots one GPU can sustain.

### 7.1 Input and storage scale

| Workload | Raw compressed input | Current local source plus materialized state |
|---|---:|---:|
| One group for one day | about 275 GB | about 1.12 TiB |
| 100 continuous groups | about 2.55 Gbit/s and 27.5 TB/day | not an acceptable production write shape |
| 500 recording-group h/day | about 5.74 TB/day | about 25.67 TB/day if current amplification persisted |
| 500 aggregate camera-video h/day | about 0.956 TB/day | about 4.278 TB/day if current amplification persisted |

If each camera publishes one 1 KiB manifest per second, 100 continuous six-camera groups produce
about 53 GB/day of broker metadata, roughly 519 times less than the raw video payload. This is why
the broker carries references and the object layer carries bytes.

The candidate local physical-storage target is one authoritative compressed copy, reference-only
views, lazy event-selected images, and compact database rows. An initial unvalidated target is total
physical bytes of 1.3-1.6 times raw input, excluding configured replication, backup, and mandated
retention copies.

### 7.2 Accelerator reference, not sizing evidence

The NVIDIA DeepStream 9.1 performance page reports end-to-end 1080p H.265 RT-DETR throughput of
380 frames/s on RTX 4500 and 643 frames/s on L40S for its specified configuration. At 70% target
utilization and this sample's 179.75 aggregate frames/s, the derived arithmetic is one and two
complete six-camera groups respectively. The same page shows large variation across models, so
these external vendor benchmarks cannot size Robata's future VLM or temporal reasoning path.

The local host has an RTX 4060 Laptop GPU with 8,188 MiB. There is no accepted comparable Robata
measurement for it. A DeepStream/Triton PoC is useful only after the target model or representative
injected service profile, batch policy, and stage metrics are fixed.

## 8. Candidate engineering targets

These are unvalidated local targets, not governed production SLOs or a `PASS` verdict:

| Signal | Candidate target |
|---|---|
| Incremental evidence latency | p95 within 5 seconds of window close; p99 within 15 seconds |
| Refined boundary evidence | p95 within 15 seconds after required post-context becomes available |
| Stable utilization | no measured stage above 70% at the accepted offered load |
| Backlog | p95 oldest required work age below two window hops; no monotonic growth in soak |
| Duplicate effects | zero duplicate business facts or sink rows; at-least-once duplicate publish attempts are counted and deduplicated |
| Committed local metadata RPO | zero under the tested local crash model |
| Local recovery RTO | below 10 minutes for the accepted state size |
| Physical write amplification | 1.3-1.6 times raw input under the defined retention profile |
| Quality accounting | every degraded, skipped, late, failed, or abstained unit remains explicit |

A latency approximation is:

```text
window close wait + allowed lateness + queue age + inference + durable commit
```

The service reports each term separately. A five-second aggregate is not accepted if it hides a
growing queue or silently skipped work. The overlay's OD-SLO-001 and OD-QUALITY-001 remain
unresolved, so these targets cannot become promotion evidence.

## 9. Next-iteration work packages

Work proceeds in this order. Later work may be prototyped behind a port, but a package does not
claim completion until its acceptance gate passes.

### WP0: Instrument the current fresh path

Deliver:

- stable spans and counters for inspect, demux, decode, export, materialize, quality, inference,
  serialization, validation, evidence transactions, completion, relay, and review;
- CPU, RSS, disk read/write bytes, SQLite transaction count/time, queue age, and artifact-byte
  accounting; and
- one machine-readable baseline bound to commit, source digest, runtime, and policy versions.

Accept when repeated fresh runs reconcile stage time to wall time, byte accounting to the state
tree, and work/call counts to authoritative ledgers. Optimization does not start from file
timestamps alone.

### WP1: Register incremental identities and Wire contracts

Deliver:

- one accepted architecture decision or ADR for the stable pre-EOS source scope, its immutable
  mapping to final source identity, and the live expected-window append/seal protocol;
- reviewed segment, incremental-window, window-result, expected-plan/sealed-manifest,
  terminal-closure, and finalization-map schemas;
- a versioned scheduler work contract that does not overload `WorkItemPlan@1.0.mcap_id`;
- semantic validators and explicit projection/policy versions;
- golden identity vectors, exact replay, late/absence cases, and mutation rejection; and
- atomic catalog publication with no edits to an existing published schema.

Accept when two implementations or independent reference paths reproduce every identity vector,
two different captures with identical repeated/black frame bytes cannot collide, purpose and parent
lineage cannot collide, locators do not affect semantic identity, and EOS finalization maps
incremental facts without rewriting them. The accepted planner decision must preserve
planner-before-child publication and prohibit deriving the expected set from outcomes.

### WP2: Build single-pass local media ingest

Deliver:

- one ordered MCAP traversal feeding all six camera indexes and incremental quality state;
- bounded encoded rings and lazy frame materialization;
- immutable segment/spool references with exact digests;
- a durable asynchronous six-MP4 export/validation branch that produces the fixed manifest without
  a second source traversal; and
- removal of full six-MP4 export and validation from the incremental analysis prerequisite path.

Accept when canonical order and relevant source facts match the current baseline, memory stays
within the configured bound, repeat bytes/identities are deterministic, and required video is not
physically copied into both registry and view trees. The six ordered MP4 records and their manifest
remain exact, immutable, independently probed, restart-safe deliverables.

### WP3: Compose the durable window scheduler

Deliver:

- window DAG creation, dependency progression, lease/fence claims, deadlines, and invalidation;
- the versioned pre-EOS scheduler subject selected by WP1;
- an outcome-independent append-before-publish expected-window plan, immutable EOS sealed manifest,
  separate window-terminal closure, and expected-export manifest/barrier;
- composed backpressure and explicit pressure-class outcomes;
- restart reconstruction and exact evidence reuse; and
- EOS scheduling of recording finalization.

Accept when injected crashes at every work boundary produce no duplicate business fact, stale
workers cannot commit, completed calls are not redispatched, no child can publish before its expected
declaration is durable, an omitted declared member cannot close the terminal barrier, and
backlog/age remain queryable.

### WP4: Batch inference and evidence persistence

Deliver:

- bounded worker pools and purpose-compatible microbatching;
- deterministic split and canonical merge independent of completion order;
- one accepted-call evidence transaction or equivalent batched persistence boundary;
- content-addressed large payloads with compact database references; and
- startup/checkpoint/completion audit boundaries with corruption tests.

Accept when serial and parallel execution preserve logical identities and semantic results, batch
settings are observational, failures expose no partial accepted call, and instrumentation isolates
the remaining provider-independent overhead.

### WP5: Add incremental evidence commit and recording finalization

Deliver:

- one local transaction for incremental evidence, window-work terminalization, and its separately
  versioned delivery row;
- relay and reconciliation from both fresh and recovered incremental evidence;
- immutable incremental-to-final mapping;
- recording-wide six-camera QA aggregation, cross-window proposal merge/deduplication, candidate
  closure, and recording-level boundary/output reduction;
- exact sealed expected-window manifest, separate terminal closure, export barrier, ordered
  membership proof, output decision, hypothesis closure, and detailed-result reference construction
  for ADR 0012;
- the existing recording-scoped primary completion and primary outbox only at valid finalization;
  and
- nonblocking review routes for degradation, abstention, policy sampling, and disagreement.

Accept when crash tests cover artifact-before-transaction, transaction-before-relay,
relay-before-ack, and EOS finalization; recovery selects the appropriate database ledger as
authority for each distinct evidence class. Cross-window tests prove one physical action spanning
multiple windows yields one recording ActionEvent, while missing or failed required members cannot
produce a successful primary completion.

### WP6: Qualify the local streaming shape

Before execution, publish a content-addressed benchmark manifest that pins:

- candidate commit, source digest, host CPU/GPU/memory, driver, OS, power mode, runtime, and lockfile;
- cold/fresh/replay/cache state and artifact-retention profile;
- chunk/window/lateness/ring, sampling, trigger, candidate, boundary, and fan-out policies;
- mock latency/failure distributions, retry policy, seed, and request/batch limits;
- warm-up, repetition count, burst shape, observation cutoff, and 30-minute smoke duration; and
- both 500 recording-hour/day and 500 aggregate camera-video-hour/day offered-load scenarios.

Use the same 40.890455-second source first, then at least a 30-minute repeated or generated stream.
Keep provider traffic mocked but inject the pinned latency distributions and failures. The
30-minute run is a local stability smoke, not representative long-soak qualification.

Acceptance has two unvalidated engineering gates:

1. structural gate: fresh processing is at most five times source time, required state bytes are at
   most twice raw input, and measurements show that duplicate decode/export and per-row audit work
   are no longer dominant; and
2. streaming gate: sustained service capacity is at least 1.86 recording-seconds per wall-second
   per accepted six-camera group under the provisional `h=1.3`, `u=0.70` planning assumptions,
   incremental latency meets the candidate target, and the 30-minute smoke has no growing required
   backlog.

These gates remain `NOT_MEASURED` for production qualification and do not imply real-model capacity.
The [synthetic capacity harness](synthetic-capacity-harness-v1.md) remains the arithmetic and
regression mechanism, not certifying evidence.

### WP7: Add an optional accelerated adapter

Only after WP0-WP6 expose a stable provider-neutral boundary, evaluate GStreamer/DeepStream and
Triton on Linux or an explicitly documented WSL2 PoC. Do not add the stack to the default runtime.

Accept when the accelerated adapter produces the same provider-neutral segment/window identities,
quality facts within registered tolerances, and canonical reductions; reports decode throughput,
batch fill, queue delay, GPU utilization/memory, and p95/p99 latency; and fails closed when the
optional runtime is absent.

## 10. Explicitly deferred external qualification

The internal iteration can complete contracts, local streaming orchestration, mock inference,
durability, recovery, incremental delivery, final primary outbox, review routing, observability,
and synthetic/real-media load tools. It cannot close these external or governed conditions:

- real model accuracy, calibration, boundary quality, latency, quota, and cost;
- representative governed camera corpus and adjudicated ground truth;
- approved O-03/O-04/O-10/O-11/O-12/O-13/O-14/O-16 policy decisions;
- production database, broker, object storage, credentials, isolation, and operator ownership;
- production filesystem and object-store durability/retention semantics;
- representative burst, failover, long-soak, and disaster-recovery evidence; and
- protected candidate-commit Schema baseline approval.

External dependencies remain explicit adapters and evidence gates. They do not block internal work
packages unless a package would otherwise fabricate a policy or production claim.

## 11. Required observability

Every local and future production run should expose:

- source bytes/s, frames/s, camera sequence gaps, skew, watermark lag, late/idle counts, spool
  byte/time high-water marks, and explicit source-loss intervals;
- decode and materialization throughput, ring occupancy, selected frame rate, and dropped preview
  count;
- per-stage offered/completed/failed/skipped rates, queue length, oldest age, busy/idle/pressure
  time, lease expiry, retry, and deadline counts;
- model queue delay, batch size/fill, input/infer/output time, timeout, cache hit, and GPU metrics;
- artifact read/write bytes, physical duplication, checksum failures, unreferenced objects, and
  database transaction counts/durations;
- incremental, refined, and final completion latency, replay/reconciliation counts, duplicate
  publish attempts, deduplicated sink effects, and delivery/DLQ state; and
- black/freeze/blur-proxy/missing/occlusion-review rates, abstention, and review backlog/SLA.

All throughput reports show recording-hours and camera-video-hours separately and retain failed,
skipped, late, pending, and abstained denominators. The two unresolved 500-hour interpretations in
the execution-spec overlay remain separate scenarios.

## 12. Official research references

- [NVIDIA DeepStream architecture](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html)
- [NVIDIA nvstreammux](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvstreammux.html)
- [NVIDIA DeepStream performance](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html)
- [NVIDIA Triton dynamic batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
- [NVIDIA Triton optimization](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html)
- [GStreamer queue](https://gstreamer.freedesktop.org/documentation/coreelements/queue.html)
- [GStreamer tee](https://gstreamer.freedesktop.org/documentation/coreelements/tee.html)
- [Apache Kafka design](https://kafka.apache.org/43/design/design/)
- [Apache Kafka producer configuration](https://kafka.apache.org/43/configuration/producer-configs/)
- [Apache Flink event time](https://nightlies.apache.org/flink/flink-docs-release-2.3/docs/concepts/time/)
- [Apache Flink backpressure](https://nightlies.apache.org/flink/flink-docs-release-2.3/docs/ops/monitoring/back_pressure/)
- [Debezium outbox event router](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)

## 13. Definition of done for the next internal iteration

The iteration is complete only when:

1. the current batch path has an exact instrumented baseline;
2. an accepted architecture decision or ADR closes stable pre-EOS identity and the live
   expected-window append/seal protocol, and the versioned scheduler work contract does not overload
   `WorkItemPlan@1.0.mcap_id`;
3. one source traversal continuously produces versioned immutable window subjects and the durable
   asynchronous six-camera MP4 export manifest without a second source scan;
4. the existing durable scheduler owns the window DAG, outcome-independent expected-window plan,
   sealed manifest, separate terminal closure, export-manifest barrier, and recovery path;
5. inference/evidence work is bounded, batchable, deterministic, and content-addressed;
6. incremental evidence and its delivery row share one distinct local transaction;
7. recording-scoped QA aggregation, cross-window merge/deduplication, candidate/action/boundary
   closure, membership proof, and output decision complete before primary completion;
8. EOS finalization waits for both recording and export barriers and preserves the existing
   recording canonical truth and primary outbox semantics;
9. degraded video and pressure produce explicit governed-shape outcomes rather than silent loss;
10. at-least-once duplicate publish attempts are counted while duplicate business facts and sink
   rows remain zero under replay and recovery tests;
11. all WP6 structural and streaming gates pass under the pinned benchmark manifest, including
    fresh processing at most five times source time, required state at most twice raw input,
    capacity of at least 1.86 recording-seconds per wall-second, the incremental latency target, and
    no growing required backlog during the 30-minute local smoke;
12. all new schemas were atomically registered and existing published bytes remain immutable; and
13. every result still states its local/non-production evidence class and remaining external gates.
