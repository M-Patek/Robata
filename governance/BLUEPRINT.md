# Robata Production-Ready Throughput, Quality, and Deployment Blueprint

**Cycle:** six-camera quality delivery, adaptive evidence, and production-boundary qualification
**Planning date:** 2026-07-26
**Revision:** replacement plan; this file supersedes the previous BLUEPRINT.md
**Dispatch unit:** one phase or explicitly named cross-module phase per main-agent window

## Authority and evidence boundary

This is a local construction map, not an approval workflow or production certificate.

- schemas/**, schemas/schema-catalog.json, the registered schema workflow, tracked source,
  tests, and conformance fixtures are the contract and behavior truth.
- governance/REQUIREMENTS.md is a production target/capacity reference. Its cloud choices,
  500 recording-hours/day, T+1/T+3, two-H100 estimates, costs, and model suggestions are
  hypotheses or targets until measured.
- Evidence classes are explicit: LOCAL_CONFORMANCE, LOCAL_BENCHMARK,
  REPRESENTATIVE_BENCHMARK, EXTERNAL_QUALIFICATION, and PRODUCTION_QUALIFIED.
- Real H100/vLLM, NVDEC, R2, broker/object-store, Supabase/Postgres, representative labels,
  and long-run soak evidence remain NOT_MEASURED.

## Current observed baseline and decisions

These are dated local fresh-path references, not production claims. They use a 40.8335
recording-second, six-camera fixture, PyAV CPU decode, and deterministic offline provider.

- Event-driven fresh timing is about 118.75 wall seconds versus 161.03 seconds for the
  earlier hot-loop; scheduler transactions are about 3,559 versus 18,193 and read I/O
  about 2.67 GiB versus 6.41 GiB.
- source.prepare is a parent envelope (about 77.38 seconds) containing inspect,
  capture/publish, publication validation, ledger/metadata, frame indexing, quality timing,
  materialization, and quality publication. It is not one algorithm and must not be added
  to child spans.
- Fresh-path action remains concentrated in source.stream.capture_publish (about 53.05
  seconds; window append/publish/drain), source.materialize (about 22.33 seconds;
  decode/PNG/hash/fsync/read amplification), inference-evidence SQLite scope, and
  completion auditing/serialization. Replay needs an independent profile.
- MCAP single-pass, six-camera materialization, local WAL recovery, provider-neutral calls,
  adaptive sampling, evidence lineage, completion/outbox, review routing, and local
  qualification scaffolding exist. Current dirty-tree behavior must be reprofiled by P0.
- Traditional low-cost visual signals already exist; extend the decoded-frame/adaptive
  bridge instead of creating another pixel pipeline.
- Fisheye undistortion, perspective correction, and Egocentric preprocessing are new,
  optional derived views. They require camera calibration, policy versions, exact lineage,
  and representative quality/cost evidence before becoming defaults.
- R2 stores objects, RunPod supplies GPU/provider execution, Supabase supplies database/Auth,
  and pgvector stores/searches vectors. None supplies CPU/NVMe workers, durable scheduler
  semantics, an embedding encoder, or production qualification.

## Product Outcome

**Outcome:** Robata converts six-camera recordings into complete evidence-bound 21-class
clip QA, event proposals, and annotation/search inputs through a bounded, replayable
pipeline, with local substitutes and explicitly qualified production adapters.

**Why now:** Source preparation and media I/O dominate the measured fresh path. The next
cycle must lower transaction, decode, provider, and storage cost without hiding source
failures, changing warning/fail semantics, or changing published identity.

**Success measures:**

- Every eligible clip has a complete result; sparse calls never silently remove a clip,
  camera, interval, or source failure.
- Full source integrity remains observable: camera mapping, packet/timestamp order, decode
  gaps, missing intervals, and provenance.
- Profiles report recording-hours and camera-hours separately, plus decoded/selected images,
  provider images/calls/HTTP/tokens, CPU/GPU/NVMe, queue/backlog, and I/O.
- Local fresh/replay/crash/duplicate proofs are reproducible and evidence classes visible.
- Adaptive sampling reports quality, abstention/incomplete coverage, dense-upgrade rate,
  provider amplification, and cost together.
- Only an external gate may promote a frozen report to 500 recording-hours/day, QA T+1,
  annotation T+3, and PRODUCTION_QUALIFIED.

**Non-goals for this cycle:**

- No released schema or identity formula is edited in place.
- No full-frame RGB persistence or full-recording undistortion becomes the default.
- Local recording-affine SQLite is not replaced merely to claim scale.
- No business quality thresholds are invented before governed labels and sign-off.
- Embedding/search, optional crops, and shadow work cannot block authoritative QA completion.
- R2, RunPod, Supabase, NVDEC, and two H100s are not declared supported before proof.

## Non-negotiable invariants

| Boundary | Invariant |
| --- | --- |
| Contract | Released schema bytes/catalog pins are immutable; changes use registered version and migration/upcast. |
| Source | Original MCAP/compressed authority, timestamps, mapping, decode facts, and provenance are never rewritten or hidden. |
| Media | Derived/corrected/cropped artifacts bind source frame/time, exact bytes, policy, and calibration. |
| Sampling | Base coverage and escalation reasons are explainable; visual proxies are not semantic truth. |
| QA | Every eligible clip gets a complete 21-class projection, including explicit unknown/abstained/incomplete. |
| Events | Candidate/proposal/boundary evidence retains lineage and deterministic replay. |
| Inference | Intent, raw, parsed, selection, accepted lineage, retry, and replay facts close in the ledger. |
| Stream | Work, barriers, leases, fences, retry/DLQ/backpressure, and recovery remain durable. |
| Identity/delivery | Logical identity is transport-independent; one terminal truth, no lost/duplicate completion/outbox/review. |
| Evidence | Local, representative, external, and production evidence are never conflated. |

## Overall Roadmap

| Phase | Outcome | Main module(s) | Depends on | Local proof | External follow-up |
| --- | --- | --- | --- | --- | --- |
| P0 - contract and measurement truth | Scope fingerprint, units, evidence classes, immutable baseline | contract-governance, qualification-ops | None | schema/profile/replay baseline | None |
| P1 - source and stream spine | Single-pass source plus message-independent window/work scheduling | source-media, stream-control, canonical-integration | P0 | scheduler/source/replay/crash profile | Broker later |
| P2 - feed-once media and visual sentinel | One decode supplies integrity, traditional-CV signals, bounded evidence | source-media, sampling-qa, qualification-ops | P1 | media/adaptive/I/O/RSS profile | R2/target decode |
| P3 - adaptive QA/event and evidence lineage | Base/coarse/dense/context/event execution is provider-neutral/replayable | sampling-qa, event-semantics, inference-evidence | P2 | QA/event/inference delayed-provider tests | Real VLM |
| P4 - complete product closure | Complete 21-class result and evidence-bound event boundaries | sampling-qa, event-semantics, canonical-integration | P3 | product/event/canonical/Pareto tests | Governed labels |
| P5 - terminal truth and recovery | Pre-EOS, completion, outbox, review, replay preserve one graph | identity-delivery, stream-control, canonical-integration | P1,P3,P4 | crash/retry/duplicate/recovery tests | Production delivery |
| P6 - parallel service/backpressure | Recording-affine workers scale with bounded queues and saturation evidence | stream-control, canonical-integration, qualification-ops | P1-P5 | 1/2/4 worker/backlog report | Capacity sizing |
| P7 - object/broker adapters | R2 and broker/outbox preserve bytes, leases, fences, reconciliation | contract-governance, source-media, stream-control, identity-delivery | P5,P6 | fake-client failure/replay contracts | Real services |
| P8 - target media and RunPod qualification | NVDEC/CPU and two-H100/vLLM measured under adaptive workload | source-media, inference-evidence, qualification-ops | P2,P3,P6,P7 | qualification harness/replay | H100/NVDEC/RunPod |
| P9 - Supabase/pgvector projection | Structured-first retrieval plus async versioned vector search | contract-governance, canonical-integration, qualification-ops | P4,P7 | fake DB/vector/replay contracts | Supabase/Postgres/RLS |
| P10 - representative production gate | Frozen quality/capacity/reliability/deadline/cost/soak report | qualification-ops, canonical-integration, event-semantics | P2-P9 | local qualification package | Labels, soak, sign-off |

## Module Phases

### contract-governance - P0/P7/P9: contract and adapter boundaries

**Result**

Schema, port, object-locator, calibration, broker, and vector projections evolve without
in-place wire or identity changes.

**Primary paths and entry points**

- schemas/**, schemas/schema-catalog.json, scripts/register_schema.py
- src/robata/contracts/**, src/robata/ports/**
- src/robata/ports/artifact_registry.py, src/robata/frame_cache.py
- tests/unit/test_register_schema.py, test_schema_immutability.py,
  tests/contract/test_schema_release_policy.py

**Implementation outline**

1. Freeze a scope digest over code revision, catalog, workload, policies, and identity.
2. Add minimal ports for calibration/preprocess, R2, broker, and vector projections;
   unfinished adapters fail closed.
3. Map R2 version/ETag/presigned URL as locator metadata, never content identity.
4. Register a new schema for a published semantic/preprocessing shape change.

**Keep intact**

Schema bytes, catalog pins, exact/semantic hashing, logical keys, idempotency keys,
fences, and transport-independent identity.

**Done when**

- [ ] Scope/evidence register reproduces fresh and replay profiles.
- [ ] Release tests reject mutation/unregistered changes.
- [ ] Adapter/preprocess metadata is internal-only or versioned.
- [ ] Local proofs need no cloud SDK.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_register_schema.py tests/unit/test_schema_immutability.py
python -m pytest tests/contract/test_schema_catalog.py tests/contract/test_schema_release_policy.py tests/contract/test_schema_upcasting.py
~~~

**Next boundary**

P1 consumes frozen identity; P7/P9 consume versioned ports.

### source-media - P1/P2/P7/P8: single-pass media and artifact ports

**Result**

One verified source traversal produces compressed authority, bounded quality facts, and
content-addressed artifacts; CPU, hardware, and R2 share media ports.

**Primary paths and entry points**

- src/robata/application/canonical/mcap_source.py, single_pass_video.py, bounded_media.py,
  media_quality.py
- src/robata/adapters/mcap_single_pass.py, pyav_*.py, parallel_*.py, nvdec_*.py
- src/robata/frame_cache.py, src/robata/ports/artifact_registry.py
- tests/unit/test_mcap_single_pass.py, test_pyav_interval_spool.py, test_bounded_media.py,
  test_canonical_media_quality.py, tests/integration/test_canonical_mcap_source.py

**Implementation outline**

1. Add nested source.prepare, capture/publish, drain, decode, render, hash, fsync, and
   materialization spans; remove repeated whole-file reads.
2. Fuse selected evidence and low-cost sentinel observations in one decode; never persist
   unbounded RGB.
3. Extend existing quality signals with optional lens/geometry triggers.
4. Add versioned CameraCalibrationProfile and FramePreprocessPolicy; apply fisheye/
   perspective/Egocentric views only to selected windows/model inputs until measured.
5. Implement R2 blob-first exact-hash publication and orphan/visibility reconciliation.

**Keep intact**

MCAP authority, frame IDs/timestamps, sidecar verification, artifact lineage, CPU fallback,
and frame/video ports.

**Done when**

- [ ] Fresh/replay have separate profiles and no source loss.
- [ ] Read amplification is <=10x on the fixture, target 8x, or codec/GOP lower bound is
  attributed.
- [ ] Sentinel and selected evidence share decode with bounded RSS and no raw-RGB store.
- [ ] Derived geometry binds source/time, calibration digest, policy, resolution, exact bytes.
- [ ] CPU and hardware adapters pass the same timestamp/artifact contract.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_mcap_single_pass.py tests/unit/test_pyav_interval_spool.py tests/unit/test_bounded_media.py tests/unit/test_canonical_media_quality.py tests/unit/test_media_quality_supplemental.py
python -m pytest tests/integration/test_real_mcap_single_pass.py tests/integration/test_canonical_mcap_source.py
~~~

**Next boundary**

P2 feeds sampling; P7 maps artifacts to R2; P8 measures hardware.

### sampling-qa - P2/P3/P4: adaptive coverage and product projection

**Result**

Explainable base/coarse/dense/context sampling reduces expensive work while preserving
complete 21-class clip projection.

**Primary paths and entry points**

- src/robata/sampling/adaptive.py, signals.py, grid.py, materializer.py
- src/robata/qa_pipeline/fast_detector.py, coarse.py, dense.py, product.py
- src/robata/application/canonical/media_quality_source_binding.py,
  media_quality_supplemental.py, supplemental_qa_evidence.py, product_qa.py
- tests/unit/test_sampling_adaptive.py, test_qa_pipeline_core.py, test_local_qa_product.py,
  test_media_quality_supplemental.py, tests/integration/test_canonical_offline.py

**Implementation outline**

1. Keep base coverage and bounded pre/post context for every camera/clip.
2. Map source-quality, uncertainty, cross-camera disagreement, event, and boundary reasons
   to deterministic upgrades.
3. Keep traditional CV as signal/trigger; full-frame context remains QA evidence.
4. Treat ROI/undistorted/Egocentric views as supplemental until equivalence is proven.
5. Project all 21 classes, warning marks, fail, abstained, and incomplete.

**Keep intact**

QA vocabulary, warning/fail semantics, clip times, provenance, and policy/version identity.

**Done when**

- [ ] Every eligible clip has deterministic 21-class projection.
- [ ] Upgrades preserve timestamps/context.
- [ ] Three or more policies have a quality/cost Pareto report.
- [ ] Optional embedding/crop/geometry cannot block completion.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_sampling_adaptive.py tests/unit/test_qa_pipeline_core.py tests/unit/test_local_qa_product.py tests/unit/test_media_quality_supplemental.py
python -m pytest tests/integration/test_canonical_offline.py
~~~

**Next boundary**

P3 consumes call plans; P4 consumes QA/event facts.

### event-semantics - P3/P4: evidence-bound event semantics

**Result**

Candidates, proposals, and boundary revisions remain evidence-bound, deterministic, and replayable.

**Primary paths and entry points**

- src/robata/event_pipeline/candidate.py, evidence.py, proposer.py, boundary_refinement.py
- tests/unit/test_event_pipeline_core.py, test_event_projection_guards.py,
  test_supplemental_temporal_package.py
- tests/integration/test_canonical_action_event_revision.py

**Implementation outline**

1. Build candidates from base/coarse/dense facts and context triggers.
2. Refine onset/offset locally while retaining upstream evidence and source timestamps.
3. Preserve proposal/revision ordering; distinguish abstention from no-event.
4. Coordinate identity changes with contract-governance and identity-delivery.

**Keep intact**

Event identity/revision formulas, temporal roles, evidence references, and replay order.

**Done when**

- [ ] Boundary events acquire pre/post context.
- [ ] No unbound or duplicate revision is created.
- [ ] Replay identities are identical.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py tests/unit/test_supplemental_temporal_package.py
python -m pytest tests/integration/test_canonical_action_event_revision.py
~~~

**Next boundary**

P4 reduction and P5 completion.

### inference-evidence - P3/P8: provider-neutral inference and replay

**Result**

Provider-neutral calls and evidence lineage are bounded and replayable; RunPod is a later
qualified adapter, not a new canonical path.

**Primary paths and entry points**

- src/robata/inference/input_plan.py, preparation.py, orchestrator.py, evidence.py,
  adapter.py, runpod.py
- src/robata/adapters/sqlite_inference_evidence.py
- tests/unit/test_inference_input_plan.py, test_inference_orchestrator.py,
  test_sqlite_inference_evidence.py, test_runpod_adapter.py

**Implementation outline**

1. Keep call identity transport-independent and batch only compatible groups.
2. Measure existing lifecycle connection/cache behavior before changing append granularity.
3. Persist intent, raw, parsed, selection, lineage, retry, timeout, and partial failure.
4. Pin provider model/version/engine/quantization/topology/limits/retry for P8.

**Keep intact**

Request identity, accepted lineage, raw responses, and single-request fallback.

**Done when**

- [ ] Delayed provider proves ordering, batch, timeout, retry, partial failure.
- [ ] Remaining call-proportional connection/transaction work is attributed.
- [ ] Real reports include queue/TTFT/E2E/tokens/retry/OOM/memory/utilization.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_inference_input_plan.py tests/unit/test_inference_orchestrator.py tests/unit/test_sqlite_inference_evidence.py tests/unit/test_runpod_adapter.py
python -m pytest tests/integration/test_canonical_offline.py
~~~

**Next boundary**

P3 canonical reduction; P8 real provider.

### stream-control - P1/P5/P6/P7: durable scheduling and backpressure

**Result**

Local durable scheduling remains authoritative while bounded queues and a production broker
preserve leases, fences, retries, DLQ, and backpressure.

**Primary paths and entry points**

- src/robata/queue/**
- src/robata/adapters/sqlite_work_scheduler.py, sqlite_stream_work_ledger.py,
  sqlite_stream_delivery.py, sqlite_barrier.py
- src/robata/queue/redis_adapter.py
- src/robata/application/canonical/durable_work.py, stream_scheduler.py,
  stream_recording_reduction.py, parallel_service.py
- tests/unit/test_barrier.py, test_sqlite_barrier.py, test_sqlite_work_scheduler.py,
  test_stream_recording_reduction.py, test_redis_task_queue.py

**Implementation outline**

1. Keep source messages in memory until window/state transitions; batch append/plan/publish.
2. Drain only newly ready work; reserve full scans for restart/reconciliation.
3. Keep recording-affine bounded queues and explicit optional-work shedding.
4. Add broker delivery only as a projection over the authoritative scheduler.

**Keep intact**

SQLite WAL same-host scope, work identities, lease/fence semantics, and locator independence.

**Done when**

- [ ] Transactions scale with windows/transitions, not messages.
- [ ] Queue admission/rejection/retry/lease/DLQ/backlog metrics are deterministic.
- [ ] Duplicate/delayed-ack/restart/stale-lease preserve terminal truth.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_barrier.py tests/unit/test_sqlite_barrier.py tests/unit/test_sqlite_work_scheduler.py tests/unit/test_stream_recording_reduction.py tests/unit/test_redis_task_queue.py
python -m pytest tests/integration/test_canonical_local_command.py
~~~

**Next boundary**

P1 source, P6 scaling, P7 broker.

### identity-delivery - P5/P7: terminal completion and delivery

**Result**

One terminal result is durably completed, delivered, reviewed, and reconciled under crashes
and duplicates.

**Primary paths and entry points**

- src/robata/adapters/sqlite_primary_completion.py, sqlite_event_identity_registry.py,
  sqlite_outbox.py, sqlite_review_queue.py
- src/robata/application/canonical/logical_nodes.py, primary_completion.py,
  local_outbox_delivery.py, local_review_routing.py
- src/robata/admission/**, src/robata/review/**
- tests/unit/test_event_identity_registry.py, test_admission_ledgers.py, test_review_routing.py,
  test_redis_outbox.py
- tests/integration/test_sqlite_primary_completion.py, tests/integration/test_sqlite_outbox_relay.py,
  tests/integration/test_canonical_local_review_routing.py

**Implementation outline**

1. Carry prevalidated bytes/digests/identities/evidence into completion.
2. Replace repeated scans only with incremental seal/run verification; retain offline audit.
3. Exercise crash-before/after commit, timeout, stale lease, duplicate and different-bytes.
4. Map production delivery only after local recovery proof.

**Keep intact**

Logical/revision identity, command bytes, terminal meaning, outbox idempotency, review routing.

**Done when**

- [ ] Fresh/replay terminal/result/outbox/review digests are identical.
- [ ] Crash/duplicate creates no lost or duplicate terminal.
- [ ] Completion spans are attributed and improved without weaker durability.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_event_identity_registry.py tests/unit/test_admission_ledgers.py tests/unit/test_review_routing.py tests/unit/test_redis_outbox.py
python -m pytest tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py tests/integration/test_canonical_local_review_routing.py
~~~

**Next boundary**

P6 parallel service; P7 production delivery.

### canonical-integration - P1/P4/P5/P6/P9/P10: canonical composition and closure

**Result**

The canonical runner advances source, stream, media, QA, event, inference, completion, and
retrieval deterministically with explicit local/external adapters.

**Primary paths and entry points**

- src/robata/application/canonical/{runner,runner_support,local_composition,result_validation,
  models,projections,reduction,output_admission,boundary_windows,mcap_source,durable_work,
  primary_completion,logical_nodes,pre_eos_execution}.py
- src/robata/application/canonical_offline.py
- tests/unit/test_canonical_offline_reducer.py, test_canonical_run_membership.py,
  test_canonical_offline_call_part_concurrency.py
- tests/integration/test_canonical_offline.py, tests/integration/test_canonical_local_command.py

**Implementation outline**

1. Make the graph explicit: integrity -> media/sentinel -> adaptive evidence -> provider
   inference -> QA/event reduction -> completion/outbox/review.
2. Execute eligible windows pre-EOS while EOS remains authoritative and replayable.
3. Turn degraded provider/source input into explicit abstention/incomplete, never false success.
4. Index retrieval only after terminal closure; indexing failure cannot reopen QA completion.

**Keep intact**

Stage ordering, reduction determinism, compatibility facades, replay equivalence, terminal semantics.

**Done when**

- [ ] Fresh/replay/restart/duplicate-window converge to one graph.
- [ ] Provider failure and incomplete source facts are visible.
- [ ] Retrieval cannot block or rewrite QA/event completion.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_canonical_offline_reducer.py tests/unit/test_canonical_run_membership.py tests/unit/test_canonical_offline_call_part_concurrency.py
python -m pytest tests/integration/test_canonical_offline.py tests/integration/test_canonical_local_command.py
~~~

**Next boundary**

P5/P6 consume terminals; P9 consumes closed events; P10 qualifies assembly.

### qualification-ops - P0/P2/P6/P8/P10: measurement and production qualification

**Result**

Every profile states workload, evidence class, hardware/provider, units, quality, resources,
costs, and unresolved gaps.

**Primary paths and entry points**

- src/robata/runtime/{capacity,canonical_profile,benchmark,local_streaming_benchmark}.py
- src/robata/benchmark/{metrics,pareto,splits,promotion,qualification,provider_qualification,evidence}.py
- tests/unit/test_runtime_capacity.py, test_canonical_profile.py, test_benchmark.py,
  test_benchmark_qualification.py, test_provider_qualification.py
- tests/unit/test_representative_production_qualification.py,
  tests/integration/test_canonical_recovery_qualification_evidence.py

**Implementation outline**

1. Use one profile schema for fresh/replay and local/representative/external evidence;
   report recording-hours and camera-hours separately.
2. Attribute demux, decode, resize/colorspace, CV, encoding, hashing, SQLite, provider,
   completion, delivery, queue, and backlog.
3. Produce Pareto reports for sampling, geometry, provider batching, workers, and shedding.
4. Keep production_eligible false until external gates pass.

**Keep intact**

Evidence labels, denominator-safe capacity math, split/leakage rules, and NOT_MEASURED.

**Done when**

- [ ] Fresh and replay cannot be confused.
- [ ] Claims include workload/config/hardware/duration and before/after timing.
- [ ] Quality/capacity/reliability/I/O/token/cost axes appear together.

**Run locally**

~~~powershell
python -m pytest tests/unit/test_runtime_capacity.py tests/unit/test_canonical_profile.py tests/unit/test_benchmark.py tests/unit/test_benchmark_qualification.py tests/unit/test_provider_qualification.py
python -m pytest tests/unit/test_local_streaming_benchmark.py tests/unit/test_representative_production_qualification.py
~~~

**Next boundary**

P8-P10 and external review.

## Cross-Module Phases

### P0 - Contract, workload, and evidence truth

**Participating modules:** contract-governance, qualification-ops

**End-to-end result:** Every timing/quality claim identifies source, schema, policy, provider,
hardware, workload, and evidence class.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| contract-governance | Scope digest, schema/identity baseline, seam versions | schemas/**, contracts/**, ports/** | catalog/immutability |
| qualification-ops | Workload fingerprint, fresh/replay profile, units/counters | runtime/**, benchmark/** | profile/capacity |

**Implementation outline**

1. Record source duration, six-camera load, frame counts, bytes, images/calls/tokens,
   CPU/GPU/NVMe, queues, and terminals.
2. Digest code, catalog, policies, and workload; date historical snapshots.
3. Never promote local fixtures or old smoke snapshots to production evidence.

**Compatibility notes:** No existing wire change; report scope is internal or registered.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_register_schema.py tests/unit/test_schema_immutability.py tests/unit/test_runtime_capacity.py tests/unit/test_canonical_profile.py tests/unit/test_benchmark.py
~~~

### P1 - Source single-pass and durable stream spine

**Participating modules:** source-media, stream-control, canonical-integration

**End-to-end result:** MCAP is traversed once into verified authority while work is emitted at
bounded window transitions, not per source message.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| source-media | H264 spools and packet/timestamp/decode facts | mcap_source.py, single_pass_video.py, mcap_single_pass.py | mcap/source tests |
| stream-control | batch append/plan/publish and indexed lookup | sqlite_work_scheduler.py, sqlite_stream_work_ledger.py, stream_scheduler.py | scheduler tests |
| canonical-integration | source-to-stream recovery bridge | local_composition.py, durable_work.py | canonical local command |

**Compatibility notes:** Preserve leases, fences, idempotency, window/work identities, and
local SQLite recovery.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_mcap_single_pass.py tests/unit/test_sqlite_work_scheduler.py tests/unit/test_stream_recording_reduction.py
python -m pytest tests/integration/test_canonical_mcap_source.py tests/integration/test_canonical_local_command.py
~~~

### P2 - Feed-once media, traditional CV, and optional geometry

**Participating modules:** source-media, sampling-qa, qualification-ops

**End-to-end result:** One decode supplies structural/visual facts and bounded artifacts;
signals reduce provider work without hiding source evidence.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| source-media | Feed-once decode, sentinel facts, and versioned derived geometry | mcap_source.py, bounded_media.py, media_quality.py, artifact_registry.py | media/lineage tests |
| sampling-qa | Deterministic coverage, upgrade triggers, and class projection | sampling/adaptive.py, signals.py, qa_pipeline/** | sampling/QA tests |
| qualification-ops | I/O, RSS, provider-amplification, and Pareto measurements | runtime/**, benchmark/** | media/adaptive profile |

**Implementation outline**

1. Run raw structural checks before correction.
2. Compute low-resolution black/freeze/exposure/edge/blur/motion/scene facts in the current
   observer; add lens/geometry triggers only as supplemental facts.
3. Apply fisheye/perspective/Egocentric views only to selected windows/model inputs with
   cached maps and deterministic pass-through.
4. Bind derived frame IDs to source/time, calibration digest, policy, resolution, exact bytes.
5. Compare baseline, sentinel-only, and selective-geometry quality/cost/I/O/RSS policies.

**Compatibility notes:** Keep media-quality v1 proxy fields; new published semantic fields
require registration. Full-frame context and raw authority remain mandatory.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_canonical_media_quality.py tests/unit/test_sampling_adaptive.py tests/unit/test_media_quality_supplemental.py
python -m pytest tests/integration/test_canonical_mcap_source.py tests/integration/test_canonical_offline.py
~~~

### P3 - Adaptive provider-neutral QA/event cascade

**Participating modules:** sampling-qa, event-semantics, inference-evidence, canonical-integration

**End-to-end result:** Base/coarse/dense/context/event work is one provider-neutral,
lineage-complete execution graph.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| sampling-qa | coverage/upgrade and coarse/dense evidence | sampling/adaptive.py, qa_pipeline/coarse.py, dense.py | QA tests |
| event-semantics | candidate/proposal/boundary evidence | event_pipeline/** | event tests |
| inference-evidence | intent/raw/parsed/selection/retry | inference/**, sqlite_inference_evidence.py | inference tests |
| canonical-integration | stage order and pre-EOS execution | runner.py, pre_eos_execution.py, local_stream_finalization.py | delayed-provider |

**Compatibility notes:** Batching changes grouping only, not call identity, lineage, event identity,
or terminal semantics.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_sampling_adaptive.py tests/unit/test_qa_pipeline_core.py tests/unit/test_event_pipeline_core.py tests/unit/test_inference_orchestrator.py tests/unit/test_sqlite_inference_evidence.py
python -m pytest tests/integration/test_canonical_offline.py tests/integration/test_pre_eos_factory_wiring.py
~~~

### P4 - Complete 21-class and event closure

**Participating modules:** sampling-qa, event-semantics, canonical-integration

**End-to-end result:** Complete clip QA and evidence-bound event revisions exist for eligible
input, including explicit abstention/incomplete.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| sampling-qa | 21-class projection with warning/fail/abstained/incomplete states | qa_pipeline/product.py, application/canonical/product_qa.py | QA product tests |
| event-semantics | Candidate boundaries, proposals, revisions, and evidence references | event_pipeline/** | event guard tests |
| canonical-integration | Deterministic reduction and canonical result admission | canonical/reduction.py, projections.py, runner.py | canonical offline tests |

**Implementation outline**

1. Preserve 21 classes, warning marks, whole-recording fail semantics, and exact times.
2. Distinguish no-event, abstained, incomplete, warning, and fail.
3. Refine candidate-local boundaries and add signed thresholds only after governed labels.

**Compatibility notes:** No new result meaning without schema/version decision; ROI/corrected
views remain supplemental until representative equivalence.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_local_qa_product.py tests/unit/test_event_projection_guards.py tests/unit/test_supplemental_temporal_package.py
python -m pytest tests/integration/test_canonical_offline.py tests/integration/test_canonical_action_event_revision.py
~~~

### P5 - Durable terminal truth and recovery

**Participating modules:** identity-delivery, stream-control, canonical-integration

**End-to-end result:** Pre-EOS/EOS, completion, outbox, review, and replay produce one graph
under crash, retry, duplicate, and lease expiry.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| identity-delivery | Terminal completion, outbox, review, and identity equality | primary_completion.py, sqlite_primary_completion.py, sqlite_outbox.py | completion/outbox tests |
| stream-control | Leases, fences, barriers, retry/DLQ, and recovery inputs | durable_work.py, sqlite_work_scheduler.py, sqlite_barrier.py | scheduler/recovery tests |
| canonical-integration | Pre-EOS/EOS closure and crash/replay orchestration | pre_eos_execution.py, local_stream_finalization.py | recovery integration |

**Implementation outline**

1. Carry prevalidated bytes/digests/evidence into completion.
2. Replace scans only with incremental seal/run verification; keep offline audit.
3. Inject crash-before/after commit, timeout, stale lease, duplicate window/outbox, and
   different-bytes conflicts.
4. Keep optional/degraded branches non-blocking and explicit.

**Compatibility notes:** Logical keys, fences, exact bytes, outbox IDs, review routing, and
terminal status meaning stay unchanged.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_sqlite_work_scheduler.py tests/unit/test_local_stream_finalization.py tests/unit/test_review_routing.py
python -m pytest tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py tests/integration/test_canonical_recovery_qualification_evidence.py
~~~

### P6 - Recording-level parallelism and bounded backpressure

**Participating modules:** canonical-integration, stream-control, qualification-ops

**End-to-end result:** Recording-affine workers share bounded provider capacity, shed optional
work in a stated order, and expose saturation/backlog.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| canonical-integration | Recording worker lifecycle, admission, cancellation, and drain | parallel_service.py, runner.py, local_composition.py | parallel-service tests |
| stream-control | Bounded queues, lease renewal, backpressure, and shedding | queue/**, stream_scheduler.py, durable_work.py | queue/scheduler tests |
| qualification-ops | 1/2/4-worker profiles, saturation, and sizing math | runtime/local_streaming_benchmark.py, runtime/capacity.py | capacity profile |

**Implementation outline**

1. One recording per CPU/NVMe worker; never share SQLite WAL over network storage.
2. Measure 1/2/4 workers; separate media/provider/completion/outbox concurrency.
3. Verify admission, rejection, cancellation, lease recovery, backlog drain, and shedding.
4. Convert measured throughput to deployment sizing, not production qualification.

**Compatibility notes:** Process topology and queue locators are not canonical identity.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_local_streaming_benchmark.py tests/unit/test_runtime_capacity.py tests/unit/test_canonical_parallel_service.py
python -m pytest tests/integration/test_canonical_local_command.py
~~~

### P7 - R2 artifact and broker/outbox boundaries

**Participating modules:** contract-governance, source-media, stream-control, identity-delivery

**End-to-end result:** Object and delivery services implement existing ports without changing
artifact lineage, scheduler authority, completion, or outbox semantics.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| contract-governance | Versioned object/broker ports and locator contracts | contracts/**, ports/**, schemas/** | contract/release tests |
| source-media | Exact-byte blob-first publication and artifact reconciliation | ports/artifact_registry.py, frame_cache.py | artifact registry tests |
| stream-control | Broker projection for leases, fences, retry, DLQ, and stale acknowledgements | queue/redis_adapter.py, sqlite_stream_delivery.py | queue/reconciliation tests |
| identity-delivery | Idempotent outbox/review delivery and duplicate handling | sqlite_outbox.py, local_outbox_delivery.py, local_review_routing.py | outbox/review tests |

**Implementation outline**

1. R2 uses versioned object keys/manifests, exact SHA-256/size checks, and signed URLs as
   locators only.
2. Blob-first publish precedes metadata/DAG commit; reconcile orphan/partial/visibility/
   retention/duplicate cases.
3. SQLite work/terminal ledgers remain authoritative until broker lease/heartbeat/fence/
   retry/DLQ/stale-ack/reconciliation behavior is proven.
4. Add metrics and failure injection to Redis/managed broker and outbox.

**Compatibility notes:** ETags/version IDs are not content identity; new message shapes use
registered schemas.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_local_artifact_registry_observability.py tests/unit/test_redis_task_queue.py tests/unit/test_redis_outbox.py
python -m pytest tests/integration/test_redis_outbox_reconciliation.py tests/integration/test_sqlite_outbox_relay.py
~~~

### P8 - Target media and two-H100 RunPod qualification

**Participating modules:** source-media, inference-evidence, qualification-ops

**End-to-end result:** Target media and real RunPod/vLLM are measured under the adaptive
workload with a safe envelope and no automatic promotion.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| source-media | CPU/NVDEC decode and materialization parity with timestamp/artifact checks | adapters/pyav_*.py, adapters/nvdec_*.py | media adapter tests |
| inference-evidence | RunPod/vLLM mapping, batching, retry, telemetry, and replay | inference/adapter.py, inference/runpod.py | provider adapter tests |
| qualification-ops | Topology/concurrency matrix, quality/capacity counters, and safe envelope | benchmark/provider_qualification.py, benchmark/qualification.py | qualification tests |

**Implementation outline**

1. Freeze model/version/engine/quantization/topology/contracts/limits/batch/concurrency/retry.
2. Measure CPU/NVDEC decode, resize, and materialization at 125 camera-seconds/s average
   and 150 margin; capture codec/resolution/FPS/GOP/transfer path.
3. Compare two single-card replicas and tensor parallel only when justified; scan concurrency
   and capture queue, TTFT, E2E P50/P95/P99, images/calls/tokens, retries, rejects, OOM,
   memory, KV/cache, and utilization.
4. Exercise restart/timeout/invalid output/partial failure in fresh namespaces.
5. Keep production_eligible false until external review.

**Compatibility notes:** Recorded responses replay through validators/ledger; credentials and
endpoint locators never enter request identity.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_runpod_adapter.py tests/unit/test_provider_qualification.py tests/unit/test_provider_qualification_count_binding.py tests/unit/test_runtime_capacity.py
python -m pytest tests/unit/test_representative_production_qualification.py
~~~

### P9 - Supabase/Postgres/pgvector retrieval projection

**Participating modules:** contract-governance, canonical-integration, qualification-ops

**End-to-end result:** Structured retrieval remains authoritative while async versioned
text/vision vectors are added without blocking QA completion.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| contract-governance | Versioned embedding/vector projection, locator, and retention contracts | contracts/**, ports/**, schemas/** | schema/retrieval contract tests |
| canonical-integration | Terminal-to-structured EventIndex projection and replay-safe membership | canonical/projections.py, reduction.py | membership/retrieval tests |
| qualification-ops | Async backfill, vector recall, filtering, latency, and cost profiles | benchmark/**, retrieval/** | retrieval profile tests |

**Implementation outline**

1. Keep EventIndex and structured/facet retrieval as the first usable search path.
2. Define a Postgres/Supabase projection for selected revisions, clips, labels, provenance,
   vectors, RLS, retention, backfill, and replay.
3. pgvector stores/indexes/searches vectors; an explicit CPU/API/RunPod encoder creates them.
4. Make embedding writes async/idempotent by event revision/artifact identity; outages do
   not reopen or delay QA completion.
5. Add vector recall plus structured facet filtering/reranking.

**Compatibility notes:** Embedding ID/model/dimension/index policy are versioned metadata;
structured retrieval and canonical identities stay unchanged.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_retrieval.py tests/unit/test_canonical_run_membership.py
python -m pytest tests/integration/test_canonical_offline.py
~~~

Supabase/RLS/index/retention/cost evidence remains external.

### P10 - Representative production qualification and operating gate

**Participating modules:** qualification-ops, canonical-integration, event-semantics, source-media,
sampling-qa, inference-evidence, stream-control, identity-delivery

**End-to-end result:** One frozen report demonstrates quality, capacity, recovery, deadlines,
cost, and operations on representative data and selected external services.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| qualification-ops | Frozen manifest, quality/capacity/reliability evidence, and gate report | benchmark/qualification.py, evidence.py, promotion.py | qualification tests |
| canonical-integration | End-to-end run, terminal/recovery/replay aggregation | canonical/runner.py, local_composition.py | canonical recovery tests |
| source-media | Representative codec/camera/geometry media matrix | adapters/mcap_*.py, adapters/pyav_*.py, application/canonical/media_quality.py | source/media tests |
| sampling-qa | All classes, sampler policies, calibration, and leakage-safe splits | sampling/adaptive.py, qa_pipeline/product.py | QA/Pareto tests |
| event-semantics | Candidate/proposal/boundary metrics and temporal adjudication | event_pipeline/candidate.py, proposer.py, boundary_refinement.py | event/temporal quality tests |
| inference-evidence | Provider latency/error/token/cost evidence | inference/**, inference/runpod.py | provider qualification tests |
| stream-control | Arrival, backlog, lease, failure, and soak behavior | queue/**, durable_work.py | stress/recovery tests |
| identity-delivery | Deadline, outbox, review, and terminal integrity | primary_completion.py, sqlite_outbox.py, review/** | completion tests |

**Implementation outline**

1. Freeze code/catalog/media/calibration/sampler/preprocess/prompt/model/provider/hardware/
   adapters/arrival distribution/split.
2. Cover codec/resolution/FPS/GOP, fisheye/Egocentric views, all 21 classes, warning/fail,
   missing/decode gaps, and event boundaries.
3. Set signed thresholds only after adjudicated labels and leakage-safe splits.
4. Run 500 recording-hours/day-equivalent arrivals, QA T+1, annotation T+3, provider/
   storage/broker failures, crashes, lease expiry, duplicates, backlog drain, and 24h soak.
5. Report recording-hours/camera-hours, provider images/calls/tokens, CPU/GPU/NVMe,
   queues/backlog, R2 storage/egress, DB/queue cost, P50/P95/P99, and recovery.
6. External review alone may promote PRODUCTION_QUALIFIED.

**External qualification gates (not local completion)**

| Gate | Required evidence | Current boundary |
| --- | --- | --- |
| E0 - evidence freeze | Code/catalog, workload, model, sampler, preprocess, hardware, and data digests | Local preparation; external freeze pending |
| E1 - quality sign-off | Governed labels, leakage-safe splits, per-class/temporal metrics, calibration, abstention/incomplete review | Representative labels NOT_MEASURED |
| E2 - media and storage | Target codec/NVDEC parity plus R2 exact-byte, range, retention, and reconciliation runs | External media/R2 NOT_MEASURED |
| E3 - provider qualification | RunPod/vLLM two-H100 topology, safe concurrency, latency/error/token and cost envelope | Real H100/provider NOT_MEASURED |
| E4 - reliability and soak | Crash/retry/lease/outbox proofs, managed-service faults, backlog drain, and 24-hour soak | External soak NOT_MEASURED |
| E5 - capacity/SLA/cost | 500 recording-hours/day-equivalent arrivals, QA T+1, annotation T+3, resource and cost bounds | Production capacity NOT_MEASURED |
| E6 - go/no-go | Independent review of the frozen report and runbook; only then set production_eligible | Must remain false until E0-E5 pass |

**Compatibility notes:** Qualification freezes semantics; it does not create a second path or
relax a failed threshold.

**Combined proof**

~~~powershell
python -m pytest tests/unit/test_benchmark_qualification.py tests/unit/test_representative_production_qualification.py tests/unit/test_runtime_capacity.py
python -m pytest tests/integration/test_canonical_recovery_qualification_evidence.py tests/integration/test_canonical_local_command.py
~~~

## Blockers and External Dependencies

| Condition | What can still be completed locally | Temporary substitute | Later external proof |
| --- | --- | --- | --- |
| Real VLM/two H100s | Request mapping, batch/retry, evidence, telemetry, replay | Delayed deterministic HTTP provider | Endpoint/topology matrix and safe envelope |
| Target media/NVDEC | CPU port equivalence and attribution | PyAV CPU | Target SKU/driver/SDK/codec/GOP/resolution throughput |
| R2/object store | Exact-hash artifact/feed-once contract, blob-first/reconciliation | Local registry/filesystem cache | PUT/GET/HEAD/range/multipart/visibility/retention/egress/failure |
| Durable broker | SQLite authority, fake/Redis lease/fence/retry/DLQ tests | Local Redis/fake client | Managed broker outage/stale-ack/duplicate/recovery |
| Supabase/Postgres/pgvector | Structured EventIndex, projection/fake-vector contract, async failure | In-memory index | RLS/auth/ANN/retention/backfill/cost/failover |
| Representative labels/media | Split/metric/Pareto framework and fixture regression | Versioned local corpus | Governed adjudication and per-class/temporal quality |
| Arrival/deadline/soak | Burst/backlog/failure harness and runbook draft | Short local stress | Production arrival, T+1/T+3, 24h soak |
| Product thresholds | Metric and acceptance-register implementation | Empty/unpromoted thresholds | Product-owner signed thresholds |

## Acceptance and Verification

- [ ] Schema/wire/identity/hash/logical-key/idempotency/fence/semantic changes have a
  registered version/migration decision.
- [ ] Fresh/replay and local/representative/external/production evidence are distinct.
- [ ] Throughput claims include workload fingerprint, hardware/provider/configuration,
  duration, before/after timing, queues/resources, and separate recording/camera units.
- [ ] Quality claims include data manifest, leakage-safe split, model/sampler/preprocess
  versions, per-class metrics, abstention/incomplete, calibration, and boundary metrics.
- [ ] Traditional-CV/geometry claims include quality impact and removed provider work; no
  proxy silently becomes semantic truth.
- [ ] Artifact claims include exact bytes/hash, timestamps, provenance, reuse, and reconcile.
- [ ] Crash/retry/duplicate/lease-expiry proofs show no lost/duplicate completion, outbox,
  review, or accepted evidence.
- [ ] External limits stay NOT_MEASURED until the required run is performed and reviewed.
- [ ] Final gate includes quality, capacity, reliability, deadline, cost, security/retention,
  and runbook evidence.

## Suggested Dispatch Prompt

~~~text
Work on contract-governance + qualification-ops / P0 - contract and measurement truth.
Read AGENTS.md, governance/BLUEPRINT.md, the template, and module cards. Freeze scope,
workload/evidence units, and contract identity. Preserve schema/hash/idempotency/fence rules.
Run focused P0 proofs and report fingerprints and external blockers.
~~~

~~~text
Work on source-media + stream-control + canonical-integration / P1 - source and stream spine.
Reduce message-proportional transactions and repeated reads while preserving source/timestamp/
decode provenance and durable work. Run P1 proofs; report transactions, wall time, I/O,
fresh/replay/crash behavior.
~~~

~~~text
Work on source-media + sampling-qa / P2 - feed-once media and visual sentinel.
Reuse one decode for structural facts, traditional-CV signals, selected evidence, and optional
versioned geometry views. Preserve raw authority/full-frame context/lineage/bounded memory.
Report I/O, RSS, selected images, upgrade rate, and quality class.
~~~

~~~text
Work on sampling-qa + event-semantics + inference-evidence + canonical-integration / P3-P4.
Implement adaptive provider-neutral QA/event plans, evidence lineage, boundary context, and
complete 21-class projection. Preserve replay, warning/fail, and explicit abstention/incomplete.
Report provider/image/call amplification.
~~~

~~~text
Work on identity-delivery + stream-control + canonical-integration / P5.
Optimize completion inputs without changing terminal meaning. Exercise crash, retry, duplicate,
lease expiry, outbox, review, and replay. Report terminal/outbox/review identity equality.
~~~

~~~text
Work on canonical-integration + stream-control + qualification-ops / P6.
Measure 1/2/4 workers, bounded queues, saturation, optional-work shedding, cancellation,
and backlog drain. Never share SQLite WAL over network storage. Report the named bottleneck.
~~~

~~~text
Work on contract-governance + source-media + stream-control + identity-delivery / P7.
Implement R2/object, broker, and outbox ports with fake failure/reconciliation fixtures.
Preserve exact bytes, identity, lease/fence/retry/DLQ, and local authority; do not claim
production support without external evidence.
~~~

~~~text
Work on source-media + inference-evidence + qualification-ops / P8.
Freeze media/provider configuration, run target decode and two-H100 matrices, capture telemetry
and safe saturation, and replay recorded responses. Keep production_eligible false until review.
~~~

~~~text
Work on contract-governance + canonical-integration + qualification-ops / P9.
Keep structured retrieval authoritative; make embedding/indexing async, versioned, idempotent,
and non-blocking. Treat pgvector as vector storage/search, not an encoder. Report external
database/RLS/index/retention blockers.
~~~

~~~text
Work on qualification-ops + canonical-integration / P10.
Freeze inputs, run governed quality/capacity/recovery/deadline/cost/soak evidence, and produce
one promotion report. Do not self-promote PRODUCTION_QUALIFIED; report every unresolved gap.
~~~
