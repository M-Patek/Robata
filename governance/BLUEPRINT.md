# Robata Throughput and Quality Delivery Blueprint

**Cycle:** 500 recording-hours/day internal engineering closure and external
qualification readiness

**Planning date:** 2026-07-24

**Dispatch unit:** one phase below per main-agent window

This blueprint is the concrete planning output derived from
`governance/BLUEPRINT_TEMPLATE.md`. It is a local construction guide, not an approval
workflow or task service.

## Product Outcome

**Outcome:** Robata can continuously turn six-camera recordings into complete,
21-class clip-level QA results and evidence-bound event results through a bounded,
replayable streaming pipeline. The internally testable path is complete with local
storage and deterministic providers; the same path is ready to be qualified against a
real two-H100 inference service and production storage/broker adapters.

**Why now:** The control plane and canonical result chain are substantially connected,
but the latest real-media profile still spends most wall time before inference, and the
current pre-EOS work executor proves durable scheduling with local conformance outputs
rather than real provider-neutral QA/event execution. The next cycle must increase
throughput without weakening black-screen, blur, obstruction, freeze, temporal context,
or replay coverage.

**Success measures:**

- Product capacity target: sustain `500 recording-hours/day`, equal to `20.833x`
  recording real time; the engineering target is `25x` to retain 20% service margin.
- Every eligible clip receives a complete product QA result covering the existing 21
  issue classes. Sparse/adaptive model calls are an implementation choice, not a reason
  to return an incomplete clip result.
- Black frames, freeze, exposure failure, blur, obstruction uncertainty, missing camera,
  decode/timestamp gaps, and event onset/offset context remain observable after sampling
  is optimized.
- Local fresh/replay evidence states its workload, timing, resource use, provider mode,
  and evidence class. Mock results are never presented as real-model capacity or quality.
- Before external qualification, the repository can exercise bounded queues, durable
  recovery, adaptive sampling, provider batching, completion, outbox reconciliation, and
  non-blocking review end to end with local substitutes.

**Non-goals for this cycle:**

- Do not invent business acceptance thresholds for recall, precision, calibration, or
  event boundary error without representative governed labels.
- Do not introduce a new approval framework, remote task service, or heavyweight process.
- Do not replace local SQLite with a production database merely to claim scale. Local
  recording-affine SQLite remains the recovery proof; production storage/broker adapters
  are an external deployment concern.
- Do not change a published schema, logical identity, semantic projection, idempotency
  key, or fence unless the phase explicitly includes a version or migration decision.

## Minimal Engineering Invariants

Only these three cross-cutting constraints apply to every phase:

1. **Contract and identity stability.** Published wire shapes and identity formulas stay
   unchanged unless explicitly versioned through the registered schema workflow.
2. **Durable terminal truth.** Completion, outbox delivery, replay, and reconciliation
   must not lose or duplicate an admitted terminal result.
3. **Truthful evidence class.** Local fixtures, mocks, representative benchmarks, and
   production qualification are reported as different evidence classes.

These invariants protect product correctness; they are not additional workflow gates.

## Current Measured Baseline

### Workload and machine

The current frozen real-media slice contains 40.8335 recording-seconds across six
cameras, 7,350 indexed frame observations, and 492 materialized evidence frames. The
latest measured environment reports Python 3.13.5, PyAV 18.0.0, 32 logical CPUs, and no
real GPU/provider work. The inference provider is the deterministic offline fixture.

The baseline is therefore useful for code-path attribution and regression, but it is
not evidence that a real model can serve production load.

### Latest fresh-path comparison

| Profile | Wall time | Recording RTF | Serial capacity | Scheduler transactions | Read I/O | Meaning |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `causal-rr4-hotloop-20260724-fresh.json` | 161.03 s | 0.254x | 6.09 rec-h/day | 18,193 | 6.41 GiB | prior hot-loop behavior |
| `causal-rr4-eventdriven-20260724-fresh.json` | 118.75 s | 0.344x | 8.25 rec-h/day | 3,559 | 2.67 GiB | current event-driven baseline |

The event-driven revision reduced wall time by about 26%, scheduler transactions by
about 80%, and read I/O by about 58%. This is strong evidence that transaction/query
shape, rather than SQLite itself, was a major source-path bottleneck.

The current profile still takes about `2.91 wall-seconds / recording-second`. A naive
linear comparison would require about 61 identical single-process fixture workers to
cover 500 recording-hours/day. That number is **not** a deployment estimate: the local
process averages only about two CPU cores, uses a mock provider, and does not measure
recording-level horizontal concurrency.

### Remaining local hot path

| Span or resource | Current measurement | Interpretation |
| --- | ---: | --- |
| `source.prepare` | 77.38 s | largest end-to-end stage |
| `source.stream.capture_publish` | 53.05 s | still performs 3,559 scheduler transactions while reading 16,210 messages |
| `source.materialize` | 22.33 s | six-camera parallel materialization already exists; optimize traversal and retained work rather than re-adding a thread pool |
| `inference.pipeline` | 16.56 s | mock execution is dominated by evidence/validation and stage work, not network/GPU latency |
| inference-evidence SQLite | 226 connections / 227 transactions | connection and read-back scope is proportional to evidence operations |
| `completion.commit` | 7.71 s | authoritative close remains intentionally durable, but serialization/audit cost needs attribution |
| process read I/O | 2.67 GiB | about 22x the 130.3 MB source artifact; still too much for 500 h/day |
| process write I/O | 0.53 GiB | about 4.4x the source artifact |

### Existing capabilities to reuse

The next phases must not reimplement work that is already present:

- MCAP single-pass packet capture and six-way parallel video export exist.
- Selected-frame materialization already uses six camera workers.
- SQLite adapters already use WAL; the next gain is transaction/query shape, persistent
  connections, and ownership boundaries, not merely setting `journal_mode=WAL` again.
- Canonical call parts already use bounded `asyncio` concurrency, and the inference
  orchestrator already has a micro-batch dispatcher.
- Durable window/work state, recovery, completion, outbox relay/reconciliation, and
  review routing already exist locally.
- The 21-class QA product contract and benchmark metrics/split logic already exist.
- `sampling/adaptive.py`, QA projectors/planners, and event projectors contain real domain
  behavior. Compatibility facades that fail closed are not the canonical execution path.

## Target Architecture

```text
recording ingress
    |
    v
compressed packet + timeline lane --------------+
  - one source traversal                         |
  - camera/timestamp/decode integrity             |
  - compressed per-camera spool                  |
    |                                             |
    +--> window clock --> durable window DAG -----+----> bounded work queues
    |                                                   |       |       |
    +--> low-cost visual sentinel                       |       |       |
    |     black/freeze/exposure/blur/change             |       |       |
    |                                                   |       |       |
    +--> adaptive evidence selector --------------------+       |       |
          base coverage + suspicion upgrade                     |       |
          context neighbours + candidate boundaries             |       |
                                                               v       v
                                                  provider-neutral inference
                                                  - coarse QA
                                                  - dense QA only when needed
                                                  - event proposal/evidence
                                                  - boundary refinement
                                                  - task/shape/deadline batching
                                                               |
                                                               v
                                                  durable evidence ledger
                                                               |
                                                               v
clip-level 21-class QA + event reduction --> authoritative completion/outbox
                                                     |
                                                     +--> non-blocking review
```

### Separation of work

1. **Structural integrity lane** examines the whole recording timeline: camera mapping,
   packet order, timestamps, decode failures, missing intervals, and source lineage.
   Sampling must never hide these facts.
2. **Visual sentinel lane** uses cheap, downscaled frames or luminance features to retain
   continuous coverage for black/frozen/overexposed/blurred or suspicious views.
3. **Semantic lane** spends VLM capacity adaptively. Every clip still receives a complete
   result; only uncertain or suspicious regions escalate to dense/context/event calls.

### Deployment shape

- A recording is sticky to one CPU/NVMe worker while its local durable state is active.
  This avoids sharing SQLite over a network filesystem; SQLite WAL is intended for
  same-host concurrency.
- Multiple recording workers run in parallel and submit provider-neutral batches to a
  shared GPU inference service. GPU work is decoupled from source traversal and local
  ledger commits by bounded queues.
- If the selected VLM fits on one H100, prefer one model replica per card for additive
  throughput and independent failure domains. Use tensor parallelism across both cards
  only when the model cannot fit or a measured throughput result justifies it.
- Object storage and a broker replace local artifact/outbox transports at deployment
  boundaries; they do not change canonical identity or terminal semantics.

## Capacity Model

### Product units

For `500 recording-hours/day` and six cameras:

| Quantity | Required average |
| --- | ---: |
| Recording service | 20.833 recording-s/s |
| Engineering target with 20% margin | 25.0 recording-s/s |
| Camera media load | 125 camera-s/s |
| Camera load at 20% margin | 150 camera-s/s |
| Full 30 FPS decode, no margin | 3,750 decoded frames/s |
| Full 30 FPS decode, 20% margin | 4,500 decoded frames/s |

`recording-hour`, `camera-hour`, and dense/event work-scope hours are different units and
must be reported separately.

### Sampling and provider load

For six-camera recordings, the unique selected-image rate is:

```text
unique_images_per_second = 125 * effective_fps_per_camera
effective_fps = base_fps + upgrade_fraction * (dense_fps - base_fps)
```

| Effective FPS/camera | Unique images/s | Unique images/day |
| ---: | ---: | ---: |
| 0.2 | 25 | 2.16 M |
| 0.5 | 62.5 | 5.40 M |
| 1.0 | 125 | 10.80 M |
| 2.0 | 250 | 21.60 M |
| 5.0 | 625 | 54.00 M |

With 1 FPS base sampling and 5 FPS dense sampling, a 10% dense upgrade rate produces
1.4 effective FPS, 175 unique images/s, and 15.12 million unique images/day.

Window overlap can multiply provider payload even when unique sampling is unchanged:

```text
windows_per_second = 20.833 / hop_seconds
logical_calls_per_second = windows_per_second
    * sum(stage_fraction * calls_per_window / windows_per_batch)
```

For 2-second windows with a 1-second hop, six cameras, and 1 FPS, one fused call contains
12 images and sends 250 provider-images/s because each unique frame appears in two
windows. Therefore every profile must report both unique images and provider images,
plus logical calls, HTTP requests, batch size, split count, retry count, and tokens.

### Data and NVMe load

For average per-camera encoded bitrate `b Mbps`:

```text
aggregate_ingress_MB_per_second = 15.625 * b
raw_storage_TB_per_day = 1.35 * b
```

At 4 Mbps/camera the six-camera workload is about 62.5 MB/s and 5.4 TB/day; at
8 Mbps/camera it is about 125 MB/s and 10.8 TB/day. Persisting decoded RGB is not viable:
even 1 FPS of 1080p RGB would be about 67 TB/day. The data plane therefore retains the
compressed authority and materializes only bounded evidence artifacts.

CPU sizing is measured rather than guessed:

```text
required_cpu_cores = decoded_frames_per_second
    * cpu_milliseconds_per_decoded_frame / 1000
    / target_cpu_utilization
```

The benchmark must separate demux, decode, resize/colorspace, local quality features,
image encoding, SQLite, and hashing. Selected frames are not equivalent to decoded
frames because inter-frame codecs may require GOP traversal.

### Two-H100 budget

At 70% average GPU utilization, two cards provide:

```text
2 * 86,400 * 0.70 = 120,960 aggregate GPU-seconds/day
120,960 / 500 = 241.9 GPU-seconds/recording-hour
                 = 4.03 aggregate GPU-minutes/recording-hour
```

This is a useful qualification budget, not proof that an unspecified VLM fits it. The
real model, image tokenization, prompt/output length, quantization, batch/concurrency,
KV cache, retries, and preprocessing must be measured together.

The selected H100 SKU's NVIDIA documentation and runtime inventory must be recorded before
qualification. GPU decode/NVDEC is an optional candidate adapter only; this blueprint
makes no decode-capacity claim from engine counts. Codec, resolution, GOP, concurrency,
transfer path, and fallback behavior require a target-SKU benchmark against the
125-camera-seconds/s workload.

## Quality Model

### Failure-mode coverage after sampling optimization

| Failure or quality risk | Cheap continuous evidence | Escalation | Result behavior |
| --- | --- | --- | --- |
| Black screen | luminance/black-pixel ratio on sentinel frames plus decode/timeline facts | increase cadence around transition | retain interval and affected cameras |
| Freeze/stale view | temporal luma/hash/feature change with source timestamps | dense neighbours around start/end | distinguish true static scene from uncertain freeze |
| Blur | focus/edge statistic on downscaled frames | choose nearby sharper evidence and retain original failure observation | never silently replace the bad frame without recording it |
| Exposure failure | luminance histogram/clipping | local interval upgrade | emit evidence even if semantic model abstains |
| Obstruction/occlusion | entropy/edge/ROI-coverage change and coarse semantic uncertainty | dense full-frame plus optional ROI/context | route unresolved cases to review; do not treat a crop as complete context |
| Missing camera or timestamp gap | packet/timeline ledger over all source messages | no model call required | fail/mark incomplete according to existing product semantics |
| Decode corruption | decoder error/packet continuity | retry bounded decode or isolate interval | no silent loss |
| Short action/event | motion/change candidate plus base coverage | pre/post neighbours and candidate-centered dense sampling | preserve exact source timestamps |
| Imprecise onset/offset | candidate evidence closure | boundary refinement ONSET/OFFSET roles | revised event remains linked to upstream evidence |

ROI crops and selected frames are supplemental evidence. Until a representative quality
study proves equivalence, they do not replace the contextual full frame needed for QA or
event interpretation. Likewise, unselected frames are not deleted from source authority;
they are simply not decoded/materialized on the expensive path.

### Complete clip output without one call per clip

The product requirement is interpreted as **one complete QA result per eligible clip**,
not **one dense model request per clip**. A clip result may be reduced from:

- structural recording facts;
- base-coverage visual observations;
- coarse model outputs;
- selectively upgraded dense outputs;
- temporal/event evidence; and
- explicit abstention or incomplete-input facts.

This preserves the external result shape while allowing the internal execution graph to
spend most compute on uncertain or high-value regions.

### Quality evidence

Reuse the existing benchmark implementation rather than creating a second framework:

- QA: per-class precision/recall/F1, macro/micro scores, critical-issue recall,
  abstention, unknown/incomplete coverage, ECE, and Brier score.
- Events: recall at temporal IoU thresholds, duplicate rate, boundary MAE/P95, event IoU,
  and action/evidence consistency.
- Integrity: camera/timeline coverage, no silent source loss, replay equivalence, terminal
  completeness, and no duplicate outbox publication.
- Efficiency: CPU-s/camera-hour, GPU-s/recording-hour, source/read/write bytes,
  unique/provider images, calls, tokens, dense-upgrade rate, and cost per accepted result.

Numeric promotion thresholds belong in a small versioned acceptance register after the
representative label set and risk owners exist. This cycle implements the metrics,
frozen local regressions, and Pareto reports without fabricating business numbers.

## Overall Roadmap

| Phase | Outcome | Main module(s) | Depends on | Local proof | External follow-up |
| --- | --- | --- | --- | --- | --- |
| P0 - measurement truth | One profile describes product units, call/image amplification, DB work, and stage resources | `qualification-ops` | none | profile unit tests plus fresh/replay report | none |
| P1 - transaction-scale stream scheduling | Scheduler work scales with emitted windows/work, not source-message count | `stream-control`, `source-media` | P0 | scheduler tests + same 40.83 s fresh profile | broker adapter later |
| P2 - bounded media and visual sentinel | One bounded media data path supplies integrity, cheap quality coverage, and selected evidence | `source-media`, `sampling-qa` | P1 | media unit/integration tests + exact 40.83 s profile | extended representative source and P9 hardware qualification |
| P3 - evidence and provider hot path | Persistent/batched evidence writes and real adapter batching remove per-record connection overhead | `inference-evidence`, `canonical-integration` | P0 | inference tests + mock/delayed-provider profile | real VLM endpoint |
| P4 - authoritative completion hot path | Completion keeps one durable truth while removing repeated full-result validation, serialization, and scans | `identity-delivery`, `canonical-integration` | P0, P3 | completion/outbox tests + fresh/replay profile | none |
| P5 - real pre-EOS execution | Window work executes the provider-neutral QA/event chain before EOS rather than only conformance terminals | `stream-control`, `canonical-integration`, `inference-evidence` | P1, P3, P4 | delayed-provider stream integration + crash replay | none |
| P6 - adaptive quality cascade | Complete 21-class clip QA and events use base coverage, suspicion upgrades, context, and abstention | `sampling-qa`, `event-semantics`, `canonical-integration` | P2, P5 | frozen fixture/Pareto integration | governed labels |
| P7 - recording-level parallel service | Multiple recording-affine workers feed bounded shared provider queues with backpressure | `canonical-integration`, `stream-control`, `qualification-ops` | P1-P6 | 1/2/4-worker local scaling report | production scheduler/broker |
| P8 - qualification package | Capacity and quality are reported together, with replay and failure evidence | `qualification-ops` | P6, P7 | local qualification report | representative data and hardware |
| P9 - target-SKU media adapter | Hardware decode implements the existing media ports without changing source/evidence semantics | `source-media` | P2, P8 | CPU/adapter port contract tests | target H100 SKU and NVDEC runtime |
| P10 - two-H100 provider qualification | The P3 provider adapter is configured and measured against the chosen real VLM | `inference-evidence`, `qualification-ops` | P3, P8 | local provider fixture and report harness | real endpoint and two H100s |
| P11 - production transport adapters | Object/broker transports preserve durable work, outbox, reconciliation, and review semantics | `stream-control`, `identity-delivery` | P4, P5, P8 | adapter contract and failure tests | chosen storage and broker |
| P12 - representative production qualification | The complete path proves the capacity envelope and signed quality thresholds | `qualification-ops`, `canonical-integration` | P6-P11 | qualification runner and local fixtures | representative labels, hardware, 24 h soak |

## Module and Cross-Module Phases

### `qualification-ops` - P0: measurement truth

**Result**

Every timing run can answer whether throughput changed because of media, scheduling,
SQLite, provider service, or horizontal concurrency. Capacity output uses recording,
camera, image, call, and token units correctly.

**Primary paths and entry points**

- `src/robata/runtime/canonical_profile.py` - profile schema and observer projection.
- `src/robata/runtime/capacity.py` - unit-safe capacity calculations.
- `scripts/profile_canonical_mcap.py` - real-media profile command.
- `tests/unit/test_canonical_profile.py` and `tests/unit/test_runtime_capacity.py`.

**Implementation outline**

1. Add explicit counts/rates for unique images, provider images, logical calls, HTTP
   requests, retries, batches, input/output tokens, and dense-upgrade fraction.
2. Attribute source bytes, SQLite read/write bytes, connection/transaction counts, and
   per-stage CPU time; report recording-hours and camera-hours separately.
3. Add a comparison output for fresh versus replay and one-versus-many recording workers.
4. Keep `LOCAL_CONFORMANCE`, representative benchmark, and production qualification
   visibly distinct.

**Keep intact**

- Existing profile exact-input manifest, evidence class, and production eligibility.

**Done when**

- [ ] A profile cannot report capacity without workload duration and provider mode.
- [ ] The report exposes every multiplier in the sampling/window/call formulas above.
- [ ] The current event-driven baseline can be reproduced and compared by stage.

**Run locally**

```powershell
python -m pytest tests/unit/test_canonical_profile.py tests/unit/test_runtime_capacity.py

# Local prerequisite; no upload is required. A clean checkout may need this sample restored.
$source = "data\source\sample-medium.mcap"
$expected = "9FD5094BF29CD4EE50CD8C7D8C053E89D1C93660A0F4E57DAAA726BAE2B6156C"
if ((Get-FileHash $source -Algorithm SHA256).Hash -ne $expected) { throw "source digest mismatch" }

.\.venv\Scripts\python.exe scripts\profile_canonical_mcap.py $source `
  --mapping-config config\genrobot-observed-v0.json `
  --allow-unapproved-profile `
  --state-dir tmp\profiles\blueprint-baseline-state `
  --run-key throughput-quality-blueprint-v1 `
  --max-duration-seconds 45 `
  --output tmp\profiles\blueprint-baseline-fresh.json

# Run the same command with the same state/run key and a replay output path.
.\.venv\Scripts\python.exe scripts\profile_canonical_mcap.py $source `
  --mapping-config config\genrobot-observed-v0.json `
  --allow-unapproved-profile `
  --state-dir tmp\profiles\blueprint-baseline-state `
  --run-key throughput-quality-blueprint-v1 `
  --max-duration-seconds 45 `
  --output tmp\profiles\blueprint-baseline-replay.json
```

**Next boundary:** P1-P4 and P7 use the same counters for acceptance.

### P1 - transaction-scale stream scheduling

**Participating modules:** `stream-control`, `source-media`

**End-to-end result:** Reading a source message updates only in-memory cursors until a
window or state transition is actually due. Emitting a window persists its window record,
work DAG, and publish state in bounded transactions without repeatedly scanning all
windows or pending work.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `stream-control` | finish the event-driven ready path; batch publication/ready transitions; indexed bounded lookups | `src/robata/adapters/sqlite_work_scheduler.py`, `sqlite_stream_work_ledger.py`, `src/robata/application/canonical/stream_scheduler.py` | `test_sqlite_work_scheduler.py`, `test_stream_scheduler_composition.py`, `test_local_stream_finalization.py` |
| `source-media` | call scheduler only at window/timeline events, not for every MCAP message | `src/robata/application/canonical/mcap_source.py` | `test_mcap_single_pass.py`, `test_canonical_mcap_source.py` |

**Implementation outline**

1. Complete the existing event-driven conversion by removing the remaining
   message-proportional `windows`, `pending_work_rows`, and execution-scope projections
   from the capture loop.
2. Reuse the existing atomic `append_window(..., work_plans=new_work)` operation; do not
   rebuild it. Batch the remaining per-row publication and ready-state work, prioritizing
   the measured `get_work`, `plan`, and `mark_published` transaction families.
3. Drain only work made ready by the latest transition and return the changed rows/cursors
   directly. Retain reconciliation scans for restart, not the hot loop.
4. Preserve leases, fences, idempotency keys, and deterministic work identities.

**Done when**

- [ ] Scheduler transactions on the 40.83 s fixture are at most 1,500 and do not grow
  with non-window MCAP message count.
- [ ] `source.stream.capture_publish` is at most 40 s on the recorded baseline machine,
  or an attributed lower-bound report identifies the next media cost.
- [ ] Fresh, replay, crash, and duplicate-window tests produce the same terminal graph.

**Combined proof**

```powershell
python -m pytest tests/unit/test_sqlite_work_scheduler.py tests/unit/test_stream_scheduler_composition.py tests/unit/test_local_stream_finalization.py tests/unit/test_mcap_single_pass.py
python -m pytest tests/integration/test_canonical_mcap_source.py tests/integration/test_canonical_local_command.py
```

**Compatibility notes:** no wire or identity change. If batching requires a persisted
shape change, use a new registered schema version rather than editing a published file.

### P2 - bounded media and visual sentinel

**Participating modules:** `source-media`, `sampling-qa`

**End-to-end result:** A source traversal creates the compressed camera/timeline authority,
cheap continuous quality facts, and a bounded set of selected evidence artifacts. The
pipeline avoids repeated full-file reads and never persists raw decoded RGB.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `source-media` | coalesce per-camera target timestamps; fuse selected materialization and sentinel decode where practical; reuse compressed spools/artifacts | `application/canonical/mcap_source.py`, `single_pass_video.py`, `bounded_media.py`, `media_quality.py`, `adapters/pyav_*`, `parallel_*` | media fast tests and real MCAP integration |
| `sampling-qa` | consume actual decoded image/luma representations; expose sentinel signals to the adaptive policy | `src/robata/sampling/adaptive.py`, `sampling/signals.py`, media-quality bridges | adaptive and supplemental QA tests |

**Implementation outline**

1. Keep whole-source packet/timeline integrity separate from expensive visual decode.
2. Define a small decoded-frame view used consistently by black/exposure/freeze/blur and
   adaptive detectors; remove assumptions that arbitrary encoded bytes are raw grayscale.
3. Decode each camera sequentially per required interval, satisfying all selected targets
   and sentinel observations in the same traversal where possible.
4. Make base sentinel cadence, resize, and evidence encoding configurable and visible in
   the profile. Keep original compressed bytes authoritative.
5. Keep decode/materialization behind the existing `ports/frame_materialization.py` and
   `ports/video_export.py` boundaries. P2 proves the PyAV CPU path and port equivalence;
   P9 supplies the target-SKU hardware implementation.

**Done when**

- [ ] Detector tests use the same pixel representation produced by the real media path.
- [ ] Black/freeze/exposure/blur fixture outcomes and selected semantic IDs do not regress.
- [ ] Read amplification on the 40.83 s fixture is at most 10x source bytes, with a target
  of 8x; any unavoidable codec/GOP floor is separately attributed.
- [ ] The complete 40.83 s `sample-medium.mcap` source can be profiled with bounded RSS
  and no decoded-RGB persistence.
- [ ] PyAV CPU materialization/export satisfies the existing media ports. A 180 s or longer
  representative source is useful extended evidence but is not a P2 local blocker.

**Combined proof**

```powershell
python -m pytest tests/unit/test_mcap_single_pass.py tests/unit/test_pyav_interval_spool.py tests/unit/test_bounded_media.py tests/unit/test_canonical_media_quality.py tests/unit/test_sampling_adaptive.py tests/unit/test_media_quality_supplemental.py
python -m pytest tests/integration/test_real_mcap_single_pass.py tests/integration/test_canonical_mcap_source.py
.\.venv\Scripts\python.exe scripts\profile_canonical_mcap.py data\source\sample-medium.mcap --mapping-config config\genrobot-observed-v0.json --allow-unapproved-profile --state-dir tmp\profiles\p2-media-state --run-key p2-media-v1 --max-duration-seconds 45 --output tmp\profiles\p2-media-fresh.json
```

**Compatibility notes:** artifact identities continue to bind exact source timestamps and
bytes. Encoding changes create new artifacts; they do not mutate published artifacts.

### P3 - evidence and provider hot path

**Participating modules:** `inference-evidence`, `canonical-integration`

**End-to-end result:** Inference uses long-lived run/stage resources, bounded concurrency,
and provider batching. Evidence remains replayable without opening a SQLite connection and
performing multiple read-backs for every individual row.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `inference-evidence` | persistent connection ownership; attempt/stage transactions; optional `infer_batch`; provider metrics | `adapters/sqlite_inference_evidence.py`, `inference/orchestrator.py`, `inference/adapter.py`, `inference/runpod.py` | inference ledger/orchestrator/RunPod tests |
| `canonical-integration` | pass bounded batch/deadline configuration and reuse returned evidence instead of redundant reads | `application/canonical/runner.py`, `runner_support.py`, `local_composition.py` | concurrency and canonical integration tests |

**Implementation outline**

1. Own one inference-evidence connection/writer for a run or stage; serialize SQLite writes
   through that owner instead of reconnecting per operation.
2. Preserve crash-visible checkpoints: intent before dispatch, raw response after receipt,
   and terminal/selection/accepted-lineage atomically after parse. Avoid read-back when the
   just-written typed value is already authoritative in memory.
3. Activate micro-batches only for compatible task/model/shape/deadline groups. Return
   outputs in deterministic call-part order regardless of provider completion order.
4. Implement RunPod batch/concurrency, timeout, retry, and partial-failure behavior against
   a local HTTP fixture; make single-request fallback explicit.
5. Keep completion/outbox databases on their existing authoritative durability settings.
   Do not trade away terminal recovery for a benchmark number.

**Done when**

- [ ] The 16-call fixture opens at most 12 inference-evidence connections.
- [ ] Inference-evidence transactions are at most 80 for the fixture while all accepted
  lineage and tamper/replay tests remain green.
- [ ] Mock `inference.pipeline` is at most 10 s on the baseline machine.
- [ ] A delayed local provider proves concurrency, batch boundaries, ordering, timeout,
  retry, and partial failure without a real external endpoint.

**Combined proof**

```powershell
python -m pytest tests/unit/test_inference_orchestrator.py tests/unit/test_sqlite_inference_evidence.py tests/unit/test_runpod_adapter.py tests/unit/test_canonical_offline_call_part_concurrency.py
python -m pytest tests/integration/test_canonical_offline.py
```

**Compatibility notes:** provider batching changes execution grouping, not input-plan or
evidence identity. Any formula change requires an explicit identity-policy version.

### P4 - authoritative completion hot path

**Participating modules:** `identity-delivery`, `canonical-integration`

**End-to-end result:** Primary completion, event/outbox publication, and recovery keep the
same authoritative semantics while avoiding repeated full-result serialization, database
integrity scans, and validation of facts already proven earlier in the run.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `identity-delivery` | incremental evidence seal, exact command bytes/digests, bounded transactional commit and reconciliation | `src/robata/adapters/sqlite_primary_completion.py`, `sqlite_event_identity_registry.py`, `sqlite_outbox.py`, `src/robata/application/canonical/local_outbox_delivery.py` | identity, primary-completion, outbox, and review tests |
| `canonical-integration` | carry forward prevalidated typed values/bytes and avoid reconstructing the terminal command multiple times | `application/canonical/local_composition.py`, `primary_completion.py`, `result_validation.py`, `logical_nodes.py` | canonical membership and local command tests |

**Implementation outline**

1. Attribute `completion.evidence.audit`, `completion.command.serialize_validate`, and the
   internal `completion.commit` substeps separately before changing behavior.
2. Carry canonical bytes, digests, prepared identities, and validated evidence references
   from their producing stage instead of serializing/parsing the same full model again.
3. Replace a whole-ledger integrity scan at completion with an incrementally maintained
   seal or run-scoped verification set, while retaining a full offline integrity command.
4. Keep the authoritative completion/outbox transaction and reconciliation semantics
   unchanged. Optimize the data supplied to it, not the meaning of commit.

**Done when**

- [ ] The combined evidence-audit, command-build, and completion-commit spans are at most
  8 s on the 40.83 s baseline machine, compared with about 14 s today.
- [ ] Fresh and exact replay produce identical command digest, event/revision identities,
  outbox rows, review routing, and final result.
- [ ] A crash before and after the authoritative transaction reconciles without a lost or
  duplicate terminal publication.
- [ ] Full offline integrity verification remains available and detects tampering.

**Combined proof**

```powershell
python -m pytest tests/unit/test_event_identity_registry.py tests/unit/test_sqlite_event_identity_registry.py tests/unit/test_admission_ledgers.py tests/unit/test_review_routing.py tests/unit/test_canonical_run_membership.py
python -m pytest tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py tests/integration/test_canonical_local_review_routing.py tests/integration/test_canonical_local_command.py
```

**Compatibility notes:** no schema, identity, or commit-point change is intended. If a
seal becomes persisted wire state, register a new schema version before implementation.

### P5 - real provider-neutral pre-EOS execution

**Participating modules:** `stream-control`, `canonical-integration`, `inference-evidence`

**End-to-end result:** When a source window closes, its ready QA/event work can run through
the real provider-neutral executor before recording EOS. Local conformance terminals remain
available for fast deterministic tests but are no longer the only incremental executor.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `stream-control` | expose ready work and accept real typed terminals with bounded leases/backpressure | `application/canonical/stream_scheduler.py`, `durable_work.py`, SQLite scheduler adapters | scheduler/recovery tests |
| `canonical-integration` | window executor reuses canonical call/reduction components and joins final closure | `runner.py`, `local_composition.py`, `local_stream_finalization.py`, `stream_recording_reduction.py` | local command/offline integration |
| `inference-evidence` | dispatch and persist window-scoped real or fixture provider evidence | orchestrator and evidence ledger | inference integration |

**Implementation outline**

1. Extract/reuse the smallest canonical stage executor rather than duplicating QA/event
   logic inside the stream scheduler.
2. Start eligible work as soon as window dependencies close; use bounded provider and DB
   queues so source capture cannot create unbounded memory or lease churn.
3. At EOS, close only the remaining graph and reduce already completed window evidence.
4. On restart, claim unfinished work, reuse persisted evidence, and avoid duplicate provider
   dispatch when a valid terminal already exists.

**Done when**

- [ ] A delayed-provider fixture records at least one typed QA terminal through the
  pre-EOS executor before EOS.
- [ ] Final clip/event output is identical between uninterrupted and crash/replay runs.
- [ ] Duplicate window delivery does not duplicate provider evidence, terminal completion,
  outbox rows, or review tasks.
- [ ] The conformance mock remains a selectable local test mode and is labelled as such.

**Combined proof**

```powershell
python -m pytest tests/unit/test_stream_scheduler_composition.py tests/unit/test_local_stream_finalization.py tests/unit/test_inference_orchestrator.py tests/unit/test_review_routing.py
python -m pytest tests/integration/test_canonical_local_command.py tests/integration/test_canonical_offline.py tests/integration/test_sqlite_outbox_relay.py tests/integration/test_canonical_local_review_routing.py
```

**Compatibility notes:** no new result semantics. The phase moves execution earlier and
reuses existing terminals; it must not create a second canonical result path.

### P6 - adaptive quality cascade

**Participating modules:** `sampling-qa`, `event-semantics`, `canonical-integration`

**End-to-end result:** Every eligible clip returns a complete 21-class QA result while
dense/context/event work is concentrated on suspicious or uncertain intervals. Short
events and boundary context retain exact evidence links.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `sampling-qa` | base coverage, signal upgrades, neighbour selection, coarse/dense reduction, 21-class clip projection | `sampling/adaptive.py`, `sampling/signals.py`, `qa_pipeline/coarse.py`, `dense.py` | adaptive, QA core, product tests |
| `event-semantics` | candidate-centered evidence and onset/offset refinement | `event_pipeline/candidate.py`, `evidence.py`, `proposer.py` | event core/projection tests |
| `canonical-integration` | one clip-level reduction and abstention/review behavior | `runner.py`, `reduction.py`, `output_admission.py`, supplemental bridges | canonical integrations |

**Implementation outline**

1. Define a base coverage budget for every camera/clip and explicit upgrade reasons:
   source quality signal, coarse uncertainty, cross-camera disagreement, event candidate,
   or boundary refinement.
2. Select best nearby frames without erasing the original bad-frame observation. Include
   pre/post context and retain exact timestamps used by every call.
3. Use full-frame context for product QA; add ROI crops only as supplemental evidence.
4. Reuse the existing `ProductQAIssue` 21-class vocabulary and `QAClassifier`; do not
   recreate the taxonomy. Integrate structural, local visual, coarse, dense, and event
   facts into one complete clip result with explicit unknown/incomplete/abstained states.
5. Emit the existing benchmark metrics across sampling/dense-rate configurations.

**Done when**

- [ ] All 21 product classes have deterministic projection coverage for every eligible
  clip, including no-issue, abstained, and incomplete-input cases.
- [ ] Black/blur/freeze/exposure/obstruction fixtures exercise base and upgrade paths.
- [ ] A short event near a window edge pulls pre/post evidence and completes boundary
  refinement without breaking upstream closure.
- [ ] A local Pareto report compares at least three sampling/dense policies and reports
  quality metrics plus images/calls/CPU cost. No mock score is called production quality.

**Combined proof**

```powershell
python -m pytest tests/unit/test_sampling_adaptive.py tests/unit/test_adaptive_sampler_runtime.py tests/unit/test_qa_pipeline_core.py tests/unit/test_local_qa_product.py tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py tests/unit/test_supplemental_temporal_package.py
python -m pytest tests/integration/test_canonical_offline.py tests/integration/test_canonical_action_event_revision.py
```

**Compatibility notes:** if sampling or projection participates in a hash/logical key,
version the policy before changing the formula. Ordinary runtime thresholds remain config.

### P7 - recording-level parallel service and backpressure

**Participating modules:** `canonical-integration`, `stream-control`, `qualification-ops`

**End-to-end result:** A local service can process multiple recordings concurrently, keep
each recording's durable state affine to one worker/NVMe location, share a bounded provider
client, and expose stable queue/backlog behavior.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `canonical-integration` | recording worker lifecycle and shared provider composition | `local_composition.py`, canonical command/runner entry points | concurrent local command tests |
| `stream-control` | bounded ingress/provider/publish queues, admission, retry, cancellation, and drain | scheduler/queue adapters | work-state tests |
| `qualification-ops` | 1/2/4/N worker harness and saturation report | `runtime/benchmark.py`, `capacity.py`, profile script | benchmark tests |

**Implementation outline**

1. Shard by recording identity; never open one recording's SQLite state from multiple
   hosts or place WAL files on network storage.
2. Bound each queue and propagate backpressure to ingress. Separate CPU media concurrency
   from provider concurrency and completion/outbox workers.
3. Prefer processes for recording-level isolation; keep camera-level thread pools inside
   the recording worker where PyAV/encoding already scales.
4. Measure 1, 2, and 4 workers, then increase until CPU, NVMe, RAM, or provider queue is
   the identified saturation point.

**Done when**

- [ ] Four concurrent local recordings finish without cross-run identity/state leakage.
- [ ] Four-worker throughput is at least 2.5x one-worker throughput on the same workload,
  or the profile proves a named shared resource limit that the next deployment must size.
- [ ] Queue sizes remain bounded, backlog drains after a burst, and cancellation/restart
  leaves replayable work.
- [ ] The capacity report converts measured worker throughput into required CPU/NVMe
  worker count for 25 recording-RTF without calling the projection a production proof.

**Combined proof**

```powershell
python -m pytest tests/unit/test_local_streaming_benchmark.py tests/unit/test_runtime_capacity.py tests/unit/test_sqlite_work_scheduler.py
python -m pytest tests/integration/test_canonical_local_command.py
```

**Compatibility notes:** process topology is not part of canonical identity.

### `qualification-ops` - P8: quality-capacity qualification package

**Result**

One report shows the quality/throughput Pareto frontier for a fixed code, sampler, prompt,
model/provider, hardware, and dataset manifest. It is impossible to claim 500 h/day from a
mock-only or single-stage result.

**Primary paths and entry points**

- `src/robata/benchmark/metrics.py`, `splits.py`, and `promotion.py`.
- `src/robata/runtime/capacity.py`, `canonical_profile.py`, and benchmark helpers.
- `tests/unit/test_benchmark*.py`, `test_runtime_capacity.py`.

**Implementation outline**

1. Produce frozen workload manifests and leakage-safe recording/session splits.
2. Run the policy matrix: base FPS, dense FPS, upgrade fraction, window/hop, batch size,
   provider concurrency, and model configuration.
3. Report quality, integrity, latency, backlog, CPU/GPU/NVMe, and call/token amplification
   in one artifact.
4. Add restart/retry/provider-timeout scenarios and confirm terminal/outbox reconciliation.
5. Leave business thresholds empty until the representative label owners sign them.

**Keep intact**

- Existing benchmark metric definitions and evidence-class distinctions.

**Done when**

- [ ] Local fixtures produce a reproducible Pareto report and catch metric regressions.
- [ ] Representative-data and real-hardware rows are clearly `NOT_MEASURED` until run.
- [ ] Every capacity claim names recording count/duration, six-camera load, model mode,
  hardware, concurrency, and run duration.

**Run locally**

```powershell
python -m pytest tests/unit/test_benchmark.py tests/unit/test_benchmark_metrics_promotion.py tests/unit/test_benchmark_splits.py tests/unit/test_runtime_capacity.py tests/unit/test_canonical_profile.py
```

**Next boundary:** P9-P12 fill the hardware, provider, transport, label, and soak rows.

### `source-media` - P9: target-SKU media adapter

**Result**

An NVDEC/DeepStream implementation, when selected, satisfies the same frame
materialization and video-export ports already proven by P2. This phase does not redesign
sampling, artifact identity, or CPU fallback behavior.

**Primary paths and entry points**

- `src/robata/ports/frame_materialization.py` and `video_export.py` - existing boundaries.
- `src/robata/adapters/pyav_frame_materializer.py` and `pyav_mp4_exporter.py` - CPU reference.
- `src/robata/adapters/nvdec_frame_materializer.py` and `nvdec_video_export.py` - planned
  target-SKU implementations if the deployment selects this path.
- Source-media fast tests plus a planned hardware-marked adapter test.

**Implementation outline**

1. Inventory the exact H100 SKU, driver, Video Codec SDK/DeepStream version, codecs,
   resolutions, FPS, GOP distributions, and host-to-device transfer path.
2. Implement the existing ports; do not add a second source or artifact identity model.
3. Batch compatible streams, expose fallback/error facts, and retain PyAV for unsupported
   inputs and local development.
4. Benchmark decode-only, decode+resize, and decode+materialize separately against the
   required 125 camera-seconds/s average and 150 camera-seconds/s 20%-margin target.

**Done when**

- [ ] CPU and GPU adapters pass the same semantic timestamp/artifact contract tests.
- [ ] Unsupported codecs and device failure fall back or fail explicitly without silent
  frame loss.
- [ ] The target-SKU report states streams, codec, resolution, FPS, GOP, transfer path,
  utilization, errors, and sustained camera-seconds/s.

**Run locally before hardware**

```powershell
python -m pytest tests/unit/test_pyav_interval_spool.py tests/unit/test_bounded_media.py tests/unit/test_mcap_single_pass.py tests/integration/test_real_mcap_single_pass.py
```

**Next boundary:** P12 consumes the measured media capacity. Hardware is the only blocker.

### P10 - two-H100 real-provider qualification

**Participating modules:** `inference-evidence`, `qualification-ops`

**End-to-end result:** The RunPod/provider behavior completed in P3 is configured, not
reimplemented, for the chosen VLM and two-H100 topology, then measured under the exact
adaptive workload produced by P6.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `inference-evidence` | real endpoint configuration, response compatibility fixes, batch/concurrency limits, metrics | `src/robata/inference/runpod.py`, `adapter.py`, `orchestrator.py` | existing RunPod/orchestrator tests plus recorded-response fixtures |
| `qualification-ops` | saturation matrix and GPU/call/token report | `src/robata/runtime/capacity.py`, `canonical_profile.py`, benchmark helpers | runtime/benchmark tests |

**Implementation outline**

1. Fix model version, precision/quantization, inference engine, card topology, maximum
   images, prompt/context/output limits, batch policy, and concurrency.
2. Configure authentication and endpoint values outside committed source. P3 already owns
   request mapping, timeout, retry, batching, and local mock-server behavior.
3. Measure GPU-s/recording-hour, provider images/s, calls/s, input/output tokens/s, queue
   time, TTFT, end-to-end stage latency, P50/P95/P99, KV cache, OOM, retries, memory, and
   utilization across the sampling policy matrix.
4. Compare two single-card replicas with two-card tensor parallel only if the model and
   engine support both topologies.

**Done when**

- [ ] Recorded real responses replay through the same evidence and result validators.
- [ ] A measured saturation curve identifies the safe calls/images/tokens envelope and
  aggregate GPU-minutes/recording-hour.
- [ ] Provider restart, timeout, partial batch failure, and retry do not duplicate accepted
  evidence or terminal output.

**Combined proof**

```powershell
python -m pytest tests/unit/test_runpod_adapter.py tests/unit/test_inference_orchestrator.py tests/unit/test_sqlite_inference_evidence.py tests/unit/test_canonical_offline_call_part_concurrency.py
```

**Compatibility notes:** real credentials and responses are external. Contract or identity
changes discovered here require the normal version decision; they are not patched in place.

### P11 - production transport adapters

**Participating modules:** `stream-control`, `identity-delivery`

**End-to-end result:** The selected object store and broker implement existing durable work,
artifact, outbox, acknowledgement, reconciliation, and review behavior without changing
canonical result semantics.

**Change map**

| Module | Expected contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `stream-control` | broker task adapter with leases, fences, retry, DLQ, and backpressure | `src/robata/ports/task_queue.py`, `src/robata/queue/redis_adapter.py`, scheduler ports/adapters | queue and scheduler contract tests |
| `identity-delivery` | object/outbox delivery adapter, idempotent acknowledgement, reconciliation, review handoff | `src/robata/application/canonical/local_outbox_delivery.py`, `src/robata/adapters/sqlite_outbox.py`, artifact/review ports | outbox, completion, and review tests |

**Implementation outline**

1. Choose services and map them to the existing ports; keep the local SQLite adapters as
   deterministic recovery references.
2. Make enqueue/publish/acknowledge fail closed, idempotent, and observable. Do not revive
   a skeleton adapter that returns false success.
3. Exercise duplicate delivery, delayed acknowledgement, lease expiry, broker restart,
   object visibility delay, DLQ, and reconciliation with local service fixtures first.
4. Register a new wire schema only if the selected transport requires a genuinely new
   published message shape.

**Done when**

- [ ] Contract tests run against local and production adapter implementations.
- [ ] Restart and duplicate-delivery scenarios create no lost/duplicate completion,
  outbox delivery, or review task.
- [ ] Queue/backlog and delivery latency are exposed to the P12 qualification report.

**Combined local proof**

```powershell
python -m pytest tests/unit/test_sqlite_work_scheduler.py tests/unit/test_review_routing.py tests/unit/test_sqlite_event_identity_registry.py
python -m pytest tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py tests/integration/test_canonical_local_review_routing.py
```

**Compatibility notes:** transport locators must not enter logical identity. Published
message changes use atomic schema registration and a new version.

### P12 - representative 500 h/day production qualification

**Participating modules:** `qualification-ops`, `canonical-integration`

**End-to-end result:** The unchanged canonical pipeline is exercised with the chosen VLM,
two H100s, representative media/labels, and selected storage/broker endpoints. One report
contains capacity, quality, recovery, and resource evidence.

**Primary paths and entry points**

- `src/robata/runtime/capacity.py` and `canonical_profile.py` - capacity evidence.
- `src/robata/benchmark/**` - quality metrics, splits, and promotion evidence.
- `scripts/profile_canonical_mcap.py` and the P8 qualification runner/report.
- Canonical fresh/replay, completion, outbox, and review integrations.

**Qualification matrix**

1. Freeze exact code, schemas/catalog, sampler, prompt, model/engine, hardware, media,
   labels/splits, adapter versions, and arrival distribution.
2. Run representative codec/resolution/FPS/GOP combinations, sampling policies, and
   measured arrival peaks.
3. Exercise restart, provider failure, broker/object-store failure, backlog drain, and at
   least one 24-hour soak.
4. Evaluate the governed quality split and fill the versioned acceptance thresholds.

**Done when**

- [ ] Sustained service is at least 25 recording-RTF with stable backlog on the agreed
  representative workload. This is a 20% service-margin target and does not imply a 70%
  GPU-utilization envelope.
- [ ] A separate preferred operating-envelope target is: at nominal 500 h/day, average
  GPU utilization is at most 70%, supported by a measured saturation curve showing at
  least 29.762 recording-RTF at the agreed workload mix. The associated 4.03 aggregate
  GPU-minutes/recording-hour is a qualification budget, not inferred H100 capability.
- [ ] Product-signed QA/event/calibration thresholds pass on leakage-safe labels.
- [ ] A restart/provider/transport failure creates no lost or duplicate authoritative
  terminals, outbox deliveries, or review tasks.

**Combined local proof before external execution**

```powershell
python -m pytest tests/unit/test_benchmark.py tests/unit/test_runtime_capacity.py tests/unit/test_canonical_profile.py tests/unit/test_runpod_adapter.py tests/unit/test_mcap_single_pass.py tests/unit/test_sqlite_work_scheduler.py tests/unit/test_review_routing.py
python -m pytest tests/integration/test_canonical_local_command.py tests/integration/test_sqlite_outbox_relay.py tests/integration/test_sqlite_primary_completion.py tests/integration/test_canonical_local_review_routing.py
```

**Compatibility notes:** P12 qualifies the assembled system; it does not introduce a new
canonical path or silently relax a failed quality/capacity threshold.

## Blockers and External Dependencies

P9-P12 require conditions that cannot be settled entirely from the repository.

| Condition | What can still be completed locally | Temporary substitute | Later external proof |
| --- | --- | --- | --- |
| Chosen VLM and two H100s | adapter, batch/concurrency, evidence, timeout, metrics, replay | delayed deterministic HTTP fixture | real model capacity/quality matrix |
| Hardware decode environment | decode port and CPU equivalence tests | PyAV 18 CPU fallback | codec/resolution/NVDEC benchmark |
| Production object storage/broker | ports, idempotency, failure/reconciliation tests | local artifact store and SQLite outbox | load, failure, retention, recovery |
| Representative governed videos and labels | metric/split/Pareto framework and local fixtures | frozen repository fixtures | signed quality evaluation |
| Arrival peaks and deadline policy | formula, synthetic burst/load harness | explicit scenario values | production traffic/operations evidence |
| Long-running hardware | soak/failure runner | short local stress | 24-hour or longer representative soak |

## Research Basis and Adopted Lessons

The blueprint uses official vendor documentation for implementation patterns, not for an
unmeasured Robata capacity claim:

- [NVIDIA Triton dynamic batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html): combine compatible requests, bound queue delay, and expose queue policy.
- [NVIDIA Triton model instances](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html): use explicit model instance groups for concurrent execution.
- [NVIDIA NVDEC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html): parallel decode contexts and codec-specific decode behavior require direct measurement.
- [NVIDIA H100 product specifications](https://www.nvidia.com/en-us/data-center/h100/): SKU documentation identifies available hardware features; it is not evidence of Robata decode or inference capacity.
- [NVIDIA DeepStream performance guidance](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html): disable rendering/OSD work in throughput-only pipelines and measure the intended stream configuration.
- [vLLM documentation](https://docs.vllm.ai/en/stable/): continuous batching and prefix/cache features are candidate serving mechanisms for supported multimodal models.
- [vLLM scaling guidance](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html): prefer single-GPU replicas when the model fits; use tensor/pipeline/data parallelism according to model and hardware constraints.
- [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics.html): retain queue, time-to-first-token, inter-token, end-to-end, token, and cache measurements.
- [SQLite WAL documentation](https://sqlite.org/wal.html): WAL improves same-host reader/writer concurrency but is not a network-filesystem coordination mechanism.

## Acceptance and Verification

- [ ] Each completed phase runs only its focused tests plus the named small integration.
- [ ] A throughput claim includes a before/after profile on the same workload and machine.
- [ ] A quality claim includes the dataset/fixture, split, metric, and model/sampling version.
- [ ] Schema/identity checks run only when a phase changes those surfaces.
- [ ] External limits stay explicit; local mocks do not become production evidence.
- [ ] Final independent acceptance runs the full relevant suite and fresh/replay/soak
  evidence after the dispatched phases are merged.

## Suggested Dispatch Prompts

Open one window with one of these compact phase names:

```text
qualification-ops / P0 - measurement truth
stream-control + source-media / P1 - transaction-scale stream scheduling
source-media + sampling-qa / P2 - bounded media and visual sentinel
inference-evidence + canonical-integration / P3 - evidence and provider hot path
identity-delivery + canonical-integration / P4 - authoritative completion hot path
stream-control + canonical-integration + inference-evidence / P5 - real pre-EOS execution
sampling-qa + event-semantics + canonical-integration / P6 - adaptive quality cascade
canonical-integration + stream-control + qualification-ops / P7 - recording-level parallel service
qualification-ops / P8 - quality-capacity qualification package
source-media / P9 - target-SKU media adapter
inference-evidence + qualification-ops / P10 - two-H100 real-provider qualification
stream-control + identity-delivery / P11 - production transport adapters
qualification-ops + canonical-integration / P12 - representative 500 h/day qualification
```

For each window: read `AGENTS.md`, this phase, and the named module cards; modify only the
named paths unless a concrete dependency is discovered; run the listed local proof; report
changed files, command results, measured result, and any truly external blocker.
