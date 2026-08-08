# Robata Mage Stream-Oriented Perception vNext Blueprint

**Cycle:** replace the window-oriented multi-invocation DAG with Mage-native continuous perception
**Planning date:** 2026-08-07
**Dispatch unit:** one numbered phase per focused development window
**Default implementation branch:** `codex/mage-stream-vnext-20260807`

## Authority and Evidence Boundary

This file is a local execution map, not a product contract, migration approval, or production certificate.

- Published schemas and `schemas/schema-catalog.json` govern wire contracts; tracked source,
  tests, and conformance fixtures govern executable behavior.
- Released schema bytes are immutable. New identity, hash, logical-key, idempotency-key,
  fence, projection, closure, or wire semantics require a new family or registered successor.
- Existing window contracts remain readable. vNext must not reinterpret
  `incremental-window@1.0.0`, `stream-window-result@1.0.0`, or their hashes.
- Mage recurrent state, visual features, KV cache, and gate state are ephemeral accelerators.
  Durable authority is media/segment hashes, an explicit context manifest, model/policy
  revisions, raw inference artifacts, and deterministic projections.

## Current Concrete Baseline

The repository has strong streaming and durability primitives, but the active semantic path
is still window-oriented:

- `application/canonical/bounded_media.py` defaults to 1-second logical segments and
  overlapping 2-second windows with a 1-second hop.
- `contracts/stream_common.py` is provider-neutral at the stream layer, but its purposes and
  stages mirror separate QA, event, action-dense, and boundary invocations.
- `application/canonical/stream_scheduler.py` uses `stream-window-dag-v4` with
  `WINDOW -> QA_COARSE -> QA_DENSE -> EVENT_PROPOSAL -> WINDOW_REDUCTION`; its execution
  projection maps back to `Stage.QWEN_*` in `queue/stage.py`.
- `application/canonical/pre_eos_execution.py` executes separate `VisionTask` calls.
- `inference/local_hf_runtime.py` and `local_hf_endpoint.py` are image-only. The released
  `local-hf-vision-request-v1` contains PNGs, not native video/codec context.
- `application/canonical/local_real_model.py` binds six Qwen-named task policies.
- `event_pipeline/fusion.py` preserves useful evidence judging, but v1 confidence uses
  `sum(reliabilities) / 6` even when fewer cameras may be observable or selected.

A local Mage run through the old physical DAG on the 40.8335-second, six-camera fixture made
164 fresh generations: 41 coarse-QA, 41 event-proposal, and 82 action-evidence calls. It took
1,164.389 seconds total, with 970.639 seconds in generation and provider-dispatch GPU use
averaging about 94.4%. This is local architecture evidence, not production capacity evidence:
the GPU was busy, while repeated business-stage generation dominated.

## Product Outcome

**Outcome:** Robata consumes aligned codec streams once, emits one durable
`MageObservation` per non-overlapping perception segment, deterministically projects
QA/event/evidence views, reconciles cross-segment event tracks, invokes Mage refinement only
for explicit ambiguity, and publishes facts with complete lineage and recovery evidence.

**Why now:** Mage's native codec/video and recurrent streaming design invalidates the old
assumption that every business judgment needs a separate VLM call.

**Success measures:**

- The default vNext route has one normal expensive stage, `PERCEPTION_OBSERVE`, plus
  exceptional `PERCEPTION_REFINE`; all other physical stages are deterministic.
- With an initial 8-second policy, the 40.8335-second fixture schedules at most
  `ceil(duration / 8s) = 6` normal Mage generations plus separately reported refinements,
  and zero Qwen QA/event/evidence/boundary calls.
- Normal perception uses native codec/video inputs, not PNG extraction or mosaic re-encode.
- One observation produces independent QA, event, and evidence facts without another model
  call; projector replay is deterministic.
- Events may cross any number of segment boundaries without duplicate final events.
- Exact replay reads accepted raw artifact bytes. Model recomputation is never called
  byte-exact replay.
- Cache/hidden-state loss is recoverable from durable context manifests and bounded segment
  replay without changing accepted artifact identity.
- Fusion normalizes over an explicit observable/selected population, not a fixed six.
- Legacy window runs stay readable and rollback-capable, but Qwen is not loaded by default.

**Non-goals:** no `web/` changes; no physical deletion of
`D:\HuggingFace\Qwen3-VL-4B-Instruct`; no production gate admission or gate training; no
2x3 mosaic as production multi-view; no durable hidden tensors; no local claim of H100,
R2, PostgreSQL/Supabase, long-run, or production qualification; no deletion/rewrite of
legacy schemas or rows.

## Target Architecture and Time Model

```text
MCAP / camera codec streams
  -> INGEST + ALIGNMENT
  -> NON-OVERLAP SEGMENTS -----------------> deterministic MEDIA_HEALTH
  -> PERCEPTION CONTEXT MANIFEST
  -> MAGE NATIVE CODEC ENCODE (per camera / batched)
  -> ONE MULTI-VIEW DECODER GENERATION
  -> MageObservation V1
       -> QA PROJECTOR
       -> EVENT PROJECTOR
       -> EVIDENCE PROJECTOR
  -> TEMPORAL RECONCILE -> MULTI-VIEW FUSION
       -> RESOLVED, or targeted MAGE_REFINE when AMBIGUOUS
  -> FINALIZE + PUBLISH
```

Three time concepts remain distinct:

1. **Storage/perception segment:** immutable, non-overlapping media boundary with absolute
   time and content identity. First production candidate: 8 seconds plus a shorter tail.
2. **Inference context:** current segment plus a bounded, explicit prior context manifest;
   cache state is optional and reconstructible.
3. **Event interval/track:** arbitrary onset/offset spanning any segments.

A later policy may separate a 2-second gate scan from an 8-12-second reasoning horizon. That
is a versioned policy change, not a return to overlapping full generations.

## Contract, Schema, and Identity Decisions

P1 registers additive contracts before publication:

| Contract | Release decision | Purpose |
| --- | --- | --- |
| `stream-segment@1.0.0` | Reuse unchanged where sufficient | Immutable media membership and packet/artifact evidence |
| `perception-context-manifest@1.0.0` | New family | Ordered current/prior segments, camera membership/absence, context/codec policies |
| `mage-observation@1.0.0` | New family | Model-level QA/action/camera/boundary/gate observations, not final facts |
| `observation-projection-bundle@1.0.0` | New family | Deterministic QA/event/evidence references |
| `event-track@1.0.0` | New family | `CANDIDATE -> OPEN -> UPDATED -> CLOSED -> FINALIZED` lineage |
| `perception-refine-request/result@1.0.0` | New families | Narrow reason-coded refinement and bounded context |
| `perception-terminal-closure@1.0.0` | New family | Segment/context membership and terminal outcomes, not window closure |
| `local-stream-recording-result@5.0.0` | Successor only if result root changes | vNext result; v1-v4 stay immutable |
| `primary-completion-record@4.0.0` | Conditional successor only if completion root changes | Reuse v3 exactly when possible; otherwise version/migrate explicitly |

The local Mage codec endpoint drafts named `*-v1` were never registered in
`schemas/schema-catalog.json`, never released, and have no production or migration-bearing durable
rows. They are discarded pre-release rather than mutated or dual-read. The first supported endpoint
family is therefore explicitly versioned as follows:

| Endpoint/runtime contract | Release decision | Identity or migration rule |
| --- | --- | --- |
| `mage-video-model-identity-v2` | First supported family; unreleased v1 draft discarded | Binds checkpoint-manifest digest and versioned resident runtime identity, including the exact load profile (`native_bf16_v1` or `bitsandbytes_4bit_nf4_v1`) |
| `mage-video-codec-request-v2` / `mage-video-codec-response-v2` | First supported wire family at `/v2/mage-video/infer` | Request binds input/context/codec plus decoder, prompt, output-schema and generation policy; response binds that inference identity and an exact result-artifact digest |
| `mage-video-codec-policy-v2` | First supported codec policy; unreleased v1 draft discarded | Binds traditional/neural engine controls and an explicit preprocessing device (`cpu` or `cuda`) so local DCVC CPU preparation and production CUDA preparation cannot share an identity |
| `mage-video-result-artifact-v2` | First supported result family | Service authors `created_at`; raw model output remains semantic-only and cannot author provenance timestamps |
| `mage-video-unified-observation-prompt-contract-v2` | Compact first supported prompt/output policy | Mage returns only selected-camera semantic observations; Robata deterministically supplies known camera IDs, segment hashes, six-slot expansion and business projections |
| `mage-video-idempotency-policy-v2` | New isolated key space/table | Exact request-body replay only; no draft-v1 rows exist to migrate and no v1 reader is retained |

This pre-release decision is intentionally one-way: implementation and tests must delete/rewrite the
unreleased v1 draft names rather than silently changing their meaning, but no published artifact or
durable row may be reinterpreted. Numeric strings are normalized only for an enumerated set of known
numeric leaves before strict validation; duplicate keys, unknown fields, non-finite numbers and
non-integral interval bounds remain errors.

The candidate catalog entries and files for `perception-context-manifest@1.0.0` and
`mage-observation@1.0.0` were created only in this uncommitted construction worktree and have never
been released, tagged, consumed, or persisted outside its tests. Their first generated bytes
incorrectly erased `SixCameraMap` value types and exact six-key cardinality. The explicit pre-release
decision is to discard those two invalid candidate byte sets/catalog pins and rerun the registered
exporter and registration workflow to produce the first valid `1.0.0` release. There is no
predecessor, upcaster, dual-reader, or durable-row migration because the invalid candidate never became
a product contract. Once the corrected bytes are committed, normal immutability applies.

The vNext execution vocabulary is provider-neutral:

```text
MEDIA_SCAN
PERCEPTION_OBSERVE
OBSERVATION_PROJECT
TEMPORAL_RECONCILE
FUSION
PERCEPTION_REFINE
FINALIZE
```

Implement a new typed enum/contract. Do not rename `Stage.QWEN_*` or reinterpret persisted
legacy rows. Initial identities are semantic hashes of source recording, ordered current and
prior segment hashes, camera context, model/checkpoint revision, codec/context/prompt policy,
and schema/projection version. Projection IDs hash `observation_id + projector_version`;
refine IDs hash `track/hypothesis + reason + bounded context + policy`. Locators, attempts,
leases, workers, timestamps, caches, and GPU state remain outside semantic identity. Raw
response bytes receive a separate exact SHA-256 and CAS identity.

## Overall Roadmap

| Phase | Outcome | Main module(s) | Depends on | Local proof | External follow-up |
| --- | --- | --- | --- | --- | --- |
| P0 - Freeze decisions/baseline | v1/vNext boundary and old-DAG baseline are explicit | `contract-governance`, `qualification-ops` | none | contract decision and profile tests | none |
| P1 - Add vNext contracts | Context/observation/track/refine/closure wire and identity rules | `contract-governance` | P0 | schema/catalog/immutability/upcasting | none |
| P2 - Non-overlap media/health | Immutable perception segments and deterministic health replace inference windows | `source-media`, `sampling-qa` | P1 | bounded media, MCAP, health tests | target codec matrix |
| P3 - Mage codec adapter | Mage loads without Qwen and consumes native video/codec via a versioned endpoint | `inference-evidence`, `source-media` | P1-P2 | fake endpoint plus conditional real Mage smoke | target GPU/runtime |
| P4 - One observation, three views | One generation creates deterministic QA/event/evidence projections | `inference-evidence`, `sampling-qa`, `event-semantics` | P1-P3 | projector determinism and raw replay | governed labels |
| P5 - Tracks and refine | Cross-segment events reconcile; only ambiguity triggers bounded refine | `event-semantics`, `inference-evidence` | P4 | track/restart/refine identity tests | boundary evaluation |
| P6 - Fusion v2 | Confidence uses declared observable/selected cameras | `event-semantics`, `sampling-qa` | P4-P5 | v1 compatibility and denominator tests | calibration |
| P7 - Perception scheduler | Durable segment DAG replaces default window DAG | `stream-control`, `canonical-integration` | P2-P6 | scheduler/recovery/local command | broker/load |
| P8 - Evidence/recovery | Observation-to-publication lineage and cache-loss recovery are durable | `inference-evidence`, `identity-delivery`, `canonical-integration` | P7 | crash matrix, outbox, completion, replay | production storage faults |
| P9 - Migration/rollback | Mage vNext is default; legacy is readable and rollback is explicit | `contract-governance`, `canonical-integration`, `identity-delivery` | P8 | dual-read/route/rollback tests | later removal decision |
| P10 - Qualification | Call reduction, quality, latency, resource, and recovery are measured | `qualification-ops`, all modules | P9 | full fixture profile and isolated CI shards | H100/R2/soak |

## Module and Cross-Module Phases

### P0 - Freeze v1/vNext Decisions and the Architecture Baseline

**Participating modules:** `contract-governance`, `qualification-ops`

**End-to-end result:** Implementation begins from a checked decision record rather than
silently mutating window contracts or comparing non-equivalent runs.

**Primary paths and entry points**

- `src/robata/contracts/phase_contract_decisions.py` - record `REUSE`, `NEW_FAMILY`, or
  `SUCCESSOR_VERSION` for each wire/root decision.
- `src/robata/runtime/canonical_profile.py` and `src/robata/benchmark/metrics.py` - retain
  separate source, provider, generation, projection, refinement, and publication spans.
- `tests/unit/test_schema_immutability.py` and `test_canonical_profile.py` - immutable bytes
  and metric units.

**Implementation outline**

1. Inventory every persisted use of `Stage.QWEN_*`, `StreamPurpose`, `StreamStage`, window
   keys, window closure, local result v1-v4, and primary completion v3.
2. Record a version decision for every P1 contract and identity change.
3. Bind the old-DAG baseline to source digest/duration/cameras, commit, model/checkpoint,
   policies, call counts, fresh/cached state, and report digest.
4. Add metric names for observation calls, refinement calls, deterministic projections,
   native codec preparation, and artifact replay reads.

**Keep intact:** released bytes, old identities, exact raw artifacts, and the distinction
between local evidence and production qualification.

**Done when**

- [ ] Every P1 identity/wire change has a version decision.
- [ ] The 164-call baseline is reproducible without summing overlapping spans.
- [ ] No throughput target is inferred from GPU utilization alone.

**Run locally**

```powershell
python -m pytest tests/unit/test_schema_immutability.py tests/contract/test_schema_release_policy.py
python -m pytest tests/unit/test_canonical_profile.py tests/unit/test_benchmark.py
```

**Next boundary:** P1 and P10.

### `contract-governance` - P1: Add the Perception Contract Family

**Result:** New producers can publish context, observation, projection, track, refinement,
and segment terminal artifacts without changing any released window schema.

**Primary paths and entry points**

- `src/robata/contracts/perception_stream.py` - strict models, semantic projections,
  identities, stage/refinement enums, and references.
- `schemas/v1/perception-context-manifest.schema.json`
- `schemas/v1/mage-observation.schema.json`
- `schemas/v1/observation-projection-bundle.schema.json`
- `schemas/v1/event-track.schema.json`
- `schemas/v1/perception-refine-request.schema.json`
- `schemas/v1/perception-refine-result.schema.json`
- `schemas/v1/perception-terminal-closure.schema.json`
- `schemas/schema-catalog.json`, `scripts/register_schema.py`, and when required
  `src/robata/contracts/schema_upcasting.py`.
- New `tests/unit/test_perception_stream_contracts.py`.

**Implementation outline**

1. Define six-camera membership with explicit absence/health; never silently omit a slot.
2. Make `MageObservation` model evidence, not a final event. Require bounded absolute
   intervals, local references, camera support/contradiction/visibility, confidence and
   boundary features, semantic QA, and optional shadow `gate_score`.
3. Keep model confidence as an uncalibrated feature.
4. Define projection/track/refine identities from semantic inputs only and exact artifact
   references separately.
5. Register all new schema bytes/catalog pins. If local-result or completion roots change,
   register the named successor and supported predecessor/upcaster; never edit old bytes.

**Keep intact:** RFC8785 canonicalization, lowercase SHA-256, schema-ref verification,
exact-byte artifact identity, and all old stream/window identities.

**Done when**

- [ ] New schemas register and catalog digests reproduce.
- [ ] Camera/observation ordering is canonical or rejected.
- [ ] Operational fields cannot change semantic identity.
- [ ] Old fixtures remain byte-identical.

**Run locally**

```powershell
python -m pytest tests/unit/test_perception_stream_contracts.py tests/unit/test_register_schema.py tests/unit/test_schema_immutability.py
python -m pytest tests/contract/test_schema_catalog.py tests/contract/test_schema_release_policy.py tests/contract/test_schema_upcasting.py
```

**Next boundary:** P2, P3, and P4.

### P2 - Emit Non-Overlapping Perception Segments and Deterministic Media Health

**Participating modules:** `source-media`, `sampling-qa`

**End-to-end result:** A single MCAP pass emits immutable non-overlap segment references and
cheap health evidence; vNext does not create overlapping inference windows or PNG sampling
packages for normal perception.

| Module | Contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `source-media` | Add an additive perception media policy/planner using existing segment primitives, codec packet spans, timestamps, and exact hashes | `application/canonical/bounded_media.py`, `single_pass_video.py`, `mcap_source.py`, `contracts/stream_source.py` | bounded media, single-pass MCAP tests |
| `sampling-qa` | Emit black/frozen/gap/cadence/exposure/decode health independently of semantic QA | `application/canonical/media_quality.py`, `media_quality_source_binding.py`, `qa_pipeline/**` | media-quality tests |

**Implementation outline**

1. Reuse `StreamSegmentManifest@1.0.0` if sufficient; put causal context in the new context
   manifest rather than changing the segment schema.
2. Add a versioned initial 8-second non-overlap perception policy with an exact shorter tail.
   Keep storage and context policies explicit so a later 2-second scan/8-12-second reasoning
   policy remains additive.
3. Publish native codec packet/file references from aligned source data. Do not decode all
   frames, build a 2x3 mosaic, or re-encode camera streams by default.
4. Produce deterministic health even if Mage is silent or fails; semantic QA is fused later.
5. Preserve bounded rings/cache, lateness facts, and durable source retention under
   backpressure before inference admission.

**Compatibility notes:** old `BoundedMediaPolicy` and incremental-window production remain
available only for the explicit legacy route.

**Combined proof**

```powershell
python -m pytest tests/unit/test_bounded_media.py tests/unit/test_mcap_single_pass.py tests/unit/test_canonical_media_quality.py
python -m pytest tests/integration/test_real_mcap_single_pass.py tests/integration/test_canonical_mcap_source.py
```

**Next boundary:** P3 consumes codec artifacts; P4 consumes health.

### P3 - Mage Native Codec/Video Runtime and Adapter

**Participating modules:** `inference-evidence`, `source-media`

**End-to-end result:** The default local real-model process loads Mage only and consumes
versioned native video/codec requests. `local-hf-vision-request-v1` remains a compatibility
endpoint, not the vNext transport.

**Primary paths and entry points**

- New `src/robata/inference/mage_video_runtime.py` - Mage loader, native codec processing,
  bounded context replay, and optional feature cache.
- New `src/robata/inference/mage_video_endpoint.py` - request/response, health, exact-body
  idempotency, and raw response CAS.
- New `src/robata/inference/mage_video_adapter.py` - capabilities and local/remote transport.
- `src/robata/inference/local_hf_endpoint.py` and `local_hf_adapter.py` - retain v1 image
  compatibility and share only safe utilities.
- `src/robata/application/canonical/local_real_model.py` - Mage vNext binding; Qwen behind an
  explicitly named legacy factory.
- `pyproject.toml`, `Dockerfile`, and production image files - a separately installable Mage
  codec extra and pinned ffmpeg/ffprobe/codec-video-prep dependencies.
- New `tests/unit/test_mage_video_runtime.py`, `test_mage_video_endpoint.py`, and
  `test_mage_video_adapter.py`.

**Implementation outline**

1. Load Mage with `trust_remote_code=True`, but create checkpoint manifest v2 hashing model
   shards, config, tokenizer/template assets, remote-code Python, codec implementation,
   cognition-gate code, and gate weights.
2. Use the native video API equivalent to
   `processor(videos=[video], video_backend="codec", codec_config=...)`; when vNext requires
   codec mode, reject any hidden PNG/image fallback.
3. Version the endpoint request around `PerceptionContextManifest` and bounded camera video
   parts. Hashes/policies define semantics; local paths, mounts, and signed locators are
   transport data and must be root/host allowlisted.
4. Preserve exact request-body idempotency and raw response bytes; key/body conflicts fail.
5. Expose capabilities `SINGLE_VIDEO_NATIVE_CODEC`, `PER_CAMERA_CODEC_ONE_DECODER`, and
   future `DYNAMIC_TOP_K`. The six-camera profile completes only when six per-camera codec
   representations feed one decoder generation; it must not silently become six business
   calls or a mosaic.
6. If the public hook initially supports one video, keep a valid one-video codec smoke while
   implementing multi-view feature fan-in behind the second capability. Production
   composition fails closed when requested multi-view is unproven.
7. Starting vNext must not load, probe, or route to Qwen. Checkpoint deletion is separate.

**Keep intact:** inference intent/attempt/accepted-call lineage, exact endpoint idempotency,
provider-neutral adapters, bounded payload/runtime limits, and legacy v1 decoding.

**Done when**

- [ ] Fake runtime proves strict validation, idempotent replay, timeout, and raw persistence.
- [ ] Opt-in real Mage smoke consumes video/codec without PNG materialization and records
      backend/model/checkpoint identity.
- [ ] Starting vNext leaves Qwen unloaded.
- [ ] Unsupported multi-view mode fails closed.

**Run locally**

```powershell
python -m pytest tests/unit/test_mage_video_runtime.py tests/unit/test_mage_video_endpoint.py tests/unit/test_mage_video_adapter.py
python -m pytest tests/unit/test_sqlite_inference_evidence.py tests/unit/test_inference_input_plan.py
# Explicit opt-in; skip cleanly when local Mage/codec dependencies are absent.
python scripts/run_local_mage_codec_smoke.py --model D:\HuggingFace\Mage-VL --require-native-codec
```

**Next boundary:** P4 turns the raw response into one observation and deterministic views.

### P4 - Persist One Mage Observation and Project QA, Events, and Evidence

**Participating modules:** `inference-evidence`, `sampling-qa`, `event-semantics`

**End-to-end result:** One accepted Mage call yields one immutable observation artifact and
three independently versioned deterministic business projections; no projector invokes a
model.

| Module | Contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `inference-evidence` | Strict parsing and raw/parsed artifact lineage/replay | `inference/models.py`, `evidence.py`, `orchestrator.py`, new `mage_observation.py` | evidence ledger/replay tests |
| `sampling-qa` | Fuse deterministic media health with observation semantic-QA | `qa_pipeline/**`, `application/canonical/product_qa.py` | QA/media-health tests |
| `event-semantics` | Project action hypotheses and per-camera support/contradiction | `event_pipeline/candidate.py`, `evidence.py`, new `observation_projection.py` | event core/guards |

**Implementation outline**

1. Persist raw provider bytes before parsing. Parse failure produces typed incomplete or
   quarantined evidence; it never erases raw bytes or fabricates facts.
2. Parse one observation containing context identity, semantic QA, local action references,
   arbitrary bounded intervals, actor/object features, per-camera support/visibility,
   boundary features, and optional gate score.
3. Implement pure, versioned `QAProjector`, `EventProjector`, and `EvidenceProjector`.
   Re-running them with identical observation/health inputs reproduces bytes and IDs.
4. Keep logical QA/event/evidence contracts distinct while deleting their separate physical
   VLM invocations.
5. Treat the cognition gate as shadow evidence; a low score cannot suppress perception yet.

**Compatibility notes:** legacy task artifacts remain readable, but vNext never pretends
independent legacy calls were one original `MageObservation`.

**Combined proof**

```powershell
python -m pytest tests/unit/test_sqlite_inference_evidence.py tests/unit/test_inference_evidence_accepted_lineage.py
python -m pytest tests/unit/test_qa_pipeline_core.py tests/unit/test_media_quality_supplemental.py
python -m pytest tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py tests/unit/test_mage_observation_projection.py
```

**Next boundary:** P5 causal tracking; P6 evidence judgment.

### P5 - Cross-Segment Event Tracks and Exceptional Mage Refinement

**Participating modules:** `event-semantics`, `inference-evidence`

**End-to-end result:** Segment observations update durable event tracks, and only a specific
unresolved ambiguity may schedule bounded, single-purpose Mage refinement.

**Primary paths and entry points**

- New `src/robata/event_pipeline/track.py` - deterministic state machine/reducer.
- New `src/robata/event_pipeline/temporal_reconcile.py` - association, continuation,
  closure, and ordering policy.
- `src/robata/event_pipeline/boundary_refinement.py` - consume targeted refine evidence.
- New `src/robata/inference/mage_refine.py` - reason-coded prompt/context builder.
- New `tests/unit/test_event_track.py` and `test_mage_refine.py`.

**Implementation outline**

1. Define `CANDIDATE -> OPEN -> UPDATED -> CLOSED -> FINALIZED` transitions ordered by
   segment sequence/absolute time and safe under redelivery.
2. Associate using versioned label/object/actor/temporal/camera rules. A track can cross
   segments; final identity derives from ordered accepted observation/refine evidence.
3. Emit explicit reasons: `UNCERTAIN_ONSET`, `UNCERTAIN_OFFSET`, `LABEL_CONFLICT`,
   `CAMERA_CONFLICT`, or `INSUFFICIENT_SUPPORT`.
4. Ask narrow questions over bounded media, for example only the precise end boundary in a
   2-second neighborhood; never repeat full perception.
5. Record refine rate/reason/context/generation time and whether it changed/closed the track.

**Keep intact:** upstream provenance, deterministic ordering, visible ambiguity, boundary
fallback evidence, and idempotent redelivery.

**Done when**

- [ ] Start/continue/end observations across three segments create one final interval.
- [ ] Duplicate delivery leaves the track byte-identical.
- [ ] Resolved tracks schedule zero refinements.
- [ ] Every refine identity is stable and context-bounded.

**Run locally**

```powershell
python -m pytest tests/unit/test_event_track.py tests/unit/test_mage_refine.py
python -m pytest tests/unit/test_event_pipeline_core.py tests/unit/test_event_projection_guards.py
python -m pytest tests/integration/test_canonical_action_event_revision.py
```

**Next boundary:** P6 and P7.

### `event-semantics` - P6: Observable-Camera Fusion v2

**Result:** Fusion remains the evidence judge, but confidence/support derive from an
explicit observable/selected population rather than an unconditional six-camera divisor.

**Primary paths and entry points**

- `src/robata/event_pipeline/fusion.py` - keep v1 behavior for legacy input; add a versioned
  v2 engine/policy or denominator strategy.
- `src/robata/event_pipeline/evidence.py` - observable/selected/relevant camera facts.
- `src/robata/application/canonical/media_quality.py` - unavailable/unusable cameras.
- New `tests/unit/test_fusion_observable_cameras.py`.

**Implementation outline**

1. Declare physically observable, policy-selected, and action-relevant camera sets; preserve
   absence/QA reasons for every nonparticipant.
2. Normalize reliability using the versioned eligible denominator/weights, never fixed `/6`
   when a smaller declared set can contribute.
3. Retain minimum camera support, contradiction, visibility, coverage, QA weighting, and
   explicit insufficient-support/low-confidence ambiguity.
4. Keep v1 fusion identity/output unchanged. v2 policy and denominator inputs participate in
   v2 identity.

**Keep intact:** Mage produces evidence; Robata decides whether it is enough for a physical
fact. Provider confidence alone never becomes production confidence.

**Done when**

- [ ] Six-observable-camera fixtures retain expected v1-equivalent behavior.
- [ ] Two selected good cameras are not penalized as four implicit zeros.
- [ ] Missing/unusable cameras remain explicit and cannot inflate confidence.
- [ ] Dynamic top-K remains disabled until separately qualified.

**Run locally**

```powershell
python -m pytest tests/unit/test_fusion_observable_cameras.py tests/unit/test_event_pipeline_core.py
python -m pytest tests/unit/test_canonical_media_quality.py tests/unit/test_event_projection_guards.py
```

**Next boundary:** P7.

### P7 - Replace the Default Window DAG with a Durable Perception DAG

**Participating modules:** `stream-control`, `canonical-integration`

**End-to-end result:** The default canonical route schedules segment/context work through a
provider-neutral perception DAG, retaining bounded backpressure, leases, fences, recovery,
and one normal model stage per perception context.

| Module | Contribution | Key paths | Local check |
| --- | --- | --- | --- |
| `stream-control` | Add `DurablePerceptionScheduler` or additive vNext policy/table family; segment closure replaces expected-window closure | `queue/**`, `application/canonical/stream_scheduler.py`, `durable_work.py` | scheduler/barrier/lease/recovery tests |
| `canonical-integration` | Compose source -> observation -> projectors -> tracks -> fusion/refine -> finalization | `application/canonical/pre_eos_execution.py`, `local_composition.py`, `runner.py`, `reduction.py`, `local_real_model.py` | local command/offline tests |

**Implementation outline**

1. Add the topology
   `MEDIA_SCAN -> PERCEPTION_OBSERVE -> OBSERVATION_PROJECT -> TEMPORAL_RECONCILE -> FUSION`,
   with `PERCEPTION_REFINE` only from typed ambiguity and `FINALIZE` after segment closure/EOS.
2. Replace expected-window declaration/seal/closure with additive segment/context membership
   and `perception-terminal-closure@1.0.0`; retain all legacy window tables/readers.
3. Preserve lease fencing, retries, late input, failed/incomplete outcomes, recording
   fairness, and bounded upstream retention under backpressure.
4. Backpressure the expensive observe queue, not unbounded decoded frames. Source/codec
   artifacts must already be durable before admission.
5. Make Mage vNext the default real-model composition. Qwen factories require an explicit
   legacy profile and never load as a startup side effect.
6. Count physical calls at the scheduler boundary. Deterministic projectors cannot depend on
   a provider adapter.

**Compatibility notes:** do not rename old queue stages or reuse old work logical keys.
Separate policy/table namespaces make rollback a route choice rather than a data rewrite.

**Combined proof**

```powershell
python -m pytest tests/unit/test_sqlite_work_scheduler.py tests/unit/test_bounded_stream_work_queues.py tests/unit/test_stream_scheduler_composition.py
python -m pytest tests/unit/test_stream_recording_reduction.py tests/unit/test_local_stream_finalization.py
python -m pytest tests/integration/test_canonical_local_command.py tests/integration/test_canonical_offline.py
```

**Next boundary:** P8.

### P8 - Observation-Centric Evidence, Identity, Replay, and Recovery

**Participating modules:** `inference-evidence`, `identity-delivery`, `canonical-integration`

**End-to-end result:** Every published fact traces to one accepted perception artifact and
all deterministic projections; cache loss, process death, retry, completion, and outbox
reconciliation preserve identity.

**Primary paths and entry points**

- `src/robata/adapters/sqlite_inference_evidence.py`
- `src/robata/adapters/sqlite_event_identity_registry.py`
- `src/robata/adapters/sqlite_primary_completion.py`
- `src/robata/adapters/sqlite_outbox.py`
- `src/robata/application/canonical/primary_completion.py`
- `src/robata/application/canonical/local_outbox_delivery.py`
- `src/robata/application/canonical/logical_nodes.py`
- New observation/track durable adapters under `src/robata/adapters/`.

**Implementation outline**

1. Persist the chain: intent -> exact request -> raw response -> parsed observation ->
   projector results -> track revisions -> fusion -> optional refine -> final fact ->
   completion/outbox.
2. Define artifact replay as reading accepted raw bytes. Define recomputation as a new
   recorded attempt with the same semantic input; compare it but never promise byte equality.
3. On worker loss, discard cache state, reload the context manifest, replay a bounded prior
   segment horizon, and resume. Existing accepted raw evidence prevents a new logical call.
4. Add crash points before raw commit, after raw/before parse, after observation/before
   projection, during track update, after completion/before outbox, and during relay.
5. If final/completion roots change, implement the P1 successor/upcaster/storage migration;
   otherwise prove existing roots consume vNext facts without projection change.

**Keep intact:** pending-terminal recovery, lease fencing, completion immutability,
exact-byte CAS, idempotent outbox delivery, and explicit incomplete/quarantine evidence.

**Done when**

- [ ] Accepted raw bytes replay without Mage.
- [ ] Forced cache loss plus bounded context reconstruction resumes deterministically.
- [ ] Every event has observation/projector/track/fusion/refine lineage.
- [ ] Completion/outbox crash reconciliation neither duplicates nor loses delivery.

**Run locally**

```powershell
python -m pytest tests/unit/test_sqlite_inference_evidence.py tests/unit/test_sqlite_inference_raw_cas.py tests/unit/test_event_identity_registry.py
python -m pytest tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py
python -m pytest tests/integration/test_canonical_recovery_qualification_evidence.py tests/integration/test_canonical_local_review_routing.py
```

**Next boundary:** P9.

### P9 - Additive Migration, Rollback, and Qwen-Route Retirement

**Participating modules:** `contract-governance`, `canonical-integration`, `identity-delivery`

**End-to-end result:** Configuration can choose legacy-window read/replay or Mage vNext
execution without mixing identities. Default startup uses Mage vNext and no Qwen process.

**Implementation outline**

1. Add explicit profiles such as `legacy_window_v1` and `mage_stream_vnext_v1`. New runs
   default to vNext only after acceptance; each existing run remains bound to its creation
   profile.
2. Use additive tables/namespaces/schema refs. Never rewrite window keys into segment keys or
   downgrade vNext artifacts during rollback.
3. Provide dual readers where review/product surfaces need history. Independent legacy calls
   remain legacy evidence and cannot be backfilled as an authoritative observation.
4. Rollback stops new vNext admission, drains or terminalizes admitted work according to its
   recorded policy, then starts explicitly legacy runs. It does not delete observations or
   mutate completion roots.
5. Remove Qwen from default health checks, Docker startup, production routing, and local
   autoload. Keep explicit legacy adapter/tests for one compatibility retention period.
6. External checkpoint, old schema, or row deletion requires a later approved retention and
   destruction phase with inventory and restore proof.

**Keep intact:** historical readability, per-run profile binding, schema authority, exact
evidence, and rollback without cross-family identity collisions.

**Done when**

- [ ] Default vNext startup proves no Qwen process/model residency.
- [ ] Legacy artifacts and committed runs remain readable.
- [ ] Mid-run route changes are rejected; new-run profile changes are explicit.
- [ ] Rollback retains both families and delivers each completion once.

**Run locally**

```powershell
python -m pytest tests/unit/test_inference_routing.py tests/unit/test_stream_scheduler_composition.py
python -m pytest tests/contract/test_schema_upcasting.py tests/contract/test_schema_release_policy.py
python -m pytest tests/integration/test_canonical_local_command.py tests/integration/test_sqlite_primary_completion.py tests/integration/test_sqlite_outbox_relay.py
```

**Next boundary:** P10.

### `qualification-ops` - P10: End-to-End Performance, Quality, and Recovery Qualification

**Result:** The vNext claim is supported by a like-for-like report explaining call count,
codec work, GPU work, projections/refinement, quality, and recovery, while naming every
external boundary still unmeasured.

**Primary paths and entry points**

- `src/robata/runtime/canonical_profile.py`, `runtime/capacity.py`
- `src/robata/benchmark/metrics.py`
- `scripts/run_local_streaming_smoke.py`
- New `scripts/run_local_mage_stream_vnext.py`
- New `tests/unit/test_mage_streaming_benchmark.py`

**Implementation outline**

1. Run the same full 40.8335-second six-camera source with fresh state and the 8-second vNext
   policy. Record source digest, commit, checkpoint, codec/context/prompt/projector/fusion/
   refine policies, hardware/software, call count, and report digest.
2. Report non-overlapping attribution: ingest/alignment, codec preparation, perception wait,
   Mage encode/generation, raw persistence, parsing, projectors, reconcile, fusion, refine,
   finalization, and outbox.
3. Enforce the architecture gate: at most six normal observation generations, zero old
   business-stage VLM calls, and every extra call classified as reason-coded refinement.
4. Report GPU/VRAM, CPU, RAM, disk I/O, codec token/frame counts, recording-seconds and
   camera-seconds per wall-second, p50/p95/p99 segment latency, backlog slope, and restart
   replay overhead.
5. Report QA/event/boundary/track quality where labels exist, plus invalid, incomplete,
   abstention, conflict, refine, and publication rates. Never infer quality from call count.
6. Log gate score distributions and counterfactual skips only; admission remains disabled
   until governed recall and false-silence limits pass.
7. Fault-inject endpoint death, cache loss, raw replay, scheduler restart, storage errors, and
   outbox relay. Run the full suite in isolated GitHub Actions shards to avoid local SQLite
   lock coupling.
8. Compare old/new as a Pareto surface across quality, latency, throughput, resource/cost,
   recovery, and ambiguity.

**Keep intact:** recording-hours and camera-hours remain separate; cached/fresh source runs
are named; local evidence remains `NOT_PRODUCTION_QUALIFIED` until external gates run.

**Done when**

- [ ] The full fixture publishes end to end with no Qwen loaded.
- [ ] The six-call normal-perception gate passes, or failure is retained exactly.
- [ ] Artifact replay performs zero provider calls; cache-loss recovery is measured.
- [ ] Quality and ambiguity/refinement effects are reported, not assumed.
- [ ] CI shards pass and no unrelated or `web/` changes exist.

**Run locally**

```powershell
python -m pytest tests/unit/test_mage_streaming_benchmark.py tests/unit/test_runtime_capacity.py tests/unit/test_canonical_profile.py
python -m pytest tests/unit/test_local_streaming_smoke.py tests/unit/test_local_streaming_benchmark.py
python -m pytest tests/integration/test_canonical_local_command.py tests/integration/test_canonical_recovery_qualification_evidence.py
python scripts/run_local_mage_stream_vnext.py --source <mcap> --mapping <mapping> --segment-seconds 8 --fresh-state --report-dir <report>
```

**Next boundary:** H100/RunPod, production storage, representative quality, and soak gates,
then an explicit production release decision.

## Migration and Rollback Matrix

| Situation | Required behavior | Forbidden shortcut |
| --- | --- | --- |
| Legacy run replay | Read original window schemas/stages/identities/raw artifacts | Rehash as a vNext observation |
| New vNext run | Bind Mage profile, segment/context policy, checkpoint, schemas at creation | Switch provider/profile silently mid-run |
| Failure before raw commit | Retry under same intent and fenced attempt policy | Publish projected facts without raw evidence |
| Raw committed, parse fails | Preserve bytes; record incomplete/quarantine and policy-bounded retry | Discard or invisibly repair evidence |
| Cache/hidden state lost | Rebuild from manifest and bounded prior segments | Treat GPU memory as durable truth |
| Default route rollback | Stop new vNext admission, drain/terminalize, start explicit legacy runs | Delete vNext rows or downgrade schemas |
| Completion root unchanged | Reuse primary completion v3 exactly | Add a gratuitous incompatible version |
| Completion root changed | Register v4 plus upcaster/storage/rollback reader | Edit v3 in place |
| Retention ends | Inventory, backup/restore proof, separate deletion approval | Delete Qwen/checkpoints/schemas/rows during ordinary deploy |

## Blockers and External Dependencies

Ordinary implementation is not a blocker. Interfaces, strict failure behavior, fake
runtimes, migrations, and recovery tests remain local work.

| Condition | What can still be completed locally | Temporary substitute | Later external proof |
| --- | --- | --- | --- |
| Mage codec Python/system dependencies | Versioned runtime/endpoint, checks, fail-closed codec requirement | Fake codec runtime and optional local smoke | Reproducible target container load/generation |
| Public Mage path is single-video oriented | Capability model, feature fan-in abstraction, one-decoder fake/runtime tests | Single-video native codec smoke; no production claim | Six-camera per-codec encode plus one decoder |
| Two-H100/RunPod topology | Routing, idempotency, timeout, context manifests, metrics | One local GPU/fake provider | Saturation, endpoint failure, capacity report |
| R2/object storage | Artifact locator port, hashes, local CAS, staging/recovery | Local filesystem CAS | R2 upload/range/reconciliation/fault test |
| PostgreSQL/Supabase | Contracts, SQLite recovery semantics, migration SQL/tests | SQLite durable adapters | Concurrency, backup/restore, RLS qualification |
| Governed six-camera labels | Metrics, deterministic fixtures, gate-shadow logging | Existing sample/conformance fixtures | Blinded event/track/boundary/support evaluation |
| 500 recording-hours/day target | Capacity harness, backpressure model, short fault profile | Local short run/CI shards | Representative production soak/cost run |

## Acceptance and Verification

- [ ] Focused unit tests cover every new semantic decision and state transition.
- [ ] New wire shapes are additive and released bytes remain immutable.
- [ ] Provider/model names do not appear in vNext scheduler stages.
- [ ] Default vNext startup loads Mage only and uses native codec/video.
- [ ] One observation deterministically produces QA, event, and evidence projections.
- [ ] Segment boundaries never force event boundaries.
- [ ] Normal perception and exceptional refine calls are counted separately.
- [ ] Gate scores remain shadow-only until governed recall/false-silence gates pass.
- [ ] Fusion v2 declares its denominator and explicit absent/unusable facts.
- [ ] Artifact replay and model recomputation are named/tested separately.
- [ ] Cache loss, scheduler restart, completion, and outbox crashes are covered.
- [ ] Legacy runs remain readable and rollback never rewrites identity.
- [ ] Performance claims name workload, model, policies, hardware, state, and report digest.
- [ ] External gates remain `NOT_MEASURED` until executed.
- [ ] No `web/` file changes in this cycle.

## Suggested Dispatch Prompt

```text
Work on <module-id> / P<n> - <phase name>.

Read AGENTS.md, governance/BLUEPRINT.md, and every governance/modules/<module-id>.md named
by the phase. Implement only that phase's end-to-end result.
Goal: <copy the phase result>.
Primary paths: <copy the phase paths/change map>.
Preserve: released schema bytes, old window identities, exact raw evidence, durable
recovery, and provider-neutral vNext stage names.
Run locally: <copy the phase commands>.
Report: changed files, command results, schema/identity decision, normal/refine call counts
when applicable, measured result, and real external blockers. Do not touch web/ and do not
physically delete model checkpoints.
```
