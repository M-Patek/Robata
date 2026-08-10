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

---

# 2026-08-09 DCVC Provider V2 Cold-Path Qualification Addendum

## Product Outcome

**Outcome:** Robata prepares Mage native-DCVC assets through one explicit, identity-bound
Provider V2 configuration, proves that `max_side` reaches the DCVC process, reuses one
resident DCVC engine across a sequence of single-camera segments, and admits only exact,
receipt-backed cache assets to the existing Mage endpoint.

**Why now:** the observed upstream path accepts `max_side` in Robata's policy but the child
readiness process reads the checkpoint's static `preprocessor_config.json`. A requested
`max_side=448` can therefore be recorded in parent-authored metadata while the child actually
runs full resolution. The current child-per-segment topology also reloads the DCVC networks.

**Success measures:**

- Provider V2 full-resolution (`max_side=0`) is exact-byte equivalent to the observed-v1
  reference for provider payload assets before any bounded-resolution default is considered.
- Every V2 cache entry binds provider implementation, canonical effective configuration,
  child/worker receipt, and exact output assets in a new recipe/cache namespace.
- One persistent worker handles the five non-overlapping segments serially and proves one
  DCVC engine load, one sequence reset per segment, no cross-camera or cross-segment DPB reuse,
  atomic publication, restart cleanup, and fail-closed admission.
- The same 40-second/five-segment sample reports cold wall time, RTF, provider startup/load,
  per-segment preparation, GPU/VRAM/temperature, asset identities, Mage output, and downstream
  QA/event/evidence/track/fusion deltas for observed-v1, V2 full-resolution, and bounded
  `max_side` candidates.
- A bounded-resolution default changes only after identity, recovery, exact-asset validation,
  and quality gates pass. H100 topology remains target configuration until externally run.

**Non-goals:** multi-camera fan-in, high request concurrency, cross-camera latent reuse,
cross-segment DPB reuse, or treating `sequence_length_frames` as a DCVC compute limit.

## Contract and Identity Decision

These are internal operational contracts, not published wire schemas. No file in `schemas/`
is changed in this phase.

| Contract | Decision | Compatibility rule |
| --- | --- | --- |
| `mage-dcvc-provider-job-v2` | New canonical internal job | Binds immutable source bytes, checkpoint, provider implementation, explicit effective config, sampling semantics, and output intent. Unknown or missing fields fail. |
| `mage-dcvc-provider-receipt-v2` | New canonical internal receipt | Worker-authored proof of the configuration actually used, engine-load generation, processed frame extent, timing, and exact payload asset set. Parent-authored claims are not accepted as proof. |
| `mage-dcvc-readiness-explicit-v2` | New effective recipe | Replaces `mage-dcvc-readiness-observed-v1` only for V2 builds; both remain readable and never share a namespace. |
| cache entry/manifest/namespace v2 | New internal families | Add provider/config/receipt identities. Existing v1 artifacts remain read-only rollback evidence and are never rewritten or admitted as V2. |
| qualified Mage checkpoint | New model revision and checkpoint manifest | Provider implementation used for preparation must be copied into a separate qualified model tree and included in its checkpoint manifest. The original `D:\HuggingFace\Mage-VL` stays unchanged. This makes the existing endpoint v2 model+policy inference identity sufficient for this cycle. |

Provider V2 initially requires `sequence_length_frames=0` and
`canvas_token_side=null`. Uniform sampling may select fewer output frames, but DCVC still
advances its recurrent chain from frame zero through the largest sampled frame. Reports must
not describe a sampled-frame count as a temporal compute cap.

The persistent worker is fixed to one implementation/config/device tuple. `reset_interval`
and `intra_period` are process identity fields because the DCVC engine binds them at load.
Each job calls `reset_sequence`; QP may be applied at reset. A different fixed tuple requires
a separate worker and recipe identity.

On the local RTX 4060 profile, CUDA codec preparation and Mage generation are mutually
exclusive GPU phases. The worker may persist between phases, but the scheduler must not run
both simultaneously. The two-H100 target may place one serial codec worker and one serial
Mage decoder on different GPUs, but this remains `NOT_LOCALLY_VALIDATED`.

## Roadmap

| Phase | Outcome | Modules | Local proof | External follow-up |
| --- | --- | --- | --- | --- |
| P11 | Canonical Provider V2 job/receipt, qualified checkpoint, v2 cache identity | `contract-governance`, `inference-evidence` | identity/parity/tamper tests; no schema diff | none |
| P12 | Explicit fixed-config provider and persistent one-job worker | `inference-evidence`, `stream-control` | protocol, one-load/multi-reset, crash cleanup, stale-result rejection | none |
| P13 | Strict cache pre-admission and local serialization guard | `canonical-integration`, `identity-delivery` | endpoint rejects v1/wrong recipe/receipt/config; GPU phase overlap rejected | none |
| P14 | Same-sample A/B and quality qualification | `qualification-ops`, `event-semantics` | observed-v1 vs V2 full-res vs bounded max-side report | repeat on H100 |
| P15 | Target deployment profile and adoption decision | `qualification-ops` | rendered config and report explicitly mark unmeasured claims | RunPod/H100 soak |

## P11-P15 Implementation Boundaries

1. Build the qualified model tree reproducibly with a pinned source-manifest precondition;
   copy/link unchanged bytes, copy the exact Provider V2 implementation into the qualified
   tree, assign a new model revision, and generate a new checkpoint manifest. Never patch the
   source checkpoint in place.
2. Pass the effective configuration to the provider before readiness/DCVC modules import.
   The provider may use a versioned configuration-module overlay, but may not mutate an
   already-imported unknown Mage object or depend on hidden checkpoint defaults.
3. Generate into a same-filesystem temporary directory, validate receipt and payload hashes,
   then atomically publish. On restart, delete only verified stale temp children under the
   qualified cache root; never recursively delete a computed external path.
4. Establish observed-v1 -> V2 `max_side=0` payload parity first. Then vary only `max_side`.
   Do not mix `sequence_length_frames`, `canvas_token_side`, cross-segment state, or camera
   selection into the same experiment.
5. Keep one camera, one provider worker, and one Mage generation in flight. Higher-performance
   hardware changes placement, not the logical contract.

## Measured Qualification Result ? August 9, 2026

The controlled local A/B used the same 40-second recording, five non-overlapping segments, one
camera, one preparation worker, one NF4 Mage generation lane, and zero executed refinement calls.
The retained evidence shows:

- observed-v1 cold preparation: 178.146 seconds;
- Provider V2 `max_side=0`: 144.383 seconds;
- Provider V2 `max_side=448`: 50.512 seconds, a 3.527x speedup and 71.65% wall reduction;
- all nine non-metadata provider assets for all five segments are exact-equal across the three
  variants;
- all five Mage outputs and all downstream QA/event/evidence/track/fusion comparison projections
  are equal;
- exact Provider V2 cache consumption performs zero new `_run_dcvc_rt` calls.

Decision: Provider V2 `max_side=448` is the local/pre-production candidate default. The Provider
V2 prewarm CLI therefore defaults to 448; `--max-side 0` remains the explicit full-resolution
control. observed-v1 remains an explicit non-destructive rollback composition. The decision is
not production eligibility evidence and does not qualify Linux/H100, multiple cameras, or target
capacity.

## Acceptance and Rollback

- [x] Original Mage directory and observed-v1 cache remain unchanged and readable.
- [x] V2 job/config/implementation changes produce different recipe, namespace, and qualified
      checkpoint identities.
- [x] Missing/mismatched receipt, config, implementation, source, or asset hashes fail closed.
- [x] Full-resolution V2 payload parity passes before persistent execution is called equivalent.
- [x] A one-shot Provider V2 execution topology is not admitted. If one is introduced later, it
      requires exact parity or a successor recipe rather than silent substitution.
- [x] Bounded `max_side` adoption passed explicit asset, Mage-output, and downstream semantic
      quality gates; speed alone did not change the default.
- [x] RTX 4060 evidence never claims simultaneous codec+decoder GPU execution.
- [x] H100 topology is documented as target-only until measured on the target hardware.
- [x] No `web/` or published-schema changes are included.

Rollback stops new V2 cache admission and restarts the endpoint with the original model
revision, checkpoint manifest, codec policy, and observed-v1 manifest. It does not delete or
rehash either cache family. Strict Provider V2 admission never silently falls back to dynamic
DCVC recomputation.

---

# Mage 25x Throughput Convergence and Conditional Qwen Batch Pivot - August 9, 2026

## Product Outcome

**Outcome:** Robata has one evidence-driven production-path decision for ingesting at least
500 camera-hours per day with 20% capacity headroom. The primary candidate remains the
single-camera Mage native-codec stream, but it must prove a segment-ready bounded pipeline,
measured steady-state costs, quality parity, and an economically plausible path to an aggregate
**25x real-time**. If those gates fail after avoidable overhead is removed, the repository keeps
the Mage evidence intact and activates an isolated Qwen Batch candidate under the same input,
output, quality, and hardware comparison rules.

**Why now:** Provider V2 removed a major false-configuration and per-segment model-load cost, but
the retained qualification still measures preparation and generation as two sequential runs.
For the controlled 40-second recording, bounded Provider V2 preparation is 50.3102506 seconds and
the Mage stream run is 21.9618685 seconds. Their serial recurring total is 72.2721191 seconds, or
only 0.5535x real-time. Neither a cache-only generation number nor an unmeasured H100 multiplier
is sufficient evidence for the 500-camera-hour requirement.

**Capacity definition:**

```text
required aggregate RTF
  = daily camera-hours * headroom / 24 hours
  = 500 * 1.20 / 24
  = 25.0x real-time

camera-hours/day per lane
  = 24 * sustained lane RTF

logical lanes required
  = ceil(25.0 / sustained lane RTF)
```

Recording-hours and camera-hours must remain separate. This cycle is locked to one camera so
40 seconds of recording equals 40 camera-seconds. A future six-camera recording contains six
times the camera-hours even when its wall-clock recording duration is unchanged.

**Capacity-unit conflict that must remain visible:** this goal cycle follows the operator's
explicit assumption of **500 camera-hours/day**, which yields 25x. The tracked production
requirement in `governance/REQUIREMENTS.md` currently says **500 recording-hours/day**. If one
recording contains six independently processed camera streams, that contract is equivalent to
3,000 camera-hours/day, or **150x camera real-time** with 20% headroom. If a future multi-view
request processes all six cameras jointly, neither 25x nor a simple 150x multiplication is a valid
capacity result until that request is measured. P22 cannot issue production `GO` while this unit
conflict is unresolved.

**Success measures:**

- The exact 40-second, `cam_01`, five-segment control remains reproducible on the RTX 4060 Laptop
  GPU with one preparation worker, one generation lane, and the same observation/projector/fusion
  semantics.
- Every report separates one-time process/model/kernel warm-up from recurring materialization,
  selected codec preparation, processor, generation, handoff, projection, durable scheduling, and
  finalization costs.
- The integrated Mage path no longer prewarms the complete recording before inference. Segment 0
  becomes observable as soon as its exact selected-provider entry is admitted; at most one
  subsequent segment may wait in a bounded queue.
- Same-device local execution never overlaps DCVC CUDA work with Mage generation. Separate-device
  overlap is represented by an explicit profile and remains unqualified until run on the target.
- Any continuous DCVC state experiment uses a successor internal recipe/cache/session identity;
  Provider V2 reset-per-job artifacts are never silently reinterpreted.
- Mage receives a numeric `GO`, `HOLD`, or `STOP` decision from measured steady-state throughput,
  quality parity, recovery evidence, and target capacity/cost assumptions. Unmeasured H100 values
  are labeled scenarios rather than observations.
- If Mage is `STOP`, Qwen Batch is measured with the same media interval, selected camera,
  deterministic business-output projection, quality comparison, local GPU, and one-worker
  constraint before any production recommendation.

**Non-goals for this cycle:**

- Six-camera fusion, multi-camera latent compression, a learned small encoder, or cross-camera
  DCVC reference reuse.
- High local request concurrency. The local design remains one preparation worker and one model
  generation lane; scaling is modeled as independent logical lanes after single-lane convergence.
- Claiming Linux, RunPod, H100, BF16, R2, PostgreSQL/Supabase, or 25x production qualification
  from Windows/RTX 4060 evidence.
- Weakening exact cache admission, artifact replay, identity isolation, durable work fencing,
  fail-closed behavior, or downstream quality gates to improve a benchmark.
- Editing `web/` or any published schema in this cycle.

## Frozen Local Baseline

Source of truth:
`docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json`, variant
`provider-v2-max-side-448`. Derived values below are calculations over retained measured fields;
they are not new GPU measurements.

| Boundary | Retained measurement | Derived lane RTF | Camera-hours/day/lane |
| --- | ---: | ---: | ---: |
| Provider V2 preparation, cache-manifest wall | 50.3102506 s / 40 s media | 0.7951x | 19.08 |
| Provider V2 preparation, sum of five worker jobs | 37.4077382 s / 40 s media | 1.0693x | 25.66 |
| Mage stream run | 21.9618685 s / 40 s media | 1.8213x | 43.71 |
| Mage generation sum | 19.9577439 s / 40 s media | 2.0042x | 48.10 |
| Current per-recording observed wall with resident decoder | 72.2721191 s / 40 s media | 0.5535x | 13.28 |
| Ideal two-stage overlap using retained whole-stage walls | max(50.3102506, 21.9618685) s | 0.7951x | 19.08 |

The five bounded preparation jobs measured 9.3185024, 7.1167458, 7.2323062,
6.5289990, and 7.2111848 seconds. The five Mage generations measured 7.6713050,
3.1115310, 2.8752010, 3.1762730, and 3.1234340 seconds. Segment 0 is a cold/long-output
outlier; segments 1-4 average 3.0716098 generation seconds. Model load and endpoint readiness
are one-time lifecycle costs and are excluded from a steady-state rate only when the same worker
is demonstrably kept resident across recordings.

The 72.2721191-second sum is not a pure recurring kernel total: the 50.3102506-second
preparation wall includes one 0.781124-second DCVC engine load plus an unresolved 12.9025124-second
difference between cache-manifest wall and worker-job sum. Existing evidence cannot honestly
classify that whole difference as either lifecycle or recurring work.

Warm segments 1-4 provide a narrower four-sample service estimate: DCVC averages 7.022309 seconds
per eight-second segment and observation averages 3.150942 seconds. The corresponding ideal
separate-stage pipeline is bottlenecked at about 1.1392x by DCVC; it would still require 22
local-equivalent lanes for the 25x camera-hour scenario. Four samples are insufficient for a
credible p95.

The retained serial result would require `ceil(25 / 0.5535) = 46` equivalent local lanes. Even
perfect overlap of the retained whole-stage walls would require `ceil(25 / 0.7951) = 32` lanes.
These figures are diagnosis, not the expected H100 count.

## Contract, Identity, and Measurement Decisions

No published JSON Schema changes are authorized by this plan. All additions below are internal
operational models and machine-readable benchmark documents.

| Area | Decision | Compatibility and rollback rule |
| --- | --- | --- |
| Provider V2 | Remains the control and rollback implementation | `mage-dcvc-readiness-explicit-v2` continues to mean one sequence reset per segment. Its receipts and cache entries are immutable. |
| Traditional H.264/HEVC candidate | Alternative codec engine, not a cold-start stage or a Provider V2 alias | A successor internal provider/recipe/cache/binding identity pins the exact `codec-video-prep` toolchain and arguments. Existing DCVC V2 entries remain immutable rollback evidence; the DCVC-only exact-cache binding is never loosened in place. |
| Segment-ready orchestration | Additive execution policy | It may consume and publish existing exact V2 entries incrementally, but may not change V2 artifact bytes or admission semantics. |
| Bounded queue | Depth is identity-bound operational policy | Local accepted depths are 1 (serial) and 2 (one next segment). Queue depth does not permit two concurrent Mage generations. |
| Same-device guard | Required on the RTX 4060 path | DCVC CUDA and generation share an exclusive guard. Queueing may overlap CPU/I/O only; reports must show zero forbidden GPU-phase overlap. |
| Separate-device profile | Target-only profile | One codec GPU and one decoder GPU may overlap. Until measured, capacity fields are `UNMEASURED_SCENARIO`. |
| Continuous DCVC session | Successor internal recipe is mandatory | A session that reuses DPB/recurrent state must use new provider/config/receipt/cache/session versions. Crashes rebuild ephemeral state from immutable media and an explicit replay anchor. V2 remains readable rollback evidence. |
| Model hot path | Operational profile is identity-bound | NF4/BF16, attention implementation, compile mode, fixed shapes, token budget, prompt contract, and observation schema remain part of the effective model/decoder identity. |
| Artifact replay | Remains exact | A repeated logical inference may reuse the accepted raw artifact byte-for-byte. GPU recomputation is measured separately and is not called exact replay. |
| Qwen Batch pivot | Separate candidate and route | It cannot overwrite Mage identities or reuse a Mage result as a Qwen result. Common comparison projections may be equal while durable identities differ. |

## Overall Roadmap

| Phase | Outcome | Main module(s) | Depends on | Local proof | External follow-up |
| --- | --- | --- | --- | --- | --- |
| P16 - Freeze baseline and 25x gate | One canonical performance ledger and capacity formula | `qualification-ops` | P15 | report parser tests, exact hash pins, derived capacity table | none |
| P17 - Incremental Provider V2 session | One resident worker admits one segment at a time and emits `CODEC_READY` without closing the process | `inference-evidence`, `stream-control` | P16 | protocol/session unit tests, crash/EOF/timeout tests, exact cache parity | none |
| P18T - Traditional H.264/HEVC candidate | Pin a Linux `cv-preinfer` provider, prepare exact segments without neural re-encoding, and select or reject it by the same 40-second A/B | `source-media`, `inference-evidence`, `qualification-ops` | P17 | pinned-container smoke, exact-cache/recovery tests, five-segment timing and quality evidence | target Linux CPU sizing |
| P18 - Segment-ready bounded Mage pipeline | The selected codec-ready segment enters the one-lane Mage consumer immediately; queue depth is 1 or 2 | `canonical-integration`, `stream-control` | P17, P18T decision | fake-clock overlap/backpressure tests plus real 40-second profile | target topology run |
| P19 - Conditional DCVC hot path and continuous experiment | Only if DCVC remains selected or is retained as a hedge, bind the real kernel backend, remove avoidable synchronization, and retain continuous DPB only if it earns its complexity | `inference-evidence`, `identity-delivery`, `qualification-ops` | P18 | backend identity, sync/guard timing, parity, restart and cache-isolation tests | CUDA-extension/H100 validation if retained |
| P20 - Resident decoder hot path | Stable-shape/model-warm/token-budget experiments with quality gates | `inference-evidence`, `qualification-ops` | P18 | real per-segment TTFT/generation/VRAM/GPU telemetry and parity | BF16/H100 repeat |
| P21 - Same-sample convergence report | Machine-readable A/B for every accepted optimization | `qualification-ops`, `event-semantics` | P18-P20 | report validator, hashes, common quality projection | none |
| P22 - Mage go/stop decision | Numeric lane/cost requirements and explicit route decision | `qualification-ops`, `canonical-integration` | P21 | deterministic decision tests and rollback rendering | target confirmation |
| P23 - Conditional Qwen Batch candidate | If Mage stops, a fair one-worker batch route is optimized and compared | `inference-evidence`, `canonical-integration`, `qualification-ops` | P22=`STOP` | same 40-second local A/B, batch/micro-batch tests, quality comparison | H100 batch run |
| P24 - Target qualification | Replace scenario factors with measured two-H100/Linux/container evidence | `qualification-ops` | selected route | not locally satisfiable | RunPod/H100, storage and DB participation |

## Module Phases

### `qualification-ops` - P16: freeze baseline and capacity ledger

**Result**

A versioned local convergence report reads retained evidence rather than copying numbers by hand,
separates lifecycle and recurring work, and computes RTF, camera-hours/day, and required lanes.

**Primary paths and entry points**

- `docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json` - retained control.
- `src/robata/benchmark/` or a focused new `src/robata/inference/mage_throughput.py` - pure
  calculations and validation.
- `scripts/compare_local_mage_25x.py` - deterministic report construction from exact inputs.
- `tests/unit/test_compare_local_mage_25x.py` - formulas, missing evidence, and label discipline.

**Implementation outline**

1. Pin source report exact and semantic hashes and reject changed/missing evidence.
2. Represent lifecycle, recurring stage, per-segment, resource, and quality measurements without
   turning absent facts into zero.
3. Emit local observed rates separately from H100 scenarios. Every scenario records its source,
   multiplier, and `measured=false`.
4. Calculate bottleneck service rate for serial, same-device guarded, and separate-device pipeline
   compositions.

**Keep intact**

- No retained report is edited in place.
- One-time model load is excluded only for a proven resident lifecycle.
- `camera-hours` is never mislabeled as `recording-hours`.

**Done when**

- [ ] Exact input hashes and formulas are tested.
- [ ] Current local baseline values reproduce the table above within numeric tolerance.
- [ ] Missing stage measurements produce `UNAVAILABLE`, not fabricated estimates.

**Run locally**

```powershell
python -m pytest -q tests/unit/test_compare_local_mage_25x.py
python scripts/compare_local_mage_25x.py --help
```

### `inference-evidence` / `stream-control` - P17: incremental Provider V2 session

**Result**

The existing binary-safe JSONL worker can be opened once, receive one request at a time, return an
exact verified response, stay resident, and close with complete process telemetry. A caller can
therefore act on segment 0 before jobs 1-4 exist.

**Primary paths and entry points**

- `src/robata/inference/mage_dcvc_prewarm.py` - refactor the all-at-once helper around a reusable
  resident session instead of changing the wire.
- `src/robata/inference/mage_dcvc_preparation_worker.py` - unchanged V2 reset-per-job backend.
- `tests/unit/test_mage_dcvc_prewarm.py` - incremental response, timeout, crash, extra output,
  close, and one-load semantics.

**Implementation outline**

1. Extract a context-managed resident worker session from `_run_worker_process`.
2. Keep exactly one outstanding JSONL job for V2. `submit/receive` concurrency is not added to the
   worker protocol.
3. After each admitted worker response, build the existing `MageCodecCacheEntryV2`, write its
   sidecar through a same-directory temporary file plus atomic create-if-absent hard-link
   publication, and re-verify source,
   provider artifact, and every asset before reporting readiness. Exact existing sidecars are
   recovery hits; truncated or different sidecars fail closed and are never overwritten.
4. Preserve canonical JSON validation, request/response matching, stderr hashing, timeout,
   termination, and clean EOF rules.
5. Reimplement the existing whole-prewarm convenience function over the session so old callers and
   evidence remain equivalent. Seal the aggregate immutable manifest only at end-of-stream; it is
   no longer the prerequisite for observing segment 0.

**Keep intact**

- V2 still resets sequence per job and publishes atomically.
- A failed or timed-out job never publishes a cache manifest.
- The worker process never becomes authoritative inference state.

**Done when**

- [ ] First response is available before later requests are sent.
- [ ] Existing V2 prewarm tests remain green without semantic drift.
- [ ] Session telemetry proves one process start and one engine load.

**Run locally**

```powershell
python -m pytest -q tests/unit/test_mage_dcvc_prewarm.py tests/unit/test_mage_dcvc_preparation_worker.py
```

### `source-media` / `inference-evidence` / `qualification-ops` - P18T: traditional H.264/HEVC candidate

**Decision boundary**

Mage's traditional codec path is a complete alternative preprocessing engine, not the first
stage of DCVC and not model cold-start. The checkpoint's `CodecConfig` defaults to `engine="hevc"`;
on a cache miss it invokes the external `cv-preinfer` tool, while `engine="dcvc-rt"` invokes
the bundled neural codec. Both paths return the same processor interface (`images`,
`src_positions`, `fps`, `out_dir`, and `meta`), but they use different score sources and are not
assumed to produce equal canvases, observations, or business facts.

Traditional preprocessing reads H.264/HEVC codec structure and selected decoded pixels from the
existing compressed input. It avoids the neural DCVC re-encoding/recurrent CUDA pass, but it still
performs packet probing, selected-frame decode, bit-cost extraction, readiness grouping, canvas
construction, and JPEG publication. Therefore it can remove the current DCVC bottleneck only if
those measured CPU/native stages are faster; it is not described as zero-copy or free.

The controlled source is already H.264/AVC (`4337aefbc597a28fa97c10f17ea24555ad03f694b10d3710ac5f96022a565b47`,
1600x1300, 1,225 frames, about 40.833 seconds). The retained five focus inputs are codec-preserving
eight-second H.264 stream copies with 240 frames and a keyframe at each segment start. This makes
them valid for the first traditional A/B; it does not prove that arbitrary fixed-duration cuts are
safe. Production input admission must either use keyframe-aligned segments or prove decodability
and timestamp/packet-boundary fidelity for each exact materialized segment.

**What the candidate can and cannot improve**

| Cost | Expected effect | Required accounting |
| --- | --- | --- |
| Endpoint/model load and first CUDA kernels | none | remains a separate one-time lifecycle measurement |
| First `cv-preinfer` process/native-library load | candidate-specific cold lifecycle | measured once, never averaged into recurring per-segment service |
| First preparation of each unique source segment | primary optimization target | clean-cache miss for every measured entry |
| Exact verified cache hit | only asset load/processor work remains | measured separately from preparation |
| Mage generation | no direct speedup | same model, decoder, prompt, token budget, and generation lane |
| Same-host pipeline occupancy | may improve | CPU/native traditional preparation may overlap GPU generation; CPU, disk, and processor-lock contention are measured |

Current warm observation service averages 3.150942 seconds per eight-second segment. Even a
zero-cost codec stage would therefore cap this local lane near `8 / 3.150942 = 2.5389x`, or about
60.93 camera-hours/day. It would still require 10 local-equivalent lanes for the 25x camera-hour
scenario and 60 lanes for the unresolved 150x six-independent-camera scenario. Traditional codec
can remove a stage bottleneck and free a target GPU from DCVC work; it cannot by itself make one
local lane satisfy 25x.

**Pinned toolchain and identity**

1. Qualify only in a pinned Linux container. The available upstream wheel is CPython 3.12
   manylinux x86_64, not a native Windows package. Windows Docker execution is valid local
   evidence only when the image digest and Linux toolchain are recorded.
2. Pin `codec-video-prep==0.2.5` and its exact wheel SHA-256
   `1fdf52a26a3499b915a3921926391ab78afe0bc703697eacf7da187c43bfbab6`, plus the
   Python base-image digest, CPython ABI, bundled decoder/shared-library hashes, NumPy/OpenCV/Pillow
   locks, Mage checkpoint remote-code hash, and every effective CLI/default value.
3. Set the traditional policy to `codec_mode=traditional` and `preprocess_device=cpu`. A CUDA
   declaration is rejected because the selected provider has no CUDA preparation phase.
4. Do not use Mage's upstream path-derived MD5 cache locator as authority. The Robata entry binds
   exact source bytes, materialized interval/segment identity, codec name, toolchain, provider
   implementation, full effective config, canvas list, `meta.json`, `src_patch_position.npy`,
   and all asset hashes.
5. Create a successor internal provider/recipe/cache namespace and exact-cache binding. The current
   `MageVideoCodecCacheBinding` and runtime admission explicitly accept only `engine=dcvc-rt`; they
   must remain fail-closed for Provider V2 rather than being silently generalized.
6. Publish one complete entry atomically, verify it after publication, and then emit `CODEC_READY`.
   Exact existing entries are recovery hits; incomplete, symlinked, path-escaped, source-mismatched,
   or toolchain-mismatched entries are rejected and never overwritten.

**Segment-ready execution**

1. Start with one independent exact traditional preparation per existing eight-second focus
   segment. Use one resident provider process so process/native-library load is lifecycle rather
   than five repeated starts.
2. Run preparation outside the Mage runtime's processor lock. The producer performs `cv-preinfer`,
   hashes and atomically admits the entry, then the endpoint consumes only the exact bound assets
   through Mage's own `_load_codec_result` surface.
3. Permit CPU/native preparation of segment `n+1` while the GPU generates segment `n`, with queue
   depth one. Traditional preparation must report no CUDA work; CPU time, thread count, RSS, disk
   read/write, page-cache state, queue wait, and consumer starvation are retained.
4. Treat `parallel_segments`, `parallel_decode_cv_reader`, decode backend, native thread type/count,
   and guard-frame settings as measured provider recipe fields. They are not silently enabled
   because internal package parallelism can contend with the decoder host and change output.
5. Measure the integrated union directly:

   ```text
   first_prep + sum(max(next_prep, current_generation)) + last_generation + handoff/commit
   ```

   rather than adding independently timed stage totals or calling a cache-hit run end-to-end.

**Why whole-recording precompute is deferred**

The upstream traditional path samples and groups over the complete file supplied to one call and
publishes its cache directory only after the call completes. Running the 40-second original once
with the current 64-frame/8-canvas budget would spread that budget across the whole recording
instead of applying it independently to each eight-second observation, and readiness groups may
cross Robata segment boundaries. Increasing the budget does not make one resulting asset directory
equivalent to five independently addressable Mage requests. A later single-source packet/bit-cost
scan is allowed only if it emits incremental per-segment assets that are byte-equal to the accepted
independent recipe, preserves timestamp/keyframe semantics, and passes crash replay. It is not part
of the first P18T implementation.

**Same-sample A/B and quality gate**

- Lock the exact five segment bytes and intervals, `cam_01`, target canvas 8, group size 8, one
  image per group, patch 16, max pixels 65,536, group limits 8/128, checkpoint/model identity,
  observation prompt/schema, 256-token ceiling, one worker, and one generation lane.
- Record traditional defaults that differ from DCVC, including input-codec bit-cost score source,
  uniform-count sampling, keyframe avoidance, readiness threshold mode/value, bit-cost grid, decode
  backend, and canvas encoding. Same interface is not quality equivalence.
- Compare canvas count/geometry, sampled frame IDs, timestamp coverage, patch-position validity,
  prompt and visual token counts, output exhaustion, raw text, normalized `MageObservation`, and
  QA/event/evidence/track/fusion comparison projections. Visual asset hashes are expected to differ.
- The five-segment run may establish functional A/B and mean service only. Warm p95 requires at
  least 10 isolated clean-cache repetitions (50 segment preparations) with the first native load
  labeled separately. Production quality requires representative labeled data and an explicit
  non-inferiority margin; one recording cannot promote the route beyond local `HOLD`.

**Adopt, hold, and stop gates**

- Prefer traditional for P18 when its warm preparation p95 is no slower than the measured warm
  observation p95, so the codec stage is no longer the bottleneck, and all correctness/recovery
  gates pass.
- It may remain a candidate when it does not fully shift the bottleneck only if the measured
  integrated recurring 40-second pipeline improves at least 20% over selected DCVC V2 and reduces
  the accepted target lane/GPU cost. A faster standalone CLI that does not improve the integrated
  union is not adopted.
- Record `STOP_TRADITIONAL` if the pinned container cannot process every exact segment, strict cache
  identity/admission cannot be preserved, preparation unexpectedly uses or contends for CUDA,
  quality/recovery fails, or the integrated improvement is below 20%. DCVC V2 remains rollback.
- Record `HOLD` rather than `GO` while the 25x camera-hour versus 150x recording-hour contract
  conflict, target Linux CPU supply rate, two-H100 decoder service, or representative quality is
  unmeasured. With two equal decoder lanes, 25x requires at least 12.5x per lane (at most 0.640
  seconds service per eight-second segment); this is a target measurement, not a local inference.

**Done when**

- [ ] The report explicitly states that traditional H.264/HEVC is an alternate engine, not cold-start.
- [ ] One pinned Linux provider produces and replays all five exact entries without neural DCVC.
- [ ] A successor exact-cache identity rejects tool, source, config, asset, and engine tamper.
- [ ] Integrated queue-depth-one execution demonstrates or disproves CPU-prep/GPU-generation overlap.
- [ ] Same-sample observation and downstream quality evidence is adjudicated without requiring false asset equality.
- [ ] A numeric `SELECT_TRADITIONAL`, `HOLD_TRADITIONAL`, or `STOP_TRADITIONAL` decision feeds P18.


### `canonical-integration` / `stream-control` - P18: segment-ready bounded pipeline

**Result**

One producer materializes and prepares exact assets from the P18T-selected codec provider in
ordinal order while one Mage consumer generates in ordinal order. Segment 0 begins inference
immediately after `CODEC_READY`. The producer may be at most one segment ahead, and failure cancels
future production.

**Primary paths and entry points**

- `src/robata/application/canonical/mage_stream_execution.py` - existing bounded execution and
  stage timings.
- A focused additive composition such as
  `src/robata/application/canonical/mage_codec_stream_execution.py` - selected provider session
  plus the existing perception consumer.
- `scripts/run_local_mage_stream.py` or a dedicated benchmark entry point - explicit profile and
  artifact roots.
- `tests/unit/test_local_mage_stream_execution.py` and focused new integration tests.

**Implementation outline**

1. Emit explicit lifecycle events: `SEGMENT_MATERIALIZED`, `CODEC_PREPARATION_STARTED`,
   provider-specific substage events, `PROVIDER_ARTIFACT_COMMITTED`, `CACHE_ENTRY_VERIFIED`,
   `CODEC_READY`, `OBSERVATION_STARTED`, `OBSERVATION_COMMITTED`, and `BACKPRESSURE_WAIT`.
2. Add a versioned internal ready receipt/live namespace. The endpoint may start with an empty
   namespace, but every request must dynamically load the ready receipt and re-verify the exact
   selected-provider entry, source bytes, provider artifact, assets, checkpoint, policy, provider,
   config, and root containment. Missing readiness fails closed; it never invokes dynamic codec
   preparation.
3. Use a bounded queue with capacity one between preparation and generation. No full-recording list
   of prepared cache entries is required before observation 0. End-of-stream seals the selected
   provider's complete manifest for static replay and qualification.
4. For DCVC on the shared RTX 4060 device, the coordinator explicitly schedules GPU turns. The
   current file guard is non-blocking and remains a final safety fence; it is not used as a wait
   queue. CPU materialization/packing may overlap generation, but DCVC CUDA and generation CUDA
   must have zero overlap. For traditional H.264/HEVC, CPU/native preparation may overlap
   generation while CPU/RSS/I/O contention is measured and preparation CUDA use remains zero.
5. Persist exact request/receipt/cache/artifact bindings before advancing durable stage state.
6. Report interval unions, overlap, guard wait, queue wait, stage idle, consumer starvation, and
   ready-queue high-water mark.
7. Before a real integrated RTX 4060 run, perform a provider-specific resource smoke: Mage NF4
   plus resident DCVC for the neural route, or Mage NF4 plus the pinned CPU/native provider for the
   traditional route. OOM, CPU starvation, or allocator instability is an external boundary, not a
   reason to hide unload/reload or queue-stall cost.

**Keep intact**

- Ordinal deterministic projection/tracking/fusion.
- Queue depth never becomes model generation concurrency.
- Backpressure and cancellation prevent unbounded materialization or orphan provider jobs.

**Done when**

- [ ] A fake slow consumer proves producer depth never exceeds one.
- [ ] A fake provider failure proves no later segment is prepared.
- [ ] Provider-specific tests prove forbidden GPU overlap is rejected and traditional preparation
      reports zero CUDA work.
- [ ] Live endpoint admission accepts a newly published exact entry without an aggregate manifest.
- [ ] A provider-specific resource smoke records VRAM/OOM and CPU/RSS/I/O before the real integrated run.
- [ ] A real 40-second report starts inference after the first cache admission, not after the fifth.

**Run locally**

```powershell
python -m pytest -q tests/unit/test_local_mage_stream_execution.py tests/unit/test_mage_dcvc_prewarm.py
```

### `inference-evidence` / `identity-delivery` / `qualification-ops` - P19: conditional DCVC hot path and continuous experiment

**Result**

If P18T keeps DCVC selected or explicitly retains it as a hedge, Robata first makes the actually
executed backend explicit and attacks avoidable synchronization and over-wide guard time.
Continuous DPB is a bounded Provider/Cache V3 semantic experiment only; it is retained only if
measured benefit justifies its identity and recovery cost. If traditional is selected without a
DCVC hedge, P19 is recorded `NOT_APPLICABLE` rather than delaying P20.

**Implementation outline**

1. Bind `kernel_backend` into a successor effective/provider identity. Current local evidence uses
   `pytorch-fallback`; the custom `inference_extensions_cuda` binary is absent. A declared custom
   backend binds the extension exact SHA, Torch/CUDA ABI, target compute capability, and fails
   startup if import falls back. Provider hashing must include the compiled binary rather than only
   `.py/.cpp/.cu` sources.
2. Measure decode, H2D, recurrent GPU step, sampled bitmap D2H synchronization, readiness grouping,
   canvas packing, and artifact commit separately. Batch or defer `.cpu().numpy()` synchronization
   where upstream semantics permit it.
3. Narrow shared-device exclusion to the GPU phase. CPU grouping/canvas/JPEG/hash/commit may overlap
   Mage generation; DCVC CUDA and generation CUDA may not.
4. Run a low-cost 40-second whole-recording readiness upper-bound experiment before implementing
   DPB persistence. Provider V2 already keeps the engine resident; continuous state removes almost
   none of the 1,200-frame recurrent work and must not be assumed to provide a large speedup.
5. Only if viable, create successor internal versions for effective config, preparation/session
   identity, receipt, cache namespace, and provider implementation. Bind recording, camera,
   ordered segment hashes, predecessor chain, global frame index, geometry, kernel backend,
   discontinuity policy, replay anchor, and provider revision.
6. Treat DPB/recurrent tensors as ephemeral. `reset_interval` is not an independent recovery anchor;
   crash recovery replays from an explicit neural intra anchor and immutable media.
7. Compare provider assets, Mage observations, and downstream content against V2. Changed visual
   representation is a quality experiment, not exact parity.

**Done when**

- [ ] The report proves whether custom CUDA or PyTorch fallback actually executed.
- [ ] Backend change alters identity and silent fallback is rejected.
- [ ] GPU guard time and CPU-pack overlap are directly measured.
- [ ] Cross-recording/camera/session reuse and geometry discontinuity are rejected.
- [ ] V2 and successor caches cannot collide.
- [ ] Continuous V3 continues only if recurring DCVC wall improves at least 5%, crash replay matches
      uninterrupted V3, and quality gates pass; otherwise it is frozen as `REJECTED`.

### `inference-evidence` / `qualification-ops` - P20: resident decoder hot path

**Result**

Accepted decoder optimizations reduce recurring generation without changing the observation
contract or hiding a cold-start cost.

**Experiment order**

1. Prove one endpoint/model lifecycle across repeated recordings; separate model load, first CUDA
   kernel warm-up, first long output, and steady requests.
2. Bound output to the compact observation grammar. Compare token counts and stop sequences before
   changing `max_new_tokens`; reject truncation or semantic loss.
3. Measure attention backend and NF4/BF16 profiles independently. Do not infer H100 BF16 behavior
   from local NF4 results.
4. Evaluate fixed input/output shapes, compile, or CUDA Graph only when the Mage remote-code path
   is compatible. Report compilation amortization and fallback counts.
5. Replace Python token callbacks only when they are measured overhead. CUDA Events improve GPU
   timing attribution; they do not count as a throughput optimization by themselves.

**Done when**

- [ ] Segment 0 cold causes are decomposed into lifecycle, TTFT, and output-length contributions.
- [ ] Warm p50/p95 generation, tokens/s, GPU utilization, VRAM, temperature, and power are recorded.
- [ ] Every accepted optimization passes normalized output and downstream quality gates.

### `qualification-ops` - P21/P22: convergence report and Mage decision

**Mage decision states**

- `GO`: all correctness/recovery gates pass, target hardware has measured stage service rates, and
  a 25x composition with 20% headroom uses an explicitly accepted number of logical lanes/GPUs.
- `HOLD`: local architecture and quality pass but target service rates or cost facts are absent.
  `HOLD` is the expected local-only outcome; it is not production approval.
- `STOP`: after avoidable overhead is removed, the path still requires an unverified custom codec
  fork, fails quality/recovery, or its measured/scenario lane requirement exceeds the accepted
  production budget.

**Performance gates**

- Approximately one second of recurring service per eight-second segment is an `8x` lane, not a
  complete 25x deployment. It needs at least four equally sustainable lanes to exceed 25x.
- A single two-stage pipeline that alone satisfies the 25x camera-hour scenario requires each
  stage to sustain at most `8 / 25 = 0.320` seconds per segment. The six-independent-camera
  150x scenario would require 0.0533 seconds per segment per single lane and is not a realistic
  unmeasured claim.
- For one bounded local optimization cycle, a deliberately optimistic scenario of 8x codec and
  10x decoder target scaling implies local warm gates of codec <=2.56 seconds and observation
  <=3.20 seconds per segment. Current four-sample means are 7.022 and 3.151 seconds respectively;
  therefore Mage remains `HOLD`, with DCVC needing at least another 2.74x local improvement even
  under that optimistic scenario.
- The report calculates the minimum sustainable requirement directly: selected-stage service
  rates multiplied by the accepted lane count must produce at least 25.0x aggregate RTF (or the
  resolved production-contract target) after 20% headroom.
- No optimization is accepted solely because mean GPU utilization rises. Throughput, quality,
  memory safety, thermals, and recovery must all pass.

**Quality and recovery gates**

- Same `MageObservation` parse validity and no output-budget exhaustion.
- Common QA/event/evidence/track/fusion comparison projection equal to control, or a labeled-data
  result with an explicit non-inferiority margin.
- Exact artifact replay without media or GPU and fail-closed behavior for identity/cache tamper.
- Restart from durable media/context/artifact bindings; no dependence on lost hidden state.

### P20/P21 local spatial-sampling evidence - 2026-08-09

The traditional provider now has a same-source spatial sweep under the fixed one-camera,
five-segment, one-generation-lane qualification boundary. The authoritative local report is
`docs/mage-spatial-sampling-qualification-2026-08-09.json` (exact SHA-256
`5975e82ea9b445bb435c0f3994b56d1d0212bb81518ff68c93768a4f2d95820a`, semantic SHA-256
`34b7c283b9ac05787b4b7bd912a00097dddb0e592a1ed67f751a297db5501681`). It remains
`LOCAL_NONPRODUCTION_ONLY`.

| Profile | Prompt tokens / segment | Output tokens total | Stream wall | Local RTF | Peak VRAM | Quality disposition |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 8 canvases / 65,536 pixels | 767 | 357 | 32.046 s | 1.248x | 4,928 MiB | `HOLD`: unsupported green-book claim |
| 8 canvases / 98,304 pixels | 935 | 311 | 28.079 s | 1.425x | 5,152 MiB | `HOLD`: duplicate first-segment claims and garment-to-pants drift |
| 8 canvases / 131,072 pixels, fresh run A | 1,215 | 236 | 23.724 s | 1.686x | 5,455 MiB | `HOLD_PARETO_CANDIDATE` |
| 8 canvases / 131,072 pixels, fresh run B | 1,215 | 236 | 24.305 s | 1.646x | measured locally | same ordered output text as run A |

The 131K recomputations used isolated endpoint state and produced the exact same ordered output-text
digest (`755ac127a8e617eb1809cc8f9effea3a88705e0f9407852eebff33d0bed1c170`). They remove the
first-segment repetition and the green-book hallucination, but still call the same garment a
`drawstring bag` in one segment and a `pant` in another. Therefore higher spatial resolution is
selected only as a quality-first local candidate; it is not a production default or representative
quality pass.

The observed 131K wall improvement must not be misread as cheaper high-resolution inference. Warm
generation cost rose from about 88.95 ms/output-token at 98K to 96.11 ms/output-token at 131K.
Total wall fell because the deterministic output became shorter. The conservative repeated local
RTF is 1.646x, requiring 16 equivalent local lanes for the 25x camera-hour interpretation and 92
for the unresolved 150x six-independent-camera interpretation. No H100 scaling claim is made.

The proposed 6-canvas / 131K run is `NOT_RUN_NOT_JUSTIFIED_BY_CURRENT_QUALITY_GATE`: reducing
temporal canvases does not address cross-segment object identity and risks missing short actions or
boundaries. The next quality lever is a bounded temporal object-consistency or label-refine path,
followed by representative labeled-data evaluation.

A new `scripts/run_bounded_mage_benchmark.py` owns endpoint and benchmark subprocesses under
startup, benchmark, overall, and shutdown deadlines. It writes child logs to files, uses a Windows
kill-on-close Job Object, always terminate/wait/kill/reaps in `finally`, and verifies that its
loopback port is closed. This replaces manual `Start-Process` qualification, which could leave the
Codex command executor waiting indefinitely on a deliberately long-lived endpoint. The two real
131K runs completed in 65.280 and 61.593 seconds of bounded lifecycle wall and both proved endpoint
reaping and port closure.

### P20/P21 temporal-memory A/B evidence - 2026-08-09

The bounded serial temporal-memory candidate has now been exercised against a fresh same-source
8-canvas / 131K full-v1 control. The authoritative local report is
`docs/mage-temporal-memory-qualification-2026-08-09.json` (exact SHA-256
`9e06dba340f3b51fcf385811cd4b10cb0442a4680d97570a284576bf0c1be6dd`, semantic SHA-256
`1f3d020be441e1b87ddede0754959494c09b72b452770db5b00c64a6890216a8`). Both bounded
lifecycles succeeded, reaped their endpoint process, and proved the loopback port closed. Model load
was 33.040 seconds for control and 32.932 seconds for the candidate; these one-time values are not
included in recurring stream RTF.

| Measurement | full-v1 control | temporal-memory-v1 candidate |
| --- | ---: | ---: |
| Bounded lifecycle wall | 83.134 s | 79.194 s |
| Recurring 40 s stream wall | 24.263 s | 21.112 s |
| Local end-to-end RTF | 1.649x | 1.895x |
| Generation sum | 22.618 s | 19.560 s |
| Warm generation mean | 3.554 s | 3.529 s |
| Prompt tokens, five calls | 6,075 | 7,261 |
| Output tokens, five calls | 236 | 188 |
| Peak VRAM | 5,447 MiB | 6,115 MiB |
| Nominal local lanes for 25x | 16 | 14, **not accepted** |

The candidate persisted five exact temporal-memory artifacts and five policy-isolated predecessor
links, reloaded four direct predecessors, retained zero hidden authoritative state, and passed CAS,
observation-lineage, and SQLite integrity checks. Therefore the rejection is not caused by a replay
or durability defect.

The candidate is rejected because all five segments collapse to the identical
`pick up a green garment` action. Four later segments repeat the predecessor label instead of the
control's distinct folding/holding phases, and the first segment loses the control's separate folding
action. The 12.99% stream-wall reduction is coupled to this semantic collapse and a 20.34% output-token
reduction, while prompt tokens rise 19.52% and peak VRAM rises 668 MiB. This fails semantic
non-inferiority; the apparent 14-lane capacity must not be counted.

Decision: `temporal-memory-v1 = REJECT`, rollback to explicit `full-v1 8x131K`, keep Mage overall
`HOLD`, and activate the isolated Qwen Batch hedge requested by the operator. This is single-recording
agent visual inspection rather than labeled ground truth, remains `LOCAL_NONPRODUCTION_ONLY`, and
makes no H100 scaling claim.
### `inference-evidence` / `canonical-integration` / `qualification-ops` - P23: conditional Qwen Batch pivot

**Activation rule**

P23 starts only when P22 records Mage `STOP`, or when an operator explicitly requests a parallel
production hedge while Mage remains `HOLD`. It does not delete Mage code or evidence.

**Fair comparison lock**

| Dimension | Required equality or explicit normalization |
| --- | --- |
| Source | same exact 40-second source bytes, interval, and `cam_01` |
| Hardware | same RTX 4060 Laptop, no simultaneous resident Mage model |
| Worker count | one model worker for the local comparison |
| Business output | same common QA/event/evidence/track/fusion comparison projection |
| Quality | parse validity, output exhaustion, event labels/intervals/confidence, QA and fusion dispositions |
| Timing | lifecycle separated from recurring; media materialization and cache hits reported independently |
| Identity | Qwen provider/model/prompt/batch policy has its own identity and artifacts |

**Optimization order**

1. Preserve the original Qwen window/stage composition as the unmodified control.
2. Keep the model resident and reuse decoded frames/tensors across overlapping windows.
3. Replace per-window calls with bounded micro-batches and length-aware packing where the model
   implementation supports it.
4. Consolidate prompts/output budgets only when common business projections remain equivalent.
5. Measure batch fill, padding waste, GPU utilization, VRAM, tokens/s, and end-to-end RTF.

**Done when**

- [ ] The control and optimized Qwen paths are independently replayable.
- [ ] The same-sample report compares Mage and Qwen without mixing cache-only and cold results.
- [ ] The selected route has a numeric 25x lane/cost model and explicit rollback.

### P23 local Qwen native-batch evidence - 2026-08-09

The frozen r12 corpus was reconstructed directly from the read-only inference-evidence SQLite
database rather than repeating the 854-second MCAP export/materialization path. The corpus identity
is `d4bd44f5e573b2abc13000cf9421134ac0e8d00fe92890fc6a7fa265c84425ed`: 51 exact
requests (`41 QA_COARSE`, `10 QA_DENSE`), 306 verified PNG references, 276 unique PNG files,
and six cameras. The machine-readable qualification is
`docs/qwen-native-batch-qualification-2026-08-09.json` (exact SHA-256
`0844c9b7c43bf1db7396977c91eb8575755a971ebf2c1d5b28310fff641084fc`, semantic
SHA-256 `319f334ba4def2cd5446ad0000f0726ee1c947b3cce914b984c7915928f4b956`). Every
real model process was owned by a bounded parent with a Windows kill-on-close Job Object, hard
deadlines, durable logs, and verified child reaping.

| Local RTX 4060 route | Recurring execution wall | Physical generation sum | Normalized parity | Camera RTF | Local lanes for 25x camera-hours | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unchanged serial control | 247.995 s | 245.561 s | 51/51 | 0.988x | 26 | control |
| all-native batch 2 | 60.041 s | 57.068 s | 50/51 | 4.081x | 7 nominal | reject exact-parity gate |
| all-native batch 4 | 55.935 s | 53.570 s | 50/51 | 4.380x | 6 nominal | reject exact-parity gate |
| all-native batch 8 | 75.499 s | 73.050 s | 49/51 | 3.245x | 8 nominal | reject: slower and more drift |
| batch 4 hybrid, recomputation 1 | 65.459 s | 63.032 s | 51/51 raw and normalized | 3.743x | 7 | pass local parity |
| batch 4 hybrid, recomputation 2 | 65.718 s | 63.181 s | 51/51 raw and normalized | 3.728x | 7 | pass local parity |

The accepted local candidate is `task-claim-group-v1` packing with physical batch size four for
single-claim requests and explicit serial generation for multi-claim requests. The latter guard is
not an error fallback: all-native batching changed one two-claim QA decision from `GOOD` to
`DEGRADED`, while the hybrid policy reproduced every raw and normalized serial output in two fresh
model lifecycles. Median recurring speedup is 3.781x and recurring wall falls 73.55%. Batch eight
reaches 7,915 MiB and regresses both throughput and parity; it is not retained.

Decision: accept the batch-4 hybrid only for a versioned local endpoint/adapter/canonical-binding
integration. Qwen remains `HOLD_FULL_PRODUCTION_QUALIFICATION`: this is a six-camera QA-only
workload, not the same `cam_01` 5x8-second MageObservation/event/evidence/track/fusion projection,
not labeled ground truth, and not an H100/BF16 measurement. Seven local-equivalent lanes is a
capacity model for this exact QA corpus, not proof that seven workers satisfy the full production
pipeline. Rollback remains selection of the unchanged serial binding; no artifact, cache, or identity
is rewritten.

## External Dependencies and Honest Boundaries

- Two H100 GPUs, Linux containers, RunPod networking, and BF16 measurements are external to this
  local cycle. Their values remain absent until measured.
- Production R2 and PostgreSQL/Supabase participation is orthogonal to model throughput. Target
  qualification must record whether each dependency participated; nonparticipation narrows the
  audit scope but does not invalidate measured model-stage evidence.
- Labeled multi-camera quality and cognition-gate recall require representative production data.
- A custom DCVC CUDA extension or upstream fork must be pinned, hashed, licensed, and included in
  provider/checkpoint identity before it can support a production claim.

## Acceptance, Safety, and Rollback

- [ ] No `web/` diff and no published-schema diff.
- [ ] All new internal identities are versioned; V2 artifacts remain immutable and readable.
- [ ] The exact source, model/checkpoint, provider bundle, policy, report, and code commit are pinned.
- [ ] Same-device device guard, queue bound, timeout, cancellation, atomic publication, and stale
      artifact rejection have focused tests.
- [ ] Real benchmark reports include GPU utilization, VRAM, temperature, power, per-stage wall,
      interval overlap, queue wait, and output/quality evidence.
- [ ] Local reports state `production_eligible=false` and do not imply measured H100 scaling.
- [ ] Rollback is route/profile selection only; it never deletes caches or rewrites durable identity.
- [ ] Focused tests, static checks, full regression, Draft PR checks, and branch hygiene pass before
      the cycle is marked complete.

## Suggested Dispatch Prompt

```text
Work on <module-id> / P<16-24> - <phase name>.

Read AGENTS.md, governance/BLUEPRINT.md, and the named module card.
Use D:\Github\Robata\.worktrees\mage-25x-convergence-20260809 only.
Keep the 40-second cam_01 five-segment Provider V2 qualification as the control.
Do not edit web/ or published schemas. Preserve exact-cache admission, artifact replay,
identity isolation, durable fencing, and fail-closed recovery.
Run the phase-local tests and report exact commands, timings, hashes, and external blockers.
```

---

## P20-P21 — 25× Local Convergence, Route Decision, and Bounded Execution (2026-08-09)

This addendum records the current convergence gate without changing the published wire schemas.

### P20 — Fixed common comparison and capacity gate

The fixed local comparison is a 40.0-second `cam_01` recording represented by five non-overlapping segments, one preparation worker, one generation lane, and an RTX 4060 Laptop GPU. Capacity is reported as a local-equivalent lane count only:

```text
ceil(25 / measured_realtime_factor)
```

The machine-readable decision is `docs/robata-25x-route-decision-2026-08-09.json` and the human-readable decision is `docs/ROBATA_25X_ROUTE_DECISION_2026-08-09.md`.

Done evidence:

- Mage native DCVC v2: recurring stream wall 21.962s, 1.821× local RTF, codec-only qualification; production hold.
- Mage traditional H.264/HEVC + 8×131K: hot wall 24.263s, 1.248× local RTF, held for an object-class hallucination.
- Mage temporal memory: rejected for semantic action collapse despite lower wall time.
- Qwen common v2 serial: 5/5 strict parse and downstream recomputation.
- Qwen common v2 Batch4: 5/5 strict parse and downstream recomputation, 1.629× recurring speedup versus serial.
- Real Qwen Batch4 endpoint smoke: one physical batch call, exact replay with no second generation, four-member raw/normalized parity.

No local result is a production or H100 qualification claim.

### P21 — Process ownership and timeout safety

All real local Qwen benchmark invocations must run under `scripts/run_bounded_qwen_batch_benchmark.py` (or an equivalent owned-process runner) with:

- Windows Job Object kill-on-close containment;
- explicit benchmark, overall, and shutdown deadlines;
- durable lifecycle log and child exit code;
- child reaping before the wrapper returns;
- post-run GPU/process cleanup verification.

An outer PowerShell wait with no model child, no GPU progress, or no CPU progress is classified as `OUTER_ORCHESTRATION_FAILURE`, not as model latency. It must not be included in capacity evidence.

### P22 — Route selection and rollback

The selected local hedge is `local-qwen-task-claim-group-hybrid-batch-v1` (native Batch4 for exactly one claim group, explicit serial path for multi-claim requests). Mage stream vNext remains the architectural default and a held candidate. Rollback is selecting the unchanged serial Qwen binding; no artifact, cache, evidence, or idempotency migration is required.

Production admission remains blocked on representative labeled quality, full-pipeline qualification, Linux/H100 sustained capacity, and canary/shadow participation evidence.
