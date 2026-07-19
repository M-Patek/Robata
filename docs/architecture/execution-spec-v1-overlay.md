# Execution Specification V1 Normalization Overlay

- Status: Normative integration overlay; open decisions remain non-promotional
- Date: 2026-07-18
- Source: `large_scale_6camera_video_agent_execution_spec.md`
- Source bytes: `37,694`
- Source lines: `2,587`
- Source SHA-256: `434902fed026726e9e4924042dd1f3f2d2ec26172011efe188aac3f7986e3c0a`
- Governing decisions: Architecture V1.1 Section 25, ADR 0001, ADR 0002, ADR 0003,
  ADR 0004, ADR 0005

## 1. Purpose and interpretation

This overlay preserves the source instruction and turns its product intent into
testable requirements without treating illustrative examples as wire schemas. Source
line references below use the immutable digest above.

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative only in this overlay, Architecture
V1.1, accepted ADRs, and registered schemas. Where these authorities conflict, ADR 0002
applies. Open decisions remain explicit; an implementation cannot close one by choosing
a convenient default.

## 2. Fixed requirement register

The source's fixed list is at lines 2559-2572. The normalized requirements are:

| ID | Requirement | Source | Executable interpretation |
|---|---|---|---|
| FX-001 | Native six-camera source model | 16-20, 52-70, 2561-2563 | Domain slots are exactly `cam_01` through `cam_06`. Unexpected or ambiguous mapping produces validation evidence and cannot publish READY; Architecture Sections 4-6 and 25.6 govern admission. |
| FX-002 | MCAP to six MP4 | 74-95, 2128-2136, 2562 | Produce one immutable `CameraVideoExportManifest` with six ordered `CameraVideoExportRecord` values under ADR 0002. MP4 is derived input, never raw identity or READY proof. |
| FX-003 | Future Qwen primary role | 99-111, 1009-1035, 2564 | Qwen is the intended primary VLM after V1.1 Phase 3 gates. Business logic remains provider-neutral; no current provider implementation is implied. |
| FX-004 | Future GPT shadow role | 115-151, 1039-1089, 2565 | GPT is asynchronous and nonblocking, uses a separate input plan/ledger, and cannot affect primary completion. It remains Phase 8 after governance and operational qualification. |
| FX-005 | Exact timestamp traceability | 499-515, 2566 | Use int64 nanoseconds, explicit clock provenance, half-open intervals, mapping/alignment revisions, and source frame/message locators. Architecture Sections 1.3, 6, 25.3, and 25.6 govern. |
| FX-006 | QA across six cameras | 806-829, 2567 | Evaluate all canonical camera slots and retain camera-level plus recording-level evidence. QA degradation does not rewrite source validation or alignment state. Registered QA policy and quality gates are required for promotion. |
| FX-007 | ActionEvent core entity | 362-391, 1726-1745, 2568 | Events are recording-scoped, immutable revisions with deterministic current selection and complete source lineage under Architecture Sections 14 and 25.9. |
| FX-008 | T+1 QA target | 22-28, 1526-1571, 2569 | Measure against the parameterized SLO in Section 7 below. Passing is prohibited until OD-SLO-001 is resolved. |
| FX-009 | Approximately 500 hours/day | 22-38, 1749-1799, 2570 | Report both workload interpretations in Section 6 until OD-SCALE-001 is resolved. |
| FX-010 | Structured storage | 1635-1654, 2571 | Store registered immutable artifacts, revisions, lineage, ledgers, and selected projections; disconnected JSON or path identity is nonconforming. Architecture Section 16 governs. |
| FX-011 | Action-level retrieval | 1694-1745, 2572 | Resolve an event to source identity, exact interval, camera evidence, and synchronized derived clips with lineage. Architecture Sections 16.5 and 21 govern. |
| FX-012 | Speed subject to quality | 40-44, 833-873 | Optimize cost/throughput only inside a registered quality floor; skipped, incomplete, and failed work remain visible in denominators. |

## 3. Experimental register

The hypotheses listed at source lines 2574-2585 are not production mandates. They require
a benchmark manifest, fixed evaluation cohort, registered ground truth, baseline,
repetitions, resource accounting, uncertainty, and an explicit promotion decision under
Architecture Section 18.

| ID | Hypothesis | Source examples |
|---|---|---|
| EXP-001 | Adaptive/sparse-to-dense sampling | 519-612, 2418-2426 |
| EXP-002 | ROI, resolution, and provider-neutral visual budgets | 616-689, 2400-2414 |
| EXP-003 | Progressive camera evidence | 691-759, 2430-2438 |
| EXP-004 | Qwen size/model comparison | 1093-1172, 2442-2454 |
| EXP-005 | Hierarchical model cascades/student models | 1134-1208, 2207-2217 |
| EXP-006 | Hardware frame decode and serving framework | 1212-1275, 2183-2192 |
| EXP-007 | Dynamic, cost-model-aware batching | 1278-1336, 2472-2480 |
| EXP-008 | GPU topology and data parallelism | 1339-1395, 2484-2492 |
| EXP-009 | Encoder disaggregation/cache/reuse | 1358-1432, 2496-2498 |
| EXP-010 | FP8 or other quantization | 1436-1457, 2458-2468 |
| EXP-011 | Semantic video/token compression | 1494-1523, 2221-2229 |
| EXP-012 | Specific GPU and serving technologies | 1951-1986, 2069-2093 |

The expressions at source lines 622-630 and 2404-2406 are conceptual, not valid metric
equations. Normalize them as:

```text
input_pixels = sum(frame_or_roi_width * frame_or_roi_height)
estimated_visual_tokens = estimator(input_pixels, frame_layout, model_capability_version)
estimated_compute = calibrated_cost_model(frames, input_pixels,
                                          estimated_visual_tokens, text_tokens)
```

FPS, frame count, resolution, ROI, total pixels, and tokens are recorded separately;
derived variables are not multiplied or added as if they had the same unit.

## 4. Normalized contracts

### 4.1 Time and intervals

All canonical time values are signed int64 nanoseconds. Wire values use canonical decimal
strings; in-process values use integers. Intervals are `[start_ns, end_ns)`, with
`end_ns > start_ns`. Every value names its clock/source domain and mapping revision.
Floating-point examples at source lines 230-275 and 287-386 are illustrative only and
MUST NOT appear on registered wires.

Alignment is evidence, not an inference from similar timestamps. Its state, offset/drift
model, residuals, uncertainty, missing-frame policy, and admissibility are separate from
source READY, per Architecture Sections 6 and 25.6.

### 4.2 Identity, revisions, and status

- `recording_identity` derives from namespace and verified raw content digest, never a
  path, MP4, processing run, or provider handle.
- Logical IDs are run-independent. Processing attempts link to logical nodes rather than
  changing their identity, per Architecture Sections 20.2 and 25.4.
- Corrections create immutable revisions. A deterministic selected projection identifies
  the current revision, per Section 25.5.
- `MCAPValidationReport`, `MCAPReadyManifest`, alignment state, derived-video export
  state, primary outcome, and shadow outcome are distinct artifacts/ledgers. No field in
  one silently mutates another.

### 4.3 Confidence and model output

The bare `confidence` examples at source lines 287-415 are not a domain probability.
Any retained confidence MUST include value/range, producer, method/version, calibration
status, and calibration cohort or explicitly be marked `UNCALIBRATED`. Provider claims
remain untrusted provider output. Orchestrator enrichment, reference validation, and
selected-attempt semantics follow Architecture Sections 25.1 and 9.

`TemporalVisualPackage` is immutable and provider-neutral. Provider/model limits,
uploads, resize/transcode choices, request layout, and capability snapshots belong to a
separate immutable `InferenceInputPlan`, per Architecture Section 25.2. Progressive view
changes create a new provider-neutral sampling/package derivation; they do not mutate a
published package.

### 4.4 Camera video export manifest

FX-002 uses the registered `CameraVideoExportManifest` contract in ADR 0002. Its
`cameras` array MUST contain exactly six ordered `CameraVideoExportRecord` values for
`cam_01` through `cam_06`. Manifest-level provenance includes the recording identity,
source content SHA-256 and byte size, mapping profile version/digest/approval, READY and
alignment references/status, and exporter implementation/profile/config identity.

The closed execution-mode invariants are:

```text
LOCAL_DEVELOPMENT_OVERRIDE
  => ready_manifest_id = null
  => mapping_profile.approved = false
  => alignment_status = UNVERIFIED

GOVERNED_READY
  => ready_manifest_id != null
  => mapping_profile.approved = true

alignment_status = VALID => alignment_id != null
```

`GOVERNED_READY` records the source-admission context only; it is not alignment or
primary-admission proof. Every exported packet MUST have a timestamp-sidecar row, and
per-camera conservation MUST reconcile using the actual wire fields:

```text
input_message_count =
    leading_drops.count + exported_packet_count + trailing_drops.count

timestamp_sidecar_artifact.row_count = exported_packet_count
```

`leading_drops` and `trailing_drops` retain stable reason codes and nullable/required
source-time extrema according to their count. `MediaTimeMapping` records exact integer
timebase, first/last PTS, final-sample duration, rounding policy, and maximum mapping
error. Fields ending in `first_observed...` or `last_observed...` are timestamp instants,
not half-open interval endpoints.

The V1 manifest does not carry `artifact_id`, lifecycle, immutable locator version, or
explicit parent artifact IDs and remains frozen as a reader contract. ADR 0003 introduces
V2 instead of changing V1 or fabricating lineage through an upcast. V2 pins exact schema
references, embeds registered input and camera-artifact references, and stores lifecycle,
immutable locator version, exact bytes, and typed parents in the external artifact-registry
entry/snapshot. The manifest's own artifact ID and exact-byte digest remain external so
the manifest does not hash itself.

Timestamp-sidecar NDJSON rows continue to use the exact pinned
`CameraVideoTimestampRow` schema and retain their schema/export-profile versions;
wire/immutability conformance remains mandatory. The local V2 implementation closes the
former registry blocker for this derived-artifact slice only. ADR 0004 separately closes
the generic logical-node identity/association primitive with two-run replay evidence. ADR
0005 subsequently implements the generic node-scoped immutable-revision, append-only
selection-decision, atomic current-selection, and deterministic rebuild primitive. These
generic primitives do not admit a concrete producer, establish business eligibility or a
selection policy, or implement current-validity/invalid-descendant work planning; those
gates remain open in Phase 1A. Export output never supplies source validation, READY, or
alignment proof.

### 4.5 Media terminology

| Term | Meaning |
|---|---|
| extract | Read selected encoded access units and metadata from MCAP. |
| remux | Package compatible encoded access units in MP4 without decoding/re-encoding. |
| transcode | Decode and re-encode video into a configured output codec/profile. |
| frame decode | Convert encoded access units to pixel frames for analytics or probes. |

Source phrases such as "decode to MP4" at lines 14, 456, 2338, and 2562 normalize to an
explicit extract-plus-remux or extract-plus-transcode plan. The `MP4 Queue -> Decode
Queue` sequence at lines 1581-1585 means an MP4-artifact queue followed by a frame-decode
queue; implementations MUST use unambiguous names.

## 5. Correct conceptual flow

Primary and shadow work are not serial dependencies:

```text
Raw MCAP artifact
  -> MCAPValidationReport
  -> selected VALID + durable source + exact mapping
  -> MCAPReadyManifest
       -> independent alignment evidence/admissibility ------+
       -> CameraVideoExportManifest (six derived MP4s) ------+
                                                            |
       selected admissible alignment + complete video manifest
  -> video analytics / sampling / immutable TemporalVisualPackage
  -> provider-specific InferenceInputPlan (Phase 3 or later)
  -> Qwen primary attempt
       -> camera/MCAP QA
       -> event proposal / dense evidence / boundary refinement
       -> multi-view fusion -> ActionEvent
       -> structured storage and retrieval

Selected immutable primary evidence
  -> asynchronous GPT shadow plan/attempt (Phase 8)
  -> separate evaluation/disagreement storage
```

GPT failure, expiry, budget shedding, or non-selection never blocks or changes the primary
outcome. Easy and hard primary cases both rejoin their applicable QA/event/fusion and
storage barriers. This supersedes only the ambiguous arrows in source lines 2334-2391;
it does not alter the product intent.

## 6. Workload normalization

Until OD-SCALE-001 is resolved, every capacity report MUST show both scenarios and MUST
NOT claim the fixed workload is met:

| Scenario | Incoming per day | Required average recording-h/wall-h | Required average camera-video-h/wall-h |
|---|---:|---:|---:|
| A: 500 recording hours/day | 500 recording h; 3,000 camera-video h | 20.833333 | 125.000000 |
| B: 500 aggregate camera-video hours/day | 83.333333 recording h; 500 camera-video h | 3.472222 | 20.833333 |

These equations assume exactly six equal-duration camera streams and a 24-hour arrival
period. Reports state deviations, selected denominators, completed/skipped/failed counts,
GPU-hours as the sum of occupied device wall time, and the measurement window. Terms such
as `video-hours/hour`, `processed-video-hour`, and `accepted-video-hour` from source lines
1749-1772 and 1976-1986 are prohibited unless qualified as recording or camera-video and
their acceptance denominator is defined. Architecture Section 19 governs accounting.

## 7. SLO and quality normalization

OD-SLO-001 must fix the business timezone, day boundary, calendar/business-day rule,
source-complete watermark, late-arrival policy, due-time function, eligible denominator,
required completion percentage, exclusions, and sustained observation period. Until
then, report arrival-to-QA-complete latency and parameterized T+1 projections, but report
the SLO verdict as `UNRESOLVED`, never PASS.

OD-QUALITY-001 must register the QA taxonomy/policy, annotation/event ontology, ground
truth/adjudication process, unit of analysis, micro/macro aggregation, minimum recall and
precision by severity, action metric, temporal boundary metric, confidence intervals,
and allowed regression from baseline. The normalized source-policy input in
`docs/architecture/qa-policy-input-v0.md` preserves known ambiguities and is not itself a
promotion threshold. Architecture Sections 18.2-18.4 govern evaluation.

Load shedding may remove shadow/research work first, but it cannot hide primary skipped,
incomplete, expired, or failed work or reduce the registered primary quality floor.

## 8. Open decisions

| ID | Decision required before promotion |
|---|---|
| OD-SCALE-001 | Is 500 hours/day recording duration or aggregate camera-video duration? |
| OD-SLO-001 | Exact T+1 clock, cohort, due-time, completion percentage, and backlog limits. |
| OD-QUALITY-001 | Registered QA/action/annotation ground truth, metrics, floors, and regression budget. |
| OD-MEDIA-001 | Approved MP4 export profile: remux/transcode policy, codec/profile, timebase, keyframe, metadata, and determinism. |
| OD-SOURCE-001 | Approved topics, schemas, roles, auxiliary-channel policy, codecs, and camera mapping (Architecture O-03). |
| OD-CLOCK-001 | Clock provenance, sync evidence, drift/reset behavior, residual/tolerance, and missing-frame policy (Architecture O-04). |
| OD-QA-001 | Six-camera QA aggregation and exact 5-second source ambiguities (Architecture O-10). |
| OD-PROVIDER-001 | Qwen model/deployment/capability and GPT data-use, residency, retention, budget, and shadow approval. |

## 9. Phase mapping and immediate scope

Architecture Section 25.10 supersedes source lines 2097-2229. The binding order is Phase
0, 1A, 1B, 2, 3, 4, 5, 6, 7, 8, and optional 9. The detailed legacy mapping is in ADR
0002. Synthetic or local development work may exercise later interfaces, but cannot
claim a phase exit or process governed production data before its predecessors pass.

The local-media slice normalized in Section 10 is MCAP-to-six-MP4 derived-artifact
evidence. It includes no Qwen/GPT calls, no production provider plan, no source READY
publication, and no Phase 1B promotion. Independent Phase 1A contract primitives may be
implemented under their own ADRs and evidence without changing this media acceptance
scope.

## 10. Executable acceptance: local MP4 slice

The slice passes only when automated tests or a machine-readable conformance report prove
all of the following:

1. **Scope and immutability:** Input bytes and SHA-256 are unchanged. Outputs are written
   atomically as immutable artifacts; temporary or partial outputs are not published.
2. **Six-slot completeness:** The readable local medium sample yields one semantic-valid
   V2 `CameraVideoExportManifest` whose six nonempty `video_artifact` records are ordered
   `cam_01` through `cam_06`, with no duplicate/missing slot.
3. **Exact provenance:** The manifest records source SHA-256/size, recording identity,
   mapping profile version/digest/approval, source topic/channel/`schema_name`/codec, exporter
   implementation/version/mode/profile/config digest, timestamp-sidecar digest, and each
   video content digest/size. The registry snapshot additionally pins schema artifacts,
   artifact IDs, lifecycle, immutable locator versions, and typed parent IDs. The
   manifest's own ID, locator, and exact-byte digest are authoritative only in its
   external registry entry. URIs are locators only.
4. **Media validity:** An independent media probe recognizes MP4/H.264 for every slot;
   each stream decodes at least one `1600 x 1300` frame. Probe/tool versions are captured.
5. **Timestamp conservation:** Each camera satisfies the two Section 4.4 count equations.
   Sidecar rows retain source log/publish/header time, source sequence, packet index,
   PTS/DTS, duration, and keyframe facts. `MediaTimeMapping` has valid first/last PTS and
   positive `last_duration` in its integer timebase. Initial undecodable access units or
   any other drops use structured reason/range provenance and are never silent. Every
   NDJSON row validates against the registered `CameraVideoTimestampRow` schema.
6. **Determinism:** Repeating export with identical input, exporter implementation/profile
   versions, and `canonical_config_sha256` yields the same six content SHA-256 values,
   timestamp-sidecar digests, and semantic manifest digest. Exporter-generated timestamps
   or random container metadata are normalized rather than ignored by ad hoc test code.
7. **Failure closure:** The known corrupt local sample returns the stable corruption code,
   publishes no camera artifact/complete manifest, and leaves no result masquerading as
   success. Injected write/probe failure also prevents complete-manifest publication.
8. **Boundary mode:** The local result records
   `execution_mode = LOCAL_DEVELOPMENT_OVERRIDE`, `ready_manifest_id = null`,
   `mapping_profile.approved = false`, and `alignment_status = UNVERIFIED`. It is not
   named or counted as an `MCAPReadyManifest` and does not change source/alignment
   ledgers. Tests separately enforce the `GOVERNED_READY` invariants in Section 4.4.
9. **Provider isolation:** Network/provider request count is zero; Qwen and GPT usage,
   frames sent, tokens, and provider cost are all zero.
10. **Conformance:** Registered manifest and timestamp-sidecar wire validation, semantic
    six-slot/count/mode validation, artifact-registry provenance traversal, negative
    fixtures, Ruff, formatting, strict mypy, and the full test suite pass. Local-real-data
    tests remain explicitly opt-in and non-promotional. The V2 registry implementation
    and evidence must be used; the frozen V1 output alone cannot satisfy this item.

The sample discovery and corruption observations are documented in
`reports/local-mcap-spike-2026-07-18.md`. The completed local six-camera exercise,
determinism evidence, independent decode, and resource limitations are documented in
`reports/local-six-camera-video-export-2026-07-18.md`. The V2 exact-schema, registry,
lineage, replay, materialized-view, and real-media evidence is documented in
`reports/local-artifact-registry-v2-2026-07-18.md`. ADR 0004 and
`reports/local-logical-node-membership-2026-07-18.md` separately cover the generic
logical-node identity/association primitive. ADR 0005 and
`reports/local-immutable-revision-selection-2026-07-18.md` cover the subsequent generic
immutable-revision/current-selection primitive and its final local verification. These
slices do not complete Phase 0, remaining concrete producer/business-policy work in Phase
1A, Phase 1B, any open decision, or any throughput, quality, SLA, or production-capacity
claim.
