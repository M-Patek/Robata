# ADR 0002: Execution Specification Integration and MP4 Artifact Boundary

- Status: Accepted; V1 wire frozen and registry evolution supplemented by ADR 0003
- Date: 2026-07-18
- Scope: Authority, phase ordering, and the provider-neutral MCAP-to-MP4 boundary

## Context

`large_scale_6camera_video_agent_execution_spec.md` is a new product and research
instruction. It fixes the intended six-camera product shape, MCAP-to-MP4 workflow,
future Qwen primary/GPT shadow roles, throughput target, research matrix, and reporting
expectations. Its own lines 26-28 leave the unit of `500 hours/day` unresolved, and its
legacy phase order at lines 2097-2229 predates the dependency gates in Architecture V1.1.

The source instruction is not a wire contract. Several examples use floating-point
seconds, path-like identities, untyped confidence, and combined source/alignment states.
Architecture V1.1 Section 25 and ADR 0001 already define stricter security, identity,
time, state, schema, and dependency rules. Silently choosing between these authorities
would make artifacts and phase claims non-reproducible.

The instruction also makes `MCAP -> 6 MP4` a fixed product requirement (lines 74-95 and
2559-2572). MP4 is useful as a video-analytics input, but it is derived data. It cannot
replace raw-source identity, source validation evidence, camera mapping, or READY
publication.

## Decision

### Split authority

The following authority split is normative:

1. `large_scale_6camera_video_agent_execution_spec.md` owns product intent, fixed
   functional goals, workload/SLO questions, experiment priorities, and report content.
2. Architecture V1.1 Section 25 owns security and provider trust, domain and wire
   boundaries, exact time, logical identity, immutable revisions, status ledgers, schema
   evolution, and implementation dependency order.
3. ADR 0001 owns the executable Python baseline, ports-and-adapters boundary, contract
   authority, and Phase 0/1A/1B promotion gates.
4. Checked-in registered JSON Schemas own wire compatibility. Domain validators own
   cross-record invariants that JSON Schema cannot express.

Architecture Sections 1.3, 3, 4.2, 5.3, 6, 8-9, 16, and 18-20 remain applicable where
Section 25 does not supersede them. The normalized interpretation of the new instruction
is recorded in `docs/architecture/execution-spec-v1-overlay.md`.

A conflict is never resolved only in application code or prose comments. A material
change requires all applicable evidence:

- an accepted ADR or architecture revision that states the decision and migration;
- a registered schema revision when the wire shape changes;
- semantic validation and golden/conformance tests;
- replay or migration evidence for existing immutable artifacts;
- an updated requirement-to-evidence mapping.

The source execution specification remains unchanged as provenance.

### CameraVideoExportManifest

The registered provider-neutral wire artifact is `CameraVideoExportManifest`. Its
manifest-level fields are `schema_version`, `execution_mode`, `recording_identity`,
`source_content_sha256`, `source_size_bytes`, `mapping_profile`, `ready_manifest_id`,
`alignment_id`, `alignment_status`, `exporter`, and `cameras`.

`cameras` contains exactly six `CameraVideoExportRecord` values in canonical order
`cam_01` through `cam_06`. Each record contains:

- canonical camera ID and `SourceVideoStream` topic/channel/`schema_name`/codec provenance;
- input-message count, observed source/export timestamp extrema, structured leading and
  trailing drop provenance, exported packet/frame/keyframe counts, width, and height;
- a content-addressed `video_artifact` and `timestamp_sidecar_artifact`, each with URI,
  exact SHA-256, byte size, and media type; the sidecar also records row count;
- `MediaTimeMapping` with `zero_source_ns`, integer timebase numerator/denominator,
  `first_pts`, `last_pts`, `last_duration`, tail-duration policy, `HALF_EVEN` rounding,
  and maximum timestamp-mapping error.

The exporter identity records implementation name/version, `REMUX` or `TRANSCODE` mode,
export-profile ID/version, and canonical config SHA-256. Observed first/last timestamp
fields are instants, not interval endpoints. Nanosecond instants use canonical int64
rules; any declared interval uses the half-open convention in Architecture Sections 1.3
and 25.3. In media ticks, `last_pts + last_duration` is the exclusive end of the final
sample and must remain in signed-int64 range.

Execution mode has exactly two values and records source-admission context:

- `LOCAL_DEVELOPMENT_OVERRIDE` requires `ready_manifest_id = null`,
  `mapping_profile.approved = false`, and `alignment_status = UNVERIFIED`.
- `GOVERNED_READY` requires a non-null `ready_manifest_id` and
  `mapping_profile.approved = true`.
- For either mode, `alignment_status = VALID` requires a non-null `alignment_id`.

`GOVERNED_READY` does not itself prove alignment admissibility or primary-processing
admission. Production consumers still require the selected READY manifest and an
alignment state admissible for their policy, as required by Architecture Section 25.6.
Neither mode publishes READY or changes a source/alignment ledger.

The V1 wire content-addresses derived bytes but does not carry the artifact registry's
`artifact_id`, lifecycle, immutable locator version, or explicit parent artifact IDs. It
is frozen as a reader contract. ADR 0003 closes that local derived-artifact gap through a
new V2 wire, exact schema catalog, external registry entries, typed lineage, replay, and
registry-backed materialized views; it does not mutate V1 or define a V1-to-V2 upcast.
Timestamp-sidecar NDJSON rows remain governed by the exact pinned
`CameraVideoTimestampRow` schema and retain schema version, export profile ID/version,
source timestamps/sequence, packet timing, duration, and keyframe facts. The V1 evidence
alone cannot satisfy artifact-registry acceptance, and the V2 local evidence is still
non-promotional.

Locators, processing run IDs, hostnames, and provider handles do not define logical
identity. Temporary files are not artifacts, consistent with Architecture Section 5.3.
Partial export and failure evidence are recorded separately and never published as a
complete `CameraVideoExportManifest`.

`remux`, `transcode`, and frame `decode` are different operations. The exporter records
which occurred; the generic phrase "decode to MP4" is not used in executable contracts.

### Dependency order

The legacy phases in execution-spec lines 2097-2229 are intent groupings, not promotion
gates. Architecture Section 25.10 is the only phase dependency order:

| Legacy execution-spec phase | V1.1 interpretation |
|---|---|
| 1, Architecture Baseline | Design input spanning 0, 1A, 1B, and 2; it does not waive any exit gate. |
| 2, Baseline Production Pipeline | Split across 1B source/time admission, the MP4 derived-artifact slice, 2 provider-neutral evidence, 3 Qwen boundary, and 4 QA. |
| 3, Event and Action | Phases 5 and 6. |
| 4, GPT Shadow | Phase 8, after Phase 7 operational qualification. |
| 5, Adaptive Compute | Experiments feeding Phases 2, 4, 5, and 7; promotion requires benchmark evidence. |
| 6, Serving Throughput | Phase 7 qualification. |
| 7, Progressive Camera Evidence | Provider-neutral package experiments before provider planning; promotion is qualified in Phase 7. |
| 8, Model Cascade | Post-baseline experiment; provider use remains gated by Phases 0, 3, 7, and 8. |
| 9, Advanced Research | Optional Phase 9 or later research after prerequisite gates. |

The immediate scope is contract and local derived-media evidence. No Qwen or GPT
implementation or call is authorized by this ADR. No local sample or derived MP4 may be
called production READY or used to claim Phase 1B exit.

## Consequences

- Product intent is retained without weakening existing security or artifact contracts.
- MP4 becomes reproducible, attributable derived data rather than an untracked path.
- Source validity, source READY, alignment, derived-video completeness, and provider
  inference remain independently auditable states.
- Capacity is reported under both 500-hour interpretations until the product owner fixes
  the unit; neither scenario may be silently selected.
- Qwen/GPT work stays deferred until its V1.1 predecessors and provider-governance gates
  pass.
