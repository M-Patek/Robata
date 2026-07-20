# Implementation Plan

## Objective

Implement the smallest executable foundation in the dependency order required by Architecture V1.1 Section 25. Phase 0 is a security/privacy/data-governance hard gate. Phase 1A establishes reusable contracts without claiming source admission. Phase 1B admits representative real MCAPs only after Phase 0, Phase 1A, governance approval, O-03, and O-04.

Phase labels report evidence, not elapsed work. A later-phase synthetic spike may run in isolation but cannot satisfy that phase's exit gate or process governed production data.

### Authority and Execution-Spec Integration

`large_scale_6camera_video_agent_execution_spec.md` is authoritative for product intent,
fixed functional goals, research priorities, and reporting expectations. Its normalized
interpretation is `docs/architecture/execution-spec-v1-overlay.md`, under the authority
split accepted by `docs/adr/0002-execution-spec-integration.md`.

Architecture V1.1 Section 25, ADR 0001, accepted ADRs, and registered schemas remain
authoritative for security and provider trust, wire/domain contracts, logical identity,
status ledgers, exact time, and dependency order. A conflict requires an explicit ADR or
architecture revision plus applicable schema, semantic-validator, migration/replay, and
conformance-test changes; implementation code cannot silently choose an interpretation.

The live-tree implementation map is `docs/current-implementation-status.md`. Files under
`archive/old_mvp` are historical, non-normative evidence and do not establish the status of
the current worktree unless their claims are independently reproduced by current checks.

## Phase 0: Security, Privacy, and Data Governance

### Deliverables

- Approved data classification and threat model covering MCAPs, frames, prompts, provider output, annotations, metadata, and derived artifacts.
- Least-privilege service/user identities, RBAC matrix, secrets lifecycle, and credential-rotation procedure.
- Encryption requirements for data at rest and in transit, with key ownership and recovery rules.
- Audit-event contract and immutable audit coverage for source access, artifact derivation, provider transmission, review, export, and deletion.
- Retention, deletion, legal-hold, redaction, residency, and incident-response policies with accountable owners.
- Explicit trust and artifact boundaries for local workers, durable stores, logs/traces, providers, shadow routes, and human review.
- Approved provider data-use/residency terms and separate approval for every shadow route.
- Incident and credential-rotation exercises, including proof that no alternate provider route bypasses controls.
- Written corpus-use decision for local real-data assets. `file.zip` currently inventories 37 MCAP members but is ignored, unadmitted, and non-promotional.

### Acceptance

- No governed production frame or prompt leaves the approved trust boundary before all applicable controls pass.
- Least privilege, secret rotation, audit capture, retention/deletion/legal hold, and incident paths have executable evidence.
- Production-like tests show that every provider and shadow route is fail-closed when approval or credentials are absent.
- Local possession or discovery of a source never implies permission to inspect, derive, transmit, retain, or promote it.

## Phase 1A: Executable Contract Foundation

### Deliverables

- Python project constrained to `>=3.12,<3.14`, managed and locked with `uv`.
- Modular-monolith package layout with domain, application, port, adapter, and CLI boundaries.
- Immutable schema registry entries containing schema ID/version/artifact/digest, owner, compatibility mode, lifecycle, canonicalization/projection version, and supported software range.
- Machine-verifiable schema compatibility checks and deterministic registered-upcaster fixtures; unknown digests and ambiguous paths fail closed.
- Common domain types for canonical camera slots, canonical integer nanoseconds, half-open intervals, digests, versions, and validation outcomes.
- Exact rational sampling-grid types/vectors using integer arithmetic, half-even rounding, tolerance, tie-break, clipping, and dedupe policy versions.
- Deterministic canonical JSON, semantic digest, exact-byte digest, and run-independent logical identity utilities.
- Artifact-registry primitives that distinguish semantic identity from exact bytes and reject mutation.
- Run-independent logical nodes plus run-node membership links, with replay under different run IDs reusing valid nodes.
- Immutable revision, append-only selection-decision, and deterministic current-selection projection primitives.
- Authoritative checked-in JSON Schemas and semantic validators for the contract kernel, including six-slot, ordering, interval, count, mapping, identity, and lineage invariants.
- Unit, schema-conformance, property, golden-vector, mutation-rejection, and replay tests.
- Tooling for formatting, linting, type checking, and tests through documented `uv` commands.

### Acceptance

- A clean environment installs from the lock file and runs all checks through documented `uv` commands.
- Wire payloads reject numeric `*_ns`, non-canonical decimal strings, unknown fields, invalid intervals, and missing or extra camera slots.
- Python models serialize to payloads accepted by the checked-in authoritative schemas.
- Semantic validators reject structurally valid states that violate Architecture V1.1.
- Canonical hashing and rational-grid vectors are deterministic across key order, supported runtimes, edge cases, and repeated runs; no digest includes itself or execution-local identity in its preimage.
- Replay under two run IDs reuses logical nodes while preserving both run-node memberships.
- Published artifacts/revisions reject mutation, and current-selection projection rebuild is deterministic.
- No database, broker, provider, or source-specific assumption leaks into the domain layer.
- Synthetic source fixtures, if used for an isolated Phase 1B spike, are labeled synthetic and do not count toward Phase 1B exit.

## Current Run-Independent Logical-Node Slice: Phase 1A

**Status: local executable primitive implemented and evidenced; Phase 1A remains open
for concrete producer identity/revision admission, business eligibility, and other
applicable gates.**

This slice implements the identity and association subset of Architecture V1.1 Section
25.4 without treating processing runs as semantic input. ADR 0004, the exact `LogicalNode` and
`ProcessingRunNodeMembership` schemas, and a dedicated SQLite adapter define the
boundary. The node identity is exactly `(node_type, node_logical_key)`. The membership
unique key is exactly `(run_id, node_type, node_logical_key, role)`.

### Deliverables

- Closed version 1.0 logical-node and processing-run membership contracts with exact
  schema-catalog pins.
- Namespaced semantic-digest keys whose node records structurally exclude run, work,
  lease, attempt, time, path, locator, and provider-handle fields.
- Producer responsibility for typed, run-independent semantic projections; the generic
  registry does not accept arbitrary mappings as identity preimages.
- Atomic local attach that derives `CREATED` for the first run and `REUSED` for later
  normal attachments.
- Immutable `CREATED`, `REUSED`, `INVALIDATED`, and `OBSERVED` memberships with the exact
  Architecture V1.1 four-part unique key.
- Canonical JSON plus normalized-column verification, one creator per node, composite
  foreign-key restriction, exact retry, commit-uncertain recovery, and fail-closed
  conflicts.

### Acceptance

- Replay under two run IDs stores one logical node and preserves two independently
  queryable memberships.
- Two concurrent adapters converge on one node, one `CREATED` membership, and all
  distinct run memberships.
- A failed transaction exposes neither a new node nor its membership.
- Node/membership record-digest, canonical JSON, normalized-column, creator, foreign-key,
  and database-schema corruption fail closed on verified reads.
- `INVALIDATED` and `OBSERVED` require an existing verified node and never mutate global
  node content or prior memberships.

Historical evidence and explicit limitations are retained in
`archive/old_mvp/reports/local-logical-node-membership-2026-07-18.md`. ADR 0005 and its separate evidence
report cover the subsequent generic revision/current-selection primitive. The combined
local primitives do not implement concrete-producer identity/revision admission, business
selection policy, work scheduling, global validity, cross-registry atomicity, or a
production database.

## Current Immutable Revision/Selection Slice: Phase 1A

**Status: local executable generic primitive implemented; final local repository
verification passed. Phase 1A remains open for concrete producers, business eligibility and
selection policy, and other applicable gates.**

This slice implements the immutable-revision and current-selection subset of Architecture
V1.1 Sections 25.4, 25.5, and 25.11. ADR 0005, three exact version 1.0 wire contracts,
and schema version 2 of the local logical-node SQLite adapter define the boundary. Every
revision and decision is owned by an existing run-independent logical node; current
selection is a replaceable projection of verified append-only authority.

### Deliverables

- Closed `ImmutableNodeRevision`, `SelectionDecision`, and `CurrentSelection` contracts
  with exact schema-catalog pins; the catalog contains 16 documents and no upcaster.
- Semantic revision and decision keys that exclude UUIDs, timestamps, execution identity,
  mutable current state, database order, and their own digest/key outputs.
- Immutable publication with exact semantic retry, supersedes ID/logical-key parity,
  same-subject ownership, and publication-time `ELIGIBLE`/`INELIGIBLE` preservation.
- One linear selection-decision chain per subject, with one genesis, at most one successor,
  contiguous sequence, verified predecessor identity, and no deselection operation.
- Atomic decision append plus compare-and-swap of the exact Architecture current-selection
  association; stale expectations and ineligible revisions fail without partial state.
- Deterministic current-projection rebuild from verified chain tails without timestamp,
  UUID, `rowid`, insertion, or query-order selection.
- Transactional SQLite v1-to-v2 migration, enforced composite foreign keys, immutable
  history triggers, canonical-record/normalized-column checks, and uncertain-commit
  recovery.

### Acceptance

- Revision mutation and logical-key collisions fail closed; exact semantic retry returns
  the first immutable record.
- Run/work/current/timestamp/UUID changes do not alter semantic identity, while every
  included projection change does.
- Missing, foreign-subject, self-superseding, mismatched predecessor, and `INELIGIBLE`
  inputs leave no published selection side effect.
- Exact retry, concurrent genesis/successor attempts, and stale compare-and-swap preserve
  one contiguous, fork-free decision chain.
- Injected transaction failure exposes neither a partial decision nor a projection change;
  uncertain commit resolves through verified semantic identity.
- Rebuild after projection deletion or corruption produces the exact verified tail for
  every subject and fails closed on malformed history.
- Foreign-key, canonical JSON, record-digest, normalized-column, chain, trigger, index,
  and database-schema tampering fail closed within the local threat model.

Implementation scope, exact contract pins, threat limits, final local automated
verification, and remaining gates are recorded in
`archive/old_mvp/reports/local-immutable-revision-selection-2026-07-18.md`. This slice does not admit a
concrete producer, define business truth or authorization, implement invalidation/work
propagation, provide cross-registry atomicity, qualify production storage, or declare
Phase 0, Phase 1A, or Phase 1B complete.

## Current Local MP4 Derived-Artifact Slice: Phase 1A/1B Boundary

**Status: local executable V2 artifact-registry slice implemented and its local media
subcriteria are evidenced. Overall Section 10 acceptance and promotion remain blocked by
upstream gates.**

This slice adds the fixed `MCAP -> 6 MP4` product goal without making MP4 a source
identity or admission artifact. Its registered contracts and deterministic fixtures are
Phase 1A work. Exercising an unapproved local MCAP is an isolated, non-promotional
development probe at the Phase 1B boundary.

Historical V2 registry/lineage/replay evidence is retained in
`archive/old_mvp/reports/local-artifact-registry-v2-2026-07-18.md`. The earlier frozen-V1
media exercise is retained in
`archive/old_mvp/reports/local-six-camera-video-export-2026-07-18.md`.

### Deliverables

- Registered immutable `CameraVideoExportManifest` wire contract containing exactly six
  `CameraVideoExportRecord` values in canonical order `cam_01` through `cam_06`, with
  semantic validators, registry entry, compatibility policy, and golden fixtures.
- Manifest-level `execution_mode`, recording identity, source SHA-256/size, mapping
  profile, READY/alignment references and status, exporter identity, and ordered camera
  records. Partial or failed output cannot publish a complete manifest.
- Per-camera source topic/channel/`schema_name`/codec, observed timestamp extrema, exported
  packet/frame/keyframe counts, dimensions, content-addressed video/sidecar artifacts,
  and exact `MediaTimeMapping` with integer timebase, first/last PTS, `last_duration`,
  tail policy, HALF_EVEN rounding, and maximum mapping error.
- Structured `leading_drops` and `trailing_drops` with count, stable reason code, and
  source-time range. Input count equals both drop counts plus exported packet count, and
  timestamp-sidecar row count equals exported packet count.
- Per-camera immutable timestamp sidecars mapping source access-unit/message provenance
  and canonical int64 recording nanoseconds to MP4 packet index, PTS, DTS, duration, and
  integer track timebase.
- Registered `CameraVideoTimestampRow` wire schema plus row conformance, immutability,
  canonical-serialization, and manifest-row-count tests.
- A `REMUX` path for compatible encoded H.264 access units. Executable contracts use
  `EXTRACT`, `REMUX`, `TRANSCODE`, and `FRAME_DECODE` precisely; they do not call MP4
  container creation decode.
- Atomic publication, cleanup on failure, content verification, independent media probe,
  and deterministic rerun evidence.
- Atomic six-camera application orchestration and
  `scripts/export_camera_videos.py`, with authorization before source access, post-export
  source rehash, complete staging-directory publication, and verified idempotent reuse.
- An explicit local-development mode for the observed unapproved mapping:
  `execution_mode = LOCAL_DEVELOPMENT_OVERRIDE`, `ready_manifest_id = null`,
  `mapping_profile.approved = false`, and `alignment_status = UNVERIFIED`.
- Closed-mode negative tests: `GOVERNED_READY` requires a READY manifest ID and approved
  mapping; `alignment_status = VALID` requires an alignment ID. Governed READY remains
  separate from alignment and primary-admission predicates.
- V2 external registry entries supply artifact IDs, lifecycle policy identity, immutable
  locator/object versions, exact payload-schema pins, and explicit typed parent artifact
  IDs without changing the frozen V1 wire.
- Zero provider traffic. This slice creates no Qwen/GPT plan or call, publishes no
  `MCAPReadyManifest`, mutates no source/alignment ledger, and satisfies no Phase 1B exit
  gate.

### Acceptance

- ADR 0002 defines the derived-artifact boundary; the executable criteria in
  `docs/architecture/execution-spec-v1-overlay.md` Section 10 are the acceptance
  authority.
- The opt-in readable local case must produce six independently probeable artifacts and
  a semantic-valid ordered `CameraVideoExportManifest` with exact provenance, mode
  invariants, and timestamp/count conservation.
- The known corrupt case and injected write/probe failures must publish no complete
  manifest and must preserve stable failure evidence.
- Identical input, exporter implementation/profile versions, and canonical config digest
  must reproduce the same six content digests, timestamp-sidecar digests, and semantic
  manifest digest.
- Registered wire validation, semantic negative fixtures, Ruff, formatting, strict
  mypy, and the full test suite pass for the implemented local subset.
- The V2 path must resolve and verify its complete typed artifact DAG and every exact blob
  before reuse; a V1 manifest alone remains insufficient for promotion.
- Passing this local acceptance remains development evidence only. Phase 0, final generic-
  slice verification and concrete producer/business-policy work in Phase 1A, governed
  representative data, O-03, O-04, and Phase 1B replay remain separate prerequisites;
  this plan does not claim any of them complete.

## Historical Isolated Fake-Model Mainline: Development Evidence

**Status: retained as a legacy smoke path; not the current unified execution chain and not a
phase-gate result.**

The `application/mainline.py` path composes the registered V2 video export with legacy
provider-neutral package/request contracts and a deterministic fake adapter. It does not yet
consume the canonical materialized `TemporalPackageSet`, `InferenceInputPlan`, ingestion/READY
publication ledger, or current benchmark evidence context. No network client, credential, Qwen/GPT
call, or model-quality evaluation is part of this isolated path.

### Historical delivered path

- One command composes mapping authorization, official MCAP inspection, registered six-
  camera H.264 remux, exact 13-file view publication, strict sidecar/manifest/PTS checks,
  selected-frame PNG materialization, and atomic output publication.
- `TemporalVisualPackage` carries six camera packages, frame IDs, content digests, source
  video URIs, and half-open nanosecond windows. The model port receives the package and its
  materialized PNG root, so a real adapter has no hidden filesystem or provider boundary.
- The fake adapter implements `QA_COARSE`, `EVENT_PROPOSAL`, `QA_DENSE`,
  `ACTION_EVIDENCE`, and `BOUNDARY_REFINEMENT` with deterministic outputs and a distinct
  no-event path. It reports zero external requests.
- The bundle preserves both coarse and dense QA aggregates, validates request/package/
  outcome identity, resolves provider frame ordinals to authoritative frame IDs, and
  persists source recording identity plus exact/semantic video-manifest hashes in the run
  report.
- Action-event fusion emits at most one development event with explicit six-camera
  provenance, boundary uncertainty, candidate, action/boundary inference IDs, and
  `production_eligible = false`. Out-of-window provider intervals are rejected before
  fusion.
- CLI publication uses a sibling staging directory with inherited ACLs and renames the
  complete `video + analysis` tree only after accounting validation. Failure leaves no
  final output or partial staging tree.
- Root-level `execution-manifest.json` inventories every published regular artifact with
  exact byte length and SHA-256; `execution-audit.ndjson` records canonical stage accounting
  without source paths, credentials, or raw frames. Both are written inside the atomic staging
  tree and are absent from failed runs.
- `scripts/preflight_local_mainline.py` performs an offline Python/dependency/mapping/source/
  output/registry/spec-hash readiness check and always reports `provider_requests = 0`.
  Historical operating context is retained in
  `archive/old_mvp/docs/operations/local-mainline-runbook.md` and
  `archive/old_mvp/docs/operations/provider-adapter-readiness.md`.

### Historical local acceptance evidence

The real local `sample-medium.mcap` run on 2026-07-19 produced `PRIMARY_COMPLETE`, one
`FINAL` `object_interaction` event, five successful fake inferences, two packages, two QA
aggregates, 13 video/sidecar/manifest files, 510 PNG frames, 10 inference records, and
zero provider requests. The exact source/manifest/bundle IDs, hashes, output counts, and
limitations are retained in
`archive/old_mvp/reports/local-full-mainline-fake-model-2026-07-19.md`. Current-tree status
and verification do not inherit that historical report's result, and the legacy path must not
be used as evidence that the current canonical slices are wired together.

This evidence does **not** close Phase 0, producer identity/admission, O-03, O-04, O-10,
real-model quality, provider policy, capacity, or Phase 1B promotion. The sample is local
development data and the mapping remains explicitly unapproved/unverified.

### Current convergence boundary

The live tree now has a canonical post-admission offline conformance spine from a resolved
`AdmittedRecordingContextV2` through root window, materialized provider-neutral package set,
strict provider-specific input plan/catalog, single-part barrier and selected attempt, raw-byte
persistence before parsing, provider claims, orchestrator enrichment, local output admission, and
recording-scoped fenced identity/outbox assignment. Exact replay and fail-closed negative paths are
covered locally with zero network calls.

This spine does not replace the legacy CLI, produce V2 admission evidence from raw MCAP, dispatch
multi-part plans, or provide production persistence. Its adapter is fixture-only and its ledgers,
barriers, raw store, registry, and outbox are in-memory. The remaining convergence work is to add
the missing durable Section 25.7 boundaries and multi-part reducer/recovery behavior, then compose
the raw-admission and downstream task paths when their governance and policy inputs are resolved.
Until then, a legacy fake run remains a smoke test and a canonical fixture run remains local
conformance evidence only.

## Phase 1B: Real MCAP and Source-Time Admission

### Entry Gate

- Phase 0 and Phase 1A have passed.
- The intended real-data use is approved by data governance.
- O-03 and O-04 are resolved as versioned source, mapping, decoder, and clock/alignment policies.

The local zip inventory contains 37 MCAP members. That count is discovery evidence only; it does not establish governance approval, representativeness, schema support, valid six-camera mapping, clock admissibility, or promotion eligibility.

### Input

- An approved immutable local MCAP source reference from a registered representative corpus.
- A versioned topic-to-camera mapping policy grounded in O-03.
- An explicit clock/alignment policy grounded in O-04.

### Deliverables

- CLI/application path that submits one approved MCAP to the admission service.
- Streaming source size and SHA-256 verification plus deterministic recording identity and raw provenance.
- MCAP summary/index inspection with a scan fallback through an MCAP reader port.
- Preservation of all raw channels separately from the six logical camera mappings.
- Versioned mapping resolution for exactly `cam_01` through `cam_06`.
- Immutable `MCAPValidationReport` with all checks/diagnostics and `VALID`, `INVALID`, or `INCONCLUSIVE` verdict; infrastructure failure is `INCONCLUSIVE`.
- Immutable `MCAPReadyManifest` publication only after a selected `VALID` report, durable source artifact, and exactly-six-camera mapping pass.
- Structured validation evidence for corrupt containers, invalid or ambiguous mappings, missing timestamps, zero duration, and unsupported schemas/codecs.
- Decoder-probe port; only implemented and successful probes may contribute to a `VALID` report selected for READY publication.
- Alignment transform and report generation under the explicit policy. Lack of clock evidence produces `UNVERIFIED`, not assumed alignment.
- Separate source-admission and alignment ledgers with independently reconcilable denominators and current outcomes.
- Primary admission predicate exposed as selected READY manifest plus selected alignment admissible for the consuming policy.
- Immutable local artifact adapter for reports, manifests, ledger evidence, and exact byte digests.
- Representative real expected/invalid cases plus approved deterministic synthetic fixtures for failure coverage.
- A Phase 1B report using the reporting fields from Architecture V1.1 Section 19.

### Acceptance

- Five-camera, seven-camera, duplicate-slot, duplicate-stream, ambiguous-topic, and missing-topic cases publish no READY manifest and enqueue no downstream primary work.
- Invalid or `INCONCLUSIVE` reports can never publish READY.
- A structurally readable source cannot become READY when its schema, codec, or decoder path is unsupported.
- Source timestamps and canonical timestamps remain exact beyond the IEEE-754 safe-integer range.
- Clock resets, non-monotonic timestamps, gaps, and missing overlap produce explicit alignment evidence and policy-derived status.
- READY does not imply aligned, and invalid alignment does not rewrite source validity.
- Reprocessing identical bytes reuses recording identity and semantic results; moving identical content changes source aliases, not content identity.
- Changed bytes, mapping policy, clock policy, or algorithm version create a distinct derivation.
- Produced reports and manifests pass authoritative wire-schema and semantic validation.
- Source and alignment ledgers reconcile independently, and the derived primary-admission predicate is tested.
- A governed real-data replay covers representative expected and invalid inputs without inferred-but-unrecorded policy.

## QA Policy Intake for Later Phase 4

`docs/architecture/qa-policy-input-v0.md` records the verified source digests, 21 issue codes, `INFO`/`WARNING`/`ERROR` guidance, 3-second/5-second rules, Step 2 short-circuit, hand visibility, completeness/authenticity, SST mismatch, and annotation principles. It is policy input, not O-10 resolution.

QA implementation and promotion must preserve six-camera aggregation: a single camera cannot automatically reject an MCAP. Task relevance requires versioned task metadata. Diversity is a same-collector, consecutive-three-or-more cross-recording decision with two source triggers: records described as almost completely identical **or** records with quantified similarity greater than 95%. The qualitative branch is not silently reduced to the quantified threshold, and its exact operational interpretation remains unresolved rather than becoming a single-MCAP or single-camera check.

## Measurement Status

The following remain `NOT_MEASURED` until an applicable phase has a governed, registered, runnable corpus and workload:

| Measurement | Status | Reason |
|---|---|---|
| Recording and camera-video hours | `NOT_MEASURED` | The 37-member local inventory is not admitted or characterized. |
| Throughput and wall time | `NOT_MEASURED` | No governed production-shaped replay exists. |
| Queue, service, and API latency percentiles | `NOT_MEASURED` | Only in-memory barriers and an offline fixture are exercised; there is no production broker or provider API. |
| CPU, memory, disk, network, and storage rates | `NOT_MEASURED` | Workload shape and actual source codecs remain unresolved until O-03. |
| Qwen/GPT requests, tokens, latency, and cost | `NOT_MEASURED` | The canonical fixture makes zero real-model requests and provides no provider latency, token, or cost evidence. |
| Quality metrics and confidence intervals | `NOT_MEASURED` | No registered ground truth or promoted O-10 policy exists. |
| Capacity against either 500-hours/day interpretation | `NOT_MEASURED` | O-01 and production-shaped measurements are unresolved. |

Tests may report deterministic case counts and local durations, but those values are non-certifying and must not be promoted to capacity or quality claims.

## Blocking Inputs and Decisions

- **Phase 0 governance approval:** Required before governed real MCAP inspection or any provider transmission.
- **Representative real MCAP corpus:** Required for Phase 1B; the local 37-member zip inventory alone is insufficient.
- **O-03:** Required source topics, schemas, camera roles, auxiliary channels, codecs, resolutions, rates, keyframe behavior, and decoder probes.
- **O-04:** Required clock relationships, sync evidence, reset behavior, alignment tolerance, drift model, and missing-frame policy.
- **O-06/O-08:** Required before real primary-model integration and promoted provider encoding,
  limits, split policy, latency, quota, cost, and data-handling claims.
- **O-10:** Required before six-camera QA aggregation or acceptance promotion; the source summary does not settle it.
- **O-12:** Required before promoting fusion, candidate identity, merge/split, boundary tolerance,
  confidence calibration, and review behavior beyond the exact-fingerprint local resolver.
- **O-14:** Database, broker, object store, and deployment selection is not required for the local contract/admission foundation, but is required before production durability and concurrency claims.
- **Section 25.7 durable surfaces:** Raw/parsed/enriched/output-decision schemas and ledgers,
  multi-part barrier/reducer recovery, transactional recording-scoped identity, and an outbox
  publisher remain required for a production path.
- **O-15:** Retention, access, encryption, data handling, residency, and audit rules are part of the Phase 0 hard gate.

## Promotion Rule

Phase 0 is complete only when its security/privacy/governance controls and non-bypass tests pass. Phase 1A is complete only when its executable contract foundation and conformance evidence pass. Phase 1B is complete only after Phase 0 and 1A, governed representative real data, O-03, O-04, separate validation/READY publication, independent source/alignment reconciliation, and real replay evidence all pass.

The presence of 37 local MCAPs, synthetic fixtures, a readable container, a successful legacy
command, or a successful canonical fixture/integration run does not waive a predecessor gate or
promote any phase.
