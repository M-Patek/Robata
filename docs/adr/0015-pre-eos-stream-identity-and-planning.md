# ADR 0015: Pre-EOS Stream Identity and Planning Authority

- Status: Accepted for local conformance
- Date: 2026-07-22
- Governing authority: Architecture V1.1 Sections 17.3, 25.2, 25.4, 25.7, and
  25.11; ADRs 0004, 0011, 0012, and 0014
- Design source:
  [Streaming Throughput Next Iteration](../architecture/streaming-throughput-next-iteration-v1.md)
  Sections 5 through 6.6

## Context

The final recording identity depends on the complete source digest, duration,
mapping, and alignment closure. A live or tailed source does not have those
facts before end of stream (EOS), but durable segment, window, inference, and
work identities must exist before EOS.

The current `WorkItemPlan@1.0` and registered
`work-message@1.0.0` require an `mcap_id`. A pre-EOS
capture is not an MCAP. Putting a capture-session ID, partial digest, path, or
run ID into that field would silently reinterpret the published contract.
Deriving identity from observed media bytes is also insufficient: two distinct
captures can legitimately contain identical black, frozen, or repeated bytes.

An open-ended source also cannot publish its complete expected-window set
before work begins. Deriving that set later from successful scheduler rows
would make failed or omitted work disappear from the completion proof. The
planner needs an append-before-child protocol, a source-derived one-way EOS
seal, and a terminal closure that is separate from the expected set.

This decision freezes those identities and ordering rules before any related
JSON Schema is published. Schema publication remains a separate atomic
registration step.

## Decision

### Authority and vocabulary

The streaming path has four distinct identity domains:

1. A source authority issues one immutable pre-EOS capture subject.
2. Media and planning components derive immutable segment and window subjects
   under that capture subject.
3. The durable scheduler creates execution-scoped stream work for those typed
   subjects.
4. At EOS, a finalizer maps the preserved incremental subjects to final
   recording-scoped subjects.

A semantic digest is the lower-case SHA-256 of an RFC 8785 canonical JSON
projection owned by its producer. Every formula below includes its named
projection or policy version. Wire schema versions, semantic projection
versions, identity-policy versions, and key namespaces are independent as
required by ADR 0011.

Paths, URIs, broker offsets, host and worker IDs, leases, attempts, processing
timestamps, random database row IDs, and materialized-view locations never
enter source, segment, window, or evidence semantic identity. They remain
provenance. An authority-issued acquisition identity is not an execution ID;
it is semantic capture lineage and is deliberately included.

### Authority-issued pre-EOS capture subject

`PreEosCaptureSubject` is issued and durably registered before
the first segment or window is admitted. Its semantic fields are:

```text
subject_type = PRE_EOS_CAPTURE
capture_authority_id
capture_authority_epoch
capture_assignment_policy_version
acquisition_id
acquisition_epoch
channel_bindings[6], ordered cam_01 through cam_06:
  camera_id
  source_channel_id
  source_channel_epoch
  channel_binding_semantic_sha256
mapping_authority:
  authority_id
  authority_epoch
  mapping_policy_version
  initial_mapping_semantic_sha256
clock_authority:
  authority_id
  authority_epoch
  clock_policy_version
  initial_clock_binding_semantic_sha256
semantic_projection_version = pre-eos-capture-subject-semantic-v1
identity_policy_version = pre-eos-capture-identity-v1
```

Its future Wire envelope also carries its own `schema_version` and
exact `SchemaRef`. Those governance fields are validated before the
semantic projection; they cannot be replaced by a caller-supplied version
label.

The six camera IDs occur exactly once and in canonical order. Source channel
bindings are distinct even when their current bytes or timestamps happen to
match. The source authority guarantees uniqueness of:

```text
(capture_authority_id, capture_authority_epoch,
 acquisition_id, acquisition_epoch)
```

within its namespace. Reuse of that tuple with different bindings, mapping, or
clock facts is an integrity conflict.

The identity is:

```text
capture_scope_digest = sha256_rfc8785({
  semantic_projection_version,
  identity_policy_version,
  capture_authority_id,
  capture_authority_epoch,
  capture_assignment_policy_version,
  acquisition_id,
  acquisition_epoch,
  ordered_channel_bindings,
  mapping_authority,
  clock_authority
})

capture_scope_key =
  "pre-eos-capture-v1:" + capture_scope_digest

capture_scope_id =
  uuid5(PRE_EOS_CAPTURE_V1_NAMESPACE, capture_scope_key)
```

The registered reference shape will carry `subject_type`,
`capture_scope_id`, `capture_scope_key`,
`capture_scope_digest`, `identity_policy_version`, and the
exact `SchemaRef`. A reference does not repeat or weaken
the authority record.

Two captures with identical complete media bytes still differ when their
authority-issued acquisition tuple differs. Black frames, frozen streams,
timestamp resets, filesystem paths, or a repeated upload cannot collapse
distinct captures into one identity.

For local conformance, a local capture-authority adapter may issue and persist
the acquisition tuple. Replay must use its persisted authority receipt. A
worker cannot mint a replacement from a path, run ID, current time, or random
execution session.

### Reconnect and restart rules

A worker restart, lease retry, broker redelivery, or process relocation keeps
the same capture subject.

A transport reconnect keeps the same subject only when the capture authority
proves all of the following:

- the authority and acquisition tuple is unchanged;
- all six channel bindings remain unchanged;
- resumed packet or source sequence is monotonic, or an explicit immutable gap
  fact closes the discontinuity;
- timestamp continuity is proven under the bound clock authority, or an
  explicit clock-discontinuity fact is appended; and
- no mapping or clock authority field in the capture subject is replaced.

Replacing a source channel, resetting an acquisition without a provable
continuation, changing the six-slot mapping authority, or changing the clock
authority requires a new `acquisition_epoch` and therefore a new
capture subject. Similar-looking bytes, paths, topics, or timestamps cannot be
used to join the old and new subjects.

Connection observations and gap facts are append-only descendants. They do not
mutate the capture subject. An attempted reconnect that conflicts with an
existing authority receipt fails closed before segment or work publication.

### Segment and incremental-window identities

`StreamSegmentManifest` and `IncrementalWindow` are
non-final subjects. Their initial formula namespaces are:

| Subject | Projection version | Identity policy | Key namespace |
| --- | --- | --- | --- |
| segment | `stream-segment-semantic-v1` | `stream-segment-identity-v1` | `stream-segment-v1` |
| window | `incremental-window-semantic-v1` | `incremental-window-identity-v1` | `incremental-window-v1` |

The segment formula is:

```text
segment_digest = sha256_rfc8785({
  semantic_projection_version,
  identity_policy_version,
  capture_scope_digest,
  camera_id,
  requested_interval,
  effective_interval,
  ordered_packet_or_sequence_closure,
  exact_content_sha256,
  mapping_semantic_sha256,
  clock_or_alignment_semantic_sha256,
  segmentation_policy_version
})

segment_key = "stream-segment-v1:" + segment_digest
segment_id = uuid5(STREAM_SEGMENT_V1_NAMESPACE, segment_key)
```

The window formula is:

```text
window_digest = sha256_rfc8785({
  semantic_projection_version,
  identity_policy_version,
  capture_scope_digest,
  purpose,
  requested_interval,
  effective_interval,
  ordered_six_slot_segment_or_explicit_absence_closure,
  mapping_semantic_sha256,
  clock_or_alignment_semantic_sha256,
  parent_subject_key_or_none,
  refinement_role_or_none,
  refinement_generation,
  window_policy_version
})

window_key = "incremental-window-v1:" + window_digest
window_id = uuid5(INCREMENTAL_WINDOW_V1_NAMESPACE, window_key)
```

Logical inference below a window is also separately versioned:

```text
inference_digest = sha256_rfc8785({
  inference_projection_version: "stream-inference-semantic-v1",
  inference_identity_policy_version: "stream-inference-identity-v1",
  window_key,
  window_semantic_sha256,
  purpose,
  input_plan_semantic_sha256
})

inference_key = "stream-inference-v1:" + inference_digest
stream_inference_logical_id = uuid5(STREAM_INFERENCE_V1_NAMESPACE, inference_key)
```

The input-plan digest is the complete provider-neutral
`InferenceInputPlan` projection governed by ADR 0011. A model label,
package label, provider request ID, or batch ID cannot replace it.

This identity is independent of provider, deployment, attempt number, retry count,
and raw response bytes. A retry creates a new execution attempt but does not
create a new logical invocation. `stream_inference_logical_id` is the stable
logical invocation identity. A separate `inference_attempt_id` is allocated for
each dispatch attempt and is never substituted into the logical preimage. Runtime
adapters that reuse the existing inference ledger map the former to
`logical_invocation_id` and the latter to the persisted attempt `inference_id`;
no V1 field is relabeled on the wire.

Intervals are half-open signed-int64 nanosecond intervals. The six-slot closure
contains exactly one segment reference or one typed absence fact for each
camera in canonical order. An absent, late, black, frozen, or degraded camera
is evidence; it is not represented by omitting a slot.

Purpose, parent lineage, refinement role, and generation are identity-bearing.
Execution concurrency, batch membership, queue delay, and worker assignment
are excluded. A changed projection, canonicalization rule, or semantic policy
requires a new version and key namespace; no V1 key is reinterpreted.

### Distinct stream work and delivery contracts

The streaming scheduler will use a new `StreamWorkItemPlan` domain
contract and a separately published `stream-work-message` Wire contract.
Neither extends, relabels, or upcasts `WorkItemPlan@1.0` or
`work-message@1.0.0`. Publication occurs only after the models, independent
vectors, and atomic registration command are ready.

`StreamWorkItemPlan` contains:

```text
schema_version
work_item_id
work_logical_key
stream_run_id
source_subject: PreEosCaptureSubjectRef
stage
subject:
  subject_type
  subject_key
  subject_semantic_sha256
  capture_scope_digest
  identity_policy_version
  schema_ref
ordered_dependencies
input_semantic_sha256
config_semantic_sha256
work_projection_version = stream-work-plan-semantic-v1
work_key_policy_version = stream-work-key-v1
priority
sla_deadline_at
execution_expiry_at
max_attempts
trace_id
created_at
```

The typed `source_subject` replaces `mcap_id`; there is no
compatibility alias or sentinel MCAP UUID. A subject validator cross-binds the
source capture digest, subject key, subject digest, purpose, and parent lineage.
It also requires `subject.capture_scope_digest` to equal
`source_subject.capture_scope_digest`; a cross-capture subject fails closed even
when its local semantic digest matches.

`created_at` is assigned by the authority in the first successful transaction
that creates the stream work row. It is excluded from `work_digest`, callers do
not supply a replacement timestamp, and exact replay returns the persisted value.

Scheduling identity is execution scoped:

```text
work_digest = sha256_rfc8785({
  work_projection_version,
  work_key_policy_version,
  stream_run_id,
  capture_scope_digest,
  stage,
  typed_subject_key,
  typed_subject_semantic_sha256,
  ordered_dependency_projections,
  input_semantic_sha256,
  config_semantic_sha256
})

work_logical_key = "stream-work-v1:" + work_digest
work_item_id = uuid5(STREAM_WORK_V1_NAMESPACE, work_logical_key)
```

Each dependency projection contains at least the upstream work logical key and
its `criticality`, in canonical order. Criticality is semantic scheduling input
and cannot be omitted from the work identity preimage: changing a dependency
from critical to advisory (or the reverse) creates a new work identity.
The wire field `ordered_dependencies` is serialized into this canonical
`ordered_dependency_projections` value; no implementation may hash only a list
of opaque dependency IDs.

`stream_run_id` is included because this key owns one durable
execution DAG. It does not enter the reusable capture, segment, window,
inference, or evidence identity. Exact evidence reuse across runs resolves
those semantic identities and records run membership separately.

`stream-work-message` is produced only from an active authoritative
stream lease. It carries the exact plan subject, ordered dependencies, lease
epoch, fencing token, attempt, cancellation state, and exact Schema pin.
Broker redelivery cannot create work or advance state. The scheduler remains
authority.

### Expected-window plan append-before-child protocol

Each capture has one `ExpectedWindowPlan` under a complete
planning-policy binding:

```text
plan_projection_version = expected-window-plan-semantic-v1
plan_identity_policy_version = expected-window-plan-identity-v1
plan_key_namespace = expected-window-plan-v1
capture_scope_digest
segmentation_policy_version and semantic digest
window_policy_version and semantic digest
watermark_policy_version and semantic digest
lateness_policy_version and semantic digest
idle_source_policy_version and semantic digest
planner_version
state = OPEN | SEALED
```

```text
plan_digest = sha256_rfc8785({
  plan_projection_version,
  plan_identity_policy_version,
  capture_scope_digest,
  segmentation_policy_binding,
  window_policy_binding,
  watermark_policy_binding,
  lateness_policy_binding,
  idle_source_policy_binding,
  planner_version
})

plan_key = "expected-window-plan-v1:" + plan_digest
plan_id = uuid5(EXPECTED_WINDOW_PLAN_V1_NAMESPACE, plan_key)
```

Mutable state is excluded from that identity. Reusing the key with different
policy facts is a conflict.

While the plan is `OPEN`, the planner appends
`ExpectedWindowDeclaration` records in contiguous ordinal order. A
declaration contains:

```text
plan_key
ordinal
window_key and window_semantic_sha256
requested_interval and effective_interval
ordered six-slot segment-or-absence closure
watermark_source_facts_sha256
declaration_projection_version =
  expected-window-declaration-semantic-v1
declaration_semantic_sha256
previous_append_chain_sha256
append_chain_sha256
```

The append chain is:

```text
append_chain_0 = sha256_rfc8785({
  version: "expected-window-plan-append-v1",
  plan_key,
  ordinal: 0,
  declaration_semantic_sha256,
  previous: null
})

append_chain_n = sha256_rfc8785({
  version: "expected-window-plan-append-v1",
  plan_key,
  ordinal: n,
  declaration_semantic_sha256,
  previous: append_chain_(n-1)
})
```

One authoritative local metadata transaction:

1. verifies the plan is `OPEN` and the next ordinal is exact;
2. inserts the immutable declaration and advances the append-chain head; and
3. inserts the exact child-delivery row whose payload is the root
   `StreamWorkItemPlan` for that declaration.

The child-delivery row has a foreign key to the declaration and a deterministic
ID:

```text
child_delivery_id = uuid5(
  EXPECTED_WINDOW_CHILD_DELIVERY_V1_NAMESPACE,
  plan_key + ":" + ordinal + ":" + declaration_semantic_sha256
    + ":" + child_work_logical_key
)
```

The delivery namespace is `expected-window-child-delivery-v1`, and
the row has a unique `(plan_key, ordinal)` constraint. A relay may
publish only committed child rows. Thus no crash point can expose child work
without its expected declaration. A crash before commit exposes neither row.
A crash after commit leaves both rows and relayable delivery. Exact replay is
a no-op; same ordinal, declaration ID, delivery ID, or logical key with
different semantic or exact bytes is a conflict.

The expected set is derived from capture timeline facts and the bound policies.
It is never reconstructed from scheduler rows, provider calls, terminal
outcomes, or available evidence.

### One-way EOS seal

`ExpectedWindowPlan` transitions only from `OPEN` to
`SEALED`. There is no unseal, reopen, truncate, or member replacement
operation.

An `ExpectedWindowPlanSeal` contains:

```text
plan_key
capture_scope_digest
eos_source_receipt_semantic_sha256
final_source_timeline_semantic_sha256
final_duration_ns
ordered_six_channel_health_closure_sha256
mapping_closure_semantic_sha256
clock_or_alignment_closure_semantic_sha256
the exact planning-policy bindings from the OPEN plan
expected_member_count
first_ordinal and last_ordinal_or_none
final_append_chain_sha256
ordered_expected_member_root_sha256
seal_projection_version = expected-window-plan-seal-semantic-v1
seal_semantic_sha256
```

The seal command may read the OPEN plan and immutable source, mapping, clock,
health, and policy facts. It must not read work state, inference state,
terminal outcomes, output decisions, or evidence availability when deriving
membership. The repository verifies contiguous ordinals and recomputes the
ordered root before atomically writing the immutable seal and changing state
to `SEALED`.

An exact repeated seal returns the existing seal. A different EOS receipt,
duration, policy binding, member count, chain head, or ordered root is a
conflict. Appends after seal fail closed.

### Separate terminal closure

`WindowTerminalClosure` is not part of the expected plan or seal. It
reconciles execution to the already sealed expected set.

Each immutable closure member binds:

```text
plan_key
expected_ordinal
window_key and window_semantic_sha256
terminal_outcome
terminal_work_item_id and terminal_work_logical_key
terminal_evidence_ref:
  artifact_id
  exact_sha256
  byte_count
  media_type
  schema_ref
terminal_policy_version
member_projection_version = window-terminal-member-semantic-v1
member_semantic_sha256
```

Every outcome, including failure, cancellation, expiry, policy skip,
abstention, no event, late or incomplete input, and invalidation, requires an
exact terminal evidence reference. A null reference cannot make a member
disappear.

This terminal evidence requirement belongs to the stream closure contract. It is
stored in additive stream closure/work tables and does not change V1
`work_items`, V1 attempts, or V1 result eligibility. V1 terminal semantics and
published bytes remain immutable.

The closure completes only when the plan is sealed and every expected ordinal
has exactly one byte-consistent terminal member. It cannot add an undeclared
window, substitute another window key, or omit a failed member. Its ordered
root and semantic digest are:

```text
terminal_member_root = sha256_rfc8785({
  version: "window-terminal-member-root-v1",
  plan_seal_semantic_sha256,
  ordered_member_semantic_sha256_values
})

terminal_closure_digest = sha256_rfc8785({
  projection_version: "window-terminal-closure-semantic-v1",
  plan_key,
  plan_seal_semantic_sha256,
  expected_member_count,
  terminal_member_root
})
```

Closure completion does not itself assert final recording success; recording
policy evaluates the explicit outcomes together with the export and
aggregation barriers.

### Immutable EOS finalization map

`RecordingFinalizationMap` links, rather than rewrites, incremental
history. It is published only after the final source digest and recording
identity are known. It contains:

```text
capture_scope_key and capture_scope_digest
final_source_subject_type
final_source_subject_id
final_source_exact_sha256
final_recording_identity
final_duration_ns
final_mapping_semantic_sha256
final_alignment_semantic_sha256
expected_plan_seal_semantic_sha256
window_terminal_closure_semantic_sha256
export_manifest_semantic_sha256
ordered_subject_mappings:
  incremental_subject_type
  incremental_subject_key
  incremental_subject_semantic_sha256
  final_subject_type
  final_subject_key
  final_subject_semantic_sha256
finalization_projection_version = recording-finalization-map-semantic-v1
finalization_policy_version = recording-finalization-policy-v1
finalization_semantic_sha256
```

The finalization digest is the RFC 8785 digest of the complete typed projection
above. Its key namespace is `recording-finalization-map-v1`.

Each incremental key and digest remains valid, queryable, and immutable.
Consumers resolve the map explicitly; repositories do not update incremental
rows to replace capture scope with final MCAP or recording scope. An
incremental subject maps at most once under a finalization-policy version.
Same capture or incremental identity with a different final target is an
integrity conflict.

The map alone is not primary completion. ADR 0012 recording completion still
requires the sealed expected set, complete terminal closure, export-manifest
barrier, recording aggregation, ordered membership proof, output decision,
and authoritative completion transaction.

### Late data and conflict behavior

The local V1 late-data action is
`late-data-reject-and-record-v1`. Watermark and allowed-lateness
policy delay declaration until the window is eligible to close. Data arriving
after its immutable declaration:

- is recorded as an exact `LateSourceFact`;
- cannot mutate its segment, window, declaration, plan chain, terminal member,
  or finalization map;
- cannot create an undeclared expected member after seal; and
- makes the affected local result explicitly incomplete or review-routed when
  required by finalization policy.

A future correction or revision path requires new registered contracts,
projection versions, and selection semantics. It cannot be introduced by
updating a V1 row in place.

Malformed authority receipts, duplicate camera slots, missing slots, channel
rebinding, non-contiguous declaration ordinals, conflicting exact replay,
out-of-policy intervals, stale planner fences, appends after seal,
outcome-derived membership, missing terminal evidence, undeclared closure
members, and conflicting EOS mappings all fail closed.

### Mixed V1 and stream worker isolation

The local SQLite scheduler advances its internal database schema additively.
It retains the current V1 work, dependency, and attempt tables without
rewriting their `mcap_id` meaning. New stream work, dependency,
attempt, expected-plan, child-delivery, seal, terminal-closure, and
finalization tables live in the same scheduler authority database and
transaction owner. Sharing one scheduler state machine does not make the Wire
contracts interchangeable.

Worker registration advertises an exact capability set:

```text
supported_work_contracts:
  plan_schema_ref: SchemaRef
  message_schema_ref: SchemaRef
  work_projection_version
  work_key_policy_version
```

Claim capability is an exact four-field pin (`schema_id`, `semantic_version`,
`artifact_id`, `exact_sha256`), not a partial schema label or digest. Both
stream contract references, the projection version, and the key-policy version
are matched against the current catalog before lease acquisition; pins are
canonicalized, unique, and stale or unknown pins fail closed.
Claim queries filter by this set before leasing. V1-only workers receive only
V1 MCAP work and `work-message@1.0.0`. Stream-capable workers receive
only supported `StreamWorkItemPlan` rows and matching
`stream-work-message` pins. Delivery topics and serializer dispatch
are contract-qualified. An unknown, missing, stale, or mismatched capability
pin fails before lease publication.

Existing V1 rows, attempts, lease epochs, fencing tokens, and messages are not
copied, relabeled, or requeued during migration. A process that understands
only the old internal SQLite version refuses the upgraded database rather than
ignoring stream tables. No automatic V1-to-stream upcaster exists because V1
does not contain a pre-EOS source subject or its authority evidence.

### Clarifications before Schema publication

The following are publication blockers, not optional implementation notes:

- **Capture scope binding.** `StreamSubjectRef` MUST carry
  `capture_scope_digest` and a validator MUST cross-bind it to the source
  subject. A matching local digest from another capture is not valid.
- **Logical versus attempt identity.** `stream_inference_logical_id` is the
  stable invocation identity; every dispatch gets a separate
  `inference_attempt_id`. Retries never alter the logical preimage, and the
  existing V1 ledger field names are not relabeled on the wire.
- **Dependency criticality.** The canonical ordered dependency projection
  includes the upstream logical key and `criticality`. A critical/advisory
  change MUST produce a new work identity.
- **Authority timestamps.** `created_at` is assigned by the first successful
  authoritative durable transaction, excluded from the work digest, and reused
  on replay.
- **Capability pins.** Stream claims use exact `SchemaRef` values for both the
  plan and message contracts (`schema_id`, semantic version, artifact ID, and
  exact SHA-256), plus the projection and key-policy versions. Partial pins are
  invalid and catalog validation occurs before leasing.
- **Identity-policy migration.** A change to any semantic projection used by a
  logical, idempotency, or fence key MUST raise both the projection/policy
  version and its key namespace. A wire schema version alone is not a migration
  policy.
- **Artifact boundary.** The published V2 `ArtifactType` contract is closed.
  Stream segment/spool, window-result, declaration/seal/closure, and
  finalization artifacts require additive Artifact Registry V3 contracts (or a
  separately governed stream artifact contract). Extending the V2 enum in place
  is forbidden; V1 readers retain their old meaning.
- **Model boundary.** Existing V1 inference/evidence models are MCAP-bound.
  Stream dispatch MUST use typed-source stream contracts or a versioned adapter;
  a stream subject may not be placed in `mcap_id`.
- **Atomic bundle publication.** All related schema documents, exact artifacts,
  catalog entries, golden pins, and `$ref` checks are staged in a temporary full
  registry using fixed LF and UTF-8 bytes, then installed in one deterministic
  bundle transaction. The transaction rejects any byte change to a published
  version, including formatting-only changes. Sequential independent registration
  is not WP1 completion evidence; if a bundle command is unavailable, publication
  remains pending.

## Local Boundary and External Gates

The accepted local boundary is:

- a local capture authority with durable receipts and injected deterministic
  vector values;
- one SQLite authority for stream scheduling, expected-plan append and seal,
  child delivery, terminal closure, and finalization metadata;
- immutable local content-addressed artifacts with exact references;
- fixture or mock model inference; and
- local at-least-once relays and nonblocking review routing.

This proves contract, identity, atomicity, fencing, replay, and recovery
behavior only. It remains `LOCAL_CONFORMANCE` and
`production_eligible=false`.

Production qualification still requires external decisions and evidence for:

- the real capture authority, acquisition uniqueness, reconnect attestation,
  and source-authentication boundary;
- governed mapping, clock, watermark, idle-source, lateness, correction, and
  finalization policies;
- production database, object storage, broker, transactional outbox or CDC,
  retention, backup, disaster recovery, and reconciliation topology;
- real model-provider capability, quality, calibration, and failure evidence;
- representative capacity, long-duration stability, monitoring, alerting,
  security, operator DLQ, and review operations; and
- governed benchmark and production promotion approval.

No local adapter or mock result may mint production evidence for those gates.

## Verification

Before related schemas are registered, implementation must provide:

1. Python and an independent Node or second reference implementation that
   reproduce golden capture, segment, window, work, append-chain, seal,
   terminal-closure, and finalization digests and IDs.
2. Mutation vectors proving every included field changes the owning identity
   and paths, worker IDs, attempts, leases, timestamps, and locators do not.
3. Two authority receipts with identical black or repeated media bytes and
   different acquisition tuples that produce different capture, segment, and
   window identities.
4. Restart and reconnect cases covering exact continuation, explicit gap,
   clock discontinuity, channel replacement, epoch rollover, and conflicting
   authority replay.
5. Injected crashes before plan append, after declaration insert, after child-
   delivery insert, after commit, before relay, and after relay. Only the
   shared committed transaction may expose a child.
6. Plan tests proving contiguous append, exact replay, immutable declarations,
   source-and-policy-only seal, append-after-seal rejection, and prohibition of
   deriving members from outcomes.
7. Closure tests proving one terminal evidence reference per sealed member,
   rejection of missing and extra members, and canonical reduction independent
   of completion order.
8. Late-data tests proving post-declaration facts cannot mutate published
   identities and cannot append a sealed member.
9. Finalization tests proving incremental facts remain byte-identical and one
   conflicting final target fails closed.
10. Migration and mixed-worker tests proving unchanged V1 behavior and bytes,
    capability-filtered claims, stale-capability rejection, stream/V1 message
    isolation, lease/fence recovery, and refusal by an old database reader.

Schema work begins only after these models and vectors agree with this
decision. Publication then uses one atomic multi-schema registration bundle,
creates new schema IDs or versions, and does not modify any published V1 schema
bytes. The bundle validates the complete temporary catalog before a single
catalog/marker replacement is committed.

## Migration and Consequences

- Existing local batch data remains V1 batch data. It is not promoted into
  stream identity by relabeling `mcap_id`.
- Pre-production stream prototypes that used paths, run IDs, partial file
  digests, or unversioned hashes are discarded and rebuilt from authority
  receipts.
- The SQLite migration is forward, additive, tested on a byte-preserving copy,
  and records its internal schema version. It does not synthesize stream rows
  for existing V1 work.
- Stable pre-EOS identity permits retry, parallel work, and recovery before
  the final source digest exists while preventing distinct repeated captures
  from colliding.
- Append-before-child publication makes omissions visible and recoverable. It
  adds one local authoritative transaction and relay obligation per expected
  window.
- Separating expected membership, terminal closure, and finalization prevents
  success outcomes from defining truth, but requires explicit failure and
  absence evidence.
- Immutable EOS mapping preserves audit history and makes final reconciliation
  queryable instead of destructively renaming incremental records.
- New Wire contracts and worker capability isolation add migration work, but
  preserve the published meaning of `WorkItemPlan@1.0` and
  `work-message@1.0.0`.
