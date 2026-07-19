# ADR 0005: Immutable Node Revisions and Current Selection

- Status: Accepted for implementation of the local Phase 1A generic primitive
- Date: 2026-07-18
- Governing authority: Architecture V1.1 Sections 25.4, 25.5, and 25.11; ADRs 0001 and 0004

## Context

Architecture V1.1 makes published revisions immutable and separates them from the fact
that one revision is current. A correction, review, recalibration, policy change, or
replay must not rewrite published content. It creates a new revision or an append-only
selection decision. The current value is a replaceable query projection rebuilt from
those decisions.

Earlier ActionEvent examples contain `is_current`, a mutable current pointer, and a
`SUPERSEDED` status. Section 25.5 supersedes those representations: `is_current` is a
derived view, the stable node is not updated, and selection does not mutate a revision's
status-at-publication.

ADR 0004 established run-independent logical nodes and processing-run membership. This
decision adds a node-scoped revision and selection primitive to the same local SQLite
schema owner so ownership, predecessor, and current-selection relationships can use
enforced foreign keys. It does not admit a concrete revision producer or define a
business selection policy.

## Decision

### Node-scoped ownership

Every revision and selection belongs to one existing logical node. The subject identity
is exactly `(subject_type, subject_id)`, where `subject_type` is the logical node's
`node_type` and `subject_id` is its `node_logical_key`. The local database enforces a
composite foreign key from that pair to the logical-node table.

The generic layer accepts no alternate row UUID, run ID, artifact ID, or caller-defined
alias as subject identity. It verifies the logical node and every referenced revision or
decision before publication or selection.

### Immutable revision contract

`ImmutableNodeRevision` is a closed version `1.0` wire contract with exactly these
fields:

```text
schema_version
revision_id
subject_type
subject_id
revision_key_namespace
revision_logical_key
semantic_sha256
payload_sha256
lineage_sha256
status_at_publication
eligibility_at_publication
revision_policy_version
supersedes_revision_id
supersedes_revision_logical_key
published_at
```

The field rules are:

- `revision_id` is a globally unique, lower-case UUID-shaped storage identifier. It is
  immutable but is not semantic identity.
- `revision_key_namespace` identifies the producer and semantic-projection policy.
  `revision_logical_key` is exactly
  `<revision_key_namespace>:<semantic_sha256>`.
- `payload_sha256` identifies the producer's immutable typed payload projection.
  `lineage_sha256` identifies its immutable canonical lineage projection.
- `status_at_publication` is an open upper-case token. The generic layer preserves it
  exactly and assigns no business transition semantics to it.
- `eligibility_at_publication` is exactly `ELIGIBLE` or `INELIGIBLE`. An `INELIGIBLE`
  revision remains valid immutable history but cannot be selected.
- `revision_policy_version` identifies the policy that defines the revision projection
  and publication-time status and eligibility semantics.
- `supersedes_revision_id` and `supersedes_revision_logical_key` are either both null or
  both non-null. When present, they resolve the same existing revision under the same
  subject. A revision cannot supersede itself. Selection, not the supersedes link,
  determines which revision is current.
- `published_at` is a timezone-bearing RFC 3339 audit timestamp. It is preserved from
  first publication and never participates in semantic identity.

Published revision records are immutable. The same subject and revision logical key can
identify only one canonical record. An exact semantic retry returns the first published
record, including its original UUID and publication time. A collision or any attempt to
change payload, lineage, publication status, eligibility, policy, or supersedes facts
fails closed.

### Revision semantic projection

The revision semantic digest is exactly:

```text
semantic_sha256 = SHA256(RFC8785({
  semantic_projection_version: "immutable-node-revision-semantic-v1",
  subject_type,
  subject_id,
  payload_sha256,
  lineage_sha256,
  status_at_publication,
  eligibility_at_publication,
  revision_policy_version,
  supersedes_revision_logical_key
}))
```

The projection uses the superseded revision's logical key, not its row UUID. It excludes
`revision_id`, `supersedes_revision_id`, `published_at`, every current-selection field,
processing-run identity, work/lease/attempt identity, enqueue or attachment time,
database row order, and all hash/key outputs. Changing an included value creates a new
revision identity. Changing execution or current-selection context cannot do so.

### Append-only selection-decision contract

`SelectionDecision` is a closed version `1.0` wire contract with exactly these fields:

```text
schema_version
selection_decision_id
selection_key_namespace
selection_decision_logical_key
semantic_sha256
subject_type
subject_id
selected_revision_id
selected_revision_logical_key
previous_selection_decision_id
previous_selection_decision_logical_key
selection_sequence
selection_policy_version
projection_version
selected_at
```

The field rules are:

- `selection_decision_id` is a globally unique, lower-case UUID-shaped storage
  identifier. It is immutable but not semantic identity.
- `selection_decision_logical_key` is exactly
  `<selection_key_namespace>:<semantic_sha256>`.
- `selected_revision_id` and `selected_revision_logical_key` must resolve the same
  verified `ELIGIBLE` revision owned by the decision's subject.
- The two `previous_selection_decision_*` fields are either both null or both non-null.
  Null identifies the one genesis decision for a subject. Non-null fields must resolve
  the same immediately preceding decision under that subject.
- `selection_sequence` is a positive integer. Genesis is sequence 1; every successor is
  exactly its predecessor's sequence plus one.
- `selection_policy_version` identifies the policy authorizing the choice.
  `projection_version` identifies the deterministic current-projection algorithm.
- `selected_at` is a timezone-bearing RFC 3339 audit timestamp. It is copied to the
  projection but never orders decisions or contributes to semantic identity.

Selection decisions are append-only. The same subject and selection-decision logical key
can identify only one canonical decision. Exact retry returns the original decision and
does not append another chain element.

### Selection-decision semantic projection

The decision semantic digest is exactly:

```text
semantic_sha256 = SHA256(RFC8785({
  semantic_projection_version: "selection-decision-semantic-v1",
  subject_type,
  subject_id,
  selected_revision_logical_key,
  previous_selection_decision_logical_key,
  selection_policy_version,
  projection_version
}))
```

The projection excludes all revision and decision UUIDs, `selection_sequence`,
`selected_at`, current-projection state, processing runs, work/lease/attempt identity,
enqueue or attachment time, database row order, and all hash/key outputs. The previous
decision logical key supplies semantic chain position; sequence is a separately verified
structural invariant.

### Linear chain and atomic selection

Each subject has at most one genesis decision and each decision has at most one
successor. The database enforces both rules in addition to decision logical-key
uniqueness and predecessor foreign keys. Forks, gaps, cycles, cross-subject predecessors,
and non-contiguous sequences fail closed.

The selection command supplies the expected prior decision identity. In one transaction,
the adapter:

1. verifies the subject node and selected revision, including stored integrity,
   ownership, and `ELIGIBLE` publication state;
2. verifies the expected prior decision against the current projection, or verifies that
   both are absent for genesis;
3. constructs and appends the immutable successor decision;
4. compare-and-swaps the current projection to that decision; and
5. commits before returning the verified decision and projection.

A stale expectation appends no decision and changes no projection. Commit-uncertain
recovery resolves the decision logical key and verifies the chain and projection before
reporting success. A retry of an earlier successfully applied decision never makes it
current again after a later decision has advanced the chain.

### Current-selection projection

`CurrentSelection` is a closed version `1.0` wire contract. Apart from the required
`schema_version` marker, its fields are exactly the Architecture V1.1 association:

```text
subject_type
subject_id
selected_revision_id
selection_decision_id
selection_policy_version
projection_version
selected_at
```

Its unique key is exactly `(subject_type, subject_id)`. It is a replaceable query
projection, not append-only authority. Its selected revision, policy, projection version,
and timestamp must equal those of the referenced tail decision. Reads return the
selection-decision ID used.

This version has no deselection or tombstone operation. A subject with no decision has no
current-selection row. Once a chain exists, a selection can advance only to an eligible
revision through another append-only decision.

### Deterministic rebuild

Rebuild starts from the subject's verified genesis and follows the unique verified
successor chain. It validates predecessor ID/logical-key parity, subject ownership,
contiguous sequence, semantic digests, selected revisions, and projection-version
support. The tail decision determines the complete current projection.

Rebuild never chooses by `selected_at`, UUID ordering, SQLite `rowid`, insertion/query
order, or first-writer timing. The same verified decision set and projection algorithm
version therefore produce the same current projection. Missing, forked, orphaned,
cyclic, non-contiguous, or tampered history fails closed rather than selecting a best
effort current value.

### Local persistence boundary

The first adapter stores logical nodes, revisions, selection decisions, and current
projections under one SQLite schema owner. This permits enforced composite foreign keys
from revisions to nodes, decisions to their selected revisions and predecessors, and
current projections to both the selected revision and tail decision. Lineage uses
`ON DELETE RESTRICT`; published history is never cascade-deleted or updated.

Schema initialization and migration are owned as one unit. A partial schema, unsupported
schema version, missing constraint/index, foreign-key violation, or canonical-record
tampering fails closed. The current projection may be transactionally replaced or
rebuilt; revision and decision rows may not be repaired by mutation.

This SQLite transaction does not own artifact-registry blobs or external payload and
lineage stores. A workflow requiring atomic publication across those boundaries still
needs an explicit shared transaction, outbox, or reconciliation design.

## Eligibility boundary

`eligibility_at_publication = ELIGIBLE` is necessary for selection but is not sufficient
to admit a concrete producer or business workflow. The producer remains responsible for
a typed payload projection, complete lineage projection, status vocabulary, semantic
validation, and policy-specific eligibility evidence. A concrete selection policy must
also prove that its subject type, revision policy, selection reason, authorization, and
any domain prerequisites are valid.

The generic adapter verifies only registered record integrity, node ownership, supported
contract/policy shape, publication eligibility, and chain/CAS invariants. It does not
infer business truth from an open status token or caller assertion.

## Scope and limitations

This decision does not implement or imply:

- a concrete ActionEvent, source-admission, inference-attempt, mapping, QA, or review
  revision producer;
- a business status transition matrix or a producer-specific selection policy;
- mutable validity, invalidation decisions, dependency propagation, affected-subgraph
  replay, or missing/invalid descendant work creation;
- processing-run/work-item membership, work ledgers, leases, attempts, barriers, fences,
  outbox publication, retries, or scheduling;
- event identity assignment, split/merge resolution, or recording-scoped generation
  serialization;
- deselection, deletion, history compaction, or rewriting a revision as `SUPERSEDED`;
- cross-registry atomicity, production database durability, replication, backup, restore,
  or multi-service transactions; or
- completion of Phase 0, all of Phase 1A, or any Phase 1B admission gate.

Work `INVALIDATED`, processing-run membership `INVALIDATED`, and future domain
invalidation facts are separate concepts. None changes revision bytes or silently clears
current selection. A future invalidation policy may append decisions and create
replacement work, but it must not mutate this history.

## Consequences

- Revision content and selection history remain independently reproducible and auditable.
- Current reads are efficient without making a mutable pointer authoritative.
- UUIDs and timestamps remain useful audit metadata without determining semantic identity
  or rebuild order.
- The predecessor logical key gives retries and downstream derivations a run-independent
  selection fact while the sequence detects structural corruption.
- Node, revision, decision, and projection foreign keys can be enforced atomically by the
  bounded local adapter.
- Producer-specific projection and eligibility work remains explicit rather than being
  hidden inside a generic registry.

## Promotion evidence

Evidence for this generic primitive must include version-pinned schemas, semantic
validators, canonical digest vectors, and automated tests proving:

1. Revision mutation of payload, lineage, publication status/eligibility, policy,
   supersedes facts, canonical bytes, or digests fails closed.
2. Run/work/current/timestamp/UUID changes do not alter revision or decision semantic
   identity, while every included projection change does.
3. Supersedes ID/logical-key parity, same-subject ownership, existence, and self-reference
   rules are enforced.
4. `INELIGIBLE`, missing, foreign-subject, or tampered revisions cannot be selected and
   leave neither a decision nor projection change.
5. Genesis, successor, sequence, one-successor, and stale-CAS constraints preserve one
   linear chain under exact retry and concurrent selection.
6. Decision append and current compare-and-swap are atomic; injected failure leaves no
   partial decision or projection, and uncertain commit is recovered by verified identity.
7. Rebuild after projection deletion or corruption produces the same exact projection
   independent of decision query order, timestamps, UUIDs, and SQLite row order.
8. Selecting or superseding never changes prior revision bytes or semantic digests, and
   every current read exposes the governing selection-decision ID.
9. Foreign-key, canonical-record, normalized-column, chain, and database-schema tampering
   fails closed.
10. No deselection, invalidation propagation, work-ledger behavior, concrete producer
    admission, or Phase 1A completion is claimed by this evidence.
