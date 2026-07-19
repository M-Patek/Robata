# ADR 0004: Run-Independent Logical Nodes and Run Memberships

- Status: Accepted for the local Phase 1A logical-node slice
- Date: 2026-07-18
- Governing authority: Architecture V1.1 Sections 25.4 and 25.11; ADR 0001

## Context

Processing runs, queue work, leases, and attempts describe execution and audit scope.
They do not describe the semantic result of a reusable derivation. If those values enter
a derivation preimage, replay under another run creates a second identity for the same
result and defeats the V1.1 reuse requirement.

The artifact registry established by ADR 0003 identifies immutable stored artifacts and
their byte lineage. It does not supply the generic, run-independent identity of a domain
node or preserve every run that created, reused, invalidated, or merely observed that
node. Logical-node identity and run membership therefore require a separate contract and
persistence boundary.

## Decision

### Logical-node contract and identity

1. `LogicalNode` is a closed version `1.0` wire contract. In addition to its
   `schema_version` marker, it contains only `node_type`, `key_namespace`,
   `node_logical_key`, `semantic_sha256`, and `identity_policy_version`.
2. Its reusable identity is exactly `(node_type, node_logical_key)`. The key has the
   form `<key_namespace>:<semantic_sha256>`, where `semantic_sha256` is the lower-case
   SHA-256 of the producer's canonical typed semantic projection. A row UUID, database
   primary key, or run identifier cannot stand in for either value.
3. `key_namespace` separates producer and projection domains. A producer changing the
   meaning, field set, or canonicalization of its identity projection must use a new
   namespace and the corresponding policy/migration decision; changing only
   `identity_policy_version` cannot create a new identity because that audit field is not
   part of the key. A producer may additionally include a policy version in its typed
   semantic projection when that version is itself semantic.
4. A second record with the same identity but different namespace, digest, policy
   version, or canonical bytes is an identity conflict and fails closed. Published node
   records are immutable.

The common logical-node layer does not accept an arbitrary mapping and calculate a key
on the caller's behalf. A generic mapping cannot prove that execution-local fields were
excluded or that all semantic fields were included. Each producer owns a typed semantic
projection and canonicalization routine, and must pass producer-specific conformance
vectors before its node type is admitted. The generic layer verifies the resulting
contract, key form, digest consistency, immutability, and collision behavior.

Producer preimages must exclude execution-local and publication-local identity,
including:

- `run_id`, `qa_run_id`, work-item identity, and enqueue time;
- lease, fencing, attempt-row, and retry identity;
- attachment, creation, or update timestamps;
- worker, process, and host identity;
- filesystem paths, object locators, materialized-view locations, and URIs;
- provider request, session, response, or other opaque handles; and
- random database or artifact row UUIDs used only as references.

This exclusion does not remove genuinely semantic inputs. For example, a selected
attempt's immutable content digest and its selection-decision logical key may be
semantic, while the selected attempt row ID is not. A model or policy version that can
change the result may likewise be semantic, while a provider request handle is not.

### Processing-run membership contract

`ProcessingRunNodeMembership` is a closed version `1.0` wire contract. Apart from its
`schema_version` marker, its fields are exactly the V1.1 association:

```text
run_id, node_type, node_logical_key, role,
disposition, first_work_item_id, attached_at
```

The membership identity and database unique key are exactly
`(run_id, node_type, node_logical_key, role)`. `role` is an open, upper-case token rather
than a business-specific enumeration; producer or workflow layers own any narrower role
vocabulary. `disposition` is exactly one of `CREATED`, `REUSED`, `INVALIDATED`, or
`OBSERVED`.

The local minimum implementation requires UUID-shaped `run_id` and
`first_work_item_id`; `first_work_item_id` is non-null. That requirement is a local
implementation decision because V1.1 does not define its nullability. It preserves the
first work item associated with the attachment and is never replaced by a later retry.
The local database does not yet enforce a foreign key to a work ledger.

Membership rows are append-only and immutable. A retry of the same attachment command
returns the original membership, including its original disposition, first work item,
and attachment time. Reusing the four-part identity with incompatible facts is a
membership conflict and fails closed; an existing row is never updated to reconcile the
conflict.

When commit outcome is uncertain, recovery may prove that the requested immutable rows
exist but cannot prove which concurrent transaction inserted them. The returned
insertion-attribution flags are then unknown rather than assigning another transaction's
write to the recovering caller.

### Attachment semantics

The port exposes one atomic node-and-membership attachment operation. `CREATED` and
`REUSED` are derived results, not caller assertions:

- A normal attachment to a missing node inserts the immutable node and a `CREATED`
  membership in one transaction.
- A normal attachment to an existing, byte-consistent node inserts a `REUSED`
  membership.
- An `INVALIDATED` attachment is allowed only for an existing, byte-consistent node. It
  records what that run encountered; it does not update, delete, or globally mark the
  node invalid.
- An `OBSERVED` attachment is allowed only for an existing, byte-consistent node. It
  records audit or input context without asserting creation or reuse.

The adapter checks for an existing membership before deriving a new normal disposition.
Consequently, an exact retry of the transaction that first created a node returns its
original `CREATED` membership instead of conflicting with a newly derived `REUSED`
result.

This slice has no import or administrative pre-seeding path. Every node is introduced by
the atomic normal attachment above, so every node has exactly one `CREATED` membership.
The database enforces at most one `CREATED` membership for a node, the node/membership
foreign key, the four-part membership uniqueness rule, and the disposition vocabulary.
A missing node cannot be manufactured through `REUSED`, `INVALIDATED`, or `OBSERVED`.

Replay under another run ID therefore resolves the same logical node, adds a distinct
`REUSED` membership, and preserves both run-node associations. Work creation remains a
workflow concern: a replay schedules work only for descendants that its separate
validity policy finds missing or invalid.

### Local persistence boundary

The first adapter uses a dedicated SQLite database behind a logical-node registry port.
It stores immutable canonical node and membership documents, uses the semantic and
membership keys as database constraints, and commits a newly introduced node with its
`CREATED` membership atomically. Transaction failure exposes neither record. Reads
revalidate canonical bytes and normalized columns so coherent identity conflicts or
storage tampering fail closed.

This database is deliberately separate from the artifact registry introduced by ADR
0003. A logical node is not an artifact-registry row, and neither registry silently owns
the other's transaction. This slice makes no claim that publishing artifacts and
attaching a logical node are atomic across the two databases. A workflow that needs both
must add an explicit outbox, recovery, or reconciliation design before promotion of that
combined publication path.

## Failure semantics

Stable registry failures distinguish malformed identity, key/digest inconsistency, node
identity conflicts, missing nodes, membership conflicts, unsupported dispositions,
persisted-integrity failures, and transaction/storage failures. Forbidden or
nonconformant producer identity projections fail producer admission/conformance before
the generic registry is called. Retrying after an uncertain commit first resolves the
four-part membership key and verifies the stored node; it does not insert a competing
node or rewrite a membership.

## Scope and limitations

This decision does not implement or imply:

- revision-bearing domain rows, correction chains, or immutable revision payloads;
- append-only selection decisions or a deterministic `current_selection` projection;
- invalidation reason records, invalidation dependency propagation, or current-validity
  projection;
- a work-ledger foreign key, queue/lease state, attempt selection, or descendant
  scheduling policy;
- a node import protocol or migration of nodes created outside this registry;
- cross-registry atomicity with artifacts, blobs, schemas, or materialized views; or
- production database durability, replication, disaster recovery, or multi-service
  transaction guarantees.

Those are separate Phase 1A or later workflow decisions. In particular,
`INVALIDATED` is an immutable run-membership disposition, not a mutable validity flag.

## Consequences

- Two executions can share one semantic node while retaining independent audit scope.
- Execution retries and new run IDs cannot perturb producer identity.
- Producer-specific typed projections carry more implementation work, but make forbidden
  inputs and semantic omissions testable instead of trusting a generic dictionary.
- An open role token keeps the generic association reusable without silently defining a
  workflow role matrix.
- Requiring a first work-item UUID gives the local slice deterministic traceability but
  may require a later compatibility decision if a governed workflow needs system-created
  memberships without work items.
- Separate SQLite registries keep the initial adapters bounded, while explicitly
  deferring the combined publication transaction.

## Promotion evidence

Evidence for the generic registry slice must include checked-in version-pinned schemas,
semantic validation, and automated tests proving items 2 through 6 below. Item 1 is an
additional mandatory admission gate for each concrete producer; this slice does not
claim that a concrete producer is admitted.

1. Producer-specific vectors reject run, work, lease, attempt-row, enqueue-time, path, locator,
   host, random-row, and provider-handle inputs while semantic input changes change the
   logical key.
2. Two run IDs attach to one node, the first as `CREATED` and the second as `REUSED`,
   with both memberships preserved and exactly one creator.
3. Exact retries are idempotent; conflicting node bytes, first-work facts, dispositions,
   or membership payloads fail closed without mutation.
4. `INVALIDATED` and `OBSERVED` reject a missing node and never mutate or delete an
   existing node.
5. Concurrent same-node attachment converges on one logical node and preserves every
   distinct run membership.
6. Injected transaction failure leaves neither a new node nor membership, and persisted
   canonical/normalized-column tampering is detected on read.

Passing this evidence promotes only the run-independent logical-node/run-membership
primitive. It does not complete Phase 1A, satisfy Phase 0, admit real data for Phase 1B,
or qualify a cross-registry production workflow.
