# Local Logical-Node and Run-Membership Evidence - 2026-07-18

## Status and Scope

**Evidence class: local development, non-promotional.**

This report records the generic run-independent logical-node and immutable processing-run
membership primitive defined by ADR 0004. It implements the Architecture V1.1 Section
25.4 association and the applicable two-run replay evidence from Section 25.11.

It does not declare Phase 0 or Phase 1A complete, admit a source into Phase 1B, publish a
READY manifest, or call any provider. At capture, immutable revision/current-selection
was separate work; ADR 0005 subsequently implements that generic local primitive. A
concrete producer and business policy, a complete work ledger, descendant validity/
scheduling, and cross-registry atomicity remain separate work.

## Governing Inputs

| Item | Exact value |
|---|---|
| Architecture | `ARCHITECTURE_DESIGN_V1.md`, V1.1 Sections 25.4 and 25.11 |
| Decision | `docs/adr/0004-run-independent-logical-nodes.md` |
| Execution specification | `large_scale_6camera_video_agent_execution_spec.md` |
| Execution-spec bytes | 37,694 |
| Execution-spec SHA-256 | `434902fed026726e9e4924042dd1f3f2d2ec26172011efe188aac3f7986e3c0a` |
| Provider requests / uploads / frames / tokens / cost | 0 / 0 / 0 / 0 / 0 |

The execution-spec bytes and digest are unchanged by this work.

## Exact Schema Authority

At the time this evidence was captured, the checked-in catalog contained 13 exact JSON
Schema 2020-12 documents and no registered upcaster. Six pre-existing V1 documents
remained byte-frozen, five V2 documents governed artifact-aware export, and these two new
first-version documents governed the logical-node primitive. ADR 0005 subsequently added
three first-version revision/selection documents, bringing the current catalog to 16:

| Logical schema | Version | Schema artifact ID | Exact-byte SHA-256 |
|---|---:|---|---|
| `https://schemas.robata.dev/logical-node` | `1.0.0` | `b4cbbe14-96cd-5018-5111-32834707feb2` | `fa5562e212d5834d932c237b84ea1864bb1d222d348ea90c31129bec38076e40` |
| `https://schemas.robata.dev/processing-run-node-membership` | `1.0.0` | `c55ccc4d-89b6-7845-9e31-7d55e0a427db` | `8a472d73825e581bb2fea948db4f3613272c95717f0a2787d232d0fc5e03033e` |

Both wires are closed to unknown fields and carry `schema_version = "1.0"`. The catalog
pins document ID/path, exact bytes, owner, lifecycle, compatibility mode,
canonicalization/projection version, and supported software range.

## Identity Boundary

`LogicalNode` contains:

```text
schema_version, node_type, key_namespace, node_logical_key,
semantic_sha256, identity_policy_version
```

Its database identity is exactly `(node_type, node_logical_key)`. The logical key must
equal `<key_namespace>:<semantic_sha256>`. The node record has no run, work, lease,
attempt-row, enqueue, attachment time, filesystem path, locator, host, or provider-handle
field.

The common helper binds an already computed semantic digest to a namespaced node key. It
does not accept an arbitrary mapping as an identity preimage. This distinction is
deliberate: the registry can verify the node contract, key/digest binding, canonical
record, and immutable collision behavior, but cannot infer which values a producer used
to calculate an opaque digest. Each admitted producer still requires a typed semantic
projection and producer-specific negative vectors proving execution-local values are
absent while semantic changes alter identity.

`ProcessingRunNodeMembership` contains the Architecture V1.1 association plus its wire
marker:

```text
schema_version, run_id, node_type, node_logical_key, role,
disposition, first_work_item_id, attached_at
```

Its unique identity is exactly `(run_id, node_type, node_logical_key, role)`. Role remains
an open uppercase token. Disposition is closed to `CREATED`, `REUSED`, `INVALIDATED`, and
`OBSERVED`. The local contract requires UUID-shaped run/work IDs and a timezone-bearing
RFC3339 attachment timestamp.

## Atomic Attachment

The port does not allow a caller to claim `CREATED`. In one `BEGIN IMMEDIATE`
transaction, the local adapter:

1. validates the complete node and requested membership facts;
2. checks the exact membership key before deriving a new disposition;
3. creates a missing node and its `CREATED` membership together for a normal attach;
4. records `REUSED` for a normal attach to an existing verified node;
5. permits `INVALIDATED` or `OBSERVED` only on an existing verified node;
6. rejects immutable node or membership conflicts without update/replace behavior; and
7. commits, then re-reads and verifies the result.

An exact retry of the first attach returns the original `CREATED` membership, first work
item, and timestamp. A second run obtains `REUSED`. Concurrent adapters serialize at the
write boundary, so the winner is nondeterministic but the final state is one node, one
creator, and every distinct run membership.

A commit error is treated as outcome-uncertain. Recovery resolves the exact membership
key and verifies the node before returning success; absence remains a transaction
failure. Recovered insertion-attribution flags are `None` because the adapter cannot
prove whether its transaction or an exact concurrent transaction committed the rows.

## Local Storage Checks

The dedicated database is `logical-nodes.sqlite3`. It is not the artifact registry. The
adapter uses WAL, `synchronous=FULL`, a 30-second busy timeout, foreign-key enforcement,
`trusted_schema=OFF`, and schema `user_version = 1`.

The database enforces:

- node primary key `(node_type, node_logical_key)`;
- membership primary key `(run_id, node_type, node_logical_key, role)`;
- a restricted composite membership-to-node foreign key;
- a partial unique index allowing at most one `CREATED` membership per node; and
- the four-value disposition vocabulary.

Every node and membership stores canonical RFC 8785 JSON, an independent exact-record
SHA-256, and normalized query columns. Verified reads check exact canonical DDL (declared
types, checks, primary/foreign keys, `WITHOUT ROWID`, and index definitions), SQLite
quick-check and foreign-key-check, record digest, strict model parsing, canonical
round-trip bytes, normalized-column equality, membership parentage, and exactly one
creator.

These local checks detect inconsistent or accidental storage tampering. They are not a
cryptographic trust root against an attacker able to rewrite canonical JSON, normalized
columns, and their record digests coherently. That stronger threat requires external
signed audit or trusted storage controls under Phase 0/production design.

## Automated Evidence

Targeted verification on Python 3.13.5:

- Logical-node/schema-catalog contract and registry adapter tests: `60 passed in 3.32s`.
- Schema verifier: 13 pinned documents plus all offline references passed.
- Strict Mypy: 32 source files passed.
- Repository-wide Ruff lint and format checks passed across 57 formatted files.
- Full repository suite: `318 passed, 2 skipped in 30.45s` (320 collected).

Those counts preserve this report's capture. Final verification of the current 16-document
catalog and ADR 0005 implementation is tracked in the subsequent revision/selection
report and is now finalized there.

The 60 targeted tests cover closed/strict/frozen wires, exact schema pins, key/digest binding,
invalid timestamp semantics, all four dispositions, the exact four-part membership key,
two-run replay, role coexistence, exact retries, immutable conflicts, canonical query
ordering, restart persistence, pre-commit rollback, commit-uncertain recovery, concurrent
competitor attribution, two-run convergence, concurrent same-membership idempotency,
record/column/digest tamper, creator removal, orphan detection, FK deletion restriction,
canonical DDL/type drift, and damaged database schema.

## Remaining Work

- Add typed identity projections and negative vectors for every concrete node producer
  before that producer is admitted to a governed workflow.
- ADR 0005 subsequently defines and implements the generic immutable revision,
  append-only selection-decision, atomic current-selection, and deterministic rebuild
  primitive. Concrete producer revision admission, business eligibility evidence,
  authorization, selection reasons, and domain-specific policy remain open.
- Define append-only invalidation decisions and descendant validity/work scheduling;
  membership `INVALIDATED` is not a mutable global validity flag.
- Add the authoritative processing-run/work/attempt ledger and applicable foreign keys.
- Design outbox/reconciliation or a shared transaction owner before claiming atomic
  artifact publication plus run membership.
- Select and qualify production storage, replication, backup, restore, audit, and
  multi-service durability under the applicable governance gates.
