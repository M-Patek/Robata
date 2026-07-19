# Local Immutable Revision and Current-Selection Evidence - 2026-07-18

## Status and Scope

**Evidence class: local development, non-promotional.**

The generic node-scoped immutable-revision, append-only selection-decision, and
replaceable current-selection implementation defined by ADR 0005 is present in the local
contract and SQLite adapter boundary. Final local repository verification is recorded in
the Automated Evidence section below.

This slice does not declare Phase 0, Phase 1A, or Phase 1B complete. It does not admit a
concrete revision producer, assert business eligibility, authorize a selection, publish a
READY source, propagate invalidation, schedule descendant work, or call a provider.

## Governing Inputs

| Item | Exact value |
|---|---|
| Architecture | [Architecture Design V1.1](../ARCHITECTURE_DESIGN_V1.md), Sections 25.4, 25.5, and 25.11 |
| Logical-node decision | [ADR 0004](../docs/adr/0004-run-independent-logical-nodes.md) |
| Revision/selection decision | [ADR 0005](../docs/adr/0005-immutable-revisions-and-current-selection.md) |
| Normalization overlay | [Execution Specification V1 Overlay](../docs/architecture/execution-spec-v1-overlay.md) |
| Execution specification | [Large-Scale Agent Execution Specification](../large_scale_6camera_video_agent_execution_spec.md) |
| Execution-spec bytes | 37,694 |
| Execution-spec lines | 2,587 |
| Execution-spec SHA-256 | `434902fed026726e9e4924042dd1f3f2d2ec26172011efe188aac3f7986e3c0a` |
| Provider requests / uploads / frames / tokens / cost | 0 / 0 / 0 / 0 / 0 |

## Exact Schema Authority

The current catalog contains 16 exact JSON Schema 2020-12 documents and no registered
upcaster. ADR 0005 adds these three closed first-version contracts:

| Logical schema | Version | Schema artifact ID | Exact-byte SHA-256 | Catalog projection |
|---|---:|---|---|---|
| `https://schemas.robata.dev/immutable-node-revision` | `1.0.0` | `b32fe5ed-da66-ec11-9149-c8b5ebc34442` | `2fe93c147d1290d7ca980bb9cdd0addfdb8dfff5398489cb5a23cfc1d8920dd6` | `immutable-node-revision-v1` |
| `https://schemas.robata.dev/selection-decision` | `1.0.0` | `83f8a99d-0e04-f365-e323-9a04bcb18830` | `2098d47dfddf58705e47ef9f1a149a1455e0bb5c918879587de6a10b488092b8` | `selection-decision-v1` |
| `https://schemas.robata.dev/current-selection` | `1.0.0` | `c7cb333b-1e64-a38b-3401-87229bf97457` | `71c819d7f14f3b16f6a5f50c60e15cdfb4e5cab44b716fa1a35d0cb853c1c625` | `current-selection-v1` |

The catalog pins the exact documents at
[`schemas/v1/immutable-node-revision.schema.json`](../schemas/v1/immutable-node-revision.schema.json),
[`schemas/v1/selection-decision.schema.json`](../schemas/v1/selection-decision.schema.json),
and [`schemas/v1/current-selection.schema.json`](../schemas/v1/current-selection.schema.json).
No upcast path fabricates revision identity, selection history, or current state.

Both catalog-backed and frozen-directory schema-registry entry points use the same local
RFC3339 shape/calendar checker and strict JSON-integer checker. Validation therefore does
not depend on an optional `jsonschema` format package, rejects calendar-invalid timestamps,
and rejects floating representations such as `selection_sequence = 1.0` consistently with
the strict Pydantic contracts.

## Architecture and Ownership

The local primitive preserves the Architecture V1.1 separation among:

- one run-independent logical subject;
- immutable published revisions under that subject;
- append-only facts that one eligible revision was selected; and
- a replaceable query projection of the verified decision-chain tail.

Subject identity is exactly `(subject_type, subject_id)`. These values must resolve the
same existing logical node as `(node_type, node_logical_key)`. A row UUID, run ID,
artifact ID, work ID, path, provider handle, or caller alias cannot substitute for that
identity. The implementation is in the strict contracts module and the same local schema
owner as the logical-node registry:

- [`src/robata/contracts/revisions.py`](../src/robata/contracts/revisions.py)
- [`src/robata/adapters/local_logical_node_registry.py`](../src/robata/adapters/local_logical_node_registry.py)

## Immutable Revision Contract

`ImmutableNodeRevision` carries the subject, a storage UUID, namespaced logical key,
semantic/payload/lineage digests, publication status and eligibility, revision policy,
optional supersedes identity, and audit publication time. Its semantic preimage is
exactly:

```text
{
  semantic_projection_version: "immutable-node-revision-semantic-v1",
  subject_type,
  subject_id,
  payload_sha256,
  lineage_sha256,
  status_at_publication,
  eligibility_at_publication,
  revision_policy_version,
  supersedes_revision_logical_key
}
```

`semantic_sha256` is SHA-256 over the RFC 8785 canonical projection, and
`revision_logical_key` is exactly
`<revision_key_namespace>:<semantic_sha256>`. The preimage excludes `revision_id`,
`supersedes_revision_id`, `published_at`, current-selection state, run/work/lease/attempt
identity, enqueue and attachment times, database ordering, and all digest/key outputs.

The optional supersedes UUID and logical key must be both null or both non-null, must
resolve the same earlier revision under the same subject, and cannot refer to the new
revision itself. Superseding or selecting never rewrites prior status, eligibility,
payload, lineage, bytes, or digest. An exact semantic retry resolves the first immutable
record, including its original UUID and timestamp; conflicting content fails closed.

`eligibility_at_publication` is exactly `ELIGIBLE` or `INELIGIBLE`. The generic adapter
rejects selection of `INELIGIBLE` history, but `ELIGIBLE` is only a necessary generic
gate. It is not proof of producer correctness, business truth, authorization, review, or
domain readiness.

## Selection and Current Contracts

`SelectionDecision` binds one selected revision to its subject and optional immediate
predecessor. Its semantic preimage is exactly:

```text
{
  semantic_projection_version: "selection-decision-semantic-v1",
  subject_type,
  subject_id,
  selected_revision_logical_key,
  previous_selection_decision_logical_key,
  selection_policy_version,
  projection_version
}
```

Its namespaced logical key is derived from that canonical digest. The preimage excludes
revision and decision UUIDs, `selection_sequence`, `selected_at`, mutable projection
state, execution identity, database order, and digest/key outputs. The predecessor's
logical key gives semantic chain position; the positive sequence is an independently
verified structural invariant.

`CurrentSelection` contains `schema_version` plus exactly the Architecture association:
`subject_type`, `subject_id`, `selected_revision_id`, `selection_decision_id`,
`selection_policy_version`, `projection_version`, and `selected_at`. Its unique key is
exactly `(subject_type, subject_id)`. It is a replaceable projection, never the authority
for selection history. This contract has no deselection or tombstone operation.

## SQLite Schema Version 2

The adapter owns logical nodes, run memberships, revisions, decisions, and current
projections in one `logical-nodes.sqlite3` schema. Opening a schema-version 1 database
verifies its canonical schema and health, enters `BEGIN IMMEDIATE`, creates the revision
tables, indexes, and triggers, sets `PRAGMA user_version = 2`, and commits. Any migration
failure rolls back; unknown versions and partial or drifted schemas fail closed. Existing
version 1 logical nodes and memberships remain in place.

Connections enable foreign keys and use WAL, `synchronous = FULL`, a bounded busy timeout,
and `trusted_schema = OFF`. Version 2 enforces:

- revision ownership by composite foreign key to the logical node;
- supersedes ownership by same-subject revision foreign key plus verified UUID/logical-key
  parity;
- selected revision and predecessor ownership through composite decision foreign keys;
- current projection references to both its selected revision and the full projection
  source tuple on its decision;
- globally unique revision/decision UUIDs, subject-scoped semantic logical keys, and
  subject-scoped selection sequences;
- one partial-index genesis and at most one indexed successor for each predecessor; and
- `ON UPDATE RESTRICT`/`ON DELETE RESTRICT` lineage plus triggers that reject update or
  deletion of revision and decision history.

The current projection intentionally remains replaceable so it can advance or rebuild;
the immutable rows cannot be repaired by mutation.

## Linear Decision Chain and Atomic CAS

For each subject, genesis has null predecessor fields and sequence 1. Every successor
references the immediately preceding decision by matching UUID and logical key and uses
exactly predecessor sequence plus one. One-genesis and one-successor constraints prevent
forks; verified reads also reject gaps, cycles, cross-subject references, mismatched
identity pairs, unsupported projection versions, and selected-revision corruption.

A selection command supplies the expected prior decision UUID. In one `BEGIN IMMEDIATE`
transaction the adapter verifies the subject and selected eligible revision, checks that
expectation against the verified current projection, constructs and inserts one immutable
decision, compare-and-swaps the current row, and commits before returning verified state.
Genesis requires both expected predecessor and current row to be absent. A successor
updates only the current row governed by its expected predecessor.

A stale expectation or ineligible target appends no decision and changes no projection.
An exact retry returns the original decision and does not make an older decision current
again after the chain advances. A pre-commit failure rolls back both changes. Commit-
uncertain recovery resolves the immutable logical identity and verifies both chain and
projection before reporting success.

## Deterministic Rebuild

Rebuild starts an immediate transaction, verifies schema and database health, and allows
only repairable current-projection content drift. For every subject, it starts at the one
verified genesis and follows the unique verified successor chain. The verified tail
supplies every `CurrentSelection` field. The adapter replaces all current rows atomically
and returns the projection at that commit linearization point. Commit-uncertain recovery
accepts only a separately read, fully verified projection; a legal selection that advances
after the rebuild commit is not misclassified as corruption.

Rebuild never selects by `selected_at`, UUID ordering, SQLite `rowid`, insertion order,
query order, or first-writer timing. The same valid decision set and supported projection
version therefore produce the same exact projection. Missing, orphaned, forked, cyclic,
non-contiguous, or coherently mismatched history fails closed instead of producing a
best-effort current value. In particular, a dangling current foreign key is not treated as
repairable projection drift because it can indicate deletion of an immutable revision or
decision tail.

## Integrity and Threat Limits

Verified operations check the canonical database schema, table columns and primary keys,
index definitions, trigger definitions, SQLite health, and foreign keys. Stored contracts
are reconstructed under strict validators; canonical JSON, record SHA-256, normalized
columns, semantic digest/key binding, ownership, predecessor parity, eligibility, chain
shape, and current-tail equality must all agree.

These controls detect accidental corruption and an attacker who changes only a record,
column, digest, relationship, or schema component. They are not a cryptographic trust root
against an attacker able to rewrite canonical JSON, every normalized column, all related
digests, the decision chain, and the database schema coherently. That stronger threat
requires external signed audit, trusted storage, access control, and production recovery
controls under Phase 0 and production design. Without an external audit anchor, the local
database also cannot prove that a decision ever existed if an attacker coherently removes
both an immutable tail and its current projection while restoring a canonical schema.

The SQLite transaction does not own artifact-registry blobs or external payload/lineage
stores. Cross-boundary publication still requires a shared transaction owner, outbox, or
explicit reconciliation protocol.

## Automated Evidence

Final local verification on Python 3.13.5:

- Focused revision/schema/logical-node suite: `150 passed in 8.45s` (150 collected).
- Full repository suite: `410 passed, 2 skipped in 36.69s` (412 collected).
- Schema verifier: all 16 exact pins and offline references passed.
- Ruff lint passed; Ruff format check reported all 61 files already formatted.
- Strict Mypy: all 34 source files passed.
- `uv lock --check`: the existing lock resolved 34 packages with no lockfile drift.
- Execution specification: 37,694 bytes, 2,587 LF bytes, SHA-256
  `434902fed026726e9e4924042dd1f3f2d2ec26172011efe188aac3f7986e3c0a`, unchanged.
- Provider requests / uploads / frames / tokens / cost: `0 / 0 / 0 / 0 / 0`.

The focused suite includes strict model/schema acceptance parity, exact catalog pins,
semantic projection vectors, publication retry/conflict behavior, same-subject supersedes
parity, linear selection and stale CAS, eligibility precedence, transaction failure and
commit-uncertain recovery, deterministic rebuild, forced read/rebuild concurrency
interleavings, immutable-tail deletion, canonical-record and normalized-column tampering,
foreign keys, triggers, migration, and exact DDL drift. This remains local development
evidence and is not a promotion or production-capacity claim.

## Remaining Gates

- Define typed payload and lineage projections, status vocabulary, negative vectors, and
  publication eligibility evidence for every concrete revision producer.
- Define domain-specific selection reason, authorization, subject/revision-policy
  compatibility, business prerequisites, and audit evidence before a workflow selects a
  revision.
- Define append-only invalidation decisions, dependency propagation, affected-subgraph
  replay, and creation of missing or invalid descendant work without mutating history.
- Add the authoritative processing-run/work/attempt ledger and any required fences,
  barriers, retries, outbox, and reconciliation behavior.
- Resolve cross-registry atomicity and qualify production storage, replication, backup,
  restore, audit, disaster recovery, and multi-service concurrency.
- Complete Phase 0 controls, the remaining Phase 1A contract and producer gates, governed
  Phase 1B source/time admission, open decisions, and measured quality/capacity evidence.
