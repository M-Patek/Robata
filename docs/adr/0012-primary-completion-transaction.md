# ADR 0012: Authoritative Primary Completion Transaction

- Status: Accepted; compact completion contract implemented locally; authoritative transaction not implemented
- Date: 2026-07-20
- Governing authority: Architecture V1.1 Sections 3, 17.6, 20.4,
  25.5, 25.7, and 25.9; ADRs 0003, 0004, and 0005

## Context

An earlier local canonical slice committed related facts through independent
state owners and could publish identity/outbox rows before durable run
completion. The `canonical-offline-v2` composition removes that standalone
identity dependency and stops after a local output decision and immutable event
hypotheses. Its result requires `identity_result=None`, and it does not mutate
the separately tested identity/outbox repository.

This removes the known partial-publication path from the current local
composition, but it does not implement authoritative production completion.
Inference evidence and logical-node memberships may be durable while call
barriers, output decisions, processing-run completion, and the returned run
result remain in process. Production identity, revision, and outbox publication
still need one explicit authority rather than a new sequence of independent
commits.

Primary completion needs one metadata transaction. This does not require large
artifacts to share a database, a distributed transaction with object storage, or
a framework for general sagas.

## Decision

### One authoritative transaction

One repository transaction is the only authority for primary completion. For
an event-producing outcome, it atomically validates and records:

1. the processing-run binding and compare-and-swap from `RUNNING` to its exact
   terminal primary status;
2. an ordered run-membership proof containing the ordinal, node type, logical
   key, role, immutable membership digest, and first work-item identity for each
   retained member;
3. the terminal barrier definition, member coverage, aggregate outcome, and
   reduction reference or explicit empty result;
4. the output-decision fact, its policy identity, semantic digest, and exact
   reduction lineage;
5. the canonical ordered hypothesis references and semantic digests;
6. the prepared recording-scoped identity mutation, including new or reused
   assignments, new identities, split/merge/ambiguity relations, and the next
   generation and fence;
7. every immutable `ActionEvent` revision published by this completion and its
   append-only selection decision and current-selection projection;
8. the compact primary-completion record and exact detailed-result artifact
   reference; and
9. all primary successor outbox rows.

When the terminal outcome is `NO_EVENTS`, the empty hypothesis, identity,
relation, revision, and selection sets are explicit transaction facts rather
than omitted unknowns. A failed, incomplete, or locally abstained run may be
durably terminalized, but it is not a production primary completion and cannot
emit a primary-success publication.

The source logical-node registry may have created immutable nodes and
memberships earlier. The authoritative transaction nevertheless stores the
ordered membership proof and binds its digest to the completion record. This is
necessary because the source registry's canonical query order is not execution
lineage order. Likewise, inference attempts and intermediate artifacts may be
written before completion, but their existence alone never implies that a run
completed.

If a durable work ledger is present, the same transaction also verifies the
current work lease epoch and fencing token, terminalizes the primary publish
work item, and records its attempt/checkpoint facts. This is a production
adapter obligation, not a reason to add a distributed lock to the local
conformance adapter.

### Artifact boundary

Large or byte-bearing data is written to immutable content-addressed storage
before the authoritative transaction. This includes source media, frame and
package manifests, rendered provider inputs, raw provider responses, complete
parsed or enriched documents, full input plans and catalogs, diagnostics,
metrics payloads, and the detailed canonical run result.

The transaction stores only typed references to those objects: artifact ID,
exact byte SHA-256, byte count, media type, and the exact schema-registry
four-tuple of schema ID, semantic version, schema artifact ID, and schema
SHA-256. Semantic hashes and logical keys remain separate from exact-byte
digests. Required artifact metadata and schema pins are verified before the
completion row becomes visible.

Writing an artifact before the database transaction can leave an unreferenced
object after a crash or stale fence. Such an object is not published state and
may be garbage-collected after a safety interval. The system does not attempt
an object-store/database two-phase commit.

### Identity is prepared, then applied once

Identity resolution is split into preparation and application. Preparation
reads one recording-scoped snapshot and returns:

- the expected registry generation and fence;
- the complete proposed identity mutation, or an explicit replay-only result;
- the expected batch result and canonical assignment order; and
- all hypothesis-to-assignment bindings required by completion.

Preparation writes nothing. The primary-completion repository applies the
prepared identity mutation inside its final transaction alongside run,
revision, selection, and outbox facts. The canonical path must not call a
standalone identity `commit()` and then attempt to append completion later.

A stale generation or fence aborts the whole transaction. The caller obtains a
new snapshot, prepares again, and retries. No run terminal state, revision,
selection, or outbox row from the stale command may remain.

### Port shape

The application boundary is intentionally small:

```text
PrimaryCompletionRepository
  begin_run(context) -> DurableRunRecord
  get(run_id) -> CommittedPrimaryCompletion | None
  commit(command) -> CommittedPrimaryCompletion
```

`begin_run` inserts a `RUNNING` record or returns an exact existing binding. A
different recording, MCAP, pipeline version, configuration digest, or start
time for the same run ID is a conflict. `get` is the authoritative recovery
read. `commit` compare-and-swaps the run and applies one complete command.

`PrimaryCompletionCommand` is an application type containing the expected run
version or fence, terminal facts, ordered membership proof, barrier proof,
output fact, hypothesis references, prepared identity batch, revision and
selection writes, result-artifact reference, and successor outbox commands.
`CommittedPrimaryCompletion` returns the registered compact completion record,
the committed identity result, outbox references, and whether the operation was
an exact replay.

The compact completion record now defaults to an immutable registered V2 schema
and V2 semantic projection. The frozen V1 schema remains exact-readable but is not
accepted by the default creator or validator and has no automatic upcast. The
detailed result remains a referenced external JSON artifact and still requires
its own immutable registered schema and exact-byte verification before governed
persistence. Registering the compact record does not implement this port, the
repository transaction, or durable completion.

### Semantic-policy migration

The published V1 record checked timestamp shape but did not prove that the
calendar date and numeric timezone offset were valid or that `completed_at` was
at or after `started_at`. Its exact bytes and catalog pin remain immutable.

This is a semantic-policy change even though the primary field set is unchanged.
V2 therefore advances all three namespaces together:

- catalog schema version `2.0.0`;
- wire `schema_version="2.0"`; and
- `semantic_projection_version="primary-completion-record-semantic-v2"`.

The V2 model strictly parses both timezone-bearing RFC3339 timestamps, rejects
invalid calendar and offset values, and compares the represented instants across
offsets. New code creates and validates V2 only. V1 remains available through
explicit exact registry lookup for historical inspection; the live catalog has
no V1-to-V2 upcaster. Any future change to a hash projection, logical key,
idempotency key, fence meaning, or semantic acceptance rule must likewise
advance its policy/projection namespace instead of silently reinterpreting
published evidence.

### Local conformance adapter

The local implementation will be a new aggregate SQLite repository with one
schema owner and one `BEGIN IMMEDIATE` transaction for the final write. It will
implement the identity snapshot boundary needed by preparation and the primary
completion port needed by application. The canonical composition will inject
that same repository for both responsibilities.

It will not:

- layer a completion database after the standalone identity database;
- use SQLite `ATTACH` to simulate a cross-database atomic commit;
- extend the already large standalone event-identity or logical-node registry
  adapters with unrelated run/work orchestration;
- introduce two-phase commit, distributed locks, or a general saga framework;
  or
- claim to select the O-14 production database or recovery topology.

Reusable domain validation and transaction-level identity mutation helpers may
be extracted from the standalone adapter. Its public `commit()` cannot be
called from the aggregate repository because it owns its own connection and
transaction.

### Crash and replay semantics

- A crash before database commit leaves no primary completion, identity
  mutation, revision, selection, or primary outbox publication. Redelivery or
  resume repeats preparation and commit.
- A crash after database commit but before the caller receives the result is
  recovered with `get(run_id)`. An exact command digest returns the committed
  result with replay status and creates no duplicate outbox rows.
- Reusing a run ID with different terminal facts, lineage, result artifact, or
  command digest is a conflict. The repository never chooses the latest or
  most complete-looking value.
- A stale recording generation/fence rolls back the entire command and requires
  fresh identity preparation.
- Outbox rows without their referenced completion must be structurally
  impossible through foreign keys and the shared transaction. Detecting such a
  state is an integrity failure, not permission to synthesize a successful run.
- A committed completion with pending outbox delivery remains complete. The
  publisher delivers at least once and records delivery idempotently without
  changing the completion facts.

No replay rule promotes old identity-only outbox rows into primary completion.
They remain unbound local evidence from the standalone adapter unless a new
governed run performs the complete transaction under this decision.

## Current implementation boundary

The transaction and repository in this ADR remain a target, not implementation
evidence. The compact `PrimaryCompletionRecord` contract is implemented and
validated against the exact registered `primary-completion-record` V2 schema.
The V1 exact schema remains frozen and readable, but the default model rejects
its wire/projection namespace and no upcaster is registered.
It carries typed count/root proofs and an exact reference to a detailed-result
artifact, but it does not write either object, verify the referenced detailed
result bytes, or atomically commit any run, identity, revision, selection, or
outbox fact. The current canonical output decision, processing-run record, and
detailed run result also lack the registered durable contracts and persistence
required here.

The canonical flow stops before publishing a concrete immutable `ActionEvent`
revision and current selection. It also stops before identity assignment and
outbox publication and does not inject the standalone identity service. That
standalone SQLite adapter is component conformance evidence only and cannot
satisfy the production primary-completion predicate.

The current run binding is `canonical-offline-v2`; old
`canonical-offline-v1` records, including unfinished `RUNNING` records, fail
closed instead of resuming across the changed completion semantics.

Until the registered detailed-result contract, exact artifact verification,
identity preparation API, aggregate repository, ActionEvent revision producer,
selection policy, canonical composition, and fault-injection tests are connected,
any transaction exercise is `LOCAL_CONFORMANCE` only. It must not be reported as
implemented production completion, production eligibility, or an O-14 decision.

## Implemented contract evidence

The following live files prove only the compact contract boundary:

- `src/robata/contracts/primary_completion.py` defines the strict compact model,
  detailed-result reference, and semantic projection;
- `schemas/v1/primary-completion-record.schema.json` is the frozen historical
  wire schema, and `schemas/v2/primary-completion-record.schema.json` is the
  immutable default V2 wire schema;
- `schemas/schema-catalog.json` exact-pins both versions and their distinct
  projection namespaces;
  and
- `tests/contract/test_primary_completion_contract.py` checks exact catalog
  resolution, V1 readability without implicit migration, V2 round-trip and
  timestamp ordering, projection invariants, closed outcomes, and tamper
  rejection.

The live catalog still declares `upcasters=[]`. This evidence is not a
`PrimaryCompletionRepository`, aggregate SQLite adapter, transaction, durable
run-result artifact implementation, or canonical completion path.

## Consequences

- Run status, event publication, and successor delivery have one recoverable
  source of truth.
- Immutable stage evidence remains reusable without making partial work appear
  complete.
- Crash recovery uses exact replay and bounded fence retry rather than a
  compensating workflow.
- Large artifacts remain outside the metadata transaction without weakening
  exact-byte or schema traceability.
- Production completion remains blocked on explicit contracts and ActionEvent
  revision/selection composition rather than being inferred from local identity
  rows.
