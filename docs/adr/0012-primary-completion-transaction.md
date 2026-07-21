# ADR 0012: Authoritative Primary Completion Transaction

- Status: Accepted; local-conformance aggregate and operator composition implemented; production authority not implemented
- Date: 2026-07-20
- Governing authority: Architecture V1.1 Sections 3, 17.6, 20.4,
  25.5, 25.7, and 25.9; ADRs 0003, 0004, and 0005

## Context

An earlier local canonical slice committed related facts through independent
state owners and could publish identity/outbox rows before durable run
completion. The `canonical-offline-v2` composition removed that standalone
identity dependency and stops after a local output decision and immutable event
hypotheses. Its result requires `identity_result=None`, and it does not mutate
the separately tested identity/outbox repository.

That removed the known partial-publication path from the then-current local
composition, but it does not implement authoritative production completion.
Inference evidence and logical-node memberships may be durable while call
barriers and intermediate output decisions remain in process. Production
identity, revision, and outbox publication still need one explicit authority
rather than a new sequence of independent commits. A local command builder,
aggregate SQLite repository, and fixture/raw-MCAP operator composition now
exercise that authority shape. The production composition and infrastructure
authority remain deliberately undecided.

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

The local conformance adapter deliberately embeds the exact detailed-result
JSON bytes in the same SQLite database as the metadata transaction. This proves
the binding and recovery semantics without selecting an object store. It is not
the production artifact boundary, an O-14 storage decision, or evidence for
cross-store recovery.

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

The compact completion record now defaults to immutable registered V3 wire and
semantic-projection namespaces. Frozen V1/V2 schemas remain exact-readable but are not
accepted by the default creator or validator and have no automatic upcast. Detailed
completion V4 is independently registered and exact-pinned; production persistence still
requires the external artifact boundary described above. Registering either record does
not by itself implement this port, repository transaction, or durable completion.

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
offsets. V1 remains available through explicit exact registry lookup for historical
inspection; the live catalog has no upcasters.

Terminal authority later changed again: valid `NO_EVENTS` outcomes can occur at event
proposal or provisional fusion before any final-fusion barrier or output decision exists.
Compact V3 therefore adds `terminal_stage` and
`terminal_evidence_semantic_sha256`, makes final-fusion evidence explicitly nullable only
for pre-final outcomes, and advances catalog, wire, and semantic-projection namespaces to
`3.0.0`, `3.0`, and `primary-completion-record-semantic-v3`. Detailed completion V4 retains
the complete stage chain for those early outcomes and successful final fusion. Older exact
bytes remain immutable and no implicit upcast is registered. Any future change to a hash
projection, logical key, idempotency key, fence meaning, or semantic acceptance rule must
likewise advance its policy/projection namespace instead of silently reinterpreting
published evidence.

### Local conformance adapter

The local implementation is an aggregate SQLite repository with one schema
owner and one `BEGIN IMMEDIATE` transaction for the final write. It implements
the identity snapshot boundary needed by preparation and the primary-completion
port needed by application. The canonical local composition injects that same
repository for both responsibilities.

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

The compact `PrimaryCompletionRecord` V1/V2/V3 contracts and the
`canonical-primary-completion-detail` V1/V2/V3/V4 contracts are registered and exact
pinned. Current detailed commands emit V4 with coarse/dense QA, proposal, candidate,
action-evidence, provisional-fusion, boundary-refinement, final-fusion, and publication
facts. Compact V3 binds the exact terminal stage and its semantic evidence, so early
`NO_EVENTS` requires no fabricated final-fusion fields. Older records remain exact-readable,
but the default models reject old wire/projection namespaces and no upcaster is registered.
A schema-validated command builder now computes versioned ordered collection
roots, exact detailed-result bytes, the deterministic artifact ID, and the
compact completion record without performing writes.

`SQLitePrimaryCompletionRepository` is the local-conformance aggregate. In one
`BEGIN IMMEDIATE` transaction it compare-and-swaps the processing run,
applies the prepared identity mutation, records deterministic genesis
`ActionEvent` revision/selection/current facts, embeds the exact detailed
result, writes the compact completion, and appends pending identity outbox
rows. `get(run_id)` is the recovery read. Exact command replay creates no
duplicate business result or outbox row; a stale identity fence rolls back the
entire transaction. Focused tests also simulate a commit that succeeds before
the caller loses the response, a failure after aggregate facts are staged,
cross-run replay-only reuse, and multi-event outbox recovery order.

The canonical runner itself still stops before identity assignment, revision
publication, and outbox publication, preserving its stage boundary. The
`local_composition.py` application service now drives either a source fixture or an
explicitly authorized raw MCAP, runner, identity preparation, revision
preparation, and aggregate commit as one invocation.
`scripts/run_canonical_fixture.py` and `scripts/run_canonical_mcap.py` are the
matching operator commands; the same source binding, state directory, and run
key recover or exactly replay the committed result.

`EventIdentityRegistryService.prepare_batch()` implements the explicit
repository-side-effect-free preparation boundary. It consumes a supplied
recording snapshot and returns canonical assignment bindings plus either a
snapshot-bound mutation or an explicit replay-only preparation. The aggregate
repository applies that preparation; the runner does not yet call the complete
sequence.

`prepare_initial_action_event_publications()` is the matching side-effect-free
local producer. For CREATED/AMBIGUOUS assignments and their exact replay it
derives deterministic stable-event subjects, internal citation-aware payloads,
immutable genesis revisions, selection decisions, current projections, and
identity current-revision references. Every payload is
`evidence_class=LOCAL_CONFORMANCE` and `production_eligible=false`; six camera
slots report only `CITED` or neutral `NOT_CITED`. A REUSED assignment fails
closed until predecessor revision/selection facts are supplied for a governed
successor policy. Generic revision `ELIGIBLE` means locally selectable only,
not production-qualified. The producer performs no repository read or write.

The current run binding is `canonical-offline-v5` with local composition v13; older
processing-run and composition namespaces, including unfinished `RUNNING` records, fail
closed instead of resuming across changed inference, final-fusion, or terminal-publication
semantics.

The aggregate supports only `SUCCEEDED` and explicit `NO_EVENTS` primary
completion. ABSTAINED/failed terminalization, durable work and barrier ledgers,
REUSED successor publication, a governed production ActionEvent contract,
external artifact storage, outbox delivery, and O-14 recovery topology remain
outside this local adapter. Every detailed result and ActionEvent payload is
`LOCAL_CONFORMANCE` with `production_eligible=false`; the transaction must
not be reported as production completion, production eligibility, or an O-14
decision.

## Implemented contract evidence

The following live files prove the local contract and aggregate boundary:

- `src/robata/contracts/primary_completion.py` defines the strict compact model,
  detailed-result reference, and semantic projection;
- `schemas/v1/primary-completion-record.schema.json` is the frozen historical
  wire schema; V2 remains immutable historical evidence, and
  `schemas/v3/primary-completion-record.schema.json` is the immutable default V3 wire schema;
- `schemas/schema-catalog.json` exact-pins every version and its distinct projection
  namespace;
- `schemas/v1` through `schemas/v4/canonical-primary-completion-detail.schema.json`
  preserve the detailed-result evolution, with V4 as the current exact-pinned contract;
- `src/robata/application/canonical/primary_completion.py` defines the command
  builder, port, deterministic roots, and recovery models;
- `src/robata/adapters/sqlite_primary_completion.py` implements the single
  local SQLite transaction;
- `src/robata/application/canonical/mcap_source.py` derives the real raw-MCAP,
  exact-schema, decoder-probe, registered-media, frame-index, and local V2
  admission facts;
- `src/robata/application/canonical/local_composition.py`,
  `scripts/run_canonical_fixture.py`, and `scripts/run_canonical_mcap.py`
  provide the fixture/raw one-command execution and recovery paths; and
- `tests/contract/test_primary_completion_contract.py` checks exact catalog
  resolution, V1 readability without implicit migration, V3 round-trip and terminal-stage
  semantics, timestamp ordering, projection invariants, closed outcomes, and tamper
  rejection, while `tests/integration/test_sqlite_primary_completion.py`
  checks normal commit/reopen/replay, strict command admission, compact/detail
  agreement, lost-response recovery, staged-write rollback, stale-fence
  rollback, replay-only reuse, and multi-event outbox order.
  `tests/integration/test_canonical_local_command.py` verifies fixture first
  execution, exact same-run replay, and cross-run reuse without provider
  redispatch or duplicate outbox.
  `tests/integration/test_canonical_mcap_source.py` verifies raw-MCAP source
  derivation, first commit, exact replay, and corrupt-input rejection.

The live catalog still declares `upcasters=[]`. This evidence is not a
production repository, external artifact store, outbox publisher, durable work
system, or phase-promotion claim.

## Consequences

- Run status, event publication, and successor delivery have one recoverable
  source of truth.
- Immutable stage evidence remains reusable without making partial work appear
  complete.
- Crash recovery uses exact replay and bounded fence retry rather than a
  compensating workflow.
- The production design keeps large artifacts outside the metadata
  transaction; the local adapter embeds detailed-result bytes only to exercise
  exact binding and recovery.
- Production completion remains blocked on governed contracts, infrastructure,
  delivery, and production qualification rather than being inferred from this
  local transaction.
