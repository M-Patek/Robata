# ADR 0003: Artifact Registry and Schema Evolution

- Status: Accepted and implemented for the local V2 schema/registry/export slice
- Date: 2026-07-18
- Governing authority: Architecture V1.1 Sections 5.3, 8.2, 16.2, 20.2,
  25.4, 25.7, and 25.10; ADR 0002

## Context

The V1 camera-video export wire records content digests and local locators but has no
artifact-registry identity, immutable locator version, lifecycle, or explicit parent
artifact relationships. The checked-in Schema registry validates one directory by short
name but does not pin an immutable schema artifact, exact bytes, semantic version, or
compatibility/upcast path.

Adding the missing artifact fields to the existing V1 documents would change registered
wire bytes and introduce required fields without a compatible reader. A manifest also
cannot contain its own exact-byte digest without creating a circular preimage.

## Decision

### Versioned Schema authority

1. Existing `schemas/v1` documents and document IDs remain frozen. Artifact-aware wires
   use a new major version and distinct document IDs.
2. A checked-in closed catalog is the authority for every supported schema artifact. An
   entry pins logical schema ID, semantic version, wire marker, schema artifact ID,
   document ID/path, exact-byte SHA-256, owner, lifecycle, compatibility mode,
   canonicalization/projection versions, and supported software range.
3. Persisted boundaries resolve and validate an exact
   `(schema_id, version, artifact_id, sha256)` reference. Short-name lookup is development
   convenience only and fails when ambiguous.
4. Compatibility is declared and tested, never inferred from a version number. V1 to V2
   camera-video export has no automatic upcaster because artifact identity and lineage
   cannot be fabricated. Stored V1 bytes remain readable through their frozen schema.
5. Upcasters, when later registered, are pure functions with exact source/target schema
   refs, code/runtime digests, golden vectors, one unambiguous path, and immutable
   provenance.

### Artifact records and identity

1. An artifact entry has an opaque `artifact_id`, artifact type, semantic SHA-256,
   exact-byte SHA-256/size/media type, producer identity, immutable URI/object version,
   ACTIVE lifecycle plus policy version, optional exact payload-schema reference,
   creation timestamp, and ordered typed parent references.
2. Semantic identity excludes artifact IDs, run/work IDs, hostnames, locators, lifecycle
   projection, creation time, and provider handles. Exact-byte identity hashes the stored
   bytes. A URI is never logical identity.
3. The local adapter may allocate stable opaque UUIDs from its private registry namespace
   and semantic key so concurrent/replayed local work converges. Idempotency is still
   enforced by the separately stored semantic key; consumers must not interpret an ID.
   A production registry may use UUIDv7 allocation without changing the port contract.
4. The current immutable publication state is ACTIVE. Retirement, quarantine, deletion,
   retention, and legal-hold changes require append-only lifecycle decisions and a
   deterministic projection after Phase 0 policy is approved; artifact entries are not
   updated in place.
5. Parents must already exist in the committed bundle or transaction, be unique,
   canonically ordered, non-self-referential, and acyclic. Referential deletion is
   restricted.

### Camera-video V2 publication

1. Raw MCAP bytes, mapping-profile bytes, export-config bytes, six MP4s, six timestamp
   maps, and the manifest are registered artifacts. MP4 and timestamp-map parents identify
   their source, mapping, and export configuration. The manifest artifact parents identify
   all inputs and all twelve camera artifacts.
2. The manifest body carries its exact schema reference, a semantic-content digest,
   input artifact references, and registered camera-artifact references. Its semantic
   projection excludes opaque IDs and locator/publication metadata.
3. The manifest's own `artifact_id`, locator, object version, and exact-byte digest exist
   only in its external registry entry. The exact digest is computed after canonical
   manifest bytes exist, so no field hashes itself.
4. V1 remains a frozen reader contract. V2 is a new derivation, not an in-place V1
   migration or an upcast that invents provenance.

### Local transaction and materialized views

1. The local adapter stores immutable blobs by SHA-256 and publishes metadata, typed
   edges, and the logical derivation in one SQLite transaction. Blob publication and
   verification happen before the registry commit; the database commit is the authority
   for derivation visibility.
2. A failed pre-commit attempt exposes no derivation. Orphan blobs and transaction
   staging are recoverable cache/garbage-collection inputs, not published artifacts.
3. User output directories are materialized views reconstructed from committed registry
   blobs. They are not identity or publication authority. A view failure does not roll
   back an already committed derivation; retry resolves the logical key and rematerializes
   it.
4. Reuse begins with logical-key lookup, exact registry/DAG validation, blob rehashing,
   and current-input comparison. Only then may an existing view be reused. Coherent
   modification of a view and its manifest cannot replace registry authority.
5. Commit-uncertain recovery queries the logical key/transaction state. It never blindly
   creates a competing derivation.

## Failure semantics

Stable failures distinguish invalid records, byte mismatches, artifact-ID conflicts,
locator-version conflicts, semantic nondeterminism, missing/invalid parents, lineage
cycles, schema-reference mismatches, registry commit failure, and materialized-view
failure. Cleanup errors do not replace the primary failure.

## Consequences

- The V1 local export evidence remains valid for its exact frozen wire but is not
  artifact-registry promotion evidence.
- The V2 local slice can prove immutable registry identity, exact bytes, replay,
  provenance traversal, and coherent-tamper detection without claiming Phase 0, Phase 1B,
  governed READY, approved media policy, or production durability.
- SQLite and the local blob layout are adapter choices, not domain contracts or the
  production database/object-store decision.
- At this work item's completion, Phase 1A still required the separate
  logical-node/run-membership and immutable revision/current-selection primitives. ADR
  0004 subsequently implements the former, and ADR 0005 subsequently implements the
  latter as a generic local primitive. Concrete producer admission, business eligibility
  and selection policy, other Phase 1A gates, and phase completion remain open.

## Implementation evidence

The exact schema-catalog closure, 20-entry derivation snapshot, registry database
accounting, two registry-backed 13-file views, real-media replay timings, and automated
verification are recorded in
`reports/local-artifact-registry-v2-2026-07-18.md`. This evidence is local and
non-promotional; it does not change the remaining Phase 0, Phase 1A, or Phase 1B gates.
