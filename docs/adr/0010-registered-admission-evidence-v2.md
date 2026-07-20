# ADR 0010: Registered Admission Evidence V2

- Status: Accepted and implemented for local V2 contracts and admission-context resolution
- Date: 2026-07-19
- Governing authority: Architecture V1.1 Sections 25.4, 25.6, 25.7,
  and 25.10; ADRs 0001 and 0003

## Context

The frozen V1 MCAP validation, READY, and alignment schemas preserve the first
local ingestion slice. They do not contain every fact required by Architecture
Section 25.6: the validation report omits the exact validator, individual check
outcomes, decoder probes, and source-schema support evidence. The three persisted
documents also omit their own exact schema-registry quartet and explicit semantic
bindings between source content, mapping, READY selection, and alignment.

Adding those required fields to V1 would change immutable schema bytes and would
make old stored payloads fail validation. A V1-to-V2 upcaster also cannot recover
checks, probes, durability proof, schema pins, or semantic stream identities that
were never stored.

## Decision

### New major-version wires

1. `mcap-validation-report`, `mcap-manifest`, and `alignment-manifest` each gain a
   separately registered `2.0.0` schema under `schemas/v2`. The V1 documents and
   their catalog entries remain unchanged.
2. Every V2 body carries `schema_ref` with exact `schema_id`, semantic version,
   schema artifact ID, and exact-byte SHA-256. A persisted boundary must resolve
   that quartet and validate against the pinned schema; matching only the logical
   ID or version is insufficient.
3. Compatibility is `NONE`, supported predecessors are empty, and no upcaster is
   registered. V1 remains readable as V1 but cannot be promoted to V2 evidence.

### Validation evidence

1. `MCAPValidationReportV2` binds a verified source-content digest, recording
   identity, mapping-policy candidate and mapping semantic digest, exact validator
   code/configuration identity, all named check outcomes, diagnostics, per-stream
   source-schema support evidence, decoder-probe results, probed stream facts, and
   the candidate camera mappings.
2. `FAIL`, source-schema `UNSUPPORTED`, or decoder `FAILED` evidence derives an
   `INVALID` verdict. Infrastructure or otherwise incomplete evidence uses
   `INCONCLUSIVE`; it cannot characterize source bytes as invalid. Only fully
   supported, successfully probed, canonical six-camera evidence may be `VALID`.
3. Stream association UUIDs are paired with stream semantic digests. Candidate
   mappings and probe/schema evidence must agree on both association and content.

### READY and alignment evidence

1. `MCAPReadyManifestV2` has no status field. Existence is the READY publication
   fact. It references the selected V2 validation report by ID, semantic digest,
   and exact schema ref; binds mapping/admission policy digests; carries positive
   source-durability evidence; and contains exactly six canonically ordered camera
   rows with distinct stream semantic digests.
2. `AlignmentManifestV2` references the selected READY manifest by ID, semantic
   digest, and exact schema ref. It binds source and mapping semantic digests,
   versioned algorithm/policy identities, six stream-bound rational transforms,
   explicit validation checks and diagnostics, and a derived alignment status.
3. READY publication and alignment remain separate facts. A READY body cannot
   assert alignment admission, and an invalid/unverified alignment does not alter
   the source-validation verdict.

### Semantic identity

1. Each V2 body carries and self-validates a named semantic SHA-256 computed over
   an explicit RFC 8785 projection.
2. The projections include schema ID/version/exact digest, source content,
   semantic policy/component identities, evidence, and referenced artifact
   semantic digests.
3. Opaque report/manifest/MCAP/mapping/alignment/stream/segment IDs, source URI and
   object version, and validation/publication timestamps are excluded. Their
   content digests replace them in identity-bearing positions.
4. Exact stored bytes remain a separate artifact-registry concern; a semantic
   digest is not an exact-byte digest.

### Consumer boundary

The V2 registration helper resolves the full schema quartet before wire validation.
A canonical admission-context resolver cross-binds the three validated bodies to the selected
source/alignment ledger decisions and consuming policy. The materializer, canonical offline
pipeline, and event identity boundary consume that resolved context rather than trusting loose
caller-provided IDs or digests.
The existing V1 local resolver may remain for compatibility, but it is not a V2
adapter and must not fabricate missing evidence. Canonical V2 callers consume the
self-carried digests rather than accepting duplicate caller-supplied digest values.

## Consequences

- The contract slice can represent complete local validation, READY, and alignment
  evidence without changing frozen V1 payloads.
- Moving identical bytes or allocating different association/publication rows does
  not change semantic identity; changing source, mapping, policy, validator, probe,
  or transform evidence does.
- A forged or unknown schema digest fails before payload validation.
- Existing V1 data requires a new derivation from the source bytes to become V2;
  it cannot be relabeled or automatically upcast.
- This decision supplies contracts and local semantic validation only. It does not
  claim governed source approval, production artifact durability, O-03/O-04 policy
  resolution, or Phase 1B completion.
- The canonical offline pipeline starts with an already resolved context; it does not inspect raw
  MCAP, publish V2 evidence, or replace durable source/alignment ledgers.

## Verification

Contract tests pin all three exact schema artifacts, validate representative V2
payloads through the catalog, reject forged pins and stale semantic digests, prove
URI/UUID/time independence, keep infrastructure failure `INCONCLUSIVE`, and reject
a mutable READY status.

## Implementation evidence

- `src/robata/contracts/admission_v2.py`
- `src/robata/admission/context.py`
- `src/robata/sampling/materializer.py`
- `src/robata/application/canonical/models.py`
- `src/robata/application/canonical/runner_support.py`
- `src/robata/application/canonical/runner.py`
- `src/robata/application/canonical_offline.py` (public re-export facade only)
- `tests/contract/test_admission_evidence_v2_contract.py`
- `tests/unit/test_admission_context.py`
- `tests/integration/test_canonical_offline.py`
