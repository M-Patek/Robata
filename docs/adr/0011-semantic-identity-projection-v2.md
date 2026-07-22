# ADR 0011: Semantic Identity Projection V2

- Status: Accepted for the pre-production canonical path
- Date: 2026-07-20
- Governing authority: Architecture V1.1 Sections 25.2 and 25.4; ADR 0004

## Context

The request-catalog projection correctly stopped hashing
`manifest_bytes_sha256`: an exact serialization digest is audit evidence, not
logical input identity. That correction changed the projection formula while
the surrounding registered input-planning wire objects still carried
`schema_version="1.0"` and callers could still use a `planner-v1` label.
Wire compatibility therefore hid an identity-policy migration.

An unversioned formula change can place old and new meanings in the same
logical-key namespace. It can also silently alter downstream call-plan, part,
barrier, idempotency, or fence identities through transitive hashes.

## Decision

Wire schema versions and semantic projection versions are independent. The
published input-planning V1 wire shapes remain unchanged. The then-unregistered
local `OutputAdmissionProof`, `PlatformEnrichedEventHypothesis`, and
`CanonicalOutputAdmissionDecision` shapes changed with the evidence metadata
and therefore use `schema_version="2.0"`; their V1 payloads fail closed. The
canonical chain uses the following hash-bearing constants:

| Identity | Projection or policy version |
| --- | --- |
| input planner implementation | `inference-input-planner-v2` |
| request catalog | `request-catalog-semantic-v2` |
| input plan | `inference-input-plan-semantic-v2` |
| call plan | `inference-call-plan-semantic-v2` |
| call part | `inference-call-part-semantic-v2` |
| call barrier | `inference-call-barrier-semantic-v2` |
| provider idempotency key | `inference-call-idempotency-key-v2` |
| primary admission proof | `admission-proof-semantic-v2` |
| output admission proof | `output-admission-semantic-v2` |
| canonical output decision | `canonical-output-decision-semantic-v2` |
| event hypothesis | `event-hypothesis-semantic-v2` |
| canonical offline processing-run binding | `canonical-offline-v2` |

Each projection or key-policy value is included in the canonical preimage it
versions. This implementation accepts only
`planner_version=inference-input-planner-v2`; the existing
`target.planner_version` field is also part of the input-plan and call-plan
preimages. Caller-supplied provider-idempotency and reduction versions remain
semantic input but do not replace formula-level versions.

Every affected key domain moves to an explicit V2 namespace:

| Key | V2 namespace |
| --- | --- |
| canonical input-plan logical node and input-plan UUID | `inference-input-plan-v2` |
| call-part logical node | `inference-input-call-part-v2` |
| call barrier logical node | `inference-input-barrier-v2` |
| provider idempotency key | `inference-input-call-v2` |
| request-catalog UUID | `provider-request-catalog-v2` |
| canonical output decision logical node | `output-admission-decision-v2` |
| canonical output-decision UUID | `canonical-output-admission-v2` |
| event hypothesis | `event-hypothesis-v2` |

The affected canonical logical-node producers also publish matching
`canonical-*-node-v2` identity-policy versions. Changing only that audit
field is never sufficient because logical-node identity does not include it;
the key namespace must change as well.

Output outcome and evidence strength are separate dimensions. Output decisions
use `ADMITTED`, `NO_EVENTS`, or `ABSTAINED`. Admission proofs and
decisions also carry `evidence_class` with one of
`LOCAL_CONFORMANCE`, `GOVERNED_BENCHMARK`, or
`PRODUCTION_QUALIFIED`, plus `production_eligible`. The production enum
value is reserved, but cannot be minted by a factory, direct model validation,
or identity-service configuration until a governed qualification gateway is
implemented. Every currently constructible evidence class has
`production_eligible=false`.

The offline canonical composition uses the neutral `AdmissionProof` and
`OutputAdmissionProof` contracts through explicit local-conformance factories.
It stops after the local output decision and immutable event hypotheses. The
identity registry is neither injected nor called, and the result requires
`identity_result=None`. It therefore cannot satisfy the Section 25.9
production-only gate. The legacy `PRODUCTION_ADMITTED` status, V1 local output
payloads, and V1 output/event key domains fail closed rather than being
relabeled.

Any future change to a projection used by a semantic hash, logical key,
idempotency key, barrier, or fence must:

1. increase the directly owned projection or policy version;
2. increase every downstream projection version whose meaning changes;
3. allocate a new namespace for every affected logical or idempotency key;
4. update deterministic UUID namespaces where those UUIDs identify the
   migrated derivation; and
5. add conformance tests for excluded fields, hash-bearing version markers,
   and rejection of stale namespaces.

This rule applies to field additions or removals, canonicalization changes, and
meaning changes even when the wire schema is byte-for-byte unchanged.

The frozen registered V1 input-planning wire objects already carry
`target.planner_version`, so this migration uses that field plus hash-bearing
formula constants and does not modify the published schema. A future registered
wire-shape change, including adding a dedicated projection-version field to the
payload, requires a newly registered schema version rather than an in-place V1
edit. At adoption the local output contracts were not registered schemas, but
still advanced their explicit wire version because their shape changed.

### Registration follow-up (2026-07-21)

The unchanged V2 payload shapes are now registered independently as
`canonical-output-decision@2.0.0`, `output-admission-proof@2.0.0`, and
`event-hypothesis@2.0.0`. Their validation entry points require the exact
`SchemaRef` as out-of-band artifact or transport metadata before validating the
payload. The quartet is deliberately not inserted into the domain objects:
doing so would alter their already published embedding in canonical completion
detail V4. This registration does not relabel or upcast any V1 payload.

## Migration

The project is pre-production. Local request catalogs, input plans, call
members, barriers, output decisions, event hypotheses, and idempotency records
derived under the old formula are rebuilt. Local V1 output-proof, decision, and
hypothesis payloads are not relabeled or upcast. `canonical-offline-v1` run
records are not resumed by the V2 composition, including records whose status
is still `RUNNING`. No general migration framework is introduced.

Exact manifest bytes, artifact locators, row UUIDs, and timestamps remain
outside semantic identity. Their audit and storage roles are unchanged.

## Consequences

- Old and new projection meanings cannot collide in one live key namespace.
- A registered input wire `schema_version="1.0"` no longer implies
  identity-policy V1.
- Formula migrations are explicit and testable without modifying published
  schema bytes.
- V1 local derivations, output payloads, and run records require rebuild before
  use by the canonical V2 path.

## Implementation evidence

- `src/robata/inference/input_plan.py`
- `src/robata/application/canonical/projections.py`
- `src/robata/application/canonical/logical_nodes.py`
- `src/robata/application/canonical/output_admission.py`
- `src/robata/event_pipeline/identity_registry.py`
- `src/robata/application/canonical/result_validation.py`
- `src/robata/application/canonical/runner.py`
- `src/robata/application/canonical_offline.py` (public re-export facade only)
- `tests/unit/test_inference_input_plan.py`
- `tests/unit/test_canonical_logical_node_producers.py`
- `tests/contract/test_output_wire_contract.py`
- `tests/integration/test_canonical_offline.py`
