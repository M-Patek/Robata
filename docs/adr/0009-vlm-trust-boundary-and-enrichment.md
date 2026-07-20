# ADR 0009: VLM Trust Boundary and Orchestrator Enrichment

- Status: Accepted and implemented for local contracts; exercised end to end in the
  post-admission single- and multi-part offline fusion slice
- Date: 2026-07-19
- Last implementation update: 2026-07-20
- Governing authority: Architecture V1.1 Sections 9.1, 10.2, 25.1, 25.4, 25.7,
  and 25.11

## Context

The earlier adapter contract allowed a normalized payload to look like a
persisted domain result. That shape made it possible for a provider response to
appear to author event IDs, source lineage, evidence references, or trusted
confidence. Architecture V1.1 requires raw provider bytes, parsed provider
claims, and orchestrator-enriched output to remain separate immutable artifacts.

## Decision

1. `ProviderClaimPayload` is the provider-facing wire. It contains only bounded
   claim ordinals, local package/camera ordinals, intervals, labels, enumerated
   observations, opaque correlation tokens, and an optional model-reported
   score. Unknown fields are rejected. Provider payloads cannot allocate UUIDs,
   logical keys, source locators, package/frame IDs, mapping/alignment IDs,
   prompt/schema lineage, event/revision IDs, or calibration state.
2. `ProviderReferenceCatalog` is derived from one immutable `InferenceInputPlan`.
   Correlation tokens are deterministic opaque join hints. The enricher matches
   them exactly, rejects missing/duplicate/out-of-catalog references, and binds
   local coordinates to the request catalog before adding authority.
3. `RawProviderResponseArtifact`, `ParsedProviderClaimArtifact`, and
   `OrchestratorEnrichedOutput` have distinct identities and digests. Parsed
   claims retain the raw-byte digest; selected-attempt output binds immutable raw
   and parsed attempt content plus the selection-decision logical key. Inference
   and selection IDs remain audit locators and are excluded from semantic identity.
   Enriched logical identity includes selected output digest,
   request-catalog digest, target enriched-schema digest, and enrichment-policy
   version.
4. Enriched claims inject recording, package, camera, frame, mapping, alignment,
   prompt, inference-attempt, and work-node references. A provider score is
   stored only as `MODEL_REPORTED_UNCALIBRATED` with explicit model-attempt
   provenance. It is never called a probability, evidence strength, or policy
   confidence.
5. Provider-claim and enriched-output JSON Schemas are separate exact-pinned
   registry entries. Enriched-output v2 carries the required selection proof;
   v1 remains frozen and no selection lineage is fabricated for it. The local
   `ProviderClaimEnricher` validates both schemas,
   the InputPlan schema digests, task-specific claim kinds, and six-camera
   coverage before publishing an enriched object.
6. Intent, terminal attempt, attempt selection, typed raw response, parsed claim,
   selected output, and enriched output use exact-pinned schemas in one local
   append-only SQLite evidence graph. Exact provider bytes are committed before
   parsing; a terminal that references those bytes also creates the typed raw
   artifact, including for invalid output. Reopening the database through fresh
   ledger, adapter, and pipeline instances may reuse an already selected success
   without another provider dispatch.

## Consequences

- The strict offline fixture adapter persists exact raw bytes before parsing and can exercise the
  trust boundary without network access or real model credentials.
- Package, prompt, schema, attempt, selection-row, and artifact locators remain available for exact
  audit and fail-closed matching but are excluded from reusable logical preimages; content or policy
  changes still produce distinct identities.
- This SQLite graph is restartable local conformance evidence. It does not choose the O-14
  production database, object store, isolation model, recovery topology, or retention policy.
- The old normalized adapter envelope remains for compatibility. The legacy application runner
  has been removed from the live package; archived results do not prove conformance to this
  boundary.
- Durable output-decision/run/work/barrier records, production artifact storage, and a real
  provider adapter remain open governance/integration work.

## Implementation evidence

- `src/robata/inference/enrichment.py`
- `src/robata/inference/offline_fixture.py`
- `src/robata/inference/evidence.py`
- `src/robata/adapters/sqlite_inference_evidence.py`
- `src/robata/application/canonical/runner.py`
- `src/robata/application/canonical/result_validation.py`
- `src/robata/application/canonical/output_admission.py`
- `src/robata/application/canonical_offline.py` (public re-export facade only)
- `src/robata/event_pipeline/identity_registry.py`
- `schemas/v1/provider-claim-payload.schema.json`
- `schemas/v1/inference-intent.schema.json`
- `schemas/v1/model-inference.schema.json`
- `schemas/v1/inference-attempt-selection.schema.json`
- `schemas/v1/raw-provider-response-artifact.schema.json`
- `schemas/v1/parsed-provider-claim-artifact.schema.json`
- `schemas/v1/selected-attempt-output.schema.json`
- `schemas/v1/orchestrator-enriched-output.schema.json`
- `schemas/v2/orchestrator-enriched-output.schema.json`
- `tests/unit/test_inference_enrichment.py`
- `tests/unit/test_offline_fixture.py`
- `tests/unit/test_sqlite_inference_evidence.py`
- `tests/integration/test_canonical_offline.py`
- `scripts/verify_schema_registry.py` verifies the exact checked-in schema catalog locally.
