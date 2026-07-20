# ADR 0009: VLM Trust Boundary and Orchestrator Enrichment

- Status: Accepted and implemented for local contracts; exercised end to end in the
  post-admission single-part offline fusion slice
- Date: 2026-07-19
- Governing authority: Architecture V1.1 Sections 9.1, 10.2, 25.1, 25.7,
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
   claims retain the raw-byte digest; selected-attempt output binds raw and
   parsed digests; enriched logical identity includes selected output digest,
   request-catalog digest, target enriched-schema digest, and enrichment-policy
   version.
4. Enriched claims inject recording, package, camera, frame, mapping, alignment,
   prompt, inference-attempt, and work-node references. A provider score is
   stored only as `MODEL_REPORTED_UNCALIBRATED` with explicit model-attempt
   provenance. It is never called a probability, evidence strength, or policy
   confidence.
5. Provider-claim and enriched-output JSON Schemas are separate exact-pinned
   registry entries. The local `ProviderClaimEnricher` validates both schemas,
   the InputPlan schema digests, task-specific claim kinds, and six-camera
   coverage before publishing an enriched object.

## Consequences

- The strict offline fixture adapter persists exact raw bytes before parsing and can exercise the
  trust boundary without network access or real model credentials.
- The old normalized adapter envelope remains for compatibility; it is not
  evidence that the legacy application mainline has been rewired to this
  boundary.
- Durable raw/parsed/enriched/output-decision ledgers, complete schema quartets required by
  Section 25.7, production artifact storage, and a real provider adapter remain open
  governance/integration work.

## Implementation evidence

- `src/robata/inference/enrichment.py`
- `src/robata/inference/offline_fixture.py`
- `src/robata/application/canonical_offline.py`
- `src/robata/event_pipeline/identity_registry.py`
- `schemas/v1/provider-claim-payload.schema.json`
- `schemas/v1/orchestrator-enriched-output.schema.json`
- `tests/unit/test_inference_enrichment.py`
- `tests/unit/test_offline_fixture.py`
- `tests/integration/test_canonical_offline.py`
- `scripts/verify_schema_registry.py` verifies the exact checked-in schema catalog locally.
