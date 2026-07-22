from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any

import pytest
from pydantic import ValidationError

from robata.application.canonical.output_admission import (
    CANONICAL_OUTPUT_DECISION_SCHEMA_ID,
    CANONICAL_OUTPUT_DECISION_SCHEMA_VERSION,
    CanonicalOutputAdmissionDecision,
    validate_registered_output_admission_decision,
)
from robata.application.canonical.projections import (
    CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE,
    _canonical_output_decision_projection_values,
    _stable_uuid,
)
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import (
    SchemaPinMismatchError,
    SchemaRef,
    SchemaRegistry,
    SchemaValidationError,
)
from robata.event_pipeline.identity_registry import (
    EVENT_HYPOTHESIS_SCHEMA_ID,
    EVENT_HYPOTHESIS_SCHEMA_VERSION,
    OUTPUT_ADMISSION_PROOF_SCHEMA_ID,
    OUTPUT_ADMISSION_PROOF_SCHEMA_VERSION,
    AdmissionEvidenceClass,
    AdmissionProof,
    OutputAdmissionProof,
    PlatformEnrichedEventHypothesis,
    PlatformEnrichedOutputReference,
    ProductionAdmittedHypothesisFact,
    ProductionOutputAdmissionPolicyRef,
    validate_registered_event_hypothesis,
    validate_registered_output_admission_proof,
)

WirePayload = (
    CanonicalOutputAdmissionDecision | OutputAdmissionProof | PlatformEnrichedEventHypothesis
)
WireValidator = Callable[[Any, SchemaRef, SchemaRegistry | None], Any]


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _payloads() -> tuple[
    CanonicalOutputAdmissionDecision,
    OutputAdmissionProof,
    PlatformEnrichedEventHypothesis,
]:
    recording_identity = _digest("registered-output-recording")
    enrichment = PlatformEnrichedOutputReference(
        authority="ORCHESTRATOR_ENRICHED",
        recording_identity=recording_identity,
        enrichment_logical_key=f"orchestrator-enrichment:{_digest('enrichment')}",
        enriched_output_semantic_sha256=_digest("enriched-output"),
        enrichment_policy_version="orchestrator-enrichment-v2",
    )
    fact = ProductionAdmittedHypothesisFact(
        fusion_output_ordinal=0,
        effective_interval=NanosecondInterval(start_ns=10, end_ns=20),
        semantic_fingerprint_sha256=_digest("event-fingerprint"),
        fusion_logical_key=f"fusion:{_digest('fusion')}",
    )
    policy = ProductionOutputAdmissionPolicyRef(
        version="output-admission-policy-v2",
        semantic_sha256=_digest("output-admission-policy"),
    )
    proof = OutputAdmissionProof.create_local_conformance(
        recording_identity=recording_identity,
        source_enrichments=(enrichment,),
        admitted_hypothesis_facts=(fact,),
        policy=policy,
    )
    hypothesis = PlatformEnrichedEventHypothesis.create(
        recording_identity=recording_identity,
        effective_interval=fact.effective_interval,
        semantic_fingerprint_sha256=fact.semantic_fingerprint_sha256,
        fusion_logical_key=fact.fusion_logical_key,
        fusion_output_ordinal=fact.fusion_output_ordinal,
        source_enrichments=(enrichment,),
        production_admission=AdmissionProof(
            decision="ADMITTED",
            evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
            production_eligible=False,
            recording_identity=recording_identity,
            admitted_context_semantic_sha256=_digest("admitted-context"),
            admission_policy_version="primary-admission-v2",
            admission_policy_sha256=_digest("primary-admission-policy"),
        ),
        production_output_admission=proof,
    )
    fusion_digest = _digest("fusion-reduction")
    decision_digest = semantic_sha256(
        _canonical_output_decision_projection_values(
            decision="ADMITTED",
            evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
            production_eligible=False,
            recording_identity=recording_identity,
            source_enrichments=(enrichment,),
            fusion_reduction_logical_key=f"fusion-reduction:{fusion_digest}",
            fusion_reduction_semantic_sha256=fusion_digest,
            policy_version=policy.version,
            policy_sha256=policy.semantic_sha256,
            admitted_claim_ordinals=(0,),
            reason_code="CLAIMS_ADMITTED",
            production_output_admission=proof,
        )
    )
    decision = CanonicalOutputAdmissionDecision(
        schema_version="2.0",
        decision_id=_stable_uuid(
            CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE,
            decision_digest,
        ),
        decision="ADMITTED",
        evidence_class=AdmissionEvidenceClass.LOCAL_CONFORMANCE,
        production_eligible=False,
        semantic_sha256=decision_digest,
        recording_identity=recording_identity,
        source_enrichments=(enrichment,),
        fusion_reduction_logical_key=f"fusion-reduction:{fusion_digest}",
        fusion_reduction_semantic_sha256=fusion_digest,
        policy_version=policy.version,
        policy_sha256=policy.semantic_sha256,
        admitted_claim_ordinals=(0,),
        reason_code="CLAIMS_ADMITTED",
        production_output_admission=proof,
    )
    return decision, proof, hypothesis


def _contracts(
    registry: SchemaRegistry,
) -> tuple[tuple[WirePayload, SchemaRef, WireValidator], ...]:
    decision, proof, hypothesis = _payloads()
    return (
        (
            decision,
            registry.resolve_version(
                CANONICAL_OUTPUT_DECISION_SCHEMA_ID,
                CANONICAL_OUTPUT_DECISION_SCHEMA_VERSION,
            ).ref,
            validate_registered_output_admission_decision,
        ),
        (
            proof,
            registry.resolve_version(
                OUTPUT_ADMISSION_PROOF_SCHEMA_ID,
                OUTPUT_ADMISSION_PROOF_SCHEMA_VERSION,
            ).ref,
            validate_registered_output_admission_proof,
        ),
        (
            hypothesis,
            registry.resolve_version(
                EVENT_HYPOTHESIS_SCHEMA_ID,
                EVENT_HYPOTHESIS_SCHEMA_VERSION,
            ).ref,
            validate_registered_event_hypothesis,
        ),
    )


def test_output_wire_contracts_are_exact_pinned_and_round_trip() -> None:
    registry = SchemaRegistry()
    expected = {
        CANONICAL_OUTPUT_DECISION_SCHEMA_ID: SchemaRef(
            schema_id=CANONICAL_OUTPUT_DECISION_SCHEMA_ID,
            version="2.0.0",
            artifact_id="36948170-81ea-9f1c-9da5-a36702671043",
            sha256="ad530439a64c6154af3dfeb056dfea06ccf354b496e6526bd818070740fb0288",
        ),
        OUTPUT_ADMISSION_PROOF_SCHEMA_ID: SchemaRef(
            schema_id=OUTPUT_ADMISSION_PROOF_SCHEMA_ID,
            version="2.0.0",
            artifact_id="0c2ab969-b49f-6d68-a6fb-bbf3cf140022",
            sha256="c755fb5c49b99c37c8736b9f5ac55448bbe751036d23ccc1f4cd1da642858596",
        ),
        EVENT_HYPOTHESIS_SCHEMA_ID: SchemaRef(
            schema_id=EVENT_HYPOTHESIS_SCHEMA_ID,
            version="2.0.0",
            artifact_id="90cfed99-d698-db3b-dd9b-fd964b0c62a9",
            sha256="ea58b7cd8047609b2455fe87b8b097dc220e34a335a21f303771a25fcd0e4646",
        ),
    }

    for payload, reference, validator in _contracts(registry):
        assert reference == expected[reference.schema_id]
        assert validator(payload, reference, registry) == payload
        assert type(payload).model_validate_json(payload.model_dump_json()) == payload
        assert "schema_ref" not in type(payload).model_fields
        document = registry.get_schema(reference)
        assert document["additionalProperties"] is False
        assert set(document["required"]) == set(document["properties"])
        assert document["properties"]["schema_version"]["const"] == "2.0"


def test_output_wire_validation_rejects_forged_pins_and_wrong_contracts() -> None:
    registry = SchemaRegistry()
    contracts = _contracts(registry)

    for payload, reference, validator in contracts:
        forged = reference.model_copy(update={"sha256": "0" * 64})
        with pytest.raises(SchemaPinMismatchError):
            validator(payload, forged, registry)

    decision, _, decision_validator = contracts[0]
    wrong_reference = contracts[1][1]
    with pytest.raises(ValueError, match="schema_ref must identify"):
        decision_validator(decision, wrong_reference, registry)


def test_output_wire_validation_combines_closed_wire_and_semantic_checks() -> None:
    registry = SchemaRegistry()

    for payload, reference, validator in _contracts(registry):
        extra = payload.model_dump(mode="json") | {"unexpected": True}
        with pytest.raises(SchemaValidationError) as caught:
            registry.validate_pinned(reference, extra)
        assert caught.value.json_path == "$.unexpected"

        tampered = payload.model_copy(update={"semantic_sha256": "0" * 64})

        with pytest.raises(ValidationError, match="semantic_sha256 is inconsistent"):
            validator(tampered, reference, registry)
