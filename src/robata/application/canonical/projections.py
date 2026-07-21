"""Semantic projections and deterministic identity helpers for the canonical flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final
from uuid import NAMESPACE_URL, uuid5

from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import semantic_sha256
from robata.contracts.pipeline import SamplingPurpose
from robata.event_pipeline.identity_registry import (
    AdmissionEvidenceClass,
    OutputAdmissionProof,
    PlatformEnrichedOutputReference,
    ProductionOutputAdmissionPolicyRef,
    platform_enriched_output_logical_projection,
    validate_evidence_eligibility,
)
from robata.inference.enrichment import EnrichedProviderClaim

if TYPE_CHECKING:
    from robata.application.canonical.models import CanonicalOfflineExecutionPolicy
    from robata.application.canonical.output_admission import CanonicalOutputAdmissionDecision
    from robata.application.canonical.reduction import (
        CanonicalFusionPartSource,
        CanonicalFusionReduction,
        CanonicalReducedFusionClaim,
    )


CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION: Final = (
    "canonical-output-decision-semantic-v2"
)
CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE: Final = "output-admission-decision-v2"
CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE: Final = "canonical-output-admission-v2"
CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION: Final = "canonical-output-decision-node-v2"
CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION: Final = (
    "canonical-offline-execution-policy-semantic-v3"
)


def canonical_fusion_reduction_projection(
    reduction: CanonicalFusionReduction,
) -> dict[str, object]:
    return _canonical_fusion_reduction_projection_values(
        schema_version=reduction.schema_version,
        input_plan_semantic_sha256=reduction.input_plan_semantic_sha256,
        barrier_reduction_semantic_sha256=reduction.barrier_reduction_semantic_sha256,
        reduction_policy=reduction.reduction_policy,
        reduction_policy_version=reduction.reduction_policy_version,
        outcome=reduction.outcome,
        parts=reduction.parts,
        claims=reduction.claims,
    )


def _canonical_fusion_reduction_projection_values(
    *,
    schema_version: str,
    input_plan_semantic_sha256: str,
    barrier_reduction_semantic_sha256: str,
    reduction_policy: str,
    reduction_policy_version: str,
    outcome: str,
    parts: Sequence[CanonicalFusionPartSource],
    claims: Sequence[CanonicalReducedFusionClaim],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "input_plan_semantic_sha256": input_plan_semantic_sha256,
        "barrier_reduction_semantic_sha256": barrier_reduction_semantic_sha256,
        "reduction_policy": reduction_policy,
        "reduction_policy_version": reduction_policy_version,
        "outcome": outcome,
        "parts": [
            {
                "part_ordinal": part.part_ordinal,
                "part_semantic_sha256": part.part_semantic_sha256,
                "selected_attempt_output_sha256": part.selected_attempt_output_sha256,
                "enrichment": platform_enriched_output_logical_projection(part.enrichment),
                "abstained": part.abstained,
            }
            for part in parts
        ],
        "claims": [
            {
                "fusion_output_ordinal": claim.fusion_output_ordinal,
                "claim_semantic_sha256": claim.claim_semantic_sha256,
                "representative": _fusion_claim_reduction_projection(claim.representative),
                "sources": [
                    {
                        "part_ordinal": item.part_ordinal,
                        "source_claim_ordinal": item.source_claim_ordinal,
                        "enrichment_logical_key": item.enrichment_logical_key,
                    }
                    for item in claim.sources
                ],
            }
            for claim in claims
        ],
    }


def canonical_root_window_projection_values(values: Mapping[str, object]) -> dict[str, object]:
    """Project root-window semantics without row IDs, associations, or clocks."""

    requested = values["requested_interval"]
    effective = values["interval"]
    if not isinstance(requested, NanosecondInterval) or not isinstance(
        effective, NanosecondInterval
    ):
        raise TypeError("window intervals must be NanosecondInterval values")
    purpose = values["purpose"]
    if not isinstance(purpose, SamplingPurpose):
        raise TypeError("window purpose must be a SamplingPurpose")
    return {
        "source_content_sha256": values["source_content_sha256"],
        "camera_mapping_semantic_sha256": values["camera_mapping_semantic_sha256"],
        "alignment_semantic_sha256": values["alignment_semantic_sha256"],
        "requested_interval": {
            "start_ns": str(requested.start_ns),
            "end_ns": str(requested.end_ns),
        },
        "interval": {
            "start_ns": str(effective.start_ns),
            "end_ns": str(effective.end_ns),
        },
        "purpose": purpose.value,
        "window_policy_version": values["window_policy_version"],
        "source_subject_type": values["source_subject_type"],
        "source_subject_logical_key": values["source_subject_logical_key"],
        "parent_window_logical_key": values["parent_window_logical_key"],
        "source_lineage_sha256": values["source_lineage_sha256"],
        "refinement_role": values["refinement_role"],
        "generation": values["generation"],
    }


def canonical_execution_policy_projection_values(values: Mapping[str, object]) -> dict[str, object]:
    output_policy = values["output_admission_policy"]
    if not isinstance(output_policy, ProductionOutputAdmissionPolicyRef):
        raise TypeError("output_admission_policy must be a ProductionOutputAdmissionPolicyRef")
    return {
        "semantic_projection_version": CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION,
        "policy_version": values["policy_version"],
        "window_policy_version": values["window_policy_version"],
        "token_policy_version": values["token_policy_version"],
        "parser_version": values["parser_version"],
        "enrichment_policy_version": values["enrichment_policy_version"],
        "projector_policy_version": values["projector_policy_version"],
        "reduction_policy": values["reduction_policy"],
        "reduction_policy_version": values["reduction_policy_version"],
        "provisional_fusion_policy_version": values["provisional_fusion_policy_version"],
        "boundary_refinement_policy_version": values["boundary_refinement_policy_version"],
        "max_attempts": values["max_attempts"],
        "output_admission_policy": output_policy.model_dump(mode="json"),
    }


def canonical_execution_policy_projection(
    policy: CanonicalOfflineExecutionPolicy,
) -> dict[str, object]:
    return canonical_execution_policy_projection_values(
        {
            "policy_version": policy.policy_version,
            "window_policy_version": policy.window_policy_version,
            "token_policy_version": policy.token_policy_version,
            "parser_version": policy.parser_version,
            "enrichment_policy_version": policy.enrichment_policy_version,
            "projector_policy_version": policy.projector_policy_version,
            "reduction_policy": policy.reduction_policy,
            "reduction_policy_version": policy.reduction_policy_version,
            "provisional_fusion_policy_version": policy.provisional_fusion_policy_version,
            "boundary_refinement_policy_version": (policy.boundary_refinement_policy_version),
            "max_attempts": policy.max_attempts,
            "output_admission_policy": policy.output_admission_policy,
        }
    )


def canonical_output_decision_projection(
    decision: CanonicalOutputAdmissionDecision,
) -> dict[str, object]:
    return _canonical_output_decision_projection_values(
        decision=decision.decision,
        evidence_class=decision.evidence_class,
        production_eligible=decision.production_eligible,
        recording_identity=decision.recording_identity,
        source_enrichments=decision.source_enrichments,
        fusion_reduction_logical_key=decision.fusion_reduction_logical_key,
        fusion_reduction_semantic_sha256=decision.fusion_reduction_semantic_sha256,
        policy_version=decision.policy_version,
        policy_sha256=decision.policy_sha256,
        admitted_claim_ordinals=decision.admitted_claim_ordinals,
        reason_code=decision.reason_code,
        production_output_admission=decision.production_output_admission,
    )


def _canonical_output_decision_projection_values(
    *,
    decision: str,
    evidence_class: AdmissionEvidenceClass,
    production_eligible: bool,
    recording_identity: str,
    source_enrichments: Sequence[PlatformEnrichedOutputReference],
    fusion_reduction_logical_key: str,
    fusion_reduction_semantic_sha256: str,
    policy_version: str,
    policy_sha256: str,
    admitted_claim_ordinals: Sequence[int],
    reason_code: str,
    production_output_admission: OutputAdmissionProof | None,
) -> dict[str, object]:
    validate_evidence_eligibility(evidence_class, production_eligible)
    return {
        "semantic_projection_version": CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION,
        "decision": decision,
        "evidence_class": evidence_class.value,
        "production_eligible": production_eligible,
        "recording_identity": recording_identity,
        "source_enrichments": [
            platform_enriched_output_logical_projection(item) for item in source_enrichments
        ],
        "fusion_reduction_logical_key": fusion_reduction_logical_key,
        "fusion_reduction_semantic_sha256": fusion_reduction_semantic_sha256,
        "policy_version": policy_version,
        "policy_sha256": policy_sha256,
        "admitted_claim_ordinals": list(admitted_claim_ordinals),
        "reason_code": reason_code,
        "production_output_admission_semantic_sha256": (
            None
            if production_output_admission is None
            else production_output_admission.semantic_sha256
        ),
    }


def _fusion_claim_reduction_projection(
    claim: EnrichedProviderClaim,
) -> dict[str, object]:
    """Project enriched claim content without row IDs or storage locators."""

    confidence = claim.model_reported_confidence
    return {
        "kind": claim.kind.value,
        "package_ordinal": claim.package_ordinal,
        "camera_id": None if claim.camera_id is None else claim.camera_id.value,
        "interval": None if claim.interval is None else claim.interval.model_dump(mode="json"),
        "label": claim.label,
        "observation": claim.observation.value,
        "evidence": [
            {
                "package_ordinal": item.package_ordinal,
                "package_semantic_content_sha256": item.package_semantic_content_sha256,
                "camera_id": item.camera_id.value,
                "camera_ordinal": item.camera_ordinal,
                "frame_ordinal": item.frame_ordinal,
                "aligned_timestamp_ns": str(item.aligned_timestamp_ns),
                "source_timestamp_ns": str(item.source_timestamp_ns),
                "source_artifact_sha256": item.source_artifact_sha256,
            }
            for item in claim.evidence
        ],
        "model_reported_confidence": (
            None
            if confidence is None
            else {
                "kind": confidence.kind,
                "semantics": confidence.semantics,
                "producer_type": confidence.producer_type,
                "producer_version": confidence.producer_version,
                "value": confidence.value,
            }
        ),
        "conflict_codes": list(claim.conflict_codes),
    }


def _fusion_claim_reduction_digest(claim: EnrichedProviderClaim) -> str:
    return semantic_sha256(_fusion_claim_reduction_projection(claim))


def _stable_uuid(namespace: str, *parts: object) -> str:
    material = ":".join(str(item) for item in parts)
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{material}"))


__all__ = [
    "CANONICAL_EXECUTION_POLICY_SEMANTIC_PROJECTION_VERSION",
    "CANONICAL_OUTPUT_DECISION_IDENTITY_POLICY_VERSION",
    "CANONICAL_OUTPUT_DECISION_LOGICAL_KEY_NAMESPACE",
    "CANONICAL_OUTPUT_DECISION_SEMANTIC_PROJECTION_VERSION",
    "CANONICAL_OUTPUT_DECISION_UUID_NAMESPACE",
    "canonical_execution_policy_projection",
    "canonical_execution_policy_projection_values",
    "canonical_fusion_reduction_projection",
    "canonical_output_decision_projection",
    "canonical_root_window_projection_values",
]
