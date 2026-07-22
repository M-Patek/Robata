"""Single persisted Wire envelope for local supplemental QA evidence."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.qa_pipeline.supplemental import (
    SupplementalQaDenseInputPlan,
    SupplementalQaDenseResult,
)
from robata.sampling.supplemental import (
    FrozenSupplementalTargetPlan,
    ProviderNeutralSupplementalPackage,
    SupplementalEvidenceClass,
)

LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID = (
    "https://schemas.robata.dev/local-supplemental-qa-evidence"
)
LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION = "2.0.0"
LOCAL_SUPPLEMENTAL_QA_EVIDENCE_WIRE_VERSION = "2.0"
LOCAL_SUPPLEMENTAL_QA_EVIDENCE_PROJECTION_VERSION = "local-supplemental-qa-evidence-semantic-v2"


class LocalSupplementalQaEvidence(StrictModel):
    """Persisted local-only chain from frozen targets through QA_DENSE consumption."""

    schema_version: Literal["2.0"] = "2.0"
    schema_ref: SchemaRef
    semantic_sha256: Sha256Digest
    frozen_plan: FrozenSupplementalTargetPlan
    package: ProviderNeutralSupplementalPackage
    package_manifest_sha256: Sha256Digest
    input_plan: SupplementalQaDenseInputPlan
    result: SupplementalQaDenseResult
    projection_version: Literal["local-supplemental-qa-evidence-semantic-v2"] = (
        "local-supplemental-qa-evidence-semantic-v2"
    )
    evidence_class: Literal[SupplementalEvidenceClass.LOCAL_CONFORMANCE] = (
        SupplementalEvidenceClass.LOCAL_CONFORMANCE
    )
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_chain(self) -> Self:
        if (
            self.schema_ref.schema_id != LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID
            or self.schema_ref.version != LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION
        ):
            raise ValueError("schema_ref must identify local-supplemental-qa-evidence@2.0.0")
        if (
            self.package.target_plan_id != self.frozen_plan.plan_id
            or self.package.target_plan_semantic_sha256 != self.frozen_plan.semantic_sha256
            or self.package.source_content_sha256 != self.frozen_plan.source_content_sha256
            or self.package.camera_mapping_semantic_sha256
            != self.frozen_plan.camera_mapping_semantic_sha256
            or self.package.alignment_semantic_sha256 != self.frozen_plan.alignment_semantic_sha256
        ):
            raise ValueError("supplemental package does not bind the frozen target plan")
        expected_manifest_sha256 = exact_bytes_sha256(canonical_json_bytes(self.package))
        if self.package_manifest_sha256 != expected_manifest_sha256:
            raise ValueError("package_manifest_sha256 does not match exact package bytes")
        if (
            self.input_plan.package_id != self.package.package_id
            or self.input_plan.package_semantic_content_sha256
            != self.package.semantic_content_sha256
            or self.input_plan.package_manifest_sha256 != self.package_manifest_sha256
            or self.input_plan.target_plan_id != self.frozen_plan.plan_id
            or self.input_plan.target_plan_semantic_sha256 != self.frozen_plan.semantic_sha256
        ):
            raise ValueError("supplemental QA_DENSE input does not bind plan/package evidence")
        if (
            self.result.input_plan_id != self.input_plan.input_plan_id
            or self.result.input_plan_semantic_sha256 != self.input_plan.semantic_sha256
            or self.result.package_id != self.package.package_id
            or self.result.package_semantic_content_sha256 != self.package.semantic_content_sha256
            or self.result.package_manifest_sha256 != self.package_manifest_sha256
            or self.result.consumer_policy_version != self.input_plan.consumer_policy_version
        ):
            raise ValueError("supplemental QA_DENSE result does not bind input/package evidence")
        if len(self.result.consumptions) != len(self.frozen_plan.targets):
            raise ValueError("supplemental QA_DENSE result does not cover every frozen target")
        for target, outcome, consumption in zip(
            self.frozen_plan.targets,
            self.package.outcomes,
            self.result.consumptions,
            strict=True,
        ):
            if (
                outcome.target != target
                or consumption.target_ordinal != target.ordinal
                or consumption.camera_id is not target.camera_id
                or consumption.target_ns != target.target_ns
                or consumption.package_status is not outcome.status
            ):
                raise ValueError("supplemental target/package/result coordinates disagree")
        expected_digest = semantic_sha256(local_supplemental_qa_evidence_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match supplemental QA evidence")
        return self


def local_supplemental_qa_evidence_projection(
    evidence: LocalSupplementalQaEvidence,
) -> dict[str, object]:
    """Return the top-level versioned evidence-chain identity projection."""

    return {
        "schema_version": evidence.schema_version,
        "projection_version": evidence.projection_version,
        "frozen_plan": {
            "plan_id": evidence.frozen_plan.plan_id,
            "semantic_sha256": evidence.frozen_plan.semantic_sha256,
        },
        "package": {
            "package_id": evidence.package.package_id,
            "semantic_content_sha256": evidence.package.semantic_content_sha256,
        },
        "input_plan": {
            "input_plan_id": evidence.input_plan.input_plan_id,
            "semantic_sha256": evidence.input_plan.semantic_sha256,
        },
        "result": {
            "result_id": evidence.result.result_id,
            "semantic_sha256": evidence.result.semantic_sha256,
        },
        "evidence_class": evidence.evidence_class.value,
        "production_eligible": evidence.production_eligible,
    }


__all__ = [
    "LOCAL_SUPPLEMENTAL_QA_EVIDENCE_PROJECTION_VERSION",
    "LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_ID",
    "LOCAL_SUPPLEMENTAL_QA_EVIDENCE_SCHEMA_VERSION",
    "LOCAL_SUPPLEMENTAL_QA_EVIDENCE_WIRE_VERSION",
    "LocalSupplementalQaEvidence",
    "local_supplemental_qa_evidence_projection",
]
