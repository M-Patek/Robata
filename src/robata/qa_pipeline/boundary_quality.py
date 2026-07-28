"""Explicit non-geometric QA evidence for P12 boundary qualification.

This module deliberately does not model a timestamp, frame coordinate, or
boundary claim.  It records only whether independently accepted camera-quality
evidence may be considered by a *non-authoritative* boundary experiment.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, model_validator

from robata.contracts.cameras import CameraId
from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid

QualityMillionths = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]

BOUNDARY_CAMERA_QUALITY_EVIDENCE_PROJECTION_VERSION: Final = (
    "boundary-camera-quality-evidence-semantic-v1"
)
BOUNDARY_CAMERA_QUALITY_EVIDENCE_LOGICAL_KEY_NAMESPACE: Final = (
    "boundary-camera-quality-evidence-v1"
)


class BoundaryQualityApplicability(StrEnum):
    """Whether a QA fact may be consumed by a qualification-only candidate."""

    APPLICABLE = "APPLICABLE"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class BoundaryCameraCondition(StrEnum):
    """QA condition category, not a geometric observation."""

    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    UNUSABLE = "UNUSABLE"
    UNKNOWN = "UNKNOWN"
    INCOMPLETE = "INCOMPLETE"


class BoundaryCameraQualityEvidence(StrictModel):
    """One exact QA/calibration citation usable by P12 experimentation only."""

    mcap_id: OpaqueUuid
    recording_identity: Sha256Digest
    source_content_sha256: Sha256Digest
    camera_mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    camera_id: CameraId
    qa_result_logical_key: NodeLogicalKey
    qa_result_semantic_sha256: Sha256Digest
    qa_result_exact_sha256: Sha256Digest
    condition: BoundaryCameraCondition
    applicability: BoundaryQualityApplicability
    quality_millionths: QualityMillionths | None = None
    calibration_association_logical_key: NodeLogicalKey | None = None
    calibration_association_semantic_sha256: Sha256Digest | None = None
    calibration_association_exact_sha256: Sha256Digest | None = None
    policy_version: SchemaVersion
    projection_version: Literal["boundary-camera-quality-evidence-semantic-v1"] = (
        BOUNDARY_CAMERA_QUALITY_EVIDENCE_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest
    logical_key: NodeLogicalKey
    production_eligible: Literal[False] = False

    @classmethod
    def create(
        cls,
        *,
        mcap_id: str,
        recording_identity: str,
        source_content_sha256: str,
        camera_mapping_semantic_sha256: str,
        alignment_semantic_sha256: str,
        camera_id: CameraId,
        qa_result_logical_key: str,
        qa_result_semantic_sha256: str,
        qa_result_exact_sha256: str,
        condition: BoundaryCameraCondition,
        applicability: BoundaryQualityApplicability,
        quality_millionths: int | None,
        policy_version: str,
        calibration_association_logical_key: str | None = None,
        calibration_association_semantic_sha256: str | None = None,
        calibration_association_exact_sha256: str | None = None,
    ) -> Self:
        values: dict[str, Any] = {
            "mcap_id": mcap_id,
            "recording_identity": recording_identity,
            "source_content_sha256": source_content_sha256,
            "camera_mapping_semantic_sha256": camera_mapping_semantic_sha256,
            "alignment_semantic_sha256": alignment_semantic_sha256,
            "camera_id": camera_id,
            "qa_result_logical_key": qa_result_logical_key,
            "qa_result_semantic_sha256": qa_result_semantic_sha256,
            "qa_result_exact_sha256": qa_result_exact_sha256,
            "condition": condition,
            "applicability": applicability,
            "quality_millionths": quality_millionths,
            "calibration_association_logical_key": calibration_association_logical_key,
            "calibration_association_semantic_sha256": calibration_association_semantic_sha256,
            "calibration_association_exact_sha256": calibration_association_exact_sha256,
            "policy_version": policy_version,
            "projection_version": BOUNDARY_CAMERA_QUALITY_EVIDENCE_PROJECTION_VERSION,
            "production_eligible": False,
        }
        draft = cls.model_construct(
            semantic_sha256="0" * 64,
            logical_key=f"{BOUNDARY_CAMERA_QUALITY_EVIDENCE_LOGICAL_KEY_NAMESPACE}:{'0' * 64}",
            **values,
        )
        digest = semantic_sha256(boundary_camera_quality_evidence_projection(draft))
        return cls.model_validate(
            {
                **values,
                "semantic_sha256": digest,
                "logical_key": f"{BOUNDARY_CAMERA_QUALITY_EVIDENCE_LOGICAL_KEY_NAMESPACE}:{digest}",
            },
            strict=True,
        )

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        _require_logical_key_digest(
            self.qa_result_logical_key,
            self.qa_result_semantic_sha256,
            "QA result",
        )
        calibration_values = (
            self.calibration_association_logical_key,
            self.calibration_association_semantic_sha256,
            self.calibration_association_exact_sha256,
        )
        if any(value is None for value in calibration_values) and any(
            value is not None for value in calibration_values
        ):
            raise ValueError("calibration association citation must be complete or absent")
        if self.calibration_association_logical_key is not None:
            _require_logical_key_digest(
                self.calibration_association_logical_key,
                self.calibration_association_semantic_sha256,
                "calibration association",
            )
        if self.applicability is BoundaryQualityApplicability.APPLICABLE:
            if self.quality_millionths is None:
                raise ValueError("applicable boundary quality requires a quality value")
        elif self.quality_millionths is not None:
            raise ValueError(
                "missing or inapplicable boundary quality cannot carry a quality value"
            )
        digest = semantic_sha256(boundary_camera_quality_evidence_projection(self))
        if self.semantic_sha256 != digest:
            raise ValueError("boundary camera quality semantic identity is inconsistent")
        if self.logical_key != f"{BOUNDARY_CAMERA_QUALITY_EVIDENCE_LOGICAL_KEY_NAMESPACE}:{digest}":
            raise ValueError("boundary camera quality logical key is inconsistent")
        return self


def boundary_camera_quality_evidence_projection(
    evidence: BoundaryCameraQualityEvidence,
) -> dict[str, object]:
    """Return provenance and applicability only; QA is never geometry."""

    return {
        "semantic_projection_version": evidence.projection_version,
        "mcap_id": evidence.mcap_id,
        "recording_identity": evidence.recording_identity,
        "source_content_sha256": evidence.source_content_sha256,
        "camera_mapping_semantic_sha256": evidence.camera_mapping_semantic_sha256,
        "alignment_semantic_sha256": evidence.alignment_semantic_sha256,
        "camera_id": evidence.camera_id.value,
        "qa_result_logical_key": evidence.qa_result_logical_key,
        "qa_result_semantic_sha256": evidence.qa_result_semantic_sha256,
        "qa_result_exact_sha256": evidence.qa_result_exact_sha256,
        "condition": evidence.condition.value,
        "applicability": evidence.applicability.value,
        "quality_millionths": evidence.quality_millionths,
        "calibration_association_logical_key": evidence.calibration_association_logical_key,
        "calibration_association_semantic_sha256": evidence.calibration_association_semantic_sha256,
        "calibration_association_exact_sha256": evidence.calibration_association_exact_sha256,
        "policy_version": evidence.policy_version,
        "production_eligible": evidence.production_eligible,
        "geometry_excluded": True,
    }


def _require_logical_key_digest(logical_key: str, digest: str | None, subject: str) -> None:
    if digest is None or logical_key.rsplit(":", 1)[-1] != digest:
        raise ValueError(f"{subject} logical key must end with its semantic SHA-256")


__all__ = [
    "BOUNDARY_CAMERA_QUALITY_EVIDENCE_LOGICAL_KEY_NAMESPACE",
    "BOUNDARY_CAMERA_QUALITY_EVIDENCE_PROJECTION_VERSION",
    "BoundaryCameraCondition",
    "BoundaryCameraQualityEvidence",
    "BoundaryQualityApplicability",
    "boundary_camera_quality_evidence_projection",
]
