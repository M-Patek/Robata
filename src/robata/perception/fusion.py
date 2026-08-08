"""Deterministic multi-view fusion for stream-oriented perception.

Unlike the legacy window fusion, confidence is normalized over cameras that were
selected *and* observable for the event track.  Missing or deliberately unselected
cameras remain explicit evidence slots but do not silently depress the denominator.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final, Self, cast
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import NodeLogicalKey, OpaqueUuid
from robata.contracts.perception_stream import CameraEvidenceRelation, RefineReason, UnitInterval
from robata.perception.projectors import (
    EvidenceProjection,
    ProjectedQaDisposition,
    QaProjection,
)
from robata.perception.tracking import EventTrackRevision

PERCEPTION_FUSION_PROJECTION_VERSION: Final = "perception-fusion-semantic-v1"
PERCEPTION_FUSION_KEY_NAMESPACE: Final = "perception-fusion-v1"
PERCEPTION_EVENT_UUID_NAMESPACE: Final = "robata:perception-event-v1"

PositiveCameraCount = Annotated[int, Field(strict=True, ge=1, le=6)]
CameraCount = Annotated[int, Field(strict=True, ge=0, le=6)]


class FusionAmbiguity(StrEnum):
    """Machine-readable reasons why a track cannot be published without caution."""

    INSUFFICIENT_OBSERVABLE_CAMERAS = "INSUFFICIENT_OBSERVABLE_CAMERAS"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
    CONTRADICTING_EVIDENCE = "CONTRADICTING_EVIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    START_BOUNDARY_UNCERTAIN = "START_BOUNDARY_UNCERTAIN"
    END_BOUNDARY_UNCERTAIN = "END_BOUNDARY_UNCERTAIN"
    QA_UNUSABLE = "QA_UNUSABLE"


class PerceptionFusionPolicy(StrictModel):
    """Versioned dynamic-denominator fusion policy."""

    version: SchemaVersion
    minimum_observable_cameras: PositiveCameraCount = 1
    minimum_supporting_cameras: PositiveCameraCount = 2
    final_confidence_threshold: UnitInterval = 0.5
    boundary_refine_threshold: UnitInterval = 0.65
    indeterminate_support_value: UnitInterval = 0.25
    default_visibility: UnitInterval = 0.5


class CameraFusionAssessment(StrictModel):
    """One camera's deterministic contribution to a fused track."""

    camera_id: CameraId
    selected: bool
    observable: bool
    supporting: bool
    contradicting: bool
    mean_reliability: UnitInterval
    mean_support_value: UnitInterval


class PerceptionFusionDecision(StrictModel):
    """One deterministic event decision after temporal reconciliation."""

    schema_version: str = "1.0"
    projection_version: str = PERCEPTION_FUSION_PROJECTION_VERSION
    fusion_key: NodeLogicalKey
    fusion_semantic_sha256: Sha256Digest
    event_id: OpaqueUuid
    source_event_track_key: NodeLogicalKey
    source_event_track_revision_semantic_sha256: Sha256Digest
    interval: NanosecondInterval
    action: str
    confidence: UnitInterval
    selected_camera_count: CameraCount
    observable_camera_count: CameraCount
    supporting_camera_count: CameraCount
    contradicting_camera_count: CameraCount
    camera_assessments: tuple[CameraFusionAssessment, ...]
    ambiguity_reasons: tuple[FusionAmbiguity, ...]
    refine_reasons: tuple[RefineReason, ...]
    policy_version: SchemaVersion
    production_eligible: bool = False

    @property
    def resolved(self) -> bool:
        return not self.ambiguity_reasons

    @property
    def requires_refinement(self) -> bool:
        return bool(self.refine_reasons)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if tuple(item.camera_id for item in self.camera_assessments) != CAMERA_IDS:
            raise ValueError("camera assessments must be ordered cam_01 through cam_06")
        if self.ambiguity_reasons != tuple(
            sorted(set(self.ambiguity_reasons), key=lambda item: item.value)
        ):
            raise ValueError("fusion ambiguity reasons must be unique and canonically ordered")
        if self.refine_reasons != tuple(
            sorted(set(self.refine_reasons), key=lambda item: item.value)
        ):
            raise ValueError("fusion refine reasons must be unique and canonically ordered")
        expected = perception_fusion_semantic_sha256(self)
        if (
            self.fusion_semantic_sha256 != expected
            or self.fusion_key != f"{PERCEPTION_FUSION_KEY_NAMESPACE}:{expected}"
        ):
            raise ValueError("fusion decision identity is inconsistent")
        expected_event_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{PERCEPTION_EVENT_UUID_NAMESPACE}:{self.source_event_track_key}",
            )
        )
        if self.event_id != expected_event_id:
            raise ValueError("event_id is not derived from the reconciled track")
        return self


def _qa_factor(disposition: ProjectedQaDisposition) -> float:
    return {
        ProjectedQaDisposition.USABLE: 1.0,
        ProjectedQaDisposition.DEGRADED: 0.65,
        ProjectedQaDisposition.UNUSABLE: 0.0,
        ProjectedQaDisposition.UNAVAILABLE: 0.0,
        ProjectedQaDisposition.UNKNOWN: 0.0,
    }[disposition]


def _relation_value(relation: CameraEvidenceRelation, policy: PerceptionFusionPolicy) -> float:
    if relation is CameraEvidenceRelation.SUPPORTS:
        return 1.0
    if relation is CameraEvidenceRelation.INDETERMINATE:
        return float(policy.indeterminate_support_value)
    return 0.0


def perception_fusion_semantic_projection(
    decision: PerceptionFusionDecision,
) -> dict[str, object]:
    return {
        "projection_version": decision.projection_version,
        "source_event_track_key": decision.source_event_track_key,
        "source_event_track_revision_semantic_sha256": (
            decision.source_event_track_revision_semantic_sha256
        ),
        "interval": decision.interval.model_dump(mode="json"),
        "action": decision.action,
        "confidence": decision.confidence,
        "selected_camera_count": decision.selected_camera_count,
        "observable_camera_count": decision.observable_camera_count,
        "supporting_camera_count": decision.supporting_camera_count,
        "contradicting_camera_count": decision.contradicting_camera_count,
        "camera_assessments": [
            item.model_dump(mode="json") for item in decision.camera_assessments
        ],
        "ambiguity_reasons": [item.value for item in decision.ambiguity_reasons],
        "refine_reasons": [item.value for item in decision.refine_reasons],
        "policy_version": decision.policy_version,
    }


def perception_fusion_semantic_sha256(decision: PerceptionFusionDecision) -> Sha256Digest:
    return semantic_sha256(perception_fusion_semantic_projection(decision))


class PerceptionFusionEngine:
    """Judge Mage-produced evidence after deterministic projection and tracking."""

    def __init__(self, policy: PerceptionFusionPolicy) -> None:
        self._policy = policy

    def fuse(
        self,
        track: EventTrackRevision,
        *,
        evidence_projections: tuple[EvidenceProjection, ...],
        qa_projections: tuple[QaProjection, ...],
    ) -> PerceptionFusionDecision:
        evidence_by_hypothesis = {
            item.hypothesis_logical_key: item
            for projection in evidence_projections
            for item in projection.event_evidence
        }
        qa_by_observation = {
            projection.source_observation_logical_key: projection for projection in qa_projections
        }

        assessments: list[CameraFusionAssessment] = []
        selected_count = observable_count = supporting_count = contradicting_count = 0
        weighted_support = 0.0
        reliability_denominator = 0.0
        any_qa_unusable = False

        for camera_id in CAMERA_IDS:
            relations: list[CameraEvidenceRelation] = []
            reliabilities: list[float] = []
            selected = False
            for reference in track.source_hypotheses:
                bundle = evidence_by_hypothesis.get(reference.hypothesis_logical_key)
                if bundle is None:
                    raise ValueError("fusion is missing evidence for an event-track hypothesis")
                fact = bundle.cameras[camera_id]
                selected = selected or fact.selected_for_inference
                if not fact.selected_for_inference:
                    continue
                qa_projection = qa_by_observation.get(reference.source_observation_logical_key)
                if qa_projection is None:
                    raise ValueError(
                        "fusion is missing QA projection for an event-track hypothesis"
                    )
                qa_disposition = qa_projection.camera_facts[camera_id].disposition
                qa_factor = _qa_factor(qa_disposition)
                if qa_disposition in {
                    ProjectedQaDisposition.UNUSABLE,
                    ProjectedQaDisposition.UNAVAILABLE,
                }:
                    any_qa_unusable = True
                if fact.relation is CameraEvidenceRelation.NOT_OBSERVABLE:
                    continue
                visibility = (
                    float(fact.visibility)
                    if fact.visibility is not None
                    else float(self._policy.default_visibility)
                )
                relations.append(fact.relation)
                reliabilities.append(max(0.0, min(1.0, visibility * qa_factor)))

            observable = bool(relations)
            supporting = CameraEvidenceRelation.SUPPORTS in relations
            contradicting = CameraEvidenceRelation.CONTRADICTS in relations
            mean_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.0
            mean_support = (
                sum(_relation_value(item, self._policy) for item in relations) / len(relations)
                if relations
                else 0.0
            )
            if selected:
                selected_count += 1
            if observable:
                observable_count += 1
                reliability_denominator += mean_reliability
                weighted_support += mean_reliability * mean_support
            if supporting:
                supporting_count += 1
            if contradicting:
                contradicting_count += 1
            assessments.append(
                CameraFusionAssessment(
                    camera_id=camera_id,
                    selected=selected,
                    observable=observable,
                    supporting=supporting,
                    contradicting=contradicting,
                    mean_reliability=mean_reliability,
                    mean_support_value=mean_support,
                )
            )

        confidence = (
            max(0.0, min(1.0, weighted_support / reliability_denominator))
            if reliability_denominator > 0
            else 0.0
        )
        ambiguity: set[FusionAmbiguity] = set()
        refine: set[RefineReason] = set()
        if observable_count < self._policy.minimum_observable_cameras:
            ambiguity.add(FusionAmbiguity.INSUFFICIENT_OBSERVABLE_CAMERAS)
        if supporting_count < self._policy.minimum_supporting_cameras:
            ambiguity.add(FusionAmbiguity.INSUFFICIENT_SUPPORT)
        if contradicting_count:
            ambiguity.add(FusionAmbiguity.CONTRADICTING_EVIDENCE)
            refine.add(RefineReason.CONFLICT)
        if confidence < self._policy.final_confidence_threshold:
            ambiguity.add(FusionAmbiguity.LOW_CONFIDENCE)
        if track.start_confidence < self._policy.boundary_refine_threshold:
            ambiguity.add(FusionAmbiguity.START_BOUNDARY_UNCERTAIN)
            refine.add(RefineReason.BOUNDARY)
        if track.end_confidence < self._policy.boundary_refine_threshold:
            ambiguity.add(FusionAmbiguity.END_BOUNDARY_UNCERTAIN)
            refine.add(RefineReason.BOUNDARY)
        if any_qa_unusable:
            ambiguity.add(FusionAmbiguity.QA_UNUSABLE)
            refine.add(RefineReason.QA)

        values = {
            "event_id": str(
                uuid5(
                    NAMESPACE_URL,
                    f"{PERCEPTION_EVENT_UUID_NAMESPACE}:{track.event_track_key}",
                )
            ),
            "source_event_track_key": track.event_track_key,
            "source_event_track_revision_semantic_sha256": track.revision_semantic_sha256,
            "interval": track.interval,
            "action": track.action,
            "confidence": confidence,
            "selected_camera_count": selected_count,
            "observable_camera_count": observable_count,
            "supporting_camera_count": supporting_count,
            "contradicting_camera_count": contradicting_count,
            "camera_assessments": tuple(assessments),
            "ambiguity_reasons": tuple(sorted(ambiguity, key=lambda item: item.value)),
            "refine_reasons": tuple(sorted(refine, key=lambda item: item.value)),
            "policy_version": self._policy.version,
            "production_eligible": False,
        }
        typed_values = cast(dict[str, Any], values)
        draft = PerceptionFusionDecision.model_construct(
            fusion_key=f"{PERCEPTION_FUSION_KEY_NAMESPACE}:{'0' * 64}",
            fusion_semantic_sha256="0" * 64,
            **typed_values,
        )
        digest = perception_fusion_semantic_sha256(draft)
        return PerceptionFusionDecision(
            fusion_key=f"{PERCEPTION_FUSION_KEY_NAMESPACE}:{digest}",
            fusion_semantic_sha256=digest,
            **typed_values,
        )


__all__ = [
    "PERCEPTION_FUSION_PROJECTION_VERSION",
    "CameraFusionAssessment",
    "FusionAmbiguity",
    "PerceptionFusionDecision",
    "PerceptionFusionEngine",
    "PerceptionFusionPolicy",
    "perception_fusion_semantic_projection",
    "perception_fusion_semantic_sha256",
]
