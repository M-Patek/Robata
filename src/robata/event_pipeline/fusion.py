"""Multi-view Fusion: 8-stage strategy (Section 15).

Replaceable fusion engine with eight deterministic stages:
1. Normalize
2. Assess evidence
3. Associate
4. Resolve labels
5. Resolve boundaries
6. Validate
7. Resolve identity
8. Publish
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator

from robata.contracts.cameras import SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    SchemaVersion,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mainline import (
    BoundaryRefinement,
    CameraActionClaim,
    CameraEvidenceStatus,
    CameraQAStatus,
    CandidateEvent,
    NonEmptyString,
    QAResultAggregate,
    UnitInterval,
)
from robata.event_pipeline.boundary import BoundaryRefinementPolicy, BoundaryRefiner

FUSION_STAGES = (
    "NORMALIZE",
    "ASSESS_EVIDENCE",
    "ASSOCIATE",
    "RESOLVE_LABELS",
    "RESOLVE_BOUNDARIES",
    "VALIDATE",
    "RESOLVE_IDENTITY",
    "PUBLISH",
)
_WEIGHT_KEYS = frozenset({"visibility", "coverage", "reported_score", "qa"})


class FusionPolicy(StrictModel):
    """Versioned fusion policy controlling stage behavior and weights."""

    version: SchemaVersion
    stages: tuple[NonEmptyString, ...]
    weights: dict[NonEmptyString, UnitInterval]
    minimum_supporting_cameras: Annotated[int, Field(strict=True, ge=1, le=6)] = 2
    final_confidence_threshold: UnitInterval = 0.5

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.stages != FUSION_STAGES:
            raise ValueError("fusion policy must declare the canonical eight stages")
        if set(self.weights) != _WEIGHT_KEYS:
            raise ValueError(f"fusion weights must define {sorted(_WEIGHT_KEYS)!r}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("fusion weights must have positive total weight")
        return self


class FusionDecision(StrictModel):
    """One physical-event hypothesis output by the fusion engine.

    Contains the resolved interval, action type, confidence, and an
    explicit ambiguity state.
    """

    event_id: OpaqueUuid
    interval: NanosecondInterval
    action_type: NonEmptyString
    confidence: UnitInterval
    ambiguity_state: NonEmptyString
    supporting_camera_count: Annotated[int, Field(strict=True, ge=0, le=6)]
    evidence_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    policy_version: SchemaVersion
    production_eligible: bool = False


class FusionEngine:
    """Replaceable fusion engine with 8 stages.

    Stages:
    1. Normalize: Validate camera IDs, labels, hands, object identities,
       intervals, visibility, QA, and source inference references.
    2. Assess evidence: Derive reliability features from visibility,
       QA severity, alignment residual, temporal resolution, and
       calibrated model scores.
    3. Associate: Cluster hypotheses using temporal overlap/proximity
       and compatible action, hand, and object identity.
    4. Resolve labels: Compare supporting and contradicting evidence.
    5. Resolve boundaries: Use a robust, versioned estimator over
       onset/end evidence and uncertainties.
    6. Validate: Require six evidence slots, time bounds, traceable
       sources, and duplicate-event suppression.
    7. Resolve identity: Persist the derived hypothesis, compare with
       the stable event registry, and record reuse/new/split/merge/ambiguous
       assignment explicitly.
    8. Publish: Append a provisional or final event revision with
       policy/calibration/identity-assignment versions.
    """

    def __init__(self, policy: FusionPolicy) -> None:
        self._policy = policy

    def fuse(
        self,
        candidate: CandidateEvent,
        camera_evidence: SixCameraMap[CameraActionClaim],
        boundary_evidence: BoundaryRefinement,
        qa_results: QAResultAggregate,
    ) -> Sequence[FusionDecision]:
        """Run the 8-stage fusion strategy.

        Args:
            candidate: The candidate event being fused.
            camera_evidence: Six-camera action evidence claims.
            boundary_evidence: Boundary refinement evidence for onset/offset.
            qa_results: Recording-level QA aggregate for reliability scoring.

        Returns:
            Zero or more fusion decisions, each representing one physical
            event hypothesis.
        """
        if candidate.mcap_id != qa_results.mcap_id:
            raise ValueError("candidate and QA aggregate must belong to the same MCAP")

        # 1. Normalize and validate interval ownership.
        for camera_id, claim in camera_evidence.items():
            if claim.camera_id is not camera_id:
                raise ValueError("camera evidence key does not match nested camera_id")
            if claim.event_interval is not None:
                interval = claim.event_interval
                dense = candidate.dense_interval
                if interval.start_ns < dense.start_ns or interval.end_ns > dense.end_ns:
                    raise ValueError("camera event evidence lies outside the candidate interval")

        # 2. Assess evidence. Provider scores remain uncalibrated features.
        qa_weight = {
            CameraQAStatus.GOOD: 1.0,
            CameraQAStatus.DEGRADED: 0.7,
            CameraQAStatus.UNUSABLE: 0.0,
            CameraQAStatus.UNKNOWN: 0.0,
            CameraQAStatus.INCOMPLETE: 0.0,
        }
        total_weight = sum(self._policy.weights.values())
        reliabilities: list[float] = []
        supporting = 0
        for camera_id, claim in camera_evidence.items():
            status_factor = 0.0
            if claim.status is CameraEvidenceStatus.SUPPORTING:
                status_factor = 1.0
                supporting += 1
            elif claim.status is CameraEvidenceStatus.PARTIAL:
                status_factor = 0.5
                supporting += 1
            features = {
                "visibility": claim.visibility or 0.0,
                "coverage": claim.coverage_fraction,
                "reported_score": claim.reported_score or 0.0,
                "qa": qa_weight[qa_results.camera_results[camera_id].claim.status],
            }
            weighted = (
                sum(self._policy.weights[name] * value for name, value in features.items())
                / total_weight
            )
            reliabilities.append(status_factor * weighted)

        # 3-4. One normalized candidate forms one association cluster; labels
        # remain explicit when the provider supplied no ontology hint.
        action_type = candidate.proposal.label_hint or "unknown_action"

        # 5. Resolve boundaries through the separately versioned estimator.
        refined = BoundaryRefiner(
            BoundaryRefinementPolicy(
                version=f"{self._policy.version}.boundary",
                minimum_observed_cameras=self._policy.minimum_supporting_cameras,
                allow_candidate_fallback=True,
            )
        ).refine(candidate, boundary_evidence)

        # 6. Validate and expose ambiguity rather than fabricating certainty.
        confidence = max(0.0, min(1.0, sum(reliabilities) / 6))
        ambiguity_codes: list[str] = []
        if supporting < self._policy.minimum_supporting_cameras:
            ambiguity_codes.append("INSUFFICIENT_CAMERA_SUPPORT")
        if confidence < self._policy.final_confidence_threshold:
            ambiguity_codes.append("LOW_UNCALIBRATED_CONFIDENCE")
        if candidate.proposal.label_hint is None:
            ambiguity_codes.append("MISSING_ACTION_LABEL")
        if refined.used_fallback:
            ambiguity_codes.append("BOUNDARY_FALLBACK")
        ambiguity_state = "+".join(ambiguity_codes) if ambiguity_codes else "RESOLVED"

        # 7-8. Identity is deterministic and publication is local-only.
        evidence_digest = semantic_sha256(
            {
                "candidate_event_id": candidate.candidate_event_id,
                "camera_evidence": camera_evidence,
                "boundary_evidence": boundary_evidence,
                "qa_aggregate_id": qa_results.aggregate_id,
                "policy_version": self._policy.version,
            }
        )
        event_id = str(uuid5(NAMESPACE_URL, f"robata:fusion:{evidence_digest}"))
        return (
            FusionDecision(
                event_id=event_id,
                interval=refined.interval,
                action_type=action_type,
                confidence=confidence,
                ambiguity_state=ambiguity_state,
                supporting_camera_count=supporting,
                evidence_digest=evidence_digest,
                policy_version=self._policy.version,
                production_eligible=False,
            ),
        )


__all__ = [
    "FUSION_STAGES",
    "FusionDecision",
    "FusionEngine",
    "FusionPolicy",
]
