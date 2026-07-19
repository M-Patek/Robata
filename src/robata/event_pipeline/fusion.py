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

from robata.contracts.cameras import SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    StrictModel,
)
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mainline import (
    BoundaryRefinement,
    CameraActionClaim,
    CandidateEvent,
    NonEmptyString,
    QAResultAggregate,
    SchemaVersion,
    UnitInterval,
)


class FusionPolicy(StrictModel):
    """Versioned fusion policy controlling stage behavior and weights."""

    version: SchemaVersion
    stages: tuple[NonEmptyString, ...]
    weights: dict[NonEmptyString, UnitInterval]


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
        # Stage 1: Normalize
        # Stage 2: Assess evidence
        # Stage 3: Associate
        # Stage 4: Resolve labels
        # Stage 5: Resolve boundaries
        # Stage 6: Validate
        # Stage 7: Resolve identity
        # Stage 8: Publish
        _ = candidate
        _ = camera_evidence
        _ = boundary_evidence
        _ = qa_results
        # Skeleton: 8-stage fusion to be implemented per Section 15.
        return []


__all__ = [
    "FusionDecision",
    "FusionEngine",
    "FusionPolicy",
]
