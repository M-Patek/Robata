"""Ground-truth annotation contracts (Section 18.2).

Defines the annotation protocol for per-camera QA issues, recording usability,
physical actions, and inter-annotator agreement.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.cameras import CameraId, SixCameraMap
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    StrictModel,
)
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class RecordingUsability(StrictModel):
    """Recording-level usability that permits partial camera degradation."""

    status: str  # USABLE, DEGRADED, UNUSABLE
    usable_camera_count: int
    reason: NonEmptyString | None = None


class QAIssueAnnotation(StrictModel):
    """One per-camera QA issue annotated by a human reviewer."""

    camera_id: CameraId
    issue_code: NonEmptyString
    interval: NanosecondInterval
    severity: str  # LOW, MEDIUM, HIGH
    confidence: UnitInterval | None = None


class CameraEvidenceAnnotation(StrictModel):
    """Ground-truth evidence annotation for one camera."""

    camera_id: CameraId
    status: str  # OBSERVED, OCCLUDED, UNUSABLE, MISSING
    event_interval: NanosecondInterval | None = None
    observed_interval: NanosecondInterval | None = None
    visibility: UnitInterval | None = None


class PhysicalActionAnnotation(StrictModel):
    """One physical action annotated on the ground-truth timeline."""

    action_id: OpaqueUuid
    action_type: NonEmptyString
    object_class: NonEmptyString | None = None
    object_instance: NonEmptyString | None = None
    active_hand: str | None = None  # LEFT, RIGHT, BOTH, NONE
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    confidence: UnitInterval | None = None
    camera_evidence: SixCameraMap[CameraEvidenceAnnotation]
    ambiguity_state: str | None = None  # e.g., "AMBIGUOUS", "CLEAR"


class BoundaryExample(StrictModel):
    """A hard-boundary example for training and evaluation."""

    action_id: OpaqueUuid
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    annotator_id: NonEmptyString
    annotation_protocol_version: NonEmptyString


class Interval(StrictModel):
    """A simple temporal interval for idle/negative regions."""

    start_ns: Nanoseconds
    end_ns: Nanoseconds


class GroundTruthAnnotation(StrictModel):
    """Complete ground-truth annotation for one MCAP recording.

    Produced with synchronized six-camera playback and canonical timestamps.
    """

    annotation_id: OpaqueUuid
    mcap_id: OpaqueUuid
    annotator_id: NonEmptyString
    annotation_protocol_version: NonEmptyString
    per_camera_qa_issues: tuple[QAIssueAnnotation, ...]
    recording_usability: RecordingUsability
    physical_actions: tuple[PhysicalActionAnnotation, ...]
    idle_negative_intervals: tuple[Interval, ...]
    boundary_examples: tuple[BoundaryExample, ...]


class InterAnnotatorAgreement(StrictModel):
    """Inter-annotator agreement metrics for a pair of annotators.

    Double-label a stratified portion and adjudicate conflicts.
    """

    agreement_id: OpaqueUuid
    annotator_pair: tuple[NonEmptyString, NonEmptyString]
    categorical_agreement: UnitInterval | None = None
    boundary_agreement: UnitInterval | None = None
    adjudicated_conflicts: tuple[OpaqueUuid, ...]


__all__ = [
    "BoundaryExample",
    "CameraEvidenceAnnotation",
    "GroundTruthAnnotation",
    "InterAnnotatorAgreement",
    "Interval",
    "PhysicalActionAnnotation",
    "QAIssueAnnotation",
    "RecordingUsability",
]
