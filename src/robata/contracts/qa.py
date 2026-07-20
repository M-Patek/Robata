"""QA wire contracts for Architecture V1.1.

Replaces the old MVP ``robata.qa`` module with canonical V1.1 contracts
that align with the two-stage QA pipeline (Section 12) and the
``contracts/mainline.py`` models.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import NanosecondInterval, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]


class QAIssueSeverity(StrEnum):
    """Severity of a detected QA issue."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class QAStatus(StrEnum):
    """Overall QA status for a camera or recording."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class QAIssue(StrEnum):
    """Canonical QA issue taxonomy."""

    MOTION_BLUR = "MOTION_BLUR"
    SCENE_CHANGE = "SCENE_CHANGE"
    OCCLUSION = "OCCLUSION"
    HAND_OCCLUSION = "HAND_OCCLUSION"
    OBJECT_OCCLUSION = "OBJECT_OCCLUSION"
    FRAMING = "FRAMING"
    EXPOSURE = "EXPOSURE"
    CORRUPTED_STREAM = "CORRUPTED_STREAM"
    UNUSABLE_VIEW = "UNUSABLE_VIEW"
    OTHER = "OTHER"


class QAIssueClaim(StrictModel):
    """One provider-authored QA issue claim."""

    code: NonEmptyString
    interval: NanosecondInterval
    severity: QAIssueSeverity
    reported_score: UnitInterval | None = None


class CameraQAClaim(StrictModel):
    """One provider-authored camera QA observation."""

    camera_id: str
    observed_interval: NanosecondInterval
    status: QAStatus
    issues: tuple[QAIssueClaim, ...] = ()
    reported_score: UnitInterval | None = None
    frame_ordinals: tuple[int, ...] = ()


class QAOutput(StrictModel):
    """Six-camera provider QA output."""

    cameras: dict[str, CameraQAClaim]


class ClipMark(StrictModel):
    """One QA issue interval (backward-compatible alias)."""

    start_sec: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    end_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    issue: QAIssue
    confidence: UnitInterval


class QAAssessment(StrictModel):
    """Recording-level QA assessment (backward-compatible alias)."""

    recording_id: NonEmptyString
    duration_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    status: QAStatus
    clip_marks: tuple[ClipMark, ...] = ()
    effective_duration_sec: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    retained: bool
    delete_source: bool


class QAClassifier:
    """Deterministic QA classifier (skeleton)."""

    def assess(
        self,
        recording_id: str,
        duration_sec: float,
        clip_marks: tuple[ClipMark, ...] = (),
    ) -> QAAssessment:
        raise NotImplementedError("QAClassifier.assess is a skeleton.")


__all__ = [
    "CameraQAClaim",
    "ClipMark",
    "QAAssessment",
    "QAClassifier",
    "QAIssue",
    "QAIssueClaim",
    "QAIssueSeverity",
    "QAOutput",
    "QAStatus",
]
