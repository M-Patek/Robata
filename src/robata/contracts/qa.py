"""QA wire contracts for Architecture V1.1.

Replaces the old MVP ``robata.qa`` module with canonical V1.1 contracts
that align with the two-stage QA pipeline (Section 12) and the
``contracts/pipeline.py`` models.
"""

from __future__ import annotations

from enum import StrEnum
from math import isclose
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CameraId
from robata.contracts.common import NanosecondInterval, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
NonEmptyReferences = Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

_NANOSECONDS_PER_SECOND = 1_000_000_000


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
    """Generic internal detector taxonomy."""

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


class ProductQAIssue(StrEnum):
    """Local product vocabulary matching the complete 21-item issue list."""

    BLACK_SCREEN = "BLACK_SCREEN"
    GLITCHED_SCREEN = "GLITCHED_SCREEN"
    BLURRY_LENS = "BLURRY_LENS"
    EXCESSIVE_SPEED = "EXCESSIVE_SPEED"
    EGO_DEVICE_WORN_BACKWARDS = "EGO_DEVICE_WORN_BACKWARDS"
    EGO_HAND_NOT_CENTERED = "EGO_HAND_NOT_CENTERED"
    CAMERA_STATIONARY_OVER_5S = "CAMERA_STATIONARY_OVER_5S"
    HAIR_BLOCKING_VIEW = "HAIR_BLOCKING_VIEW"
    IRRELEVANT_ACTION_PARTIAL_SEGMENT = "IRRELEVANT_ACTION_PARTIAL_SEGMENT"
    TASK_IRRELEVANT_ACTION = "TASK_IRRELEVANT_ACTION"
    ARM_HAND_OBSTRUCTED = "ARM_HAND_OBSTRUCTED"
    HAND_OVERLAP_CONTACT_CROSSING = "HAND_OVERLAP_CONTACT_CROSSING"
    INCOMPLETE_TASK = "INCOMPLETE_TASK"
    LACK_OF_DIVERSITY = "LACK_OF_DIVERSITY"
    LACK_OF_AUTHENTICITY = "LACK_OF_AUTHENTICITY"
    VIDEO_ABNORMAL_ENDING = "VIDEO_ABNORMAL_ENDING"
    TOO_DARK_OR_OVEREXPOSED = "TOO_DARK_OR_OVEREXPOSED"
    UNAUTHORIZED_PERSON_OR_ANIMAL = "UNAUTHORIZED_PERSON_OR_ANIMAL"
    REVEALING_OUTFIT = "REVEALING_OUTFIT"
    PERFORMED_OTHER_EXISTING_TASK = "PERFORMED_OTHER_EXISTING_TASK"
    OTHER = "OTHER"


class ProductQAScopeKind(StrEnum):
    """Evidence scope retained before simplifying an issue to a clip mark."""

    CAMERA_INTERVAL = "CAMERA_INTERVAL"
    CAMERA_RECORDING = "CAMERA_RECORDING"
    TASK_INTERVAL = "TASK_INTERVAL"
    TASK_RECORDING = "TASK_RECORDING"
    CROSS_RECORDING_SEQUENCE = "CROSS_RECORDING_SEQUENCE"


class ProductQAConfidenceKind(StrEnum):
    """Origin of a local score; none of these values claims calibration."""

    DETECTOR_REPORTED = "DETECTOR_REPORTED"
    MODEL_REPORTED = "MODEL_REPORTED"
    POLICY_DERIVED = "POLICY_DERIVED"


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
    """Simplified local product projection for one QA issue interval."""

    start_sec: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    end_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    issue: ProductQAIssue
    confidence: UnitInterval

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.start_sec >= self.end_sec:
            raise ValueError("start_sec must be less than end_sec")
        return self


class ProductQAEvidenceScope(StrictModel):
    """Authoritative local subjects to which one product issue applies."""

    kind: ProductQAScopeKind
    subject_refs: NonEmptyReferences
    camera_id: CameraId | None = None

    @model_validator(mode="after")
    def validate_camera_scope(self) -> Self:
        if len(set(self.subject_refs)) != len(self.subject_refs):
            raise ValueError("subject_refs must be unique")
        if any(not reference.strip() for reference in self.subject_refs):
            raise ValueError("subject_refs must not contain blank values")
        if (
            self.kind
            in {
                ProductQAScopeKind.CAMERA_INTERVAL,
                ProductQAScopeKind.CAMERA_RECORDING,
            }
            and self.camera_id is None
        ):
            raise ValueError("camera evidence scopes require camera_id")
        return self


class ProductQAIssueEvidence(StrictModel):
    """Rich local issue evidence retained behind the simplified product view."""

    issue: ProductQAIssue
    scope: ProductQAEvidenceScope
    interval: NanosecondInterval | None = None
    confidence: UnitInterval
    confidence_kind: ProductQAConfidenceKind
    evidence_refs: NonEmptyReferences
    note: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("evidence_refs must be unique")
        if any(not reference.strip() for reference in self.evidence_refs):
            raise ValueError("evidence_refs must not contain blank values")
        if (
            self.scope.kind
            in {
                ProductQAScopeKind.CAMERA_INTERVAL,
                ProductQAScopeKind.TASK_INTERVAL,
            }
            and self.interval is None
        ):
            raise ValueError("interval evidence scopes require an interval")
        if (
            self.scope.kind is ProductQAScopeKind.CROSS_RECORDING_SEQUENCE
            and self.interval is not None
        ):
            raise ValueError("cross-recording evidence cannot claim one recording interval")
        if self.interval is not None and self.interval.start_ns < 0:
            raise ValueError("local product intervals must start at or after recording time zero")
        if self.issue is ProductQAIssue.OTHER and (self.note is None or not self.note.strip()):
            raise ValueError("OTHER issue evidence requires a nonempty note")
        return self

    def to_clip_mark(self, *, recording_duration_ns: int | None = None) -> ClipMark:
        """Project evidence to the intentionally small local product shape."""

        interval = self.interval
        if interval is None:
            if self.scope.kind not in {
                ProductQAScopeKind.CAMERA_RECORDING,
                ProductQAScopeKind.TASK_RECORDING,
            }:
                raise ValueError("temporal interval is required for ClipMark projection")
            if recording_duration_ns is None or recording_duration_ns <= 0:
                raise ValueError("recording_duration_ns is required for recording scope")
            start_sec = 0.0
            end_sec = recording_duration_ns / _NANOSECONDS_PER_SECOND
        else:
            if recording_duration_ns is not None and interval.end_ns > recording_duration_ns:
                raise ValueError("issue interval must lie within recording_duration_ns")
            start_sec = interval.start_ns / _NANOSECONDS_PER_SECOND
            end_sec = interval.end_ns / _NANOSECONDS_PER_SECOND
        return ClipMark(
            start_sec=start_sec,
            end_sec=end_sec,
            issue=self.issue,
            confidence=self.confidence,
        )


_RECORDING_FAILURE_ISSUES = frozenset(
    {
        ProductQAIssue.EGO_DEVICE_WORN_BACKWARDS,
        ProductQAIssue.BLURRY_LENS,
        ProductQAIssue.INCOMPLETE_TASK,
        ProductQAIssue.LACK_OF_AUTHENTICITY,
        ProductQAIssue.REVEALING_OUTFIT,
    }
)
_FULL_COVERAGE_FAILURE_ISSUES = frozenset(
    {
        ProductQAIssue.BLACK_SCREEN,
        ProductQAIssue.TOO_DARK_OR_OVEREXPOSED,
    }
)


def _ordered_clip_marks(clip_marks: tuple[ClipMark, ...]) -> tuple[ClipMark, ...]:
    return tuple(
        sorted(
            clip_marks,
            key=lambda mark: (
                mark.start_sec,
                mark.end_sec,
                mark.issue.value,
                mark.confidence,
            ),
        )
    )


def _merged_intervals(clip_marks: tuple[ClipMark, ...]) -> tuple[tuple[float, float], ...]:
    if not clip_marks:
        return ()

    intervals = sorted((mark.start_sec, mark.end_sec) for mark in clip_marks)
    merged: list[tuple[float, float]] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return tuple(merged)


def _effective_duration_sec(
    duration_sec: float,
    clip_marks: tuple[ClipMark, ...],
) -> float:
    invalid_duration = sum(end - start for start, end in _merged_intervals(clip_marks))
    return max(0.0, duration_sec - invalid_duration)


def _covers_recording(duration_sec: float, clip_marks: tuple[ClipMark, ...]) -> bool:
    merged = _merged_intervals(clip_marks)
    return len(merged) == 1 and merged[0] == (0.0, duration_sec)


def _local_status(
    duration_sec: float,
    clip_marks: tuple[ClipMark, ...],
    unprojected_issue_count: int = 0,
) -> QAStatus:
    if not clip_marks:
        return QAStatus.WARNING if unprojected_issue_count else QAStatus.PASS
    if any(mark.issue in _RECORDING_FAILURE_ISSUES for mark in clip_marks):
        return QAStatus.FAIL
    for issue in _FULL_COVERAGE_FAILURE_ISSUES:
        issue_marks = tuple(mark for mark in clip_marks if mark.issue is issue)
        if _covers_recording(duration_sec, issue_marks):
            return QAStatus.FAIL
    return QAStatus.WARNING


class QAAssessment(StrictModel):
    """Local recording assessment; it never authorizes source deletion."""

    recording_id: NonEmptyString
    duration_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    status: QAStatus
    clip_marks: tuple[ClipMark, ...] = ()
    unprojected_issue_count: NonNegativeInt = 0
    effective_duration_sec: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    retained: Literal[True] = True
    delete_source: Literal[False] = False
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        if any(mark.end_sec > self.duration_sec for mark in self.clip_marks):
            raise ValueError("clip marks must lie within duration_sec")

        expected_duration = _effective_duration_sec(self.duration_sec, self.clip_marks)
        if not isclose(
            self.effective_duration_sec,
            expected_duration,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("effective_duration_sec must subtract the interval union")

        expected_status = _local_status(
            self.duration_sec,
            self.clip_marks,
            self.unprojected_issue_count,
        )
        if self.status is not expected_status:
            raise ValueError("status must match the local QA disposition policy")
        return self


class LocalQARecordingResult(StrictModel):
    """Rich local result retaining evidence omitted from product clip marks."""

    assessment: QAAssessment
    issue_evidence: tuple[ProductQAIssueEvidence, ...] = ()
    unprojected_issue_evidence: tuple[ProductQAIssueEvidence, ...] = ()
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_evidence_partition(self) -> Self:
        if self.assessment.unprojected_issue_count != len(self.unprojected_issue_evidence):
            raise ValueError("assessment must count every unprojected issue")
        if any(issue not in self.issue_evidence for issue in self.unprojected_issue_evidence):
            raise ValueError("unprojected issues must also be retained in issue_evidence")
        return self


class QAClassifier:
    """Deterministic local-only product classifier."""

    def assess(
        self,
        recording_id: str,
        duration_sec: float,
        clip_marks: tuple[ClipMark, ...] = (),
    ) -> QAAssessment:
        ordered_marks = _ordered_clip_marks(clip_marks)
        return QAAssessment(
            recording_id=recording_id,
            duration_sec=duration_sec,
            status=_local_status(duration_sec, ordered_marks),
            clip_marks=ordered_marks,
            effective_duration_sec=_effective_duration_sec(duration_sec, ordered_marks),
        )

    def assess_evidence(
        self,
        recording_id: str,
        duration_ns: int,
        issues: tuple[ProductQAIssueEvidence, ...] = (),
    ) -> LocalQARecordingResult:
        """Produce clip marks while retaining evidence that has no clip projection."""

        if isinstance(duration_ns, bool) or duration_ns <= 0:
            raise ValueError("duration_ns must be a positive integer")
        projected_marks: list[ClipMark] = []
        unprojected_issues: list[ProductQAIssueEvidence] = []
        for issue in issues:
            if issue.scope.kind is ProductQAScopeKind.CROSS_RECORDING_SEQUENCE:
                unprojected_issues.append(issue)
                continue
            projected_marks.append(issue.to_clip_mark(recording_duration_ns=duration_ns))

        duration_sec = duration_ns / _NANOSECONDS_PER_SECOND
        ordered_marks = _ordered_clip_marks(tuple(projected_marks))
        assessment = QAAssessment(
            recording_id=recording_id,
            duration_sec=duration_sec,
            status=_local_status(
                duration_sec,
                ordered_marks,
                len(unprojected_issues),
            ),
            clip_marks=ordered_marks,
            unprojected_issue_count=len(unprojected_issues),
            effective_duration_sec=_effective_duration_sec(duration_sec, ordered_marks),
        )
        return LocalQARecordingResult(
            assessment=assessment,
            issue_evidence=issues,
            unprojected_issue_evidence=tuple(unprojected_issues),
        )


__all__ = [
    "CameraQAClaim",
    "ClipMark",
    "LocalQARecordingResult",
    "ProductQAConfidenceKind",
    "ProductQAEvidenceScope",
    "ProductQAIssue",
    "ProductQAIssueEvidence",
    "ProductQAScopeKind",
    "QAAssessment",
    "QAClassifier",
    "QAIssue",
    "QAIssueClaim",
    "QAIssueSeverity",
    "QAOutput",
    "QAStatus",
]
