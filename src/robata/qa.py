"""Provider-neutral quality assessment contracts and policy.

The production QA boundary is deliberately independent from a model provider.  A detector (or
human reviewer) emits time-local :class:`ClipMark` records and :class:`QAAssessment` applies the
requirements taxonomy to derive ``pass``/``warning``/``fail``.  The source recording is never
mutated; invalid intervals are retained as provenance and downstream consumers decide whether to
skip or down-weight them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Annotated, ClassVar

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFinite = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


def _normalize_issue_text(value: str) -> str:
    return " ".join(
        value.strip().replace("_", " ").replace("-", " ").replace("/", " ").split()
    ).casefold()


class QAStatus(StrEnum):
    """Video-level disposition required by the production workflow."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class QAIssue(StrEnum):
    """Canonical 21-item QA vocabulary from REQUIREMENTS.md."""

    BLACK_SCREEN = "Black Screen"
    GLITCHED_SCREEN = "Glitched Screen"
    BLURRY_LENS = "Blurry Lens"
    EXCESSIVE_SPEED = "Excessive Speed"
    DEVICE_WORN_BACKWARDS = "Ego - Device worn backwards"
    HAND_NOT_CENTERED = "Ego - Hand not centered in frame"
    CAMERA_STATIONARY_OVER_5S = "Camera stationary for more than 5s"
    HAIR_BLOCKING_VIEW = "Hair blocking view"
    IRRELEVANT_ACTIONS_PARTIAL = "Irrelevant actions in partial segments"
    TASK_IRRELEVANT_ACTIONS = "Task irrelevant actions"
    ARM_HAND_OBSTRUCTED = "Arm/Hand obstructed"
    HAND_OVERLAP_CROSSING = "Hand overlap / contact / crossing"
    INCOMPLETE_TASK = "Incomplete task"
    LACK_OF_DIVERSITY = "Lack of diversity"
    LACK_OF_AUTHENTICITY = "Lack of authenticity"
    VIDEO_ABNORMALLY_ENDING = "Video Abnormally Ending"
    TOO_DARK_OVEREXPOSED = "Too Dark / Overexposed"
    UNAUTHORIZED_PERSON_ANIMAL = "Unauthorized Person/Animal Entering Frame"
    REVEALING_OUTFIT = "Revealing outfit"
    OTHER_EXISTING_TASK = "Performed other existing Tasks"
    OTHER = "Other (please specify)"

    @classmethod
    def parse(cls, value: QAIssue | str) -> QAIssue:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("issue must be a QAIssue or string")
        normalized = _normalize_issue_text(value)
        for issue in cls:
            if normalized in {
                _normalize_issue_text(issue.value),
                _normalize_issue_text(issue.name),
            }:
                return issue
        aliases = {
            "black screen": cls.BLACK_SCREEN,
            "glitch": cls.GLITCHED_SCREEN,
            "glitched screen": cls.GLITCHED_SCREEN,
            "blur": cls.BLURRY_LENS,
            "blurry lens": cls.BLURRY_LENS,
            "device worn backwards": cls.DEVICE_WORN_BACKWARDS,
            "ego device worn backwards": cls.DEVICE_WORN_BACKWARDS,
            "hand not centered": cls.HAND_NOT_CENTERED,
            "stationary >5s": cls.CAMERA_STATIONARY_OVER_5S,
            "camera stationary >5s": cls.CAMERA_STATIONARY_OVER_5S,
            "too dark overexposed": cls.TOO_DARK_OVEREXPOSED,
            "unauthorized person animal": cls.UNAUTHORIZED_PERSON_ANIMAL,
            "other existing task": cls.OTHER_EXISTING_TASK,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown QA issue: {value!r}") from exc


class IssueDisposition(StrEnum):
    """How an issue participates in video-level QA."""

    LOCAL_WARNING = "local_warning"
    WHOLE_RECORDING_FAIL = "whole_recording_fail"
    CROSS_VIDEO_REVIEW = "cross_video_review"
    TAIL_REPAIR = "tail_repair"
    CONTEXTUAL_REVIEW = "contextual_review"


# A stable policy table: local defects are retained as warning marks, while recording-level
# compliance/completeness defects fail.  Cross-video issues are represented but do not force a
# destructive decision in this per-video stage.
ISSUE_DISPOSITION: Mapping[QAIssue, IssueDisposition] = {
    QAIssue.BLACK_SCREEN: IssueDisposition.LOCAL_WARNING,
    QAIssue.GLITCHED_SCREEN: IssueDisposition.LOCAL_WARNING,
    QAIssue.BLURRY_LENS: IssueDisposition.WHOLE_RECORDING_FAIL,
    QAIssue.EXCESSIVE_SPEED: IssueDisposition.LOCAL_WARNING,
    QAIssue.DEVICE_WORN_BACKWARDS: IssueDisposition.WHOLE_RECORDING_FAIL,
    QAIssue.HAND_NOT_CENTERED: IssueDisposition.LOCAL_WARNING,
    QAIssue.CAMERA_STATIONARY_OVER_5S: IssueDisposition.LOCAL_WARNING,
    QAIssue.HAIR_BLOCKING_VIEW: IssueDisposition.LOCAL_WARNING,
    QAIssue.IRRELEVANT_ACTIONS_PARTIAL: IssueDisposition.LOCAL_WARNING,
    QAIssue.TASK_IRRELEVANT_ACTIONS: IssueDisposition.LOCAL_WARNING,
    QAIssue.ARM_HAND_OBSTRUCTED: IssueDisposition.LOCAL_WARNING,
    QAIssue.HAND_OVERLAP_CROSSING: IssueDisposition.LOCAL_WARNING,
    QAIssue.INCOMPLETE_TASK: IssueDisposition.WHOLE_RECORDING_FAIL,
    QAIssue.LACK_OF_DIVERSITY: IssueDisposition.CROSS_VIDEO_REVIEW,
    QAIssue.LACK_OF_AUTHENTICITY: IssueDisposition.WHOLE_RECORDING_FAIL,
    QAIssue.VIDEO_ABNORMALLY_ENDING: IssueDisposition.TAIL_REPAIR,
    QAIssue.TOO_DARK_OVEREXPOSED: IssueDisposition.LOCAL_WARNING,
    QAIssue.UNAUTHORIZED_PERSON_ANIMAL: IssueDisposition.LOCAL_WARNING,
    QAIssue.REVEALING_OUTFIT: IssueDisposition.WHOLE_RECORDING_FAIL,
    QAIssue.OTHER_EXISTING_TASK: IssueDisposition.CROSS_VIDEO_REVIEW,
    QAIssue.OTHER: IssueDisposition.CONTEXTUAL_REVIEW,
}


class ClipMark(StrictModel):
    """One QA issue interval in seconds.

    ``issue`` intentionally carries the canonical issue string on the wire so the contract can be
    consumed by systems that do not import Python enums.  Parsing still accepts enum members and
    common aliases.
    """

    start_sec: NonNegativeFinite
    end_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    issue: QAIssue
    confidence: UnitInterval

    @field_validator("issue", mode="before")
    @classmethod
    def _parse_issue(cls, value: QAIssue | str) -> QAIssue:
        return QAIssue.parse(value)

    @model_validator(mode="after")
    def validate_interval(self) -> ClipMark:
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        return self

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def disposition(self) -> IssueDisposition:
        return ISSUE_DISPOSITION[self.issue]


class QAAssessment(StrictModel):
    """Immutable recording-level QA result with all clip marks retained."""

    recording_id: NonEmptyString
    duration_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    status: QAStatus
    clip_marks: tuple[ClipMark, ...] = ()
    effective_duration_sec: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    retained: bool
    delete_source: bool
    cross_video_review_required: bool = False
    source_immutable: bool = True

    @model_validator(mode="after")
    def validate_result(self) -> QAAssessment:
        if self.effective_duration_sec > self.duration_sec + 1e-9:
            raise ValueError("effective_duration_sec cannot exceed duration_sec")
        expected_retained = self.status is not QAStatus.FAIL
        if self.retained != expected_retained:
            raise ValueError("only fail recordings may be deleted")
        if self.delete_source == self.retained:
            raise ValueError("delete_source must be true only for fail recordings")
        if self.status is QAStatus.PASS and self.clip_marks:
            raise ValueError("pass recordings cannot contain clip marks")
        if self.status is QAStatus.FAIL and not self.delete_source:
            raise ValueError("fail recordings must be marked for deletion")
        return self

    @property
    def warning_marks(self) -> tuple[ClipMark, ...]:
        return self.clip_marks if self.status is QAStatus.WARNING else ()


@dataclass(frozen=True, slots=True)
class QAClassifier:
    """Deterministic policy evaluator used by fake/local detectors and tests."""

    full_coverage_tolerance_sec: float = 0.05
    minimum_duration_sec: float = 1e-6
    taxonomy: ClassVar[Mapping[QAIssue, IssueDisposition]] = ISSUE_DISPOSITION

    def __post_init__(self) -> None:
        if not isfinite(self.full_coverage_tolerance_sec) or self.full_coverage_tolerance_sec < 0:
            raise ValueError("full_coverage_tolerance_sec must be finite and non-negative")
        if not isfinite(self.minimum_duration_sec) or self.minimum_duration_sec <= 0:
            raise ValueError("minimum_duration_sec must be finite and positive")

    def assess(
        self,
        recording_id: str,
        duration_sec: float,
        clip_marks: Iterable[ClipMark | Mapping[str, object]] = (),
    ) -> QAAssessment:
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ValueError("recording_id must be a non-empty string")
        if isinstance(duration_sec, bool) or not isinstance(duration_sec, (int, float)):
            raise TypeError("duration_sec must be numeric")
        duration = float(duration_sec)
        if not isfinite(duration) or duration <= 0:
            raise ValueError("duration_sec must be finite and positive")
        marks = tuple(self._coerce_mark(mark, duration) for mark in clip_marks)
        # Preserve every mark in deterministic order; sorting avoids provider-order drift while
        # retaining duplicate intervals/issues (all evidence is valuable for review).
        marks = tuple(
            sorted(
                marks,
                key=lambda mark: (mark.start_sec, mark.end_sec, mark.issue.value, -mark.confidence),
            )
        )
        full_recording_fail = any(
            mark.disposition is IssueDisposition.WHOLE_RECORDING_FAIL for mark in marks
        )
        for issue in {mark.issue for mark in marks}:
            if issue in {
                QAIssue.BLACK_SCREEN,
                QAIssue.TOO_DARK_OVEREXPOSED,
            } and self._covers_duration(marks, issue, duration):
                full_recording_fail = True
        status = (
            QAStatus.FAIL if full_recording_fail else QAStatus.WARNING if marks else QAStatus.PASS
        )
        invalid_duration = _union_duration(marks)
        effective = max(0.0, duration - invalid_duration)
        cross_review = any(
            mark.disposition
            in {IssueDisposition.CROSS_VIDEO_REVIEW, IssueDisposition.CONTEXTUAL_REVIEW}
            for mark in marks
        )
        return QAAssessment(
            recording_id=recording_id,
            duration_sec=duration,
            status=status,
            clip_marks=marks,
            effective_duration_sec=effective,
            retained=status is not QAStatus.FAIL,
            delete_source=status is QAStatus.FAIL,
            cross_video_review_required=cross_review,
        )

    def _coerce_mark(self, mark: ClipMark | Mapping[str, object], duration: float) -> ClipMark:
        if isinstance(mark, ClipMark):
            value = mark
        elif isinstance(mark, Mapping):
            payload = dict(mark)
            payload["issue"] = QAIssue.parse(payload.get("issue"))  # type: ignore[arg-type]
            value = ClipMark.model_validate(payload)
        else:
            raise TypeError("clip_marks must contain ClipMark or mapping values")
        if value.end_sec > duration + self.full_coverage_tolerance_sec:
            raise ValueError("clip mark end_sec cannot exceed recording duration")
        if value.start_sec >= duration:
            raise ValueError("clip mark must overlap recording duration")
        # Clamp tiny detector overrun to the source duration while preserving strict interval rules.
        if value.end_sec > duration:
            value = value.model_copy(update={"end_sec": duration})
        return value

    def _covers_duration(self, marks: Sequence[ClipMark], issue: QAIssue, duration: float) -> bool:
        intervals = sorted(
            (mark.start_sec, min(mark.end_sec, duration)) for mark in marks if mark.issue is issue
        )
        if not intervals:
            return False
        cursor = 0.0
        for start, end in intervals:
            if start > cursor + self.full_coverage_tolerance_sec:
                return False
            cursor = max(cursor, end)
            if cursor >= duration - self.full_coverage_tolerance_sec:
                return True
        return cursor >= duration - self.full_coverage_tolerance_sec


def _union_duration(marks: Sequence[ClipMark]) -> float:
    if not marks:
        return 0.0
    intervals = sorted((mark.start_sec, mark.end_sec) for mark in marks)
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += end - start
        start, end = next_start, next_end
    return total + end - start


# Group names mirror the vocabulary used by the source QA UI.
QA_ISSUE_GROUPS: Mapping[str, tuple[QAIssue, ...]] = {
    "Device Issues": (QAIssue.BLACK_SCREEN, QAIssue.GLITCHED_SCREEN, QAIssue.BLURRY_LENS),
    "Collector Operation Issues": (
        QAIssue.EXCESSIVE_SPEED,
        QAIssue.DEVICE_WORN_BACKWARDS,
        QAIssue.HAND_NOT_CENTERED,
        QAIssue.CAMERA_STATIONARY_OVER_5S,
        QAIssue.HAIR_BLOCKING_VIEW,
        QAIssue.IRRELEVANT_ACTIONS_PARTIAL,
        QAIssue.TASK_IRRELEVANT_ACTIONS,
        QAIssue.ARM_HAND_OBSTRUCTED,
        QAIssue.HAND_OVERLAP_CROSSING,
        QAIssue.INCOMPLETE_TASK,
        QAIssue.LACK_OF_DIVERSITY,
        QAIssue.LACK_OF_AUTHENTICITY,
        QAIssue.VIDEO_ABNORMALLY_ENDING,
    ),
    "Environmental Issues": (
        QAIssue.TOO_DARK_OVEREXPOSED,
        QAIssue.UNAUTHORIZED_PERSON_ANIMAL,
        QAIssue.REVEALING_OUTFIT,
    ),
    "Task Set Issues": (QAIssue.OTHER_EXISTING_TASK,),
    "Others": (QAIssue.OTHER,),
}

# Friendly aliases for callers that use the terminology from the requirements document.
QAResult = QAAssessment
ClipQAResult = QAAssessment
QualityAssessment = QAClassifier
QAIssueType = QAIssue
VideoQAStatus = QAStatus
ClipQAStatus = QAStatus

__all__ = [
    "ISSUE_DISPOSITION",
    "QA_ISSUE_GROUPS",
    "ClipMark",
    "ClipQAResult",
    "ClipQAStatus",
    "IssueDisposition",
    "QAAssessment",
    "QAClassifier",
    "QAIssue",
    "QAIssueType",
    "QAResult",
    "QAStatus",
    "QualityAssessment",
    "VideoQAStatus",
]
