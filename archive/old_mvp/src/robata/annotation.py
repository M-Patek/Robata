"""Provider-neutral AI pre-annotation contracts and local principal pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, cast

from pydantic import AliasChoices, Field, StringConstraints, field_validator, model_validator

from robata.contracts.common import StrictModel
from robata.frame_cache import FrameFeedManifest, FrameRef
from robata.qa import ClipMark, QAAssessment, QAStatus

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]


class StructuredLabels(StrictModel):
    """Searchable action-object labels produced by annotation principal."""

    verb: NonEmptyString
    noun: NonEmptyString
    attributes: tuple[NonEmptyString, ...] = ()
    location: NonEmptyString | None = None
    hand: NonEmptyString | None = None

    @field_validator("attributes", mode="before")
    @classmethod
    def _coerce_attributes(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Mapping):
            value = tuple(
                str(key) if item in (None, True) else f"{key}:{item}" for key, item in value.items()
            )
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("attributes must be a string, mapping, or sequence of strings")
        return tuple(value)

    @field_validator("attributes")
    @classmethod
    def _dedupe_attributes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for item in value:
            normalized = item.strip()
            if not normalized:
                raise ValueError("attributes cannot contain empty strings")
            key = normalized.casefold()
            if key not in seen:
                result.append(normalized)
                seen.add(key)
        return tuple(result)


class AnnotationSegmentDraft(StrictModel):
    """A human-reviewable action segment draft."""

    segment_id: NonEmptyString
    video_id: NonEmptyString
    start_sec: NonNegative = Field(
        validation_alias=AliasChoices("start_sec", "start_time_sec"),
        serialization_alias="start_sec",
    )
    end_sec: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)] = Field(
        validation_alias=AliasChoices("end_sec", "end_time_sec"),
        serialization_alias="end_sec",
    )
    structured_labels: StructuredLabels
    confidence: UnitInterval = 0.5
    qa_clip_marks: tuple[ClipMark, ...] = ()
    source: NonEmptyString = "annotation_principal"
    review_status: NonEmptyString = "draft"
    frame_ids: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def _validate_segment(self) -> AnnotationSegmentDraft:
        if self.end_sec <= self.start_sec:
            raise ValueError("end_sec must be greater than start_sec")
        for mark in self.qa_clip_marks:
            if mark.end_sec <= self.start_sec or mark.start_sec >= self.end_sec:
                continue
        return self

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def start_time_sec(self) -> float:
        return self.start_sec

    @property
    def end_time_sec(self) -> float:
        return self.end_sec

    @property
    def has_warning_overlap(self) -> bool:
        return any(
            mark.start_sec < self.end_sec and self.start_sec < mark.end_sec
            for mark in self.qa_clip_marks
        )

    def playback_target(self, base_uri: str | None = None) -> str:
        """Return a direct clip target suitable for a player deep link."""
        if base_uri is None:
            base_uri = self.video_id
        separator = "&" if "?" in base_uri else "?"
        return f"{base_uri}{separator}start={self.start_sec:g}&end={self.end_sec:g}"


class AnnotationBatchResult(StrictModel):
    """Batch output and explicit fail exclusion accounting."""

    drafts: tuple[AnnotationSegmentDraft, ...] = ()
    accepted_video_ids: tuple[NonEmptyString, ...] = ()
    skipped_fail_video_ids: tuple[NonEmptyString, ...] = ()
    warning_video_ids: tuple[NonEmptyString, ...] = ()

    @property
    def draft_count(self) -> int:
        return len(self.drafts)

    @property
    def results(self) -> tuple[AnnotationSegmentDraft, ...]:
        return self.drafts


@dataclass(frozen=True, slots=True)
class PrincipalContext:
    video_id: str
    duration_sec: float
    qa_clip_marks: tuple[ClipMark, ...]
    frames: tuple[FrameRef, ...] = ()


class AnnotationPrincipal(Protocol):
    """Provider-neutral boundary for a model-backed annotation principal."""

    def annotate(
        self, context: PrincipalContext
    ) -> Sequence[AnnotationSegmentDraft | Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class DeterministicAnnotationPrincipal:
    """Local deterministic principal used until a real model is approved."""

    segment_duration_sec: float = 10.0
    verb: str = "interact"
    noun: str = "object"
    attributes: tuple[str, ...] = ()
    location: str = "unknown"
    hand: str = "unknown"
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.segment_duration_sec <= 0:
            raise ValueError("segment_duration_sec must be positive")
        if not self.verb.strip() or not self.noun.strip():
            raise ValueError("verb and noun must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")

    def annotate(self, context: PrincipalContext) -> tuple[AnnotationSegmentDraft, ...]:
        if context.duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        labels = StructuredLabels(
            verb=self.verb,
            noun=self.noun,
            attributes=self.attributes,
            location=self.location,
            hand=self.hand,
        )
        result: list[AnnotationSegmentDraft] = []
        start = 0.0
        ordinal = 0
        while start < context.duration_sec:
            end = min(context.duration_sec, start + self.segment_duration_sec)
            if end <= start:
                break
            marks = tuple(
                mark
                for mark in context.qa_clip_marks
                if mark.start_sec < end and start < mark.end_sec
            )
            frames = tuple(
                frame.frame_id for frame in context.frames if start <= frame.timestamp_sec < end
            )
            result.append(
                AnnotationSegmentDraft(
                    segment_id=f"{context.video_id}:segment:{ordinal}",
                    video_id=context.video_id,
                    start_sec=start,
                    end_sec=end,
                    structured_labels=labels,
                    confidence=self.confidence,
                    qa_clip_marks=marks,
                    frame_ids=frames,
                )
            )
            ordinal += 1
            start = end
        return tuple(result)


class _CallablePrincipal:
    def __init__(self, callback: Any) -> None:
        self._callback = callback

    def annotate(
        self, context: PrincipalContext
    ) -> Sequence[AnnotationSegmentDraft | Mapping[str, Any]]:
        return cast(Sequence[AnnotationSegmentDraft | Mapping[str, Any]], self._callback(context))


class AnnotationPipeline:
    """Run annotation for every pass/warning video and never for fail videos."""

    def __init__(self, principal: AnnotationPrincipal | Any | None = None) -> None:
        selected = principal or DeterministicAnnotationPrincipal()
        self.principal: AnnotationPrincipal | _CallablePrincipal
        if callable(selected) and not hasattr(selected, "annotate"):
            self.principal = _CallablePrincipal(selected)
        else:
            self.principal = cast(AnnotationPrincipal, selected)
        if not hasattr(self.principal, "annotate") or not callable(self.principal.annotate):
            raise TypeError("principal must implement annotate(context)")

    def run(
        self,
        assessments: Iterable[QAAssessment],
        *,
        frame_manifests: Mapping[str, FrameFeedManifest] | None = None,
    ) -> AnnotationBatchResult:
        manifests = frame_manifests or {}
        drafts: list[AnnotationSegmentDraft] = []
        accepted: list[str] = []
        skipped: list[str] = []
        warning_ids: list[str] = []
        for assessment in assessments:
            if not isinstance(assessment, QAAssessment):
                if isinstance(assessment, Mapping):
                    assessment = QAAssessment.model_validate(assessment)
                else:
                    raise TypeError("assessments must contain QAAssessment values")
            if assessment.status is QAStatus.FAIL:
                skipped.append(assessment.recording_id)
                continue
            accepted.append(assessment.recording_id)
            if assessment.status is QAStatus.WARNING:
                warning_ids.append(assessment.recording_id)
            manifest = manifests.get(assessment.recording_id)
            context = PrincipalContext(
                video_id=assessment.recording_id,
                duration_sec=assessment.duration_sec,
                qa_clip_marks=assessment.clip_marks,
                frames=manifest.frames if manifest else (),
            )
            produced = self.principal.annotate(context)
            if produced is None:
                raise ValueError("annotation principal returned None")
            for item in produced:
                draft = (
                    item
                    if isinstance(item, AnnotationSegmentDraft)
                    else AnnotationSegmentDraft.model_validate(item)
                )
                if draft.video_id != assessment.recording_id:
                    raise ValueError("annotation draft video_id does not match assessment")
                # Principal output is required to carry all intersecting QA marks.  We attach any
                # omitted marks here so warning provenance cannot be silently lost downstream.
                expected = tuple(
                    mark
                    for mark in assessment.clip_marks
                    if mark.start_sec < draft.end_sec and draft.start_sec < mark.end_sec
                )
                if draft.qa_clip_marks != expected:
                    draft = draft.model_copy(update={"qa_clip_marks": expected})
                drafts.append(draft)
        return AnnotationBatchResult(
            drafts=tuple(drafts),
            accepted_video_ids=tuple(accepted),
            skipped_fail_video_ids=tuple(skipped),
            warning_video_ids=tuple(warning_ids),
        )

    process = run


PreAnnotationPipeline = AnnotationPipeline
ActionSegmentDraft = AnnotationSegmentDraft

__all__ = [
    "ActionSegmentDraft",
    "AnnotationBatchResult",
    "AnnotationPipeline",
    "AnnotationPrincipal",
    "AnnotationSegmentDraft",
    "DeterministicAnnotationPrincipal",
    "PreAnnotationPipeline",
    "PrincipalContext",
    "StructuredLabels",
]
