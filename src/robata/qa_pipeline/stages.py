"""QA pipeline stage definitions and shared models.

This module defines the core enumerations and data models used by the two-stage
QA pipeline (coarse -> dense -> final) described in Architecture V1 Section 12.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import (
    NanosecondInterval,
    StrictModel,
)

# ---------------------------------------------------------------------------
# Re-export canonical status enums from the mainline contract so pipeline
# modules can import them from a single location.
# ---------------------------------------------------------------------------
from robata.contracts.mainline import (
    CameraQAStatus,
    QAIssueSeverity,
    RecordingQAStatus,
)

__all__ = [
    "CameraQAStatus",
    "QAStage",
    "QAStageResult",
    "QAIssueSeverity",
    "RecordingQAStatus",
    "QAStageIssue",
]


class QAStage(StrEnum):
    """QA pipeline stage identifiers.

    The two-stage pipeline (coarse -> dense) is book-ended by a deterministic
    fast-detector stage and a final aggregation stage.
    """

    COARSE = "COARSE"
    """Coarse QA: full-recording, all-six-camera screening."""

    DENSE = "DENSE"
    """Dense QA: targeted high-rate analysis on suspicious intervals."""

    FINAL = "FINAL"
    """Final aggregation: camera results merged into recording-level QA."""


class QAStageIssue(StrictModel):
    """One issue detected during a QA pipeline stage.

    Issues are produced by both deterministic fast detectors and VLM-backed
    inference.  The ``producer`` field distinguishes the source.
    """

    code: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
    severity: QAIssueSeverity
    interval: NanosecondInterval
    score: Annotated[
        float | None,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = None
    confidence: Annotated[
        float | None,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ] = None
    producer: Annotated[
        str,
        StringConstraints(strict=True, min_length=1),
    ] = "FAST_DETECTOR"


class QAStageResult(StrictModel):
    """Immutable result for one QA pipeline stage execution.

    A stage result captures the outcome of a single pass (coarse, dense, or final)
    over a camera or set of cameras.  It is intentionally lightweight so that
    downstream reducers and aggregators can compose it without carrying full
    inference envelopes.
    """

    stage: QAStage
    status: CameraQAStatus
    issues: Annotated[
        tuple[QAStageIssue, ...],
        Field(default_factory=tuple),
    ]
    quality_score: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ]
    confidence: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ]

    @model_validator(mode="after")
    def _validate_stage_result(self) -> Self:
        # INCOMPLETE is only meaningful when there are no issues.
        if self.status is CameraQAStatus.INCOMPLETE and self.issues:
            raise ValueError("INCOMPLETE stage result cannot contain issues")
        return self
