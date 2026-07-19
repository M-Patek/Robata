"""Coarse QA: full-recording, all-six-camera screening.

This module implements the coarse stage of the two-stage QA pipeline
(Architecture V1 Section 12.1).  It covers the complete recording for all
six cameras using benchmark-selected sampling rates, producing per-camera
results and a set of suspicious intervals for the dense stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from robata.contracts.cameras import CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mainline import (
    CameraQAStatus,
    CameraQAResult,
    QAIssueSeverity,
    TemporalVisualPackage,
)
from robata.qa import QAClassifier
from robata.qa_pipeline.stages import QAStage, QAStageIssue, QAStageResult

__all__ = [
    "CameraCoarseResult",
    "CoarseQAPipeline",
    "CoarseQAResult",
    "SamplingPlan",
    "SuspiciousInterval",
]


class SamplingPlan(StrictModel):
    """Benchmark-selected sampling configuration for coarse QA.

    The sampling rate is chosen by the benchmark (Section 18) and is applied
    uniformly across the full recording duration.
    """

    target_fps: Annotated[
        float,
        Field(strict=True, gt=0.0, allow_inf_nan=False),
    ]
    frames_per_window: Annotated[int, Field(strict=True, ge=1)] = 64
    window_overlap_sec: Annotated[float, Field(strict=True, ge=0.0)] = 0.0
    policy_version: Annotated[str, Field(strict=True, min_length=1)]


class SuspiciousInterval(StrictModel):
    """One interval flagged by coarse QA as requiring dense analysis.

    The reducer (see :mod:`suspicion_reducer`) merges overlapping intervals
    and adds padding before the dense stage consumes them.
    """

    start_ns: Annotated[int, Field(strict=True)]
    end_ns: Annotated[int, Field(strict=True)]
    camera_id: CameraId
    issue_type: Annotated[str, Field(strict=True, min_length=1)]
    confidence: Annotated[
        float,
        Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
    ]


class CameraCoarseResult(StrictModel):
    """Per-camera coarse QA result.

    Wraps the canonical :class:`CameraQAResult` with additional coarse-stage
    metadata such as the window count and suspicious intervals.
    """

    camera_id: CameraId
    qa_result: CameraQAResult
    suspicious_intervals: tuple[SuspiciousInterval, ...]
    window_count: Annotated[int, Field(strict=True, ge=0)]
    package_count: Annotated[int, Field(strict=True, ge=0)]


class CoarseQAResult(StrictModel):
    """Aggregate result of the coarse QA stage for one recording.

    Contains per-camera results and the union of suspicious intervals that
    will drive the dense stage.
    """

    recording_id: OpaqueUuid
    camera_results: SixCameraMap[CameraCoarseResult]
    overall_status: Annotated[str, Field(strict=True, min_length=1)]
    suspicious_intervals: tuple[SuspiciousInterval, ...]
    duration_ns: Annotated[int, Field(strict=True, ge=0)]
    policy_version: Annotated[str, Field(strict=True, min_length=1)]


class CoarseQAPipeline:
    """Coarse QA: full-recording, all-six-camera screening.

    The pipeline runs the classifier over the complete recording, producing
    per-camera QA results and a set of suspicious intervals for the dense
    stage.  It uses the benchmark-selected sampling rate and does not
    attempt to vary density per camera at this stage.
    """

    def __init__(
        self,
        classifier: QAClassifier,
        sampler: "Sampler",  # type: ignore[name-defined]
    ) -> None:
        self.classifier = classifier
        self.sampler = sampler

    def run_coarse(
        self,
        recording: "MCAPRecording",  # type: ignore[name-defined]
        sampling_plan: SamplingPlan,
    ) -> CoarseQAResult:
        """Run coarse QA over the complete recording.

        Parameters
        ----------
        recording:
            The source MCAP recording to analyse.
        sampling_plan:
            Benchmark-selected sampling configuration.

        Returns
        -------
        CoarseQAResult:
            Per-camera results and suspicious intervals for the dense stage.
        """
        raise NotImplementedError(
            "CoarseQAPipeline.run_coarse is a skeleton; "
            "implementation requires MCAPRecording, Sampler, and inference wiring."
        )
