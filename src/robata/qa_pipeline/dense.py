"""Future contract for dense QA over suspicious intervals.

This non-runnable skeleton defines the planned dense stage of the two-stage QA
pipeline (Architecture V1 Section 12.1). It will increase sampling for target
cameras while retaining synchronized context from other views.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from robata.contracts.cameras import CameraId, SixCameraMap
from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.mcap import MCAPRecording
from robata.contracts.pipeline import CameraQAResult
from robata.contracts.qa import QAClassifier
from robata.qa_pipeline.suspicion_reducer import ReducedInterval

__all__ = [
    "CameraDenseResult",
    "DenseQAPipeline",
    "DenseQAResult",
    "IntervalDenseResult",
]


class IntervalDenseResult(StrictModel):
    """Dense QA result for one reduced interval.

    Contains per-camera results for all six cameras, even those that were
    not the primary target of dense analysis.  Non-target cameras retain
    low-rate context frames.
    """

    interval_start_ns: Annotated[int, Field(strict=True)]
    interval_end_ns: Annotated[int, Field(strict=True)]
    camera_results: SixCameraMap[CameraQAResult]
    target_cameras: tuple[CameraId, ...]
    context_cameras: tuple[CameraId, ...]
    sampling_rate_fps: Annotated[
        float,
        Field(strict=True, gt=0.0, allow_inf_nan=False),
    ]
    context_sampling_rate_fps: Annotated[
        float,
        Field(strict=True, gt=0.0, allow_inf_nan=False),
    ]


class CameraDenseResult(StrictModel):
    """Per-camera result from the dense QA stage.

    Wraps the canonical :class:`CameraQAResult` with dense-stage metadata
    such as whether this camera was a target or context camera.
    """

    camera_id: CameraId
    qa_result: CameraQAResult
    is_target: bool
    dense_interval_count: Annotated[int, Field(strict=True, ge=0)]


class DenseQAResult(StrictModel):
    """Aggregate result of the dense QA stage for one recording.

    Contains per-interval results and the overall dense-stage status.
    """

    recording_id: OpaqueUuid
    interval_results: tuple[IntervalDenseResult, ...]
    overall_status: Annotated[str, Field(strict=True, min_length=1)]
    camera_results: SixCameraMap[CameraDenseResult]
    policy_version: Annotated[str, Field(strict=True, min_length=1)]


class DenseQAPipeline:
    """Future dense-QA execution contract; currently non-runnable.

    For each reduced interval, the pipeline will:
    1. Identifies target cameras (those with suspicious intervals).
    2. Increases sampling rate for target cameras.
    3. Retains low-rate synchronized context from other views.
    4. Run VLM inference on the dense package.
    5. Produce per-camera dense QA results.
    """

    def __init__(self, classifier: QAClassifier) -> None:
        self.classifier = classifier

    def run_dense(
        self,
        reduced_intervals: Sequence[ReducedInterval],
        recording: MCAPRecording,
    ) -> DenseQAResult:
        """Run dense QA over the reduced suspicious intervals.

        Parameters
        ----------
        reduced_intervals:
            Merged, padded intervals produced by the SuspiciousIntervalReducer.
        recording:
            The source MCAP recording.

        Returns
        -------
        DenseQAResult:
            Per-interval and per-camera dense QA results.
        """
        raise NotImplementedError(
            "DenseQAPipeline.run_dense is a skeleton; "
            "implementation requires MCAPRecording, QAClassifier, and inference wiring."
        )
