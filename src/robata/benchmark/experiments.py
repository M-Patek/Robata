"""Experiment matrix definitions (Section 18.3).

Defines the benchmark experiment types: QA sampling, event proposal,
dense sampling, shadow comparison, and camera ablation.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import StrictModel
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class QAExperiment(StrictModel):
    """QA sampling experiment comparing uniform and adaptive rates.

    Tests sampling rates: 0.2, 0.5, 1, 2, 5 FPS.
    Compares coarse-only, two-stage, and full-dense policies.
    """

    experiment_id: OpaqueUuid
    name: NonEmptyString
    sampling_rates: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)
    adaptive_policies: tuple[NonEmptyString, ...] = ()
    two_stage_comparison: bool = True
    metrics: tuple[NonEmptyString, ...] = (
        "precision",
        "recall",
        "f1",
        "iou",
        "calibration",
        "cost",
    )


class EventProposalExperiment(StrictModel):
    """Event proposal experiment comparing event-rate grids.

    Tests event rates: 0.2, 0.5, 1, 2, 5 FPS.
    """

    experiment_id: OpaqueUuid
    name: NonEmptyString
    event_rates: tuple[float, ...] = (0.2, 0.5, 1.0, 2.0, 5.0)
    window_length_hop: tuple[int, int] = (5, 2)  # seconds
    overlap_threshold: UnitInterval = 0.5
    metrics: tuple[NonEmptyString, ...] = (
        "recall",
        "iou",
        "start_end_hit_rate",
        "false_candidates",
    )


class DenseSamplingExperiment(StrictModel):
    """Dense sampling and boundary refinement experiment.

    Tests dense rates: 2, 5, 10, 20 FPS.
    """

    experiment_id: OpaqueUuid
    name: NonEmptyString
    dense_rates: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
    context_padding_seconds: int = 1
    metrics: tuple[NonEmptyString, ...] = (
        "mae",
        "iou",
        "classification_accuracy",
    )


class ShadowComparisonExperiment(StrictModel):
    """Qwen vs GPT shadow comparison experiment.

    Tests shadow sample ratios: 0.01, 0.05, 0.10, 1.00.
    """

    experiment_id: OpaqueUuid
    name: NonEmptyString
    shadow_sample_ratios: tuple[float, ...] = (0.01, 0.05, 0.10, 1.0)
    hard_case_routing: bool = False
    metrics: tuple[NonEmptyString, ...] = (
        "f1",
        "latency",
        "cost",
        "disagreement",
    )


class CameraAblationExperiment(StrictModel):
    """Camera ablation experiment.

    Compares 1, 2, 3, and 6 camera conditions on the same recordings.
    """

    experiment_id: OpaqueUuid
    name: NonEmptyString
    camera_conditions: tuple[NonEmptyString, ...] = (
        "single_cam_01",
        "single_cam_02",
        "single_cam_03",
        "single_cam_04",
        "single_cam_05",
        "single_cam_06",
        "leave_one_out_cam_01",
        "leave_one_out_cam_02",
        "leave_one_out_cam_03",
        "leave_one_out_cam_04",
        "leave_one_out_cam_05",
        "leave_one_out_cam_06",
        "all_six",
    )
    metrics: tuple[NonEmptyString, ...] = (
        "per_camera",
        "leave_one_out",
        "subset_combinations",
    )


class ExperimentMatrix(StrictModel):
    """Complete experiment matrix for a benchmark run.

    Collects all experiment types under one benchmark identity.
    """

    matrix_id: OpaqueUuid
    benchmark_id: OpaqueUuid
    experiments: tuple[
        QAExperiment
        | EventProposalExperiment
        | DenseSamplingExperiment
        | ShadowComparisonExperiment
        | CameraAblationExperiment,
        ...,
    ]


__all__ = [
    "CameraAblationExperiment",
    "DenseSamplingExperiment",
    "EventProposalExperiment",
    "ExperimentMatrix",
    "QAExperiment",
    "ShadowComparisonExperiment",
]
