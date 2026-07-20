"""Alignment service-layer models.

These complement the wire-contract models in ``robata.contracts.alignment``
with service-internal and persistence-facing types.  All frozen Pydantic v2
models inherit ``StrictModel``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.alignment import (
    AlignmentMethod,
    AlignmentRun,
    AlignmentSegment,
    AlignmentStatus,
    CameraAlignment,
    CanonicalOrigin,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import Nanoseconds, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class FrameAlignmentProjection(StrictModel):
    """Immutable aligned timestamp projection for one source frame.

    Each projection is separately keyed by ``(source_frame_id, alignment_id)``
    and references its transform segment.  Re-alignment appends projections
    and never overwrites prior aligned timestamps.
    """

    projection_id: OpaqueUuid
    source_frame_id: OpaqueUuid
    alignment_id: OpaqueUuid
    camera_id: CameraId
    segment_id: NonEmptyString
    source_timestamp_ns: Nanoseconds
    aligned_timestamp_ns: Nanoseconds
    delta_to_target_ns: Nanoseconds


class AlignmentValidationMetrics(StrictModel):
    """Per-camera validation statistics produced by the alignment service."""

    camera_id: CameraId
    residual_p50_ns: NonNegativeInt
    residual_p95_ns: NonNegativeInt
    max_error_ns: NonNegativeInt
    derived_drift_ppm: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    coverage: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    gap_count: NonNegativeInt = 0
    duplicate_count: NonNegativeInt = 0
    out_of_range_count: NonNegativeInt = 0


class AlignmentValidationResult(StrictModel):
    """Aggregate validation outcome for one alignment run."""

    alignment_id: OpaqueUuid
    per_camera: tuple[AlignmentValidationMetrics, ...]
    overall_status: AlignmentStatus
    issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_six_camera_metrics(self) -> AlignmentValidationResult:
        actual = tuple(metric.camera_id for metric in self.per_camera)
        if actual != CAMERA_IDS:
            raise ValueError("alignment validation metrics must be ordered cam_01 through cam_06")
        return self


__all__ = [
    "AlignmentMethod",
    "AlignmentRun",
    "AlignmentSegment",
    "AlignmentStatus",
    "AlignmentValidationMetrics",
    "AlignmentValidationResult",
    "CameraAlignment",
    "CanonicalOrigin",
    "FrameAlignmentProjection",
]
