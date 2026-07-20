"""Exact timestamp-alignment primitives."""

from robata.alignment.models import (
    AlignmentMethod,
    AlignmentRun,
    AlignmentSegment,
    AlignmentStatus,
    AlignmentValidationMetrics,
    AlignmentValidationResult,
    CameraAlignment,
    CanonicalOrigin,
    FrameAlignmentProjection,
)
from robata.alignment.rational_time import (
    PiecewiseAlignment,
    RationalTransformSegment,
    round_half_even,
)
from robata.alignment.service import (
    AlignmentCapabilityError,
    AlignmentError,
    AlignmentService,
)

__all__ = [
    "AlignmentCapabilityError",
    "AlignmentError",
    "AlignmentMethod",
    "AlignmentRun",
    "AlignmentSegment",
    "AlignmentService",
    "AlignmentStatus",
    "AlignmentValidationMetrics",
    "AlignmentValidationResult",
    "CameraAlignment",
    "CanonicalOrigin",
    "FrameAlignmentProjection",
    "PiecewiseAlignment",
    "RationalTransformSegment",
    "round_half_even",
]
