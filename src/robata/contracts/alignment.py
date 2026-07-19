"""Timestamp alignment contracts with rational transforms."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, SchemaVersion, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$",
    ),
]


class AlignmentMethod(StrEnum):
    """Alignment method in decreasing order of direct evidence."""

    HARDWARE_SYNC = "hardware_sync"
    SENSOR_CLOCK = "sensor_clock"
    MCAP_LOG_TIME = "mcap_log_time"
    CROSS_CORRELATION = "cross_correlation"
    MANUAL = "manual"


class AlignmentStatus(StrEnum):
    """Alignment quality status."""

    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"
    UNVERIFIED = "UNVERIFIED"


class AlignmentSegment(StrictModel):
    """One piecewise rational transform segment."""

    segment_id: NonEmptyString
    source_epoch_id: NonEmptyString
    source_order_start: NonNegativeInt
    source_order_end: NonNegativeInt
    source_start_ns: Nanoseconds
    source_end_ns: Nanoseconds
    source_anchor_ns: Nanoseconds
    canonical_anchor_ns: Nanoseconds
    rate_numerator: NonEmptyString
    rate_denominator: NonEmptyString
    rounding: Literal["HALF_EVEN", "HALF_UP", "HALF_DOWN", "CEILING", "FLOOR"]


class CameraAlignment(StrictModel):
    """Alignment summary for one camera."""

    camera_id: NonEmptyString
    source_clock_id: NonEmptyString
    source_timestamp_unit: Literal["ns", "us", "ms", "s"]
    derived_drift_ppm: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    residual_p95_ns: NonEmptyString
    max_error_ns: NonEmptyString
    coverage: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    segments: tuple[AlignmentSegment, ...]
    status: AlignmentStatus


class AlignmentRun(StrictModel):
    """One alignment run for an MCAP recording."""

    schema_version: Literal["1.0"]
    alignment_id: NonEmptyString
    mcap_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString
    reference_timebase: NonEmptyString
    canonical_origin: dict[str, object]
    method: AlignmentMethod
    algorithm_version: SchemaVersion
    status: AlignmentStatus
    cameras: dict[str, CameraAlignment]
    policy_version: SchemaVersion
    created_at: Rfc3339Timestamp


__all__ = [
    "AlignmentMethod",
    "AlignmentRun",
    "AlignmentSegment",
    "AlignmentStatus",
    "CameraAlignment",
]
