"""Timestamp alignment contracts with rational transforms."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.contracts.common import INT64_MAX, Nanoseconds, SchemaVersion, StrictModel
from robata.contracts.logical_nodes import OpaqueUuid

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeNanoseconds = Annotated[Nanoseconds, Field(ge=0)]
PositiveInt64String = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=19, pattern=r"^[1-9][0-9]*$"),
]
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

    segment_id: OpaqueUuid
    source_epoch_id: NonEmptyString
    source_order_start: NonNegativeInt
    source_order_end: NonNegativeInt
    source_start_ns: Nanoseconds
    source_end_ns: Nanoseconds
    source_anchor_ns: Nanoseconds
    canonical_anchor_ns: Nanoseconds
    rate_numerator: PositiveInt64String
    rate_denominator: PositiveInt64String
    rounding: Literal["HALF_EVEN"]

    @field_validator("rate_numerator", "rate_denominator")
    @classmethod
    def require_int64_rate(cls, value: str) -> str:
        if int(value) > INT64_MAX:
            raise ValueError("alignment rate must fit positive signed int64")
        return value


class CanonicalOrigin(StrictModel):
    """Reference instant from which recording-relative canonical time starts."""

    source: NonEmptyString
    reference_timestamp_ns: Nanoseconds
    utc: Rfc3339Timestamp | None


class CameraAlignment(StrictModel):
    """Alignment summary for one camera."""

    source_clock_id: NonEmptyString
    source_timestamp_unit: Literal["ns"]
    derived_drift_ppm: Annotated[float, Field(strict=True, allow_inf_nan=False)]
    residual_p95_ns: NonNegativeNanoseconds
    max_error_ns: NonNegativeNanoseconds
    coverage: Annotated[float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)]
    segments: Annotated[tuple[AlignmentSegment, ...], Field(min_length=1)]
    status: AlignmentStatus


class AlignmentRun(StrictModel):
    """One alignment run for an MCAP recording."""

    schema_version: Literal["1.0"]
    alignment_id: OpaqueUuid
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    reference_timebase: NonEmptyString
    canonical_origin: CanonicalOrigin
    method: AlignmentMethod
    algorithm_version: SchemaVersion
    status: AlignmentStatus
    cameras: dict[str, CameraAlignment]
    policy_version: SchemaVersion
    created_at: Rfc3339Timestamp

    @model_validator(mode="after")
    def require_exact_camera_keys(self) -> AlignmentRun:
        if set(self.cameras) != set(CAMERA_ID_VALUES):
            raise ValueError("alignment cameras must contain cam_01 through cam_06")
        return self


__all__ = [
    "AlignmentMethod",
    "AlignmentRun",
    "AlignmentSegment",
    "AlignmentStatus",
    "CameraAlignment",
    "CanonicalOrigin",
]
