"""Temporal window and package set contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from robata.contracts.common import Nanoseconds, SchemaVersion, StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
Rfc3339Timestamp = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$",
    ),
]


class SplitReason(StrEnum):
    """Reason for splitting a temporal window into multiple packages."""

    FRAME_BUDGET = "FRAME_BUDGET"
    PROVIDER_LIMIT = "PROVIDER_LIMIT"
    NONE = "NONE"


class TemporalPackageSetMember(StrictModel):
    """One member of a split temporal package set."""

    package_id: NonEmptyString
    ordinal: Annotated[int, Field(strict=True, ge=0)]
    part_count: Annotated[int, Field(strict=True, ge=1)]
    requested_start_ns: Nanoseconds
    requested_end_ns: Nanoseconds
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    overlap_before_ns: Nanoseconds = 0
    overlap_after_ns: Nanoseconds = 0


class TemporalPackageSet(StrictModel):
    """A set of temporal packages derived from one window, possibly split."""

    schema_version: Literal["1.0"]
    package_set_id: NonEmptyString
    split_group_id: NonEmptyString
    mcap_id: NonEmptyString
    window_id: NonEmptyString
    camera_mapping_run_id: NonEmptyString
    alignment_id: NonEmptyString
    requested_start_ns: Nanoseconds
    requested_end_ns: Nanoseconds
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    split_reason: SplitReason
    split_policy_version: SchemaVersion
    capability_snapshot_digest: NonEmptyString | None = None
    split_plan_digest: NonEmptyString
    members: tuple[TemporalPackageSetMember, ...]
    member_manifest_sha256: NonEmptyString
    reduction_policy_version: SchemaVersion
    created_at: Rfc3339Timestamp


class FrameSelectionManifest(StrictModel):
    """One selected frame within a temporal package."""

    frame_id: NonEmptyString
    alignment_projection_id: NonEmptyString
    ordinal: Annotated[int, Field(strict=True, ge=0)]
    aligned_timestamp_ns: Nanoseconds
    source_timestamp_ns: Nanoseconds
    delta_to_target_ns: Nanoseconds
    source_locator: dict[str, object]
    materialized_artifact: dict[str, object] | None = None
    width: Annotated[int, Field(strict=True, ge=1)]
    height: Annotated[int, Field(strict=True, ge=1)]
    quality_flags: tuple[NonEmptyString, ...] = ()


class CameraSamplingSummary(StrictModel):
    """Sampling metadata for one camera within a package."""

    strategy: NonEmptyString
    target_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    actual_fps: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
    target_count: Annotated[int, Field(strict=True, ge=0)]
    actual_count: Annotated[int, Field(strict=True, ge=0)]
    missed_targets: Annotated[int, Field(strict=True, ge=0)]
    trigger_features: tuple[NonEmptyString, ...] = ()


class TemporalPackageCameraEntry(StrictModel):
    """One camera entry within a temporal visual package."""

    status: NonEmptyString
    stream_id: NonEmptyString | None = None
    frames: tuple[FrameSelectionManifest, ...] = ()
    sampling: CameraSamplingSummary | None = None
    missing_reason: NonEmptyString | None = None


__all__ = [
    "CameraSamplingSummary",
    "FrameSelectionManifest",
    "SplitReason",
    "TemporalPackageCameraEntry",
    "TemporalPackageSet",
    "TemporalPackageSetMember",
]
