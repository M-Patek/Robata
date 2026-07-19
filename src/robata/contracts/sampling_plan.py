"""Sampling plan and adaptive sampling configuration contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, StringConstraints

from robata.contracts.common import StrictModel

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
SchemaVersion = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._+-]*[A-Za-z0-9])?$",
    ),
]


class SamplingMode(StrEnum):
    """Sampling strategy mode."""

    UNIFORM = "UNIFORM"
    ADAPTIVE = "ADAPTIVE"
    DENSE = "DENSE"
    NOT_REQUESTED = "NOT_REQUESTED"


class OverflowPolicy(StrEnum):
    """Policy when frame budget is exceeded."""

    SPLIT_WINDOW = "SPLIT_WINDOW"
    DROP_FRAMES = "DROP_FRAMES"
    FAIL = "FAIL"


class AdaptiveTrigger(StrEnum):
    """Signal types that can trigger adaptive sampling rate changes."""

    MOTION_DELTA = "motion_delta"
    SCENE_CHANGE = "scene_change"
    BLUR_CHANGE = "blur_change"
    OCCLUSION_CHANGE = "occlusion_change"
    HAND_PRESENCE = "hand_presence"
    HAND_OBJECT_DISTANCE = "hand_object_distance"
    OBJECT_MOTION = "object_motion"


class PerCameraOverride(StrictModel):
    """Per-camera sampling rate override."""

    camera_id: NonEmptyString
    dense_sampling_rate_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]


class AdaptivePolicy(StrictModel):
    """Adaptive sampling policy configuration."""

    version: SchemaVersion
    min_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    max_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    triggers: tuple[AdaptiveTrigger, ...]
    hysteresis_sec: Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)] = 0.5


class FrameBudget(StrictModel):
    """Frame budget constraints."""

    max_frames_per_camera: Annotated[int, Field(strict=True, ge=1)]
    max_frames_total: Annotated[int, Field(strict=True, ge=1)]
    overflow_policy: OverflowPolicy = OverflowPolicy.SPLIT_WINDOW


class SamplingPlan(StrictModel):
    """Immutable sampling plan configuration."""

    sampling_plan_id: NonEmptyString
    version: SchemaVersion
    qa_sampling_rate_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    event_sampling_rate_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    dense_sampling_rate_fps: Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]
    per_camera: tuple[PerCameraOverride, ...] = ()
    adaptive_policy: AdaptivePolicy | None = None
    frame_budget: FrameBudget


__all__ = [
    "AdaptivePolicy",
    "AdaptiveTrigger",
    "FrameBudget",
    "OverflowPolicy",
    "PerCameraOverride",
    "SamplingMode",
    "SamplingPlan",
]
