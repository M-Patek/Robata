"""Deterministic adaptive detection, rate reduction, and frozen target resolution.

The runtime sampler reduces registered detector triggers to a bounded window-average
rate. The exact resolver separately consumes frozen policy decisions and produces
identity-bound integer-nanosecond targets; runtime observations never bypass that
publication boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from itertools import pairwise
from math import gcd
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.artifacts import ArtifactId
from robata.contracts.cameras import CameraId
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256 as compute_semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.sampling.grid import SamplingGrid, SamplingRate

if TYPE_CHECKING:
    from robata.frame_cache import FramePayload

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION: Literal["adaptive-target-plan-semantic-v1"] = (
    "adaptive-target-plan-semantic-v1"
)


class AdaptiveSignal(StrEnum):
    """Lightweight per-camera signal types consumed by adaptive sampling."""

    MOTION_ENERGY = "MOTION_ENERGY"
    SCENE_CHANGE = "SCENE_CHANGE"
    BLUR_CHANGE = "BLUR_CHANGE"
    OCCLUSION_CHANGE = "OCCLUSION_CHANGE"
    HAND_PRESENCE = "HAND_PRESENCE"
    HAND_OBJECT_DISTANCE = "HAND_OBJECT_DISTANCE"
    OBJECT_MOTION = "OBJECT_MOTION"


class SignalTrigger(StrictModel):
    """One concrete trigger produced by a :class:`SignalDetector`."""

    signal_type: AdaptiveSignal
    timestamp_ns: Nanoseconds
    strength: NonNegativeFiniteFloat
    confidence: UnitInterval
    camera_id: CameraId | None = None


class AdaptiveSamplingPolicy(StrictModel):
    """Versioned policy that maps signal ranges to min/max FPS and hysteresis."""

    version: SchemaVersion
    min_fps: PositiveFiniteFloat
    max_fps: PositiveFiniteFloat
    triggers: tuple[AdaptiveSignal, ...]
    hysteresis_sec: NonNegativeFiniteFloat

    def model_post_init(self, __context: object) -> None:
        if self.min_fps > self.max_fps:
            raise ValueError("min_fps must not exceed max_fps")


class AdaptiveSamplingResult(StrictModel):
    """Immutable outcome of adaptive sampling for one temporal window."""

    window_id: NonEmptyString
    actual_fps: NonNegativeFiniteFloat
    trigger_count: NonNegativeInt
    trigger_features: tuple[SignalTrigger, ...]


class AdaptiveResolutionMode(StrEnum):
    """The frozen coordinate form supplied to the exact resolver."""

    GRID_SEGMENTS = "GRID_SEGMENTS"
    EXPLICIT_TARGETS = "EXPLICIT_TARGETS"


class FrozenAdaptiveTriggerArtifactRef(StrictModel):
    """Exact and semantic identity of one immutable trigger artifact."""

    artifact_id: ArtifactId
    schema_ref: SchemaRef
    exact_bytes_sha256: Sha256Digest
    semantic_sha256: Sha256Digest


class CanonicalAdaptiveGridSegment(StrictModel):
    """One ordered interval with an already-selected exact rational rate."""

    interval: NanosecondInterval
    rate_num: PositiveInt
    rate_den: PositiveInt

    @model_validator(mode="after")
    def validate_reduced_rate(self) -> Self:
        if gcd(self.rate_num, self.rate_den) != 1:
            raise ValueError("adaptive segment rate must be a reduced rational")
        return self


class FrozenAdaptiveResolutionRequest(StrictModel):
    """A frozen, policy-resolved input to exact adaptive target enumeration."""

    trigger_artifact: FrozenAdaptiveTriggerArtifactRef
    camera_id: CameraId
    effective_interval: NanosecondInterval
    grid_origin_ns: Nanoseconds
    resolution_mode: AdaptiveResolutionMode
    segments: tuple[CanonicalAdaptiveGridSegment, ...] = ()
    explicit_targets_ns: tuple[Nanoseconds, ...] = ()
    max_target_count: NonNegativeInt
    resolution_policy_version: SchemaVersion
    semantic_projection_version: Literal["adaptive-target-plan-semantic-v1"] = (
        ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def validate_frozen_coordinates(self) -> Self:
        _validate_frozen_coordinates(
            effective_interval=self.effective_interval,
            resolution_mode=self.resolution_mode,
            segments=self.segments,
            explicit_targets_ns=self.explicit_targets_ns,
        )
        return self


class ResolvedAdaptiveTarget(StrictModel):
    """One canonical integer-nanosecond adaptive target with grid provenance."""

    ordinal: NonNegativeInt
    target_ns: Nanoseconds
    segment_ordinal: NonNegativeInt | None = None
    grid_k: int | None = None

    @model_validator(mode="after")
    def validate_grid_provenance(self) -> Self:
        if (self.segment_ordinal is None) != (self.grid_k is None):
            raise ValueError("segment_ordinal and grid_k must be supplied together")
        return self


class ResolvedAdaptivePlan(StrictModel):
    """Deterministic target plan derived from one frozen adaptive input."""

    trigger_artifact: FrozenAdaptiveTriggerArtifactRef
    camera_id: CameraId
    effective_interval: NanosecondInterval
    grid_origin_ns: Nanoseconds
    resolution_mode: AdaptiveResolutionMode
    segments: tuple[CanonicalAdaptiveGridSegment, ...] = ()
    explicit_targets_ns: tuple[Nanoseconds, ...] = ()
    targets: tuple[ResolvedAdaptiveTarget, ...]
    max_target_count: NonNegativeInt
    resolution_policy_version: SchemaVersion
    semantic_projection_version: Literal["adaptive-target-plan-semantic-v1"] = (
        ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_resolved_plan(self) -> Self:
        _validate_frozen_coordinates(
            effective_interval=self.effective_interval,
            resolution_mode=self.resolution_mode,
            segments=self.segments,
            explicit_targets_ns=self.explicit_targets_ns,
        )

        expected_targets = _resolve_canonical_targets(
            grid_origin_ns=self.grid_origin_ns,
            resolution_mode=self.resolution_mode,
            segments=self.segments,
            explicit_targets_ns=self.explicit_targets_ns,
            max_target_count=self.max_target_count,
        )
        if self.targets != expected_targets:
            raise ValueError("resolved targets do not match the complete canonical target set")

        expected_digest = compute_semantic_sha256(adaptive_target_plan_semantic_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match the adaptive target plan projection")
        return self


def _validate_frozen_coordinates(
    *,
    effective_interval: NanosecondInterval,
    resolution_mode: AdaptiveResolutionMode,
    segments: tuple[CanonicalAdaptiveGridSegment, ...],
    explicit_targets_ns: tuple[int, ...],
) -> None:
    if resolution_mode is AdaptiveResolutionMode.GRID_SEGMENTS:
        if not segments:
            raise ValueError("GRID_SEGMENTS resolution requires at least one segment")
        if explicit_targets_ns:
            raise ValueError("GRID_SEGMENTS resolution cannot contain explicit targets")

        previous_end_ns: int | None = None
        for segment in segments:
            if segment.interval.start_ns < effective_interval.start_ns:
                raise ValueError("adaptive segments must lie inside effective_interval")
            if segment.interval.end_ns > effective_interval.end_ns:
                raise ValueError("adaptive segments must lie inside effective_interval")
            if previous_end_ns is not None and segment.interval.start_ns < previous_end_ns:
                raise ValueError("adaptive segments must be ordered and non-overlapping")
            previous_end_ns = segment.interval.end_ns
        return

    if segments:
        raise ValueError("EXPLICIT_TARGETS resolution cannot contain grid segments")
    if any(left >= right for left, right in pairwise(explicit_targets_ns)):
        raise ValueError("explicit targets must be strictly increasing")
    if any(not effective_interval.contains(target_ns) for target_ns in explicit_targets_ns):
        raise ValueError("explicit targets must lie inside effective_interval")


def _resolve_canonical_targets(
    *,
    grid_origin_ns: int,
    resolution_mode: AdaptiveResolutionMode,
    segments: tuple[CanonicalAdaptiveGridSegment, ...],
    explicit_targets_ns: tuple[int, ...],
    max_target_count: int,
) -> tuple[ResolvedAdaptiveTarget, ...]:
    if resolution_mode is AdaptiveResolutionMode.EXPLICIT_TARGETS:
        if len(explicit_targets_ns) > max_target_count:
            raise ValueError("adaptive target count exceeds max_target_count")
        return tuple(
            ResolvedAdaptiveTarget(ordinal=ordinal, target_ns=target_ns)
            for ordinal, target_ns in enumerate(explicit_targets_ns)
        )

    targets: list[ResolvedAdaptiveTarget] = []
    seen_target_ns: set[int] = set()
    for segment_ordinal, segment in enumerate(segments):
        grid = SamplingGrid(
            grid_origin_ns=grid_origin_ns,
            rate=SamplingRate(segment.rate_num, segment.rate_den),
        )
        for grid_target in grid.iter_unique_targets(
            segment.interval.start_ns,
            segment.interval.end_ns,
        ):
            if grid_target.target_ns in seen_target_ns:
                raise ValueError("adaptive grid segments resolved to a duplicate target")
            if len(targets) >= max_target_count:
                raise ValueError("adaptive target count exceeds max_target_count")
            seen_target_ns.add(grid_target.target_ns)
            targets.append(
                ResolvedAdaptiveTarget(
                    ordinal=len(targets),
                    target_ns=grid_target.target_ns,
                    segment_ordinal=segment_ordinal,
                    grid_k=grid_target.k,
                )
            )
    return tuple(targets)


def _adaptive_target_plan_projection(
    *,
    trigger_artifact: FrozenAdaptiveTriggerArtifactRef,
    camera_id: CameraId,
    effective_interval: NanosecondInterval,
    grid_origin_ns: int,
    resolution_mode: AdaptiveResolutionMode,
    segments: tuple[CanonicalAdaptiveGridSegment, ...],
    explicit_targets_ns: tuple[int, ...],
    targets: tuple[ResolvedAdaptiveTarget, ...],
    max_target_count: int,
    resolution_policy_version: str,
    semantic_projection_version: Literal["adaptive-target-plan-semantic-v1"],
) -> dict[str, object]:
    return {
        "semantic_projection_version": semantic_projection_version,
        "resolution_policy_version": resolution_policy_version,
        "trigger_artifact": {
            "schema_ref": {
                "schema_id": trigger_artifact.schema_ref.schema_id,
                "version": trigger_artifact.schema_ref.version,
            },
            "semantic_sha256": trigger_artifact.semantic_sha256,
        },
        "camera_id": camera_id.value,
        "effective_interval": {
            "start_ns": str(effective_interval.start_ns),
            "end_ns": str(effective_interval.end_ns),
        },
        "grid_origin_ns": str(grid_origin_ns),
        "resolution_mode": resolution_mode.value,
        "max_target_count": max_target_count,
        "segments": [
            {
                "start_ns": str(segment.interval.start_ns),
                "end_ns": str(segment.interval.end_ns),
                "rate_num": str(segment.rate_num),
                "rate_den": str(segment.rate_den),
            }
            for segment in segments
        ],
        "explicit_targets_ns": [str(target_ns) for target_ns in explicit_targets_ns],
        "resolved_targets": [
            {
                "ordinal": target.ordinal,
                "target_ns": str(target.target_ns),
                "segment_ordinal": target.segment_ordinal,
                "grid_k": None if target.grid_k is None else str(target.grid_k),
            }
            for target in targets
        ],
    }


def adaptive_target_plan_semantic_projection(plan: ResolvedAdaptivePlan) -> dict[str, object]:
    """Return the explicit, versioned semantic projection of a resolved plan."""

    if not isinstance(plan, ResolvedAdaptivePlan):
        raise TypeError("plan must be a ResolvedAdaptivePlan")
    return _adaptive_target_plan_projection(
        trigger_artifact=plan.trigger_artifact,
        camera_id=plan.camera_id,
        effective_interval=plan.effective_interval,
        grid_origin_ns=plan.grid_origin_ns,
        resolution_mode=plan.resolution_mode,
        segments=plan.segments,
        explicit_targets_ns=plan.explicit_targets_ns,
        targets=plan.targets,
        max_target_count=plan.max_target_count,
        resolution_policy_version=plan.resolution_policy_version,
        semantic_projection_version=plan.semantic_projection_version,
    )


def resolve_frozen_adaptive_targets(
    request: FrozenAdaptiveResolutionRequest,
) -> ResolvedAdaptivePlan:
    """Resolve frozen rational segments or exact timestamps without policy inference."""

    if not isinstance(request, FrozenAdaptiveResolutionRequest):
        raise TypeError("request must be a FrozenAdaptiveResolutionRequest")

    resolved_targets = _resolve_canonical_targets(
        grid_origin_ns=request.grid_origin_ns,
        resolution_mode=request.resolution_mode,
        segments=request.segments,
        explicit_targets_ns=request.explicit_targets_ns,
        max_target_count=request.max_target_count,
    )
    projection = _adaptive_target_plan_projection(
        trigger_artifact=request.trigger_artifact,
        camera_id=request.camera_id,
        effective_interval=request.effective_interval,
        grid_origin_ns=request.grid_origin_ns,
        resolution_mode=request.resolution_mode,
        segments=request.segments,
        explicit_targets_ns=request.explicit_targets_ns,
        targets=resolved_targets,
        max_target_count=request.max_target_count,
        resolution_policy_version=request.resolution_policy_version,
        semantic_projection_version=request.semantic_projection_version,
    )
    return ResolvedAdaptivePlan(
        trigger_artifact=request.trigger_artifact,
        camera_id=request.camera_id,
        effective_interval=request.effective_interval,
        grid_origin_ns=request.grid_origin_ns,
        resolution_mode=request.resolution_mode,
        segments=request.segments,
        explicit_targets_ns=request.explicit_targets_ns,
        targets=resolved_targets,
        max_target_count=request.max_target_count,
        resolution_policy_version=request.resolution_policy_version,
        semantic_projection_version=request.semantic_projection_version,
        semantic_sha256=compute_semantic_sha256(projection),
    )


class SignalDetector:
    """Protocol/abstract base class for per-camera signal detection.

    Concrete implementations must override :meth:`detect` and return a sequence
    of :class:`SignalTrigger` values ordered by ``timestamp_ns``.
    """

    def detect(
        self,
        frames: Iterable[FramePayload],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        """Detect signal triggers across the supplied frame sequence.

        Args:
            frames: Ordered iterable of decoded frame payloads for one camera.
            camera_id: Canonical camera identifier (e.g. ``cam_01``).

        Returns:
            Ordered sequence of detected triggers.
        """
        raise NotImplementedError("SignalDetector subclasses must implement detect()")


class AdaptiveSampler:
    """Run registered detectors and reduce triggers under a versioned policy."""

    def __init__(
        self,
        policy: AdaptiveSamplingPolicy,
        detectors: Sequence[SignalDetector],
    ) -> None:
        if not detectors:
            raise ValueError("at least one SignalDetector is required")
        self._policy = policy
        self._detectors = tuple(detectors)

    @property
    def policy(self) -> AdaptiveSamplingPolicy:
        """The immutable policy governing this sampler."""
        return self._policy

    @property
    def detectors(self) -> tuple[SignalDetector, ...]:
        """The ordered detectors registered with this sampler."""
        return self._detectors

    def sample(
        self,
        window: object,
        frames: dict[CameraId, Sequence[FramePayload]],
    ) -> AdaptiveSamplingResult:
        """Return a deterministic window-average rate without materializing frames."""

        window_id = getattr(window, "window_id", None)
        interval = getattr(window, "interval", None)
        if not isinstance(window_id, str) or not window_id:
            raise TypeError("window must expose a nonempty window_id")
        if not isinstance(interval, NanosecondInterval):
            raise TypeError("window must expose a NanosecondInterval interval")
        if not isinstance(frames, dict) or set(frames) != set(CameraId):
            raise ValueError("frames must contain every canonical camera exactly once")

        triggers: list[SignalTrigger] = []
        enabled = set(self._policy.triggers)
        for camera_id in CameraId:
            camera_frames = frames[camera_id]
            for detector in self._detectors:
                detected = detector.detect(camera_frames, camera_id=camera_id.value)
                for trigger in detected:
                    if not isinstance(trigger, SignalTrigger):
                        raise TypeError("signal detectors must return SignalTrigger values")
                    if trigger.signal_type not in enabled:
                        continue
                    if not interval.contains(trigger.timestamp_ns):
                        continue
                    if trigger.camera_id not in {None, camera_id}:
                        raise ValueError("signal trigger camera_id disagrees with detector input")
                    triggers.append(trigger.model_copy(update={"camera_id": camera_id}))

        canonical = tuple(
            sorted(
                triggers,
                key=lambda item: (
                    item.timestamp_ns,
                    item.camera_id.value if item.camera_id is not None else "",
                    item.signal_type.value,
                    item.strength,
                    item.confidence,
                ),
            )
        )
        actual_fps = _window_average_fps(
            interval=interval,
            triggers=canonical,
            policy=self._policy,
        )
        return AdaptiveSamplingResult(
            window_id=window_id,
            actual_fps=actual_fps,
            trigger_count=len(canonical),
            trigger_features=canonical,
        )


def _window_average_fps(
    *,
    interval: NanosecondInterval,
    triggers: tuple[SignalTrigger, ...],
    policy: AdaptiveSamplingPolicy,
) -> float:
    if not triggers:
        return policy.min_fps
    hysteresis_ns = int(policy.hysteresis_sec * 1_000_000_000)
    if hysteresis_ns <= 0:
        return policy.max_fps

    promoted = sorted(
        (
            trigger.timestamp_ns,
            min(interval.end_ns, trigger.timestamp_ns + hysteresis_ns),
        )
        for trigger in triggers
    )
    covered_ns = 0
    current_start, current_end = promoted[0]
    for start_ns, end_ns in promoted[1:]:
        if start_ns <= current_end:
            current_end = max(current_end, end_ns)
        else:
            covered_ns += current_end - current_start
            current_start, current_end = start_ns, end_ns
    covered_ns += current_end - current_start
    ratio = covered_ns / (interval.end_ns - interval.start_ns)
    return policy.min_fps + (policy.max_fps - policy.min_fps) * ratio


__all__ = [
    "ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION",
    "AdaptiveResolutionMode",
    "AdaptiveSampler",
    "AdaptiveSamplingPolicy",
    "AdaptiveSamplingResult",
    "AdaptiveSignal",
    "CanonicalAdaptiveGridSegment",
    "FrozenAdaptiveResolutionRequest",
    "FrozenAdaptiveTriggerArtifactRef",
    "ResolvedAdaptivePlan",
    "ResolvedAdaptiveTarget",
    "SignalDetector",
    "SignalTrigger",
    "adaptive_target_plan_semantic_projection",
    "resolve_frozen_adaptive_targets",
]
