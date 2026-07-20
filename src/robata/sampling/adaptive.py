"""Adaptive sampling core logic for Section 7.2 of Architecture V1.

This module defines the signal taxonomy, detector protocol, policy models,
and the :class:`AdaptiveSampler` entry point that coordinates lightweight
per-camera signal detection with configurable rate modulation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from pydantic import Field, StringConstraints

from robata.contracts.cameras import CameraId
from robata.contracts.common import Nanoseconds, SchemaVersion, StrictModel

if TYPE_CHECKING:
    from robata.frame_cache import FramePayload

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


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
    """Coordinate adaptive sampling given a policy and a collection of detectors.

    The sampler runs every registered :class:`SignalDetector` over the supplied
    six-camera stream set, collects triggers, and produces an
    :class:`AdaptiveSamplingResult` that records the effective FPS and all
    detected features.  It does **not** materialize frames; that responsibility
    remains with the caller.
    """

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
        window: object,  # TemporalWindow not imported to avoid circular deps
        frames: dict[CameraId, Sequence[FramePayload]],
    ) -> AdaptiveSamplingResult:
        """Run all detectors and produce an adaptive sampling result.

        Args:
            window: The temporal window being sampled (duck-typed; expected
                to expose a ``window_id`` attribute).
            frames: Mapping from canonical :class:`CameraId` to ordered
                frame payloads for that camera.

        Returns:
            An :class:`AdaptiveSamplingResult` carrying the computed
            ``actual_fps`` and every trigger discovered across all cameras.
        """
        window_id = getattr(window, "window_id", "")
        if not window_id:
            raise ValueError("window must expose a window_id attribute")

        all_triggers: list[SignalTrigger] = []
        for camera_id, camera_frames in frames.items():
            for detector in self._detectors:
                triggers = detector.detect(camera_frames, camera_id=camera_id.value)
                all_triggers.extend(triggers)

        all_triggers.sort(key=lambda t: t.timestamp_ns)
        trigger_count = len(all_triggers)

        # Effective FPS is clamped to the policy range and modulated by
        # trigger density.  This is intentionally a skeleton; production
        # implementations may replace the heuristic with learned models.
        duration_frames = sum(len(f) for f in frames.values())
        if duration_frames > 0 and trigger_count > 0:
            # Simple heuristic: more triggers -> higher FPS, bounded by policy
            density_factor = min(trigger_count / max(duration_frames, 1), 1.0)
            actual_fps = (
                self._policy.min_fps
                + (self._policy.max_fps - self._policy.min_fps) * density_factor
            )
        else:
            actual_fps = self._policy.min_fps

        return AdaptiveSamplingResult(
            window_id=str(window_id),
            actual_fps=actual_fps,
            trigger_count=trigger_count,
            trigger_features=tuple(all_triggers),
        )


__all__ = [
    "AdaptiveSampler",
    "AdaptiveSamplingPolicy",
    "AdaptiveSamplingResult",
    "AdaptiveSignal",
    "SignalDetector",
    "SignalTrigger",
]
