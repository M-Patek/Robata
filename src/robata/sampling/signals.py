"""Lightweight signal detectors over compact decoded grayscale views.

Each detector consumes :class:`~robata.ports.decoded_frame.DecodedFrameView`,
whose row-major luminance bytes and dimensions are explicit. NumPy is optional;
OpenCV is intentionally avoided to keep the package dependency footprint minimal.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ModuleNotFoundError:  # optional signal-detector dependency
        np = None

from robata.ports.decoded_frame import DecodedFrameView
from robata.sampling.adaptive import AdaptiveSignal, SignalDetector, SignalTrigger


def _require_numpy() -> object:
    if np is None:
        raise ModuleNotFoundError(
            "NumPy is required for adaptive signal detectors; install the optional signal extra"
        )
    return np


def _to_gray_ndarray(frame: DecodedFrameView) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Return the explicit 2-D luminance raster from one normalized frame view."""

    if not isinstance(frame, DecodedFrameView):
        raise TypeError("adaptive signal frames must be DecodedFrameView values")
    _require_numpy()
    return np.frombuffer(frame.gray_pixels, dtype=np.uint8).reshape((frame.height, frame.width))


def _overlapping_rasters(
    previous: np.ndarray[Any, np.dtype[np.uint8]],
    current: np.ndarray[Any, np.dtype[np.uint8]],
) -> tuple[np.ndarray[Any, np.dtype[np.uint8]], np.ndarray[Any, np.dtype[np.uint8]]]:
    """Align two explicitly shaped rasters to their common top-left region."""

    height = min(previous.shape[0], current.shape[0])
    width = min(previous.shape[1], current.shape[1])
    return previous[:height, :width], current[:height, :width]


class MotionEnergyDetector(SignalDetector):
    """Detect motion energy from frame-to-frame differences.

    Computes the mean absolute difference between consecutive grayscale
    frames.  When the difference exceeds a configurable threshold a
    :class:`SignalTrigger` is emitted.
    """

    def __init__(self, threshold: float = 15.0) -> None:
        self._threshold = threshold

    def detect(
        self,
        frames: Iterable[DecodedFrameView],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        triggers: list[SignalTrigger] = []
        prev_gray: np.ndarray[Any, np.dtype[np.uint8]] | None = None
        for frame in frames:
            gray = _to_gray_ndarray(frame)
            if prev_gray is not None:
                previous, current = _overlapping_rasters(prev_gray, gray)
                diff = np.mean(np.abs(previous.astype(np.float32) - current.astype(np.float32)))
                if diff > self._threshold:
                    triggers.append(
                        SignalTrigger(
                            signal_type=AdaptiveSignal.MOTION_ENERGY,
                            timestamp_ns=frame.timestamp_ns,
                            strength=float(diff),
                            confidence=min(diff / (self._threshold * 2), 1.0),
                        )
                    )
            prev_gray = gray
        return triggers


class SceneChangeDetector(SignalDetector):
    """Detect scene changes via histogram differences.

    Compares the grayscale histogram of each frame with the previous one.
    When the chi-square distance exceeds a threshold a trigger is emitted.
    """

    def __init__(self, threshold: float = 0.3, bins: int = 64) -> None:
        self._threshold = threshold
        self._bins = bins

    def detect(
        self,
        frames: Iterable[DecodedFrameView],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        triggers: list[SignalTrigger] = []
        prev_hist: np.ndarray[Any, np.dtype[np.float32]] | None = None
        for frame in frames:
            gray = _to_gray_ndarray(frame)
            hist, _ = np.histogram(gray, bins=self._bins, range=(0, 256))
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            if prev_hist is not None:
                # Chi-square-like distance
                distance = np.sum((hist - prev_hist) ** 2)
                if distance > self._threshold:
                    triggers.append(
                        SignalTrigger(
                            signal_type=AdaptiveSignal.SCENE_CHANGE,
                            timestamp_ns=frame.timestamp_ns,
                            strength=float(distance),
                            confidence=min(distance / (self._threshold * 2), 1.0),
                        )
                    )
            prev_hist = hist
        return triggers


class BlurDetector(SignalDetector):
    """Detect blur changes via Laplacian variance.

    Computes the variance of the Laplacian (a simple focus measure) on
    each frame.  A sudden drop in variance relative to the running mean
    indicates increased blur.
    """

    def __init__(self, threshold: float = 100.0) -> None:
        self._threshold = threshold

    def detect(
        self,
        frames: Iterable[DecodedFrameView],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        triggers: list[SignalTrigger] = []
        prev_variance: float | None = None
        for frame in frames:
            gray = _to_gray_ndarray(frame)
            # Approximate 2-D Laplacian via finite differences.
            laplacian = (
                -4 * gray.astype(np.float32)
                + np.roll(gray, 1, axis=0).astype(np.float32)
                + np.roll(gray, -1, axis=0).astype(np.float32)
                + np.roll(gray, 1, axis=1).astype(np.float32)
                + np.roll(gray, -1, axis=1).astype(np.float32)
            )
            variance = float(np.var(laplacian))
            if prev_variance is not None:
                drop = max(0.0, prev_variance - variance)
                if drop > self._threshold:
                    triggers.append(
                        SignalTrigger(
                            signal_type=AdaptiveSignal.BLUR_CHANGE,
                            timestamp_ns=frame.timestamp_ns,
                            strength=drop,
                            confidence=min(drop / (self._threshold * 2), 1.0),
                        )
                    )
            prev_variance = variance
        return triggers


__all__ = [
    "BlurDetector",
    "MotionEnergyDetector",
    "SceneChangeDetector",
]
