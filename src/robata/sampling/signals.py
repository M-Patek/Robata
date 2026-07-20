"""Lightweight signal detector implementations for adaptive sampling.

Each detector operates on raw frame payloads using only NumPy.  Where NumPy
suffices the implementation is pure NumPy; OpenCV is avoided to keep the
package dependency footprint minimal.
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

from robata.contracts.common import INT64_MAX, INT64_MIN
from robata.sampling.adaptive import AdaptiveSignal, SignalDetector, SignalTrigger

if TYPE_CHECKING:
    from robata.frame_cache import FramePayload


def _require_numpy() -> object:
    if np is None:
        raise ModuleNotFoundError(
            "NumPy is required for adaptive signal detectors; install the optional signal extra"
        )
    return np


def _to_gray_ndarray(data: bytes) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Best-effort decode of raw frame bytes into a grayscale NumPy array.

    The implementation assumes the payload is a raw 8-bit grayscale image
    whose width and height can be inferred from the payload length.  If
    the payload cannot be decoded, an empty array is returned so the
    detector degrades gracefully.
    """
    _require_numpy()
    arr = np.frombuffer(data, dtype=np.uint8)
    # Try common square-ish sizes first, then fall back to a linear array.
    for size in (1920, 1280, 1080, 640, 480, 256):
        if len(arr) == size * size:
            return arr.reshape((size, size))
        if len(arr) == size * (size // 2):
            return arr.reshape((size, size // 2))
    # Fallback: treat as 1-D signal; downstream detectors will handle it.
    return arr


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
        frames: Iterable[FramePayload],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        triggers: list[SignalTrigger] = []
        prev_gray: np.ndarray[Any, np.dtype[np.uint8]] | None = None
        for payload in frames:
            gray = _to_gray_ndarray(payload.data)
            if gray.size == 0:
                continue
            if prev_gray is not None:
                # Resize to the smaller common shape if dimensions differ.
                if prev_gray.shape != gray.shape:
                    min_shape = tuple(
                        min(a, b) for a, b in zip(prev_gray.shape, gray.shape, strict=True)
                    )
                    prev_gray = prev_gray[: min_shape[0], : min_shape[1]]
                    gray = gray[: min_shape[0], : min_shape[1]]
                diff = np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32)))
                if diff > self._threshold:
                    ts_ns = int(payload.timestamp_sec * 1_000_000_000)
                    if INT64_MIN <= ts_ns <= INT64_MAX:
                        triggers.append(
                            SignalTrigger(
                                signal_type=AdaptiveSignal.MOTION_ENERGY,
                                timestamp_ns=ts_ns,
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
        frames: Iterable[FramePayload],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        triggers: list[SignalTrigger] = []
        prev_hist: np.ndarray[Any, np.dtype[np.float32]] | None = None
        for payload in frames:
            gray = _to_gray_ndarray(payload.data)
            if gray.size == 0:
                continue
            hist, _ = np.histogram(gray, bins=self._bins, range=(0, 256))
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            if prev_hist is not None:
                # Chi-square-like distance
                distance = np.sum((hist - prev_hist) ** 2)
                if distance > self._threshold:
                    ts_ns = int(payload.timestamp_sec * 1_000_000_000)
                    if INT64_MIN <= ts_ns <= INT64_MAX:
                        triggers.append(
                            SignalTrigger(
                                signal_type=AdaptiveSignal.SCENE_CHANGE,
                                timestamp_ns=ts_ns,
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
        frames: Iterable[FramePayload],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        triggers: list[SignalTrigger] = []
        prev_variance: float | None = None
        for payload in frames:
            gray = _to_gray_ndarray(payload.data)
            if gray.size == 0:
                continue
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
                    ts_ns = int(payload.timestamp_sec * 1_000_000_000)
                    if INT64_MIN <= ts_ns <= INT64_MAX:
                        triggers.append(
                            SignalTrigger(
                                signal_type=AdaptiveSignal.BLUR_CHANGE,
                                timestamp_ns=ts_ns,
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
