"""Bounded native-video inputs for Qwen3-VL benchmark runs.

This module is intentionally benchmark-only.  It materializes a complete bounded
interval as ordered frames and carries the source timeline alongside the bytes so a
Qwen3-VL processor can use its native ``video`` path instead of receiving several
unrelated image tokens.  It does not alter any published Robata wire contract.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QWEN_NATIVE_VIDEO_INPUT_VERSION = "qwen-native-video-input-v1"
QWEN_NATIVE_VIDEO_MAX_FRAMES = 32


class QwenNativeVideoInputError(ValueError):
    """A bounded native-video benchmark input is invalid or cannot be decoded."""


@dataclass(frozen=True, slots=True)
class QwenVideoFrame:
    """One encoded frame plus its position in the source timeline."""

    payload: bytes
    frame_index: int
    timestamp_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes) or not self.payload:
            raise QwenNativeVideoInputError("frame payload must be nonempty bytes")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise QwenNativeVideoInputError("frame_index must be an integer")
        if self.frame_index < 0:
            raise QwenNativeVideoInputError("frame_index must be non-negative")
        _finite_nonnegative(self.timestamp_seconds, "timestamp_seconds")

    @property
    def sha256(self) -> str:
        """Digest retained for evidence lineage without embedding frame bytes in JSON."""

        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True, slots=True)
class QwenNativeVideoInput:
    """A complete bounded action window for Qwen's native video processor."""

    frames: tuple[QwenVideoFrame, ...]
    source_fps: float
    total_num_frames: int
    width: int
    height: int
    duration_seconds: float | None
    interval_start_seconds: float
    interval_end_seconds: float
    context_before_seconds: float = 0.0
    context_after_seconds: float = 0.0
    source_path: str | None = None
    adapter_version: str = QWEN_NATIVE_VIDEO_INPUT_VERSION

    def __post_init__(self) -> None:
        if not self.frames:
            raise QwenNativeVideoInputError("at least one video frame is required")
        if len(self.frames) > QWEN_NATIVE_VIDEO_MAX_FRAMES:
            raise QwenNativeVideoInputError(
                f"frame count must not exceed {QWEN_NATIVE_VIDEO_MAX_FRAMES}"
            )
        if isinstance(self.total_num_frames, bool) or not isinstance(self.total_num_frames, int):
            raise QwenNativeVideoInputError("total_num_frames must be an integer")
        if self.total_num_frames <= 0:
            raise QwenNativeVideoInputError("total_num_frames must be positive")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width <= 0:
            raise QwenNativeVideoInputError("width must be a positive integer")
        if isinstance(self.height, bool) or not isinstance(self.height, int) or self.height <= 0:
            raise QwenNativeVideoInputError("height must be a positive integer")
        _finite_positive(self.source_fps, "source_fps")
        _finite_nonnegative(self.interval_start_seconds, "interval_start_seconds")
        _finite_nonnegative(self.interval_end_seconds, "interval_end_seconds")
        if self.interval_end_seconds < self.interval_start_seconds:
            raise QwenNativeVideoInputError("interval_end_seconds must not precede start")
        _finite_nonnegative(self.context_before_seconds, "context_before_seconds")
        _finite_nonnegative(self.context_after_seconds, "context_after_seconds")
        if self.duration_seconds is not None:
            _finite_positive(self.duration_seconds, "duration_seconds")
        previous_index = -1
        previous_timestamp = -1.0
        for frame in self.frames:
            if frame.frame_index <= previous_index:
                raise QwenNativeVideoInputError("frame indices must be strictly increasing")
            if frame.frame_index >= self.total_num_frames:
                raise QwenNativeVideoInputError("frame index exceeds total_num_frames")
            if frame.timestamp_seconds <= previous_timestamp:
                raise QwenNativeVideoInputError("frame timestamps must be strictly increasing")
            previous_index = frame.frame_index
            previous_timestamp = frame.timestamp_seconds

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(frame.frame_index for frame in self.frames)

    @property
    def frame_timestamps_seconds(self) -> tuple[float, ...]:
        return tuple(frame.timestamp_seconds for frame in self.frames)

    @property
    def frame_payloads(self) -> tuple[bytes, ...]:
        return tuple(frame.payload for frame in self.frames)

    @property
    def source_window_start_seconds(self) -> float:
        return max(0.0, self.interval_start_seconds - self.context_before_seconds)

    @property
    def source_window_end_seconds(self) -> float:
        end = self.interval_end_seconds + self.context_after_seconds
        if self.duration_seconds is not None:
            return min(self.duration_seconds, end)
        return end

    def processor_video_metadata(self) -> dict[str, Any]:
        """Return the exact metadata shape accepted by Transformers VideoMetadata."""

        metadata: dict[str, Any] = {
            "total_num_frames": self.total_num_frames,
            "fps": float(self.source_fps),
            "width": self.width,
            "height": self.height,
            "frames_indices": list(self.frame_indices),
        }
        if self.duration_seconds is not None:
            metadata["duration"] = float(self.duration_seconds)
        return metadata

    def evidence(self) -> dict[str, Any]:
        """Serialize provenance and frame evidence without serializing raw image bytes."""

        return {
            "adapter_version": self.adapter_version,
            "source_path": self.source_path,
            "source_fps": self.source_fps,
            "total_num_frames": self.total_num_frames,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "interval_start_seconds": self.interval_start_seconds,
            "interval_end_seconds": self.interval_end_seconds,
            "context_before_seconds": self.context_before_seconds,
            "context_after_seconds": self.context_after_seconds,
            "source_window_start_seconds": self.source_window_start_seconds,
            "source_window_end_seconds": self.source_window_end_seconds,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "sha256": frame.sha256,
                    "payload_bytes": len(frame.payload),
                }
                for frame in self.frames
            ],
        }


def sample_qwen_native_video(
    path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    frame_count: int = 8,
    context_before_seconds: float = 0.25,
    context_after_seconds: float = 0.25,
    jpeg_quality: int = 92,
) -> QwenNativeVideoInput:
    """Materialize a complete bounded interval with source frame indices.

    The function deliberately reads the whole requested window before inference and
    never creates a streaming ``image`` request.  It is suitable for benchmark
    qualification; production source adapters should provide equivalent provenance
    from their canonical media layer.
    """

    if not isinstance(path, Path):
        path = Path(path)
    _finite_nonnegative(start_seconds, "start_seconds")
    _finite_nonnegative(end_seconds, "end_seconds")
    if end_seconds < start_seconds:
        raise QwenNativeVideoInputError("end_seconds must not precede start_seconds")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise QwenNativeVideoInputError("frame_count must be a positive integer")
    if frame_count > QWEN_NATIVE_VIDEO_MAX_FRAMES:
        raise QwenNativeVideoInputError(
            f"frame_count must not exceed {QWEN_NATIVE_VIDEO_MAX_FRAMES}"
        )
    if (
        isinstance(jpeg_quality, bool)
        or not isinstance(jpeg_quality, int)
        or not 1 <= jpeg_quality <= 100
    ):
        raise QwenNativeVideoInputError("jpeg_quality must be an integer from 1 to 100")
    _finite_nonnegative(context_before_seconds, "context_before_seconds")
    _finite_nonnegative(context_after_seconds, "context_after_seconds")
    if not path.is_file():
        raise QwenNativeVideoInputError(f"video path is not a file: {path}")

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise QwenNativeVideoInputError(
            "native-video sampling requires opencv-python and numpy"
        ) from error

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise QwenNativeVideoInputError(f"unable to open video: {path}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_num_frames = round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise QwenNativeVideoInputError("video does not expose a positive source FPS")
        if total_num_frames <= 0 or width <= 0 or height <= 0:
            raise QwenNativeVideoInputError("video metadata is incomplete")
        duration = total_num_frames / source_fps
        window_start = max(0.0, start_seconds - context_before_seconds)
        window_end = min(duration, end_seconds + context_after_seconds)
        first_index = max(0, min(total_num_frames - 1, math.floor(window_start * source_fps)))
        last_index = max(first_index, min(total_num_frames - 1, math.ceil(window_end * source_fps)))
        if first_index == last_index:
            indices = [first_index]
        else:
            raw = np.linspace(first_index, last_index, num=frame_count)
            indices = _unique_bounded_indices(raw, first_index, last_index)
            if len(indices) < frame_count:
                indices = _fill_indices(indices, first_index, last_index, frame_count)
        frames: list[QwenVideoFrame] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise QwenNativeVideoInputError(f"unable to decode frame {frame_index}")
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not ok:
                raise QwenNativeVideoInputError(f"unable to encode frame {frame_index}")
            frames.append(
                QwenVideoFrame(
                    payload=bytes(encoded.tobytes()),
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / source_fps,
                )
            )
        return QwenNativeVideoInput(
            frames=tuple(frames),
            source_fps=source_fps,
            total_num_frames=total_num_frames,
            width=width,
            height=height,
            duration_seconds=duration,
            interval_start_seconds=start_seconds,
            interval_end_seconds=end_seconds,
            context_before_seconds=context_before_seconds,
            context_after_seconds=context_after_seconds,
            source_path=str(path.resolve()),
        )
    finally:
        capture.release()


def _unique_bounded_indices(values: Any, lower: int, upper: int) -> list[int]:
    result: list[int] = []
    for value in values:
        index = max(lower, min(upper, round(float(value))))
        if not result or index != result[-1]:
            result.append(index)
    return result


def _fill_indices(values: list[int], lower: int, upper: int, count: int) -> list[int]:
    if upper <= lower:
        return [lower]
    result = list(values)
    for index in range(lower, upper + 1):
        if index not in result:
            result.append(index)
        if len(result) >= count:
            break
    return sorted(result)


def _finite_nonnegative(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QwenNativeVideoInputError(f"{field} must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise QwenNativeVideoInputError(f"{field} must be a finite non-negative number")


def _finite_positive(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QwenNativeVideoInputError(f"{field} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise QwenNativeVideoInputError(f"{field} must be a finite positive number")


__all__ = [
    "QWEN_NATIVE_VIDEO_INPUT_VERSION",
    "QWEN_NATIVE_VIDEO_MAX_FRAMES",
    "QwenNativeVideoInput",
    "QwenNativeVideoInputError",
    "QwenVideoFrame",
    "sample_qwen_native_video",
]
