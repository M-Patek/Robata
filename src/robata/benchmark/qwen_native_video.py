"""Bounded native-video inputs for Qwen3-VL benchmark runs.

This module is intentionally benchmark-only.  It materializes a complete bounded
interval as ordered frames and carries the source timeline alongside the bytes so a
Qwen3-VL processor can use its native ``video`` path instead of receiving several
unrelated image tokens.  It does not alter any published Robata wire contract.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

QWEN_NATIVE_VIDEO_INPUT_VERSION = "qwen-native-video-input-v1"
QWEN_NATIVE_VIDEO_MIN_FRAMES = 2
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
        """Serialize provenance and frame evidence without raw bytes or digests."""

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
        # PyAV is already the repository's canonical optional codec dependency.
        # Import it dynamically so the core package remains importable without
        # media extras while this benchmark reports a useful runtime error.
        av: Any = importlib.import_module("av")
    except ImportError as error:
        raise QwenNativeVideoInputError("native-video sampling requires the av package") from error

    try:
        container = av.open(str(path), mode="r")
    except Exception as error:
        raise QwenNativeVideoInputError(f"unable to open video: {path}") from error
    try:
        streams = list(container.streams.video)
        if not streams:
            raise QwenNativeVideoInputError("video does not contain a video stream")
        stream = streams[0]
        source_fps = _stream_frame_rate(stream)
        width, height = _stream_dimensions(stream)
        total_num_frames = _stream_frame_count(stream, source_fps)
        duration = total_num_frames / source_fps
        if total_num_frames < QWEN_NATIVE_VIDEO_MIN_FRAMES:
            raise QwenNativeVideoInputError(
                "Qwen native video requires a physical source with at least two frames"
            )
        if start_seconds >= duration:
            raise QwenNativeVideoInputError(
                "requested interval does not overlap the video duration "
                f"({start_seconds:.6f}s >= {duration:.6f}s)"
            )
        # A candidate interval may run a little past the physical recording
        # tail.  Clamp its semantic endpoint before constructing the runtime
        # request; otherwise the sampler would return an object that the native
        # runtime correctly rejects as outside ``duration_seconds``.
        interval_end = min(end_seconds, duration)
        window_start = max(0.0, start_seconds - context_before_seconds)
        window_end = min(duration, interval_end + context_after_seconds)
        first_index = max(0, min(total_num_frames - 1, math.floor(window_start * source_fps)))
        last_index = max(first_index, min(total_num_frames - 1, math.ceil(window_end * source_fps)))
        # Qwen3-VL's temporal patching path requires at least two real frames.
        # A one-frame request or a quantized zero-width interval must therefore
        # be widened to the nearest adjacent source frame.  Never duplicate a
        # payload: the resulting indices remain actual, strictly increasing
        # source-frame ordinals.
        sample_count = max(frame_count, QWEN_NATIVE_VIDEO_MIN_FRAMES)
        if first_index == last_index:
            if first_index == 0:
                last_index = 1
            else:
                first_index -= 1
        raw = [
            first_index + (last_index - first_index) * position / (sample_count - 1)
            for position in range(sample_count)
        ]
        indices = _unique_bounded_indices(raw, first_index, last_index)
        if len(indices) < sample_count:
            indices = _fill_indices(indices, first_index, last_index, sample_count)
        seek_applied = False
        seek = getattr(container, "seek", None)
        time_base = getattr(stream, "time_base", None)
        if first_index > 0 and callable(seek) and time_base is not None:
            try:
                seek_position = int((first_index / source_fps) / float(time_base))
                seek(seek_position, stream=stream, any_frame=False, backward=True)
                seek_applied = True
            except Exception as error:
                raise QwenNativeVideoInputError(
                    f"unable to seek video to requested window: {type(error).__name__}: {error}"
                ) from error

        selected: dict[int, bytes] = {}
        wanted = set(indices)
        try:
            for decoded_offset, frame in enumerate(container.decode(stream)):
                decoded_index = _decoded_frame_index(
                    frame,
                    source_fps=source_fps,
                    fallback_index=(
                        first_index + decoded_offset if seek_applied else decoded_offset
                    ),
                )
                if decoded_index < first_index:
                    continue
                if decoded_index > last_index:
                    break
                if decoded_index not in wanted:
                    continue
                selected[decoded_index] = _encode_jpeg_frame(frame, jpeg_quality, decoded_index)
                if len(selected) == len(wanted):
                    break
        except QwenNativeVideoInputError:
            raise
        except Exception as error:
            raise QwenNativeVideoInputError(
                f"unable to decode video frames from {path}: {type(error).__name__}: {error}"
            ) from error
        if len(selected) != len(wanted):
            missing = [index for index in indices if index not in selected]
            raise QwenNativeVideoInputError(f"unable to decode requested frame(s): {missing!r}")
        frames = tuple(
            QwenVideoFrame(
                payload=selected[frame_index],
                frame_index=frame_index,
                timestamp_seconds=frame_index / source_fps,
            )
            for frame_index in indices
        )
        return QwenNativeVideoInput(
            frames=frames,
            source_fps=source_fps,
            total_num_frames=total_num_frames,
            width=width,
            height=height,
            duration_seconds=duration,
            interval_start_seconds=start_seconds,
            interval_end_seconds=interval_end,
            context_before_seconds=context_before_seconds,
            context_after_seconds=context_after_seconds,
            source_path=str(path.resolve()),
        )
    finally:
        close = getattr(container, "close", None)
        if callable(close):
            close()


def _stream_frame_rate(stream: Any) -> float:
    """Read a positive constant frame rate from a PyAV video stream."""

    for name in ("average_rate", "guessed_rate", "base_rate"):
        value = cast(Any, getattr(stream, name, None))
        try:
            rate = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    raise QwenNativeVideoInputError("video does not expose a positive source FPS")


def _stream_dimensions(stream: Any) -> tuple[int, int]:
    """Read positive source dimensions from a PyAV stream or codec context."""

    context = getattr(stream, "codec_context", None)
    values: list[tuple[Any, Any]] = [
        (getattr(context, "width", None), getattr(context, "height", None)),
        (getattr(stream, "width", None), getattr(stream, "height", None)),
    ]
    for raw_width, raw_height in values:
        try:
            width, height = int(raw_width), int(raw_height)
        except (TypeError, ValueError, OverflowError):
            continue
        if width > 0 and height > 0:
            return width, height
    raise QwenNativeVideoInputError("video metadata is incomplete")


def _stream_frame_count(stream: Any, source_fps: float) -> int:
    """Read or derive a positive frame count needed for native timeline metadata."""

    raw_count = cast(Any, getattr(stream, "frames", None))
    try:
        count = int(raw_count)
    except (TypeError, ValueError, OverflowError):
        count = 0
    if count > 0:
        return count

    # Some containers omit ``nb_frames`` but expose a stream duration.  The
    # native route is intentionally constant-FPS; derive the count only when
    # the duration and time base provide a finite, positive estimate.
    raw_duration = cast(Any, getattr(stream, "duration", None))
    time_base = cast(Any, getattr(stream, "time_base", None))
    try:
        duration_seconds = float(raw_duration * time_base)
        estimated = round(duration_seconds * source_fps)
    except (TypeError, ValueError, OverflowError):
        estimated = 0
    if estimated > 0:
        return estimated
    raise QwenNativeVideoInputError("video does not expose a positive frame count")


def _encode_jpeg_frame(frame: Any, quality: int, frame_index: int) -> bytes:
    """Convert one decoded PyAV frame to the JPEG payload expected by Qwen."""

    try:
        image = frame.to_image()
        try:
            with BytesIO() as output:
                image.save(output, format="JPEG", quality=quality)
                payload = output.getvalue()
        finally:
            close = getattr(image, "close", None)
            if callable(close):
                close()
    except Exception as error:
        raise QwenNativeVideoInputError(
            f"unable to encode frame {frame_index}: {type(error).__name__}: {error}"
        ) from error
    if not payload:
        raise QwenNativeVideoInputError(f"unable to encode frame {frame_index}: empty JPEG")
    return payload


def _decoded_frame_index(frame: Any, *, source_fps: float, fallback_index: int) -> int:
    """Recover an absolute constant-FPS index from a decoded frame's PTS."""

    raw_pts = getattr(frame, "pts", None)
    raw_time_base = getattr(frame, "time_base", None)
    if raw_pts is not None and raw_time_base is not None:
        try:
            timestamp_seconds = float(raw_pts * raw_time_base)
            if math.isfinite(timestamp_seconds) and timestamp_seconds >= 0:
                return max(0, round(timestamp_seconds * source_fps))
        except (TypeError, ValueError, OverflowError):
            pass
    raw_time = getattr(frame, "time", None)
    if raw_time is not None:
        try:
            timestamp_seconds = float(raw_time)
            if math.isfinite(timestamp_seconds) and timestamp_seconds >= 0:
                return max(0, round(timestamp_seconds * source_fps))
        except (TypeError, ValueError, OverflowError):
            pass
    return fallback_index


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
    "QWEN_NATIVE_VIDEO_MIN_FRAMES",
    "QwenNativeVideoInput",
    "QwenNativeVideoInputError",
    "QwenVideoFrame",
    "sample_qwen_native_video",
]
