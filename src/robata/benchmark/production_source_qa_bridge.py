"""Lightweight, source-bound media QA observations for production recordings.

This module deliberately sits outside the production API and annotation path.  It
streams one MCAP archive member at a time, decodes the six mapped camera topics,
and samples a small number of grayscale frames for deterministic *objective*
proxies: black/overexposed content, blur, frozen content, timestamp continuity,
and decode continuity.  The result is a review aid, not a human QA decision and
never makes a production-eligibility claim.

The implementation does not persist frames, invoke a model, infer task labels,
or calculate a content hash.  The ``source_preflight`` input is optional; when it
is supplied, only members already known to be structurally readable are opened.
Malformed members are represented as excluded rows instead of being retried
indefinitely.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import IO, Any, Final

CAMERA_IDS: Final[tuple[str, ...]] = tuple(f"cam_{index:02d}" for index in range(1, 7))
DEFAULT_CAMERA_TOPICS: Final[dict[str, str]] = {
    camera_id: f"/robot0/sensor/camera{index}/compressed"
    for index, camera_id in enumerate(CAMERA_IDS)
}
SOURCE_QA_BRIDGE_VERSION: Final = "robata-production-source-qa-bridge-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"


class SourceQABridgeError(ValueError):
    """Raised when source QA bridge inputs are invalid."""


@dataclass(frozen=True, slots=True)
class SourceQAPolicy:
    """Small, explicit sampling policy for objective source observations."""

    sample_period_seconds: float = 2.0
    max_samples_per_camera: int = 300
    analysis_width: int = 64
    black_luma_max: int = 16
    black_fraction_fail: float = 0.98
    overexposed_luma_min: int = 240
    overexposed_fraction_fail: float = 0.98
    blur_laplacian_variance_max: float = 100.0
    blur_warning_fraction: float = 0.50
    freeze_delta_luma_max: float = 1.5
    freeze_min_duration_seconds: float = 5.0
    timestamp_gap_warning_seconds: float = 0.5
    decode_ratio_warning: float = 0.98
    decode_ratio_failure: float = 0.90

    def __post_init__(self) -> None:
        positive_ints = (self.max_samples_per_camera, self.analysis_width)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_ints
        ):
            raise SourceQABridgeError("sample limits must be positive integers")
        positive_floats = (
            self.sample_period_seconds,
            self.freeze_min_duration_seconds,
            self.timestamp_gap_warning_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in positive_floats
        ):
            raise SourceQABridgeError("time thresholds must be finite positive numbers")
        if not 0 <= self.black_luma_max <= 255 or not 0 <= self.overexposed_luma_min <= 255:
            raise SourceQABridgeError("luma thresholds must be in [0, 255]")
        bounded = (
            self.black_fraction_fail,
            self.overexposed_fraction_fail,
            self.blur_warning_fraction,
            self.decode_ratio_warning,
            self.decode_ratio_failure,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in bounded
        ):
            raise SourceQABridgeError("fraction thresholds must be finite values in [0, 1]")
        if self.decode_ratio_failure > self.decode_ratio_warning:
            raise SourceQABridgeError("decode failure ratio must not exceed warning ratio")
        if (
            isinstance(self.blur_laplacian_variance_max, bool)
            or not isinstance(self.blur_laplacian_variance_max, (int, float))
            or not math.isfinite(float(self.blur_laplacian_variance_max))
            or self.blur_laplacian_variance_max < 0.0
        ):
            raise SourceQABridgeError("blur threshold must be a finite nonnegative number")
        if (
            isinstance(self.freeze_delta_luma_max, bool)
            or not isinstance(self.freeze_delta_luma_max, (int, float))
            or not math.isfinite(float(self.freeze_delta_luma_max))
            or self.freeze_delta_luma_max < 0.0
        ):
            raise SourceQABridgeError("freeze threshold must be a finite nonnegative number")


DEFAULT_SOURCE_QA_POLICY: Final[SourceQAPolicy] = SourceQAPolicy()


@dataclass(frozen=True, slots=True)
class FrameSample:
    """Compact grayscale observation retained for one sampled decoded frame."""

    timestamp_ns: int
    mean_luma: float
    black_fraction: float
    overexposed_fraction: float
    edge_energy: float
    laplacian_variance: float
    frame_delta_luma: float | None


@dataclass(slots=True)
class _CameraState:
    camera_id: str
    topic: str
    expected_frames: int | None = None
    source_messages: int = 0
    decoded_frames: int = 0
    decode_errors: int = 0
    timestamps: list[int] = field(default_factory=list)
    samples: list[FrameSample] = field(default_factory=list)
    sequence_values: list[int] = field(default_factory=list)
    next_sample_ns: int | None = None
    previous_sample_pixels: tuple[int, ...] | None = None
    previous_sample_timestamp_ns: int | None = None


def normalize_camera_topics(value: object | None = None) -> tuple[tuple[str, str], ...]:
    """Return exactly six unique camera/topic pairs in canonical order."""

    raw: object = DEFAULT_CAMERA_TOPICS if value is None else value
    if isinstance(raw, Mapping):
        for wrapper in ("topics", "camera_topics"):
            if wrapper in raw:
                raw = raw[wrapper]
                break
        else:
            if "cameras" in raw and isinstance(raw["cameras"], Sequence):
                raw = raw["cameras"]
    pairs: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for camera, topic in raw.items():
            if not isinstance(camera, str) or not camera.strip():
                raise SourceQABridgeError("camera id must be a non-empty string")
            if not isinstance(topic, str) or not topic.strip():
                raise SourceQABridgeError(f"topic for {camera!r} must be non-empty")
            pairs[camera.strip()] = topic.strip()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for index, row in enumerate(raw):
            if not isinstance(row, Mapping):
                raise SourceQABridgeError(f"camera_topics[{index}] must be an object")
            camera = row.get("camera_id", row.get("id"))
            topic = row.get("topic", row.get("camera_topic"))
            if not isinstance(camera, str) or not camera.strip():
                raise SourceQABridgeError(f"camera_topics[{index}].camera_id is invalid")
            if not isinstance(topic, str) or not topic.strip():
                raise SourceQABridgeError(f"camera_topics[{index}].topic is invalid")
            if camera.strip() in pairs:
                raise SourceQABridgeError(f"duplicate camera id: {camera.strip()}")
            pairs[camera.strip()] = topic.strip()
    else:
        raise SourceQABridgeError("camera topics must be an object or array")
    expected = set(CAMERA_IDS)
    if set(pairs) != expected:
        missing = sorted(expected - set(pairs))
        extra = sorted(set(pairs) - expected)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise SourceQABridgeError(
            "camera mapping must contain cam_01..cam_06"
            + (f" ({'; '.join(detail)})" if detail else "")
        )
    ordered = tuple((camera, pairs[camera]) for camera in CAMERA_IDS)
    if len({topic for _, topic in ordered}) != len(ordered):
        raise SourceQABridgeError("camera topics must be unique")
    return ordered


def _gray_pixels(frame: Any, width: int) -> tuple[int, int, tuple[int, ...]]:
    """Convert a PyAV frame to a small, row-major grayscale tuple."""

    source_width = int(getattr(frame, "width", 0) or 0)
    source_height = int(getattr(frame, "height", 0) or 0)
    if source_width <= 0 or source_height <= 0:
        raise SourceQABridgeError("decoded frame dimensions must be positive")
    target_height = max(1, round(source_height * width / source_width))
    try:
        gray_frame = frame.reformat(width=width, height=target_height, format="gray")
        plane = gray_frame.planes[0]
        line_size = int(plane.line_size)
        raw = bytes(plane)
    except Exception as exc:  # pragma: no cover - depends on PyAV internals
        raise SourceQABridgeError(
            f"could not create grayscale frame: {type(exc).__name__}: {exc}"
        ) from exc
    pixels: list[int] = []
    for row in range(target_height):
        start = row * line_size
        pixels.extend(raw[start : start + width])
    if len(pixels) != width * target_height:
        raise SourceQABridgeError("decoded grayscale frame has unexpected row length")
    return width, target_height, tuple(pixels)


def _frame_metrics(
    pixels: tuple[int, ...],
    width: int,
    height: int,
    *,
    policy: SourceQAPolicy,
    previous: tuple[int, ...] | None,
) -> tuple[float, float, float, float, float, float | None]:
    count = len(pixels)
    mean_luma = sum(pixels) / count
    black_fraction = sum(pixel <= policy.black_luma_max for pixel in pixels) / count
    overexposed_fraction = sum(pixel >= policy.overexposed_luma_min for pixel in pixels) / count
    edge_values: list[float] = []
    laplacian_values: list[float] = []
    for row in range(height):
        row_start = row * width
        for column in range(width):
            current = pixels[row_start + column]
            if column + 1 < width:
                edge_values.append(abs(current - pixels[row_start + column + 1]))
            if row + 1 < height:
                edge_values.append(abs(current - pixels[row_start + width + column]))
            if 0 < row < height - 1 and 0 < column < width - 1:
                neighbors = (
                    pixels[row_start - width + column],
                    pixels[row_start + width + column],
                    pixels[row_start + column - 1],
                    pixels[row_start + column + 1],
                )
                laplacian_values.append(abs(4 * current - sum(neighbors)))
    edge_energy = sum(edge_values) / max(1, len(edge_values))
    if laplacian_values:
        lap_mean = sum(laplacian_values) / len(laplacian_values)
        laplacian_variance = sum((value - lap_mean) ** 2 for value in laplacian_values) / len(
            laplacian_values
        )
    else:
        laplacian_variance = 0.0
    frame_delta = None
    if previous is not None:
        frame_delta = (
            sum(abs(current - old) for current, old in zip(pixels, previous, strict=True)) / count
        )
    return (
        mean_luma,
        black_fraction,
        overexposed_fraction,
        edge_energy,
        laplacian_variance,
        frame_delta,
    )


def _message_timestamp(decoded: Any, message: Any) -> int | None:
    for candidate in (
        getattr(getattr(decoded, "header", None), "timestamp", None),
        getattr(message, "log_time", None),
    ):
        if candidate is None:
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        if value >= 0:
            return value
    return None


def _frame_timestamp(frame: Any, fallback_ns: int | None) -> int | None:
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is not None and time_base is not None:
        try:
            return int(Fraction(pts) * time_base * 1_000_000_000)
        except (TypeError, ValueError, OverflowError):
            pass
    return fallback_ns


def _camera_result(state: _CameraState, policy: SourceQAPolicy) -> dict[str, Any]:
    samples = tuple(state.samples)
    timestamp_nonmonotonic = sum(
        current <= previous for previous, current in pairwise(state.timestamps)
    )
    positive_deltas = tuple(
        current - previous for previous, current in pairwise(state.timestamps) if current > previous
    )
    median_delta = sorted(positive_deltas)[len(positive_deltas) // 2] if positive_deltas else None
    gap_threshold = max(
        int(policy.timestamp_gap_warning_seconds * 1_000_000_000),
        3 * median_delta if median_delta is not None else 0,
    )
    timestamp_gaps = sum(
        current - previous > gap_threshold for previous, current in pairwise(state.timestamps)
    )
    black_samples = sum(sample.black_fraction >= policy.black_fraction_fail for sample in samples)
    overexposed_samples = sum(
        sample.overexposed_fraction >= policy.overexposed_fraction_fail for sample in samples
    )
    blur_samples = sum(
        sample.laplacian_variance <= policy.blur_laplacian_variance_max for sample in samples
    )
    frozen_samples = 0
    frozen_start_ns: int | None = None
    frozen_intervals: list[dict[str, float]] = []
    timeline_origin_ns = state.timestamps[0] if state.timestamps else 0
    for previous, current in pairwise(samples):
        stable = (
            current.frame_delta_luma is not None
            and current.frame_delta_luma <= policy.freeze_delta_luma_max
        )
        if stable:
            frozen_samples += 1
            if frozen_start_ns is None:
                frozen_start_ns = previous.timestamp_ns
            if current.timestamp_ns - frozen_start_ns >= int(
                policy.freeze_min_duration_seconds * 1_000_000_000
            ):
                frozen_intervals.append(
                    {
                        "start_seconds": (frozen_start_ns - timeline_origin_ns) / 1_000_000_000,
                        "end_seconds": (current.timestamp_ns - timeline_origin_ns) / 1_000_000_000,
                    }
                )
        else:
            frozen_start_ns = None
    expected = state.expected_frames or state.source_messages
    decode_ratio = state.decoded_frames / expected if expected > 0 else 0.0
    checks: dict[str, dict[str, Any]] = {
        "blackout": {
            "status": "FAIL"
            if samples and black_samples == len(samples)
            else ("WARNING" if black_samples else "PASS"),
            "sample_count": len(samples),
            "affected_samples": black_samples,
            "proxy": True,
        },
        "exposure": {
            "status": "FAIL"
            if samples and overexposed_samples == len(samples)
            else ("WARNING" if overexposed_samples else "PASS"),
            "sample_count": len(samples),
            "affected_samples": overexposed_samples,
            "proxy": True,
        },
        "blur": {
            "status": "WARNING"
            if blur_samples
            and samples
            and blur_samples / len(samples) >= policy.blur_warning_fraction
            else "PASS",
            "sample_count": len(samples),
            "affected_samples": blur_samples,
            "proxy": True,
        },
        "freeze": {
            "status": "WARNING" if frozen_intervals else "PASS",
            "sample_count": len(samples),
            "affected_samples": frozen_samples,
            "intervals": frozen_intervals,
            "proxy": True,
        },
        "timestamps": {
            "status": "FAIL"
            if not state.timestamps or timestamp_nonmonotonic
            else ("WARNING" if timestamp_gaps else "PASS"),
            "message_count": len(state.timestamps),
            "nonmonotonic_count": timestamp_nonmonotonic,
            "large_gap_count": timestamp_gaps,
            "median_delta_ms": None if median_delta is None else median_delta / 1_000_000,
            "proxy": False,
        },
        "decode_continuity": {
            "status": (
                "FAIL"
                if state.decoded_frames == 0 or decode_ratio < policy.decode_ratio_failure
                else (
                    "WARNING"
                    if decode_ratio < policy.decode_ratio_warning or state.decode_errors
                    else "PASS"
                )
            ),
            "expected_frames": expected,
            "decoded_frames": state.decoded_frames,
            "decode_errors": state.decode_errors,
            "decode_ratio": decode_ratio,
            "proxy": False,
        },
    }
    statuses = tuple(item["status"] for item in checks.values())
    status = "FAIL" if "FAIL" in statuses else ("WARNING" if "WARNING" in statuses else "PASS")
    return {
        "camera_id": state.camera_id,
        "topic": state.topic,
        "status": status,
        "source_messages": state.source_messages,
        "expected_frames": expected,
        "decoded_frames": state.decoded_frames,
        "sampled_frames": len(samples),
        "decode_errors": state.decode_errors,
        "first_timestamp_ns": state.timestamps[0] if state.timestamps else None,
        "last_timestamp_ns": state.timestamps[-1] if state.timestamps else None,
        "timeline_origin_ns": timeline_origin_ns or None,
        "checks": checks,
        "sample_metrics": [
            {
                "timestamp_ns": sample.timestamp_ns,
                "relative_seconds": (sample.timestamp_ns - timeline_origin_ns) / 1_000_000_000,
                "mean_luma": sample.mean_luma,
                "black_fraction": sample.black_fraction,
                "overexposed_fraction": sample.overexposed_fraction,
                "edge_energy": sample.edge_energy,
                "laplacian_variance": sample.laplacian_variance,
                "frame_delta_luma": sample.frame_delta_luma,
            }
            for sample in samples
        ],
    }


def _scan_stream(
    stream: IO[bytes],
    *,
    source_ref: str,
    camera_topics: tuple[tuple[str, str], ...],
    expected_frames: Mapping[str, int] | None,
    policy: SourceQAPolicy,
    validate_crcs: bool,
) -> dict[str, Any]:
    """Scan one MCAP stream.  Imports of optional media packages stay lazy."""

    try:
        import av
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SourceQABridgeError("source QA scan requires av, mcap, and mcap-protobuf") from exc

    topic_to_camera = {topic: camera for camera, topic in camera_topics}
    states = {
        camera: _CameraState(
            camera, topic, None if expected_frames is None else expected_frames.get(camera)
        )
        for camera, topic in camera_topics
    }
    decoders: dict[int, Any] = {}
    codec_contexts: dict[str, Any] = {}
    reader = make_reader(stream, validate_crcs=validate_crcs)
    try:
        iterator = reader.iter_messages(topics=tuple(topic_to_camera), log_time_order=False)
        for schema, channel, message in iterator:
            topic = getattr(channel, "topic", None)
            if not isinstance(topic, str):
                continue
            camera = topic_to_camera.get(topic)
            if camera is None:
                continue
            state = states[camera]
            state.source_messages += 1
            timestamp_hint = getattr(message, "log_time", None)
            try:
                schema_id = int(getattr(schema, "id", 0))
                decoder = decoders.get(schema_id)
                if decoder is None:
                    factory = DecoderFactory()
                    decoder = factory.decoder_for(channel.message_encoding, schema)
                    if decoder is None:
                        raise SourceQABridgeError("camera channel has no protobuf decoder")
                    decoders[schema_id] = decoder
                decoded = decoder(message.data)
                image_format = str(getattr(decoded, "format", "")).strip().casefold()
                payload = getattr(decoded, "data", None)
                if image_format != "h264" or not isinstance(payload, bytes) or not payload:
                    raise SourceQABridgeError(
                        "camera payload is not a non-empty h264 CompressedImage"
                    )
                decoder_context = codec_contexts.get(camera)
                if decoder_context is None:
                    decoder_context = av.CodecContext.create("h264", "r")
                    codec_contexts[camera] = decoder_context
                packet = av.Packet(payload)
                if timestamp_hint is not None:
                    packet.pts = int(timestamp_hint)
                    packet.dts = int(timestamp_hint)
                    packet.time_base = Fraction(1, 1_000_000_000)
                source_timestamp = _message_timestamp(decoded, message)
                if source_timestamp is not None:
                    if state.timestamps and source_timestamp <= state.timestamps[-1]:
                        # Retain the row so the monotonicity result is source-bound.
                        state.timestamps.append(source_timestamp)
                    else:
                        state.timestamps.append(source_timestamp)
                frames = decoder_context.decode(packet)
                state.decoded_frames += len(frames)
            except Exception:
                state.decode_errors += 1
                continue
            for frame in frames:
                frame_timestamp = _frame_timestamp(frame, source_timestamp)
                if frame_timestamp is None:
                    continue
                if state.next_sample_ns is not None and frame_timestamp < state.next_sample_ns:
                    continue
                if len(state.samples) >= policy.max_samples_per_camera:
                    continue
                width, height, pixels = _gray_pixels(frame, policy.analysis_width)
                metrics = _frame_metrics(
                    pixels,
                    width,
                    height,
                    policy=policy,
                    previous=state.previous_sample_pixels,
                )
                state.samples.append(FrameSample(frame_timestamp, *metrics))
                state.previous_sample_pixels = pixels
                state.previous_sample_timestamp_ns = frame_timestamp
                state.next_sample_ns = frame_timestamp + int(
                    policy.sample_period_seconds * 1_000_000_000
                )
        for camera, decoder_context in codec_contexts.items():
            try:
                flushed = decoder_context.decode(None)
            except Exception:
                states[camera].decode_errors += 1
                continue
            states[camera].decoded_frames += len(flushed)
    except Exception as exc:
        raise SourceQABridgeError(
            f"could not scan MCAP {source_ref}: {type(exc).__name__}: {exc}"
        ) from exc
    cameras = [_camera_result(states[camera], policy) for camera, _ in camera_topics]
    statuses = tuple(camera["status"] for camera in cameras)
    status = "FAIL" if "FAIL" in statuses else ("WARNING" if "WARNING" in statuses else "PASS")
    return {
        "source": {"ref": source_ref, "source_bound": True},
        "status": status,
        "camera_count": len(cameras),
        "cameras": cameras,
        "policy": {
            "sample_period_seconds": policy.sample_period_seconds,
            "max_samples_per_camera": policy.max_samples_per_camera,
            "analysis_width": policy.analysis_width,
            "objective_only": True,
        },
    }


def _expected_frames_from_preflight(item: Mapping[str, Any]) -> dict[str, int]:
    rows = item.get("camera_channels")
    result: dict[str, int] = {}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return result
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        topic = row.get("topic")
        count = row.get("message_count")
        if isinstance(topic, str) and isinstance(count, int) and count >= 0:
            camera = CAMERA_IDS[index] if index < len(CAMERA_IDS) else None
            if camera is not None:
                result[camera] = count
    return result


def _load_preflight(path: Path) -> dict[str, Mapping[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceQABridgeError(f"could not read source preflight {path}: {exc}") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("items"), Sequence):
        raise SourceQABridgeError("source preflight must contain an items array")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in value["items"]:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            continue
        result[str(raw["name"])] = raw
    return result


def run_archive_source_qa(
    archive: Path,
    *,
    preflight_path: Path | None = None,
    camera_topics: object | None = None,
    policy: SourceQAPolicy = DEFAULT_SOURCE_QA_POLICY,
    validate_crcs: bool = True,
    max_recordings: int | None = None,
) -> dict[str, Any]:
    """Scan source-preflight PASS archive members one at a time.

    The returned ``qa_admission`` is intentionally ``PENDING_VISUAL_REVIEW``
    even when all objective checks pass.  This prevents deterministic proxies
    from being confused with the task/hand/appropriateness review required by
    the production QA specification.
    """

    path = Path(archive).expanduser().resolve()
    if not path.is_file():
        raise SourceQABridgeError(f"archive is not a file: {path}")
    topics = normalize_camera_topics(camera_topics)
    preflight = {} if preflight_path is None else _load_preflight(Path(preflight_path))
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive_file:
        members = [
            info for info in archive_file.infolist() if info.filename.casefold().endswith(".mcap")
        ]
        if max_recordings is not None and (
            isinstance(max_recordings, bool)
            or not isinstance(max_recordings, int)
            or max_recordings <= 0
        ):
            raise SourceQABridgeError("max_recordings must be a positive integer or null")
        selected = []
        for info in members:
            item = preflight.get(info.filename)
            if item is not None and item.get("ok") is not True:
                rows.append(
                    {
                        "member": info.filename,
                        "source_preflight": "FAIL",
                        "status": "FAIL",
                        "qa_admission": "EXCLUDED_SOURCE_PREFLIGHT_FAIL",
                        "reason": item.get("error", "source preflight failed"),
                    }
                )
                continue
            selected.append((info, item))
        if max_recordings is not None:
            selected = selected[:max_recordings]
        for info, item in selected:
            source_ref = f"{path}!{info.filename}"
            try:
                with archive_file.open(info, "r") as stream:
                    scan = _scan_stream(
                        stream,
                        source_ref=source_ref,
                        camera_topics=topics,
                        expected_frames=None
                        if item is None
                        else _expected_frames_from_preflight(item),
                        policy=policy,
                        validate_crcs=validate_crcs,
                    )
            except SourceQABridgeError as exc:
                rows.append(
                    {
                        "member": info.filename,
                        "source_preflight": "PASS" if item is not None else "NOT_RUN",
                        "status": "FAIL",
                        "qa_admission": "EXCLUDED_SOURCE_SCAN_FAIL",
                        "reason": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "member": info.filename,
                    "source_preflight": "PASS" if item is not None else "NOT_RUN",
                    "status": scan["status"],
                    "qa_admission": "PENDING_VISUAL_REVIEW",
                    "objective_only": True,
                    "scan": scan,
                }
            )
    pass_rows = [row for row in rows if row["status"] == "PASS"]
    warning_rows = [row for row in rows if row["status"] == "WARNING"]
    fail_rows = [row for row in rows if row["status"] == "FAIL"]
    return {
        "format": SOURCE_QA_BRIDGE_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "status": "FAIL" if fail_rows else ("WARNING" if warning_rows else "PASS"),
        "qa_admission": "PENDING_VISUAL_REVIEW",
        "source": {
            "archive": str(path),
            "source_bound": True,
            "members_streamed_one_at_a_time": True,
        },
        "counts": {
            "members_total": len(members),
            "members_scanned": len(selected),
            "pass": len(pass_rows),
            "warning": len(warning_rows),
            "fail": len(fail_rows),
        },
        "controls": {
            "model_invoked": False,
            "pixels_persisted": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "content_hash_computed": False,
            "full_visual_qa": False,
        },
        "limitations": [
            "Black, exposure, blur, and freeze values are objective frame proxies.",
            (
                "The bridge cannot decide hand visibility, task completeness, "
                "authenticity, or diversity."
            ),
            (
                "PASS/WARNING here does not admit a recording to production; "
                "visual review remains pending."
            ),
        ],
        "policy": {
            "sample_period_seconds": policy.sample_period_seconds,
            "max_samples_per_camera": policy.max_samples_per_camera,
            "validate_crcs": validate_crcs,
        },
        "items": rows,
    }


__all__ = [
    "AUTHORITY",
    "CAMERA_IDS",
    "DEFAULT_CAMERA_TOPICS",
    "DEFAULT_SOURCE_QA_POLICY",
    "SOURCE_QA_BRIDGE_VERSION",
    "FrameSample",
    "SourceQABridgeError",
    "SourceQAPolicy",
    "normalize_camera_topics",
    "run_archive_source_qa",
]
