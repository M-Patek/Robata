"""Build a label-neutral, six-camera production-shaped cohort manifest.

The manifest is deliberately independent of any model prediction.  It binds
only source topics and time windows; human labels remain pending until a
reviewer supplies them.  This lets WeMM, Qwen and Mage consume the same
windows without treating one model's output as another model's ground truth.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProductionCohortError(ValueError):
    """Raised for malformed source/window inputs."""


DEFAULT_CAMERA_TOPICS: dict[str, str] = {
    "cam_01": "/robot0/sensor/camera0/compressed",
    "cam_02": "/robot0/sensor/camera1/compressed",
    "cam_03": "/robot0/sensor/camera2/compressed",
    "cam_04": "/robot0/sensor/camera3/compressed",
    "cam_05": "/robot0/sensor/camera4/compressed",
    "cam_06": "/robot0/sensor/camera5/compressed",
}


@dataclass(frozen=True, slots=True)
class CameraSpan:
    camera_id: str
    topic: str
    frame_count: int
    first_timestamp_ns: int
    last_timestamp_ns: int

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.last_timestamp_ns - self.first_timestamp_ns) / 1_000_000_000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "topic": self.topic,
            "frame_count": self.frame_count,
            "first_timestamp_ns": str(self.first_timestamp_ns),
            "last_timestamp_ns": str(self.last_timestamp_ns),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class ProductionWindow:
    ordinal: int
    window_id: str
    start_seconds: float
    end_seconds: float
    camera_ids: tuple[str, ...]
    camera_topics: Mapping[str, str]
    gold_status: str = "PENDING_HUMAN_REVIEW"

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "window_id": self.window_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "camera_ids": list(self.camera_ids),
            "camera_topics": dict(self.camera_topics),
            "gold_status": self.gold_status,
            "review": {
                "required": True,
                "qa_status": "PENDING",
                "segments": [],
                "notes": None,
            },
            "model_routes": {
                "wemm": "complete_bounded_video_embedding",
                "qwen": "complete_native_video",
                "mage": "complete_bounded_native_codec",
            },
        }


def common_camera_span(spans: Sequence[CameraSpan]) -> tuple[int, int]:
    """Return the intersection of all camera timelines."""

    if not spans:
        raise ProductionCohortError("at least one camera span is required")
    start = max(item.first_timestamp_ns for item in spans)
    end = min(item.last_timestamp_ns for item in spans)
    if end <= start:
        raise ProductionCohortError("camera timelines have no common interval")
    return start, end


def build_windows(
    spans: Sequence[CameraSpan],
    *,
    window_seconds: float = 8.0,
    include_tail: bool = False,
    window_stride_seconds: float | None = None,
    window_prefix: str = "sample-medium",
) -> tuple[ProductionWindow, ...]:
    """Build non-overlapping windows over the common camera interval.

    The default five-window policy mirrors the existing 40-second Qwen/Mage
    vertical run.  ``include_tail=True`` adds a final short window rather than
    silently discarding the remainder.
    """

    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ProductionCohortError("window_seconds must be positive and finite")
    stride = window_seconds if window_stride_seconds is None else float(window_stride_seconds)
    if not math.isfinite(stride) or stride <= 0:
        raise ProductionCohortError("window_stride_seconds must be positive and finite")
    if stride > window_seconds:
        raise ProductionCohortError("window_stride_seconds must be <= window_seconds")
    start_ns, end_ns = common_camera_span(spans)
    origin_ns = start_ns
    duration = (end_ns - origin_ns) / 1_000_000_000
    # Retain the historical exact non-overlapping algorithm byte-for-byte when
    # stride is omitted/equal to the context width.  Dense temporal mode uses
    # the second branch and intentionally creates overlapping context windows;
    # these remain processing envelopes, never action boundaries.
    full_count = int(duration // window_seconds) if abs(stride - window_seconds) <= 1e-9 else 0
    windows: list[ProductionWindow] = []
    camera_ids = tuple(item.camera_id for item in spans)
    camera_topics = {item.camera_id: item.topic for item in spans}
    if abs(stride - window_seconds) <= 1e-9:
        for ordinal in range(full_count):
            left = ordinal * window_seconds
            right = min(duration, (ordinal + 1) * window_seconds)
            windows.append(
                ProductionWindow(
                    ordinal=ordinal,
                    window_id=f"{window_prefix}-w{ordinal:02d}",
                    start_seconds=left,
                    end_seconds=right,
                    camera_ids=camera_ids,
                    camera_topics=camera_topics,
                )
            )
        tail = duration - full_count * window_seconds
        if include_tail and tail > 1e-6:
            windows.append(
                ProductionWindow(
                    ordinal=full_count,
                    window_id=f"{window_prefix}-w{full_count:02d}-tail",
                    start_seconds=full_count * window_seconds,
                    end_seconds=duration,
                    camera_ids=camera_ids,
                    camera_topics=camera_topics,
                )
            )
    else:
        epsilon = 1e-9
        ordinal = 0
        # Emit every full context on the requested stride.  Once the next
        # stride would produce a short context, keep at most one final tail
        # envelope.  Emitting every short stride (for example starts 3, 4, 5
        # on a 6 s source with a 4 s/1 s grid) needlessly multiplies decode and
        # inference cost while adding no new full-width temporal evidence.
        max_full_start = max(0.0, duration - window_seconds)
        left = 0.0
        emitted_full = False
        while left <= max_full_start + epsilon:
            right = min(duration, left + window_seconds)
            if right - left >= window_seconds - epsilon:
                windows.append(
                    ProductionWindow(
                        ordinal=ordinal,
                        window_id=f"{window_prefix}-w{ordinal:03d}",
                        start_seconds=left,
                        end_seconds=right,
                        camera_ids=camera_ids,
                        camera_topics=camera_topics,
                    )
                )
                ordinal += 1
                emitted_full = True
            left += stride
        if include_tail:
            # A source shorter than one full context still gets one explicit
            # tail when requested, matching the legacy route.
            tail_start = 0.0 if not emitted_full else duration - window_seconds
            tail_start = max(0.0, tail_start)
            if not windows or tail_start > windows[-1].start_seconds + epsilon:
                windows.append(
                    ProductionWindow(
                        ordinal=ordinal,
                        window_id=f"{window_prefix}-w{ordinal:03d}-tail",
                        start_seconds=tail_start,
                        end_seconds=duration,
                        camera_ids=camera_ids,
                        camera_topics=camera_topics,
                    )
                )
    if not windows:
        raise ProductionCohortError("common interval is shorter than one requested window")
    return tuple(windows)


def inspect_mcap_camera_spans(
    source: str | Path,
    *,
    camera_topics: Mapping[str, str] = DEFAULT_CAMERA_TOPICS,
) -> tuple[CameraSpan, ...]:
    """Read camera timestamps/counts from MCAP without hashing or decoding frames."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ProductionCohortError(f"MCAP source is not a file: {path}")
    if not camera_topics:
        raise ProductionCohortError("camera_topics must not be empty")
    try:
        from mcap.reader import make_reader
    except Exception as exc:  # pragma: no cover - optional dependency boundary
        raise ProductionCohortError("mcap package is required for source inspection") from exc

    topic_to_camera = {str(topic): str(camera_id) for camera_id, topic in camera_topics.items()}
    timestamps: dict[str, list[int]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    try:
        with path.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=False)
            for _schema, channel, message in reader.iter_messages(log_time_order=False):
                camera_id = topic_to_camera.get(channel.topic)
                if camera_id is None:
                    continue
                counts[camera_id] += 1
                timestamps[camera_id].append(int(message.log_time))
    except OSError as exc:
        raise ProductionCohortError(f"could not read MCAP source {path}: {exc}") from exc

    missing = [camera_id for camera_id in camera_topics if not timestamps.get(camera_id)]
    if missing:
        raise ProductionCohortError("missing mapped camera topics: " + ", ".join(missing))
    return tuple(
        CameraSpan(
            camera_id=camera_id,
            topic=str(camera_topics[camera_id]),
            frame_count=counts[camera_id],
            first_timestamp_ns=min(timestamps[camera_id]),
            last_timestamp_ns=max(timestamps[camera_id]),
        )
        for camera_id in camera_topics
    )


def build_manifest(
    source: str | Path,
    *,
    window_seconds: float = 8.0,
    include_tail: bool = False,
    window_stride_seconds: float | None = None,
    camera_topics: Mapping[str, str] = DEFAULT_CAMERA_TOPICS,
) -> dict[str, Any]:
    """Inspect one source and return a serialisable production cohort manifest."""

    path = Path(source).expanduser().resolve()
    spans = inspect_mcap_camera_spans(path, camera_topics=camera_topics)
    windows = build_windows(
        spans,
        window_seconds=window_seconds,
        include_tail=include_tail,
        window_stride_seconds=window_stride_seconds,
    )
    start_ns, end_ns = common_camera_span(spans)
    common_duration = (end_ns - start_ns) / 1_000_000_000
    represented = sum(item.duration_seconds for item in windows)
    ordered_intervals = sorted((item.start_seconds, item.end_seconds) for item in windows)
    covered_seconds = 0.0
    covered_start: float | None = None
    covered_end: float | None = None
    for left, right in ordered_intervals:
        if covered_start is None or covered_end is None:
            covered_start, covered_end = left, right
        elif left <= covered_end + 1e-9:
            covered_end = max(covered_end, right)
        else:
            covered_seconds += covered_end - covered_start
            covered_start, covered_end = left, right
    if covered_start is not None and covered_end is not None:
        covered_seconds += covered_end - covered_start
    return {
        "format": "robata-production-shaped-cohort-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {
            "path": str(path),
            "media_type": "application/x-mcap",
            "camera_count": len(spans),
            "cameras": [item.to_dict() for item in spans],
            "common_start_timestamp_ns": str(start_ns),
            "common_end_timestamp_ns": str(end_ns),
            "common_duration_seconds": common_duration,
        },
        "window_policy": {
            "window_seconds": window_seconds,
            "window_stride_seconds": (
                window_seconds if window_stride_seconds is None else window_stride_seconds
            ),
            "overlap_seconds": max(
                0.0,
                window_seconds
                - (window_seconds if window_stride_seconds is None else window_stride_seconds),
            ),
            "context_windows_not_action_boundaries": True,
            "include_tail": include_tail,
            "represented_duration_seconds": represented,
            "context_workload_seconds": represented,
            "unique_context_coverage_seconds": covered_seconds,
            "overlap_workload_seconds": max(0.0, represented - covered_seconds),
            "excluded_tail_seconds": max(0.0, common_duration - represented),
        },
        "gold": {
            "status": "PENDING_HUMAN_REVIEW",
            "structured_label_fields": ["verb", "noun", "attributes", "location", "hand"],
            "boundary_fields": ["start_seconds", "end_seconds"],
            "model_predictions_are_not_gold": True,
        },
        "windows": [item.to_dict() for item in windows],
        "controls": {
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "sha_or_digest_computed": False,
            "frames_decoded": False,
        },
    }


__all__ = [
    "DEFAULT_CAMERA_TOPICS",
    "CameraSpan",
    "ProductionCohortError",
    "ProductionWindow",
    "build_manifest",
    "build_windows",
    "common_camera_span",
    "inspect_mcap_camera_spans",
]
