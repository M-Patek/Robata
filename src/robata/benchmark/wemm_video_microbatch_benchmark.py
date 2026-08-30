"""Bounded, non-production WeMM video microbatch benchmark helpers.

The production open-vocabulary runner intentionally remains singleton/serial.
This module is an independent experiment seam for comparing that control path
with explicit Batch2/Batch4 calls on an already-resident
:class:`WemmEmbeddingBackend`.  It accepts decoded frame groups so callers can
use a tiny fixture or one bounded 40.8335-second, six-camera cohort without
introducing a second decoder/model process.

Only order-safe metadata and numeric parity/timing facts are returned.  Frame
pixels, hashes/digests, ontology/Mapper fields, and web/API concerns are never
read or persisted here.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .wemm_embedding_backend import (
    WEMM_VIDEO_MICROBATCH_SIZES,
    WemmEmbeddingBackend,
)

WEMM_VIDEO_MICROBATCH_BENCHMARK_FORMAT: Final = "robata-wemm-video-microbatch-benchmark-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
DEFAULT_COHORT_DURATION_SECONDS: Final = 40.8335
DEFAULT_FRAME_COUNT: Final = 4
DEFAULT_BATCH_SIZES: Final = WEMM_VIDEO_MICROBATCH_SIZES
EXPECTED_CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))


class WemmVideoMicrobatchBenchmarkError(ValueError):
    """Raised when a bounded microbatch experiment cannot be prepared."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be an array")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be finite")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be finite")
    return number


def _integer(value: object, *, field: str, minimum: int | None = None) -> int:
    """Parse a strict integer while preserving the benchmark error type."""

    if isinstance(value, bool):
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be an integer")
    # ``int('3')`` is useful for JSON-ish callers, but silently truncating a
    # float (or accepting an arbitrary object) would make the plan ambiguous.
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        token = value.strip()
        digits = token[1:] if token.startswith(("+", "-")) else token
        if not digits or not digits.isdigit():
            raise WemmVideoMicrobatchBenchmarkError(f"{field} must be an integer")
        try:
            result = int(token, 10)
        except (TypeError, ValueError) as exc:
            raise WemmVideoMicrobatchBenchmarkError(f"{field} must be an integer") from exc
    else:
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be an integer")
    if minimum is not None and result < minimum:
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be >= {minimum}")
    return result


def _json_safe(value: object, *, field: str = "value") -> Any:
    """Copy bounded JSON metadata without deriving a content identity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WemmVideoMicrobatchBenchmarkError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise WemmVideoMicrobatchBenchmarkError(f"{field} keys must be strings")
            result[key] = _json_safe(child, field=f"{field}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child, field=f"{field}[]") for child in value]
    raise WemmVideoMicrobatchBenchmarkError(f"{field} must be JSON-compatible")


def _load_json(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WemmVideoMicrobatchBenchmarkError(
                f"could not read cohort manifest {path}: {exc}"
            ) from exc
    return _mapping(value, field="cohort_manifest")


def _row_list(value: object, *, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must be an embedding row")
    result: list[float] = []
    for index, item in enumerate(value):
        number = _finite(item, field=f"{field}[{index}]")
        result.append(number)
    if not result:
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must not be empty")
    return result


def _normalise_rows(
    value: Iterable[Sequence[float]], *, field: str
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(
        tuple(_row_list(row, field=f"{field}[{index}]")) for index, row in enumerate(value)
    )
    if not rows:
        raise WemmVideoMicrobatchBenchmarkError(f"{field} must not be empty")
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise WemmVideoMicrobatchBenchmarkError(f"{field} rows have inconsistent dimensions")
    return rows


def _parity(
    serial_rows: Sequence[Sequence[float]],
    batch_rows: Sequence[Sequence[float]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    if len(serial_rows) != len(batch_rows):
        return {
            "row_count_equal": False,
            "serial_row_count": len(serial_rows),
            "batch_row_count": len(batch_rows),
            "max_abs_delta": None,
            "mean_abs_delta": None,
            "within_tolerance": False,
        }
    deltas: list[float] = []
    for row_index, (serial, batched) in enumerate(zip(serial_rows, batch_rows, strict=True)):
        if len(serial) != len(batched):
            return {
                "row_count_equal": True,
                "dimension_equal": False,
                "mismatched_row": row_index,
                "serial_dimension": len(serial),
                "batch_dimension": len(batched),
                "max_abs_delta": None,
                "mean_abs_delta": None,
                "within_tolerance": False,
            }
        deltas.extend(
            abs(float(left) - float(right)) for left, right in zip(serial, batched, strict=True)
        )
    max_delta = max(deltas, default=0.0)
    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "row_count_equal": True,
        "dimension_equal": True,
        "max_abs_delta": max_delta,
        "mean_abs_delta": mean_delta,
        "within_tolerance": max_delta <= tolerance,
    }


def _observation_slice(backend: Any, start: int) -> list[dict[str, Any]]:
    observations = getattr(backend, "observations", ())
    result: list[dict[str, Any]] = []
    for item in list(observations)[start:]:
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
        elif isinstance(item, Mapping):
            payload = item
        else:
            continue
        if isinstance(payload, Mapping):
            result.append(_json_safe(payload, field="observation"))
    return result


def _context_rows(
    item_contexts: Sequence[Mapping[str, Any]] | None,
    count: int,
) -> list[dict[str, Any]]:
    if item_contexts is None:
        return [{} for _ in range(count)]
    if len(item_contexts) != count:
        raise WemmVideoMicrobatchBenchmarkError(
            "item_contexts must match the number of frame groups"
        )
    rows: list[dict[str, Any]] = []
    for index, context in enumerate(item_contexts):
        rows.append(
            _json_safe(
                dict(_mapping(context, field=f"item_contexts[{index}]")),
                field=f"item_contexts[{index}]",
            )
        )
    return rows


def run_video_microbatch_benchmark(
    backend: WemmEmbeddingBackend,
    frame_groups: Iterable[Sequence[Any]],
    *,
    metadata_groups: Iterable[Mapping[str, Any]] | None = None,
    item_contexts: Sequence[Mapping[str, Any]] | None = None,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare serial video embeddings with explicit Batch2/Batch4 calls.

    ``frame_groups`` is materialised once because the control and candidate
    arms intentionally consume identical decoded objects.  Callers should
    keep this to a bounded window/chunk (for example one 4-second window from
    the 40.8335-second six-camera fixture), not an entire production archive.
    ``item_contexts`` is optional JSON metadata used solely to prove that
    camera/window/timestamp ordering survives the comparison.
    """

    if not isinstance(backend, WemmEmbeddingBackend) and not hasattr(
        backend, "encode_video_frames"
    ):
        raise WemmVideoMicrobatchBenchmarkError("backend lacks encode_video_frames")
    tolerance = _finite(tolerance, field="tolerance")
    if tolerance < 0:
        raise WemmVideoMicrobatchBenchmarkError("tolerance must be non-negative")
    requested_sizes: list[int] = []
    for index, raw_size in enumerate(batch_sizes):
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size <= 0:
            raise WemmVideoMicrobatchBenchmarkError(
                f"batch_sizes[{index}] must be a positive integer"
            )
        if raw_size not in requested_sizes:
            requested_sizes.append(raw_size)
    if not requested_sizes:
        raise WemmVideoMicrobatchBenchmarkError("batch_sizes must not be empty")
    if not hasattr(backend, "encode_video_frames_batch"):
        raise WemmVideoMicrobatchBenchmarkError(
            "backend lacks the opt-in encode_video_frames_batch seam"
        )

    groups = tuple(tuple(group) for group in frame_groups)
    if not groups:
        raise WemmVideoMicrobatchBenchmarkError("frame_groups must not be empty")
    contexts = _context_rows(item_contexts, len(groups))
    if metadata_groups is None:
        metadata = None
    else:
        metadata = tuple(
            dict(_mapping(item, field="metadata_groups[]")) for item in metadata_groups
        )
        if len(metadata) != len(groups):
            raise WemmVideoMicrobatchBenchmarkError(
                "metadata_groups must match the number of frame groups"
            )

    serial_observation_start = len(getattr(backend, "observations", ()))
    serial_started = time.perf_counter()
    serial_raw = backend.encode_video_frames(groups, metadata_groups=metadata)
    serial_elapsed = time.perf_counter() - serial_started
    serial_rows = _normalise_rows(serial_raw, field="serial_rows")
    serial_observations = _observation_slice(backend, serial_observation_start)

    arms: list[dict[str, Any]] = []
    for requested_size in requested_sizes:
        observation_start = len(getattr(backend, "observations", ()))
        started = time.perf_counter()
        batch_raw = backend.encode_video_frames_batch(
            groups,
            metadata_groups=metadata,
            batch_size=requested_size,
        )
        elapsed = time.perf_counter() - started
        batch_rows = _normalise_rows(batch_raw, field=f"batch{requested_size}_rows")
        observations = _observation_slice(backend, observation_start)
        # Flatten only bounded row-order context; frame contents are never
        # represented in this report.
        arms.append(
            {
                "arm_id": f"batch{requested_size}",
                "batch_size": requested_size,
                "input_count": len(groups),
                "output_count": len(batch_rows),
                "elapsed_seconds": elapsed,
                "serial_elapsed_seconds": serial_elapsed,
                "speedup_vs_serial": (serial_elapsed / elapsed if elapsed > 0 else None),
                "parity": _parity(serial_rows, batch_rows, tolerance=tolerance),
                "observations": observations,
                "ordered_items": contexts,
            }
        )

    return {
        "format": WEMM_VIDEO_MICROBATCH_BENCHMARK_FORMAT,
        "authority": AUTHORITY,
        "status": "MEASURED_NONPRODUCTION" if arms else "NOT_MEASURED",
        "production_eligible": False,
        "controls": {
            "model_invoked": True,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "epic_ontology_used": False,
            "mapper_used": False,
            "hash_or_sha_used": False,
        },
        "source": {
            "input_count": len(groups),
            "camera_window_timestamp_context_preserved": item_contexts is not None,
            "bounded_input_required": True,
        },
        "control": {
            "arm_id": "serial",
            "batch_size": 1,
            "input_count": len(groups),
            "output_count": len(serial_rows),
            "elapsed_seconds": serial_elapsed,
            "observations": serial_observations,
            "ordered_items": contexts,
        },
        "arms": arms,
        "config": {
            "batch_sizes": requested_sizes,
            "tolerance": tolerance,
            "intended_arms": list(DEFAULT_BATCH_SIZES),
        },
        "limitations": [
            "This is an opt-in benchmark seam; the production open runner remains serial.",
            "Rows are embedding parity diagnostics, not calibrated quality or gold accuracy.",
            "Only bounded decoded groups should be supplied; no archive-wide "
            "accumulation is intended.",
        ],
    }


def run_decoded_video_microbatch_benchmark(
    backend: WemmEmbeddingBackend,
    decoded_groups: Mapping[str, Mapping[str, Any]],
    *,
    camera_order: Sequence[str] | None = None,
    window_order: Sequence[str] | None = None,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Benchmark a decoder-shaped ``camera -> window -> frame-group`` map.

    ``iter_decode_production_window_chunks`` already returns this shape.  The
    helper flattens it in the same window-major/camera-minor order used by the
    open runner, then delegates to :func:`run_video_microbatch_benchmark`.
    Each group's ``metadata()`` and ``to_dict()`` methods are used when
    available, so selected timestamps and camera/window IDs remain visible in
    the bounded report without exposing frames.
    """

    cameras = tuple(camera_order or sorted(str(key) for key in decoded_groups))
    if not cameras:
        raise WemmVideoMicrobatchBenchmarkError("decoded_groups must contain at least one camera")
    for camera_id in cameras:
        if camera_id not in decoded_groups:
            raise WemmVideoMicrobatchBenchmarkError(
                f"decoded_groups is missing camera {camera_id!r}"
            )
    first_camera = _mapping(decoded_groups[cameras[0]], field=f"decoded_groups.{cameras[0]}")
    windows = tuple(window_order or (str(key) for key in first_camera))
    if not windows:
        raise WemmVideoMicrobatchBenchmarkError("decoded_groups must contain at least one window")

    frame_groups: list[Sequence[Any]] = []
    metadata_groups: list[Mapping[str, Any]] = []
    contexts: list[Mapping[str, Any]] = []
    for window_id in windows:
        for camera_id in cameras:
            camera_map = _mapping(decoded_groups[camera_id], field=f"decoded_groups.{camera_id}")
            if window_id not in camera_map:
                raise WemmVideoMicrobatchBenchmarkError(
                    f"camera {camera_id!r} is missing window {window_id!r}"
                )
            group = camera_map[window_id]
            frames = getattr(group, "frames", None)
            if frames is None and isinstance(group, Mapping):
                frames = group.get("frames")
            if frames is None:
                raise WemmVideoMicrobatchBenchmarkError(
                    f"decoded group {camera_id}/{window_id} has no frames"
                )
            metadata_fn = getattr(group, "metadata", None)
            metadata_value = metadata_fn() if callable(metadata_fn) else {}
            metadata_groups.append(
                dict(_mapping(metadata_value, field=f"metadata[{camera_id}/{window_id}]"))
            )
            context_fn = getattr(group, "to_dict", None)
            context_value = context_fn() if callable(context_fn) else {}
            context = dict(_mapping(context_value, field=f"context[{camera_id}/{window_id}]"))
            context.setdefault("camera_id", camera_id)
            context.setdefault("window_id", window_id)
            contexts.append(context)
            frame_groups.append(tuple(frames))

    report = run_video_microbatch_benchmark(
        backend,
        frame_groups,
        metadata_groups=metadata_groups,
        item_contexts=contexts,
        batch_sizes=batch_sizes,
        tolerance=tolerance,
    )
    source = dict(_mapping(report.get("source"), field="report.source"))
    source.update(
        {
            "camera_count": len(cameras),
            "camera_ids": list(cameras),
            "window_count": len(windows),
            "window_ids": list(windows),
            "flatten_order": "window_major_camera_minor",
        }
    )
    report["source"] = source
    return report


def build_cohort_microbatch_plan(
    manifest: Mapping[str, Any] | str | Path,
    *,
    max_duration_seconds: float = DEFAULT_COHORT_DURATION_SECONDS,
    frame_count: int = DEFAULT_FRAME_COUNT,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
) -> dict[str, Any]:
    """Build a no-model plan for the bounded 40.8335s six-camera fixture.

    The plan reads only manifest geometry and window metadata.  It is useful
    for preflight/CI and deliberately does not decode media or instantiate a
    model.  ``f*_high`` arms are processor probes, not claims that a given cap
    will resolve to that grid on every Transformers revision.
    """

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 2:
        raise WemmVideoMicrobatchBenchmarkError("frame_count must be at least two")
    max_duration_seconds = _finite(max_duration_seconds, field="max_duration_seconds")
    if max_duration_seconds <= 0:
        raise WemmVideoMicrobatchBenchmarkError("max_duration_seconds must be positive")
    body = _load_json(manifest)
    source = _mapping(body.get("source"), field="cohort_manifest.source")
    camera_count = _integer(
        source.get("camera_count", 0) or 0,
        field="cohort_manifest.source.camera_count",
        minimum=0,
    )
    if camera_count != len(EXPECTED_CAMERA_IDS):
        raise WemmVideoMicrobatchBenchmarkError(
            f"cohort must declare six cameras, got {camera_count}"
        )
    cameras = _sequence(source.get("cameras"), field="cohort_manifest.source.cameras")
    camera_ids = tuple(str(_mapping(camera, field="camera").get("camera_id")) for camera in cameras)
    if camera_ids != EXPECTED_CAMERA_IDS:
        raise WemmVideoMicrobatchBenchmarkError(
            "cohort cameras must be cam_01 through cam_06 in order"
        )
    duration = _finite(
        source.get("common_duration_seconds"),
        field="cohort_manifest.source.common_duration_seconds",
    )
    if duration > max_duration_seconds + 1e-3:
        raise WemmVideoMicrobatchBenchmarkError(
            f"cohort duration {duration:.6f}s exceeds bounded limit {max_duration_seconds:.6f}s"
        )
    windows_raw = _sequence(body.get("windows"), field="cohort_manifest.windows")
    windows: list[dict[str, Any]] = []
    previous_ordinal = -1
    for index, raw in enumerate(windows_raw):
        window = _mapping(raw, field=f"cohort_manifest.windows[{index}]")
        ordinal = _integer(
            window.get("ordinal", index),
            field=f"cohort_manifest.windows[{index}].ordinal",
            minimum=0,
        )
        if ordinal <= previous_ordinal:
            raise WemmVideoMicrobatchBenchmarkError("windows must be in ordinal order")
        previous_ordinal = ordinal
        window_id = str(window.get("window_id", "")).strip()
        if not window_id:
            raise WemmVideoMicrobatchBenchmarkError(f"windows[{index}].window_id is required")
        start = _finite(window.get("start_seconds"), field=f"{window_id}.start_seconds")
        end = _finite(window.get("end_seconds"), field=f"{window_id}.end_seconds")
        if end <= start:
            raise WemmVideoMicrobatchBenchmarkError(f"{window_id} must have positive duration")
        declared_cameras = tuple(
            str(item)
            for item in _sequence(
                window.get("camera_ids", EXPECTED_CAMERA_IDS), field=f"{window_id}.camera_ids"
            )
        )
        if declared_cameras != EXPECTED_CAMERA_IDS:
            raise WemmVideoMicrobatchBenchmarkError(
                f"{window_id}.camera_ids must preserve six-camera order"
            )
        windows.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "start_seconds": start,
                "end_seconds": end,
                "camera_ids": list(EXPECTED_CAMERA_IDS),
            }
        )
    requested_sizes: list[int] = []
    for index, size in enumerate(batch_sizes):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise WemmVideoMicrobatchBenchmarkError(
                f"batch_sizes[{index}] must be a positive integer"
            )
        if size not in requested_sizes:
            requested_sizes.append(size)
    if not requested_sizes:
        raise WemmVideoMicrobatchBenchmarkError("batch_sizes must not be empty")

    items = [
        {
            **window,
            "camera_id": camera_id,
            # The context is copied, not interpreted as a semantic label.
            "timestamps": {
                "start_seconds": window["start_seconds"],
                "end_seconds": window["end_seconds"],
            },
        }
        for window in windows
        for camera_id in EXPECTED_CAMERA_IDS
    ]
    matrix = [
        {
            "arm_id": "f4_current",
            "frame_count": 4,
            "grid_status": "observed_fixture_default",
            "expected_video_grid_thw": [[2, 14, 16]],
            "spatial_cap": "current",
        },
        {
            "arm_id": "f8_current",
            "frame_count": 8,
            "grid_status": "probe_expected",
            "expected_video_grid_thw": [[4, 10, 12]],
            "spatial_cap": "current",
        },
        {
            "arm_id": "f4_high",
            "frame_count": 4,
            "grid_status": "probe_expected",
            "expected_video_grid_thw": [[2, 20, 24]],
            "spatial_cap": "higher_probe",
        },
        {
            "arm_id": "f8_high",
            "frame_count": 8,
            "grid_status": "probe_expected",
            "expected_video_grid_thw": [[4, 14, 16]],
            "spatial_cap": "higher_probe",
        },
    ]
    return {
        "format": WEMM_VIDEO_MICROBATCH_BENCHMARK_FORMAT,
        "authority": AUTHORITY,
        "status": "PLAN_ONLY",
        "production_eligible": False,
        "source": {
            "manifest_format": body.get("format"),
            "path": source.get("path"),
            "common_duration_seconds": duration,
            "duration_limit_seconds": max_duration_seconds,
            "camera_count": len(EXPECTED_CAMERA_IDS),
            "camera_ids": list(EXPECTED_CAMERA_IDS),
            "window_count": len(windows),
            "camera_window_input_count": len(items),
        },
        "config": {
            "frame_count": frame_count,
            "batch_sizes": requested_sizes,
            "intended_batch_arms": [f"batch{size}" for size in requested_sizes],
        },
        "windows": windows,
        "items": items,
        "matrix": matrix,
        "controls": {
            "media_decoded": False,
            "model_invoked": False,
            "gold_read": False,
            "gold_written": False,
            "hash_or_sha_used": False,
            "ontology_used": False,
            "mapper_used": False,
        },
        "limitations": [
            "Plan-only geometry; processor grids for probe arms must be measured, not assumed.",
            "No action boundaries, quality, gold, ontology, Mapper, or identity claims are made.",
            "Run only a bounded fixture/chunk; do not use this plan to launch a "
            "full archive sweep.",
        ],
    }


__all__ = [
    "AUTHORITY",
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_COHORT_DURATION_SECONDS",
    "DEFAULT_FRAME_COUNT",
    "EXPECTED_CAMERA_IDS",
    "WEMM_VIDEO_MICROBATCH_BENCHMARK_FORMAT",
    "WemmVideoMicrobatchBenchmarkError",
    "build_cohort_microbatch_plan",
    "run_decoded_video_microbatch_benchmark",
    "run_video_microbatch_benchmark",
]
