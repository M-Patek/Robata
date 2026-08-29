"""Bounded, source-shaped WeMM runtime benchmark.

This module is deliberately narrower than the production pre-annotation route:
it measures the cost and ordering of singleton versus native video microbatch
inference on the ten 4-second windows of the local six-camera cohort.  Decode
is performed once per benchmark pass and the same decoded groups feed the
serial and batch controls, so the comparison does not accidentally mix media
variance with model variance.  An explicit process-local decode cache may be
provided when a caller is comparing multiple frame/grid arms; the cache is
never enabled implicitly.  No gold, EPIC ontology, Mapper, pixels, embeddings,
hashes, or archive-wide inputs are written to the report.

The benchmark is diagnostic only.  The cohort has no established visual gold;
rank metrics below are therefore *consistency* metrics against the selected
provisional phrase catalog, not accuracy claims.
"""

from __future__ import annotations

import gc
import importlib
import json
import math
import os
import time
from collections.abc import Hashable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Final

from .production_wemm_decode_cache import ProductionWemmDecodeCache
from .production_wemm_open_runner import load_open_phrase_catalog
from .production_wemm_shadow import iter_decode_production_window_chunks
from .wemm_embedding_backend import WemmEmbeddingBackend
from .wemm_pipeline_benchmark import PipelinePhase, run_bounded_pipeline

FORMAT: Final = "robata-wemm-cohort-runtime-benchmark-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
COHORT_FORMAT: Final = "robata-production-shaped-cohort-v1"
CAMERA_IDS: Final = tuple(f"cam_{index:02d}" for index in range(1, 7))
DEFAULT_BATCH_SIZES: Final = (2, 4)
DEFAULT_PIXEL_BUDGET: Final = 256 * 32 * 32
DEFAULT_FRAME_COUNT: Final = 4
DEFAULT_DIMENSION: Final = 2048
MIN_FRAME_COUNT: Final = 2
MAX_FRAME_COUNT: Final = 64
MAX_WINDOW_COUNT: Final = 10
MAX_BATCH_SIZE: Final = 64
MAX_COHORT_DURATION_SECONDS: Final = 40.8335
DEFAULT_PARITY_TOLERANCE: Final = 1e-5
MAX_PARITY_MISMATCHES: Final = 16


class WemmCohortRuntimeBenchmarkError(ValueError):
    """Raised when a bounded cohort benchmark cannot be prepared."""


def _process_rss_bytes() -> int | None:
    """Return the current process RSS when the optional probe is available.

    ``psutil`` is intentionally optional: the benchmark must remain runnable in
    the core test environment without adding a dependency.  The probe is
    sampled only at arm boundaries, so it does not create a worker thread or a
    measurable hot-path cost.
    """

    try:
        psutil = importlib.import_module("psutil")
        value = int(psutil.Process(os.getpid()).memory_info().rss)
    except (ImportError, OSError, AttributeError, TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _cuda_memory_snapshot(backend: object) -> dict[str, Any]:
    """Read bounded CUDA allocator/device facts from a resident backend.

    The WeMM backend keeps its lazily imported torch module on ``_torch``.  A
    fake backend (used by unit tests) or a CPU run simply reports
    ``UNAVAILABLE``; this is preferable to fabricating zeroes.  All reads are
    best-effort and contain only allocator counters, never tensors or content.
    """

    torch = getattr(backend, "_torch", None)
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return {"status": "UNAVAILABLE", "reason": "torch_cuda_unavailable"}
    try:
        if not bool(is_available()):
            return {"status": "UNAVAILABLE", "reason": "cuda_not_available"}
        memory_allocated = getattr(cuda, "memory_allocated", None)
        memory_reserved = getattr(cuda, "memory_reserved", None)
        mem_get_info = getattr(cuda, "mem_get_info", None)
        if not callable(memory_allocated) or not callable(memory_reserved):
            return {"status": "UNAVAILABLE", "reason": "allocator_api_missing"}
        allocated = int(memory_allocated())
        reserved = int(memory_reserved())
        free: int | None = None
        total: int | None = None
        if callable(mem_get_info):
            raw_free, raw_total = mem_get_info()
            free = int(raw_free)
            total = int(raw_total)
        name_fn = getattr(cuda, "get_device_name", None)
        device_name: str | None = None
        if callable(name_fn):
            with suppress(Exception):
                device_name = str(name_fn(0))
        payload: dict[str, Any] = {
            "status": "AVAILABLE",
            "allocated_bytes": max(0, allocated),
            "reserved_bytes": max(0, reserved),
            "free_bytes": None if free is None else max(0, free),
            "total_bytes": None if total is None else max(0, total),
            "device_name": device_name,
            "source": "torch.cuda",
        }
        return payload
    except (RuntimeError, OSError, TypeError, ValueError):
        return {"status": "ERROR", "reason": "cuda_probe_failed"}


def _cuda_peak_bytes(backend: object) -> dict[str, int | None]:
    """Read allocator peaks after an arm, without requiring CUDA in tests."""

    torch = getattr(backend, "_torch", None)
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    try:
        if not bool(is_available()):
            return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        allocated_fn = getattr(cuda, "max_memory_allocated", None)
        reserved_fn = getattr(cuda, "max_memory_reserved", None)
        return {
            "peak_allocated_bytes": (
                max(0, int(allocated_fn())) if callable(allocated_fn) else None
            ),
            "peak_reserved_bytes": (max(0, int(reserved_fn())) if callable(reserved_fn) else None),
        }
    except (RuntimeError, OSError, TypeError, ValueError):
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}


def _reset_cuda_peak(backend: object) -> None:
    """Reset allocator peaks before one measured arm when supported."""

    torch = getattr(backend, "_torch", None)
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    reset = getattr(cuda, "reset_peak_memory_stats", None)
    if not callable(is_available) or not callable(reset):
        return
    try:
        if bool(is_available()):
            reset()
    except (RuntimeError, OSError, TypeError, ValueError):
        return


def _arm_memory_telemetry(backend: object, before: Mapping[str, Any]) -> dict[str, Any]:
    """Combine arm-boundary CPU RSS and CUDA allocator observations."""

    cpu_before_raw = before.get("cpu")
    cpu_before: Mapping[str, Any] = cpu_before_raw if isinstance(cpu_before_raw, Mapping) else {}
    cpu_after_rss = _process_rss_bytes()
    cpu_before_rss = cpu_before.get("rss_bytes")
    cpu_delta = (
        cpu_after_rss - int(cpu_before_rss)
        if isinstance(cpu_before_rss, int) and cpu_after_rss is not None
        else None
    )
    cpu: dict[str, Any] = {
        "status": "AVAILABLE" if cpu_after_rss is not None else "UNAVAILABLE",
        "rss_before_bytes": cpu_before_rss,
        "rss_after_bytes": cpu_after_rss,
        "rss_delta_bytes": cpu_delta,
        "source": "psutil.Process.memory_info.rss" if cpu_after_rss is not None else None,
        "peak_scope": "boundary_samples_only",
    }
    gpu_before_raw = before.get("gpu")
    gpu_before: Mapping[str, Any] = gpu_before_raw if isinstance(gpu_before_raw, Mapping) else {}
    gpu_after = _cuda_memory_snapshot(backend)
    gpu_peak = _cuda_peak_bytes(backend)
    gpu: dict[str, Any] = {
        "status": gpu_after.get("status", "UNAVAILABLE"),
        "device_name": gpu_after.get("device_name") or gpu_before.get("device_name"),
        "allocated_before_bytes": gpu_before.get("allocated_bytes"),
        "allocated_after_bytes": gpu_after.get("allocated_bytes"),
        "peak_allocated_bytes": gpu_peak.get("peak_allocated_bytes"),
        "reserved_before_bytes": gpu_before.get("reserved_bytes"),
        "reserved_after_bytes": gpu_after.get("reserved_bytes"),
        "peak_reserved_bytes": gpu_peak.get("peak_reserved_bytes"),
        "free_before_bytes": gpu_before.get("free_bytes"),
        "free_after_bytes": gpu_after.get("free_bytes"),
        "total_bytes": gpu_after.get("total_bytes") or gpu_before.get("total_bytes"),
        "source": "torch.cuda" if gpu_after.get("status") == "AVAILABLE" else None,
    }
    statuses = {str(cpu["status"]), str(gpu["status"])}
    return {
        "status": "AVAILABLE" if "AVAILABLE" in statuses else "UNAVAILABLE",
        "cpu": cpu,
        "gpu": gpu,
        "notes": [
            "CPU RSS is sampled at arm boundaries; it is not a process peak.",
            "GPU peak counters are reset before the arm and read after synchronized inference.",
        ],
    }


def _merge_arm_memory(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Fold one chunk's arm-boundary telemetry into an arm-wide summary."""

    if previous is None:
        return dict(current)
    old_cpu_raw = previous.get("cpu")
    new_cpu_raw = current.get("cpu")
    old_gpu_raw = previous.get("gpu")
    new_gpu_raw = current.get("gpu")
    old_cpu: Mapping[str, Any] = old_cpu_raw if isinstance(old_cpu_raw, Mapping) else {}
    new_cpu: Mapping[str, Any] = new_cpu_raw if isinstance(new_cpu_raw, Mapping) else {}
    old_gpu: Mapping[str, Any] = old_gpu_raw if isinstance(old_gpu_raw, Mapping) else {}
    new_gpu: Mapping[str, Any] = new_gpu_raw if isinstance(new_gpu_raw, Mapping) else {}
    cpu_before = old_cpu.get("rss_before_bytes")
    cpu_after = new_cpu.get("rss_after_bytes")
    cpu_delta = (
        cpu_after - int(cpu_before)
        if isinstance(cpu_before, int) and isinstance(cpu_after, int)
        else None
    )
    cpu = {
        **dict(old_cpu),
        "status": "AVAILABLE"
        if old_cpu.get("status") == "AVAILABLE" or new_cpu.get("status") == "AVAILABLE"
        else "UNAVAILABLE",
        "rss_after_bytes": cpu_after,
        "rss_delta_bytes": cpu_delta,
    }
    gpu: dict[str, Any] = dict(old_gpu)
    for key in ("status", "device_name", "source"):
        if new_gpu.get(key) not in (None, "UNAVAILABLE"):
            gpu[key] = new_gpu.get(key)
    for key in ("allocated_after_bytes", "reserved_after_bytes", "free_after_bytes"):
        gpu[key] = new_gpu.get(key)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [value for value in (old_gpu.get(key), new_gpu.get(key)) if isinstance(value, int)]
        gpu[key] = max(values) if values else None
    return {
        **dict(previous),
        "status": "AVAILABLE"
        if previous.get("status") == "AVAILABLE" or current.get("status") == "AVAILABLE"
        else "UNAVAILABLE",
        "cpu": cpu,
        "gpu": gpu,
    }


def _arm_memory_before(backend: object) -> dict[str, Any]:
    """Capture the low-cost pre-arm memory snapshot."""

    return {"cpu": {"rss_bytes": _process_rss_bytes()}, "gpu": _cuda_memory_snapshot(backend)}


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WemmCohortRuntimeBenchmarkError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WemmCohortRuntimeBenchmarkError(f"{field} must be an array")
    return value


def _positive_int(value: object, *, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WemmCohortRuntimeBenchmarkError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise WemmCohortRuntimeBenchmarkError(f"{field} must be <= {maximum}")
    return value


def _finite(value: object, *, field: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WemmCohortRuntimeBenchmarkError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise WemmCohortRuntimeBenchmarkError(f"{field} must be finite")
    return result


def _json_safe(value: object, *, field: str) -> Any:
    """Copy report metadata without retaining frames, tensors, or identities."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WemmCohortRuntimeBenchmarkError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise WemmCohortRuntimeBenchmarkError(f"{field} keys must be strings")
            copied[key] = _json_safe(child, field=f"{field}.{key}")
        return copied
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child, field=f"{field}[]") for child in value]
    raise WemmCohortRuntimeBenchmarkError(f"{field} must be JSON-compatible")


def _observation_dict(value: object, *, field: str) -> dict[str, Any]:
    """Project one backend observation to JSON-safe diagnostic metadata."""

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    return dict(_json_safe(_mapping(value, field=field), field=field))


def _group_frames(group: object, *, field: str) -> Sequence[Any]:
    frames = getattr(group, "frames", None)
    if frames is None and isinstance(group, Mapping):
        frames = group.get("frames")
    return _sequence(frames, field=f"{field}.frames")


def _group_metadata(group: object, *, field: str) -> dict[str, Any]:
    """Return the exact metadata paired with a decoded group in model order."""

    metadata_fn = getattr(group, "metadata", None)
    if callable(metadata_fn):
        value = metadata_fn()
    elif isinstance(group, Mapping):
        value = group.get("metadata", {})
        if callable(value):
            value = value()
    else:
        value = {}
    return dict(_json_safe(_mapping(value, field=f"{field}.metadata"), field=f"{field}.metadata"))


def _group_context(
    group: object,
    *,
    camera_id: str,
    window_id: str,
    row_index: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Make the no-pixel context used to verify flattened row order."""

    context_fn = getattr(group, "to_dict", None)
    if callable(context_fn):
        value = context_fn()
    elif isinstance(group, Mapping):
        value = {key: child for key, child in group.items() if key not in {"frames", "metadata"}}
    else:
        value = {}
    context = dict(_json_safe(_mapping(value, field="decoded_group"), field="decoded_group"))
    context["row_index"] = row_index
    context.setdefault("camera_id", camera_id)
    context.setdefault("window_id", window_id)
    context.setdefault("frame_count", len(_group_frames(group, field="decoded_group")))
    # This is the exact side-channel passed alongside the corresponding frame
    # group.  It proves pairing without retaining frame pixels or identifiers.
    context["processor_video_metadata"] = _json_safe(
        metadata,
        field="decoded_group.processor_video_metadata",
    )
    return context


def _backend_value(backend: object, attribute: str) -> Any:
    """Read a small backend metadata attribute without making it mandatory."""

    value = getattr(backend, attribute, None)
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    return value


def _load_json(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        try:
            value = json.loads(Path(value).expanduser().read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WemmCohortRuntimeBenchmarkError(f"could not read cohort manifest: {exc}") from exc
    return dict(_mapping(value, field="cohort_manifest"))


def _sync(backend: Any) -> None:
    """Synchronize CUDA when the optional backend exposes its torch module."""

    torch = getattr(backend, "_torch", None)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        try:
            if cuda.is_available():
                cuda.synchronize()
        except Exception:
            # Timing remains useful on CPU and on runtimes without a working
            # synchronize hook; this is not a correctness gate.
            return


def _close_groups(groups: Mapping[str, Mapping[str, Any]]) -> None:
    for camera_groups in groups.values():
        if not isinstance(camera_groups, Mapping):
            continue
        for group in camera_groups.values():
            frames = getattr(group, "frames", None)
            if frames is None and isinstance(group, Mapping):
                frames = group.get("frames", ())
            for frame in frames or ():
                close = getattr(frame, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()


def _numeric_row(value: Sequence[float], *, field: str) -> tuple[float, ...]:
    if not value:
        raise WemmCohortRuntimeBenchmarkError(f"{field} must not be empty")
    row: list[float] = []
    for index, raw in enumerate(value):
        row.append(_finite(raw, field=f"{field}[{index}]"))
    return tuple(row)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise WemmCohortRuntimeBenchmarkError("embedding rows have different dimensions")
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return a normalised cosine even for a lightweight fake backend."""

    if len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(float(item) * float(item) for item in left))
    right_norm = math.sqrt(sum(float(item) * float(item) for item in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    value = _dot(left, right) / (left_norm * right_norm)
    # Round-off around unit vectors should not yield an impossible cosine.
    return max(-1.0, min(1.0, value))


def _rank_ids(
    query: Sequence[float],
    labels: Sequence[Any],
    label_vectors: Sequence[Sequence[float]],
) -> tuple[str, ...]:
    return tuple(item[1] for item in _rank_scores(query, labels, label_vectors))


def _rank_scores(
    query: Sequence[float],
    labels: Sequence[Any],
    label_vectors: Sequence[Sequence[float]],
) -> list[tuple[float, str]]:
    rows = [
        (_dot(query, vector), str(label.provisional_id))
        for label, vector in zip(labels, label_vectors, strict=True)
    ]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return rows


def _parity(
    serial_rows: Sequence[Sequence[float]],
    batch_rows: Sequence[Sequence[float]],
    *,
    labels: Sequence[Any],
    label_vectors: Sequence[Sequence[float]],
    ordered_items: Sequence[Mapping[str, Any]] = (),
    tolerance: float = DEFAULT_PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Compare numerical and rank parity while retaining row-order evidence.

    The backend returns normalised embeddings in normal operation, but the
    comparator also handles simple test doubles.  It intentionally stores
    only bounded per-row deltas and camera/window context -- never embeddings
    or frame content.
    """

    if len(serial_rows) != len(batch_rows):
        return {
            "row_count_equal": False,
            "serial_row_count": len(serial_rows),
            "batch_row_count": len(batch_rows),
            "dimension_equal": False,
            "mean_cosine": None,
            "min_cosine": None,
            "max_abs_delta": None,
            "mean_abs_delta": None,
            "top1_equal_fraction": 0.0,
            "full_order_equal_fraction": 0.0,
            "within_tolerance": False,
            "order_context_count": len(ordered_items),
            "row_order_preserved": False,
            "mismatch_count": 1,
            "mismatches_truncated": False,
            "mismatches": [],
        }
    if not ordered_items:
        ordered_items = tuple({"row_index": index} for index in range(len(serial_rows)))
    if len(ordered_items) != len(serial_rows):
        raise WemmCohortRuntimeBenchmarkError(
            "ordered_items must match the serial and batch embedding row count"
        )
    cosines: list[float] = []
    deltas: list[float] = []
    top1 = 0
    full = 0
    dimension_equal = True
    mismatches: list[dict[str, Any]] = []
    mismatch_count = 0
    for row_index, (serial_raw, batch_raw, context) in enumerate(
        zip(serial_rows, batch_rows, ordered_items, strict=True)
    ):
        serial = _numeric_row(serial_raw, field=f"serial_rows[{row_index}]")
        batched = _numeric_row(batch_raw, field=f"batch_rows[{row_index}]")
        if len(serial) != len(batched):
            dimension_equal = False
            mismatch_count += 1
            if len(mismatches) < MAX_PARITY_MISMATCHES:
                mismatches.append(
                    {
                        "row_index": row_index,
                        "context": _json_safe(context, field=f"ordered_items[{row_index}]"),
                        "reason": "DIMENSION_MISMATCH",
                        "serial_dimension": len(serial),
                        "batch_dimension": len(batched),
                    }
                )
            continue
        row_deltas = [abs(left - right) for left, right in zip(serial, batched, strict=True)]
        row_max_delta = max(row_deltas, default=0.0)
        deltas.extend(row_deltas)
        cosine = _cosine(serial, batched)
        if cosine is not None:
            cosines.append(cosine)
        serial_rank = _rank_ids(serial, labels, label_vectors)
        batch_rank = _rank_ids(batched, labels, label_vectors)
        top1_equal = serial_rank[:1] == batch_rank[:1]
        full_order_equal = serial_rank == batch_rank
        top1 += int(top1_equal)
        full += int(full_order_equal)
        if row_max_delta > tolerance or not top1_equal or not full_order_equal:
            mismatch_count += 1
            if len(mismatches) < MAX_PARITY_MISMATCHES:
                mismatches.append(
                    {
                        "row_index": row_index,
                        "context": _json_safe(context, field=f"ordered_items[{row_index}]"),
                        "max_abs_delta": row_max_delta,
                        "cosine": cosine,
                        "top1_equal": top1_equal,
                        "full_order_equal": full_order_equal,
                        "serial_top1": serial_rank[0] if serial_rank else None,
                        "batch_top1": batch_rank[0] if batch_rank else None,
                    }
                )
    count = len(serial_rows)
    max_delta = max(deltas, default=0.0) if dimension_equal else None
    mean_delta = sum(deltas) / len(deltas) if deltas else (0.0 if dimension_equal else None)
    within_tolerance = bool(dimension_equal and max_delta is not None and max_delta <= tolerance)
    row_order_preserved = all(
        context.get("row_index") == row_index for row_index, context in enumerate(ordered_items)
    )
    return {
        "row_count_equal": True,
        "serial_row_count": count,
        "batch_row_count": count,
        "dimension_equal": dimension_equal,
        "mean_cosine": sum(cosines) / len(cosines) if cosines else 1.0,
        "min_cosine": min(cosines) if cosines else 1.0,
        "max_abs_delta": max_delta,
        "mean_abs_delta": mean_delta,
        "top1_equal_fraction": top1 / count if count else 1.0,
        "full_order_equal_fraction": full / count if count else 1.0,
        "within_tolerance": within_tolerance,
        "row_order_preserved": row_order_preserved,
        "order_context_count": len(ordered_items),
        "mismatch_count": mismatch_count,
        "mismatches_truncated": mismatch_count > len(mismatches),
        "mismatches": mismatches,
    }


def _validate_cohort(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...], float]:
    if manifest.get("format") != COHORT_FORMAT:
        raise WemmCohortRuntimeBenchmarkError(f"cohort format must be {COHORT_FORMAT!r}")
    if manifest.get("authority") != AUTHORITY:
        raise WemmCohortRuntimeBenchmarkError(f"cohort authority must be {AUTHORITY!r}")
    source = _mapping(manifest.get("source"), field="cohort.source")
    camera_count = _positive_int(source.get("camera_count"), field="cohort.source.camera_count")
    if camera_count != len(CAMERA_IDS):
        raise WemmCohortRuntimeBenchmarkError("cohort must contain exactly six cameras")
    cameras = _sequence(source.get("cameras"), field="cohort.source.cameras")
    camera_ids = tuple(str(_mapping(row, field="camera").get("camera_id")) for row in cameras)
    if camera_ids != CAMERA_IDS:
        raise WemmCohortRuntimeBenchmarkError("cohort camera order must be cam_01 through cam_06")
    duration = _finite(
        source.get("common_duration_seconds"),
        field="cohort.source.common_duration_seconds",
    )
    if duration <= 0:
        raise WemmCohortRuntimeBenchmarkError("cohort duration must be positive")
    if duration > MAX_COHORT_DURATION_SECONDS + 1e-6:
        raise WemmCohortRuntimeBenchmarkError("cohort exceeds the bounded 40.8335-second limit")
    windows_raw = _sequence(manifest.get("windows"), field="cohort.windows")
    if len(windows_raw) > MAX_WINDOW_COUNT:
        raise WemmCohortRuntimeBenchmarkError(
            f"cohort contains more than {MAX_WINDOW_COUNT} windows"
        )
    windows: list[dict[str, Any]] = []
    seen_window_ids: set[str] = set()
    for index, raw in enumerate(windows_raw):
        row = dict(_mapping(raw, field=f"cohort.windows[{index}]"))
        window_id = str(row.get("window_id", "")).strip()
        if not window_id:
            raise WemmCohortRuntimeBenchmarkError(f"cohort.windows[{index}].window_id is required")
        if window_id in seen_window_ids:
            raise WemmCohortRuntimeBenchmarkError(f"duplicate window_id: {window_id}")
        seen_window_ids.add(window_id)
        start = _finite(row.get("start_seconds"), field=f"{window_id}.start_seconds")
        end = _finite(row.get("end_seconds"), field=f"{window_id}.end_seconds")
        if start < 0 or end <= start:
            raise WemmCohortRuntimeBenchmarkError(f"invalid interval for {window_id}")
        if end > duration + 1e-6:
            raise WemmCohortRuntimeBenchmarkError(
                f"{window_id} ends beyond the bounded source duration"
            )
        windows.append(row)
    if not windows:
        raise WemmCohortRuntimeBenchmarkError("cohort contains no windows")
    return windows, camera_ids, duration


def _arm_summary(
    *,
    arm_id: str,
    batch_size: int,
    frame_count: int,
    pixel_budget: int,
    decode_seconds: float,
    inference_seconds: float,
    input_count: int,
    represented_seconds: float,
    observations: Sequence[Mapping[str, Any]],
    serial_rows: Sequence[Sequence[float]],
    rows: Sequence[Sequence[float]],
    labels: Sequence[Any],
    label_vectors: Sequence[Sequence[float]],
    serial: bool,
    ordered_items: Sequence[Mapping[str, Any]] = (),
    parity_tolerance: float = DEFAULT_PARITY_TOLERANCE,
    memory_telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    video_observations = [row for row in observations if str(row.get("modality", "")) == "video"]
    video_item_count = sum(
        int(row.get("item_count", 0))
        for row in video_observations
        if isinstance(row.get("item_count", 0), int)
    )
    payload: dict[str, Any] = {
        "arm_id": arm_id,
        "batch_size": batch_size,
        "frame_count": frame_count,
        "video_max_pixels": pixel_budget,
        "input_count": input_count,
        "observation_count": len(observations),
        "model_call_count": len(video_observations),
        "video_item_count": video_item_count,
        "decode_seconds_shared": decode_seconds,
        "inference_seconds": inference_seconds,
        "estimated_e2e_seconds": decode_seconds + inference_seconds,
        "source_camera_normalized_realtime": (
            represented_seconds * len(CAMERA_IDS) / (decode_seconds + inference_seconds)
            if decode_seconds + inference_seconds > 0
            else None
        ),
        "observations": [dict(row) for row in observations],
        "rank_diagnostic": {
            "top1_count": 0,
            "top1_fraction": 0.0,
            "rank_counts": {},
        },
    }
    if memory_telemetry is not None:
        payload["memory_telemetry"] = dict(memory_telemetry)
    if serial:
        payload["control"] = True
    else:
        payload["parity_vs_serial"] = _parity(
            serial_rows,
            rows,
            labels=labels,
            label_vectors=label_vectors,
            ordered_items=ordered_items,
            tolerance=parity_tolerance,
        )
    ranks = [_rank_ids(row, labels, label_vectors) for row in rows]
    scored_ranks = [_rank_scores(row, labels, label_vectors) for row in rows]
    top_label_counts: dict[str, int] = {}
    margins: list[float] = []
    row_diagnostics: list[dict[str, Any]] = []
    for row_index, (rank, scored) in enumerate(zip(ranks, scored_ranks, strict=True)):
        if rank:
            top_label_counts[rank[0]] = top_label_counts.get(rank[0], 0) + 1
        if len(scored) >= 2:
            margins.append(scored[0][0] - scored[1][0])
        context = ordered_items[row_index] if row_index < len(ordered_items) else {}
        row_diagnostics.append(
            {
                "row_index": row_index,
                "window_id": context.get("window_id"),
                "camera_id": context.get("camera_id"),
                "top1_not_gold": rank[0] if rank else None,
                "top1_top2_margin": (scored[0][0] - scored[1][0] if len(scored) >= 2 else None),
            }
        )
    by_window: dict[str, list[dict[str, Any]]] = {}
    for row in row_diagnostics:
        window_id = row.get("window_id")
        if isinstance(window_id, str) and window_id:
            by_window.setdefault(window_id, []).append(row)
    window_consistency: list[dict[str, Any]] = []
    for window_id, window_rows in by_window.items():
        labels_for_window = [
            str(row["top1_not_gold"]) for row in window_rows if row.get("top1_not_gold") is not None
        ]
        counts: dict[str, int] = {}
        for label_id in labels_for_window:
            counts[label_id] = counts.get(label_id, 0) + 1
        modal_count = max(counts.values(), default=0)
        window_consistency.append(
            {
                "window_id": window_id,
                "camera_count_observed": len(window_rows),
                "modal_top1_count": modal_count,
                "modal_top1_fraction": modal_count / len(window_rows) if window_rows else 0.0,
                "top1_labels_not_gold": counts,
            }
        )
    # There is no official gold in this cohort.  Keep this field explicitly
    # diagnostic rather than pretending the provisional phrase top-1 is gold.
    payload["rank_diagnostic"] = {
        "phrase_catalog_top1_count": len(ranks),
        "phrase_catalog_top1_fraction": 1.0 if ranks else 0.0,
        "top_label_counts_not_gold": top_label_counts,
        "top1_top2_margin_not_calibrated": {
            "count": len(margins),
            "mean": sum(margins) / len(margins) if margins else None,
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "camera_consistency_not_gold": {
            "window_count": len(window_consistency),
            "mean_modal_top1_fraction": (
                sum(row["modal_top1_fraction"] for row in window_consistency)
                / len(window_consistency)
                if window_consistency
                else None
            ),
            "all_camera_same_fraction": (
                sum(row["modal_top1_fraction"] == 1.0 for row in window_consistency)
                / len(window_consistency)
                if window_consistency
                else None
            ),
            "per_window": window_consistency,
        },
    }
    return payload


def run_wemm_cohort_runtime_benchmark(
    manifest: Mapping[str, Any] | str | Path,
    *,
    phrase_catalog: Mapping[str, Any] | Sequence[Any] | str | Path,
    model_directory: str | Path,
    frame_count: int = DEFAULT_FRAME_COUNT,
    pixel_budget: int = DEFAULT_PIXEL_BUDGET,
    dimension: int = DEFAULT_DIMENSION,
    device: str = "cuda",
    window_chunk_size: int = 1,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    max_windows: int | None = None,
    include_pipeline: bool = False,
    queue_capacity: int = 1,
    decode_cache: ProductionWemmDecodeCache | None = None,
    decode_scope_key: Hashable | None = None,
) -> dict[str, Any]:
    """Run serial/Batch2/Batch4 on one bounded six-camera cohort.

    The model is loaded once.  For each decoded chunk, serial and all requested
    batch arms consume the same PIL frame objects before they are released.
    ``include_pipeline`` adds a separate one-producer/one-consumer diagnostic
    pass using the largest requested batch size; it decodes the source a second
    time and is therefore reported separately.  When ``decode_cache`` is
    supplied, only the main serial/batch pass uses the cache; the pipeline arm
    remains a fresh decode so producer timing is not replaced by replay.
    """

    manifest_doc = _load_json(manifest)
    windows, camera_order, duration = _validate_cohort(manifest_doc)
    frame_count = _positive_int(frame_count, field="frame_count", maximum=MAX_FRAME_COUNT)
    if frame_count < MIN_FRAME_COUNT:
        raise WemmCohortRuntimeBenchmarkError(
            f"frame_count must be between {MIN_FRAME_COUNT} and {MAX_FRAME_COUNT}"
        )
    pixel_budget = _positive_int(pixel_budget, field="pixel_budget")
    dimension = _positive_int(dimension, field="dimension")
    window_chunk_size = _positive_int(window_chunk_size, field="window_chunk_size")
    queue_capacity = _positive_int(queue_capacity, field="queue_capacity")
    if max_windows is not None:
        max_windows = _positive_int(max_windows, field="max_windows")
        windows = windows[:max_windows]
    if not windows:
        raise WemmCohortRuntimeBenchmarkError("no windows selected")
    requested: list[int] = []
    for index, value in enumerate(batch_sizes):
        requested_size = _positive_int(
            value,
            field=f"batch_sizes[{index}]",
            maximum=MAX_BATCH_SIZE,
        )
        if requested_size not in requested:
            requested.append(requested_size)
    if not requested:
        raise WemmCohortRuntimeBenchmarkError("batch_sizes must not be empty")
    if (decode_cache is None) != (decode_scope_key is None):
        raise WemmCohortRuntimeBenchmarkError(
            "decode_cache and decode_scope_key must be supplied together"
        )
    labels, catalog_meta = load_open_phrase_catalog(phrase_catalog)

    backend = WemmEmbeddingBackend(
        model_directory=model_directory,
        device=device,
        dimension=dimension,
        video_max_pixels=pixel_budget,
    )
    text_start = time.perf_counter()
    label_rows = backend.encode_texts(
        [label.text_for("canonical") for label in labels],
        batch_size=32,
    )
    text_seconds = time.perf_counter() - text_start
    arms: list[dict[str, Any]] = []
    chunks_seen = 0
    try:
        label_by_index = tuple(label_rows)
        serial_all: list[tuple[float, ...]] = []
        batch_all: dict[int, list[tuple[float, ...]]] = {size: [] for size in requested}
        arm_observations: dict[str, list[dict[str, Any]]] = {"serial": []}
        arm_observations.update({f"batch{size}": [] for size in requested})
        arm_memory: dict[str, dict[str, Any]] = {}
        ordered_items: list[dict[str, Any]] = []
        arm_seconds: dict[str, float] = {
            "serial": 0.0,
            **{f"batch{size}": 0.0 for size in requested},
        }

        def _decode_factory() -> Any:
            return iter_decode_production_window_chunks(
                {**manifest_doc, "windows": windows},
                frame_count=frame_count,
                window_chunk_size=window_chunk_size,
            )

        decode_iter = (
            decode_cache.iter_chunks(decode_scope_key, _decode_factory)
            if decode_cache is not None
            else _decode_factory()
        )
        decode_seconds = 0.0
        chunk_index = 0
        while True:
            decode_started = time.perf_counter()
            try:
                groups = next(decode_iter)
            except StopIteration:
                break
            decode_seconds += time.perf_counter() - decode_started
            chunks_seen += 1
            chunk_windows = windows[
                chunk_index * window_chunk_size : (chunk_index + 1) * window_chunk_size
            ]
            flat_groups: list[Sequence[Any]] = []
            flat_metadata: list[Mapping[str, Any]] = []
            for window in chunk_windows:
                window_id = str(window["window_id"])
                for camera_id in camera_order:
                    group = groups[camera_id][window_id]
                    metadata = _group_metadata(
                        group,
                        field=f"{camera_id}.{window_id}",
                    )
                    ordered_items.append(
                        _group_context(
                            group,
                            camera_id=camera_id,
                            window_id=window_id,
                            row_index=len(ordered_items),
                            metadata=metadata,
                        )
                    )
                    flat_groups.append(_group_frames(group, field=f"{camera_id}.{window_id}"))
                    flat_metadata.append(metadata)

            # Warm kernels once on the first chunk without including warm-up
            # in any arm.  The same first frame group is safe to reuse.
            if chunks_seen == 1 and flat_groups:
                _sync(backend)
                backend.encode_video_frames([flat_groups[0]], metadata_groups=[flat_metadata[0]])
                for size in requested:
                    backend.encode_video_frames_batch(
                        [flat_groups[0]], metadata_groups=[flat_metadata[0]], batch_size=size
                    )
                backend.observations.clear()

            observation_start = len(backend.observations)
            _reset_cuda_peak(backend)
            serial_memory_before = _arm_memory_before(backend)
            _sync(backend)
            started = time.perf_counter()
            serial_rows = backend.encode_video_frames(flat_groups, metadata_groups=flat_metadata)
            _sync(backend)
            arm_seconds["serial"] += time.perf_counter() - started
            arm_memory["serial"] = _merge_arm_memory(
                arm_memory.get("serial"),
                _arm_memory_telemetry(backend, serial_memory_before),
            )
            serial_all.extend(serial_rows)
            arm_observations["serial"].extend(
                _observation_dict(item, field="serial_observation")
                for item in backend.observations[observation_start:]
            )

            for size in requested:
                observation_start = len(backend.observations)
                _reset_cuda_peak(backend)
                batch_memory_before = _arm_memory_before(backend)
                _sync(backend)
                started = time.perf_counter()
                rows = backend.encode_video_frames_batch(
                    flat_groups,
                    metadata_groups=flat_metadata,
                    batch_size=size,
                )
                _sync(backend)
                arm_seconds[f"batch{size}"] += time.perf_counter() - started
                arm_memory[f"batch{size}"] = _merge_arm_memory(
                    arm_memory.get(f"batch{size}"),
                    _arm_memory_telemetry(backend, batch_memory_before),
                )
                batch_all[size].extend(rows)
                arm_observations[f"batch{size}"].extend(
                    _observation_dict(item, field=f"batch{size}_observation")
                    for item in backend.observations[observation_start:]
                )
            _close_groups(groups)
            del groups
            gc.collect()
            chunk_index += 1

        close_decoder = getattr(decode_iter, "close", None)
        if callable(close_decoder):
            close_decoder()

        expected_count = len(windows) * len(camera_order)
        represented_seconds = sum(
            float(window["end_seconds"]) - float(window["start_seconds"]) for window in windows
        )
        if len(serial_all) != expected_count:
            raise WemmCohortRuntimeBenchmarkError(
                f"serial output count {len(serial_all)} != expected {expected_count}"
            )
        for size, batch_rows in batch_all.items():
            if len(batch_rows) != expected_count:
                raise WemmCohortRuntimeBenchmarkError(
                    f"batch{size} output count {len(batch_rows)} != expected {expected_count}"
                )
        # Decode is shared across arms and measured around the source iterator;
        # keeping the metric explicit prevents fake precision.
        for serial in (True, False):
            if serial:
                arms.append(
                    _arm_summary(
                        arm_id="serial",
                        batch_size=1,
                        frame_count=frame_count,
                        pixel_budget=pixel_budget,
                        decode_seconds=decode_seconds,
                        inference_seconds=arm_seconds["serial"],
                        input_count=expected_count,
                        represented_seconds=represented_seconds,
                        observations=arm_observations["serial"],
                        serial_rows=serial_all,
                        rows=serial_all,
                        labels=labels,
                        label_vectors=label_by_index,
                        serial=True,
                        ordered_items=ordered_items,
                        memory_telemetry=arm_memory.get("serial"),
                    )
                )
                continue
            for size in requested:
                arms.append(
                    _arm_summary(
                        arm_id=f"batch{size}",
                        batch_size=size,
                        frame_count=frame_count,
                        pixel_budget=pixel_budget,
                        decode_seconds=decode_seconds,
                        inference_seconds=arm_seconds[f"batch{size}"],
                        input_count=expected_count,
                        represented_seconds=represented_seconds,
                        observations=arm_observations[f"batch{size}"],
                        serial_rows=serial_all,
                        rows=batch_all[size],
                        labels=labels,
                        label_vectors=label_by_index,
                        serial=False,
                        ordered_items=ordered_items,
                        memory_telemetry=arm_memory.get(f"batch{size}"),
                    )
                )
        report = {
            "format": FORMAT,
            "authority": AUTHORITY,
            "status": "MEASURED_NONPRODUCTION",
            "production_eligible": False,
            "official_quality_status": "NOT_MEASURED",
            "official_gold_status": "NOT_ESTABLISHED",
            "source": {
                "manifest_format": manifest_doc.get("format"),
                "path": _mapping(manifest_doc.get("source"), field="cohort.source").get("path"),
                "duration_seconds": duration,
                "represented_window_seconds": sum(
                    float(window["end_seconds"]) - float(window["start_seconds"])
                    for window in windows
                ),
                "window_count": len(windows),
                "camera_count": len(camera_order),
                "camera_ids": list(camera_order),
                "camera_window_input_count": expected_count,
                "window_chunk_size": window_chunk_size,
                "chunks_seen": chunks_seen,
                "input_order": [
                    {
                        "row_index": int(context["row_index"]),
                        "window_id": context.get("window_id"),
                        "camera_id": context.get("camera_id"),
                        "frame_count": context.get("frame_count"),
                        "selected_timestamps_ns": context.get("selected_timestamps_ns"),
                    }
                    for context in ordered_items
                ],
            },
            "model": {
                "name": "WeMM-Embedding-2B",
                "directory": str(Path(model_directory).expanduser().resolve()),
                "dimension": dimension,
                "backend_identity": _backend_value(backend, "identity"),
                "backend_variant": _backend_value(backend, "variant"),
                "supported_dimensions": _backend_value(backend, "supported_dimensions"),
                "effective_video_min_pixels": _backend_value(backend, "video_min_pixels"),
                "effective_video_max_pixels": _backend_value(backend, "video_max_pixels"),
                "frame_count": frame_count,
                "video_max_pixels": pixel_budget,
                "batch_sizes": list(requested),
                "parity_tolerance": DEFAULT_PARITY_TOLERANCE,
                "text_encode_seconds": text_seconds,
                "catalog_phrase_count": len(labels),
                "catalog_status": catalog_meta.get("status"),
            },
            "arms": arms,
            "controls": {
                "media_decoded": True,
                "model_invoked": True,
                "gold_read": False,
                "gold_written": False,
                "epic_ontology_used": False,
                "mapper_used": False,
                "hash_or_sha_used": False,
            },
            "limitations": [
                "The cohort has no established gold; rank diagnostics are not accuracy.",
                (
                    "Decode wall time is measured around the iterator; model arm "
                    "timings exclude shared decode."
                ),
                (
                    "Batch numerical drift is expected from batched GPU kernels; "
                    "inspect parity before promotion."
                ),
            ],
        }
        if decode_cache is not None:
            report["decode_cache"] = {
                "scope": "process_local",
                "scope_key_supplied": True,
                **decode_cache.stats().to_dict(),
                "pipeline_uses_separate_decode": bool(include_pipeline),
            }
        if include_pipeline:
            largest_batch = max(requested)
            pipeline_iter = iter_decode_production_window_chunks(
                {**manifest_doc, "windows": windows},
                frame_count=frame_count,
                window_chunk_size=window_chunk_size,
            )
            chunk_count = (len(windows) + window_chunk_size - 1) // window_chunk_size

            def prepare_chunk(_ordinal: int, recorder: Any) -> Mapping[str, Mapping[str, Any]]:
                with recorder.phase(PipelinePhase.MEDIA_DECODE):
                    try:
                        return next(pipeline_iter)
                    except StopIteration as exc:
                        raise WemmCohortRuntimeBenchmarkError(
                            "decoder ended before the planned pipeline chunks"
                        ) from exc

            def consume_chunk(
                groups: Mapping[str, Mapping[str, Any]], recorder: Any
            ) -> dict[str, int]:
                # Identify the chunk's windows from the first camera map.  The
                # iterator preserves manifest order, so this remains stable
                # without carrying extra identity/provenance state.
                first_camera = next(iter(camera_order))
                window_ids = tuple(str(value) for value in groups[first_camera])
                flat: list[Sequence[Any]] = []
                metadata: list[Mapping[str, Any]] = []
                for window_id in window_ids:
                    for camera_id in camera_order:
                        group = groups[camera_id][window_id]
                        flat.append(_group_frames(group, field=f"{camera_id}.{window_id}"))
                        metadata.append(_group_metadata(group, field=f"{camera_id}.{window_id}"))
                _sync(backend)
                with recorder.phase(PipelinePhase.MODEL):
                    rows = backend.encode_video_frames_batch(
                        flat,
                        metadata_groups=metadata,
                        batch_size=largest_batch,
                    )
                _sync(backend)
                _close_groups(groups)
                return {"input_count": len(rows)}

            pipeline_run = run_bounded_pipeline(
                range(chunk_count),
                prepare=prepare_chunk,
                consume=consume_chunk,
                key=lambda _item, ordinal: f"cohort-chunk-{ordinal:03d}",
                queue_capacity=queue_capacity,
            )
            report["pipeline"] = {
                "batch_size": largest_batch,
                "window_chunk_size": window_chunk_size,
                "queue_capacity": queue_capacity,
                "chunk_count": chunk_count,
                "timing": pipeline_run.report.to_dict(),
                "limitations": [
                    "Pipeline pass is a separate decode of the same bounded cohort.",
                    "MODEL phase includes processor, tensor transfer, and WeMM forward.",
                ],
            }
            close_pipeline = getattr(pipeline_iter, "close", None)
            if callable(close_pipeline):
                close_pipeline()
        return report
    finally:
        backend.close()


__all__ = [
    "AUTHORITY",
    "CAMERA_IDS",
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_DIMENSION",
    "DEFAULT_FRAME_COUNT",
    "DEFAULT_PIXEL_BUDGET",
    "FORMAT",
    "WemmCohortRuntimeBenchmarkError",
    "run_wemm_cohort_runtime_benchmark",
]
