"""Bounded WeMM frame/grid quality probe with explicit decode reuse.

This module executes the small matrix planned by :mod:`wemm_frame_grid_matrix`
against the same ten-window, six-camera production-shaped cohort.  It keeps
Batch4 fixed, varies only frame count and total pixel budget, and reuses a
process-local decoded-frame cache between the two arms that share a frame
count.  The returned artifact is intentionally a light diagnostic projection:
it contains timing, observed processor grids, rank/margin diagnostics, and
Batch4-vs-serial parity, but never embeddings or frame pixels.

The cohort has no independent gold.  Consequently all quality fields remain
``NOT_MEASURED`` and rank values are explicitly provisional diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .production_wemm_decode_cache import (
    ProductionWemmDecodeCache,
    ProductionWemmDecodeCacheError,
)
from .wemm_cohort_runtime_benchmark import (
    DEFAULT_DIMENSION,
    DEFAULT_PIXEL_BUDGET,
    WemmCohortRuntimeBenchmarkError,
    run_wemm_cohort_runtime_benchmark,
)
from .wemm_frame_grid_matrix import (
    BASELINE_GRID,
    CURRENT_8_FRAME_GRID,
    HIGHER_4_FRAME_GRID,
    HIGHER_8_FRAME_GRID,
    HIGHER_TOTAL_PIXEL_BUDGET,
)

FORMAT: Final = "robata-wemm-frame-grid-quality-matrix-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
BATCH_SIZE: Final = 4
MATRIX_ARMS: Final[tuple[dict[str, int | str], ...]] = (
    {"arm_id": "f4_current_budget", "frame_count": 4, "pixel_budget": DEFAULT_PIXEL_BUDGET},
    {"arm_id": "f8_current_budget", "frame_count": 8, "pixel_budget": DEFAULT_PIXEL_BUDGET},
    {
        "arm_id": "f4_higher_budget",
        "frame_count": 4,
        "pixel_budget": HIGHER_TOTAL_PIXEL_BUDGET,
    },
    {
        "arm_id": "f8_higher_budget",
        "frame_count": 8,
        "pixel_budget": HIGHER_TOTAL_PIXEL_BUDGET,
    },
)
MATRIX_ARM_IDS: Final[tuple[str, ...]] = tuple(str(row["arm_id"]) for row in MATRIX_ARMS)
EXPECTED_GRIDS: Final[dict[str, list[int]]] = {
    "f4_current_budget": list(BASELINE_GRID),
    "f8_current_budget": list(CURRENT_8_FRAME_GRID),
    "f4_higher_budget": list(HIGHER_4_FRAME_GRID),
    "f8_higher_budget": list(HIGHER_8_FRAME_GRID),
}


class WemmFrameGridQualityMatrixError(ValueError):
    """Raised when the bounded quality matrix cannot be executed."""


def _grid_rows(observations: object) -> list[list[int]]:
    """Project unique processor grids without retaining other observations."""

    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes, bytearray)):
        return []
    seen: set[tuple[int, ...]] = set()
    grids: list[list[int]] = []
    for raw in observations:
        if not isinstance(raw, Mapping) or raw.get("modality") != "video":
            continue
        value = raw.get("video_grid_thw")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            continue
        rows: Sequence[Any]
        if (
            value
            and isinstance(value[0], Sequence)
            and not isinstance(value[0], (str, bytes, bytearray))
        ):
            rows = value
        else:
            rows = (value,)
        for row in rows:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                continue
            try:
                grid = tuple(int(item) for item in row)
            except (TypeError, ValueError, OverflowError):
                continue
            if len(grid) != 3 or any(item <= 0 for item in grid) or grid in seen:
                continue
            seen.add(grid)
            grids.append(list(grid))
    return grids


def _parity_projection(value: object) -> dict[str, Any] | None:
    """Keep parity headline metrics while omitting per-row verbose context."""

    if not isinstance(value, Mapping):
        return None
    names = (
        "row_count_equal",
        "dimension_equal",
        "mean_cosine",
        "min_cosine",
        "max_abs_delta",
        "mean_abs_delta",
        "top1_equal_fraction",
        "full_order_equal_fraction",
        "within_tolerance",
        "row_order_preserved",
        "order_context_count",
        "mismatch_count",
        "mismatches_truncated",
    )
    return {name: value.get(name) for name in names if name in value}


def _arm_projection(
    report: Mapping[str, Any],
    *,
    arm_id: str,
    frame_count: int,
    pixel_budget: int,
    cache_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_arms = report.get("arms")
    if not isinstance(raw_arms, Sequence) or isinstance(raw_arms, (str, bytes, bytearray)):
        raise WemmFrameGridQualityMatrixError(f"{arm_id} report has no arms")
    serial = next(
        (row for row in raw_arms if isinstance(row, Mapping) and row.get("control")),
        None,
    )
    batch = next(
        (
            row
            for row in raw_arms
            if isinstance(row, Mapping)
            and int(row.get("batch_size", 0) or 0) == BATCH_SIZE
            and not row.get("control")
        ),
        None,
    )
    if batch is None:
        raise WemmFrameGridQualityMatrixError(f"{arm_id} report has no Batch4 arm")

    batch_observations = batch.get("observations")
    projection: dict[str, Any] = {
        "arm_id": arm_id,
        "frame_count": frame_count,
        "pixel_budget": pixel_budget,
        "total_pixel_budget": pixel_budget,
        "expected_video_grid_thw": EXPECTED_GRIDS[arm_id],
        "status": report.get("status"),
        "production_eligible": report.get("production_eligible"),
        "official_quality_status": report.get("official_quality_status"),
        "official_gold_status": report.get("official_gold_status"),
        "input_count": batch.get("input_count"),
        "model_call_count": batch.get("model_call_count"),
        "video_item_count": batch.get("video_item_count"),
        "decode_seconds_shared": batch.get("decode_seconds_shared"),
        "inference_seconds": batch.get("inference_seconds"),
        "estimated_e2e_seconds": batch.get("estimated_e2e_seconds"),
        "source_camera_normalized_realtime": batch.get("source_camera_normalized_realtime"),
        "observed_video_grid_thw": _grid_rows(batch_observations),
        "rank_diagnostic": batch.get("rank_diagnostic"),
        "parity_vs_serial": _parity_projection(batch.get("parity_vs_serial")),
        "decode_cache": dict(cache_event) if cache_event is not None else None,
        "serial_control": {
            "inference_seconds": serial.get("inference_seconds") if serial else None,
            "estimated_e2e_seconds": serial.get("estimated_e2e_seconds") if serial else None,
        },
    }
    pipeline = report.get("pipeline")
    if isinstance(pipeline, Mapping):
        timing = pipeline.get("timing")
        if isinstance(timing, Mapping):
            projection["pipeline"] = {
                "batch_size": pipeline.get("batch_size"),
                "queue_capacity": pipeline.get("queue_capacity"),
                "wall_seconds": timing.get("wall_seconds"),
                "estimated_speedup": timing.get("estimated_speedup"),
                "dominant_phase": timing.get("dominant_phase"),
                "overlap_seconds": timing.get("overlap_seconds"),
            }
    return projection


def _scope_key(
    manifest_path: Path,
    *,
    frame_count: int,
    window_chunk_size: int,
    max_windows: int | None,
) -> tuple[str, int, int, int | None]:
    """Return the explicit cache key shared by equal-frame-count arms."""

    return (
        str(manifest_path.resolve()),
        int(frame_count),
        int(window_chunk_size),
        int(max_windows) if max_windows is not None else None,
    )


def _cache_event(before: Any, after: Any) -> dict[str, Any]:
    """Summarize one matrix arm's cache transition without exposing frames."""

    def _counter(stats: Any, name: str) -> int:
        value: object = getattr(stats, name, None)
        if value is None:
            to_dict = getattr(stats, "to_dict", None)
            if callable(to_dict):
                try:
                    value = to_dict().get(name, 0)
                except Exception:
                    value = 0
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

    before_hits = _counter(before, "hit_count")
    after_hits = _counter(after, "hit_count")
    before_misses = _counter(before, "miss_count")
    after_misses = _counter(after, "miss_count")
    return {
        "cache_hit": after_hits > before_hits,
        "hit_delta": max(0, after_hits - before_hits),
        "miss_delta": max(0, after_misses - before_misses),
    }


def run_wemm_frame_grid_quality_matrix(
    manifest: Mapping[str, Any] | str | Path,
    *,
    phrase_catalog: Mapping[str, Any] | Sequence[Any] | str | Path,
    model_directory: str | Path,
    device: str = "cuda",
    window_chunk_size: int = 1,
    max_windows: int | None = None,
    dimension: int = DEFAULT_DIMENSION,
    pipeline_arm: str | None = None,
    queue_capacity: int = 1,
    decode_cache: ProductionWemmDecodeCache | None = None,
) -> dict[str, Any]:
    """Run all four fixed matrix arms and return a light quality probe.

    The cache is keyed by manifest path, frame count, chunk size, and optional
    window cap.  Thus current/higher pixel-budget arms reuse decode, while 4f
    and 8f correctly use separate scopes.  ``decode_cache`` is injectable for
    tests or a caller that wants to combine this matrix with another bounded
    diagnostic; when omitted it is created and cleared by this function.
    """

    if pipeline_arm is not None and pipeline_arm not in MATRIX_ARM_IDS:
        raise WemmFrameGridQualityMatrixError(
            f"pipeline_arm must be one of {', '.join(MATRIX_ARM_IDS)}"
        )
    if (
        isinstance(queue_capacity, bool)
        or not isinstance(queue_capacity, int)
        or queue_capacity <= 0
    ):
        raise WemmFrameGridQualityMatrixError("queue_capacity must be a positive integer")
    if (
        isinstance(window_chunk_size, bool)
        or not isinstance(window_chunk_size, int)
        or window_chunk_size <= 0
    ):
        raise WemmFrameGridQualityMatrixError("window_chunk_size must be a positive integer")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise WemmFrameGridQualityMatrixError("dimension must be a positive integer")

    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
    elif isinstance(manifest, Mapping):
        source = manifest.get("source")
        raw_source_path = source.get("path") if isinstance(source, Mapping) else None
        manifest_path = (
            Path(raw_source_path).expanduser().resolve()
            if isinstance(raw_source_path, (str, Path)) and str(raw_source_path).strip()
            else Path("<in-memory-cohort>")
        )
    else:
        manifest_path = Path("<in-memory-cohort>")
    cache = decode_cache if decode_cache is not None else ProductionWemmDecodeCache(max_scopes=2)
    owns_cache = decode_cache is None
    reports: list[dict[str, Any]] = []
    try:
        for arm in MATRIX_ARMS:
            arm_id = str(arm["arm_id"])
            frame_count = int(arm["frame_count"])
            pixel_budget = int(arm["pixel_budget"])
            cache_before = cache.stats()
            try:
                runtime_report = run_wemm_cohort_runtime_benchmark(
                    manifest,
                    phrase_catalog=phrase_catalog,
                    model_directory=model_directory,
                    frame_count=frame_count,
                    pixel_budget=pixel_budget,
                    dimension=dimension,
                    device=device,
                    window_chunk_size=window_chunk_size,
                    batch_sizes=(BATCH_SIZE,),
                    max_windows=max_windows,
                    include_pipeline=pipeline_arm == arm_id,
                    queue_capacity=queue_capacity,
                    decode_cache=cache,
                    decode_scope_key=_scope_key(
                        manifest_path,
                        frame_count=frame_count,
                        window_chunk_size=window_chunk_size,
                        max_windows=max_windows,
                    ),
                )
            except (WemmCohortRuntimeBenchmarkError, ProductionWemmDecodeCacheError) as exc:
                raise WemmFrameGridQualityMatrixError(f"{arm_id} runtime failed: {exc}") from exc
            cache_after = cache.stats()
            reports.append(
                _arm_projection(
                    runtime_report,
                    arm_id=arm_id,
                    frame_count=frame_count,
                    pixel_budget=pixel_budget,
                    cache_event=_cache_event(cache_before, cache_after),
                )
            )
        cache_stats = cache.stats().to_dict()
        return {
            "format": FORMAT,
            "authority": AUTHORITY,
            "status": "SUCCEEDED",
            "production_eligible": False,
            "official_quality_status": "NOT_MEASURED",
            "official_gold_status": "NOT_ESTABLISHED",
            "quality_claim": False,
            "model": {
                "name": "WeMM-Embedding-2B",
                "directory": str(Path(model_directory).expanduser().resolve()),
                "dimension": dimension,
                "batch_size": BATCH_SIZE,
            },
            "source": {
                "manifest": str(manifest_path),
                "phrase_catalog": str(Path(phrase_catalog).expanduser().resolve())
                if isinstance(phrase_catalog, (str, Path))
                else "<in-memory-catalog>",
                "window_chunk_size": window_chunk_size,
                "max_windows": max_windows,
                "cohort_contract": "40.8335s/6 cameras/10 windows when full manifest is supplied",
            },
            "matrix": {
                "batch_size": BATCH_SIZE,
                "arm_ids": list(MATRIX_ARM_IDS),
                "execution_order": list(MATRIX_ARM_IDS),
                "arms": reports,
            },
            "decode_cache": {
                "scope": "process_local",
                "reuse_policy": "same_manifest_and_frame_count",
                **cache_stats,
            },
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
                (
                    "No independent source-bound gold is established; rank diagnostics "
                    "are not accuracy."
                ),
                "The decode cache is process-local and diagnostic, not a durable production cache.",
                (
                    "Only equal-frame-count pixel-budget pairs reuse decoded frames; "
                    "4f and 8f are separate scopes."
                ),
            ],
        }
    finally:
        if owns_cache:
            cache.clear()


__all__ = [
    "AUTHORITY",
    "BATCH_SIZE",
    "DEFAULT_DIMENSION",
    "EXPECTED_GRIDS",
    "FORMAT",
    "MATRIX_ARMS",
    "MATRIX_ARM_IDS",
    "WemmFrameGridQualityMatrixError",
    "run_wemm_frame_grid_quality_matrix",
]
