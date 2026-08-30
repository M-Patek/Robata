#!/usr/bin/env python3
"""Create a compact, post-hoc WeMM cohort runtime report.

Only existing JSON diagnostics are read.  No media, weights, labels, or
identity material are touched.  The projection deliberately keeps only
throughput, encoder phase breakdowns, ingest-versus-decode-reuse timing,
batch parity, frame/grid observations, camera consistency, and provisional
top-label counts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT = "robata-wemm-runtime-compact-v1"
AUTHORITY = "LOCAL_NONPRODUCTION_ONLY"
PHASE_NAMES = ("processor", "model", "postprocess", "total")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be an array")
    return value


def _load(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc
    return dict(_mapping(payload, field=str(path)))


def _number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value


def _label_counts(value: object, *, field: str) -> dict[str, int]:
    if value is None:
        return {}
    source = _mapping(value, field=field)
    result: dict[str, int] = {}
    for key, raw_count in source.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        count = _integer(raw_count, field=f"{field}.{key}")
        if count is None or count < 0:
            raise ValueError(f"{field}.{key} must be non-negative")
        result[key] = count
    return dict(sorted(result.items()))


def _parity(value: object, *, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    source = _mapping(value, field=field)
    result: dict[str, Any] = {}
    numeric_keys = (
        "mean_cosine",
        "min_cosine",
        "max_abs_delta",
        "mean_abs_delta",
        "top1_equal_fraction",
        "full_order_equal_fraction",
    )
    for key in numeric_keys:
        if key in source:
            result[key] = _number(source[key], field=f"{field}.{key}")
    for key in ("within_tolerance", "row_order_preserved"):
        if key in source:
            if not isinstance(source[key], bool):
                raise ValueError(f"{field}.{key} must be boolean")
            result[key] = source[key]
    return result


def _runtime_arm(value: object, *, index: int) -> dict[str, Any]:
    row = _mapping(value, field=f"runtime.arms[{index}]")
    arm_id = str(row.get("arm_id", f"arm-{index}"))
    diagnostic = _mapping(row.get("rank_diagnostic", {}), field=f"{arm_id}.rank_diagnostic")
    result: dict[str, Any] = {
        "arm_id": arm_id,
        "batch_size": _integer(row.get("batch_size"), field=f"{arm_id}.batch_size"),
        "frame_count": _integer(row.get("frame_count"), field=f"{arm_id}.frame_count"),
        "input_count": _integer(row.get("input_count"), field=f"{arm_id}.input_count"),
        "shared_decode_seconds": _number(
            row.get("decode_seconds_shared"), field=f"{arm_id}.decode_seconds_shared"
        ),
        "inference_seconds": _number(
            row.get("inference_seconds"), field=f"{arm_id}.inference_seconds"
        ),
        "estimated_e2e_seconds": _number(
            row.get("estimated_e2e_seconds"), field=f"{arm_id}.estimated_e2e_seconds"
        ),
        "camera_normalized_realtime": _number(
            row.get("source_camera_normalized_realtime"),
            field=f"{arm_id}.source_camera_normalized_realtime",
        ),
        "provisional_top1_distribution": _label_counts(
            diagnostic.get("top_label_counts_not_gold"),
            field=f"{arm_id}.top_label_counts_not_gold",
        ),
    }
    parity = _parity(row.get("parity_vs_serial"), field=f"{arm_id}.parity_vs_serial")
    if parity is not None:
        result["parity_vs_serial"] = parity
    camera = diagnostic.get("camera_consistency_not_gold")
    if camera is not None:
        camera_row = _mapping(camera, field=f"{arm_id}.camera_consistency_not_gold")
        result["camera_consistency"] = {
            "window_count": _integer(
                camera_row.get("window_count"), field=f"{arm_id}.camera.window_count"
            ),
            "mean_modal_top1_fraction": _number(
                camera_row.get("mean_modal_top1_fraction"),
                field=f"{arm_id}.camera.mean_modal_top1_fraction",
            ),
            "all_camera_same_fraction": _number(
                camera_row.get("all_camera_same_fraction"),
                field=f"{arm_id}.camera.all_camera_same_fraction",
            ),
        }
    return result


def _phase_breakdown(value: object, *, index: int) -> dict[str, Any]:
    """Aggregate recorded encoder phases for one runtime arm.

    Batch observations carry explicit processor/model/postprocess timings,
    while the historical singleton path only carries an arm-level inference
    total.  Preserve that distinction instead of fabricating phase values for
    serial runs; the arm total is retained as a clearly marked fallback.
    """

    row = _mapping(value, field=f"runtime.arms[{index}]")
    arm_id = str(row.get("arm_id", f"arm-{index}"))
    raw_observations = _sequence(row.get("observations", []), field=f"{arm_id}.observations")
    sums = {name: 0.0 for name in PHASE_NAMES}
    counts = {name: 0 for name in PHASE_NAMES}
    observed_count = 0
    for observation_index, raw_observation in enumerate(raw_observations):
        observation = _mapping(
            raw_observation,
            field=f"{arm_id}.observations[{observation_index}]",
        )
        raw_phases = observation.get("phase_timings")
        if raw_phases is None:
            continue
        phases = _mapping(
            raw_phases,
            field=f"{arm_id}.observations[{observation_index}].phase_timings",
        )
        observed_count += 1
        for name in PHASE_NAMES:
            if name not in phases:
                continue
            seconds = _number(
                phases[name],
                field=f"{arm_id}.observations[{observation_index}].phase_timings.{name}",
            )
            if seconds is None:
                continue
            sums[name] += seconds
            counts[name] += 1

    fallback_total = _number(row.get("inference_seconds"), field=f"{arm_id}.inference_seconds")
    if counts["total"] == 0 and fallback_total is not None:
        sums["total"] = fallback_total
        counts["total"] = 1
        total_source = "arm.inference_seconds"
    elif counts["total"]:
        total_source = "observation.phase_timings"
    else:
        total_source = "unavailable"

    # The encoder phase telemetry does not include every caller-side interval
    # (for example rank/fusion or synchronization).  Keep that small residual
    # explicit rather than implying that the phase rows exhaust the arm total.
    observed_total = (
        sums["total"] if counts["total"] and total_source == "observation.phase_timings" else None
    )
    unattributed = (
        max(0.0, fallback_total - observed_total)
        if fallback_total is not None and observed_total is not None
        else None
    )

    return {
        "arm_id": arm_id,
        "status": "MEASURED" if observed_count else "NOT_RECORDED",
        "observation_count": len(raw_observations),
        "phase_observation_count": observed_count,
        "total_source": total_source,
        "unattributed_seconds": unattributed,
        "unattributed_source": (
            "arm_minus_observation_phase_total" if unattributed is not None else None
        ),
        "sum_seconds": {name: sums[name] if counts[name] else None for name in PHASE_NAMES},
        "mean_seconds": {
            name: (sums[name] / counts[name]) if counts[name] else None for name in PHASE_NAMES
        },
        "sample_count": {name: counts[name] for name in PHASE_NAMES},
    }


def _pipeline_phase_totals(value: object) -> list[dict[str, Any]]:
    """Project optional producer/consumer phase totals into seconds."""

    if value is None:
        return []
    rows = _sequence(value, field="pipeline.timing.phase_totals")
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"pipeline.timing.phase_totals[{index}]")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"pipeline.timing.phase_totals[{index}].name must be non-empty")
        total_ns = _number(
            row.get("total_ns"),
            field=f"pipeline.timing.phase_totals[{index}].total_ns",
        )
        mean_ns = _number(
            row.get("mean_ns"),
            field=f"pipeline.timing.phase_totals[{index}].mean_ns",
        )
        max_ns = _number(
            row.get("max_ns"),
            field=f"pipeline.timing.phase_totals[{index}].max_ns",
        )
        result.append(
            {
                "name": name.strip(),
                "count": _integer(row.get("count"), field=f"phase_totals[{index}].count"),
                "total_seconds": total_ns / 1_000_000_000 if total_ns is not None else None,
                "mean_seconds": mean_ns / 1_000_000_000 if mean_ns is not None else None,
                "max_seconds": max_ns / 1_000_000_000 if max_ns is not None else None,
            }
        )
    return result


def _fraction(part: float | None, whole: float | None) -> float | None:
    if part is None or whole is None or whole <= 0.0:
        return None
    return part / whole


def _ingest_vs_cached_inference(
    runtime: Mapping[str, Any],
    pipeline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Separate one-time ingest from inference over reused decoded frames.

    The runtime artifact measures shared decode once and reports each arm's
    inference-only time.  That is a *decode-reuse estimate*, not a persistent
    cache-hit benchmark; the distinction is retained in the returned note.
    """

    raw_arms = _sequence(runtime.get("arms", []), field="runtime.arms")
    runtime_rows: list[dict[str, Any]] = []
    for index, raw_arm in enumerate(raw_arms):
        arm = _mapping(raw_arm, field=f"runtime.arms[{index}]")
        arm_id = str(arm.get("arm_id", f"arm-{index}"))
        ingest_seconds = _number(
            arm.get("decode_seconds_shared"),
            field=f"{arm_id}.decode_seconds_shared",
        )
        cached_seconds = _number(
            arm.get("inference_seconds"),
            field=f"{arm_id}.inference_seconds",
        )
        e2e_seconds = _number(
            arm.get("estimated_e2e_seconds"),
            field=f"{arm_id}.estimated_e2e_seconds",
        )
        summed_seconds = (
            ingest_seconds + cached_seconds
            if ingest_seconds is not None and cached_seconds is not None
            else None
        )
        denominator = e2e_seconds if e2e_seconds and e2e_seconds > 0.0 else summed_seconds
        runtime_rows.append(
            {
                "arm_id": arm_id,
                "ingest_seconds": ingest_seconds,
                "cached_inference_seconds": cached_seconds,
                "estimated_e2e_seconds": e2e_seconds,
                "ingest_plus_cached_seconds": summed_seconds,
                "ingest_fraction": _fraction(ingest_seconds, denominator),
                "cached_inference_fraction": _fraction(cached_seconds, denominator),
            }
        )

    pipeline_summary: dict[str, Any] | None = None
    if pipeline is not None:
        phase_rows = _sequence(pipeline.get("phase_totals", []), field="pipeline.phase_totals")
        phase_by_name = {
            str(_mapping(row, field=f"pipeline.phase_totals[{index}]").get("name")): _mapping(
                row, field=f"pipeline.phase_totals[{index}]"
            )
            for index, row in enumerate(phase_rows)
        }

        def _phase_seconds(name: str) -> float | None:
            row = phase_by_name.get(name)
            if row is None:
                return None
            return _number(row.get("total_seconds"), field=f"pipeline.phase_totals.{name}")

        ingest_seconds = _phase_seconds("media_decode")
        cached_seconds = _phase_seconds("model")
        wall_seconds = _number(pipeline.get("wall_seconds"), field="pipeline.wall_seconds")
        pipeline_summary = {
            "ingest_seconds": ingest_seconds,
            "cached_inference_seconds": cached_seconds,
            "wall_seconds": wall_seconds,
            "ingest_plus_cached_seconds": (
                ingest_seconds + cached_seconds
                if ingest_seconds is not None and cached_seconds is not None
                else None
            ),
            "ingest_fraction": _fraction(ingest_seconds, wall_seconds),
            "cached_inference_fraction": _fraction(cached_seconds, wall_seconds),
            "phase_source": "pipeline.phase_totals",
        }

    return {
        "status": "DIAGNOSTIC_ESTIMATE",
        "cache_semantics": (
            "cached_inference_seconds is inference after shared decode within this "
            "benchmark; no persistent cache-hit run was measured."
        ),
        "runtime_arms": runtime_rows,
        "pipeline": pipeline_summary,
    }


def _pipeline(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    root = _mapping(value.get("pipeline", value), field="pipeline")
    timing = _mapping(root.get("timing", {}), field="pipeline.timing")
    raw_backpressure = timing.get("producer_backpressure_ns")
    backpressure_value = _number(raw_backpressure, field="pipeline.producer_backpressure_ns")
    backpressure = backpressure_value / 1_000_000_000 if backpressure_value is not None else None
    return {
        "batch_size": _integer(root.get("batch_size"), field="pipeline.batch_size"),
        "window_chunk_size": _integer(
            root.get("window_chunk_size"), field="pipeline.window_chunk_size"
        ),
        "queue_capacity": _integer(
            timing.get("queue_capacity", root.get("queue_capacity")),
            field="pipeline.queue_capacity",
        ),
        "wall_seconds": _number(timing.get("wall_seconds"), field="pipeline.wall_seconds"),
        "overlap_seconds": _number(timing.get("overlap_seconds"), field="pipeline.overlap_seconds"),
        "estimated_speedup": _number(
            timing.get("estimated_speedup"), field="pipeline.estimated_speedup"
        ),
        "producer_utilization": _number(
            timing.get("producer_utilization"), field="pipeline.producer_utilization"
        ),
        "consumer_utilization": _number(
            timing.get("consumer_utilization"), field="pipeline.consumer_utilization"
        ),
        "backpressure_seconds": backpressure,
        "phase_totals": _pipeline_phase_totals(
            timing.get("phase_totals", root.get("phase_totals"))
        ),
    }


def _matrix_arm(value: object, *, index: int) -> dict[str, Any]:
    row = _mapping(value, field=f"matrix.arms[{index}]")
    arm_id = str(row.get("arm_id", f"arm-{index}"))
    return {
        "arm_id": arm_id,
        "frame_count": _integer(row.get("frame_count"), field=f"{arm_id}.frame_count"),
        "total_pixel_budget": _integer(
            row.get("total_pixel_budget"), field=f"{arm_id}.total_pixel_budget"
        ),
        "observed_grid_thw": row.get("observed_grid_thw", []),
        "estimated_e2e_seconds": _number(
            row.get("estimated_e2e_seconds"), field=f"{arm_id}.estimated_e2e_seconds"
        ),
        "inference_speedup_vs_serial": _number(
            row.get("inference_speedup_vs_serial"),
            field=f"{arm_id}.inference_speedup_vs_serial",
        ),
        "batch_top1_parity": _number(
            row.get("batch_top1_parity"), field=f"{arm_id}.batch_top1_parity"
        ),
        "batch_full_order_parity": _number(
            row.get("batch_full_order_parity"), field=f"{arm_id}.batch_full_order_parity"
        ),
        "batch_mean_cosine_vs_serial": _number(
            row.get("batch_mean_cosine_vs_serial"),
            field=f"{arm_id}.batch_mean_cosine_vs_serial",
        ),
        "provisional_top1_distribution": _label_counts(
            row.get("provisional_top1_distribution"),
            field=f"{arm_id}.provisional_top1_distribution",
        ),
    }


def build_compact_report(
    runtime: Mapping[str, Any],
    *,
    pipeline: Mapping[str, Any] | None = None,
    matrix: Mapping[str, Any] | None = None,
    source_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Project existing reports without inference or media access.

    ``source_paths`` is accepted for compatibility with earlier callers but is
    intentionally ignored so filesystem details do not enter the projection.
    """

    del source_paths
    source = _mapping(runtime.get("source", {}), field="runtime.source")
    arms = _sequence(runtime.get("arms", []), field="runtime.arms")
    # A runtime benchmark may embed its optional pipeline arm.  Prefer the
    # explicitly supplied sidecar, but fall back to the embedded report so a
    # caller cannot accidentally lose overlap telemetry by omitting it.
    pipeline_report = _pipeline(pipeline if pipeline is not None else runtime.get("pipeline"))
    report: dict[str, Any] = {
        "format": FORMAT,
        "authority": runtime.get("authority", AUTHORITY),
        "status": runtime.get("status", "MEASURED_NONPRODUCTION"),
        "official_gold_status": runtime.get("official_gold_status", "NOT_ESTABLISHED"),
        "official_quality_status": runtime.get("official_quality_status", "NOT_MEASURED"),
        "production_eligible": bool(runtime.get("production_eligible", False)),
        "quality_claim": False,
        "scope": {
            "window_count": _integer(source.get("window_count"), field="source.window_count"),
            "camera_count": _integer(source.get("camera_count"), field="source.camera_count"),
            "camera_window_input_count": _integer(
                source.get("camera_window_input_count"),
                field="source.camera_window_input_count",
            ),
            "represented_window_seconds": _number(
                source.get("represented_window_seconds"),
                field="source.represented_window_seconds",
            ),
        },
        "throughput": [_runtime_arm(item, index=index) for index, item in enumerate(arms)],
        "phase_breakdown": [_phase_breakdown(item, index=index) for index, item in enumerate(arms)],
        "ingest_vs_cached_inference": _ingest_vs_cached_inference(runtime, pipeline_report),
        "pipeline": pipeline_report,
        "frame_grid_matrix": [],
        "caveat": (
            "No independent source-bound gold has been adjudicated.  Provisional labels, "
            "scores and parity values are diagnostics only, not accuracy, confidence, or "
            "quality claims; fixed windows are context units rather than action boundaries."
        ),
    }
    if matrix is not None:
        matrix_arms = _sequence(matrix.get("arms", []), field="matrix.arms")
        report["frame_grid_matrix"] = [
            _matrix_arm(item, index=index) for index, item in enumerate(matrix_arms)
        ]
    return report


def build_compact_projection(
    runtime: Mapping[str, Any],
    pipeline: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility alias with the explicit three-report signature."""

    return build_compact_report(runtime, pipeline=pipeline, matrix=matrix)


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(report: Mapping[str, Any]) -> str:
    scope = _mapping(report.get("scope", {}), field="report.scope")
    lines = [
        "# WeMM runtime diagnostic compact report",
        "",
        (
            f"- Scope: {scope.get('window_count')} windows, {scope.get('camera_count')} cameras, "
            f"{scope.get('camera_window_input_count')} camera-window inputs; "
            f"{scope.get('represented_window_seconds')} s represented."
        ),
        f"- Status: `{report.get('status')}`; quality claim: `{report.get('quality_claim')}`.",
        (
            f"- Gold status: `{report.get('official_gold_status')}`; "
            f"quality status: `{report.get('official_quality_status')}`."
        ),
        "",
        "## Throughput",
        "",
        "| arm | batch | frames | inference s | estimated e2e s | camera RT factor |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for raw in _sequence(report.get("throughput", []), field="report.throughput"):
        row = _mapping(raw, field="report.throughput[]")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("arm_id")),
                    _fmt(row.get("batch_size"), 0),
                    _fmt(row.get("frame_count"), 0),
                    _fmt(row.get("inference_seconds")),
                    _fmt(row.get("estimated_e2e_seconds")),
                    _fmt(row.get("camera_normalized_realtime")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Encoder phase breakdown",
            "",
            (
                "| arm | status | processor s | model s | postprocess s | total s | "
                "other s | measured observations |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for raw in _sequence(report.get("phase_breakdown", []), field="report.phase_breakdown"):
        row = _mapping(raw, field="report.phase_breakdown[]")
        means = _mapping(row.get("mean_seconds", {}), field="phase_breakdown.mean_seconds")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("arm_id")),
                    str(row.get("status")),
                    _fmt(means.get("processor")),
                    _fmt(means.get("model")),
                    _fmt(means.get("postprocess")),
                    _fmt(means.get("total")),
                    _fmt(row.get("unattributed_seconds")),
                    _fmt(row.get("phase_observation_count"), 0),
                )
            )
            + " |"
        )
    ingest_summary = report.get("ingest_vs_cached_inference")
    if isinstance(ingest_summary, Mapping):
        lines.extend(
            [
                "",
                "## Ingest versus cached inference",
                "",
                str(ingest_summary.get("cache_semantics")),
                "",
                (
                    "| arm | ingest s | cached inference s | ingest + cached s | "
                    "ingest fraction | cached fraction |"
                ),
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for raw in _sequence(
            ingest_summary.get("runtime_arms", []),
            field="ingest_vs_cached_inference.runtime_arms",
        ):
            row = _mapping(raw, field="ingest_vs_cached_inference.runtime_arms[]")
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row.get("arm_id")),
                        _fmt(row.get("ingest_seconds")),
                        _fmt(row.get("cached_inference_seconds")),
                        _fmt(row.get("ingest_plus_cached_seconds")),
                        _fmt(row.get("ingest_fraction")),
                        _fmt(row.get("cached_inference_fraction")),
                    )
                )
                + " |"
            )
        pipeline_summary = ingest_summary.get("pipeline")
        if isinstance(pipeline_summary, Mapping):
            lines.append(
                "Pipeline: ingest "
                f"{_fmt(pipeline_summary.get('ingest_seconds'))} s; cached inference "
                f"{_fmt(pipeline_summary.get('cached_inference_seconds'))} s; wall "
                f"{_fmt(pipeline_summary.get('wall_seconds'))} s."
            )
    pipeline = report.get("pipeline")
    if isinstance(pipeline, Mapping):
        lines.extend(
            [
                "",
                "## Producer-consumer throughput",
                "",
                (
                    f"Wall {_fmt(pipeline.get('wall_seconds'))} s; overlap "
                    f"{_fmt(pipeline.get('overlap_seconds'))} s; estimated speedup "
                    f"{_fmt(pipeline.get('estimated_speedup'))}x; backpressure "
                    f"{_fmt(pipeline.get('backpressure_seconds'))} s."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Batch parity",
            "",
            "| arm | mean cosine | top-1 | full order | row order |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for raw in _sequence(report.get("throughput", []), field="report.throughput"):
        row = _mapping(raw, field="report.throughput[]")
        parity = row.get("parity_vs_serial")
        if not isinstance(parity, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("arm_id")),
                    _fmt(parity.get("mean_cosine"), 6),
                    _fmt(parity.get("top1_equal_fraction")),
                    _fmt(parity.get("full_order_equal_fraction")),
                    str(parity.get("row_order_preserved")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Frame/grid matrix",
            "",
            "| arm | frames | budget | grid | e2e s | top-1 parity | full-order parity |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for raw in _sequence(report.get("frame_grid_matrix", []), field="report.frame_grid_matrix"):
        row = _mapping(raw, field="report.frame_grid_matrix[]")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("arm_id")),
                    _fmt(row.get("frame_count"), 0),
                    _fmt(row.get("total_pixel_budget"), 0),
                    str(row.get("observed_grid_thw")),
                    _fmt(row.get("estimated_e2e_seconds")),
                    _fmt(row.get("batch_top1_parity")),
                    _fmt(row.get("batch_full_order_parity")),
                )
            )
            + " |"
        )
    lines.extend(["", "## Provisional labels and camera consistency", ""])
    for raw in _sequence(report.get("throughput", []), field="report.throughput"):
        row = _mapping(raw, field="report.throughput[]")
        lines.append(f"- `{row.get('arm_id')}` labels: {row.get('provisional_top1_distribution')}")
        camera = row.get("camera_consistency")
        if isinstance(camera, Mapping):
            lines.append(
                f"  camera modal fraction {_fmt(camera.get('mean_modal_top1_fraction'))}; "
                f"all-camera-same fraction {_fmt(camera.get('all_camera_same_fraction'))}."
            )
    lines.extend(["", "## Caveat", "", str(report.get("caveat")), ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_compact_report(
            _load(args.runtime),
            pipeline=_load(args.pipeline) if args.pipeline else None,
            matrix=_load(args.matrix) if args.matrix else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        markdown_path = args.markdown_output or args.output.with_suffix(".md")
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"compact WeMM projection failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "markdown_output": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY",
    "FORMAT",
    "build_compact_projection",
    "build_compact_report",
    "main",
]
