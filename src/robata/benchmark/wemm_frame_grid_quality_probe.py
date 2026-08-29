"""Lightweight, artifact-only frame/grid stability probe for WeMM.

The frame/grid matrix runner already records runtime and provisional ranking
telemetry, but its plan artifact intentionally does not make a quality claim.
This module joins completed *runtime reports* from the same bounded cohort and
reports only configuration/stability diagnostics:

* observed processor grids and frame counts;
* model/inference and estimated end-to-end timing;
* provisional label distributions and Top-1/Top-2 margin summaries;
* per-window modal-label agreement against an explicitly selected reference.

No model is loaded, no media is decoded, no ontology or Mapper is touched, and
no identity/hash is derived.  The probe is useful for choosing the next matrix
arm, not for estimating accuracy: official quality and gold remain
``NOT_MEASURED``/``NOT_ESTABLISHED``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

FORMAT: Final = "robata-wemm-frame-grid-quality-probe-v1"
AUTHORITY: Final = "LOCAL_NONPRODUCTION_ONLY"
DEFAULT_RUNTIME_ARM: Final = "batch4"


class WemmFrameGridQualityProbeError(ValueError):
    """Raised when runtime artifacts cannot be compared safely."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WemmFrameGridQualityProbeError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise WemmFrameGridQualityProbeError(f"{field} must be an array")
    return value


def _finite(value: object, *, field: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise WemmFrameGridQualityProbeError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise WemmFrameGridQualityProbeError(f"{field} must be finite")
    return result


def _int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise WemmFrameGridQualityProbeError(f"{field} must be an integer")
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise WemmFrameGridQualityProbeError(f"{field} must be an integer") from exc
    if result < minimum:
        raise WemmFrameGridQualityProbeError(f"{field} must be >= {minimum}")
    return result


def _grid(value: object, *, field: str) -> tuple[int, int, int]:
    outer = _sequence(value, field=field)
    row: Sequence[Any]
    if (
        len(outer) == 1
        and isinstance(outer[0], Sequence)
        and not isinstance(outer[0], (str, bytes, bytearray))
    ):
        row = outer[0]
    else:
        row = outer
    if len(row) != 3:
        raise WemmFrameGridQualityProbeError(f"{field} must contain [t, h, w]")
    return tuple(_int(item, field=f"{field}[{index}]") for index, item in enumerate(row))  # type: ignore[return-value]


def _grids(value: object, *, field: str) -> list[tuple[int, int, int]]:
    """Normalize serial and batched ``video_grid_thw`` telemetry.

    Serial observations carry ``[[t, h, w]]`` while a batched observation
    carries one ``[t, h, w]`` row per item.  Keep both forms explicit so a
    malformed nested row cannot silently become a quality signal.
    """

    outer = _sequence(value, field=field)
    if len(outer) == 3 and all(
        not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray))
        for item in outer
    ):
        return [_grid(outer, field=field)]
    result: list[tuple[int, int, int]] = []
    for index, row in enumerate(outer):
        result.append(_grid(row, field=f"{field}[{index}]"))
    return result


def _modal_label(counts: Mapping[str, Any]) -> tuple[str | None, int]:
    parsed: list[tuple[str, int]] = []
    for raw_label, raw_count in counts.items():
        label = str(raw_label).strip()
        if not label:
            continue
        count = _int(raw_count, field=f"label_counts[{label!r}]", minimum=0)
        if count == 0:
            continue
        parsed.append((label, count))
    if not parsed:
        return None, 0
    parsed.sort(key=lambda item: (-item[1], item[0]))
    return parsed[0]


def _source_signature(
    report: Mapping[str, Any], *, field: str
) -> tuple[str, tuple[str, ...], int, int]:
    source = _mapping(report.get("source"), field=f"{field}.source")
    path = str(source.get("path", "")).strip()
    if not path:
        raise WemmFrameGridQualityProbeError(f"{field}.source.path is required")
    camera_count = _int(source.get("camera_count"), field=f"{field}.source.camera_count")
    window_count = _int(source.get("window_count"), field=f"{field}.source.window_count")
    input_order = _sequence(source.get("input_order"), field=f"{field}.source.input_order")
    window_ids = tuple(
        sorted(
            {
                str(_mapping(row, field=f"{field}.source.input_order[]").get("window_id"))
                for row in input_order
                if _mapping(row, field=f"{field}.source.input_order[]").get("window_id") is not None
            }
        )
    )
    if not window_ids:
        raise WemmFrameGridQualityProbeError(f"{field}.source.input_order has no window ids")
    return path, window_ids, camera_count, window_count


def _extract_arm(report: Mapping[str, Any], *, arm_id: str, field: str) -> Mapping[str, Any]:
    if report.get("status") != "MEASURED_NONPRODUCTION":
        raise WemmFrameGridQualityProbeError(f"{field}.status must be MEASURED_NONPRODUCTION")
    if report.get("official_quality_status") != "NOT_MEASURED":
        raise WemmFrameGridQualityProbeError(
            f"{field}.official_quality_status must be NOT_MEASURED"
        )
    if report.get("official_gold_status") != "NOT_ESTABLISHED":
        raise WemmFrameGridQualityProbeError(
            f"{field}.official_gold_status must be NOT_ESTABLISHED"
        )
    arms = _sequence(report.get("arms"), field=f"{field}.arms")
    for index, raw_arm in enumerate(arms):
        arm = _mapping(raw_arm, field=f"{field}.arms[{index}]")
        if str(arm.get("arm_id")) == arm_id:
            return arm
    raise WemmFrameGridQualityProbeError(f"{field} does not contain runtime arm {arm_id!r}")


def _arm_projection(arm: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    observations = _sequence(arm.get("observations", ()), field=f"{field}.observations")
    grids: list[tuple[int, int, int]] = []
    frame_counts: list[int] = []
    for index, raw in enumerate(observations):
        row = _mapping(raw, field=f"{field}.observations[{index}]")
        if row.get("modality") != "video":
            continue
        if row.get("frame_count") is not None:
            frame_counts.append(
                _int(
                    row["frame_count"],
                    field=f"{field}.observations[{index}].frame_count",
                    minimum=1,
                )
            )
        if row.get("video_grid_thw") is not None:
            grids.extend(
                _grids(
                    row["video_grid_thw"],
                    field=f"{field}.observations[{index}].video_grid_thw",
                )
            )

    diagnostic = _mapping(arm.get("rank_diagnostic"), field=f"{field}.rank_diagnostic")
    margin = _mapping(
        diagnostic.get("top1_top2_margin_not_calibrated", {}),
        field=f"{field}.rank_diagnostic.top1_top2_margin_not_calibrated",
    )
    top_counts = _mapping(
        diagnostic.get("top_label_counts_not_gold", {}),
        field=f"{field}.rank_diagnostic.top_label_counts_not_gold",
    )
    camera = _mapping(
        diagnostic.get("camera_consistency_not_gold", {}),
        field=f"{field}.rank_diagnostic.camera_consistency_not_gold",
    )
    per_window: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(
        _sequence(camera.get("per_window", ()), field=f"{field}.camera.per_window")
    ):
        row = _mapping(raw, field=f"{field}.camera.per_window[{index}]")
        window_id = str(row.get("window_id", "")).strip()
        if not window_id:
            continue
        counts = _mapping(
            row.get("top1_labels_not_gold", {}),
            field=f"{field}.camera.per_window[{index}].top1_labels_not_gold",
        )
        modal, modal_count = _modal_label(counts)
        observed = sum(
            _int(value, field="camera label count", minimum=0) for value in counts.values()
        )
        per_window[window_id] = {
            "modal_top1": modal,
            "modal_count": modal_count,
            "camera_count_observed": _int(
                row.get("camera_count_observed", observed),
                field=f"{field}.camera.per_window[{index}].camera_count_observed",
            ),
            "modal_top1_fraction": _finite(
                row.get("modal_top1_fraction"),
                field=f"{field}.camera.per_window[{index}].modal_top1_fraction",
                allow_none=True,
            ),
            "top1_labels_not_gold": dict(counts),
        }
    return {
        "frame_count": _int(arm.get("frame_count"), field=f"{field}.frame_count", minimum=1),
        "pixel_budget": _int(
            arm.get("video_max_pixels"), field=f"{field}.video_max_pixels", minimum=1
        ),
        "input_count": _int(arm.get("input_count"), field=f"{field}.input_count", minimum=0),
        "decode_seconds_shared": _finite(
            arm.get("decode_seconds_shared"), field=f"{field}.decode_seconds_shared"
        ),
        "inference_seconds": _finite(
            arm.get("inference_seconds"), field=f"{field}.inference_seconds"
        ),
        "estimated_e2e_seconds": _finite(
            arm.get("estimated_e2e_seconds"), field=f"{field}.estimated_e2e_seconds"
        ),
        "source_camera_normalized_realtime": _finite(
            arm.get("source_camera_normalized_realtime"),
            field=f"{field}.source_camera_normalized_realtime",
            allow_none=True,
        ),
        "observed_grids": [list(grid) for grid in sorted(set(grids))],
        "observed_frame_counts": sorted(set(frame_counts)),
        "top_label_counts_not_gold": dict(top_counts),
        "margin": {
            "count": _int(margin.get("count", 0), field=f"{field}.margin.count"),
            "mean": _finite(margin.get("mean"), field=f"{field}.margin.mean", allow_none=True),
            "min": _finite(margin.get("min"), field=f"{field}.margin.min", allow_none=True),
            "max": _finite(margin.get("max"), field=f"{field}.margin.max", allow_none=True),
        },
        "camera_consistency": {
            "mean_modal_top1_fraction": _finite(
                camera.get("mean_modal_top1_fraction"),
                field=f"{field}.camera.mean_modal_top1_fraction",
                allow_none=True,
            ),
            "all_camera_same_fraction": _finite(
                camera.get("all_camera_same_fraction"),
                field=f"{field}.camera.all_camera_same_fraction",
                allow_none=True,
            ),
            "per_window": per_window,
        },
    }


def _agreement(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    ref_windows = _mapping(reference["camera_consistency"], field="reference.camera_consistency")[
        "per_window"
    ]
    cand_windows = _mapping(candidate["camera_consistency"], field="candidate.camera_consistency")[
        "per_window"
    ]
    ref_map = _mapping(ref_windows, field="reference.per_window")
    cand_map = _mapping(cand_windows, field="candidate.per_window")
    common = sorted(set(ref_map) & set(cand_map))
    matches = 0
    fractions: list[float] = []
    disagreements: list[dict[str, Any]] = []
    for window_id in common:
        ref_row = _mapping(ref_map[window_id], field=f"reference.per_window.{window_id}")
        cand_row = _mapping(cand_map[window_id], field=f"candidate.per_window.{window_id}")
        ref_label = ref_row.get("modal_top1")
        cand_label = cand_row.get("modal_top1")
        # Two missing modal labels do not constitute agreement: this is an
        # unobserved window, not evidence that the arms selected the same
        # action.
        equal = ref_label is not None and cand_label is not None and ref_label == cand_label
        matches += int(equal)
        ref_fraction = ref_row.get("modal_top1_fraction")
        cand_fraction = cand_row.get("modal_top1_fraction")
        if ref_fraction is not None and cand_fraction is not None:
            fractions.append(abs(float(ref_fraction) - float(cand_fraction)))
        if not equal:
            disagreements.append(
                {
                    "window_id": window_id,
                    "reference_modal_top1": ref_label,
                    "candidate_modal_top1": cand_label,
                }
            )
    return {
        "common_window_count": len(common),
        "modal_top1_agreement_fraction": matches / len(common) if common else None,
        "mean_camera_modal_fraction_abs_delta": sum(fractions) / len(fractions)
        if fractions
        else None,
        "disagreements": disagreements,
    }


def analyze_wemm_frame_grid_quality(
    arms: Sequence[Mapping[str, Any]],
    *,
    reference_arm_id: str | None = None,
    runtime_arm_id: str = DEFAULT_RUNTIME_ARM,
) -> dict[str, Any]:
    """Compare completed runtime artifacts without invoking media or models."""

    if not arms:
        raise WemmFrameGridQualityProbeError("arms must not be empty")
    if not isinstance(runtime_arm_id, str) or not runtime_arm_id.strip():
        raise WemmFrameGridQualityProbeError("runtime_arm_id must be non-empty")
    projections: list[dict[str, Any]] = []
    cohort: tuple[str, tuple[str, ...], int, int] | None = None
    seen_ids: set[str] = set()
    for index, raw in enumerate(arms):
        entry = _mapping(raw, field=f"arms[{index}]")
        arm_id = str(entry.get("arm_id", "")).strip()
        if not arm_id or arm_id in seen_ids:
            raise WemmFrameGridQualityProbeError(
                f"arms[{index}].arm_id must be unique and non-empty"
            )
        seen_ids.add(arm_id)
        report = _mapping(entry.get("runtime_report"), field=f"arms[{index}].runtime_report")
        signature = _source_signature(report, field=f"arms[{index}].runtime_report")
        if cohort is None:
            cohort = signature
        elif signature != cohort:
            raise WemmFrameGridQualityProbeError("runtime reports do not describe the same cohort")
        selected = _extract_arm(
            report, arm_id=runtime_arm_id, field=f"arms[{index}].runtime_report"
        )
        projection = _arm_projection(
            selected, field=f"arms[{index}].runtime_report.{runtime_arm_id}"
        )
        projection.update(
            {
                "arm_id": arm_id,
                "source_artifact": str(entry.get("source_artifact", "")),
                "runtime_arm_id": runtime_arm_id,
            }
        )
        projections.append(projection)
    assert cohort is not None
    reference_id = reference_arm_id or projections[0]["arm_id"]
    reference = next((row for row in projections if row["arm_id"] == reference_id), None)
    if reference is None:
        raise WemmFrameGridQualityProbeError(f"reference arm is not present: {reference_id!r}")
    for row in projections:
        row["agreement_vs_reference"] = (
            {"reference": reference_id, "self": True}
            if row["arm_id"] == reference_id
            else _agreement(reference, row)
        )

    return {
        "format": FORMAT,
        "authority": AUTHORITY,
        "status": "MEASURED_NONPRODUCTION_ARTIFACT_ONLY",
        "official_quality_status": "NOT_MEASURED",
        "official_gold_status": "NOT_ESTABLISHED",
        "quality_claim": False,
        "production_eligible": False,
        "scope": {
            "source_path": cohort[0],
            "window_ids": list(cohort[1]),
            "camera_count": cohort[2],
            "window_count": cohort[3],
            "runtime_arm_id": runtime_arm_id,
            "reference_arm_id": reference_id,
        },
        "arms": projections,
        "controls": {
            "model_invoked": False,
            "media_decoded": False,
            "gold_read": False,
            "gold_written": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "heldout_100_opened": False,
            "hash_or_digest_computed": False,
        },
        "limitations": [
            (
                "Agreement and label distributions are provisional stability diagnostics, "
                "not accuracy."
            ),
            "The caller owns scope keys and artifact provenance; this probe derives no identity.",
            "A source-bound Terra/human reference is required before quality promotion.",
        ],
    }


__all__ = [
    "AUTHORITY",
    "DEFAULT_RUNTIME_ARM",
    "FORMAT",
    "WemmFrameGridQualityProbeError",
    "analyze_wemm_frame_grid_quality",
]
