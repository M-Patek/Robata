from __future__ import annotations

import json

import pytest

from robata.benchmark.production_wemm_temporal import resolve_wemm_temporal_segments
from robata.benchmark.production_wemm_temporal_refinement import (
    apply_refined_boundaries,
    plan_wemm_temporal_refinement,
)
from robata.benchmark.production_wemm_temporal_score_refinement import (
    plan_wemm_score_refinement_grid,
    resolve_wemm_score_refinement,
)


def _candidate(action: str, score: float, *, rank: int = 1) -> dict[str, object]:
    return {
        "provisional_id": action,
        "label_text": action.replace("_", " "),
        "rank": rank,
        "score": score,
        "camera_support": 2,
        "evidence": [{"camera_id": "cam_01"}, {"camera_id": "cam_02"}],
    }


def _window(window_id: str, start: float, end: float, score: float) -> dict[str, object]:
    return {
        "window_id": window_id,
        "start_seconds": start,
        "end_seconds": end,
        "proposals": [{"top_k": [_candidate("open_cupboard", score)]}],
    }


def _coarse() -> dict[str, object]:
    return resolve_wemm_temporal_segments(
        [
            _window("w0", 0.0, 4.0, 0.2),
            _window("w1", 1.0, 5.0, 0.85),
            _window("w2", 2.0, 6.0, 0.80),
            _window("w3", 3.0, 7.0, 0.2),
        ],
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.1,
    )


def test_grid_has_nested_before_after_contexts_and_never_marks_edges() -> None:
    coarse = _coarse()
    parent = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    fine = plan_wemm_score_refinement_grid(
        coarse,
        parent_plan=parent,
        probe_span_seconds=0.5,
        points_per_side=2,
        levels=2,
    )

    requests = fine["requests"]
    assert len(requests) == 12  # type: ignore[arg-type]
    assert {row["probe_side"] for row in requests} == {"before", "after"}  # type: ignore[index]
    assert {row["level"] for row in requests} == {0, 1}  # type: ignore[index]
    assert all(row["request_edges_are_not_boundaries"] is True for row in requests)  # type: ignore[index]
    assert fine["controls"]["model_invoked"] is False  # type: ignore[index]
    json.dumps(fine)


def test_score_crossings_emit_parent_measured_rows_consumable_by_apply() -> None:
    coarse = _coarse()
    parent = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    fine = plan_wemm_score_refinement_grid(
        coarse,
        parent_plan=parent,
        probe_span_seconds=0.5,
        points_per_side=2,
        levels=2,
    )

    scores: dict[str, dict[str, float]] = {}
    for request in fine["requests"]:  # type: ignore[index]
        center = (request["start_seconds"] + request["end_seconds"]) / 2.0  # type: ignore[operator]
        role = request["role"]
        if role == "onset":
            value = 0.85 if center >= 2.45 else 0.20
        else:
            value = 0.80 if center < 4.65 else 0.20
        scores[request["request_id"]] = {"score": value}

    result = resolve_wemm_score_refinement(
        parent,
        fine,
        scores,
        start_threshold=0.65,
        stop_threshold=0.50,
        score_policy="absolute",
    )
    assert result["diagnostics"]["measured_result_count"] == 2  # type: ignore[index]
    rows = result["results"]  # type: ignore[index]
    assert all(row["status"] == "MEASURED" for row in rows)

    projected = apply_refined_boundaries(coarse, parent, result)
    refined = projected["refined_segments"][0]  # type: ignore[index]
    assert refined["refinement_status"] == "REFINED"
    assert refined["boundary_status"] == "MODEL_REFINED"
    assert refined["start_seconds"] == pytest.approx(2.45, abs=0.20)
    assert refined["end_seconds"] == pytest.approx(4.65, abs=0.20)
    assert refined["automatic_eligible"] is False
    assert projected["segments"] == coarse["segments"]
    json.dumps(projected)


def test_non_bracketed_or_edge_clipped_scores_stay_uncertain() -> None:
    coarse = _coarse()
    parent = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    fine = plan_wemm_score_refinement_grid(
        coarse,
        parent_plan=parent,
        probe_span_seconds=0.5,
        points_per_side=2,
        levels=1,
    )
    # Flat scores cannot establish either transition, even though every probe
    # has a valid source interval.
    scores = {row["request_id"]: {"score": 0.70} for row in fine["requests"]}  # type: ignore[index]
    result = resolve_wemm_score_refinement(parent, fine, scores, score_policy="absolute")
    assert result["diagnostics"]["measured_result_count"] == 0  # type: ignore[index]
    assert all(row["status"] == "UNCERTAIN" for row in result["results"])  # type: ignore[index]
