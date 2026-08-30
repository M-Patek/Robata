from __future__ import annotations

import pytest

from robata.benchmark.production_wemm_temporal import resolve_wemm_temporal_segments
from robata.benchmark.production_wemm_temporal_refinement import (
    ProductionWemmTemporalRefinementError,
    apply_refined_boundaries,
    plan_wemm_temporal_refinement,
    plan_wemm_temporal_refinement_from_windows,
)


def _candidate(action: str, score: float, *, rank: int = 1) -> dict[str, object]:
    return {
        "provisional_id": action,
        "label_text": action.replace("_", " "),
        "label_variant": "canonical",
        "structured_labels": {},
        "rank": rank,
        "score": score,
        "camera_support": 2,
        "evidence": [{"camera_id": "cam_01"}, {"camera_id": "cam_02"}],
    }


def _window(
    window_id: str,
    start: float,
    end: float,
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "window_id": window_id,
        "start_seconds": start,
        "end_seconds": end,
        "proposals": [{"top_k": candidates}],
    }


def _coarse_report() -> dict[str, object]:
    return resolve_wemm_temporal_segments(
        [
            _window("w0", 0.0, 4.0, [_candidate("open_cupboard", 0.2)]),
            _window("w1", 1.0, 5.0, [_candidate("open_cupboard", 0.85)]),
            _window("w2", 2.0, 6.0, [_candidate("open_cupboard", 0.80)]),
            _window("w3", 3.0, 7.0, [_candidate("open_cupboard", 0.2)]),
        ],
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.1,
    )


def test_planner_emits_short_source_relative_requests_without_model_calls() -> None:
    plan = plan_wemm_temporal_refinement(_coarse_report(), refinement_span_seconds=1.0)

    assert plan["status"] == "REFINEMENT_REQUESTS_ONLY"
    assert plan["production_eligible"] is False
    assert plan["controls"]["model_invoked"] is False  # type: ignore[index]
    assert plan["controls"]["runner_recompute_required"] is True  # type: ignore[index]
    requests = plan["requests"]  # type: ignore[index]
    assert len(requests) == 2
    assert {(row["role"], row["start_seconds"], row["end_seconds"]) for row in requests} == {
        ("onset", 2.0, 3.0),
        ("offset", 4.0, 5.0),
    }
    assert all(row["requires_model_recompute"] is True for row in requests)
    assert all(row["boundary_status"] == "PENDING_MODEL_RECOMPUTE" for row in requests)
    assert all(row["request_timestamp_basis"] == "request_relative_seconds" for row in requests)
    assert plan["diagnostics"]["shorter_than_coarse_context"] is True  # type: ignore[index]


def test_duplicate_action_role_span_is_coalesced_with_lineage() -> None:
    report = _coarse_report()
    segment = dict(report["segments"][0])  # type: ignore[index]
    segment["segment_id"] = "duplicate-segment"
    report["segments"] = [report["segments"][0], segment]  # type: ignore[index]

    plan = plan_wemm_temporal_refinement(report, refinement_span_seconds=1.0)
    assert len(plan["requests"]) == 2  # type: ignore[arg-type]
    onset = next(row for row in plan["requests"] if row["role"] == "onset")  # type: ignore[index]
    assert onset["source_segment_ids"] == [
        "duplicate-segment",
        "open_cupboard@2.500000-4.500000",
    ]
    assert plan["diagnostics"]["deduplicated_request_count"] == 2  # type: ignore[index]


def test_edge_anchor_shifts_short_span_inside_source_bounds() -> None:
    report = {
        "format": "coarse",
        "status": "PROPOSALS_ONLY",
        "context_interval": {"start_seconds": 0.0, "end_seconds": 5.0},
        "diagnostics": {"context_grid": {"context_width_seconds": 4.0}},
        "segments": [
            {
                "segment_id": "edge",
                "action_key": "open_drawer",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "boundary_confidence": 0.0,
                "supporting_window_ids": ["w0"],
                "transition_diagnostics": {
                    "onset": {
                        "boundary_seconds": 0.0,
                        "boundary_method": "observed_probe_span",
                        "crossed_threshold": False,
                        "reason": "NO_PRECEDING_PROBE",
                        "confidence": 0.0,
                    },
                    "offset": {
                        "boundary_seconds": 1.0,
                        "boundary_method": "probe_center_midpoint",
                        "crossed_threshold": True,
                        "reason": "THRESHOLD_CROSSING",
                        "confidence": 0.8,
                    },
                },
            }
        ],
    }
    plan = plan_wemm_temporal_refinement(report, refinement_span_seconds=1.0)
    onset = next(row for row in plan["requests"] if row["role"] == "onset")  # type: ignore[index]
    assert onset["start_seconds"] == pytest.approx(0.0)
    assert onset["end_seconds"] == pytest.approx(1.0)
    assert onset["coarse_boundary_reason"] == "NO_PRECEDING_PROBE"
    assert plan["diagnostics"]["edge_boundary_count"] >= 1  # type: ignore[index]
    assert plan["diagnostics"]["unbracketed_boundary_count"] >= 1  # type: ignore[index]


def test_planner_rejects_probe_that_is_not_shorter_than_coarse_context() -> None:
    with pytest.raises(ProductionWemmTemporalRefinementError, match="shorter"):
        plan_wemm_temporal_refinement(_coarse_report(), refinement_span_seconds=4.0)


def test_empty_coarse_report_is_a_valid_no_work_plan() -> None:
    plan = plan_wemm_temporal_refinement(
        {
            "format": "coarse",
            "status": "PROPOSALS_ONLY",
            "context_interval": {"start_seconds": 0.0, "end_seconds": 4.0},
            "segments": [],
        }
    )
    assert plan["requests"] == []
    assert plan["controls"]["runner_recompute_required"] is False  # type: ignore[index]
    assert plan["diagnostics"]["candidate_boundary_count"] == 0  # type: ignore[index]


def test_window_convenience_wrapper_only_resolves_and_plans() -> None:
    plan = plan_wemm_temporal_refinement_from_windows(
        [
            _window("w0", 0.0, 4.0, [_candidate("open_cupboard", 0.2)]),
            _window("w1", 1.0, 5.0, [_candidate("open_cupboard", 0.8)]),
        ],
        refinement_span_seconds=1.0,
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
    )
    assert plan["controls"]["model_invoked"] is False  # type: ignore[index]
    assert len(plan["requests"]) == 2  # type: ignore[arg-type]


def test_apply_projects_request_relative_boundaries_additively() -> None:
    coarse = _coarse_report()
    plan = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    onset = next(row for row in plan["requests"] if row["role"] == "onset")  # type: ignore[index]
    offset = next(row for row in plan["requests"] if row["role"] == "offset")  # type: ignore[index]
    projected = apply_refined_boundaries(
        coarse,
        plan,
        {
            onset["request_id"]: {
                "status": "MEASURED",
                "timestamp_basis": "request_relative_seconds",
                "start_seconds": 0.10,
                "end_seconds": 0.40,
                "confidence": 0.9,
                "evidence": "handle begins moving",
            },
            offset["request_id"]: {
                "status": "MEASURED",
                "timestamp_basis": "request_relative_seconds",
                "start_seconds": 0.20,
                "end_seconds": 0.60,
                "confidence": 0.8,
                "evidence": "water stops",
            },
        },
    )

    # The original coarse segment remains unchanged; only the additive
    # refined sidecar carries source-relative boundaries.
    assert projected["segments"] == coarse["segments"]
    refined = projected["refined_segments"][0]  # type: ignore[index]
    assert refined["refinement_status"] == "REFINED"
    assert refined["boundary_status"] == "MODEL_REFINED"
    assert refined["start_seconds"] == pytest.approx(2.10)
    assert refined["end_seconds"] == pytest.approx(4.60)
    assert projected["temporal_refinement"]["diagnostics"]["refined_segment_count"] == 1  # type: ignore[index]


def test_apply_keeps_partial_and_missing_results_review_only() -> None:
    coarse = _coarse_report()
    plan = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    onset = next(row for row in plan["requests"] if row["role"] == "onset")  # type: ignore[index]
    projected = apply_refined_boundaries(
        coarse,
        plan,
        {
            onset["request_id"]: {
                "status": "UNCERTAIN",
                "evidence": "boundary occluded",
            }
        },
    )
    refined = projected["refined_segments"][0]  # type: ignore[index]
    assert refined["refinement_status"] == "UNRESOLVED"
    assert refined["start_seconds"] is None
    assert refined["end_seconds"] is None
    assert refined["automatic_eligible"] is False
    diagnostics = projected["temporal_refinement"]["diagnostics"]  # type: ignore[index]
    assert diagnostics["missing_result_count"] == 1


def test_apply_rejects_unknown_request_and_source_relative_result_clock() -> None:
    coarse = _coarse_report()
    plan = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    onset = next(row for row in plan["requests"] if row["role"] == "onset")  # type: ignore[index]
    with pytest.raises(ProductionWemmTemporalRefinementError, match="unknown request_id"):
        apply_refined_boundaries(
            coarse,
            plan,
            {"unknown": {"status": "UNCERTAIN"}},
        )
    with pytest.raises(ProductionWemmTemporalRefinementError, match="request_relative_seconds"):
        apply_refined_boundaries(
            coarse,
            plan,
            {
                onset["request_id"]: {
                    "status": "MEASURED",
                    "timestamp_basis": "source_relative_seconds",
                    "start_seconds": 2.1,
                    "end_seconds": 2.4,
                }
            },
        )
    with pytest.raises(
        ProductionWemmTemporalRefinementError, match="declare request_relative_seconds"
    ):
        apply_refined_boundaries(
            coarse,
            plan,
            {
                onset["request_id"]: {
                    "status": "MEASURED",
                    "start_seconds": 0.1,
                    "end_seconds": 0.4,
                }
            },
        )
