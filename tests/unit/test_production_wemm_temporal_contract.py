from __future__ import annotations

from typing import Any

import pytest

from robata.benchmark.production_wemm_temporal import resolve_wemm_temporal_segments
from robata.benchmark.production_wemm_temporal_refinement import (
    apply_refined_boundaries,
    plan_wemm_temporal_refinement,
)


def _candidate(score: float) -> dict[str, Any]:
    return {
        "provisional_id": "open_cupboard",
        "label_text": "open cupboard",
        "rank": 1,
        "score": score,
        "camera_support": 2,
        "evidence": [{"camera_id": "cam_01"}, {"camera_id": "cam_02"}],
    }


def _window(window_id: str, start: float, end: float, score: float) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "start_seconds": start,
        "end_seconds": end,
        "proposals": [{"top_k": [_candidate(score)]}],
    }


def _coarse_report() -> dict[str, Any]:
    # Deliberately use the historical 4-second WeMM input contexts.  The
    # contract under test is that these remain context envelopes only.
    return resolve_wemm_temporal_segments(
        [
            _window("w0", 0.0, 4.0, 0.20),
            _window("w1", 1.0, 5.0, 0.85),
            _window("w2", 2.0, 6.0, 0.80),
            _window("w3", 3.0, 7.0, 0.20),
        ],
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.1,
    )


def test_four_second_contexts_are_not_emitted_as_action_boundaries() -> None:
    report = _coarse_report()
    segment = report["segments"][0]
    context = report["context_interval"]

    assert context["context_only"] is True
    assert report["diagnostics"]["context_grid"]["context_width_seconds"] == pytest.approx(4.0)
    assert segment["boundary_status"] == "MODEL_PROBE_BOUND"
    assert segment["boundary_source"] == "wemm_temporal_score"
    assert segment["review_required"] is True
    assert segment["automatic_eligible"] is False
    # The interval is localized from score transitions between probe centres,
    # not copied from any 4-second context edge or span.
    assert segment["boundary_method"] == "probe_center_midpoint"
    assert (segment["start_seconds"], segment["end_seconds"]) == pytest.approx((2.5, 4.5))
    context_spans = {
        (row["start_seconds"], row["end_seconds"])
        for row in (
            _window("w0", 0.0, 4.0, 0.20),
            _window("w1", 1.0, 5.0, 0.85),
            _window("w2", 2.0, 6.0, 0.80),
            _window("w3", 3.0, 7.0, 0.20),
        )
    }
    assert (segment["start_seconds"], segment["end_seconds"]) not in context_spans


def test_adaptive_interval_retains_model_derived_review_only_provenance() -> None:
    coarse = _coarse_report()
    plan = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    onset = next(row for row in plan["requests"] if row["role"] == "onset")
    offset = next(row for row in plan["requests"] if row["role"] == "offset")

    projected = apply_refined_boundaries(
        coarse,
        plan,
        {
            onset["request_id"]: {
                "status": "MEASURED",
                "timestamp_basis": "request_relative_seconds",
                "start_seconds": 0.20,
                "end_seconds": 0.40,
                "confidence": 0.90,
                "evidence": {"source": "short_probe"},
            },
            offset["request_id"]: {
                "status": "MEASURED",
                "timestamp_basis": "request_relative_seconds",
                "start_seconds": 0.20,
                "end_seconds": 0.60,
                "confidence": 0.80,
                "evidence": {"source": "short_probe"},
            },
        },
    )

    refinement = projected["temporal_refinement"]
    refined = projected["refined_segments"][0]
    assert refinement["status"] == "REFINEMENT_REVIEW_ONLY"
    assert refinement["production_eligible"] is False
    assert refinement["diagnostics"]["coarse_segments_preserved"] is True
    assert refined["coarse_interval"]["status"] == "MODEL_PROBE_BOUND"
    assert refined["boundary_status"] == "MODEL_REFINED"
    assert refined["boundary_source"] == "wemm_short_refinement"
    assert refined["boundary_method"] == "short_probe_model"
    assert refined["refinement_status"] == "REFINED"
    assert refined["review_required"] is True
    assert refined["automatic_eligible"] is False
    assert refined["decision"] == "pending"
    assert refined["start_seconds"] == pytest.approx(2.20)
    assert refined["end_seconds"] == pytest.approx(4.60)
    # The historical coarse report is additive and remains untouched.
    assert projected["segments"] == coarse["segments"]


def test_no_requests_and_uncertain_results_never_fabricate_boundaries() -> None:
    empty_coarse: dict[str, Any] = {
        "format": "coarse",
        "status": "PROPOSALS_ONLY",
        "context_interval": {"start_seconds": 0.0, "end_seconds": 8.0},
        "segments": [],
    }
    empty_plan = plan_wemm_temporal_refinement(empty_coarse)
    assert empty_plan["requests"] == []
    assert empty_plan["controls"]["runner_recompute_required"] is False
    empty_projection = apply_refined_boundaries(empty_coarse, empty_plan, {})
    assert empty_projection["refined_segments"] == []
    assert empty_projection["temporal_refinement"]["diagnostics"] == {
        "request_count": 0,
        "result_count": 0,
        "missing_result_count": 0,
        "refined_segment_count": 0,
        "partial_segment_count": 0,
        "unresolved_segment_count": 0,
        "invalid_pair_count": 0,
        "request_edge_rejected_result_count": 0,
        "coarse_segments_preserved": True,
    }

    coarse = _coarse_report()
    plan = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    onset = next(row for row in plan["requests"] if row["role"] == "onset")
    uncertain = apply_refined_boundaries(
        coarse,
        plan,
        {
            onset["request_id"]: {
                "status": "UNCERTAIN",
                "confidence": 0.0,
                "evidence": {"reason": "occluded_boundary"},
            }
        },
    )
    refined = uncertain["refined_segments"][0]
    assert refined["refinement_status"] == "UNRESOLVED"
    assert refined["boundary_status"] == "MODEL_REFINEMENT_PENDING"
    assert refined["start_seconds"] is None
    assert refined["end_seconds"] is None
    assert refined["review_required"] is True
    assert refined["automatic_eligible"] is False
    assert uncertain["temporal_refinement"]["diagnostics"]["missing_result_count"] == 1
