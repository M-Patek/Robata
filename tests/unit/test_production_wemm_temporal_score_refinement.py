from __future__ import annotations

import json

import pytest

from robata.benchmark.production_wemm_temporal import resolve_wemm_temporal_segments
from robata.benchmark.production_wemm_temporal_refinement import (
    apply_refined_boundaries,
    plan_wemm_temporal_refinement,
)
from robata.benchmark.production_wemm_temporal_score_refinement import (
    ProductionWemmTemporalScoreRefinementError,
    plan_wemm_score_refinement_grid,
    resolve_wemm_candidate_relative_score_refinement,
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
            # The offset evidence must cross on a pair that straddles the
            # coarse anchor (4.5), not on a later same-side fluctuation.
            value = 0.80 if center < 4.45 else 0.20
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


def test_normalized_preannotation_raw_candidate_fields_are_used_for_scores() -> None:
    """Fine resolution must understand the persisted envelope shape.

    The pre-annotation normalizer keeps the opaque action ID and fused camera
    coverage inside ``candidate.raw``.  Reading only the flattened fields
    would silently turn every probe into a zero-support miss.
    """

    coarse = _coarse()
    parent = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    fine = plan_wemm_score_refinement_grid(
        coarse,
        parent_plan=parent,
        probe_span_seconds=0.5,
        points_per_side=2,
        levels=2,
    )
    rows: list[dict[str, object]] = []
    for request in fine["requests"]:  # type: ignore[index]
        center = (request["start_seconds"] + request["end_seconds"]) / 2.0  # type: ignore[operator]
        role = request["role"]
        if role == "onset":
            score = 0.85 if center >= 2.45 else 0.20
        else:
            score = 0.80 if center < 4.45 else 0.20
        rows.append(
            {
                "window_id": f"temporal-refinement::{request['request_id']}",
                "proposals": [
                    {
                        "top_k": [
                            {
                                "rank": 1,
                                "score": score,
                                "raw": {
                                    "action_key": "open_cupboard",
                                    "camera_coverage": 2,
                                },
                            }
                        ]
                    }
                ],
            }
        )

    result = resolve_wemm_score_refinement(
        parent,
        fine,
        rows,
        start_threshold=0.65,
        stop_threshold=0.50,
        score_policy="absolute",
    )
    assert result["diagnostics"]["fine_score_row_count"] > 0  # type: ignore[index]
    assert result["diagnostics"]["measured_result_count"] == 2  # type: ignore[index]


def test_none_camera_support_falls_back_to_numeric_support_count() -> None:
    """A null producer field must not hide a retained support count."""

    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    rows = []
    for request in fine["requests"]:  # type: ignore[index]
        score = 0.20 if str(request["request_id"]).endswith("-before") else 0.90  # type: ignore[index]
        rows.append(
            {
                "window_id": f"temporal-refinement::{request['request_id']}",
                "top_k": [
                    {
                        "provisional_id": "open_cupboard",
                        "score": score,
                        "rank": 1,
                        "camera_support": None,
                        "camera_support_count": 2,
                    }
                ],
            }
        )

    result = resolve_wemm_score_refinement(
        parent,
        fine,
        rows,
        start_threshold=0.65,
        stop_threshold=0.50,
        score_policy="absolute",
    )

    assert result["diagnostics"]["fine_score_row_count"] == 2  # type: ignore[index]
    assert result["diagnostics"]["measured_result_count"] == 1  # type: ignore[index]
    assert result["results"][0]["status"] == "MEASURED"  # type: ignore[index]


def test_compact_scalar_score_honors_top1_rank_gate() -> None:
    """Compact keyed rows must apply the same top1 policy as candidate rows."""

    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    scores = {
        request["request_id"]: {
            "score": 0.20 if str(request["request_id"]).endswith("-before") else 0.90,
            "rank": 2,
        }
        for request in fine["requests"]  # type: ignore[index]
    }

    result = resolve_wemm_score_refinement(
        parent,
        fine,
        scores,
        start_threshold=0.65,
        stop_threshold=0.50,
        score_policy="top1",
    )

    assert result["diagnostics"]["fine_score_row_count"] == 2  # type: ignore[index]
    assert result["diagnostics"]["measured_result_count"] == 0  # type: ignore[index]
    assert result["results"][0]["status"] == "UNCERTAIN"  # type: ignore[index]
    assert result["results"][0]["evidence"]["reason"] == "NO_FINE_SCORE_THRESHOLD_CROSSING"  # type: ignore[index]


def test_normalized_envelope_prefers_raw_lower_ranked_target_and_keeps_padding_provenance() -> None:
    """Raw fused IDs must survive the normalized envelope projection.

    The review envelope intentionally strips ``provisional_id`` from each
    normalized ``top_k`` row.  A target that is rank two therefore cannot be
    recovered from that projection alone; the retained raw sidecar is the
    authoritative candidate view.  Padding metadata must still be carried
    through from the same raw rows.
    """

    from robata.benchmark.production_wemm_preannotation import build_preannotation_envelope

    coarse = _coarse()
    parent = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    fine = plan_wemm_score_refinement_grid(
        coarse,
        parent_plan=parent,
        probe_span_seconds=0.5,
        points_per_side=2,
        levels=2,
    )
    normalized_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for request in fine["requests"]:  # type: ignore[index]
        center = (request["start_seconds"] + request["end_seconds"]) / 2.0  # type: ignore[operator]
        role = request["role"]
        if role == "onset":
            score = 0.85 if center >= 2.45 else 0.20
        else:
            score = 0.80 if center < 4.45 else 0.20
        window_id = f"temporal-refinement::{request['request_id']}"
        # The normalized projection has labels/scores but no opaque IDs.
        normalized_rows.append(
            {
                "window_id": window_id,
                "proposals": [
                    {
                        "top_k": [
                            {"rank": 1, "score": 0.90, "label_text": "open drawer"},
                            {"rank": 2, "score": score, "label_text": "open cupboard"},
                        ]
                    }
                ],
            }
        )
        # The raw fused sidecar retains the IDs and camera coverage.  The
        # target is deliberately lower-ranked in every probe.
        raw_rows.append(
            {
                "window_id": window_id,
                "fused": {
                    "candidates": [
                        {
                            "action_key": "open_drawer",
                            "fused_score": 0.90,
                            "rank": 1,
                            "camera_coverage": 2,
                        },
                        {
                            "action_key": "open_cupboard",
                            "fused_score": score,
                            "rank": 2,
                            "camera_coverage": 2,
                        },
                    ]
                },
                "input_observations": [
                    {
                        "camera_id": "cam_01",
                        "frame_padding_used": False,
                        "frame_padding_indices": [],
                    }
                ],
            }
        )

    envelope = build_preannotation_envelope(
        {"path": "fine.mcap"},
        normalized_rows,
        raw_model_output={"windows": raw_rows},
    )
    normalized_candidate = envelope["windows"][0]["proposals"][0]["top_k"][1]  # type: ignore[index]
    assert "provisional_id" not in normalized_candidate
    raw_candidate = envelope["raw_model_output"]["windows"][0]["fused"]["candidates"][1]  # type: ignore[index]
    assert raw_candidate["action_key"] == "open_cupboard"

    result = resolve_wemm_score_refinement(
        parent,
        fine,
        envelope,
        start_threshold=0.65,
        stop_threshold=0.50,
        score_policy="absolute",
    )
    assert result["diagnostics"]["fine_score_row_count"] == len(fine["requests"])  # type: ignore[index]
    assert result["diagnostics"]["measured_result_count"] == 2  # type: ignore[index]
    rows = result["results"]  # type: ignore[index]
    for row in rows:
        assert row["status"] == "MEASURED"
        assert row["evidence"]["left_probe"]["padding_provenance_available"] is True
        assert row["evidence"]["left_probe"]["padding_used"] is False
        assert row["evidence"]["right_probe"]["padding_provenance_available"] is True
        assert row["evidence"]["right_probe"]["padding_used"] is False


def test_same_side_score_crossing_is_uncertain_and_keeps_diagnostic() -> None:
    """A threshold crossing away from the coarse anchor is not a boundary."""

    coarse = _coarse()
    parent = plan_wemm_temporal_refinement(coarse, refinement_span_seconds=1.0)
    fine = plan_wemm_score_refinement_grid(
        coarse,
        parent_plan=parent,
        probe_span_seconds=0.5,
        points_per_side=2,
        levels=2,
    )

    # Both trajectories transition after the corresponding anchor.  The old
    # resolver accepted these as a fallback; strict anchor bracketing must
    # leave both parent requests unresolved and explain why.
    scores: dict[str, dict[str, float]] = {}
    for request in fine["requests"]:  # type: ignore[index]
        center = (request["start_seconds"] + request["end_seconds"]) / 2.0  # type: ignore[operator]
        role = request["role"]
        if role == "onset":
            value = 0.85 if center >= 2.70 else 0.20
        else:
            value = 0.80 if center < 4.70 else 0.20
        scores[request["request_id"]] = {"score": value}

    result = resolve_wemm_score_refinement(
        parent,
        fine,
        scores,
        start_threshold=0.65,
        stop_threshold=0.50,
        score_policy="absolute",
    )

    assert result["diagnostics"]["measured_result_count"] == 0  # type: ignore[index]
    assert result["diagnostics"]["unresolved_result_count"] == 2  # type: ignore[index]
    for row in result["results"]:  # type: ignore[index]
        assert row["status"] == "UNCERTAIN"
        evidence = row["evidence"]
        assert evidence["reason"] == "NO_ANCHOR_BRACKETED_FINE_SCORE_CROSSING"
        assert evidence["threshold_crossing_count"] >= 1
        assert evidence["same_side_crossing_count"] >= 1
        assert evidence["rejected_same_side_crossings"]


def test_fine_grid_truncation_allocates_round_robin_across_parents() -> None:
    coarse: dict[str, object] = {
        "format": "synthetic-coarse",
        "context_interval": {"start_seconds": 0.0, "end_seconds": 12.0},
        "segments": [
            {
                "segment_id": "s-a",
                "action_key": "action_a",
                "start_seconds": 1.0,
                "end_seconds": 3.0,
                "supporting_window_ids": [],
                "transition_diagnostics": {
                    "onset": {
                        "boundary_seconds": 1.0,
                        "boundary_method": "coarse_segment_edge",
                        "reason": "COARSE_BOUNDARY",
                    },
                    "offset": {
                        "boundary_seconds": 3.0,
                        "boundary_method": "coarse_segment_edge",
                        "reason": "COARSE_BOUNDARY",
                    },
                },
            },
            {
                "segment_id": "s-b",
                "action_key": "action_b",
                "start_seconds": 5.0,
                "end_seconds": 7.0,
                "supporting_window_ids": [],
                "transition_diagnostics": {
                    "onset": {
                        "boundary_seconds": 5.0,
                        "boundary_method": "coarse_segment_edge",
                        "reason": "COARSE_BOUNDARY",
                    },
                    "offset": {
                        "boundary_seconds": 7.0,
                        "boundary_method": "coarse_segment_edge",
                        "reason": "COARSE_BOUNDARY",
                    },
                },
            },
            {
                "segment_id": "s-c",
                "action_key": "action_c",
                "start_seconds": 9.0,
                "end_seconds": 11.0,
                "supporting_window_ids": [],
                "transition_diagnostics": {
                    "onset": {
                        "boundary_seconds": 9.0,
                        "boundary_method": "coarse_segment_edge",
                        "reason": "COARSE_BOUNDARY",
                    },
                    "offset": {
                        "boundary_seconds": 11.0,
                        "boundary_method": "coarse_segment_edge",
                        "reason": "COARSE_BOUNDARY",
                    },
                },
            },
        ],
    }

    fine = plan_wemm_score_refinement_grid(
        coarse,
        probe_span_seconds=0.5,
        points_per_side=1,
        levels=1,
        max_requests=6,
    )

    diagnostics = fine["diagnostics"]
    assert diagnostics["planned_probe_request_count"] == 12  # type: ignore[index]
    assert diagnostics["emitted_probe_request_count"] == 6  # type: ignore[index]
    assert diagnostics["truncated"] is True  # type: ignore[index]
    allocation = diagnostics["parent_probe_request_allocation"]  # type: ignore[index]
    assert len(allocation) == 6  # type: ignore[arg-type]
    assert sum(item["emitted_count"] for item in allocation.values()) == 6  # type: ignore[union-attr]
    assert sum(item["emitted_count"] > 0 for item in allocation.values()) == 6  # type: ignore[union-attr]
    assert len({row["parent_request_id"] for row in fine["requests"]}) == 6  # type: ignore[index]


def test_padded_non_edge_probe_cannot_measure_a_score_boundary() -> None:
    parent = {
        "requests": [
            {
                "request_id": "parent-onset",
                "action_key": "open_cupboard",
                "role": "onset",
                "start_seconds": 1.0,
                "end_seconds": 3.0,
                "coarse_anchor_seconds": 2.0,
            }
        ]
    }
    fine = {
        "source": {"context_interval": {"start_seconds": 0.0, "end_seconds": 4.0}},
        "requests": [
            {
                "request_id": "probe-before",
                "parent_request_id": "parent-onset",
                "action_key": "open_cupboard",
                "role": "onset",
                "start_seconds": 1.0,
                "end_seconds": 1.5,
                "level": 0,
                "edge_clipped": False,
            },
            {
                "request_id": "probe-padded",
                "parent_request_id": "parent-onset",
                "action_key": "open_cupboard",
                "role": "onset",
                "start_seconds": 1.5,
                "end_seconds": 2.0,
                "level": 0,
                "edge_clipped": False,
            },
            {
                "request_id": "probe-after",
                "parent_request_id": "parent-onset",
                "action_key": "open_cupboard",
                "role": "onset",
                "start_seconds": 2.0,
                "end_seconds": 2.5,
                "level": 0,
                "edge_clipped": False,
            },
        ],
    }
    fine_results = {
        "windows": [
            {"window_id": "temporal-refinement::probe-before", "score": 0.20},
            {"window_id": "temporal-refinement::probe-padded", "score": 0.50},
            {"window_id": "temporal-refinement::probe-after", "score": 0.90},
        ],
        "raw_model_output": {
            "windows": [
                {
                    "window_id": "temporal-refinement::probe-before",
                    "input_observations": [
                        {
                            "camera_id": "cam_01",
                            "frame_padding_used": False,
                            "frame_padding_indices": [],
                        }
                    ],
                },
                {
                    "window_id": "temporal-refinement::probe-padded",
                    "input_observations": [
                        {
                            "camera_id": "cam_01",
                            "frame_padding_used": True,
                            "frame_padding_indices": [2],
                        }
                    ],
                },
                {
                    "window_id": "temporal-refinement::probe-after",
                    "input_observations": [
                        {
                            "camera_id": "cam_01",
                            "frame_padding_used": False,
                            "frame_padding_indices": [],
                        }
                    ],
                },
            ]
        },
    }

    result = resolve_wemm_score_refinement(
        parent,
        fine,
        fine_results,
        start_threshold=0.65,
        stop_threshold=0.50,
        score_policy="absolute",
    )

    assert result["diagnostics"]["padded_score_row_count"] == 1  # type: ignore[index]
    assert result["diagnostics"]["padded_crossing_count"] == 1  # type: ignore[index]
    assert result["diagnostics"]["padded_boundary_rejection_count"] == 1  # type: ignore[index]
    assert result["diagnostics"]["measured_result_count"] == 0  # type: ignore[index]
    row = result["results"][0]  # type: ignore[index]
    assert row["status"] == "UNCERTAIN"
    assert row["evidence"]["reason"] == "PADDED_FINE_SCORE_PROBE"
    assert "probe-padded" in row["evidence"]["padded_probe_request_ids"]
    assert row["evidence"]["rejected_padded_crossings"]
    rejected = row["evidence"]["rejected_padded_crossings"][0]
    padded_probe = next(
        probe
        for probe in (rejected["left_probe"], rejected["right_probe"])
        if probe["request_id"] == "probe-padded"
    )
    assert padded_probe["padding_used"] is True
    assert padded_probe["frame_padding_used"] is True
    assert padded_probe["padding_indices"] == [2]
    assert padded_probe["padding_camera_ids"] == ["cam_01"]


def _relative_parent(role: str) -> dict[str, object]:
    return {
        "requests": [
            {
                "request_id": f"parent-{role}",
                "action_key": "open_cupboard",
                "role": role,
                "start_seconds": 1.0,
                "end_seconds": 3.0,
                "coarse_anchor_seconds": 2.0,
            }
        ]
    }


def _relative_fine(role: str) -> dict[str, object]:
    parent_id = f"parent-{role}"
    return {
        "source": {"context_interval": {"start_seconds": 0.0, "end_seconds": 4.0}},
        "requests": [
            {
                "request_id": f"{role}-before",
                "parent_request_id": parent_id,
                "action_key": "open_cupboard",
                "role": role,
                "start_seconds": 1.0,
                "end_seconds": 1.5,
                "level": 0,
                "edge_clipped": False,
            },
            {
                "request_id": f"{role}-after",
                "parent_request_id": parent_id,
                "action_key": "open_cupboard",
                "role": role,
                "start_seconds": 2.0,
                "end_seconds": 2.5,
                "level": 0,
                "edge_clipped": False,
            },
        ],
    }


def test_candidate_relative_margin_crosses_when_absolute_scores_stay_high() -> None:
    """A signed target-vs-neighbour margin can localize a high-score transition."""

    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    fine_results = {
        "windows": [
            {
                "window_id": "temporal-refinement::onset-before",
                "proposals": [
                    {
                        "top_k": [
                            _candidate("open_cupboard", 0.70, rank=2),
                            _candidate("open_drawer", 0.72, rank=1),
                        ]
                    }
                ],
            },
            {
                "window_id": "temporal-refinement::onset-after",
                "proposals": [
                    {
                        "top_k": [
                            _candidate("open_cupboard", 0.74, rank=1),
                            _candidate("open_drawer", 0.70, rank=2),
                        ]
                    }
                ],
            },
        ]
    }

    result = resolve_wemm_candidate_relative_score_refinement(parent, fine, fine_results)

    assert result["score_policy"] == "relative_margin"  # type: ignore[index]
    assert result["effective_score_policy"] == "relative_margin"  # type: ignore[index]
    assert result["diagnostics"]["measured_result_count"] == 1  # type: ignore[index]
    row = result["results"][0]  # type: ignore[index]
    assert row["status"] == "MEASURED"
    evidence = row["evidence"]
    assert evidence["reason"] == "FINE_SCORE_MARGIN_CROSSING"
    assert evidence["target_action_key"] == "open_cupboard"
    assert evidence["competitor_action_key"] == "open_drawer"
    assert evidence["margin_before"] < 0
    assert evidence["margin_after"] > 0
    assert evidence["target_score_before"] == pytest.approx(0.70)
    assert evidence["competitor_score_after"] == pytest.approx(0.70)
    assert result["production_eligible"] is False  # type: ignore[index]
    json.dumps(result)


def test_candidate_relative_margin_reads_raw_ids_from_normalized_envelope() -> None:
    """Relative scoring must merge the raw lower-ranked target as well."""

    from robata.benchmark.production_wemm_preannotation import build_preannotation_envelope

    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    normalized_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    scores = {"onset-before": (0.70, 0.72), "onset-after": (0.74, 0.70)}
    for request in fine["requests"]:  # type: ignore[index]
        request_id = str(request["request_id"])
        target_score, competitor_score = scores[request_id]
        window_id = f"temporal-refinement::{request_id}"
        normalized_rows.append(
            {
                "window_id": window_id,
                "proposals": [
                    {
                        "top_k": [
                            {"rank": 1, "score": competitor_score, "label_text": "open drawer"},
                            {"rank": 2, "score": target_score, "label_text": "open cupboard"},
                        ]
                    }
                ],
            }
        )
        raw_rows.append(
            {
                "window_id": window_id,
                "fused": {
                    "candidates": [
                        {
                            "action_key": "open_drawer",
                            "fused_score": competitor_score,
                            "rank": 1,
                            "camera_coverage": 2,
                        },
                        {
                            "action_key": "open_cupboard",
                            "fused_score": target_score,
                            "rank": 2,
                            "camera_coverage": 2,
                        },
                    ]
                },
                "input_observations": [
                    {
                        "camera_id": "cam_01",
                        "frame_padding_used": False,
                        "frame_padding_indices": [],
                    }
                ],
            }
        )

    envelope = build_preannotation_envelope(
        {"path": "fine.mcap"},
        normalized_rows,
        raw_model_output={"windows": raw_rows},
    )
    result = resolve_wemm_candidate_relative_score_refinement(parent, fine, envelope)

    assert result["diagnostics"]["relative_candidate_row_count"] == 2  # type: ignore[index]
    assert result["diagnostics"]["relative_score_row_count"] == 2  # type: ignore[index]
    assert result["diagnostics"]["measured_result_count"] == 1  # type: ignore[index]
    row = result["results"][0]  # type: ignore[index]
    assert row["status"] == "MEASURED"
    assert row["evidence"]["left_probe"]["target_rank"] == 2
    assert row["evidence"]["left_probe"]["competitor_rank"] == 1
    assert row["evidence"]["left_probe"]["padding_provenance_available"] is True
    assert row["evidence"]["left_probe"]["padding_used"] is False


def test_candidate_relative_margin_supports_offset_direction_and_aliases() -> None:
    parent = _relative_parent("offset")
    fine = _relative_fine("offset")
    fine_results = {
        "windows": [
            {
                "window_id": "temporal-refinement::offset-before",
                "top_k": [
                    _candidate("open_cupboard", 0.74, rank=1),
                    _candidate("open_drawer", 0.70, rank=2),
                ],
            },
            {
                "window_id": "temporal-refinement::offset-after",
                "top_k": [
                    _candidate("open_cupboard", 0.70, rank=2),
                    _candidate("open_drawer", 0.73, rank=1),
                ],
            },
        ]
    }

    for policy in ("relative_margin", "candidate_relative", "contrast", "relative"):
        result = resolve_wemm_score_refinement(
            parent,
            fine,
            fine_results,
            score_policy=policy,
        )
        assert result["diagnostics"]["measured_result_count"] == 1  # type: ignore[index]
        row = result["results"][0]  # type: ignore[index]
        assert row["status"] == "MEASURED"
        assert row["evidence"]["role"] == "offset"
        assert row["evidence"]["margin_before"] > 0
        assert row["evidence"]["margin_after"] < 0


def test_candidate_relative_missing_competitor_is_unknown_not_zero() -> None:
    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    fine_results = {
        "windows": [
            {
                "window_id": "temporal-refinement::onset-before",
                "top_k": [_candidate("open_cupboard", 0.70)],
            },
            {
                "window_id": "temporal-refinement::onset-after",
                "top_k": [_candidate("open_cupboard", 0.74)],
            },
        ]
    }

    result = resolve_wemm_candidate_relative_score_refinement(parent, fine, fine_results)

    assert result["diagnostics"]["measured_result_count"] == 0  # type: ignore[index]
    assert result["diagnostics"]["relative_skipped_missing_competitor_count"] == 2  # type: ignore[index]
    row = result["results"][0]  # type: ignore[index]
    assert row["status"] == "UNCERTAIN"
    assert row["evidence"]["reason"] == "NO_FINE_SCORE_MARGIN_CROSSING"
    assert result["diagnostics"]["relative_competitor_is_not_zero_filled"] is True  # type: ignore[index]


def test_candidate_relative_filters_unsupported_runner_up_before_selection() -> None:
    """A high-scoring, under-supported runner-up must not mask a valid one."""

    parent = _relative_parent("onset")
    fine = _relative_fine("onset")

    def candidate(
        action: str,
        score: float,
        *,
        rank: int,
        support: int,
        camera_ids: list[str],
    ) -> dict[str, object]:
        return {
            "provisional_id": action,
            "score": score,
            "rank": rank,
            "camera_support": support,
            "camera_ids": camera_ids,
        }

    fine_results = {
        "windows": [
            {
                "window_id": "temporal-refinement::onset-before",
                "top_k": [
                    # The unsupported action is the numerical runner-up, but
                    # must be ignored when the support floor is two cameras.
                    candidate(
                        "open_cupboard",
                        0.70,
                        rank=2,
                        support=2,
                        camera_ids=["cam_01", "cam_02"],
                    ),
                    candidate(
                        "open_box",
                        0.99,
                        rank=1,
                        support=1,
                        camera_ids=["cam_01"],
                    ),
                    candidate(
                        "open_drawer",
                        0.72,
                        rank=3,
                        support=2,
                        camera_ids=["cam_01", "cam_02"],
                    ),
                ],
            },
            {
                "window_id": "temporal-refinement::onset-after",
                "top_k": [
                    candidate(
                        "open_cupboard",
                        0.80,
                        rank=1,
                        support=2,
                        camera_ids=["cam_01", "cam_02"],
                    ),
                    candidate(
                        "open_box",
                        0.99,
                        rank=1,
                        support=1,
                        camera_ids=["cam_01"],
                    ),
                    candidate(
                        "open_drawer",
                        0.60,
                        rank=2,
                        support=2,
                        camera_ids=["cam_01", "cam_02"],
                    ),
                ],
            },
        ]
    }

    result = resolve_wemm_candidate_relative_score_refinement(
        parent,
        fine,
        fine_results,
        min_camera_support=2,
    )

    assert result["diagnostics"]["measured_result_count"] == 1  # type: ignore[index]
    evidence = result["results"][0]["evidence"]  # type: ignore[index]
    assert evidence["competitor_action_key"] == "open_drawer"
    assert evidence["competitor_selection_source"] == "runner_up_consensus"
    assert evidence["left_probe"]["competitor_camera_support"] == 2


def test_candidate_relative_target_floor_and_camera_mismatch_remain_unresolved() -> None:
    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    fine_results = {
        "windows": [
            {
                "window_id": "temporal-refinement::onset-before",
                "top_k": [
                    {
                        "provisional_id": "open_cupboard",
                        "score": 0.55,
                        "rank": 1,
                        "camera_ids": ["cam_01"],
                    },
                    {
                        "provisional_id": "open_drawer",
                        "score": 0.50,
                        "rank": 2,
                        "camera_ids": ["cam_01"],
                    },
                ],
            },
            {
                "window_id": "temporal-refinement::onset-after",
                "top_k": [
                    {
                        "provisional_id": "open_cupboard",
                        "score": 0.75,
                        "rank": 1,
                        "camera_ids": ["cam_01"],
                    },
                    {
                        "provisional_id": "open_drawer",
                        "score": 0.60,
                        "rank": 2,
                        "camera_ids": ["cam_02"],
                    },
                ],
            },
        ]
    }

    result = resolve_wemm_candidate_relative_score_refinement(parent, fine, fine_results)

    assert result["diagnostics"]["measured_result_count"] == 0  # type: ignore[index]
    assert result["diagnostics"]["relative_skipped_target_floor_count"] == 1  # type: ignore[index]
    assert result["diagnostics"]["relative_skipped_camera_mismatch_count"] == 1  # type: ignore[index]
    assert result["results"][0]["status"] == "UNCERTAIN"  # type: ignore[index]


def test_candidate_relative_explicit_competitor_is_stable_across_runner_up_switch() -> None:
    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    fine["parents"] = [
        {
            "request_id": "parent-onset",
            "relative_action_key": "open_drawer",
        }
    ]
    fine_results = {
        "windows": [
            {
                "window_id": "temporal-refinement::onset-before",
                "top_k": [
                    _candidate("open_cupboard", 0.70, rank=2),
                    _candidate("open_drawer", 0.72, rank=1),
                    _candidate("open_box", 0.71, rank=3),
                ],
            },
            {
                "window_id": "temporal-refinement::onset-after",
                "top_k": [
                    _candidate("open_cupboard", 0.74, rank=1),
                    _candidate("open_box", 0.73, rank=2),
                    _candidate("open_drawer", 0.60, rank=3),
                ],
            },
        ]
    }

    result = resolve_wemm_candidate_relative_score_refinement(parent, fine, fine_results)

    assert result["diagnostics"]["measured_result_count"] == 1  # type: ignore[index]
    evidence = result["results"][0]["evidence"]  # type: ignore[index]
    assert evidence["competitor_action_key"] == "open_drawer"
    assert evidence["competitor_selection_source"] == "explicit_hint"


def test_relative_margin_scale_is_forwarded_and_recorded() -> None:
    parent = _relative_parent("onset")
    fine = _relative_fine("onset")
    fine_results = {
        "windows": [
            {
                "window_id": "temporal-refinement::onset-before",
                "top_k": [
                    _candidate("open_cupboard", 0.70, rank=2),
                    _candidate("open_drawer", 0.72, rank=1),
                ],
            },
            {
                "window_id": "temporal-refinement::onset-after",
                "top_k": [
                    _candidate("open_cupboard", 0.74, rank=1),
                    _candidate("open_drawer", 0.70, rank=2),
                ],
            },
        ]
    }

    result = resolve_wemm_candidate_relative_score_refinement(
        parent,
        fine,
        fine_results,
        relative_margin_scale=0.04,
    )

    assert result["parameters"]["relative_margin_scale"] == pytest.approx(0.04)  # type: ignore[index]
    assert result["diagnostics"]["relative_margin_scale"] == pytest.approx(0.04)  # type: ignore[index]


@pytest.mark.parametrize(
    "scale",
    [0.0, -0.01, float("nan"), float("inf"), True, "0.02"],
)
def test_relative_margin_scale_must_be_positive_finite_number(scale: object) -> None:
    parent = _relative_parent("onset")
    fine = _relative_fine("onset")

    with pytest.raises(
        ProductionWemmTemporalScoreRefinementError,
        match="relative_margin_scale must be positive and finite",
    ):
        resolve_wemm_score_refinement(
            parent,
            fine,
            {},
            score_policy="relative_margin",
            relative_margin_scale=scale,  # type: ignore[arg-type]
        )


def test_margin_is_not_a_score_policy_alias() -> None:
    parent = _relative_parent("onset")
    fine = _relative_fine("onset")

    with pytest.raises(
        ProductionWemmTemporalScoreRefinementError,
        match="score_policy must be one of",
    ):
        resolve_wemm_score_refinement(parent, fine, {}, score_policy="margin")
