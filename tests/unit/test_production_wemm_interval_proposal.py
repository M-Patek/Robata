from __future__ import annotations

import pytest

from robata.benchmark.production_wemm_interval_proposal import (
    ProductionWemmIntervalProposalError,
    aggregate_temporal_probes,
    parse_temporal_probes,
    propose_model_intervals,
)


def _probe(
    action: str,
    start: float,
    end: float,
    score: float,
    camera: str = "cam_01",
    window_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "action_key": action,
        "camera_id": camera,
        "start_seconds": start,
        "end_seconds": end,
        "score": score,
    }
    if window_id is not None:
        row["window_id"] = window_id
    return row


def test_camera_scores_are_fused_only_on_identical_probe_spans() -> None:
    probes = parse_temporal_probes(
        [
            _probe("open_cupboard", 0.0, 0.5, 0.8, "cam_02"),
            _probe("open_cupboard", 0.0, 0.5, 0.6, "cam_01"),
        ]
    )
    fused = aggregate_temporal_probes(probes)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(0.7)
    assert fused[0].camera_ids == ("cam_01", "cam_02")
    assert fused[0].source_scores == (0.6, 0.8)


def test_duplicate_camera_probe_is_rejected() -> None:
    with pytest.raises(ProductionWemmIntervalProposalError, match="duplicate probe"):
        parse_temporal_probes(
            [
                _probe("open_cupboard", 0.0, 0.5, 0.8),
                _probe("open_cupboard", 0.0, 0.5, 0.7),
            ]
        )


def test_hysteresis_projects_two_intervals_without_using_window_as_boundary() -> None:
    report = propose_model_intervals(
        [
            _probe("open_cupboard", 0.0, 0.5, 0.8),
            # Below activation but above release: keep the active region alive.
            _probe("open_cupboard", 0.5, 1.0, 0.55),
            # Below release: close the first region.
            _probe("open_cupboard", 1.0, 1.5, 0.4),
            _probe("open_cupboard", 1.5, 2.0, 0.75),
        ],
        window_start_seconds=0.0,
        window_end_seconds=4.0,
        merge_gap_seconds=0.1,
        min_duration_seconds=0.2,
    )

    assert report["status"] == "PROPOSALS_ONLY"
    assert report["window"]["context_only"] is True  # type: ignore[index]
    proposals = report["proposals"]
    assert len(proposals) == 2  # type: ignore[arg-type]
    assert proposals[0]["start_seconds"] == pytest.approx(0.0)  # type: ignore[index]
    assert proposals[0]["end_seconds"] == pytest.approx(1.0)  # type: ignore[index]
    assert proposals[1]["start_seconds"] == pytest.approx(1.5)  # type: ignore[index]
    assert proposals[1]["end_seconds"] == pytest.approx(2.0)  # type: ignore[index]
    assert proposals[0]["boundary_status"] == "MODEL_PROBE_BOUND"  # type: ignore[index]
    assert proposals[0]["automatic_eligible"] is False  # type: ignore[index]


def test_minimum_camera_support_filters_single_view_scores() -> None:
    report = propose_model_intervals(
        [_probe("open_cupboard", 0.0, 1.0, 0.9)],
        window_start_seconds=0.0,
        window_end_seconds=2.0,
        min_camera_support=2,
    )
    assert report["proposals"] == []
    assert report["diagnostics"]["action_count"] == 0  # type: ignore[index]


def test_probe_outside_processing_window_is_not_clipped_or_repaired() -> None:
    with pytest.raises(ProductionWemmIntervalProposalError, match="outside window"):
        propose_model_intervals(
            [_probe("open_cupboard", 1.0, 3.0, 0.9)],
            window_start_seconds=0.0,
            window_end_seconds=2.0,
        )


def test_low_scores_do_not_activate_and_controls_remain_nonproduction() -> None:
    report = propose_model_intervals(
        [_probe("open_cupboard", 0.0, 1.0, 0.4)],
        window_start_seconds=0.0,
        window_end_seconds=2.0,
    )
    assert report["proposals"] == []
    controls = report["controls"]
    assert controls["media_decoded"] is False  # type: ignore[index]
    assert controls["model_invoked"] is False  # type: ignore[index]
    assert controls["raw_probe_scores_preserved"] is True  # type: ignore[index]


def test_max_camera_fusion_is_explicit_and_parameters_are_recorded() -> None:
    report = propose_model_intervals(
        [
            _probe("open_cupboard", 0.0, 1.0, 0.9, "cam_01"),
            _probe("open_cupboard", 0.0, 1.0, 0.2, "cam_02"),
        ],
        window_start_seconds=0.0,
        window_end_seconds=2.0,
        camera_fusion="max",
    )
    assert report["parameters"]["camera_fusion"] == "max"  # type: ignore[index]
    assert report["proposals"][0]["peak_score"] == pytest.approx(0.9)  # type: ignore[index]
    assert report["proposals"][0]["boundary_source"] == "wemm_temporal_score"  # type: ignore[index]


def test_hysteresis_threshold_order_is_validated() -> None:
    with pytest.raises(ProductionWemmIntervalProposalError, match="stop_threshold"):
        propose_model_intervals(
            [],
            window_start_seconds=0.0,
            window_end_seconds=1.0,
            start_threshold=0.4,
            stop_threshold=0.5,
        )


def test_negative_probe_start_is_rejected() -> None:
    with pytest.raises(ProductionWemmIntervalProposalError, match="non-negative"):
        parse_temporal_probes([_probe("open_cupboard", -0.1, 0.5, 0.8)])


def test_direct_probe_objects_are_accepted_without_losing_provenance() -> None:
    probe = parse_temporal_probes([_probe("open_cupboard", 0.0, 0.5, 0.8)])[0]
    report = propose_model_intervals([probe], window_start_seconds=0.0, window_end_seconds=1.0)
    assert report["diagnostics"]["input_probe_count"] == 1  # type: ignore[index]


def test_transition_diagnostics_and_window_provenance_are_retained() -> None:
    report = propose_model_intervals(
        [
            _probe("open_cupboard", 0.0, 1.0, 0.20, window_id="w0"),
            _probe("open_cupboard", 1.0, 2.0, 0.80, window_id="w1"),
            _probe("open_cupboard", 2.0, 3.0, 0.70, window_id="w2"),
            _probe("open_cupboard", 3.0, 4.0, 0.30, window_id="w3"),
        ],
        window_start_seconds=0.0,
        window_end_seconds=4.0,
        merge_gap_seconds=0.0,
        min_duration_seconds=0.1,
    )

    proposal = report["proposals"][0]  # type: ignore[index]
    assert proposal["start_seconds"] == pytest.approx(1.0)  # type: ignore[index]
    assert proposal["end_seconds"] == pytest.approx(3.0)  # type: ignore[index]
    assert proposal["boundary_method"] == "observed_probe_span"  # type: ignore[index]
    assert proposal["supporting_window_ids"] == ["w1", "w2"]  # type: ignore[index]
    onset = proposal["transition_diagnostics"]["onset"]  # type: ignore[index]
    offset = proposal["transition_diagnostics"]["offset"]  # type: ignore[index]
    assert onset["crossed_threshold"] is True  # type: ignore[index]
    assert offset["crossed_threshold"] is True  # type: ignore[index]
    assert onset["score_delta"] == pytest.approx(0.6)  # type: ignore[index]
    assert offset["score_delta"] == pytest.approx(0.4)  # type: ignore[index]
    assert proposal["boundary_confidence"] == pytest.approx(0.5)  # type: ignore[index]
    assert report["diagnostics"]["boundary_bracketed_count"] == 1  # type: ignore[index]
    assert report["diagnostics"]["trajectory_count"] == 1  # type: ignore[index]


def test_context_edge_transition_is_explicitly_unbracketed() -> None:
    report = propose_model_intervals(
        [_probe("open_cupboard", 1.0, 2.0, 0.9, window_id="w1")],
        window_start_seconds=0.0,
        window_end_seconds=4.0,
    )
    proposal = report["proposals"][0]  # type: ignore[index]
    onset = proposal["transition_diagnostics"]["onset"]  # type: ignore[index]
    offset = proposal["transition_diagnostics"]["offset"]  # type: ignore[index]
    assert onset["reason"] == "NO_PRECEDING_PROBE"  # type: ignore[index]
    assert offset["reason"] == "NO_FOLLOWING_PROBE"  # type: ignore[index]
    assert proposal["boundary_confidence"] == 0.0  # type: ignore[index]


def test_midpoint_mode_uses_score_crossings_not_context_edges() -> None:
    report = propose_model_intervals(
        [
            _probe("open_cupboard", 0.0, 4.0, 0.20, window_id="w0"),
            _probe("open_cupboard", 1.0, 5.0, 0.80, window_id="w1"),
            _probe("open_cupboard", 2.0, 6.0, 0.70, window_id="w2"),
            _probe("open_cupboard", 3.0, 7.0, 0.30, window_id="w3"),
        ],
        window_start_seconds=0.0,
        window_end_seconds=8.0,
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.1,
    )
    proposal = report["proposals"][0]  # type: ignore[index]
    # The 4 s contexts overlap by 3 s; the estimated interval follows the
    # score crossings at the probe centres instead of [1, 7] context edges.
    assert proposal["start_seconds"] == pytest.approx(2.5)  # type: ignore[index]
    assert proposal["end_seconds"] == pytest.approx(4.5)  # type: ignore[index]
    assert proposal["boundary_method"] == "probe_center_midpoint"  # type: ignore[index]
    assert proposal["transition_diagnostics"]["onset"]["interpolated"] is True  # type: ignore[index]


def test_midpoint_mode_marks_unbracketed_edge_boundaries_explicitly() -> None:
    report = propose_model_intervals(
        [
            _probe("open_cupboard", 0.0, 4.0, 0.80, window_id="w0"),
            _probe("open_cupboard", 1.0, 5.0, 0.80, window_id="w1"),
            _probe("open_cupboard", 2.0, 6.0, 0.20, window_id="w2"),
        ],
        window_start_seconds=0.0,
        window_end_seconds=6.0,
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.1,
    )
    proposal = report["proposals"][0]  # type: ignore[index]
    assert proposal["boundary_method"] == "mixed_probe_boundary"  # type: ignore[index]
    assert proposal["boundary_method_by_side"] == {  # type: ignore[index]
        "onset": "observed_probe_span",
        "offset": "probe_center_midpoint",
    }
    assert proposal["boundary_edge"] == {"onset": True, "offset": False}  # type: ignore[index]
    assert proposal["transition_diagnostics"]["onset"]["reason"] == "NO_PRECEDING_PROBE"  # type: ignore[index]
    assert report["diagnostics"]["probe_grid"]["context_center_latency_seconds"] == pytest.approx(
        2.0
    )  # type: ignore[index]
    assert report["diagnostics"]["probe_grid"][
        "estimated_boundary_resolution_seconds"
    ] == pytest.approx(1.0)  # type: ignore[index]


def test_window_id_must_be_non_empty_when_supplied() -> None:
    with pytest.raises(ProductionWemmIntervalProposalError, match="window_id"):
        parse_temporal_probes([_probe("open_cupboard", 0.0, 1.0, 0.8, window_id=" ")])
