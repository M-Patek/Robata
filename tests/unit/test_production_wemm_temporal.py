from __future__ import annotations

import pytest

from robata.benchmark.production_wemm_preannotation import build_preannotation_envelope
from robata.benchmark.production_wemm_temporal import (
    ProductionWemmTemporalError,
    _stabilize_winner_sequence,
    normalize_score_policy,
    resolve_wemm_temporal_segments,
)


def _candidate(
    action: str,
    score: float,
    *,
    rank: int = 1,
    cameras: tuple[str, ...] = ("cam_01", "cam_02"),
) -> dict[str, object]:
    return {
        "provisional_id": action,
        "label_text": action.replace("_", " "),
        "label_variant": "canonical",
        "structured_labels": {},
        "rank": rank,
        "score": score,
        "camera_support": len(cameras),
        "evidence": [{"camera_id": camera, "score": score} for camera in cameras],
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


def test_dense_context_scores_produce_model_estimated_segment_and_keep_contexts() -> None:
    report = resolve_wemm_temporal_segments(
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
    assert report["status"] == "PROPOSALS_ONLY"
    assert report["context_interval"]["context_only"] is True  # type: ignore[index]
    assert len(report["segments"]) == 1  # type: ignore[arg-type]
    segment = report["segments"][0]  # type: ignore[index]
    assert segment["provisional_id"] == "open_cupboard"
    assert segment["boundary_status"] == "MODEL_PROBE_BOUND"
    assert segment["boundary_method"] == "probe_center_midpoint"
    assert segment["start_seconds"] == pytest.approx(2.5)
    assert segment["end_seconds"] == pytest.approx(4.5)
    assert segment["supporting_window_ids"] == ["w1", "w2"]
    assert segment["camera_support"] == ["cam_01", "cam_02"]
    assert segment["top_k"][0]["window_id"] == "w1"  # type: ignore[index]
    assert segment["top_k"][0]["candidates"][0]["provisional_id"] == "open_cupboard"  # type: ignore[index]
    assert segment["review_required"] is True
    assert segment["automatic_eligible"] is False
    trajectory = report["score_trajectories"][0]  # type: ignore[index]
    probe = trajectory["probes"][0]  # type: ignore[index]
    assert probe["camera_id"] == "__fused__"
    assert probe["camera_ids"] == ["cam_01", "cam_02"]
    assert probe["source_camera_ids"] == ["cam_01", "cam_02"]
    assert probe["camera_support_count"] == 2
    assert report["diagnostics"]["context_grid"]["score_reference"] == "context_center"  # type: ignore[index]


def test_missing_top_k_is_a_recorded_release_signal_not_a_new_label() -> None:
    report = resolve_wemm_temporal_segments(
        [
            _window("w0", 0.0, 2.0, [_candidate("open_cupboard", 0.8)]),
            _window("w1", 1.0, 3.0, []),
            _window("w2", 2.0, 4.0, [_candidate("open_cupboard", 0.8)]),
        ],
        boundary_mode="observed_probe",
        merge_gap_seconds=0.0,
    )
    assert report["controls"]["missing_top_k_recorded_as_zero"] is True  # type: ignore[index]
    assert report["diagnostics"]["temporal_probe_count"] == 3  # type: ignore[index]
    # The missing middle ranking row is not silently discarded from the track.
    trajectory = report["score_trajectories"][0]  # type: ignore[index]
    assert [row["score"] for row in trajectory["probes"]] == [0.8, 0.0, 0.8]


def test_camera_support_threshold_filters_weak_context_candidates() -> None:
    report = resolve_wemm_temporal_segments(
        [
            _window("w0", 0.0, 1.0, [_candidate("open_cupboard", 0.9, cameras=("cam_01",))]),
        ],
        min_camera_support=2,
    )
    assert report["segments"] == []


def test_top1_chooses_supported_lower_rank_when_rank1_is_weak() -> None:
    report = resolve_wemm_temporal_segments(
        [
            _window(
                "w0",
                0.0,
                2.0,
                [
                    _candidate("open_drawer", 0.95, rank=1, cameras=("cam_01",)),
                    _candidate("open_cupboard", 0.80, rank=2, cameras=("cam_01", "cam_02")),
                ],
            )
        ],
        min_camera_support=2,
        score_policy="top1",
    )
    assert [segment["provisional_id"] for segment in report["segments"]] == ["open_cupboard"]


def test_numeric_camera_support_is_retained_without_inventing_camera_ids() -> None:
    candidate = _candidate("open_cupboard", 0.9)
    candidate.pop("evidence")
    candidate["camera_support"] = 6
    report = resolve_wemm_temporal_segments(
        [_window("w0", 0.0, 2.0, [candidate])],
        min_camera_support=6,
    )
    trajectory = report["score_trajectories"][0]  # type: ignore[index]
    probe = trajectory["probes"][0]  # type: ignore[index]
    assert probe["source_camera_support_count"] == 6
    assert probe["source_camera_ids"] == []
    segment = report["segments"][0]  # type: ignore[index]
    assert segment["camera_support"] == []
    assert segment["camera_support_count"] == 6
    assert segment["camera_support_ids_complete"] is False


def test_empty_context_candidate_list_is_review_only_and_nonproduction() -> None:
    with pytest.raises(ProductionWemmTemporalError, match="at least one context"):
        resolve_wemm_temporal_segments([])


def test_contexts_without_candidates_return_an_empty_review_sidecar() -> None:
    report = resolve_wemm_temporal_segments(
        [_window("empty", 3.0, 7.0, [])],
    )

    assert report["status"] == "PROPOSALS_ONLY"
    assert report["segments"] == []
    assert report["context_interval"] == {
        "start_seconds": 3.0,
        "end_seconds": 7.0,
        "context_only": True,
        "is_action_boundary": False,
        "action_boundary": False,
    }


def test_top1_score_policy_prevents_flat_similarity_from_opening_every_action() -> None:
    windows = [
        _window(
            "w0",
            0.0,
            4.0,
            [
                _candidate("open_cupboard", 0.78, rank=1),
                _candidate("fold_garment", 0.77, rank=2),
            ],
        ),
        _window(
            "w1",
            1.0,
            5.0,
            [
                _candidate("fold_garment", 0.78, rank=1),
                _candidate("open_cupboard", 0.77, rank=2),
            ],
        ),
    ]

    top1 = resolve_wemm_temporal_segments(
        windows,
        score_policy="top1",
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
    )
    assert top1["parameters"]["score_policy"] == "top1"  # type: ignore[index]
    assert {segment["provisional_id"] for segment in top1["segments"]} == {  # type: ignore[index]
        "open_cupboard",
        "fold_garment",
    }
    trajectories = {
        row["action_key"]: row
        for row in top1["score_trajectories"]  # type: ignore[index]
    }
    assert [probe["score"] for probe in trajectories["open_cupboard"]["probes"]] == [
        pytest.approx(0.78),
        pytest.approx(0.0),
    ]
    assert [probe["raw_score"] for probe in trajectories["open_cupboard"]["probes"]] == [
        pytest.approx(0.78),
        pytest.approx(0.77),
    ]

    absolute = resolve_wemm_temporal_segments(
        windows,
        score_policy="absolute",
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
    )
    assert absolute["parameters"]["score_policy"] == "absolute"  # type: ignore[index]
    assert len(absolute["segments"]) == 2  # type: ignore[arg-type]
    assert all(
        segment["start_seconds"] == pytest.approx(0.0)
        and segment["end_seconds"] == pytest.approx(5.0)
        for segment in absolute["segments"]  # type: ignore[index]
    )


def test_ranking_switch_suppression_is_opt_in_and_keeps_legacy_top1() -> None:
    """A rank flip alone must not become an adaptive boundary.

    The default dense/top1 resolver remains backwards compatible, while the
    adaptive guard omits both fake fragments and keeps detached unresolved
    diagnostics instead of emitting ``MODEL_PROBE_BOUND`` rows.
    """

    windows = [
        _window(
            "w0",
            0.0,
            4.0,
            [
                _candidate("action_a", 0.80, rank=1),
                _candidate("action_b", 0.79, rank=2),
            ],
        ),
        _window(
            "w1",
            1.0,
            5.0,
            [
                _candidate("action_b", 0.80, rank=1),
                _candidate("action_a", 0.79, rank=2),
            ],
        ),
        _window(
            "w2",
            2.0,
            6.0,
            [
                _candidate("action_b", 0.80, rank=1),
                _candidate("action_a", 0.79, rank=2),
            ],
        ),
    ]

    legacy = resolve_wemm_temporal_segments(
        windows,
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
    )
    assert [segment["provisional_id"] for segment in legacy["segments"]] == [
        "action_a",
        "action_b",
    ]

    adaptive_guarded = resolve_wemm_temporal_segments(
        windows,
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        suppress_ranking_switch_boundaries=True,
    )
    assert adaptive_guarded["segments"] == []
    diagnostics = adaptive_guarded["diagnostics"]
    assert diagnostics["ranking_switch_suppression_enabled"] is True
    assert diagnostics["ranking_switch_unresolved_count"] == 2
    unresolved = diagnostics["ranking_switch_unresolved_segments"]
    assert {row["action_key"] for row in unresolved} == {"action_a", "action_b"}
    assert all(row["boundary_status"] == "UNRESOLVED" for row in unresolved)
    assert all("MODEL_PROBE_BOUND" not in str(row) for row in unresolved)


def test_ranking_switch_guard_keeps_a_real_raw_score_release() -> None:
    """A raw score crossing remains a normal proposal under the guard."""

    report = resolve_wemm_temporal_segments(
        [
            _window(
                "w0",
                0.0,
                4.0,
                [
                    _candidate("action_a", 0.80, rank=1),
                    _candidate("action_b", 0.79, rank=2),
                ],
            ),
            _window(
                "w1",
                1.0,
                5.0,
                [
                    _candidate("action_b", 0.80, rank=1),
                    _candidate("action_a", 0.20, rank=2),
                ],
            ),
        ],
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        suppress_ranking_switch_boundaries=True,
    )
    # The action whose raw evidence crossed below stop_threshold is not
    # suppressed as a ranking-only switch.
    assert [segment["provisional_id"] for segment in report["segments"]] == ["action_a"]
    assert report["diagnostics"]["ranking_switch_unresolved_count"] == 1
    assert report["diagnostics"]["ranking_switch_unresolved_segments"][0]["action_key"] == (
        "action_b"
    )


def test_ranking_switch_guard_is_inactive_for_absolute_policy() -> None:
    windows = [
        _window(
            "w0",
            0.0,
            4.0,
            [_candidate("action_a", 0.80, rank=1), _candidate("action_b", 0.79, rank=2)],
        ),
        _window(
            "w1",
            1.0,
            5.0,
            [_candidate("action_b", 0.80, rank=1), _candidate("action_a", 0.79, rank=2)],
        ),
    ]
    report = resolve_wemm_temporal_segments(
        windows,
        score_policy="absolute",
        suppress_ranking_switch_boundaries=True,
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
    )
    assert report["diagnostics"]["ranking_switch_suppression_enabled"] is True
    assert report["diagnostics"]["ranking_switch_suppression_active"] is False
    assert report["diagnostics"]["ranking_switch_unresolved_count"] == 0
    assert report["segments"]


def test_winner_stable_repairs_only_an_interior_singleton_run() -> None:
    assert _stabilize_winner_sequence(("a", "b", "a")) == ("a", "a", "a")
    assert _stabilize_winner_sequence(("a", None, "a")) == ("a", "a", "a")
    # The pass is non-cascading and deliberately leaves two-context runs and
    # sequence edges untouched.
    assert _stabilize_winner_sequence(("a", "b", "b", "a")) == ("a", "b", "b", "a")
    assert _stabilize_winner_sequence(("a", "b")) == ("a", "b")


def test_winner_stable_joins_an_alternating_winner_without_camera_fabrication() -> None:
    """A,B,A should not create three action fragments when A is a one-row miss."""

    windows = [
        _window("w0", 0.0, 2.0, [_candidate("open_cupboard", 0.90, rank=1)]),
        _window("w1", 1.0, 3.0, [_candidate("open_drawer", 0.95, rank=1)]),
        _window("w2", 2.0, 4.0, [_candidate("open_cupboard", 0.85, rank=1)]),
    ]

    report = resolve_wemm_temporal_segments(
        windows,
        score_policy="winner_stable",
        boundary_mode="observed_probe",
        merge_gap_seconds=0.0,
    )

    assert [segment["provisional_id"] for segment in report["segments"]] == ["open_cupboard"]  # type: ignore[index]
    assert report["diagnostics"]["raw_winner_sequence"] == [  # type: ignore[index]
        "open_cupboard",
        "open_drawer",
        "open_cupboard",
    ]
    assert report["diagnostics"]["stabilized_winner_sequence"] == [  # type: ignore[index]
        "open_cupboard",
        "open_cupboard",
        "open_cupboard",
    ]
    assert report["diagnostics"]["winner_stabilization_count"] == 1  # type: ignore[index]
    assert report["diagnostics"]["score_carry_count"] == 1  # type: ignore[index]
    assert report["diagnostics"]["winner_only_context_support"] is True  # type: ignore[index]

    trajectory = next(
        row
        for row in report["score_trajectories"]  # type: ignore[index]
        if row["action_key"] == "open_cupboard"
    )
    middle = trajectory["probes"][1]
    assert middle["score"] == pytest.approx(0.85)
    assert middle["score_carried"] is True
    assert middle["score_imputed"] is True
    assert middle["source_camera_ids"] == []

    segment = report["segments"][0]  # type: ignore[index]
    carried_evidence = segment["evidence"][1]
    assert carried_evidence["window_id"] == "w1"
    assert carried_evidence["score_carried"] is True
    assert carried_evidence["camera_support"] == []
    assert carried_evidence["top_k_candidate"] is None


def test_unknown_temporal_score_policy_is_rejected() -> None:
    with pytest.raises(ProductionWemmTemporalError, match="score_policy"):
        resolve_wemm_temporal_segments(
            [_window("w0", 0.0, 1.0, [_candidate("open_cupboard", 0.8)])],
            score_policy="margin",
        )


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [
        ("raw", "absolute"),
        ("winner", "top1"),
        ("stable", "winner_stable"),
        ("winner-stability", "winner_stable"),
        ("candidate-relative", "relative_margin"),
        (" relative ", "relative_margin"),
        ("CONTRAST", "relative_margin"),
    ],
)
def test_temporal_score_policy_aliases_normalize_to_canonical(
    requested: str,
    canonical: str,
) -> None:
    assert normalize_score_policy(requested) == canonical


def test_normalized_preannotation_candidates_retain_temporal_identity_in_raw() -> None:
    """Temporal resolution also accepts the normalized envelope shape.

    The pre-annotation contract keeps opaque action identity in each
    candidate's ``raw`` sidecar.  Calling the resolver on that persisted
    shape must not lose the candidate or silently produce an empty track.
    """

    raw_windows = [
        {
            "window_id": "w0",
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "camera_ids": ["cam_01"],
            "proposals": [
                {
                    "label_text": "open cupboard",
                    "structured_labels": {},
                    "top_k": [_candidate("open_cupboard", 0.2, cameras=("cam_01",))],
                }
            ],
        },
        {
            "window_id": "w1",
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "camera_ids": ["cam_01"],
            "proposals": [
                {
                    "label_text": "open cupboard",
                    "structured_labels": {},
                    "top_k": [_candidate("open_cupboard", 0.9, cameras=("cam_01",))],
                }
            ],
        },
        {
            "window_id": "w2",
            "start_seconds": 2.0,
            "end_seconds": 4.0,
            "camera_ids": ["cam_01"],
            "proposals": [
                {
                    "label_text": "open cupboard",
                    "structured_labels": {},
                    "top_k": [_candidate("open_cupboard", 0.2, cameras=("cam_01",))],
                }
            ],
        },
    ]
    envelope = build_preannotation_envelope({"path": "clip.mcap"}, raw_windows)
    report = resolve_wemm_temporal_segments(
        envelope["windows"],
        boundary_mode="observed_probe",
        merge_gap_seconds=0.0,
    )
    assert len(report["segments"]) == 1
    segment = report["segments"][0]
    assert segment["provisional_id"] == "open_cupboard"
    assert segment["camera_support"] == ["cam_01"]


def test_relative_margin_rank_switch_is_unresolved_under_adaptive_guard() -> None:
    """A competitor reorder cannot certify an action boundary."""

    report = resolve_wemm_temporal_segments(
        [
            _window(
                "w0",
                0.0,
                1.0,
                [_candidate("action_a", 0.80, rank=1), _candidate("action_b", 0.70, rank=2)],
            ),
            _window(
                "w1",
                1.0,
                2.0,
                [_candidate("action_a", 0.80, rank=2), _candidate("action_b", 0.90, rank=1)],
            ),
            _window(
                "w2",
                2.0,
                3.0,
                [_candidate("action_a", 0.80, rank=2), _candidate("action_b", 0.90, rank=1)],
            ),
        ],
        score_policy="relative_margin",
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.01,
        suppress_ranking_switch_boundaries=True,
    )

    assert report["parameters"]["ranking_switch_suppression_active"] is True  # type: ignore[index]
    assert report["segments"] == []
    diagnostics = report["diagnostics"]
    assert diagnostics["ranking_switch_unresolved_count"] == 2  # type: ignore[index]
    unresolved = diagnostics["ranking_switch_unresolved_segments"]  # type: ignore[index]
    assert {row["action_key"] for row in unresolved} == {"action_a", "action_b"}
    assert all(row["reason"] == "RANKING_SWITCH_ONLY" for row in unresolved)
    assert all(row["boundary_status"] == "UNRESOLVED" for row in unresolved)


def test_relative_margin_unknown_camera_ids_do_not_enable_switch_suppression() -> None:
    """Numeric support without IDs remains reviewable but cannot certify a switch."""

    def unknown_camera_candidate(action: str, score: float, rank: int) -> dict[str, object]:
        candidate = _candidate(action, score, rank=rank)
        candidate.pop("evidence")
        candidate["camera_support"] = 2
        return candidate

    report = resolve_wemm_temporal_segments(
        [
            _window(
                "w0",
                0.0,
                1.0,
                [
                    unknown_camera_candidate("action_a", 0.80, 1),
                    unknown_camera_candidate("action_b", 0.70, 2),
                ],
            ),
            _window(
                "w1",
                1.0,
                2.0,
                [
                    unknown_camera_candidate("action_a", 0.80, 2),
                    unknown_camera_candidate("action_b", 0.90, 1),
                ],
            ),
            _window(
                "w2",
                2.0,
                3.0,
                [
                    unknown_camera_candidate("action_a", 0.80, 2),
                    unknown_camera_candidate("action_b", 0.90, 1),
                ],
            ),
        ],
        score_policy="relative_margin",
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.01,
        suppress_ranking_switch_boundaries=True,
    )

    # The relative stream may still be emitted as review evidence, but the
    # adaptive guard must not call an unknown-camera transition ranking-only.
    assert report["segments"]
    assert report["diagnostics"]["ranking_switch_unresolved_count"] == 0
    for trajectory in report["score_trajectories"]:
        for probe in trajectory["probes"]:
            assert probe["relative_margin_camera_provenance_known"] is False
            assert probe["relative_margin_camera_sets_known"] is False
            assert probe["runner_up_camera_ids"] == []


def test_relative_margin_keeps_true_raw_target_crossing_measured() -> None:
    """A raw target floor crossing remains a normal review proposal."""

    report = resolve_wemm_temporal_segments(
        [
            _window(
                "w0",
                0.0,
                1.0,
                [_candidate("action_a", 0.20, rank=2), _candidate("action_b", 0.70, rank=1)],
            ),
            _window(
                "w1",
                1.0,
                2.0,
                [_candidate("action_a", 0.80, rank=1), _candidate("action_b", 0.70, rank=2)],
            ),
            _window(
                "w2",
                2.0,
                3.0,
                [_candidate("action_a", 0.80, rank=1), _candidate("action_b", 0.70, rank=2)],
            ),
        ],
        score_policy="relative_margin",
        boundary_mode="midpoint",
        merge_gap_seconds=0.0,
        min_duration_seconds=0.01,
        suppress_ranking_switch_boundaries=True,
    )

    assert [segment["provisional_id"] for segment in report["segments"]] == ["action_a"]
    assert report["segments"][0]["boundary_status"] == "MODEL_PROBE_BOUND"  # type: ignore[index]
    unresolved = report["diagnostics"]["ranking_switch_unresolved_segments"]  # type: ignore[index]
    assert [(row["action_key"], row["unsupported_sides"]) for row in unresolved] == [
        ("action_b", ["offset"])
    ]
