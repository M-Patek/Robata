from __future__ import annotations

import pytest

from robata.benchmark.production_wemm_temporal import (
    ProductionWemmTemporalError,
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


def test_empty_context_candidate_list_is_review_only_and_nonproduction() -> None:
    with pytest.raises(ProductionWemmTemporalError, match="at least one context"):
        resolve_wemm_temporal_segments([])


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


def test_unknown_temporal_score_policy_is_rejected() -> None:
    with pytest.raises(ProductionWemmTemporalError, match="score_policy"):
        resolve_wemm_temporal_segments(
            [_window("w0", 0.0, 1.0, [_candidate("open_cupboard", 0.8)])],
            score_policy="margin",
        )
