from __future__ import annotations

import json

import pytest

from robata.benchmark.wemm_multiview_retrieval import (
    WemmMultiviewRetrievalError,
    fuse_camera_rankings,
    fuse_multiview_candidates,
    normalize_score,
    normalize_scores,
)


def _candidate(
    action: tuple[int, int], score: float, *, rank: int | None = None
) -> dict[str, object]:
    row: dict[str, object] = {"action_key": list(action), "score": score}
    if rank is not None:
        row["rank"] = rank
    return row


def test_mean_fusion_is_order_independent_and_retains_camera_evidence() -> None:
    report = fuse_camera_rankings(
        {
            "cam_b": [_candidate((1, 2), 0.5), _candidate((2, 3), 0.2)],
            "cam_a": [_candidate((1, 2), 0.9), _candidate((3, 4), 0.1)],
        },
        top_k=2,
    )

    assert report["camera_order"] == ["cam_a", "cam_b"]
    assert [row["action_key"] for row in report["candidates"]] == [[1, 2], [2, 3]]
    first = report["candidates"][0]
    assert first["fused_score"] == pytest.approx(0.7)
    assert first["camera_coverage"] == 2
    assert [item["camera_id"] for item in first["per_camera"]] == ["cam_a", "cam_b"]
    assert first["per_camera"] == first["camera_evidence"]
    assert json.loads(json.dumps(report)) == report


def test_expected_camera_order_reports_missing_views_without_zero_evidence() -> None:
    report = fuse_multiview_candidates(
        {"cam_02": [_candidate((0, 1), 0.8)]},
        expected_cameras=["cam_01", "cam_02", "cam_03"],
        top_k=None,
    )

    assert report["camera_order"] == ["cam_01", "cam_02", "cam_03"]
    assert report["camera_coverage"] == {
        "expected_count": 3,
        "observed_count": 1,
        "fraction": pytest.approx(1 / 3),
        "observed_cameras": ["cam_02"],
        "missing_cameras": ["cam_01", "cam_03"],
    }
    assert report["camera_coverage_fraction"] == pytest.approx(1 / 3)
    assert report["candidates"][0]["fused_score"] == pytest.approx(0.8)


def test_missing_candidate_policy_is_explicit() -> None:
    rankings = {
        "a": [_candidate((1, 1), 0.8)],
        "b": [_candidate((2, 2), 0.6)],
    }
    omitted = fuse_camera_rankings(rankings, top_k=None, missing_score="omit")
    zeroed = fuse_camera_rankings(rankings, top_k=None, missing_score="zero")

    # Omit averages only available evidence; zero includes the other observed
    # camera as an explicit non-supporting observation.
    assert omitted["candidates"][0]["fused_score"] == pytest.approx(0.8)
    assert zeroed["candidates"][0]["fused_score"] == pytest.approx(0.4)
    assert zeroed["candidates"][0]["camera_coverage"] == 1


def test_rank_and_reciprocal_rank_fusion_are_deterministic() -> None:
    rankings = {
        "cam_a": [_candidate((1, 1), 0.1), _candidate((0, 0), 0.9)],
        "cam_b": [_candidate((0, 0), 0.1), _candidate((1, 1), 0.9)],
    }
    ranked = fuse_camera_rankings(rankings, fusion="rank", top_k=None)
    reciprocal = fuse_camera_rankings(rankings, fusion="rrf", top_k=None)

    # Both actions have the same aggregate rank and therefore use the stable
    # action-key tie break, independent of input mapping order.
    assert [row["action_key"] for row in ranked["candidates"]] == [[0, 0], [1, 1]]
    assert ranked["candidates"][0]["fused_score"] == pytest.approx(0.5)
    assert reciprocal["fusion"]["method"] == "rrf"
    assert reciprocal["candidates"][0]["fused_score"] == pytest.approx(0.75)


def test_embeddings_can_supply_scores_without_invoking_a_model() -> None:
    report = fuse_camera_rankings(
        {
            "cam_a": {
                "query_embedding": [1.0, 0.0],
                "candidates": [
                    {"action_key": [0, 0], "embedding": [1.0, 0.0]},
                    {"action_key": [0, 1], "embedding": [0.0, 1.0]},
                ],
            }
        },
        top_k=None,
    )

    assert report["candidates"][0]["action_key"] == [0, 0]
    assert report["candidates"][0]["fused_score"] == pytest.approx(1.0)
    assert report["candidates"][0]["per_camera"][0]["score_source"] == "embedding_cosine"
    assert report["candidates"][0]["per_camera"][0]["embedding"] == [1.0, 0.0]
    assert report["model_invoked"] is False
    assert report["gpu_invoked"] is False


def test_existing_retrieval_rows_with_to_dict_are_consumed_without_model_import() -> None:
    class RetrievalRow:
        def to_dict(self) -> dict[str, object]:
            return {
                "rank": 1,
                "action_key": [4, 9],
                "fused_score": 0.73,
                "visual_score": 0.81,
                "visual_cosine": 0.62,
                "text_score": 0.44,
                "label_text": "open door",
            }

    report = fuse_camera_rankings({"cam_a": [RetrievalRow()]}, top_k=None)

    assert report["candidates"] == [
        {
            "rank": 1,
            "action_key": [4, 9],
            "score": pytest.approx(0.73),
            "fused_score": pytest.approx(0.73),
            "camera_coverage": 1,
            "camera_coverage_fraction": 1.0,
            "expected_camera_coverage_fraction": 1.0,
            "per_camera": [
                {
                    "camera_id": "cam_a",
                    "rank": 1,
                    "raw_score": 0.73,
                    "score": 0.73,
                    "normalized_score": 0.73,
                    "score_source": "provided",
                    "visual_score": 0.81,
                    "visual_cosine": 0.62,
                    "text_score": 0.44,
                    "label_text": "open door",
                }
            ],
            "camera_evidence": [
                {
                    "camera_id": "cam_a",
                    "rank": 1,
                    "raw_score": 0.73,
                    "score": 0.73,
                    "normalized_score": 0.73,
                    "score_source": "provided",
                    "visual_score": 0.81,
                    "visual_cosine": 0.62,
                    "text_score": 0.44,
                    "label_text": "open door",
                }
            ],
            "label_text": "open door",
        }
    ]


def test_retrieval_row_projection_errors_are_wrapped_as_contract_errors() -> None:
    class BrokenRow:
        def to_dict(self) -> dict[str, object]:
            raise RuntimeError("encoder output unavailable")

    with pytest.raises(WemmMultiviewRetrievalError, match="to_dict"):
        fuse_camera_rankings({"cam_a": [BrokenRow()]})


def test_score_normalization_invariants_and_scalar_helper() -> None:
    assert normalize_scores([2.0, -1.0, 0.5]) == (1.0, 0.0, 0.5)
    assert normalize_scores([-1.0, 1.0], method="cosine") == (0.0, 1.0)
    assert normalize_scores([3.0, 3.0], method="minmax") == (1.0, 1.0)
    assert normalize_scores(["first", "second"], method="rank") == (1.0, 0.0)
    assert normalize_score(0.25) == pytest.approx(0.25)
    with pytest.raises(WemmMultiviewRetrievalError, match=r"within \[0, 1\]"):
        normalize_score(1.1, method="none")


def test_malformed_inputs_fail_closed() -> None:
    with pytest.raises(WemmMultiviewRetrievalError, match="duplicate action"):
        fuse_camera_rankings({"cam": [_candidate((0, 0), 0.5), _candidate((0, 0), 0.4)]})
    with pytest.raises(WemmMultiviewRetrievalError, match="duplicate camera"):
        fuse_camera_rankings(
            [
                {"camera_id": "cam", "candidates": []},
                {"camera_id": "cam", "candidates": []},
            ]
        )
    with pytest.raises(WemmMultiviewRetrievalError, match="dimensions"):
        fuse_camera_rankings(
            {
                "a": {
                    "query_embedding": [1.0, 0.0],
                    "candidates": [{"action_key": [0, 0], "embedding": [1.0, 0.0]}],
                },
                "b": {
                    "query_embedding": [1.0, 0.0, 0.0],
                    "candidates": [{"action_key": [0, 0], "embedding": [1.0, 0.0, 0.0]}],
                },
            }
        )
