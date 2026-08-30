from __future__ import annotations

from copy import deepcopy

import pytest

from robata.benchmark.production_machine_assisted_review import (
    MachineAssistedReviewError,
    build_machine_assisted_review,
)


def _pack() -> dict[str, object]:
    return {
        "format": "robata-production-human-review-pack-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "items": [
            {
                "window_id": "w00",
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "gold": {"status": "PENDING_HUMAN_REVIEW", "segments": []},
            },
            {
                "window_id": "w01",
                "start_seconds": 8.0,
                "end_seconds": 16.0,
                "gold": {"status": "PENDING_HUMAN_REVIEW", "segments": []},
            },
        ],
    }


def test_draft_keeps_gold_out_and_uses_window_bound_only() -> None:
    pack = _pack()
    before = deepcopy(pack)
    sidecar = {
        "windows": [
            {
                "window_id": "w00",
                "model_outputs": {
                    "wemm": {"predictions": [{"verb": "open", "noun": "drawer", "rank": 1}]},
                    "qwen": {
                        "predictions": [{"verb": "open", "noun": "drawer", "confidence": 0.8}]
                    },
                },
            }
        ]
    }

    result = build_machine_assisted_review(pack, sidecar)

    assert pack == before
    assert "gold" not in result
    item = result["items"][0]
    assert item["status"] == "MACHINE_ASSISTED_DRAFT"
    assert item["draft_type"] == "PROVISIONAL"
    assert item["segments"][0]["verb"] == "open"
    assert item["segments"][0]["noun"] == "drawer"
    assert item["segments"][0]["boundary_status"] == "WINDOW_BOUND_ONLY"
    assert item["agreement"]["distinct_models"] == 2
    assert result["controls"]["gold_written"] is False


def test_missing_predictions_abstain() -> None:
    result = build_machine_assisted_review(_pack())
    assert [item["status"] for item in result["items"]] == ["ABSTAIN", "ABSTAIN"]
    assert all(item["segments"] == [] for item in result["items"])


def test_explicit_action_key_is_supported_but_free_prose_is_not() -> None:
    sidecar = {
        "windows": [
            {
                "window_id": "w00",
                "model_outputs": {
                    "wemm": {
                        "predictions": [
                            {"action_key": "turn / tap", "rank": 1},
                            {"raw_text": "someone turns the tap"},
                        ]
                    }
                },
            }
        ]
    }
    result = build_machine_assisted_review(_pack(), sidecar)
    assert result["items"][0]["segments"][0]["verb"] == "turn"
    assert result["items"][0]["segments"][0]["noun"] == "tap"


def test_gold_fields_in_model_sidecar_are_rejected() -> None:
    sidecar = {
        "windows": [
            {"window_id": "w00", "model_outputs": {"qwen": {"official_reference": "open drawer"}}}
        ]
    }
    with pytest.raises(MachineAssistedReviewError, match=r"gold|annotation"):
        build_machine_assisted_review(_pack(), sidecar)


def test_invalid_window_interval_is_rejected() -> None:
    pack = _pack()
    pack["items"][0]["end_seconds"] = 0.0  # type: ignore[index]
    with pytest.raises(MachineAssistedReviewError, match="greater than"):
        build_machine_assisted_review(pack)


def test_only_one_model_is_high_priority_and_attributes_remain_unobserved() -> None:
    sidecar = {
        "windows": [
            {
                "window_id": "w00",
                "model_outputs": {
                    "mage": {"predictions": [{"verb": "close", "noun": "door", "confidence": 0.5}]}
                },
            }
        ]
    }
    result = build_machine_assisted_review(_pack(), sidecar)
    item = result["items"][0]
    assert item["review_priority"] == "HIGH"
    segment = item["segments"][0]
    assert segment["attributes"] is None
    assert segment["location"] is None
    assert segment["hand"] is None


def test_direct_qwen_structured_sidecar_is_consumed_without_prose_parsing() -> None:
    sidecar = {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "windows": [
            {
                "window_id": "w00",
                "status": "SUCCEEDED",
                "segments": [
                    {
                        "structured_labels": {"verb": "fold", "noun": "garment"},
                        "confidence": 0.8,
                        "evidence": ["visible fold"],
                    }
                ],
            }
        ],
    }
    result = build_machine_assisted_review(_pack(), sidecar)
    item = result["items"][0]
    assert item["segments"][0]["verb"] == "fold"
    assert item["segments"][0]["noun"] == "garment"
    assert item["agreement"]["top_support_models"] == ["qwen"]


def test_direct_mage_structured_sidecar_is_consumed() -> None:
    sidecar = {
        "format": "robata-production-mage-structured-native-shadow-v1",
        "windows": [
            {
                "window_id": "w00",
                "status": "SUCCEEDED",
                "parsed_structured": {
                    "segments": [
                        {
                            "structured_labels": {"verb": "close", "noun": "door"},
                            "confidence": 0.7,
                        }
                    ]
                },
            }
        ],
    }
    result = build_machine_assisted_review(_pack(), sidecar)
    item = result["items"][0]
    assert item["segments"][0]["verb"] == "close"
    assert item["segments"][0]["noun"] == "door"
    assert item["agreement"]["top_support_models"] == ["mage"]


def test_direct_native_rows_merge_all_cameras_for_one_window() -> None:
    sidecar = {
        "format": "robata-production-qwen-structured-native-shadow-v1",
        "windows": [
            {
                "window_id": "w00",
                "camera_id": "cam_01",
                "status": "SUCCEEDED",
                "segments": [{"structured_labels": {"verb": "fold", "noun": "garment"}}],
            },
            {
                "window_id": "w00",
                "camera_id": "cam_02",
                "status": "SUCCEEDED",
                "segments": [{"structured_labels": {"verb": "fold", "noun": "garment"}}],
            },
        ],
    }
    result = build_machine_assisted_review(_pack(), sidecar)
    item = result["items"][0]
    assert item["status"] == "MACHINE_ASSISTED_DRAFT"
    assert item["candidate_votes"][0]["support_count"] == 2
    assert {row["camera_id"] for row in item["segments"][0]["evidence"]} == {
        "cam_01",
        "cam_02",
    }
