from __future__ import annotations

from robata.benchmark.production_review_pack import build_review_pack


def _manifest() -> dict[str, object]:
    return {
        "format": "robata-production-shaped-cohort-v1",
        "source": {"path": "sample.mcap", "camera_count": 6},
        "windows": [
            {
                "ordinal": 0,
                "window_id": "sample-w00",
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "camera_ids": ["cam_01", "cam_02"],
            }
        ],
    }


def test_review_pack_has_blank_gold_and_separate_model_slots() -> None:
    payload = build_review_pack(_manifest())
    item = payload["items"][0]
    assert item["gold"]["segments"] == []
    assert item["gold"]["status"] == "PENDING_HUMAN_REVIEW"
    assert item["model_outputs"]["wemm"]["status"] == "NOT_RUN"
    assert payload["controls"]["labels_inferred"] is False
    assert payload["controls"]["model_predictions_copied"] is False


def test_review_pack_contract_contains_required_annotation_fields() -> None:
    payload = build_review_pack(_manifest())
    assert payload["review_contract"]["structured_label_fields"] == [
        "verb",
        "noun",
        "attributes",
        "location",
        "hand",
    ]
    assert payload["review_contract"]["observable_only"] is True
