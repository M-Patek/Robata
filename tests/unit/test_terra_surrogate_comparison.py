from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.benchmark.terra_surrogate_comparison import (
    TerraSurrogateComparisonError,
    compare_terra_surrogate_reviews,
    render_markdown,
)


def _segment(
    verb: str,
    noun: str,
    start: float,
    end: float,
    *,
    attributes: str | None = None,
    location: str | None = None,
    hand: str | None = None,
) -> dict[str, object]:
    return {
        "verb": verb,
        "noun": noun,
        "attributes": attributes,
        "location": location,
        "hand": hand,
        "start_seconds": start,
        "end_seconds": end,
    }


def _review_pair() -> tuple[dict[str, object], dict[str, object]]:
    confirmed = {
        "format": "robata-terra-confirmed-production-review-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "OWNER_REVIEW_PROVISIONAL_NON_GOLD",
        "official_gold": False,
        "official_gold_status": "NOT_ESTABLISHED",
        "human_adjudication": "NOT_PERFORMED",
        "windows": [
            {
                "window_id": "sample-medium-w00",
                "window_start_seconds": 0.0,
                "window_end_seconds": 4.0,
                "decision": "accept",
                "segments": [_segment("pick_up", "garment", 1.0, 3.0)],
            },
            {
                "window_id": "sample-medium-w01",
                "window_start_seconds": 4.0,
                "window_end_seconds": 8.0,
                "decision": "abstain",
                "segments": [],
                "candidate_interval": {
                    **_segment("adjust", "garment", 4.0, 8.0),
                    "confidence": 0.4,
                },
            },
            {
                "window_id": "sample-medium-w02",
                "window_start_seconds": 8.0,
                "window_end_seconds": 12.0,
                "decision": "split",
                "segments": [
                    _segment("smooth", "garment", 8.0, 9.0),
                    _segment("adjust", "garment", 9.0, 12.0),
                ],
            },
        ],
    }
    independent = {
        "format": "robata-terra-independent-production-review-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "status": "INDEPENDENT_SURROGATE_NON_GOLD",
        "official_gold_status": "NOT_ESTABLISHED",
        "human_adjudication": "NOT_PERFORMED",
        "items": [
            {
                "window_id": "sample-medium-w00",
                "window_start_seconds": 0.0,
                "window_end_seconds": 4.0,
                "recommendation": "EDIT",
                "segments": [
                    _segment(
                        "pick up",
                        "garment",
                        1.0,
                        3.0,
                        location="table",
                        hand="both hands",
                    )
                ],
            },
            {
                "window_id": "sample-medium-w01",
                "window_start_seconds": 4.0,
                "window_end_seconds": 8.0,
                "recommendation": "ABSTAIN",
                "segments": [_segment("adjust", "garment", 4.0, 8.0)],
            },
            {
                "window_id": "sample-medium-w02",
                "window_start_seconds": 8.0,
                "window_end_seconds": 12.0,
                "recommendation": "SPLIT",
                "segments": [
                    _segment("smooth", "garment", 8.0, 9.0),
                    _segment("adjust", "garment", 9.0, 12.0),
                ],
            },
        ],
    }
    return confirmed, independent


def test_terra_comparison_collapses_accept_edit_and_keeps_abstain_split_distinct() -> None:
    confirmed, independent = _review_pair()

    report = compare_terra_surrogate_reviews(confirmed, independent)

    assert report["status"] == "NON_GOLD_SURROGATE_CONSISTENCY"
    assert report["quality_claim"] is False
    assert report["decision_compatibility"] == {
        "common_windows": 3,
        "compatible_count": 3,
        "incompatible_count": 0,
        "compatibility_rate": 1.0,
        "confusion": {
            "RETAINED->RETAINED": 1,
            "ABSTAIN->ABSTAIN": 1,
            "SPLIT->SPLIT": 1,
        },
    }
    labels = report["label_interval_overlap"]
    # The abstained candidate is retained as context in the source artifact,
    # but is intentionally excluded from retained-claim overlap metrics.
    assert labels["core_label_exact_matches"] == 3
    assert labels["exact_interval_matches"] == 3
    assert labels["mean_interval_iou"] == 1.0
    abstained = next(row for row in report["per_window"] if row["window_id"] == "sample-medium-w01")
    assert abstained["confirmed"]["candidate_segment_count"] == 0
    assert abstained["independent"]["candidate_segment_count"] == 1
    assert abstained["segment_pairs"] == []
    assert report["split_agreement"]["both_split_count"] == 1
    assert report["abstain_agreement"]["both_abstain_count"] == 1
    assert any("not precision" in item.lower() for item in report["limitations"])
    assert all(value is False for value in report["controls"].values())

    markdown = render_markdown(report)
    assert "NON_GOLD_SURROGATE_CONSISTENCY" in markdown
    assert "not model accuracy" in markdown


def test_terra_comparison_refuses_explicit_official_gold() -> None:
    confirmed, independent = _review_pair()
    confirmed["official_gold"] = True

    with pytest.raises(TerraSurrogateComparisonError, match="official gold"):
        compare_terra_surrogate_reviews(confirmed, independent)


def test_recorded_terra_artifacts_have_surrogate_consistency_when_present() -> None:
    confirmed_path = Path(".agent_tmp/terra_confirmed_production_review_20260827.json")
    independent_path = Path(".agent_tmp/terra_independent_production_review_4s_16f_20260827.json")
    if not confirmed_path.is_file() or not independent_path.is_file():
        pytest.skip("local Terra artifacts are not part of a clean checkout")

    confirmed = json.loads(confirmed_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    report = compare_terra_surrogate_reviews(confirmed, independent)

    assert report["windows"]["common_count"] == 10
    assert report["decision_compatibility"]["compatible_count"] == 10
    assert report["label_interval_overlap"]["core_label_exact_matches"] == 9
    assert report["label_interval_overlap"]["exact_interval_matches"] == 9
    assert report["split_agreement"]["both_split_count"] == 1
    assert report["abstain_agreement"]["both_abstain_count"] == 2
    assert report["official_gold_status"] == "NOT_ESTABLISHED"
