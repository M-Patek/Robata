from __future__ import annotations

from robata.benchmark.production_identity_boundary_merge import (
    merge_identity_and_boundaries,
)


def _identity_row(action: str, camera: str = "cam_01") -> dict:
    return {
        "window_id": "w00",
        "ordinal": 0,
        "interval": [0.0, 4.0],
        "camera_id": camera,
        "status": "SUCCEEDED",
        "parsed_identity": {
            "parse_status": "PARSED",
            "action": action,
            "confidence": 0.9,
            "evidence": ["hand changes garment position"],
        },
    }


def _boundary_row(camera: str = "cam_01", *, start: float = 0.5, end: float = 1.5) -> dict:
    return {
        "window_id": "w00",
        "camera_id": camera,
        "status": "SUCCEEDED",
        "timestamp_mapping_status": "MAPPED",
        "segments": [
            {
                "start_time_sec": start,
                "end_time_sec": end,
                "structured_labels": {
                    "verb": "folding",
                    "noun": "clothing",
                    "attributes": "blue",
                    "location": "table",
                    "hand": "both hands",
                },
                "confidence": 0.8,
                "evidence": ["edges move inward"],
            }
        ],
    }


def test_merges_identity_wording_and_source_bound_boundary() -> None:
    report = merge_identity_and_boundaries(
        {"windows": [_identity_row("fold garment")]},
        {"windows": [_boundary_row()]},
    )
    top = report["windows"][0]["annotation_candidates"][0]
    assert top["label_text"] == "fold garment"
    assert top["start_seconds"] == 0.5
    assert top["end_seconds"] == 1.5
    assert top["attributes"] == "blue"
    assert top["boundary_status"] == "MEASURED"
    assert report["official_quality_status"] == "NOT_MEASURED"
    assert report["production_eligible"] is False


def test_measured_consensus_metric_requires_two_cameras() -> None:
    """One measured camera remains reviewable but is not a consensus metric."""

    report = merge_identity_and_boundaries(
        {"windows": [_identity_row("fold garment", "cam_01")]},
        {"windows": [_boundary_row("cam_01")]},
    )
    top = report["windows"][0]["annotation_candidates"][0]
    assert top["boundary_status"] == "MEASURED"
    assert top["camera_boundary_support"] == 1
    assert "BOUNDARY_SUPPORT_LT_2_CAMERAS" in top["reason_codes"]
    assert report["metrics"]["windows_with_measured_consensus"] == 0


def test_measured_consensus_metric_counts_two_distinct_cameras() -> None:
    report = merge_identity_and_boundaries(
        {
            "windows": [
                _identity_row("fold garment", "cam_01"),
                _identity_row("fold garment", "cam_02"),
            ]
        },
        {
            "windows": [
                _boundary_row("cam_01"),
                _boundary_row("cam_02", start=0.7, end=1.7),
            ]
        },
    )
    top = report["windows"][0]["annotation_candidates"][0]
    assert top["camera_boundary_support"] == 2
    assert top["boundary_status"] == "MEASURED"
    assert report["metrics"]["windows_with_measured_consensus"] == 1


def test_one_of_two_camera_boundaries_is_not_consensus() -> None:
    report = merge_identity_and_boundaries(
        {
            "windows": [
                _identity_row("fold garment", "cam_01"),
                _identity_row("fold garment", "cam_02"),
            ]
        },
        {"windows": [_boundary_row("cam_01")]},
    )
    top = report["windows"][0]["annotation_candidates"][0]
    assert top["camera_support"] == 2
    assert top["camera_boundary_support"] == 1
    assert top["boundary_status"] == "MEASURED"
    assert report["metrics"]["windows_with_measured_consensus"] == 0


def test_duplicate_rows_from_one_camera_do_not_fake_consensus() -> None:
    report = merge_identity_and_boundaries(
        {
            "windows": [
                _identity_row("fold garment", "cam_01"),
                _identity_row("fold garment", "cam_01"),
            ]
        },
        {"windows": [_boundary_row("cam_01")]},
    )
    top = report["windows"][0]["annotation_candidates"][0]
    assert top["camera_support"] == 2
    assert top["camera_boundary_support"] == 1
    assert report["metrics"]["windows_with_measured_consensus"] == 0


def test_missing_or_unmapped_boundary_stays_null() -> None:
    row = _boundary_row()
    row["timestamp_mapping_status"] = "FAILED"
    report = merge_identity_and_boundaries(
        {"windows": [_identity_row("fold garment")]},
        {"windows": [row]},
    )
    top = report["windows"][0]["annotation_candidates"][0]
    assert top["start_seconds"] is None
    assert top["end_seconds"] is None
    assert top["boundary_status"] == "NOT_MEASURED"
    assert "BOUNDARY_UNRESOLVED" in top["reason_codes"]


def test_non_matching_boundary_is_not_reused() -> None:
    row = _boundary_row()
    row["segments"][0]["structured_labels"]["verb"] = "reaches"
    report = merge_identity_and_boundaries(
        {"windows": [_identity_row("fold garment")]},
        {"windows": [row]},
    )
    top = report["windows"][0]["annotation_candidates"][0]
    assert top["start_seconds"] is None
    assert "BOUNDARY_FOR_IDENTITY_NOT_FOUND" in top["reason_codes"]
