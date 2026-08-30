from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcap")
pytest.importorskip("mcap_protobuf")
pytest.importorskip("av")
pytest.importorskip("PIL")

from robata.benchmark.production_review_surfaces import (
    ProductionReviewSurfacesError,
    build_production_review_surfaces,
    write_production_review_surfaces,
)
from tests.support.six_camera_mcap import SIX_CAMERA_TOPICS, write_six_camera_mcap


def _manifest(source: Path) -> dict[str, object]:
    topics = {f"cam_{index + 1:02d}": topic for index, topic in enumerate(SIX_CAMERA_TOPICS)}
    origin = 1_781_051_907_271_610_000
    return {
        "format": "robata-production-shaped-cohort-v1",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "source": {
            "path": str(source),
            "camera_count": 6,
            "common_start_timestamp_ns": str(origin),
            "cameras": [
                {"camera_id": camera_id, "topic": topic} for camera_id, topic in topics.items()
            ],
        },
        "windows": [
            {
                "ordinal": 0,
                "window_id": "fixture-w00",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "camera_ids": list(topics),
                "camera_topics": topics,
            }
        ],
    }


def test_materializes_six_camera_surface_paths_and_explicit_draft_contract(
    tmp_path: Path,
) -> None:
    source = write_six_camera_mcap(tmp_path / "source.mcap")
    bundle = tmp_path / "surfaces"
    report = build_production_review_surfaces(
        _manifest(source),
        bundle,
        frames_per_camera=2,
        thumbnail_max_side=64,
        max_messages_per_camera=8,
    )

    assert report["format"] == "robata-production-review-surfaces-v1"
    assert report["authority"] == "LOCAL_NONPRODUCTION_ONLY"
    assert report["controls"]["model_invoked"] is False
    assert report["controls"]["machine_assisted_draft_generated"] is False
    assert report["controls"]["sha_or_digest_computed"] is False
    assert report["machine_assisted_draft_contract"]["generated_status"] == (
        "MACHINE_ASSISTED_DRAFT"
    )
    assert report["machine_assisted_draft_contract"]["generated_review_state"] == ("PROVISIONAL")

    window = report["windows"][0]
    assert window["status"] == "READY"
    assert len(window["camera_surfaces"]) == 6
    assert window["window_contact_sheet_path"] == ("surfaces/fixture-w00/six-camera-overview.jpg")
    assert (bundle / "surfaces/fixture-w00/six-camera-overview.jpg").is_file()
    for camera in window["camera_surfaces"]:
        assert camera["status"] == "READY"
        assert len(camera["frames"]) == 2
        contact = str(camera["contact_sheet_path"])
        assert contact.startswith("surfaces/fixture-w00/cam_")
        assert (bundle / Path(*contact.split("/"))).is_file()
        for frame in camera["frames"]:
            frame_path = frame["frame_path"]
            assert isinstance(frame_path, str)
            assert (bundle / Path(*frame_path.split("/"))).is_file()
            assert frame["selected"] is True

    draft = window["machine_assisted_draft"]
    assert draft["status"] == "NOT_GENERATED"
    assert draft["provisional_status_if_generated"] == "MACHINE_ASSISTED_DRAFT"
    assert draft["provisional_review_state_if_generated"] == "PROVISIONAL"
    assert draft["accepted_as_gold"] is False
    assert draft["segments"] == []

    output = tmp_path / "report.json"
    write_production_review_surfaces(report, output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == report


def test_rejects_partial_camera_manifest_before_source_access(tmp_path: Path) -> None:
    source = tmp_path / "missing.mcap"
    manifest = _manifest(source)
    manifest["windows"][0]["camera_ids"] = ["cam_01"]  # type: ignore[index]
    with pytest.raises(ProductionReviewSurfacesError, match="camera_ids"):
        build_production_review_surfaces(manifest, tmp_path / "bundle")
