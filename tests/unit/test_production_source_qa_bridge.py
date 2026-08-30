from __future__ import annotations

import json
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest

from robata.benchmark.production_source_qa_bridge import (
    CAMERA_IDS,
    DEFAULT_CAMERA_TOPICS,
    FrameSample,
    SourceQABridgeError,
    SourceQAPolicy,
    _camera_result,
    _CameraState,
    normalize_camera_topics,
    run_archive_source_qa,
)
from tests.support.six_camera_mcap import SIX_CAMERA_TOPICS, build_six_camera_mcap


def test_policy_requires_explicit_positive_limits() -> None:
    with pytest.raises(SourceQABridgeError, match="positive integers"):
        SourceQAPolicy(max_samples_per_camera=0)
    with pytest.raises(SourceQABridgeError, match="time thresholds"):
        SourceQAPolicy(sample_period_seconds=0.0)
    with pytest.raises(SourceQABridgeError, match="decode failure ratio"):
        SourceQAPolicy(decode_ratio_warning=0.2, decode_ratio_failure=0.3)


def test_normalize_topics_requires_all_six_unique_cameras() -> None:
    assert normalize_camera_topics(DEFAULT_CAMERA_TOPICS) == tuple(DEFAULT_CAMERA_TOPICS.items())
    missing = dict(DEFAULT_CAMERA_TOPICS)
    missing.pop("cam_06")
    with pytest.raises(SourceQABridgeError, match="missing cam_06"):
        normalize_camera_topics(missing)


@pytest.fixture
def scratch_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / ".agent_tmp" / f"source-qa-test-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        root.rmdir()


def test_objective_camera_result_marks_blackout_and_freeze_without_gold() -> None:
    state = _CameraState("cam_01", "/camera/1", expected_frames=3)
    state.source_messages = 3
    state.decoded_frames = 3
    state.samples = [
        FrameSample(0, 1.0, 1.0, 0.0, 0.0, 0.0, None),
        FrameSample(2_000_000_000, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        FrameSample(6_000_000_000, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ]
    state.timestamps = [0, 2_000_000_000, 6_000_000_000]
    result = _camera_result(state, SourceQAPolicy())
    assert result["status"] == "FAIL"
    assert result["checks"]["blackout"]["status"] == "FAIL"
    assert result["checks"]["freeze"]["status"] == "WARNING"


def test_archive_scan_streams_fixture_member_and_keeps_admission_pending(scratch_dir: Path) -> None:
    archive = scratch_dir / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("file/sample_data_mcap/fixture.mcap", build_six_camera_mcap())
        handle.writestr("notes.txt", "not media")

    report = run_archive_source_qa(
        archive,
        camera_topics={
            camera: topic for camera, topic in zip(CAMERA_IDS, SIX_CAMERA_TOPICS, strict=True)
        },
        policy=SourceQAPolicy(
            sample_period_seconds=0.25,
            max_samples_per_camera=4,
            decode_ratio_warning=0.4,
            decode_ratio_failure=0.1,
        ),
        validate_crcs=False,
    )
    assert report["source"]["source_bound"] is True
    assert report["source"]["members_streamed_one_at_a_time"] is True
    assert report["counts"]["members_total"] == 1
    assert report["counts"]["members_scanned"] == 1
    assert report["qa_admission"] == "PENDING_VISUAL_REVIEW"
    assert report["production_eligible"] is False
    assert report["controls"]["content_hash_computed"] is False
    item = report["items"][0]
    assert item["scan"]["camera_count"] == 6
    assert {row["camera_id"] for row in item["scan"]["cameras"]} == set(CAMERA_IDS)


def test_preflight_fail_member_is_excluded_without_opening_it(scratch_dir: Path) -> None:
    archive = scratch_dir / "source.zip"
    member = "file/sample_data_mcap/bad.mcap"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, b"malformed")
    preflight = scratch_dir / "preflight.json"
    preflight.write_text(
        json.dumps({"items": [{"name": member, "ok": False, "error": "malformed"}]}),
        encoding="utf-8",
    )
    report = run_archive_source_qa(archive, preflight_path=preflight)
    assert report["counts"] == {
        "members_total": 1,
        "members_scanned": 0,
        "pass": 0,
        "warning": 0,
        "fail": 1,
    }
    assert report["items"][0]["qa_admission"] == "EXCLUDED_SOURCE_PREFLIGHT_FAIL"
