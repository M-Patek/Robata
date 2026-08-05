from __future__ import annotations

import json
from pathlib import Path

import pytest

from robata.benchmark.local_real_model_e2e import (
    LocalRealModelE2EError,
    _file_uri_path,
    _load_mapping,
    _mapping_topics,
    _normalize_loopback_endpoint_url,
    _parse_json_output,
    _structured_output_shape_valid,
)


def _mapping(tmp_path: Path, *, approved: bool) -> Path:
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {
                "profile_id": "local-six-camera",
                "approval_status": "APPROVED" if approved else "UNAPPROVED",
                "approved": approved,
                "topics": {f"cam_{index:02d}": f"/camera/{index}" for index in range(1, 7)},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_unapproved_mapping_requires_explicit_local_override(tmp_path: Path) -> None:
    path = _mapping(tmp_path, approved=False)

    with pytest.raises(LocalRealModelE2EError, match="allow-unapproved-profile"):
        _load_mapping(path, False)

    loaded = _load_mapping(path, True)
    assert loaded["approval_status"] == "UNAPPROVED"
    assert _mapping_topics(loaded)[0] == ("cam_01", "/camera/1")


def test_mapping_requires_exact_six_camera_keys(tmp_path: Path) -> None:
    path = _mapping(tmp_path, approved=True)
    loaded = _load_mapping(path, False)
    topics = loaded["topics"]
    assert isinstance(topics, dict)
    topics.pop("cam_06")

    with pytest.raises(LocalRealModelE2EError, match="cam_01 through cam_06"):
        _mapping_topics(loaded)


def test_model_output_parser_accepts_fenced_json_and_rejects_plain_text() -> None:
    assert _parse_json_output('```json\n{"scene_summary":"room"}\n```') == {"scene_summary": "room"}
    assert _parse_json_output("plain text only") is None


def test_local_file_uri_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "camera frame.png"
    target.write_bytes(b"png")

    assert _file_uri_path(target.resolve().as_uri()).resolve() == target.resolve()


def test_structured_output_shape_requires_array_fields() -> None:
    valid = {
        "scene_summary": "room",
        "observed_objects": ["table"],
        "observable_actions": ["sitting"],
        "cross_camera_consistency": "consistent",
        "uncertainties": ["identity unknown"],
    }
    invalid = dict(valid)
    invalid["observed_objects"] = "table"

    assert _structured_output_shape_valid(valid) is True
    assert _structured_output_shape_valid(invalid) is False


def test_endpoint_url_is_strictly_loopback_only() -> None:
    assert _normalize_loopback_endpoint_url("http://127.0.0.1:8101/") == "http://127.0.0.1:8101"
    assert _normalize_loopback_endpoint_url("http://[::1]:8101") == "http://[::1]:8101"
    with pytest.raises(LocalRealModelE2EError, match="loopback"):
        _normalize_loopback_endpoint_url("http://192.168.2.1:8101")
    with pytest.raises(LocalRealModelE2EError, match="API path"):
        _normalize_loopback_endpoint_url("http://127.0.0.1:8101/v1")
