from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

av = pytest.importorskip("av")
pytest.importorskip("mcap")
pytest.importorskip("mcap_protobuf")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_camera_videos.py"
MEDIUM_SAMPLE = ROOT / "data" / "source" / "sample-medium.mcap"
SMALL_SAMPLE = ROOT / "data" / "source" / "sample-small.mcap"
RUN_REAL_EXPORT = os.environ.get("ROBATA_RUN_REAL_EXPORT_ACCEPTANCE") == "1"

pytestmark = pytest.mark.acceptance


def _run(source: Path, output: Path, *, allow_unapproved: bool) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), str(source), str(output)]
    if allow_unapproved:
        command.append("--allow-unapproved-profile")
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_export_cli_rejects_unapproved_mapping_before_source_access(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"

    completed = _run(
        ROOT / "data" / "source" / "does-not-exist.mcap",
        output,
        allow_unapproved=False,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["error"]["code"] == "INVALID_CAMERA_MAPPING"
    assert not output.exists()


@pytest.mark.skipif(not SMALL_SAMPLE.exists(), reason="local corrupt sample is absent")
def test_corrupt_source_publishes_no_export_directory(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"

    completed = _run(SMALL_SAMPLE, output, allow_unapproved=True)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error"]["code"] == "CORRUPT_MCAP"
    assert payload["provider_requests"] == 0
    assert not output.exists()


@pytest.mark.skipif(
    not RUN_REAL_EXPORT or not MEDIUM_SAMPLE.exists(),
    reason="set ROBATA_RUN_REAL_EXPORT_ACCEPTANCE=1 with the local medium sample",
)
def test_real_six_camera_export_is_complete_decodable_and_deterministic(
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = _run(MEDIUM_SAMPLE, first_output, allow_unapproved=True)
    second = _run(MEDIUM_SAMPLE, second_output, allow_unapproved=True)

    assert first.returncode == second.returncode == 0
    first_payload = json.loads(first.stdout)
    second_payload = json.loads(second.stdout)
    assert first_payload["execution_mode"] == "LOCAL_DEVELOPMENT_OVERRIDE"
    assert first_payload["schema_version"] == "2.0"
    assert first_payload["alignment_status"] == "UNVERIFIED"
    assert first_payload["mapping_approved"] is False
    assert first_payload["ready_manifest_id"] is None
    assert first_payload["provider_requests"] == 0
    assert first_payload["derivation_reused"] is False
    assert first_payload["materialized_view_reused"] is False
    assert second_payload["derivation_reused"] is True
    assert second_payload["materialized_view_reused"] is False
    assert first_payload["manifest_sha256"] == second_payload["manifest_sha256"]
    assert first_payload["manifest_artifact_id"] == second_payload["manifest_artifact_id"]
    assert first_payload["logical_key"] == second_payload["logical_key"]
    assert first_payload["registry_root"] == second_payload["registry_root"]
    assert len(first_payload["cameras"]) == len(second_payload["cameras"]) == 6

    for first_camera, second_camera in zip(
        first_payload["cameras"],
        second_payload["cameras"],
        strict=True,
    ):
        assert first_camera == second_camera
        assert first_camera["input_message_count"] == 1226
        assert first_camera["exported_packet_count"] == 1225
        assert first_camera["exported_frame_count"] == 1225
        assert first_camera["leading_dropped_message_count"] == 1
        assert first_camera["trailing_dropped_message_count"] == 0
        assert first_camera["keyframe_count"] == 41
        assert (first_camera["width"], first_camera["height"]) == (1600, 1300)

        video_path = first_output / f"{first_camera['camera_id']}.mp4"
        with av.open(str(video_path), mode="r") as container:
            stream = container.streams.video[0]
            frame = next(container.decode(stream))
            assert (frame.width, frame.height) == (1600, 1300)
