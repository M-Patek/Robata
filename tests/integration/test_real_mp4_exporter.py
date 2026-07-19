from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")
pytest.importorskip("mcap")
pytest.importorskip("mcap_protobuf")

# Optional media dependencies must be checked before importing concrete adapters.
from robata.adapters import OfficialMcapInspector, PyAvH264Mp4Exporter  # noqa: E402
from robata.contracts import CameraId, canonical_json_bytes  # noqa: E402
from robata.ingestion import ExactTopicMappingPolicy, TopicMappingProfile  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MEDIUM_SAMPLE = ROOT / "data" / "source" / "sample-medium.mcap"
MAPPING_PROFILE = ROOT / "config" / "genrobot-observed-v0.json"

pytestmark = pytest.mark.acceptance


@pytest.mark.skipif(not MEDIUM_SAMPLE.exists(), reason="local sample-medium.mcap is absent")
def test_real_cam01_direct_remux_is_complete_exact_and_independently_decodable(
    tmp_path: Path,
) -> None:
    inspection = OfficialMcapInspector().inspect(MEDIUM_SAMPLE)
    profile = TopicMappingProfile.load(MAPPING_PROFILE)
    mapping = ExactTopicMappingPolicy.from_profile(
        profile,
        allow_unapproved=True,
    ).resolve(inspection)
    video_path = tmp_path / "cam_01.mp4"
    sidecar_path = tmp_path / "cam_01.timestamps.jsonl"

    facts = PyAvH264Mp4Exporter().export(
        MEDIUM_SAMPLE,
        CameraId.CAM_01,
        mapping[CameraId.CAM_01],
        video_path,
        sidecar_path,
    )

    assert facts.source_message_count == 1226
    assert facts.leading_access_unit_count == 1
    assert facts.leading_first_log_time_ns == 1781051907238275000
    assert facts.leading_last_log_time_ns == 1781051907238275000
    assert facts.trailing_access_unit_count == 0
    assert facts.trailing_first_log_time_ns is None
    assert facts.trailing_last_log_time_ns is None
    assert facts.exported_packet_count == 1225
    assert facts.decoded_frame_count == 1225
    assert facts.keyframe_count == 41
    assert (facts.width, facts.height) == (1600, 1300)
    assert facts.first_pts_ns == 0
    assert facts.last_pts_ns == 40_800_129_000
    assert facts.tail_duration_ns == 33_334_000
    assert facts.duration_ns == 40_833_463_000
    assert facts.tail_duration_policy == "MEDIAN_POSITIVE_INTERVAL"
    assert facts.max_timestamp_mapping_error_ns == 0
    assert facts.video_size_bytes == video_path.stat().st_size
    assert facts.sidecar_size_bytes == sidecar_path.stat().st_size
    assert re.fullmatch(r"[0-9a-f]{64}", facts.video_sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", facts.sidecar_sha256)

    lines = sidecar_path.read_bytes().splitlines(keepends=True)
    assert len(lines) == facts.sidecar_row_count == 1225
    rows = [json.loads(line) for line in lines]
    assert all(
        line[:-1] == canonical_json_bytes(row) for line, row in zip(lines, rows, strict=True)
    )
    assert rows[0]["relative_pts_ns"] == "0"
    assert rows[0]["source_log_time_ns"] == str(facts.export_first_source_log_time_ns)
    assert rows[0]["embedded_header_time_ns"] == rows[0]["source_log_time_ns"]
    assert rows[0]["is_keyframe"] is True
    assert rows[0]["duration_is_estimated"] is False
    assert rows[-1]["relative_pts_ns"] == str(facts.last_pts_ns)
    assert rows[-1]["duration_ns"] == str(facts.tail_duration_ns)
    assert rows[-1]["duration_is_estimated"] is True

    with av.open(str(video_path), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.time_base == Fraction(1, 1_000_000_000)
        frame_count = 0
        keyframe_count = 0
        for frame in container.decode(stream):
            assert (frame.width, frame.height) == (1600, 1300)
            frame_count += 1
            keyframe_count += int(bool(frame.key_frame))
    assert frame_count == 1225
    assert keyframe_count == 41
