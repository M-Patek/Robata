from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcap")
pytest.importorskip("mcap_protobuf")
pytest.importorskip("av")

from robata.adapters import OfficialMcapInspector, PyAvH264DecoderProbe
from robata.contracts import CAMERA_IDS
from robata.ingestion import ExactTopicMappingPolicy, TopicMappingProfile
from robata.ports import IngestionError, IngestionErrorCode

ROOT = Path(__file__).resolve().parents[2]
MEDIUM_SAMPLE = ROOT / "data" / "source" / "sample-medium.mcap"
SMALL_SAMPLE = ROOT / "data" / "source" / "sample-small.mcap"
MAPPING_PROFILE = ROOT / "config" / "genrobot-observed-v0.json"

pytestmark = pytest.mark.acceptance


def test_cli_rejects_unapproved_profile_before_source_access() -> None:
    missing_source = ROOT / "data" / "source" / "does-not-exist.mcap"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "inspect_mcap.py"), str(missing_source)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["error"]["code"] == "INVALID_CAMERA_MAPPING"
    assert "not approved" in payload["error"]["message"]


@pytest.mark.skipif(not MEDIUM_SAMPLE.exists(), reason="local sample-medium.mcap is absent")
def test_real_medium_mcap_inspection_mapping_and_decode() -> None:
    inspection = OfficialMcapInspector().inspect(MEDIUM_SAMPLE)

    assert inspection.header_profile == "Genrobot"
    assert inspection.header_library == "libmcap"
    assert inspection.summary_available
    assert inspection.channel_count == 17
    assert inspection.source_size_bytes == MEDIUM_SAMPLE.stat().st_size
    assert re.fullmatch(r"[0-9a-f]{64}", inspection.source_sha256)

    profile = TopicMappingProfile.load(MAPPING_PROFILE)
    assert profile.profile_kind == "OBSERVED"
    assert profile.approval_status == "UNAPPROVED"
    assert not profile.approved
    mapping = ExactTopicMappingPolicy.from_profile(
        profile,
        allow_unapproved=True,
    ).resolve(inspection)

    probe = PyAvH264DecoderProbe()
    for camera_id in CAMERA_IDS:
        channel = mapping[camera_id]
        assert channel.message_count == 1226
        assert channel.schema_name == "foxglove.CompressedImage"
        assert channel.codec == "h264"
        assert channel.monotonic
        result = probe.probe(MEDIUM_SAMPLE, channel)
        assert result.success
        assert result.width == 1600
        assert result.height == 1300
        assert result.first_decoded_timestamp_ns is not None
        assert result.decoded_frames >= 1


@pytest.mark.skipif(not SMALL_SAMPLE.exists(), reason="local sample-small.mcap is absent")
def test_real_small_mcap_reports_stable_structural_error() -> None:
    with pytest.raises(IngestionError) as raised:
        OfficialMcapInspector().inspect(SMALL_SAMPLE)

    assert raised.value.code is IngestionErrorCode.CORRUPT_MCAP
