from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from robata.benchmark.production_media_decode_probe import (
    CAMERA_IDS,
    COMPRESSED_IMAGE_SCHEMA,
    CameraDecodeObservation,
    ProductionMediaDecodeProbeError,
    normalize_camera_topics,
    probe_production_media,
)
from scripts import probe_production_media_decode as probe_cli


def _topics() -> dict[str, str]:
    return {
        camera_id: f"/robot0/sensor/camera{index}/compressed"
        for index, camera_id in enumerate(CAMERA_IDS)
    }


def test_normalize_camera_topics_accepts_topics_wrapper_and_canonicalizes_order() -> None:
    raw = {"topics": dict(reversed(tuple(_topics().items())))}

    assert normalize_camera_topics(raw) == tuple(_topics().items())


def test_normalize_camera_topics_accepts_camera_rows() -> None:
    raw = {
        "cameras": [
            {"id": camera_id, "camera_topic": topic}
            for camera_id, topic in reversed(tuple(_topics().items()))
        ]
    }

    assert normalize_camera_topics(raw) == tuple(_topics().items())


def test_normalize_camera_topics_rejects_missing_duplicate_and_reused_topics() -> None:
    missing = _topics()
    missing.pop("cam_06")
    with pytest.raises(ProductionMediaDecodeProbeError, match="missing cam_06"):
        normalize_camera_topics(missing)

    duplicate_camera_rows = [
        {"camera_id": "cam_01", "topic": "/camera/one"},
        {"camera_id": "cam_01", "topic": "/camera/one-again"},
    ]
    with pytest.raises(ProductionMediaDecodeProbeError, match="duplicate camera ID"):
        normalize_camera_topics(duplicate_camera_rows)

    reused_topics = _topics()
    reused_topics["cam_06"] = reused_topics["cam_05"]
    with pytest.raises(ProductionMediaDecodeProbeError, match="topics must be unique"):
        normalize_camera_topics(reused_topics)


@pytest.fixture
def scratch_dir() -> Any:
    root = Path(__file__).resolve().parents[2] / ".agent_tmp" / f"media-probe-test-{uuid4().hex}"
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


def test_probe_rejects_missing_source_and_nonpositive_bound(scratch_dir: Path) -> None:
    with pytest.raises(ProductionMediaDecodeProbeError, match="not a file"):
        probe_production_media(scratch_dir / "missing.mcap")

    source = scratch_dir / "empty.mcap"
    source.write_bytes(b"")
    with pytest.raises(ProductionMediaDecodeProbeError, match="positive integer"):
        probe_production_media(source, max_messages_per_camera=0)


def test_camera_decode_observation_serialization_is_json_compatible() -> None:
    observation = CameraDecodeObservation(
        camera_id="cam_01",
        topic="/camera/0",
        schema=COMPRESSED_IMAGE_SCHEMA,
        codec="h264",
        success=False,
        source_timestamp_ns=None,
        first_decoded_timestamp_ns=None,
        width=None,
        height=None,
        messages_examined=2,
        decoded_frames=0,
        failures=(
            {
                "code": "H264_DECODE_ERROR",
                "timestamp_ns": 12,
                "message": "bad packet",
            },
        ),
    )

    payload = observation.to_dict()
    assert payload["decode_failures"] == 1
    assert json.loads(json.dumps(payload)) == payload


def _fake_report() -> dict[str, Any]:
    return {
        "status": "SUCCEEDED",
        "camera_count": 6,
        "decoded_camera_count": 6,
        "messages_examined": 12,
        "decode_failures": 6,
    }


def test_cli_run_reads_mapping_and_writes_report(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = scratch_dir / "sample.mcap"
    source.write_bytes(b"fixture")
    mapping = scratch_dir / "mapping.json"
    mapping.write_text(json.dumps({"topics": _topics()}), encoding="utf-8")
    output = scratch_dir / "nested" / "report.json"
    seen: dict[str, Any] = {}

    def fake_probe(source_arg: Path, **kwargs: Any) -> dict[str, Any]:
        seen["source"] = source_arg
        seen.update(kwargs)
        return _fake_report()

    monkeypatch.setattr(probe_cli, "probe_production_media", fake_probe)
    report = probe_cli.run(
        source_path=source,
        mapping_config_path=mapping,
        output_path=output,
        max_messages_per_camera=7,
        validate_crcs=False,
    )

    assert report == _fake_report()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert seen["source"] == source
    assert seen["camera_topics"] == {"topics": _topics()}
    assert seen["max_messages_per_camera"] == 7
    assert seen["validate_crcs"] is False


def test_cli_main_returns_two_for_partial_probe(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = scratch_dir / "sample.mcap"
    source.write_bytes(b"fixture")
    mapping = scratch_dir / "mapping.json"
    mapping.write_text(json.dumps(_topics()), encoding="utf-8")
    output = scratch_dir / "report.json"
    partial = dict(_fake_report(), status="PARTIAL", decoded_camera_count=5)
    monkeypatch.setattr(probe_cli, "probe_production_media", lambda *_args, **_kwargs: partial)

    assert (
        probe_cli.main(
            [
                str(source),
                "--mapping-config",
                str(mapping),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert json.loads(output.read_text(encoding="utf-8")) == partial


def test_cli_main_returns_two_for_malformed_mapping(
    scratch_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = scratch_dir / "sample.mcap"
    source.write_bytes(b"fixture")
    mapping = scratch_dir / "mapping.json"
    mapping.write_text("{not-json", encoding="utf-8")

    assert (
        probe_cli.main(
            [
                str(source),
                "--mapping-config",
                str(mapping),
                "--output",
                str(scratch_dir / "report.json"),
            ]
        )
        == 2
    )
    assert "invalid mapping JSON" in capsys.readouterr().err
