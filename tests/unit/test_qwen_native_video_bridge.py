from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robata.benchmark import qwen_native_video_bridge as bridge
from robata.contracts import CameraId, SixCameraMap
from robata.ingestion.mapping import TopicMappingProfile
from robata.ports.ingestion import ChannelInspection

ROOT = Path(__file__).resolve().parents[2]
MAPPING = TopicMappingProfile.load(ROOT / "config" / "genrobot-observed-v0.json")


def _channels() -> tuple[ChannelInspection, ...]:
    return tuple(
        ChannelInspection(
            channel_id=index,
            topic=f"/robot0/sensor/camera{index - 1}/compressed",
            schema_name="foxglove.CompressedImage",
            message_encoding="protobuf",
            message_count=4,
            first_message_time_ns=100 + index,
            last_message_time_ns=400 + index,
            monotonic=True,
            codec="h264",
            frame_id=None,
            schema_encoding="protobuf",
            schema_content_sha256=None,
        )
        for index in range(1, 7)
    )


def _patch_inspection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    source = tmp_path / "source.mcap"
    source.write_bytes(b"not-read-by-fake-inspector")
    monkeypatch.setattr(
        bridge,
        "inspect_mcap_without_digests",
        lambda _source: (source, source.stat().st_size, _channels()),
    )
    return source


def test_plan_does_not_hash_and_declares_complete_six_camera_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _patch_inspection(monkeypatch, tmp_path)

    def fail_hash(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("bridge plan must not calculate a digest")

    monkeypatch.setattr(hashlib, "sha256", fail_hash)
    plan = bridge.build_qwen_native_video_plan(
        source,
        tmp_path / "video-root",
        mapping_profile=MAPPING,
        allow_unapproved_profile=True,
    )

    assert plan["status"] == "DRY_RUN"
    assert plan["controls"]["sha_or_digest_computed"] is False
    assert plan["output"]["camera_order"] == [f"cam_{index:02d}" for index in range(1, 7)]
    assert len(plan["output"]["required_files"]) == 13
    assert not (tmp_path / "video-root").exists()
    assert all("sha256" not in row and "hash" not in row for row in plan["cameras"])


def test_unapproved_profile_requires_explicit_local_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _patch_inspection(monkeypatch, tmp_path)
    with pytest.raises(bridge.QwenNativeVideoBridgeError, match="not approved"):
        bridge.build_qwen_native_video_plan(
            source,
            tmp_path / "video-root",
            mapping_profile=MAPPING,
        )


def test_mapping_profile_rejects_duplicate_topics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _patch_inspection(monkeypatch, tmp_path)
    topics = {camera_id.value: MAPPING.topics[camera_id] for camera_id in CameraId}
    topics[CameraId.CAM_02.value] = topics[CameraId.CAM_01.value]
    duplicate_profile = TopicMappingProfile(
        profile_id=MAPPING.profile_id,
        version=MAPPING.version,
        profile_kind=MAPPING.profile_kind,
        approval_status=MAPPING.approval_status,
        approved=MAPPING.approved,
        mapping_policy=MAPPING.mapping_policy,
        required_schema=MAPPING.required_schema,
        topics=SixCameraMap[str].model_validate(topics, strict=True),
    )
    with pytest.raises(bridge.QwenNativeVideoBridgeError, match="more than once"):
        bridge.build_qwen_native_video_plan(
            source,
            tmp_path / "video-root",
            mapping_profile=duplicate_profile,
            allow_unapproved_profile=True,
        )


class _FakeExporter:
    def export(
        self,
        _source: Path,
        camera_id: Any,
        channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> Any:
        video_path.write_bytes(f"video-{camera_id.value}".encode())
        sidecar_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            exported_packet_count=1,
            decoded_frame_count=1,
            keyframe_count=1,
            leading_access_unit_count=0,
            trailing_access_unit_count=0,
            width=16,
            height=16,
            export_first_source_log_time_ns=channel.first_message_time_ns,
            export_last_source_log_time_ns=channel.last_message_time_ns,
            first_pts_ns=0,
            last_pts_ns=300,
            duration_ns=301,
            time_base_numerator=1,
            time_base_denominator=1_000_000_000,
            tail_duration_ns=1,
            sidecar_row_count=1,
        )


class _EmptyExporter:
    def export(
        self,
        _source: Path,
        _camera_id: Any,
        _channel: ChannelInspection,
        video_path: Path,
        sidecar_path: Path,
    ) -> Any:
        video_path.touch()
        sidecar_path.touch()
        return None


def test_materialization_writes_qwen_view_without_digest_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _patch_inspection(monkeypatch, tmp_path)
    output = tmp_path / "video-root"
    result = bridge.materialize_qwen_native_video_inputs(
        source,
        output,
        mapping_profile=MAPPING,
        allow_unapproved_profile=True,
        exporter=_FakeExporter(),
    )

    assert result.dry_run is False
    assert result.manifest["status"] == "MATERIALIZED"
    assert set(path.name for path in output.iterdir()) == {
        *(f"cam_{index:02d}.mp4" for index in range(1, 7)),
        *(f"cam_{index:02d}.timestamps.jsonl" for index in range(1, 7)),
        bridge.QWEN_NATIVE_VIDEO_BRIDGE_MANIFEST_FILENAME,
    }
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["controls"]["sha_or_digest_computed"] is False
    assert all(
        not any(token in key.casefold() for token in ("sha", "hash", "digest"))
        for row in payload["cameras"]
        for key in row
    )


def test_materialization_rejects_empty_media_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _patch_inspection(monkeypatch, tmp_path)
    with pytest.raises(bridge.QwenNativeVideoBridgeError, match="must be non-empty"):
        bridge.materialize_qwen_native_video_inputs(
            source,
            tmp_path / "video-root",
            mapping_profile=MAPPING,
            allow_unapproved_profile=True,
            exporter=_EmptyExporter(),
        )


def test_compatibility_seam_reports_size_without_hashing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"bytes")

    def fail_hash(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("compatibility seam must not call hashlib.sha256")

    monkeypatch.setattr(hashlib, "sha256", fail_hash)
    size, digest = bridge.NoDigestPyAvH264Mp4Exporter._hash_file(path)
    assert size == 5
    assert digest is None
