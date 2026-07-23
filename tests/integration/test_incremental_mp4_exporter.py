from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

pytest.importorskip("av")

from robata.adapters import OfficialMcapInspector
from robata.adapters.mcap_single_pass import H264PacketEnvelope
from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter
from robata.application.canonical.bounded_media import (
    ACCESS_UNIT_FRAMING_VERSION,
    EncodedMediaPacket,
)
from robata.contracts import CameraId
from robata.ingestion import ExactTopicMappingPolicy, TopicMappingProfile
from robata.ports import (
    COMPRESSED_IMAGE_SCHEMA,
    ChannelInspection,
    ExportedCameraVideoFacts,
    VideoExportError,
    VideoExportErrorCode,
)

ROOT = Path(__file__).resolve().parents[2]
MEDIUM_SAMPLE = ROOT / "data" / "source" / "sample-medium.mcap"
MAPPING_PROFILE = ROOT / "config" / "genrobot-observed-v0.json"

pytestmark = pytest.mark.acceptance


def _cam01_channel() -> ChannelInspection:
    inspection = OfficialMcapInspector().inspect(MEDIUM_SAMPLE)
    profile = TopicMappingProfile.load(MAPPING_PROFILE)
    return ExactTopicMappingPolicy.from_profile(
        profile,
        allow_unapproved=True,
    ).resolve(inspection)[CameraId.CAM_01]


def _incremental_export(
    exporter: PyAvH264Mp4Exporter,
    channel: ChannelInspection,
    video_path: Path,
    sidecar_path: Path,
) -> ExportedCameraVideoFacts:
    session = exporter.begin_incremental(
        CameraId.CAM_01,
        channel,
        video_path,
        sidecar_path,
    )
    try:
        source_order = 0
        with MEDIUM_SAMPLE.open("rb") as stream:
            reader = make_reader(
                stream,
                validate_crcs=True,
                decoder_factories=[DecoderFactory()],
            )
            for schema, observed_channel, message, decoded in reader.iter_decoded_messages(
                topics=[channel.topic],
                log_time_order=False,
            ):
                if observed_channel.id != channel.channel_id:
                    continue
                assert schema is not None
                assert schema.name == COMPRESSED_IMAGE_SCHEMA
                unit = exporter._access_unit(decoded, message)
                packet = EncodedMediaPacket(
                    traversal_index=source_order,
                    camera_id=CameraId.CAM_01,
                    source_order=source_order,
                    source_sequence=unit.source_sequence,
                    source_timestamp_ns=unit.log_time_ns,
                    aligned_timestamp_ns=unit.log_time_ns,
                    source_locator=f"mcap://channel/{channel.channel_id}/packet/{source_order}",
                    payload=unit.payload,
                    is_keyframe=unit.is_keyframe,
                )
                envelope = H264PacketEnvelope(
                    packet=packet,
                    source_publish_time_ns=unit.publish_time_ns,
                    embedded_header_time_ns=unit.embedded_header_time_ns,
                    nal_types=unit.nal_types,
                )
                session.append_access_unit(
                    envelope,
                    packet.reference(),
                    framing_version=ACCESS_UNIT_FRAMING_VERSION,
                )
                source_order += 1
        session.seal()
        return session.facts
    finally:
        session.abort()


@pytest.mark.skipif(not MEDIUM_SAMPLE.exists(), reason="local sample-medium.mcap is absent")
def test_incremental_branch_matches_legacy_cam01_exactly(tmp_path: Path) -> None:
    channel = _cam01_channel()
    exporter = PyAvH264Mp4Exporter()
    legacy_video = tmp_path / "legacy.mp4"
    legacy_sidecar = tmp_path / "legacy.timestamps.jsonl"
    incremental_video = tmp_path / "incremental.mp4"
    incremental_sidecar = tmp_path / "incremental.timestamps.jsonl"

    legacy = exporter.export(
        MEDIUM_SAMPLE,
        CameraId.CAM_01,
        channel,
        legacy_video,
        legacy_sidecar,
    )
    incremental = _incremental_export(
        exporter,
        channel,
        incremental_video,
        incremental_sidecar,
    )

    assert incremental == replace(
        legacy,
        video_path=incremental_video,
        sidecar_path=incremental_sidecar,
    )
    assert incremental_video.read_bytes() == legacy_video.read_bytes()
    assert incremental_sidecar.read_bytes() == legacy_sidecar.read_bytes()
    assert not tuple(tmp_path.glob(".*.robata-*.tmp"))


def test_append_failure_abort_is_idempotent_and_leaves_no_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = ChannelInspection(
        channel_id=1,
        topic="/camera/1",
        schema_name=COMPRESSED_IMAGE_SCHEMA,
        message_encoding="protobuf",
        message_count=2,
        first_message_time_ns=0,
        last_message_time_ns=1,
        monotonic=True,
        codec="h264",
        frame_id="cam_01",
    )
    video_path = tmp_path / "aborted.mp4"
    sidecar_path = tmp_path / "aborted.timestamps.jsonl"
    session = PyAvH264Mp4Exporter().begin_incremental(
        CameraId.CAM_01,
        channel,
        video_path,
        sidecar_path,
    )
    packet = EncodedMediaPacket(
        traversal_index=0,
        camera_id=CameraId.CAM_01,
        source_order=0,
        source_sequence=0,
        source_timestamp_ns=0,
        aligned_timestamp_ns=0,
        source_locator="memory://cam01/0",
        payload=b"not-decoded",
        is_keyframe=False,
    )
    envelope = H264PacketEnvelope(
        packet=packet,
        source_publish_time_ns=0,
        embedded_header_time_ns=0,
        nal_types=(1,),
    )

    def close_failure(*, trailer_required: bool) -> None:
        raise RuntimeError(f"close failed: {trailer_required}")

    monkeypatch.setattr(session, "_close_output", close_failure)
    with pytest.raises(VideoExportError) as captured:
        session.append_access_unit(
            envelope,
            packet.reference(),
            framing_version="unsupported-framing",
        )

    assert captured.value.code is VideoExportErrorCode.INVALID_ACCESS_UNIT
    session.abort()
    session.abort()
    assert not video_path.exists()
    assert not sidecar_path.exists()
    assert not tuple(tmp_path.glob(".aborted.mp4.robata-*.tmp"))
    assert not tuple(tmp_path.glob(".aborted.timestamps.jsonl.robata-*.tmp"))
    with pytest.raises(RuntimeError, match="only after seal"):
        _ = session.facts
