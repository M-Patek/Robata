from __future__ import annotations

import io
from io import BytesIO
from pathlib import Path

import pytest
from mcap.writer import CompressionType, IndexType, Writer

from robata.adapters.mcap_inspector import OfficialMcapInspector
from robata.adapters.mcap_single_pass import (
    AppendOnlyH264SpoolBranch,
    McapSinglePassH264Tee,
    iter_h264_spool,
)
from robata.application.canonical.bounded_media import (
    BoundedMediaPolicy,
    BoundedSinglePassMediaPlanner,
)
from robata.contracts import CAMERA_IDS, SixCameraMap
from robata.ports import ChannelInspection
from tests.support.six_camera_mcap import (
    SIX_CAMERA_MCAP_SHA256,
    SIX_CAMERA_TOPICS,
    write_six_camera_mcap,
)


def test_real_mcap_single_pass_reconciles_source_and_six_camera_spools(
    tmp_path: Path,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    inspection = OfficialMcapInspector().inspect(source)
    channels = SixCameraMap[ChannelInspection].model_validate(
        {
            camera_id: inspection.channels_for_topic(topic)[0]
            for camera_id, topic in zip(CAMERA_IDS, SIX_CAMERA_TOPICS, strict=True)
        },
        strict=True,
    )
    assert inspection.first_message_time_ns is not None
    assert inspection.last_message_time_ns is not None
    planner = BoundedSinglePassMediaPlanner(
        BoundedMediaPolicy(
            source_scope_digest=inspection.source_sha256,
            mapping_semantic_sha256="b" * 64,
            alignment_semantic_sha256="c" * 64,
            source_origin_ns=inspection.first_message_time_ns,
            allowed_lateness_ns=0,
            ring_max_bytes_per_camera=1024 * 1024,
        )
    )
    branches = {
        camera_id: AppendOnlyH264SpoolBranch(
            camera_id,
            tmp_path / f"{camera_id.value}.h264.spool",
        )
        for camera_id in CAMERA_IDS
    }

    result = McapSinglePassH264Tee().traverse(
        source,
        channels,
        planner,
        branches,
        final_end_ns=inspection.last_message_time_ns + 1,
        expected_source_sha256=inspection.source_sha256,
    )

    assert result.source_sha256 == inspection.source_sha256 == SIX_CAMERA_MCAP_SHA256
    assert result.source_size_bytes == inspection.source_size_bytes == source.stat().st_size
    assert result.source_message_count == inspection.message_count
    assert result.selected_packet_count == sum(
        channel.message_count for channel in channels.values()
    )
    assert result.camera_packet_counts == SixCameraMap[int].model_validate(
        {camera_id: channels[camera_id].message_count for camera_id in CAMERA_IDS},
        strict=True,
    )
    for camera_id in CAMERA_IDS:
        branch = branches[camera_id]
        records = tuple(iter_h264_spool(branch.facts.path))
        assert len(records) == channels[camera_id].message_count
        assert tuple(record.packet.source_order for record in records) == tuple(range(len(records)))
        assert all(record.packet.camera_id is camera_id for record in records)
        assert branch.facts.size_bytes == branch.facts.path.stat().st_size
    assert all(
        snapshot.total_bytes <= planner.policy.ring_max_bytes_per_camera
        for snapshot in planner.ring_snapshots()
    )


def test_single_pass_inspection_matches_official_and_uses_one_sequential_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_six_camera_mcap(tmp_path / "six-camera.mcap")
    inspector = OfficialMcapInspector()
    expected = inspector.inspect(source)
    preflight = inspector.preflight(source)
    mapping_view = preflight.as_mapping_inspection(SIX_CAMERA_MCAP_SHA256)
    channels = SixCameraMap[ChannelInspection].model_validate(
        {
            camera_id: mapping_view.channels_for_topic(topic)[0]
            for camera_id, topic in zip(CAMERA_IDS, SIX_CAMERA_TOPICS, strict=True)
        },
        strict=True,
    )
    assert mapping_view.first_message_time_ns is not None
    assert mapping_view.last_message_time_ns is not None
    planner = BoundedSinglePassMediaPlanner(
        BoundedMediaPolicy(
            source_scope_digest=SIX_CAMERA_MCAP_SHA256,
            mapping_semantic_sha256="b" * 64,
            alignment_semantic_sha256="c" * 64,
            source_origin_ns=min(
                channel.first_message_time_ns
                for channel in channels.values()
                if channel.first_message_time_ns is not None
            ),
            allowed_lateness_ns=0,
            ring_max_bytes_per_camera=1024 * 1024,
        )
    )
    branches = {
        camera_id: AppendOnlyH264SpoolBranch(
            camera_id,
            tmp_path / f"fused-{camera_id.value}.h264.spool",
        )
        for camera_id in CAMERA_IDS
    }
    source_opens = 0
    original_open = io.open

    def observed_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        nonlocal source_opens
        if Path(file) == source and mode == "rb":  # type: ignore[arg-type]
            source_opens += 1
        return original_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(io, "open", observed_open)
    result = McapSinglePassH264Tee().traverse(
        source,
        channels,
        planner,
        branches,
        preflight=preflight,
        expected_source_sha256=SIX_CAMERA_MCAP_SHA256,
        final_end_ns=mapping_view.last_message_time_ns + 1,
    )

    assert source_opens == 1
    assert result.inspection == expected


def test_preflight_reports_summary_without_faking_message_index_facts(tmp_path: Path) -> None:
    output = BytesIO()
    writer = Writer(
        output,
        compression=CompressionType.NONE,
        index_types=IndexType.CHUNK,
        use_chunking=True,
        use_statistics=True,
    )
    writer.start(profile="no-message-index", library="test")
    channel_id = writer.register_channel(
        topic="/fixture/no-index",
        message_encoding="json",
        schema_id=0,
    )
    writer.add_message(
        channel_id=channel_id,
        log_time=1,
        publish_time=1,
        sequence=0,
        data=b"{}",
    )
    writer.finish()
    source = tmp_path / "no-message-index.mcap"
    source.write_bytes(output.getvalue())

    preflight = OfficialMcapInspector().preflight(source)

    assert preflight.message_indexes_complete is False
    assert preflight.message_count == 1
    assert len(preflight.channels) == 1
    channel = preflight.channels[0]
    assert channel.message_count == 1
    assert channel.first_message_time_ns is None
    assert channel.last_message_time_ns is None
    assert channel.monotonic is False
