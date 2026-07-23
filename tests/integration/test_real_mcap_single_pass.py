from __future__ import annotations

from pathlib import Path

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
