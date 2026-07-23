from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import robata.adapters.mcap_single_pass as subject
from robata.adapters.mcap_single_pass import (
    AppendOnlyH264SpoolBranch,
    McapSinglePassH264Tee,
    iter_h264_spool,
)
from robata.application.canonical.bounded_media import (
    BoundedMediaPolicy,
    BoundedSinglePassMediaPlanner,
)
from robata.contracts import CAMERA_IDS, CameraId, SixCameraMap
from robata.ports import COMPRESSED_IMAGE_SCHEMA, ChannelInspection


@dataclass(frozen=True)
class _FakeChannel:
    id: int
    topic: str


@dataclass(frozen=True)
class _FakeSchema:
    id: int = 100
    name: str = COMPRESSED_IMAGE_SCHEMA


class _FakeReader:
    def __init__(self, stream: Any, rows: list[tuple[Any, Any, Any]]) -> None:
        self._stream = stream
        self._rows = rows

    def iter_messages(self, *, log_time_order: bool) -> Any:
        assert log_time_order is False
        assert self._stream.seekable() is False
        self._stream.read(4)
        return iter(self._rows)


class _FakeDecoderFactory:
    @staticmethod
    def decoder_for(_message_encoding: str, _schema: Any) -> Any:
        return lambda data: data


def _channel(camera_id: CameraId) -> ChannelInspection:
    ordinal = CAMERA_IDS.index(camera_id) + 1
    return ChannelInspection(
        channel_id=ordinal,
        topic=f"/camera/{ordinal}",
        schema_name=COMPRESSED_IMAGE_SCHEMA,
        message_encoding="protobuf",
        message_count=2,
        first_message_time_ns=0,
        last_message_time_ns=500_000_000,
        monotonic=True,
        codec="h264",
        frame_id=camera_id.value,
    )


def _rows() -> list[tuple[Any, Any, Any]]:
    rows: list[tuple[Any, Any, Any]] = []
    schema = _FakeSchema()
    for source_order, timestamp_ns in enumerate((0, 500_000_000)):
        for camera_id in CAMERA_IDS:
            channel = _channel(camera_id)
            nal_type = 5 if source_order == 0 else 1
            payload = b"\x00\x00\x00\x01" + bytes([nal_type]) + camera_id.value.encode()
            decoded = SimpleNamespace(
                format="h264",
                data=payload,
                header=SimpleNamespace(timestamp=timestamp_ns + 20),
            )
            rows.append(
                (
                    schema,
                    SimpleNamespace(
                        id=channel.channel_id,
                        topic=channel.topic,
                        message_encoding="protobuf",
                    ),
                    SimpleNamespace(
                        log_time=timestamp_ns,
                        publish_time=timestamp_ns + 10,
                        sequence=source_order,
                        data=decoded,
                    ),
                )
            )
        if source_order == 0:
            rows.append(
                (
                    schema,
                    SimpleNamespace(id=999, topic="/unmapped", message_encoding="raw"),
                    SimpleNamespace(log_time=1, publish_time=1, sequence=0, data=b"raw"),
                )
            )
    return rows


def _planner() -> BoundedSinglePassMediaPlanner:
    return BoundedSinglePassMediaPlanner(
        BoundedMediaPolicy(
            source_scope_digest="a" * 64,
            mapping_semantic_sha256="b" * 64,
            alignment_semantic_sha256="c" * 64,
            source_origin_ns=0,
            allowed_lateness_ns=0,
            ring_max_bytes_per_camera=64,
        )
    )


def test_one_reader_traversal_feeds_six_ordered_spools_and_bounded_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_calls = 0

    def fake_make_reader(stream: Any, **_kwargs: Any) -> _FakeReader:
        nonlocal reader_calls
        reader_calls += 1
        return _FakeReader(stream, _rows())

    monkeypatch.setattr(subject, "make_reader", fake_make_reader)
    monkeypatch.setattr(subject, "DecoderFactory", _FakeDecoderFactory)
    source = tmp_path / "source.mcap"
    source_bytes = b"fake-mcap-source"
    source.write_bytes(source_bytes)
    channels = SixCameraMap[ChannelInspection].model_validate(
        {camera_id: _channel(camera_id) for camera_id in CAMERA_IDS}, strict=True
    )
    spool_branches = {
        camera_id: AppendOnlyH264SpoolBranch(camera_id, tmp_path / f"{camera_id.value}.spool")
        for camera_id in CAMERA_IDS
    }
    planner = _planner()

    result = McapSinglePassH264Tee().traverse(
        source,
        channels,
        planner,
        spool_branches,
        final_end_ns=1_000_000_000,
    )

    assert reader_calls == 1
    assert result.source_size_bytes == len(source_bytes)
    assert result.source_sha256 == sha256(source_bytes).hexdigest()
    assert result.source_message_count == 13
    assert result.selected_packet_count == 12
    assert tuple(result.camera_packet_counts.values()) == (2, 2, 2, 2, 2, 2)
    assert result.final_end_ns == 1_000_000_000
    assert all(snapshot.total_bytes <= 64 for snapshot in planner.ring_snapshots())

    traversal_indexes: list[int] = []
    for camera_id in CAMERA_IDS:
        branch = spool_branches[camera_id]
        records = tuple(iter_h264_spool(branch.facts.path))
        assert branch.facts.packet_count == 2
        assert branch.facts.size_bytes == branch.facts.path.stat().st_size
        assert [record.packet.source_order for record in records] == [0, 1]
        assert [record.packet.source_sequence for record in records] == [0, 1]
        assert [record.packet.source_timestamp_ns for record in records] == [0, 500_000_000]
        assert [record.source_publish_time_ns for record in records] == [10, 500_000_010]
        assert [record.embedded_header_time_ns for record in records] == [20, 500_000_020]
        assert [record.is_keyframe for record in records] == [True, False]
        traversal_indexes.extend(record.packet.traversal_index for record in records)
    assert sorted(traversal_indexes) == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def test_branch_failure_aborts_every_spool_without_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "make_reader",
        lambda stream, **_kwargs: _FakeReader(stream, _rows()),
    )
    monkeypatch.setattr(subject, "DecoderFactory", _FakeDecoderFactory)
    source = tmp_path / "source.mcap"
    source.write_bytes(b"fake")
    channels = SixCameraMap[ChannelInspection].model_validate(
        {camera_id: _channel(camera_id) for camera_id in CAMERA_IDS}, strict=True
    )
    spool_branches = {
        camera_id: AppendOnlyH264SpoolBranch(camera_id, tmp_path / f"{camera_id.value}.spool")
        for camera_id in CAMERA_IDS
    }
    failing = spool_branches[CameraId.CAM_03]

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(failing, "append_access_unit", fail)
    with pytest.raises(subject.SinglePassMcapError, match="disk full"):
        McapSinglePassH264Tee().traverse(source, channels, _planner(), spool_branches)

    assert all(branch._stream.closed for branch in spool_branches.values())
    assert all(branch._facts is None for branch in spool_branches.values())
