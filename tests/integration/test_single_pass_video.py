from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("av")

from robata.adapters.mcap_inspector import OfficialMcapInspector
from robata.adapters.mcap_single_pass import (
    PLANNING_MODE_LIVE_BOOTSTRAP,
    McapSinglePassH264Tee,
)
from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter
from robata.application.canonical import single_pass_video
from robata.application.canonical.bounded_media import (
    BoundedMediaPolicy,
    PlannerEmission,
    PlannerFinish,
)
from robata.application.canonical.mcap_source import authorize_mcap_mapping
from robata.application.canonical.single_pass_video import (
    SPOOL_SEAL_FILENAME,
    DurableSinglePassVideoProducer,
    SinglePassVideoProductionError,
    read_sealed_mcap_inspection,
)
from robata.contracts import CAMERA_IDS, CameraId, SixCameraMap, canonical_json_bytes
from robata.ports import ChannelInspection, McapInspection
from tests.support.six_camera_mcap import SIX_CAMERA_TOPICS, write_six_camera_mcap

MEDIUM_SAMPLE = Path("data/source/sample-medium.mcap")
MEDIUM_MAPPING = Path("config/genrobot-observed-v0.json")


class _CountingTee(McapSinglePassH264Tee):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def traverse(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().traverse(*args, **kwargs)  # type: ignore[arg-type]


class _RecordingPlanningSink:
    def __init__(self, source_position: Callable[[], int] | None = None) -> None:
        self.emissions: list[PlannerEmission] = []
        self.finishes: list[PlannerFinish] = []
        self.source_positions: list[int] = []
        self._source_position = source_position

    def append_emission(self, emission: PlannerEmission) -> None:
        self.emissions.append(emission)
        if self._source_position is not None:
            self.source_positions.append(self._source_position())

    def seal(self, finish: PlannerFinish) -> None:
        self.finishes.append(finish)


def _source_context(
    tmp_path: Path,
) -> tuple[McapInspection, SixCameraMap[ChannelInspection], BoundedMediaPolicy]:
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
    policy = BoundedMediaPolicy(
        source_scope_digest="d" * 64,
        mapping_semantic_sha256="b" * 64,
        alignment_semantic_sha256="c" * 64,
        source_origin_ns=inspection.first_message_time_ns,
        allowed_lateness_ns=0,
        ring_max_bytes_per_camera=1024,
    )
    return inspection, channels, policy


def test_fresh_and_recovered_spool_production_match_legacy_export_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspection, channels, policy = _source_context(tmp_path)
    spool_directory = tmp_path / "durable-spools"
    fresh_stage = tmp_path / "fresh-stage"
    legacy_stage = tmp_path / "legacy-stage"
    recovered_stage = tmp_path / "recovered-stage"
    fresh_stage.mkdir()
    legacy_stage.mkdir()
    recovered_stage.mkdir()

    fresh_tee = _CountingTee()
    fresh_sink = _RecordingPlanningSink()
    fresh_producer = DurableSinglePassVideoProducer(
        inspection=inspection,
        channels=channels,
        planner_policy=policy,
        spool_directory=spool_directory,
        max_parallel_exports=3,
        planning_sink=fresh_sink,
        tee=fresh_tee,
    )

    def reject_redundant_spool_hash(_path: Path) -> tuple[int, str]:
        raise AssertionError("fresh publication must use append-time spool digests")

    with monkeypatch.context() as fresh_guard:
        fresh_guard.setattr(single_pass_video, "_hash_file", reject_redundant_spool_hash)
        fresh = fresh_producer.produce(fresh_stage)

    assert fresh_tee.calls == 1
    assert not fresh.reused_spool_set
    assert fresh.max_parallel_exports == fresh_producer.max_parallel_exports == 3
    assert fresh.spool_set.source_sha256 == inspection.source_sha256
    assert fresh.spool_set.source_size_bytes == inspection.source_size_bytes
    assert fresh.spool_set.source_message_count == inspection.message_count
    assert fresh.spool_set.inspection == inspection
    assert fresh.traversal.inspection == inspection
    assert fresh.spool_set.seal_path == spool_directory / SPOOL_SEAL_FILENAME
    seal_document = fresh.spool_set.seal_path.read_bytes()
    assert canonical_json_bytes(json.loads(seal_document)) == seal_document
    assert len(fresh_sink.emissions) == fresh.spool_set.selected_packet_count
    assert fresh_sink.finishes == [fresh.planner_finish]
    assert all(
        snapshot.total_bytes <= policy.ring_max_bytes_per_camera
        for snapshot in fresh.planner_ring_snapshots
    )

    exporter = PyAvH264Mp4Exporter()
    legacy_facts = {}
    for camera_id in CAMERA_IDS:
        legacy_facts[camera_id] = exporter.export(
            inspection.source,
            camera_id,
            channels[camera_id],
            legacy_stage / f"{camera_id.value}.mp4",
            legacy_stage / f"{camera_id.value}.timestamps.jsonl",
        )
    for camera_id, actual in zip(CAMERA_IDS, fresh.staged_export.camera_facts, strict=True):
        assert actual == replace(
            legacy_facts[camera_id],
            video_path=fresh_stage / f"{camera_id.value}.mp4",
            sidecar_path=fresh_stage / f"{camera_id.value}.timestamps.jsonl",
        )
        assert actual.video_path.read_bytes() == legacy_facts[camera_id].video_path.read_bytes()
        assert actual.sidecar_path.read_bytes() == legacy_facts[camera_id].sidecar_path.read_bytes()

    inspection.source.unlink()
    assert (
        read_sealed_mcap_inspection(
            spool_directory,
            source=inspection.source,
            expected_source_sha256=inspection.source_sha256,
        )
        == inspection
    )
    recovery_tee = _CountingTee()
    recovery_sink = _RecordingPlanningSink()
    recovered_producer = DurableSinglePassVideoProducer(
        inspection=inspection,
        channels=channels,
        planner_policy=policy,
        spool_directory=spool_directory,
        max_parallel_exports=2,
        planning_sink=recovery_sink,
        tee=recovery_tee,
    )
    hashed_paths: list[Path] = []
    original_hash_file = single_pass_video._hash_file

    def record_recovery_hash(path: Path) -> tuple[int, str]:
        hashed_paths.append(path)
        return original_hash_file(path)

    with monkeypatch.context() as recovery_guard:
        recovery_guard.setattr(single_pass_video, "_hash_file", record_recovery_hash)
        recovered = recovered_producer.produce(recovered_stage)

    assert recovery_tee.calls == 0
    assert recovered.reused_spool_set
    assert hashed_paths == [recovered.spool_set.spools[camera_id].path for camera_id in CAMERA_IDS]
    assert recovery_sink.emissions == fresh_sink.emissions
    assert recovery_sink.finishes == [recovered.planner_finish]
    assert recovered.planner_finish == fresh.planner_finish
    assert recovered.traversal == fresh.traversal
    assert recovered.spool_set == fresh.spool_set
    assert recovered.spool_set.inspection == inspection
    for fresh_fact, recovered_fact in zip(
        fresh.staged_export.camera_facts,
        recovered.staged_export.camera_facts,
        strict=True,
    ):
        assert recovered_fact == replace(
            fresh_fact,
            video_path=recovered_stage / fresh_fact.video_path.name,
            sidecar_path=recovered_stage / fresh_fact.sidecar_path.name,
        )
        assert recovered_fact.video_path.read_bytes() == fresh_fact.video_path.read_bytes()
        assert recovered_fact.sidecar_path.read_bytes() == fresh_fact.sidecar_path.read_bytes()


def test_partial_spool_directory_is_not_reused(tmp_path: Path) -> None:
    inspection, channels, policy = _source_context(tmp_path)
    spool_directory = tmp_path / "partial-spools"
    spool_directory.mkdir()
    partial = spool_directory / f"{CameraId.CAM_01.value}.h264.spool"
    partial.write_bytes(b"partial")
    stage = tmp_path / "stage"
    stage.mkdir()
    tee = _CountingTee()
    producer = DurableSinglePassVideoProducer(
        inspection=inspection,
        channels=channels,
        planner_policy=policy,
        spool_directory=spool_directory,
        tee=tee,
    )

    with pytest.raises(SinglePassVideoProductionError, match="no complete seal"):
        producer.produce(stage)

    assert tee.calls == 0
    assert partial.read_bytes() == b"partial"


@pytest.mark.parametrize("workers", [0, 7])
def test_parallel_export_bound_is_one_to_six(tmp_path: Path, workers: int) -> None:
    inspection, channels, policy = _source_context(tmp_path)

    with pytest.raises(ValueError, match="between one and six"):
        DurableSinglePassVideoProducer(
            inspection=inspection,
            channels=channels,
            planner_policy=policy,
            spool_directory=tmp_path / "spools",
            max_parallel_exports=workers,
        )


@pytest.mark.skipif(not MEDIUM_SAMPLE.exists(), reason="local sample-medium.mcap is absent")
def test_real_no_index_capture_is_one_pass_and_recovery_never_opens_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MEDIUM_SAMPLE.resolve()
    inspector = OfficialMcapInspector()
    expected = inspector.inspect(source)
    preflight = inspector.preflight(source)
    assert preflight.message_indexes_complete is False
    authorization = authorize_mcap_mapping(
        MEDIUM_MAPPING,
        allow_unapproved_profile=True,
    )
    mapping_view = preflight.as_mapping_inspection(expected.source_sha256)
    channels = authorization.policy.resolve(mapping_view)

    def policy_factory(
        source_origin_ns: int,
    ) -> BoundedMediaPolicy:
        return BoundedMediaPolicy(
            source_scope_digest="d" * 64,
            mapping_semantic_sha256=authorization.semantic_sha256,
            alignment_semantic_sha256="c" * 64,
            source_origin_ns=source_origin_ns,
        )

    source_opens = 0
    opened_source: io.BufferedReader | None = None
    original_open = io.open

    def observed_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        nonlocal opened_source, source_opens
        stream = original_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]
        if Path(file).resolve() == source and mode == "rb":  # type: ignore[arg-type]
            source_opens += 1
            opened_source = stream
        return stream

    monkeypatch.setattr(io, "open", observed_open)
    spool_directory = tmp_path / "real-spools"
    fresh_tee = _CountingTee()
    fresh_sink = _RecordingPlanningSink(
        lambda: opened_source.tell() if opened_source is not None else expected.source_size_bytes
    )
    fresh_producer = DurableSinglePassVideoProducer(
        inspection=mapping_view,
        channels=channels,
        planner_policy=None,
        planner_policy_factory=policy_factory,
        preflight=preflight,
        spool_directory=spool_directory,
        planner_source_scope_digest="d" * 64,
        planning_sink=fresh_sink,
        tee=fresh_tee,
    )
    fresh = fresh_producer.prepare()

    assert source_opens == 1
    assert fresh_tee.calls == 1
    assert fresh.inspection == expected
    assert fresh.planning_mode == PLANNING_MODE_LIVE_BOOTSTRAP
    assert fresh.preflight_message_indexes_complete is False
    assert len(fresh_sink.emissions) == fresh.selected_packet_count
    assert fresh_sink.source_positions
    assert fresh_sink.source_positions[0] < expected.source_size_bytes
    assert fresh_producer.planner_finish == fresh_sink.finishes[0]

    final_channels = authorization.policy.resolve(expected)
    final_end_ns = (
        max(
            channel.last_message_time_ns
            for channel in final_channels.values()
            if channel.last_message_time_ns is not None
        )
        + 1
    )
    recovery_tee = _CountingTee()
    recovery_sink = _RecordingPlanningSink()
    recovered_producer = DurableSinglePassVideoProducer(
        inspection=expected,
        channels=final_channels,
        planner_policy=policy_factory(
            min(
                channel.first_message_time_ns
                for channel in final_channels.values()
                if channel.first_message_time_ns is not None
            )
        ),
        spool_directory=spool_directory,
        final_end_ns=final_end_ns,
        planning_sink=recovery_sink,
        tee=recovery_tee,
    )

    def reject_source_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if Path(file).resolve() == source:  # type: ignore[arg-type]
            raise AssertionError("sealed recovery must not open the MCAP source")
        return original_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(io, "open", reject_source_open)
    recovered = recovered_producer.prepare()

    assert recovery_tee.calls == 0
    assert recovered.inspection == expected
    assert recovered.planning_mode == fresh.planning_mode
    assert recovered.preflight_message_indexes_complete is False
    assert recovery_sink.emissions == fresh_sink.emissions
    assert recovery_sink.finishes == fresh_sink.finishes
