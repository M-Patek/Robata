from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("av")

from robata.adapters.mcap_inspector import OfficialMcapInspector
from robata.adapters.mcap_single_pass import McapSinglePassH264Tee
from robata.adapters.pyav_mp4_exporter import PyAvH264Mp4Exporter
from robata.application.canonical import single_pass_video
from robata.application.canonical.bounded_media import (
    BoundedMediaPolicy,
    PlannerEmission,
    PlannerFinish,
)
from robata.application.canonical.single_pass_video import (
    SPOOL_SEAL_FILENAME,
    DurableSinglePassVideoProducer,
    SinglePassVideoProductionError,
)
from robata.contracts import CAMERA_IDS, CameraId, SixCameraMap, canonical_json_bytes
from robata.ports import ChannelInspection, McapInspection
from tests.support.six_camera_mcap import SIX_CAMERA_TOPICS, write_six_camera_mcap


class _CountingTee(McapSinglePassH264Tee):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def traverse(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().traverse(*args, **kwargs)  # type: ignore[arg-type]


class _RecordingPlanningSink:
    def __init__(self) -> None:
        self.emissions: list[PlannerEmission] = []
        self.finishes: list[PlannerFinish] = []

    def append_emission(self, emission: PlannerEmission) -> None:
        self.emissions.append(emission)

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
        source_scope_digest=inspection.source_sha256,
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
