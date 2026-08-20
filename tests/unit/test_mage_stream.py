from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robata.application.canonical.mage_stream import (
    AbsoluteNanosecondInterval,
    FfmpegCommandResult,
    MageStreamMaterializationError,
    MageStreamMaterializer,
    MageStreamPlanningError,
    MageStreamPolicy,
    MageStreamRecording,
    MageStreamSegmentationMode,
    build_perception_context_manifest,
    plan_keyframe_aligned_mage_stream,
    plan_mage_stream,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.perception_stream import CameraAbsenceReason


def _recording(source: Path, *, start_ns: int, end_ns: int) -> MageStreamRecording:
    return MageStreamRecording(
        recording_key="local-recording",
        recording_exact_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        interval=AbsoluteNanosecondInterval(start_ns, end_ns),
    )


def test_planner_partitions_absolute_timeline_and_expands_reasoning_to_full_segments(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"native-codec-input")
    start_ns = 1_000_000_000
    plan = plan_mage_stream(
        recording=_recording(source, start_ns=start_ns, end_ns=21_000_000_123),
        policy=MageStreamPolicy(
            scan_segment_duration_ns=8_000_000_000,
            reasoning_horizon_duration_ns=12_000_000_000,
        ),
    )

    assert [item.interval.as_projection() for item in plan.storage_segments] == [
        {"start_ns": 1_000_000_000, "end_ns": 9_000_000_000},
        {"start_ns": 9_000_000_000, "end_ns": 17_000_000_000},
        {"start_ns": 17_000_000_000, "end_ns": 21_000_000_123},
    ]
    for left, right in zip(plan.storage_segments, plan.storage_segments[1:], strict=False):
        assert left.interval.end_ns == right.interval.start_ns
    final_context = plan.reasoning_contexts[-1]
    assert final_context.reasoning_horizon == AbsoluteNanosecondInterval(
        9_000_000_123,
        21_000_000_123,
    )
    assert final_context.materialized_interval == AbsoluteNanosecondInterval(
        9_000_000_000,
        21_000_000_123,
    )
    assert [item.ordinal for item in final_context.ordered_segments] == [1, 2]


def test_planner_rejects_nonpositive_intervals_and_nonpositive_policy() -> None:
    digest = hashlib.sha256(b"recording").hexdigest()
    with pytest.raises(MageStreamPlanningError, match="nonempty"):
        AbsoluteNanosecondInterval(50, 50)
    with pytest.raises(MageStreamPlanningError, match="positive"):
        MageStreamPolicy(scan_segment_duration_ns=0)
    with pytest.raises(MageStreamPlanningError, match="nonempty"):
        MageStreamRecording(
            recording_key="recording",
            recording_exact_sha256=digest,
            interval=AbsoluteNanosecondInterval(10, 9),
        )


def test_materializer_uses_ffmpeg_stream_copy_and_context_manifest_is_honest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"codec-byte-source")
    plan = plan_mage_stream(
        recording=_recording(source, start_ns=0, end_ns=16_000_000_000),
        policy=MageStreamPolicy(
            scan_segment_duration_ns=8_000_000_000,
            reasoning_horizon_duration_ns=16_000_000_000,
        ),
    )
    commands: list[tuple[str, ...]] = []

    def fake_ffmpeg(command: tuple[str, ...]) -> FfmpegCommandResult:
        commands.append(command)
        destination = Path(command[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"stream-copy-{len(commands)}".encode())
        return FfmpegCommandResult(returncode=0)

    materializer = MageStreamMaterializer(command_runner=fake_ffmpeg)
    storage = materializer.materialize_storage_segments(
        plan=plan,
        source_path=source,
        output_root=tmp_path / "output",
    )
    context_media = materializer.materialize_reasoning_context(
        context=plan.reasoning_contexts[-1],
        camera_id=CameraId.CAM_03,
        storage_segments=storage,
        output_root=tmp_path / "output",
    )
    perception_context = build_perception_context_manifest(
        plan=plan,
        context=plan.reasoning_contexts[-1],
        materialized_context=context_media,
        codec_policy_version="mage-video-codec-policy-v1",
    )

    assert len(commands) == 3
    for command in commands:
        assert (command[command.index("-c")], command[command.index("-c") + 1]) == ("-c", "copy")
        assert "-c:v" not in command
        assert "-vf" not in command
    assert commands[-1][commands[-1].index("-f") + 1] == "concat"
    segment_receipts = sorted((tmp_path / "output" / "segments").glob("*.receipt.json"))
    context_receipts = sorted((tmp_path / "output" / "contexts").glob("*.receipt.json"))
    assert len(segment_receipts) == len(plan.storage_segments)
    assert len(context_receipts) == 1
    assert all(
        path.is_file() and not path.is_symlink() for path in segment_receipts + context_receipts
    )

    replayed_storage = materializer.materialize_storage_segments(
        plan=plan,
        source_path=source,
        output_root=tmp_path / "output",
    )
    replayed_context = materializer.materialize_reasoning_context(
        context=plan.reasoning_contexts[-1],
        camera_id=CameraId.CAM_03,
        storage_segments=replayed_storage,
        output_root=tmp_path / "output",
    )
    assert len(commands) == 3
    assert replayed_context.content_exact_sha256 == context_media.content_exact_sha256
    selected = perception_context.cameras[CameraId.CAM_03]
    assert selected.available is True
    assert selected.selected_for_inference is True
    assert selected.codec_stream_exact_sha256 == context_media.content_exact_sha256
    for camera_id in CAMERA_IDS:
        if camera_id is CameraId.CAM_03:
            continue
        binding = perception_context.cameras[camera_id]
        assert binding.available is False
        assert binding.selected_for_inference is False
        assert binding.absence_reason is CameraAbsenceReason.UNAVAILABLE


def test_keyframe_aligned_planner_uses_exact_source_pts_without_overlap(tmp_path: Path) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"native-codec-input")
    recording = _recording(source, start_ns=0, end_ns=20_500_000_000)
    plan = plan_keyframe_aligned_mage_stream(
        recording=recording,
        policy=MageStreamPolicy(
            scan_segment_duration_ns=8_000_000_000,
            reasoning_horizon_duration_ns=8_000_000_000,
            segmentation_mode=MageStreamSegmentationMode.KEYFRAME_ALIGNED,
            keyframe_alignment_tolerance_ns=1_000_000,
        ),
        keyframe_offsets_ns=(0, 8_000_009_000, 15_999_996_000),
    )

    assert [item.interval.as_projection() for item in plan.storage_segments] == [
        {"start_ns": 0, "end_ns": 8_000_009_000},
        {"start_ns": 8_000_009_000, "end_ns": 15_999_996_000},
        {"start_ns": 15_999_996_000, "end_ns": 20_500_000_000},
    ]
    assert all(
        context.ordered_segments == (segment,)
        for context, segment in zip(plan.reasoning_contexts, plan.storage_segments, strict=True)
    )


def test_verified_materializer_rejects_packet_preroll_and_uses_output_seek(
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"native-codec-input")
    plan = plan_keyframe_aligned_mage_stream(
        recording=_recording(source, start_ns=0, end_ns=8_000_000_000),
        policy=MageStreamPolicy(
            segmentation_mode=MageStreamSegmentationMode.KEYFRAME_ALIGNED,
            keyframe_alignment_tolerance_ns=1_000_000,
        ),
        keyframe_offsets_ns=(0,),
    )
    commands: list[tuple[str, ...]] = []

    def fake_ffmpeg(command: tuple[str, ...]) -> FfmpegCommandResult:
        commands.append(command)
        Path(command[-1]).write_bytes(b"segment")
        return FfmpegCommandResult(returncode=0)

    good_packets = FfmpegCommandResult(
        returncode=0,
        stdout=(
            '{"packets":['
            '{"pts_time":"0.000000","dts_time":"0.000000","flags":"K__"},'
            '{"pts_time":"7.966667","dts_time":"7.966667","flags":"___"}'
            "]}"
        ),
    )
    materializer = MageStreamMaterializer(
        command_runner=fake_ffmpeg,
        probe_runner=lambda _command: good_packets,
        verify_packet_boundaries=True,
    )
    materializer.materialize_storage_segment(
        plan=plan,
        source_path=source,
        segment=plan.storage_segments[0],
        output_root=tmp_path / "good",
    )
    command = commands[0]
    assert command.index("-i") < command.index("-ss")
    assert command[command.index("-avoid_negative_ts") + 1] == "disabled"

    bad_packets = FfmpegCommandResult(
        returncode=0,
        stdout=('{"packets":[{"pts_time":"-0.500000","dts_time":"-0.500000","flags":"KD_"}]}'),
    )
    with pytest.raises(MageStreamMaterializationError, match="negative decoder pre-roll"):
        MageStreamMaterializer(
            command_runner=fake_ffmpeg,
            probe_runner=lambda _command: bad_packets,
            verify_packet_boundaries=True,
        ).materialize_storage_segment(
            plan=plan,
            source_path=source,
            segment=plan.storage_segments[0],
            output_root=tmp_path / "bad",
        )


def test_materializer_rejects_missing_or_tampered_receipts(tmp_path: Path) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"codec-byte-source")
    plan = plan_mage_stream(
        recording=_recording(source, start_ns=0, end_ns=8_000_000_000),
        policy=MageStreamPolicy(),
    )
    commands: list[tuple[str, ...]] = []

    def fake_ffmpeg(command: tuple[str, ...]) -> FfmpegCommandResult:
        commands.append(command)
        destination = Path(command[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"stream-copy")
        return FfmpegCommandResult(returncode=0)

    materializer = MageStreamMaterializer(command_runner=fake_ffmpeg)
    output_root = tmp_path / "output"
    materialized = materializer.materialize_storage_segment(
        plan=plan,
        source_path=source,
        segment=plan.storage_segments[0],
        output_root=output_root,
    )
    receipt_path = materialized.durable_path.with_name(
        materialized.durable_path.name + ".receipt.json"
    )
    receipt_path.unlink()
    with pytest.raises(MageStreamMaterializationError, match="receipt is missing"):
        materializer.materialize_storage_segment(
            plan=plan,
            source_path=source,
            segment=plan.storage_segments[0],
            output_root=output_root,
        )

    materializer.materialize_storage_segment(
        plan=plan,
        source_path=source,
        segment=plan.storage_segments[0],
        output_root=tmp_path / "tampered",
    )
    tampered = next((tmp_path / "tampered" / "segments").glob("*.mp4"))
    tampered.write_bytes(b"tampered")
    with pytest.raises(MageStreamMaterializationError, match="bytes do not match"):
        materializer.materialize_storage_segment(
            plan=plan,
            source_path=source,
            segment=plan.storage_segments[0],
            output_root=tmp_path / "tampered",
        )


def test_materializer_rejects_receipt_lineage_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"codec-byte-source")
    plan = plan_mage_stream(
        recording=_recording(source, start_ns=0, end_ns=8_000_000_000),
        policy=MageStreamPolicy(),
    )

    def fake_ffmpeg(command: tuple[str, ...]) -> FfmpegCommandResult:
        destination = Path(command[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"stream-copy")
        return FfmpegCommandResult(returncode=0)

    materializer = MageStreamMaterializer(command_runner=fake_ffmpeg)
    output_root = tmp_path / "output"
    materialized = materializer.materialize_storage_segment(
        plan=plan,
        source_path=source,
        segment=plan.storage_segments[0],
        output_root=output_root,
    )
    receipt_path = materialized.durable_path.with_name(
        materialized.durable_path.name + ".receipt.json"
    )
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    document["recording_exact_sha256"] = hashlib.sha256(b"different-source").hexdigest()
    receipt_path.write_bytes(canonical_json_bytes(document) + b"\n")
    with pytest.raises(MageStreamMaterializationError, match="lineage mismatch"):
        materializer.materialize_storage_segment(
            plan=plan,
            source_path=source,
            segment=plan.storage_segments[0],
            output_root=output_root,
        )


def test_materializer_rejects_missing_or_tampered_context_receipts(tmp_path: Path) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"codec-byte-source")
    plan = plan_mage_stream(
        recording=_recording(source, start_ns=0, end_ns=16_000_000_000),
        policy=MageStreamPolicy(
            scan_segment_duration_ns=8_000_000_000,
            reasoning_horizon_duration_ns=16_000_000_000,
        ),
    )

    def fake_ffmpeg(command: tuple[str, ...]) -> FfmpegCommandResult:
        destination = Path(command[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"output-{destination.parent.name}".encode())
        return FfmpegCommandResult(returncode=0)

    materializer = MageStreamMaterializer(command_runner=fake_ffmpeg)
    output_root = tmp_path / "output"
    storage = materializer.materialize_storage_segments(
        plan=plan,
        source_path=source,
        output_root=output_root,
    )
    context = materializer.materialize_reasoning_context(
        context=plan.reasoning_contexts[-1],
        camera_id=CameraId.CAM_03,
        storage_segments=storage,
        output_root=output_root,
    )
    receipt_path = context.durable_path.with_name(context.durable_path.name + ".receipt.json")
    receipt_path.unlink()
    with pytest.raises(MageStreamMaterializationError, match="receipt is missing"):
        materializer.materialize_reasoning_context(
            context=plan.reasoning_contexts[-1],
            camera_id=CameraId.CAM_03,
            storage_segments=storage,
            output_root=output_root,
        )

    context_root = tmp_path / "tampered"
    storage_tampered = materializer.materialize_storage_segments(
        plan=plan,
        source_path=source,
        output_root=context_root,
    )
    tampered_context = materializer.materialize_reasoning_context(
        context=plan.reasoning_contexts[-1],
        camera_id=CameraId.CAM_03,
        storage_segments=storage_tampered,
        output_root=context_root,
    )
    tampered_context.durable_path.write_bytes(b"tampered-context")
    with pytest.raises(MageStreamMaterializationError, match="bytes do not match"):
        materializer.materialize_reasoning_context(
            context=plan.reasoning_contexts[-1],
            camera_id=CameraId.CAM_03,
            storage_segments=storage_tampered,
            output_root=context_root,
        )
