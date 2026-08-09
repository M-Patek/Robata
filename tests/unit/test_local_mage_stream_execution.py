from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from robata.application.canonical.mage_stream import (
    AbsoluteNanosecondInterval,
    FfmpegCommandResult,
    MageStreamMaterializer,
    MageStreamPolicy,
    MageStreamRecording,
    build_perception_context_manifest,
    plan_mage_stream,
)
from robata.application.canonical.mage_stream_execution import (
    LocalMageStreamExecutionError,
    LocalMageStreamExecutionProfile,
    LocalMaterializedSegmentResolver,
    LocalMediaHealthScanner,
    execute_local_mage_stream,
)
from robata.contracts.cameras import CameraId
from robata.perception.durable_scheduler import SQLitePerceptionWorkScheduler
from robata.perception.fusion import PerceptionFusionEngine, PerceptionFusionPolicy
from robata.perception.pipeline import (
    LocalPerceptionArtifactStore,
    PerceptionStage,
    StreamPerceptionPipeline,
)
from robata.perception.projectors import (
    EventProjector,
    EvidenceProjector,
    MediaHealthDisposition,
    QaProjector,
)
from robata.perception.tracking import EventTrackPolicy, EventTrackReconciler
from tests.support.perception_stream import make_observation


@dataclass
class _RecordingProvider:
    events: list[str]
    fail_at_ordinal: int | None = None
    start_confidence: float = 0.86
    end_confidence: float = 0.82

    def observe(self, context):  # type: ignore[no-untyped-def]
        ordinal = context.focus_segment_ordinal
        self.events.append(f"observe-{ordinal}")
        if self.fail_at_ordinal == ordinal:
            raise RuntimeError(f"planned provider failure {ordinal}")
        start_ns = context.context_interval.start_ns + 1
        return make_observation(
            context=context,
            local_ref=f"observation-{ordinal}",
            action_start_ns=start_ns,
            action_end_ns=start_ns + 1,
            start_confidence=self.start_confidence,
            end_confidence=self.end_confidence,
            inference_artifact_seed=("stream", ordinal),
        )


class _OverlapProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.first_started = Event()
        self.second_started = Event()

    def observe(self, context):  # type: ignore[no-untyped-def]
        ordinal = context.focus_segment_ordinal
        self.events.append(f"observe-{ordinal}")
        if ordinal == 0:
            self.first_started.set()
            if not self.second_started.wait(timeout=5.0):
                raise RuntimeError("second observation did not overlap the first")
        elif ordinal == 1:
            if not self.first_started.is_set():
                raise RuntimeError("second observation started before the first")
            self.second_started.set()
        start_ns = context.context_interval.start_ns + 1
        return make_observation(
            context=context,
            local_ref=f"observation-{ordinal}",
            action_start_ns=start_ns,
            action_end_ns=start_ns + 1,
            inference_artifact_seed=("overlap", ordinal),
        )


class _RecordingMaterializer(MageStreamMaterializer):
    def __init__(self, *, events: list[str]) -> None:
        super().__init__(command_runner=_ffmpeg_success)
        self._events = events

    def materialize_storage_segment(self, **kwargs):  # type: ignore[no-untyped-def]
        segment = kwargs["segment"]
        self._events.append(f"materialize-{segment.ordinal}")
        return super().materialize_storage_segment(**kwargs)


def _ffmpeg_success(command: tuple[str, ...]) -> FfmpegCommandResult:
    destination = Path(command[-1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(f"native-{destination.name}".encode())
    return FfmpegCommandResult(returncode=0, stdout="", stderr="")


def _ffprobe_success(_command: tuple[str, ...]) -> FfmpegCommandResult:
    return FfmpegCommandResult(
        returncode=0, stdout='{"streams":[{"codec_type":"video"}]}', stderr=""
    )


def _plan(source: Path, *, horizon_ns: int = 8_000_000_000):
    return plan_mage_stream(
        recording=MageStreamRecording(
            recording_key="local-stream",
            recording_exact_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            interval=AbsoluteNanosecondInterval(0, 24_000_000_000),
        ),
        policy=MageStreamPolicy(
            scan_segment_duration_ns=8_000_000_000,
            reasoning_horizon_duration_ns=horizon_ns,
        ),
    )


def _pipeline(
    provider: _RecordingProvider,
    root: Path,
    *,
    fusion_policy: PerceptionFusionPolicy | None = None,
) -> StreamPerceptionPipeline:
    return StreamPerceptionPipeline(
        provider=provider,
        qa_projector=QaProjector(),
        event_projector=EventProjector(),
        evidence_projector=EvidenceProjector(),
        reconciler=EventTrackReconciler(EventTrackPolicy(version="test-track-policy-v1")),
        fusion_engine=PerceptionFusionEngine(
            fusion_policy or PerceptionFusionPolicy(version="test-fusion-policy-v1")
        ),
        refine_policy_version="test-refine-policy-v1",
        refine_prompt_version="test-refine-prompt-v1",
        artifact_sink=LocalPerceptionArtifactStore(root / "perception-cas"),
    )


def test_execution_materializes_scans_and_observes_each_focus_segment_in_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"local-native-video")
    plan = _plan(source)
    events: list[str] = []
    provider = _RecordingProvider(events)
    artifact_store = LocalPerceptionArtifactStore(tmp_path / "reports")

    result = execute_local_mage_stream(
        plan=plan,
        source_path=source,
        selected_camera=CameraId.CAM_01,
        materializer=_RecordingMaterializer(events=events),
        codec_policy_version="mage-video-codec-policy-v1",
        resolver=LocalMaterializedSegmentResolver(),
        pipeline=_pipeline(provider, tmp_path),
        artifact_store=artifact_store,
        media_health_scanner=LocalMediaHealthScanner(command_runner=_ffprobe_success),
        materialization_root=tmp_path / "materialized",
        max_inflight_observations=1,
    )

    assert events == [
        "materialize-0",
        "observe-0",
        "materialize-1",
        "observe-1",
        "materialize-2",
        "observe-2",
    ]
    assert result.queue_depth == 1
    assert result.execution_profile is LocalMageStreamExecutionProfile.SERIAL_NATIVE_V1
    assert result.timing.profile is LocalMageStreamExecutionProfile.SERIAL_NATIVE_V1
    assert result.timing.max_observations_in_flight == 1
    assert result.timing.end_to_end_realtime_factor > 0.0
    assert len(result.contexts) == 3
    assert all(item.normal_model_call_count == 1 for item in result.contexts)
    assert all(item.refinement_model_call_count == 0 for item in result.contexts)
    assert all(item.persisted_report_exact_sha256 is not None for item in result.contexts)
    assert result.run_manifest is not None
    run_manifest = json.loads(
        artifact_store.read(
            kind=result.run_manifest.kind, logical_key=result.run_manifest.logical_key
        )
    )
    assert run_manifest["manifest_version"] == "local-mage-stream-run-manifest-v2"
    assert run_manifest["plan_semantic_sha256"] == plan.plan_semantic_sha256
    assert run_manifest["execution_profile"] == "SERIAL_NATIVE_V1"
    assert run_manifest["timing"]["max_observations_in_flight"] == 1
    assert result.pipeline_result.terminal_artifacts is not None
    assert [item["persisted_report_exact_sha256"] for item in run_manifest["contexts"]] == [
        item.persisted_report_exact_sha256 for item in result.contexts
    ]
    assert run_manifest["terminal_artifacts"]["terminal_manifest"]["exact_sha256"] == (
        result.pipeline_result.terminal_artifacts.terminal_manifest.exact_sha256
    )
    assert run_manifest["accepted_inference_bindings"] == []
    assert run_manifest["durable_execution"] is None
    stages = {item.stage: item for item in result.pipeline_result.stage_measurements}
    assert stages[PerceptionStage.MEDIA_SCAN].invocation_count == 3
    assert stages[PerceptionStage.PERCEPTION_OBSERVE].invocation_count == 3
    assert stages[PerceptionStage.PERCEPTION_REFINE].invocation_count == 0
    assert len(tuple((tmp_path / "reports" / "local-stream-context-report").rglob("*.json"))) == 3


def test_execution_consumes_durable_vnext_scheduler_and_records_stage_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"local-native-video")
    plan = _plan(source, horizon_ns=8_000_000_000)
    scheduler = SQLitePerceptionWorkScheduler(tmp_path / "perception-vnext.sqlite3")
    artifact_store = LocalPerceptionArtifactStore(tmp_path / "reports")
    result = execute_local_mage_stream(
        plan=plan,
        source_path=source,
        selected_camera=CameraId.CAM_01,
        materializer=_RecordingMaterializer(events=[]),
        codec_policy_version="mage-video-codec-policy-v1",
        resolver=LocalMaterializedSegmentResolver(),
        pipeline=_pipeline(
            _RecordingProvider([]),
            tmp_path,
            fusion_policy=PerceptionFusionPolicy(
                version="test-fusion-policy-v1",
                minimum_supporting_cameras=1,
            ),
        ),
        artifact_store=artifact_store,
        media_health_scanner=LocalMediaHealthScanner(command_runner=_ffprobe_success),
        materialization_root=tmp_path / "materialized",
        max_inflight_observations=1,
        durable_scheduler=scheduler,
    )

    assert result.durable_execution is not None
    assert result.run_manifest is not None
    durable = result.durable_execution
    manifest = json.loads(
        artifact_store.read(
            kind=result.run_manifest.kind,
            logical_key=result.run_manifest.logical_key,
        )
    )
    assert manifest["durable_execution"]["run_key"] == durable.run.run_key
    assert manifest["durable_execution"]["finalization_state"] == (durable.finalization_state)
    assert durable.run.plan_semantic_sha256 == plan.plan_semantic_sha256
    assert durable.snapshot.run.run_key == durable.run.run_key
    counts = {item.stage: item for item in durable.snapshot.stage_counts}
    assert counts[PerceptionStage.MEDIA_SCAN].succeeded == len(plan.reasoning_contexts)
    assert counts[PerceptionStage.PERCEPTION_OBSERVE].succeeded == len(plan.reasoning_contexts)
    assert counts[PerceptionStage.OBSERVATION_PROJECT].succeeded == len(plan.reasoning_contexts)
    assert counts[PerceptionStage.TEMPORAL_RECONCILE].succeeded == len(plan.reasoning_contexts)
    assert counts[PerceptionStage.FUSION].succeeded == len(result.pipeline_result.fusion_decisions)
    assert counts[PerceptionStage.PERCEPTION_REFINE].planned == 0
    assert durable.finalization_state == "SUCCEEDED"
    assert durable.run.derived_work_sealed is True
    assert counts[PerceptionStage.FINALIZE].succeeded == 1


def test_execution_registers_request_only_refinement_without_sealing_or_finalizing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"local-native-video")
    plan = _plan(source, horizon_ns=8_000_000_000)
    scheduler = SQLitePerceptionWorkScheduler(tmp_path / "perception-vnext.sqlite3")
    result = execute_local_mage_stream(
        plan=plan,
        source_path=source,
        selected_camera=CameraId.CAM_01,
        materializer=_RecordingMaterializer(events=[]),
        codec_policy_version="mage-video-codec-policy-v1",
        resolver=LocalMaterializedSegmentResolver(),
        pipeline=_pipeline(
            _RecordingProvider([], start_confidence=0.2, end_confidence=0.2),
            tmp_path,
            fusion_policy=PerceptionFusionPolicy(
                version="test-fusion-policy-v1",
                minimum_supporting_cameras=1,
            ),
        ),
        artifact_store=LocalPerceptionArtifactStore(tmp_path / "reports"),
        media_health_scanner=LocalMediaHealthScanner(command_runner=_ffprobe_success),
        materialization_root=tmp_path / "materialized",
        max_inflight_observations=1,
        durable_scheduler=scheduler,
    )

    assert result.pipeline_result.refinement_model_call_count == 0
    assert result.durable_execution is not None
    durable = result.durable_execution
    counts = {item.stage: item for item in durable.snapshot.stage_counts}
    assert durable.finalization_state == "PENDING_REFINEMENT"
    assert durable.run.derived_work_sealed is False
    assert durable.pending_refinement_work_item_ids
    assert counts[PerceptionStage.FUSION].succeeded == len(result.pipeline_result.fusion_decisions)
    assert counts[PerceptionStage.PERCEPTION_REFINE].ready == len(
        durable.pending_refinement_work_item_ids
    )
    assert counts[PerceptionStage.PERCEPTION_REFINE].succeeded == 0
    assert counts[PerceptionStage.FINALIZE].planned == 1
    assert counts[PerceptionStage.FINALIZE].succeeded == 0
    fusion_ids = {
        item.work_item_id
        for item in scheduler.items_for_run(durable.run.run_key)
        if item.stage is PerceptionStage.FUSION
    }
    for work_item_id in durable.pending_refinement_work_item_ids:
        assert set(scheduler.dependencies(work_item_id)) <= fusion_ids
        assert set(scheduler.dependencies(work_item_id))


def test_media_health_is_measured_only_from_file_hash_and_ffprobe_facts(tmp_path: Path) -> None:
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"local-native-video")
    plan = _plan(source)
    events: list[str] = []
    materializer = _RecordingMaterializer(events=events)
    segment = materializer.materialize_storage_segment(
        plan=plan,
        source_path=source,
        segment=plan.storage_segments[0],
        output_root=tmp_path / "materialized",
    )
    materialized_context = materializer.materialize_reasoning_context(
        context=plan.reasoning_contexts[0],
        camera_id=CameraId.CAM_01,
        storage_segments=(segment,),
        output_root=tmp_path / "materialized",
    )
    context = build_perception_context_manifest(
        plan=plan,
        context=plan.reasoning_contexts[0],
        materialized_context=materialized_context,
        codec_policy_version="mage-video-codec-policy-v1",
    )

    health = LocalMediaHealthScanner(command_runner=_ffprobe_success).scan(
        context=context,
        materialized_context=materialized_context,
    )

    assert health.cameras[CameraId.CAM_01].disposition is MediaHealthDisposition.HEALTHY
    assert health.cameras[CameraId.CAM_01].issue_codes == ()
    assert health.cameras[CameraId.CAM_02].disposition is MediaHealthDisposition.UNAVAILABLE
    assert health.cameras[CameraId.CAM_02].issue_codes == ("CAMERA_UNAVAILABLE",)


def test_execution_stops_producer_after_provider_failure_without_materializing_ahead(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"local-native-video")
    plan = _plan(source)
    events: list[str] = []
    provider = _RecordingProvider(events, fail_at_ordinal=1)

    with pytest.raises(RuntimeError, match="planned provider failure 1"):
        execute_local_mage_stream(
            plan=plan,
            source_path=source,
            selected_camera=CameraId.CAM_01,
            materializer=_RecordingMaterializer(events=events),
            codec_policy_version="mage-video-codec-policy-v1",
            resolver=LocalMaterializedSegmentResolver(),
            pipeline=_pipeline(provider, tmp_path),
            media_health_scanner=LocalMediaHealthScanner(command_runner=_ffprobe_success),
            materialization_root=tmp_path / "materialized",
            max_inflight_observations=1,
        )

    assert events == ["materialize-0", "observe-0", "materialize-1", "observe-1"]


def test_execution_rejects_overlapping_reasoning_horizon_before_materialization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"local-native-video")
    plan = _plan(source, horizon_ns=16_000_000_000)
    events: list[str] = []

    with pytest.raises(LocalMageStreamExecutionError, match="requires reasoning_horizon"):
        execute_local_mage_stream(
            plan=plan,
            source_path=source,
            selected_camera=CameraId.CAM_01,
            materializer=_RecordingMaterializer(events=events),
            codec_policy_version="mage-video-codec-policy-v1",
            resolver=LocalMaterializedSegmentResolver(),
            pipeline=_pipeline(_RecordingProvider(events), tmp_path),
            media_health_scanner=LocalMediaHealthScanner(command_runner=_ffprobe_success),
            materialization_root=tmp_path / "materialized",
        )

    assert events == []


def test_execution_supports_two_inflight_observations_with_ordered_consumption(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cam_01.mp4"
    source.write_bytes(b"local-native-video")
    plan = _plan(source)
    events: list[str] = []
    result = execute_local_mage_stream(
        plan=plan,
        source_path=source,
        selected_camera=CameraId.CAM_01,
        materializer=_RecordingMaterializer(events=events),
        codec_policy_version="mage-video-codec-policy-v1",
        resolver=LocalMaterializedSegmentResolver(),
        pipeline=_pipeline(_OverlapProvider(events), tmp_path),
        media_health_scanner=LocalMediaHealthScanner(command_runner=_ffprobe_success),
        materialization_root=tmp_path / "materialized",
        max_inflight_observations=2,
    )

    assert result.queue_depth == 2
    assert result.execution_profile is LocalMageStreamExecutionProfile.BOUNDED_PREFETCH_NATIVE_V1
    assert result.timing.profile is LocalMageStreamExecutionProfile.BOUNDED_PREFETCH_NATIVE_V1
    assert result.timing.max_observations_in_flight == 2
    assert result.timing.preparation_observation_overlap_seconds > 0.0
    assert result.timing.observation_parallelism_factor > 1.0
    assert [item.focus_segment_ordinal for item in result.contexts] == [0, 1, 2]
    assert result.contexts[1].observation_started_offset_seconds < (
        result.contexts[0].observation_completed_offset_seconds
    )
    assert result.pipeline_result.normal_model_call_count == 3
