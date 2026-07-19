from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from robata.adapters.fake_vision_model import DeterministicFakeVisionModelAdapter
from robata.application.mainline import (
    LocalMainlineConfig,
    LocalMainlinePipeline,
    MainlineRunError,
    MainlineRunErrorCode,
)
from robata.application.registered_video_export import PublishedRegisteredVideoExport
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.mainline import (
    ActionEventStatus,
    ActionEvidence,
    BoundaryRefinement,
    CameraPackage,
    CameraPackageStatus,
    MainlineBundle,
    MainlineStage,
    MaterializedFrame,
    NanosecondInterval,
    RunStatus,
    SamplingPurpose,
    SamplingStrategy,
    SamplingSummary,
    StageStatus,
    TemporalVisualPackage,
    VisionInferenceOutcome,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionTask,
)
from robata.contracts.video_export import MediaTimeMapping
from robata.contracts.video_export_v2 import (
    CameraVideoExportManifestV2,
    CameraVideoExportRecordV2,
)
from robata.ports.mainline import FrameMaterializationRequest

SECOND = 1_000_000_000
SOURCE_ORIGIN_NS = 1_710_000_000_000_000_000
FIXED_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _stable_uuid(domain: str, value: object) -> str:
    digest = semantic_sha256({"domain": domain, "value": value})
    return f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


class _StubFrameMaterializer:
    def __init__(self, *, missing_camera: CameraId | None = None) -> None:
        self.missing_camera = missing_camera
        self.requests: list[FrameMaterializationRequest] = []

    def materialize(self, request: FrameMaterializationRequest) -> TemporalVisualPackage:
        self.requests.append(request)
        package_id = _stable_uuid(
            "stub-package",
            {
                "window_id": request.window.window_id,
                "purpose": request.purpose.value,
                "rate": [request.rate_num, request.rate_den],
                "tolerance_ns": request.selection_tolerance_ns,
            },
        )
        timestamp_ns = request.window.interval.start_ns + request.window.interval.duration_ns // 2
        target_fps = request.rate_num / request.rate_den
        strategy = (
            SamplingStrategy.DENSE
            if request.purpose is SamplingPurpose.ACTION_DENSE
            else SamplingStrategy.UNIFORM
        )

        camera_packages: dict[CameraId, CameraPackage] = {}
        for source_index, camera_id in enumerate(CAMERA_IDS):
            if camera_id is self.missing_camera:
                frames: tuple[MaterializedFrame, ...] = ()
                status = CameraPackageStatus.NO_FRAME
                actual_fps = 0.0
                missed_targets = 1
                missing_reason = "deliberate test fixture gap"
            else:
                frame_id = _stable_uuid(
                    "stub-frame",
                    {"package_id": package_id, "camera_id": camera_id.value},
                )
                frames = (
                    MaterializedFrame(
                        frame_id=frame_id,
                        ordinal=0,
                        source_frame_index=source_index,
                        target_timestamp_ns=timestamp_ns,
                        aligned_timestamp_ns=timestamp_ns,
                        source_timestamp_ns=SOURCE_ORIGIN_NS + timestamp_ns,
                        delta_to_target_ns=0,
                        artifact_uri=(f"artifact://frames/{package_id}/{camera_id.value}.png"),
                        artifact_sha256=semantic_sha256(
                            {"frame_id": frame_id, "fixture": "mainline"}
                        ),
                        width=640,
                        height=480,
                        quality_flags=(),
                    ),
                )
                status = CameraPackageStatus.AVAILABLE
                actual_fps = target_fps
                missed_targets = 0
                missing_reason = None

            camera_packages[camera_id] = CameraPackage(
                camera_id=camera_id,
                status=status,
                source_video_uri=f"artifact://videos/{camera_id.value}.mp4",
                frames=frames,
                sampling=SamplingSummary(
                    strategy=strategy,
                    target_fps=target_fps,
                    actual_fps=actual_fps,
                    target_count=1,
                    actual_count=len(frames),
                    missed_targets=missed_targets,
                ),
                missing_reason=missing_reason,
            )

        cameras = SixCameraMap[CameraPackage](camera_packages)
        frame_count = sum(len(camera.frames) for camera in cameras.values())
        return TemporalVisualPackage(
            schema_version="1.0",
            package_id=package_id,
            content_sha256=semantic_sha256(
                {
                    "package_id": package_id,
                    "frame_ids": tuple(
                        frame.frame_id for camera in cameras.values() for frame in camera.frames
                    ),
                }
            ),
            mcap_id=request.window.mcap_id,
            window_id=request.window.window_id,
            purpose=request.purpose,
            interval=request.window.interval,
            cameras=cameras,
            frame_count_total=frame_count,
            producer_version="test-materializer-v1",
            created_at="2026-07-18T12:00:00Z",
        )


class _CountingFakeAdapter(DeterministicFakeVisionModelAdapter):
    def __init__(self, *, no_event: bool = False) -> None:
        super().__init__(no_event=no_event)
        self.requests: list[VisionInferenceRequest] = []

    def infer(
        self,
        request: VisionInferenceRequest,
        package: TemporalVisualPackage | None = None,
        artifact_root: Path | None = None,
    ) -> VisionInferenceOutcome:
        self.requests.append(request)
        return super().infer(request, package, artifact_root)


class _OutOfWindowFakeAdapter(_CountingFakeAdapter):
    def __init__(self, task: VisionTask) -> None:
        super().__init__()
        self.task = task

    def infer(
        self,
        request: VisionInferenceRequest,
        package: TemporalVisualPackage | None = None,
        artifact_root: Path | None = None,
    ) -> VisionInferenceOutcome:
        outcome = super().infer(request, package, artifact_root)
        if request.task is not self.task or not isinstance(outcome, VisionInferenceSuccess):
            return outcome

        outside = NanosecondInterval(
            start_ns=request.interval.end_ns + SECOND,
            end_ns=request.interval.end_ns + 4 * SECOND,
        )
        inner = NanosecondInterval(
            start_ns=outside.start_ns + SECOND,
            end_ns=outside.end_ns - SECOND,
        )
        output = outcome.output
        if isinstance(output, ActionEvidence):
            cameras = SixCameraMap(
                {
                    camera_id: claim.model_copy(
                        update={
                            "observed_interval": outside,
                            "event_interval": inner,
                        }
                    )
                    for camera_id, claim in output.cameras.items()
                }
            )
            hypotheses = tuple(
                hypothesis.model_copy(update={"interval": inner})
                for hypothesis in output.cross_view_hypotheses
            )
            output = output.model_copy(
                update={"cameras": cameras, "cross_view_hypotheses": hypotheses}
            )
        elif isinstance(output, BoundaryRefinement):
            cameras = SixCameraMap(
                {
                    camera_id: claim.model_copy(
                        update={
                            "observed_interval": outside,
                            "onset_interval": inner,
                            "offset_interval": inner,
                        }
                    )
                    for camera_id, claim in output.cameras.items()
                }
            )
            output = output.model_copy(update={"cameras": cameras})
        return outcome.model_copy(update={"output": output})


def _published_video_export(tmp_path: Path) -> PublishedRegisteredVideoExport:
    video_directory = tmp_path / "video-export"
    video_directory.mkdir(exist_ok=True)
    cameras = tuple(
        CameraVideoExportRecordV2.model_construct(
            camera_id=camera_id,
            export_first_observed_source_message_ns=SOURCE_ORIGIN_NS,
            export_last_observed_source_message_ns=SOURCE_ORIGIN_NS + 7 * SECOND,
            media_time_mapping=MediaTimeMapping.model_construct(
                last_duration=1,
                time_base_numerator=1,
                time_base_denominator=1,
            ),
        )
        for camera_id in CAMERA_IDS
    )
    manifest = CameraVideoExportManifestV2.model_construct(
        recording_identity="1" * 64,
        source_content_sha256="4" * 64,
        semantic_content_sha256="2" * 64,
        mapping_profile_artifact_id=_stable_uuid("fixture", "mapping"),
        alignment_id=None,
        cameras=cameras,
    )
    return PublishedRegisteredVideoExport(
        output_directory=video_directory,
        manifest=manifest,
        manifest_sha256="3" * 64,
        manifest_artifact_id=_stable_uuid("fixture", "manifest"),
        logical_key="mainline-pipeline-test-export",
        derivation_reused=False,
        materialized_view_reused=False,
    )


def _pipeline(
    materializer: _StubFrameMaterializer,
    adapter: _CountingFakeAdapter,
    *,
    config=None,
) -> LocalMainlinePipeline:
    return LocalMainlinePipeline(
        materializer,
        adapter,
        config=config,
        clock=lambda: FIXED_NOW,
        monotonic=lambda: 100.0,
    )


def _published_file_names(bundle: MainlineBundle) -> set[str]:
    names = {
        "action-events.json",
        "candidates.json",
        "mainline-bundle.json",
        "qa-aggregates.json",
        "run-report.json",
    }
    names.update(f"packages/{package.package_id}.json" for package in bundle.packages)
    for ordinal, request in enumerate(bundle.inference_requests):
        task_name = request.task.value.lower().replace("_", "-")
        prefix = f"inferences/{ordinal:02d}-{task_name}"
        names.add(f"{prefix}-request.json")
        names.add(f"{prefix}-outcome.json")
    return names


def _assert_atomic_publication(
    *,
    output_directory: Path,
    bundle: MainlineBundle,
    bundle_sha256: str,
) -> None:
    actual_names = {
        path.relative_to(output_directory).as_posix()
        for path in output_directory.rglob("*")
        if path.is_file()
    }
    assert actual_names == _published_file_names(bundle)
    bundle_bytes = (output_directory / "mainline-bundle.json").read_bytes()
    assert bundle_bytes == canonical_json_bytes(bundle)
    assert exact_bytes_sha256(bundle_bytes) == bundle_sha256
    assert (output_directory / "run-report.json").read_bytes() == canonical_json_bytes(
        bundle.report
    )
    assert (output_directory / "qa-aggregates.json").read_bytes() == canonical_json_bytes(
        bundle.qa_aggregates
    )
    assert tuple(output_directory.parent.glob(f".{output_directory.name}.partial-*")) == ()


def test_default_path_publishes_one_six_view_action_event_atomically(
    tmp_path: Path,
) -> None:
    materializer = _StubFrameMaterializer()
    adapter = _CountingFakeAdapter()
    output_directory = tmp_path / "mainline-default"

    published = _pipeline(materializer, adapter).run(
        _published_video_export(tmp_path), output_directory
    )
    bundle = published.bundle
    report = bundle.report

    assert published.output_directory == output_directory
    assert report.status is RunStatus.PRIMARY_COMPLETE
    assert (report.window_count, report.package_count) == (2, 2)
    assert (report.inference_attempt_count, report.fake_inference_attempt_count) == (5, 5)
    assert (report.inference_success_count, report.real_provider_request_count) == (5, 0)
    assert (report.candidate_count, report.event_count) == (1, 1)
    assert len(bundle.windows) == len(materializer.requests) == 2
    assert len(bundle.packages) == 2
    assert len(bundle.qa_aggregates) == 2
    assert report.source_recording_identity == "1" * 64
    assert report.source_content_sha256 == "4" * 64
    assert report.video_manifest_sha256 == "3" * 64
    assert tuple(request.task for request in adapter.requests) == (
        VisionTask.QA_COARSE,
        VisionTask.EVENT_PROPOSAL,
        VisionTask.QA_DENSE,
        VisionTask.ACTION_EVIDENCE,
        VisionTask.BOUNDARY_REFINEMENT,
    )
    assert bundle.inference_requests == tuple(adapter.requests)

    (event,) = bundle.events
    assert event.status is ActionEventStatus.FINAL
    assert tuple(event.camera_evidence.keys()) == CAMERA_IDS
    assert event.production_eligible is False
    assert all(evidence.frame_ids for evidence in event.camera_evidence.values())
    _assert_atomic_publication(
        output_directory=output_directory,
        bundle=bundle,
        bundle_sha256=published.bundle_sha256,
    )


def test_no_event_path_stops_after_two_fake_calls_and_skips_candidate_stages(
    tmp_path: Path,
) -> None:
    materializer = _StubFrameMaterializer()
    adapter = _CountingFakeAdapter(no_event=True)
    output_directory = tmp_path / "mainline-no-event"

    published = _pipeline(materializer, adapter).run(
        _published_video_export(tmp_path), output_directory
    )
    bundle = published.bundle
    report = bundle.report

    assert report.status is RunStatus.PRIMARY_COMPLETE_NO_EVENTS
    assert (report.window_count, report.package_count) == (1, 1)
    assert (report.inference_attempt_count, report.fake_inference_attempt_count) == (2, 2)
    assert tuple(request.task for request in adapter.requests) == (
        VisionTask.QA_COARSE,
        VisionTask.EVENT_PROPOSAL,
    )
    assert bundle.candidates == ()
    assert bundle.events == ()
    assert len(bundle.qa_aggregates) == 1
    assert (report.candidate_count, report.event_count) == (0, 0)
    stages = {stage.stage: stage for stage in report.stages}
    for stage in (
        MainlineStage.ACTION_EVIDENCE,
        MainlineStage.BOUNDARY_REFINEMENT,
        MainlineStage.FUSION,
    ):
        assert stages[stage].status is StageStatus.SKIPPED
        assert (stages[stage].succeeded, stages[stage].skipped) == (0, 1)
    _assert_atomic_publication(
        output_directory=output_directory,
        bundle=bundle,
        bundle_sha256=published.bundle_sha256,
    )


def test_model_failure_leaves_no_output_or_partial_directory(tmp_path: Path) -> None:
    materializer = _StubFrameMaterializer(missing_camera=CAMERA_IDS[-1])
    adapter = _CountingFakeAdapter()
    output_directory = tmp_path / "mainline-failure"

    with pytest.raises(MainlineRunError) as caught:
        _pipeline(materializer, adapter).run(_published_video_export(tmp_path), output_directory)

    assert caught.value.code is MainlineRunErrorCode.MODEL_INFERENCE_FAILED
    assert "FAKE_INPUT_MISSING_FRAMES" in str(caught.value)
    assert tuple(request.task for request in adapter.requests) == (VisionTask.QA_COARSE,)
    assert adapter.external_provider_requests == 0
    assert not output_directory.exists()
    assert tuple(tmp_path.glob(f".{output_directory.name}.partial-*")) == ()


@pytest.mark.parametrize(
    "task",
    [VisionTask.ACTION_EVIDENCE, VisionTask.BOUNDARY_REFINEMENT],
)
def test_out_of_window_dense_claim_is_rejected_before_fusion(
    task: VisionTask,
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / f"outside-{task.value.lower()}"
    adapter = _OutOfWindowFakeAdapter(task)

    with pytest.raises(MainlineRunError) as caught:
        _pipeline(_StubFrameMaterializer(), adapter).run(
            _published_video_export(tmp_path),
            output_directory,
        )

    assert caught.value.code is MainlineRunErrorCode.INVALID_MODEL_OUTPUT
    assert "outside the requested package interval" in str(caught.value)
    assert adapter.external_provider_requests == 0
    assert not output_directory.exists()
    assert tuple(tmp_path.glob(f".{output_directory.name}.partial-*")) == ()


def test_same_input_produces_the_same_key_ids_and_event_semantics(tmp_path: Path) -> None:
    video_export = _published_video_export(tmp_path)
    first = _pipeline(_StubFrameMaterializer(), _CountingFakeAdapter()).run(
        video_export, tmp_path / "mainline-first"
    )
    second = _pipeline(_StubFrameMaterializer(), _CountingFakeAdapter()).run(
        video_export, tmp_path / "mainline-second"
    )

    assert first.bundle.report.run_id == second.bundle.report.run_id
    assert tuple(window.window_id for window in first.bundle.windows) == tuple(
        window.window_id for window in second.bundle.windows
    )
    assert tuple(package.package_id for package in first.bundle.packages) == tuple(
        package.package_id for package in second.bundle.packages
    )
    assert tuple(request.inference_id for request in first.bundle.inference_requests) == tuple(
        request.inference_id for request in second.bundle.inference_requests
    )
    assert tuple(candidate.candidate_event_id for candidate in first.bundle.candidates) == tuple(
        candidate.candidate_event_id for candidate in second.bundle.candidates
    )
    assert first.bundle.events == second.bundle.events
    assert first.bundle_sha256 == second.bundle_sha256
    assert (first.output_directory / "mainline-bundle.json").read_bytes() == (
        second.output_directory / "mainline-bundle.json"
    ).read_bytes()


def test_parallel_independent_inference_preserves_canonical_results(tmp_path: Path) -> None:
    video_export = _published_video_export(tmp_path)
    serial = _pipeline(_StubFrameMaterializer(), _CountingFakeAdapter()).run(
        video_export, tmp_path / "mainline-serial"
    )
    parallel = _pipeline(
        _StubFrameMaterializer(),
        _CountingFakeAdapter(),
        config=LocalMainlineConfig(parallel_independent_inference=True),
    ).run(video_export, tmp_path / "mainline-parallel")

    assert tuple(request.task for request in parallel.bundle.inference_requests) == (
        VisionTask.QA_COARSE,
        VisionTask.EVENT_PROPOSAL,
        VisionTask.QA_DENSE,
        VisionTask.ACTION_EVIDENCE,
        VisionTask.BOUNDARY_REFINEMENT,
    )
    assert parallel.bundle.inference_requests == serial.bundle.inference_requests
    assert parallel.bundle.inference_outcomes == serial.bundle.inference_outcomes
    assert parallel.bundle.events == serial.bundle.events
    assert parallel.bundle_sha256 == serial.bundle_sha256


class _SerialOnlyFakeAdapter(_CountingFakeAdapter):
    supports_parallel_inference = False


def test_parallel_inference_requires_adapter_capability(tmp_path: Path) -> None:
    with pytest.raises(MainlineRunError) as caught:
        _pipeline(
            _StubFrameMaterializer(),
            _SerialOnlyFakeAdapter(),
            config=LocalMainlineConfig(parallel_independent_inference=True),
        ).run(_published_video_export(tmp_path), tmp_path / "mainline-unsupported-parallel")

    assert caught.value.code is MainlineRunErrorCode.INVALID_REQUEST
    assert "capability declaration" in str(caught.value)
    assert not (tmp_path / "mainline-unsupported-parallel").exists()


def test_parallel_config_requires_two_workers() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        LocalMainlineConfig(
            parallel_independent_inference=True,
            max_parallel_inference_workers=1,
        )
