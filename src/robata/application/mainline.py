"""Synchronous local video-to-action mainline using a swappable model port."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import median_low
from typing import Any, Never

from pydantic import ValidationError

from robata.alignment.rational_time import round_half_even
from robata.application.registered_video_export import PublishedRegisteredVideoExport
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import (
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
from robata.contracts.mainline import (
    ActionEventStatus,
    ActionEvidence,
    BoundaryRefinement,
    BoundaryStatus,
    CameraEventProvenance,
    CameraEvidenceStatus,
    CameraQAResult,
    CameraQAStatus,
    CandidateEvent,
    CandidateEventStatus,
    EventProposal,
    EventProposalOutput,
    FusedActionEvent,
    FusionAdjudicationOutput,
    InferenceCameraInput,
    MainlineBundle,
    MainlineRunReport,
    MainlineStage,
    QAOutput,
    QAResultAggregate,
    RecordingQAStatus,
    RunStatus,
    SamplingPurpose,
    StageReport,
    StageStatus,
    TemporalVisualPackage,
    TemporalWindow,
    VisionInferenceFailure,
    VisionInferenceOutcome,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionTask,
)
from robata.ports.mainline import (
    FrameMaterializationError,
    FrameMaterializationRequest,
    FrameMaterializer,
    VisionModelAdapter,
)
from robata.tempfiles import make_staging_directory

PIPELINE_VERSION = "local-mainline-v0"
FUSION_POLICY_VERSION = "deterministic-median-v0"
QA_POLICY_VERSION = "six-camera-local-v0"
ONTOLOGY_VERSION = "local-interaction-v0"


class MainlineRunErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    FRAME_MATERIALIZATION_FAILED = "FRAME_MATERIALIZATION_FAILED"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    OUTPUT_IO_ERROR = "OUTPUT_IO_ERROR"
    ATOMIC_PUBLISH_FAILED = "ATOMIC_PUBLISH_FAILED"


class MainlineRunError(RuntimeError):
    def __init__(self, code: MainlineRunErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class LocalMainlineConfig:
    coarse_rate_num: int = 1
    coarse_rate_den: int = 1
    dense_rate_num: int = 2
    dense_rate_den: int = 1
    coarse_selection_tolerance_ns: int = 500_000_000
    dense_selection_tolerance_ns: int = 250_000_000
    dense_padding_ns: int = 1_000_000_000
    max_candidates: int = 1
    prompt_version: str = "local-fake-prompt-v0"
    output_contract_version: str = "1.0"
    timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        positive = (
            self.coarse_rate_num,
            self.coarse_rate_den,
            self.dense_rate_num,
            self.dense_rate_den,
            self.max_candidates,
            self.timeout_ms,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive
        ):
            raise ValueError("rates, max_candidates, and timeout_ms must be positive integers")
        nonnegative = (
            self.coarse_selection_tolerance_ns,
            self.dense_selection_tolerance_ns,
            self.dense_padding_ns,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in nonnegative
        ):
            raise ValueError("tolerances and padding must be nonnegative integers")
        if not self.prompt_version or not self.output_contract_version:
            raise ValueError("prompt and output contract versions must be nonempty")

    @property
    def semantic_projection(self) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            "coarse_rate": [self.coarse_rate_num, self.coarse_rate_den],
            "dense_rate": [self.dense_rate_num, self.dense_rate_den],
            "coarse_tolerance_ns": self.coarse_selection_tolerance_ns,
            "dense_tolerance_ns": self.dense_selection_tolerance_ns,
            "dense_padding_ns": self.dense_padding_ns,
            "max_candidates": self.max_candidates,
            "prompt_version": self.prompt_version,
            "output_contract_version": self.output_contract_version,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True, slots=True)
class PublishedMainlineRun:
    output_directory: Path
    bundle: MainlineBundle
    bundle_sha256: str


def _raise(code: MainlineRunErrorCode, message: str) -> Never:
    raise MainlineRunError(code, message)


def _stable_uuid(domain: str, value: Any) -> str:
    digest = semantic_sha256({"domain": domain, "value": value})
    return f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _elapsed_ms(started: float, monotonic: Callable[[], float]) -> int:
    return max(0, round((monotonic() - started) * 1_000))


def _contains(outer: NanosecondInterval, inner: NanosecondInterval) -> bool:
    return outer.start_ns <= inner.start_ns and inner.end_ns <= outer.end_ns


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(canonical_json_bytes(value))
        output.flush()
        os.fsync(output.fileno())


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LocalMainlinePipeline:
    """Run the complete local primary path after the existing six-video export."""

    def __init__(
        self,
        frame_materializer: FrameMaterializer,
        model_adapter: VisionModelAdapter,
        *,
        config: LocalMainlineConfig | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._frames = frame_materializer
        self._model = model_adapter
        self._config = config or LocalMainlineConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.perf_counter

    def run(
        self,
        video_export: PublishedRegisteredVideoExport,
        output_directory: Path,
    ) -> PublishedMainlineRun:
        if not isinstance(video_export, PublishedRegisteredVideoExport):
            _raise(MainlineRunErrorCode.INVALID_REQUEST, "video_export has the wrong type")
        if not isinstance(output_directory, Path) or output_directory.name in {"", ".", ".."}:
            _raise(MainlineRunErrorCode.INVALID_REQUEST, "output_directory is invalid")
        if self._model.provider != "fake":
            _raise(MainlineRunErrorCode.INVALID_REQUEST, "V0 requires the fake provider")

        try:
            target = Path(os.path.abspath(output_directory))
            if target.exists() or target.is_symlink():
                _raise(MainlineRunErrorCode.OUTPUT_EXISTS, f"output exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = make_staging_directory(target.parent, prefix=f".{target.name}.partial-")
        except MainlineRunError:
            raise
        except OSError as error:
            _raise(MainlineRunErrorCode.OUTPUT_IO_ERROR, f"cannot create staging: {error}")

        published = False
        try:
            bundle = self._execute(video_export, staging)
            bundle_bytes = canonical_json_bytes(bundle)
            _write_new(staging / "run-report.json", bundle.report)
            with (staging / "mainline-bundle.json").open("xb") as output:
                output.write(bundle_bytes)
                output.flush()
                os.fsync(output.fileno())
            _sync_directory(staging)
            try:
                staging.rename(target)
            except OSError as error:
                _raise(
                    MainlineRunErrorCode.ATOMIC_PUBLISH_FAILED,
                    f"cannot publish analysis directory: {error}",
                )
            published = True
            _sync_directory(target.parent)
            return PublishedMainlineRun(
                output_directory=target,
                bundle=bundle,
                bundle_sha256=exact_bytes_sha256(bundle_bytes),
            )
        except MainlineRunError:
            raise
        except FrameMaterializationError as error:
            _raise(
                MainlineRunErrorCode.FRAME_MATERIALIZATION_FAILED,
                f"{error.code.value}: {error}",
            )
        except (ValidationError, TypeError, ValueError) as error:
            _raise(
                MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                f"mainline contract validation failed: {error}",
            )
        except OSError as error:
            _raise(MainlineRunErrorCode.OUTPUT_IO_ERROR, f"analysis I/O failed: {error}")
        finally:
            if not published:
                with suppress(OSError):
                    shutil.rmtree(staging)

    def _execute(
        self,
        video_export: PublishedRegisteredVideoExport,
        staging: Path,
    ) -> MainlineBundle:
        run_started_at = self._clock()
        run_timer = self._monotonic()
        durations: dict[MainlineStage, int] = {stage: 0 for stage in MainlineStage}
        config_sha256 = semantic_sha256(self._config.semantic_projection)
        mcap_id = _stable_uuid("local-mainline-mcap", video_export.manifest.recording_identity)
        run_id = _stable_uuid(
            "local-mainline-run",
            {
                "manifest": video_export.manifest.semantic_content_sha256,
                "config": config_sha256,
                "provider": self._model.provider,
                "model_name": self._model.model_name,
                "model_version": self._model.model_version,
            },
        )
        coverage = self._recording_coverage(video_export)

        started = self._monotonic()
        root_window = TemporalWindow(
            schema_version="1.0",
            window_id=_stable_uuid(
                "coarse-window",
                {"mcap_id": mcap_id, "interval": coverage, "config": config_sha256},
            ),
            mcap_id=mcap_id,
            camera_mapping_run_id=None,
            alignment_id=video_export.manifest.alignment_id,
            requested_interval=coverage,
            interval=coverage,
            purpose=SamplingPurpose.QA_COARSE,
            parent_window_id=None,
            source_candidate_id=None,
            source_event_id=None,
            generation=0,
        )
        durations[MainlineStage.WINDOWING] += _elapsed_ms(started, self._monotonic)

        started = self._monotonic()
        coarse_package = self._materialize(
            video_export,
            staging,
            root_window,
            rate_num=self._config.coarse_rate_num,
            rate_den=self._config.coarse_rate_den,
            tolerance_ns=self._config.coarse_selection_tolerance_ns,
        )
        durations[MainlineStage.SAMPLING] += _elapsed_ms(started, self._monotonic)

        requests: list[VisionInferenceRequest] = []
        outcomes: list[VisionInferenceSuccess] = []
        started = self._monotonic()
        qa_request, qa_outcome = self._infer(
            run_id=run_id,
            package=coarse_package,
            artifact_root=staging,
            task=VisionTask.QA_COARSE,
            candidate_id=None,
        )
        requests.append(qa_request)
        outcomes.append(qa_outcome)
        durations[MainlineStage.QA_INFERENCE] += _elapsed_ms(started, self._monotonic)

        started = self._monotonic()
        qa_aggregate = self._aggregate_qa(coarse_package, qa_outcome)
        qa_aggregates = [qa_aggregate]
        durations[MainlineStage.QA_AGGREGATION] += _elapsed_ms(started, self._monotonic)

        started = self._monotonic()
        proposal_request, proposal_outcome = self._infer(
            run_id=run_id,
            package=coarse_package,
            artifact_root=staging,
            task=VisionTask.EVENT_PROPOSAL,
            candidate_id=None,
        )
        requests.append(proposal_request)
        outcomes.append(proposal_outcome)
        durations[MainlineStage.EVENT_PROPOSAL] += _elapsed_ms(started, self._monotonic)
        proposal_output = proposal_outcome.output
        if not isinstance(proposal_output, EventProposalOutput):
            _raise(
                MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                "EVENT_PROPOSAL returned the wrong output type",
            )

        windows = [root_window]
        packages = [coarse_package]
        candidates: list[CandidateEvent] = []
        events: list[FusedActionEvent] = []
        proposals = proposal_output.proposals[: self._config.max_candidates]
        if proposals:
            proposal = proposals[0]
            if not _contains(coverage, proposal.interval):
                _raise(
                    MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                    "event proposal lies outside sampled coverage",
                )

            started = self._monotonic()
            candidate = self._candidate(
                mcap_id=mcap_id,
                package=coarse_package,
                inference=proposal_outcome,
                proposal=proposal,
                coverage=coverage,
            )
            dense_window = self._dense_window(
                mcap_id=mcap_id,
                video_export=video_export,
                parent=root_window,
                candidate=candidate,
            )
            candidates.append(candidate)
            windows.append(dense_window)
            durations[MainlineStage.WINDOWING] += _elapsed_ms(started, self._monotonic)

            started = self._monotonic()
            dense_package = self._materialize(
                video_export,
                staging,
                dense_window,
                rate_num=self._config.dense_rate_num,
                rate_den=self._config.dense_rate_den,
                tolerance_ns=self._config.dense_selection_tolerance_ns,
            )
            packages.append(dense_package)
            durations[MainlineStage.SAMPLING] += _elapsed_ms(started, self._monotonic)

            started = self._monotonic()
            dense_qa_request, dense_qa_outcome = self._infer(
                run_id=run_id,
                package=dense_package,
                artifact_root=staging,
                task=VisionTask.QA_DENSE,
                candidate_id=None,
            )
            requests.append(dense_qa_request)
            outcomes.append(dense_qa_outcome)
            durations[MainlineStage.QA_INFERENCE] += _elapsed_ms(started, self._monotonic)

            started = self._monotonic()
            qa_aggregate = self._aggregate_qa(dense_package, dense_qa_outcome)
            qa_aggregates.append(qa_aggregate)
            durations[MainlineStage.QA_AGGREGATION] += _elapsed_ms(started, self._monotonic)

            started = self._monotonic()
            action_request, action_outcome = self._infer(
                run_id=run_id,
                package=dense_package,
                artifact_root=staging,
                task=VisionTask.ACTION_EVIDENCE,
                candidate_id=candidate.candidate_event_id,
            )
            requests.append(action_request)
            outcomes.append(action_outcome)
            durations[MainlineStage.ACTION_EVIDENCE] += _elapsed_ms(started, self._monotonic)

            started = self._monotonic()
            boundary_request, boundary_outcome = self._infer(
                run_id=run_id,
                package=dense_package,
                artifact_root=staging,
                task=VisionTask.BOUNDARY_REFINEMENT,
                candidate_id=candidate.candidate_event_id,
            )
            requests.append(boundary_request)
            outcomes.append(boundary_outcome)
            durations[MainlineStage.BOUNDARY_REFINEMENT] += _elapsed_ms(started, self._monotonic)

            started = self._monotonic()
            events.append(
                self._fuse(
                    mcap_id=mcap_id,
                    candidate=candidate,
                    package=dense_package,
                    qa=qa_aggregate,
                    action=action_outcome,
                    boundary=boundary_outcome,
                    created_at=_timestamp(run_started_at),
                )
            )
            durations[MainlineStage.FUSION] += _elapsed_ms(started, self._monotonic)

        started = self._monotonic()
        self._write_records(
            staging=staging,
            packages=tuple(packages),
            requests=tuple(requests),
            outcomes=tuple(outcomes),
            qa=tuple(qa_aggregates),
            candidates=tuple(candidates),
            events=tuple(events),
        )
        durations[MainlineStage.PUBLISH] += _elapsed_ms(started, self._monotonic)

        has_event = bool(events)
        report = MainlineRunReport(
            schema_version="1.0",
            run_id=run_id,
            source_mcap_id=mcap_id,
            source_recording_identity=video_export.manifest.recording_identity,
            source_content_sha256=video_export.manifest.source_content_sha256,
            video_manifest_artifact_id=video_export.manifest_artifact_id,
            video_manifest_sha256=video_export.manifest_sha256,
            video_manifest_semantic_sha256=video_export.manifest.semantic_content_sha256,
            pipeline_version=PIPELINE_VERSION,
            config_sha256=config_sha256,
            status=(
                RunStatus.PRIMARY_COMPLETE if has_event else RunStatus.PRIMARY_COMPLETE_NO_EVENTS
            ),
            started_at=_timestamp(run_started_at),
            completed_at=_timestamp(self._clock()),
            duration_ms=_elapsed_ms(run_timer, self._monotonic),
            stages=self._stage_reports(durations, has_event=has_event),
            window_count=len(windows),
            package_count=len(packages),
            inference_attempt_count=len(requests),
            inference_success_count=len(outcomes),
            inference_failure_count=0,
            inference_invalid_output_count=0,
            candidate_count=len(candidates),
            event_count=len(events),
            fake_inference_attempt_count=len(requests),
            real_provider_request_count=0,
        )
        return MainlineBundle(
            schema_version="1.0",
            report=report,
            windows=tuple(windows),
            packages=tuple(packages),
            inference_requests=tuple(requests),
            inference_outcomes=tuple(outcomes),
            qa_aggregates=tuple(qa_aggregates),
            candidates=tuple(candidates),
            events=tuple(events),
        )

    @staticmethod
    def _recording_coverage(
        video_export: PublishedRegisteredVideoExport,
    ) -> NanosecondInterval:
        records = video_export.manifest.cameras
        origin_ns = min(record.export_first_observed_source_message_ns for record in records)
        starts = [record.export_first_observed_source_message_ns - origin_ns for record in records]
        ends: list[int] = []
        for record in records:
            mapping = record.media_time_mapping
            tail_ns = round_half_even(
                mapping.last_duration * mapping.time_base_numerator * 1_000_000_000,
                mapping.time_base_denominator,
            )
            ends.append(record.export_last_observed_source_message_ns - origin_ns + tail_ns)
        start_ns, end_ns = max(starts), min(ends)
        if start_ns >= end_ns:
            _raise(
                MainlineRunErrorCode.INVALID_REQUEST,
                "six cameras have no common source-time coverage",
            )
        return NanosecondInterval(start_ns=start_ns, end_ns=end_ns)

    def _materialize(
        self,
        video_export: PublishedRegisteredVideoExport,
        staging: Path,
        window: TemporalWindow,
        *,
        rate_num: int,
        rate_den: int,
        tolerance_ns: int,
    ) -> TemporalVisualPackage:
        package = self._frames.materialize(
            FrameMaterializationRequest(
                video_export=video_export,
                output_directory=staging,
                window=window,
                purpose=window.purpose,
                rate_num=rate_num,
                rate_den=rate_den,
                selection_tolerance_ns=tolerance_ns,
            )
        )
        if (
            package.window_id != window.window_id
            or package.mcap_id != window.mcap_id
            or package.purpose is not window.purpose
            or package.interval != window.interval
        ):
            _raise(
                MainlineRunErrorCode.FRAME_MATERIALIZATION_FAILED,
                "materialized package does not match its requested window",
            )
        return package

    def _infer(
        self,
        *,
        run_id: str,
        package: TemporalVisualPackage,
        artifact_root: Path,
        task: VisionTask,
        candidate_id: str | None,
    ) -> tuple[VisionInferenceRequest, VisionInferenceSuccess]:
        identity = {
            "run_id": run_id,
            "package_id": package.package_id,
            "task": task.value,
            "candidate_id": candidate_id,
            "model": [
                self._model.provider,
                self._model.model_name,
                self._model.model_version,
            ],
        }
        request = VisionInferenceRequest(
            schema_version="1.0",
            inference_id=_stable_uuid("vision-inference", identity),
            request_id=_stable_uuid("vision-request", identity),
            mcap_id=package.mcap_id,
            package_id=package.package_id,
            package_content_sha256=package.content_sha256,
            interval=package.interval,
            subject_candidate_id=candidate_id,
            task=task,
            provider=self._model.provider,
            model_name=self._model.model_name,
            model_version=self._model.model_version,
            prompt_version=self._config.prompt_version,
            output_contract_version=self._config.output_contract_version,
            camera_inputs=SixCameraMap[InferenceCameraInput](
                {
                    camera_id: InferenceCameraInput(
                        camera_id=camera_id,
                        frame_ids=tuple(
                            frame.frame_id for frame in package.cameras[camera_id].frames
                        ),
                    )
                    for camera_id in CAMERA_IDS
                }
            ),
            timeout_ms=self._config.timeout_ms,
        )
        try:
            outcome: VisionInferenceOutcome = self._model.infer(
                request,
                package,
                artifact_root,
            )
        except Exception as error:
            _raise(
                MainlineRunErrorCode.MODEL_INFERENCE_FAILED,
                f"{task.value} adapter call failed: {error}",
            )
        if not isinstance(outcome, (VisionInferenceSuccess, VisionInferenceFailure)):
            _raise(
                MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                f"{task.value} returned an unknown outcome type",
            )
        if (
            outcome.inference_id != request.inference_id
            or outcome.request_id != request.request_id
            or outcome.task is not request.task
            or outcome.provider != request.provider
            or outcome.model_name != request.model_name
            or outcome.model_version != request.model_version
        ):
            _raise(
                MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                f"{task.value} outcome identity does not match its request",
            )
        if isinstance(outcome, VisionInferenceFailure):
            _raise(
                MainlineRunErrorCode.MODEL_INFERENCE_FAILED,
                f"{task.value} ended {outcome.status.value}: "
                f"{outcome.failure.code}: {outcome.failure.detail}",
            )
        self._validate_provider_output(package, outcome)
        return request, outcome

    @staticmethod
    def _validate_provider_output(
        package: TemporalVisualPackage,
        outcome: VisionInferenceSuccess,
    ) -> None:
        def require_within(interval: NanosecondInterval | None, label: str) -> None:
            if interval is not None and not _contains(package.interval, interval):
                _raise(
                    MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                    f"{outcome.task.value} {label} lies outside the requested package interval",
                )

        def require_ordinals(
            camera_id: CameraId,
            ordinals: tuple[int, ...],
            label: str,
        ) -> None:
            frame_count = len(package.cameras[camera_id].frames)
            if any(ordinal >= frame_count for ordinal in ordinals):
                _raise(
                    MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                    f"{outcome.task.value} {camera_id.value} {label} references "
                    "an absent frame ordinal",
                )

        output = outcome.output
        if isinstance(output, QAOutput):
            for camera_id, claim in output.cameras.items():
                require_within(claim.observed_interval, "observed interval")
                require_ordinals(camera_id, claim.frame_ordinals, "QA claim")
            return
        if isinstance(output, EventProposalOutput):
            for proposal in output.proposals:
                require_within(proposal.interval, "proposal interval")
                for camera_id, claim in proposal.cameras.items():
                    require_ordinals(camera_id, claim.frame_ordinals, "proposal")
            return
        if isinstance(output, ActionEvidence):
            for camera_id, claim in output.cameras.items():
                require_within(claim.observed_interval, "action observed interval")
                require_within(claim.event_interval, "action event interval")
                require_ordinals(camera_id, claim.frame_ordinals, "action claim")
            for hypothesis in output.cross_view_hypotheses:
                require_within(hypothesis.interval, "cross-view action interval")
            return
        if isinstance(output, BoundaryRefinement):
            for camera_id, claim in output.cameras.items():
                require_within(claim.observed_interval, "boundary observed interval")
                require_within(claim.onset_interval, "boundary onset interval")
                require_within(claim.offset_interval, "boundary offset interval")
                require_ordinals(camera_id, claim.frame_ordinals, "boundary claim")
            return
        if isinstance(output, FusionAdjudicationOutput):
            for fusion_hypothesis in output.hypotheses:
                require_within(fusion_hypothesis.interval, "fusion interval")
            return
        _raise(
            MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
            f"{outcome.task.value} returned an unsupported output type",
        )

    @staticmethod
    def _aggregate_qa(
        package: TemporalVisualPackage,
        outcome: VisionInferenceSuccess,
    ) -> QAResultAggregate:
        output = outcome.output
        if not isinstance(output, QAOutput):
            _raise(
                MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                f"{outcome.task.value} returned the wrong QA output type",
            )
        camera_results: dict[CameraId, CameraQAResult] = {}
        usable = 0
        for camera_id in CAMERA_IDS:
            claim = output.cameras[camera_id]
            frames = package.cameras[camera_id].frames
            try:
                evidence_ids = tuple(frames[index].frame_id for index in claim.frame_ordinals)
            except IndexError:
                _raise(
                    MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                    f"{camera_id.value} QA referenced an absent frame ordinal",
                )
            if claim.status in {CameraQAStatus.GOOD, CameraQAStatus.DEGRADED}:
                usable += 1
            result_id = _stable_uuid(
                "camera-qa-result",
                {
                    "package_id": package.package_id,
                    "inference_id": outcome.inference_id,
                    "camera_id": camera_id.value,
                    "claim": claim,
                    "frame_ids": evidence_ids,
                },
            )
            camera_results[camera_id] = CameraQAResult(
                qa_result_id=result_id,
                mcap_id=package.mcap_id,
                package_id=package.package_id,
                inference_id=outcome.inference_id,
                camera_id=camera_id,
                claim=claim,
                evidence_frame_ids=evidence_ids,
            )

        statuses = tuple(result.claim.status for result in camera_results.values())
        if usable == 0:
            overall = RecordingQAStatus.UNUSABLE
        elif usable < 4:
            overall = RecordingQAStatus.INCOMPLETE
        elif all(status is CameraQAStatus.GOOD for status in statuses):
            overall = RecordingQAStatus.USABLE
        else:
            overall = RecordingQAStatus.DEGRADED
        aggregate_id = _stable_uuid(
            "qa-aggregate",
            {
                "mcap_id": package.mcap_id,
                "scope": package.interval,
                "results": tuple(camera_results.values()),
                "policy": QA_POLICY_VERSION,
            },
        )
        return QAResultAggregate(
            aggregate_id=aggregate_id,
            mcap_id=package.mcap_id,
            scope=package.interval,
            overall_status=overall,
            usable_camera_count=usable,
            camera_results=SixCameraMap[CameraQAResult](camera_results),
            policy_version=QA_POLICY_VERSION,
        )

    def _candidate(
        self,
        *,
        mcap_id: str,
        package: TemporalVisualPackage,
        inference: VisionInferenceSuccess,
        proposal: EventProposal,
        coverage: NanosecondInterval,
    ) -> CandidateEvent:
        requested_start = proposal.interval.start_ns - self._config.dense_padding_ns
        requested_end = proposal.interval.end_ns + self._config.dense_padding_ns
        dense_interval = NanosecondInterval(
            start_ns=max(coverage.start_ns, requested_start),
            end_ns=min(coverage.end_ns, requested_end),
        )
        candidate_id = _stable_uuid(
            "candidate-event",
            {
                "mcap_id": mcap_id,
                "package_id": package.package_id,
                "inference_id": inference.inference_id,
                "proposal": proposal,
                "dense_interval": dense_interval,
                "ontology": ONTOLOGY_VERSION,
            },
        )
        return CandidateEvent(
            candidate_event_id=candidate_id,
            mcap_id=mcap_id,
            source_package_id=package.package_id,
            source_inference_id=inference.inference_id,
            proposal=proposal,
            dense_interval=dense_interval,
            ontology_version=ONTOLOGY_VERSION,
            status=CandidateEventStatus.ACCEPTED,
        )

    def _dense_window(
        self,
        *,
        mcap_id: str,
        video_export: PublishedRegisteredVideoExport,
        parent: TemporalWindow,
        candidate: CandidateEvent,
    ) -> TemporalWindow:
        proposal = candidate.proposal.interval
        requested = NanosecondInterval(
            start_ns=proposal.start_ns - self._config.dense_padding_ns,
            end_ns=proposal.end_ns + self._config.dense_padding_ns,
        )
        return TemporalWindow(
            schema_version="1.0",
            window_id=_stable_uuid(
                "dense-window",
                {
                    "candidate_id": candidate.candidate_event_id,
                    "requested": requested,
                    "effective": candidate.dense_interval,
                },
            ),
            mcap_id=mcap_id,
            camera_mapping_run_id=None,
            alignment_id=video_export.manifest.alignment_id,
            requested_interval=requested,
            interval=candidate.dense_interval,
            purpose=SamplingPurpose.ACTION_DENSE,
            parent_window_id=parent.window_id,
            source_candidate_id=candidate.candidate_event_id,
            source_event_id=None,
            generation=1,
        )

    @staticmethod
    def _boundary_estimate(
        intervals: tuple[NanosecondInterval, ...],
        fallback: int,
    ) -> tuple[int, int]:
        if not intervals:
            return fallback, 0
        centers = tuple(
            interval.start_ns + (interval.end_ns - interval.start_ns) // 2 for interval in intervals
        )
        estimate = median_low(sorted(centers))
        uncertainty = max(
            abs(center - estimate) + (interval.duration_ns + 1) // 2
            for center, interval in zip(centers, intervals, strict=True)
        )
        return estimate, uncertainty

    @staticmethod
    def _fuse(
        *,
        mcap_id: str,
        candidate: CandidateEvent,
        package: TemporalVisualPackage,
        qa: QAResultAggregate,
        action: VisionInferenceSuccess,
        boundary: VisionInferenceSuccess,
        created_at: str,
    ) -> FusedActionEvent:
        action_output = action.output
        boundary_output = boundary.output
        if not isinstance(action_output, ActionEvidence) or not isinstance(
            boundary_output, BoundaryRefinement
        ):
            _raise(
                MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                "fusion requires action evidence and boundary refinement",
            )

        action_intervals = tuple(
            claim.event_interval
            for claim in action_output.cameras.values()
            if claim.event_interval is not None
        )
        onset_intervals = tuple(
            claim.onset_interval
            for claim in boundary_output.cameras.values()
            if claim.status is BoundaryStatus.OBSERVED and claim.onset_interval is not None
        )
        offset_intervals = tuple(
            claim.offset_interval
            for claim in boundary_output.cameras.values()
            if claim.status is BoundaryStatus.OBSERVED and claim.offset_interval is not None
        )
        fallback_start = (
            median_low(sorted(interval.start_ns for interval in action_intervals))
            if action_intervals
            else candidate.proposal.interval.start_ns
        )
        fallback_end = (
            median_low(sorted(interval.end_ns for interval in action_intervals))
            if action_intervals
            else candidate.proposal.interval.end_ns
        )
        start_ns, start_uncertainty = LocalMainlinePipeline._boundary_estimate(
            onset_intervals, fallback_start
        )
        end_ns, end_uncertainty = LocalMainlinePipeline._boundary_estimate(
            offset_intervals, fallback_end
        )
        if start_ns >= end_ns:
            start_ns, end_ns = fallback_start, fallback_end
        if start_ns >= end_ns:
            _raise(
                MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                "fusion could not derive a nonempty event interval",
            )
        event_interval = NanosecondInterval(start_ns=start_ns, end_ns=end_ns)

        evidence: dict[CameraId, CameraEventProvenance] = {}
        supporting = 0
        for camera_id in CAMERA_IDS:
            claim = action_output.cameras[camera_id]
            frames = package.cameras[camera_id].frames
            try:
                frame_ids = tuple(frames[index].frame_id for index in claim.frame_ordinals)
            except IndexError:
                _raise(
                    MainlineRunErrorCode.INVALID_MODEL_OUTPUT,
                    f"{camera_id.value} action referenced an absent frame ordinal",
                )
            if claim.status in {
                CameraEvidenceStatus.SUPPORTING,
                CameraEvidenceStatus.PARTIAL,
            }:
                supporting += 1
            evidence[camera_id] = CameraEventProvenance(
                camera_id=camera_id,
                claim=claim,
                package_id=package.package_id,
                inference_id=action.inference_id,
                frame_ids=frame_ids,
                qa_result_id=qa.camera_results[camera_id].qa_result_id,
            )

        action_type = (
            action_output.cross_view_hypotheses[0].action_type
            if action_output.cross_view_hypotheses
            else candidate.proposal.label_hint or "interaction"
        )
        event_id = _stable_uuid(
            "fused-action-event",
            {
                "candidate_id": candidate.candidate_event_id,
                "interval": event_interval,
                "action_type": action_type,
                "evidence": tuple(evidence.values()),
                "fusion_policy": FUSION_POLICY_VERSION,
            },
        )
        return FusedActionEvent(
            schema_version="1.0",
            event_id=event_id,
            mcap_id=mcap_id,
            candidate_event_ids=(candidate.candidate_event_id,),
            interval=event_interval,
            action_type=action_type,
            boundary_start_uncertainty_ns=start_uncertainty,
            boundary_end_uncertainty_ns=end_uncertainty,
            camera_evidence=SixCameraMap[CameraEventProvenance](evidence),
            fusion_policy_version=FUSION_POLICY_VERSION,
            boundary_inference_id=boundary.inference_id,
            producer_provider="fake",
            status=(ActionEventStatus.FINAL if supporting >= 2 else ActionEventStatus.AMBIGUOUS),
            production_eligible=False,
            created_at=created_at,
        )

    @staticmethod
    def _write_records(
        *,
        staging: Path,
        packages: tuple[TemporalVisualPackage, ...],
        requests: tuple[VisionInferenceRequest, ...],
        outcomes: tuple[VisionInferenceSuccess, ...],
        qa: tuple[QAResultAggregate, ...],
        candidates: tuple[CandidateEvent, ...],
        events: tuple[FusedActionEvent, ...],
    ) -> None:
        for package in packages:
            _write_new(staging / "packages" / f"{package.package_id}.json", package)
        for ordinal, (request, outcome) in enumerate(zip(requests, outcomes, strict=True)):
            task_name = request.task.value.lower().replace("_", "-")
            prefix = f"{ordinal:02d}-{task_name}"
            _write_new(staging / "inferences" / f"{prefix}-request.json", request)
            _write_new(staging / "inferences" / f"{prefix}-outcome.json", outcome)
        _write_new(staging / "qa-aggregates.json", qa)
        _write_new(staging / "candidates.json", candidates)
        _write_new(staging / "action-events.json", events)

    @staticmethod
    def _stage_reports(
        durations: dict[MainlineStage, int],
        *,
        has_event: bool,
    ) -> tuple[StageReport, ...]:
        common_counts = {
            MainlineStage.WINDOWING: 2 if has_event else 1,
            MainlineStage.SAMPLING: 2 if has_event else 1,
            MainlineStage.QA_INFERENCE: 2 if has_event else 1,
            MainlineStage.QA_AGGREGATION: 2 if has_event else 1,
            MainlineStage.EVENT_PROPOSAL: 1,
            MainlineStage.PUBLISH: 1,
        }
        candidate_stages = {
            MainlineStage.ACTION_EVIDENCE,
            MainlineStage.BOUNDARY_REFINEMENT,
            MainlineStage.FUSION,
        }
        reports: list[StageReport] = []
        for stage in MainlineStage:
            if stage in candidate_stages and not has_event:
                reports.append(
                    StageReport(
                        stage=stage,
                        status=StageStatus.SKIPPED,
                        planned=1,
                        succeeded=0,
                        failed=0,
                        pending=0,
                        skipped=1,
                        duration_ms=durations[stage],
                    )
                )
                continue
            planned = 1 if stage in candidate_stages else common_counts[stage]
            reports.append(
                StageReport(
                    stage=stage,
                    status=StageStatus.SUCCEEDED,
                    planned=planned,
                    succeeded=planned,
                    failed=0,
                    pending=0,
                    skipped=0,
                    duration_ms=durations[stage],
                )
            )
        return tuple(reports)


__all__ = [
    "FUSION_POLICY_VERSION",
    "PIPELINE_VERSION",
    "QA_POLICY_VERSION",
    "LocalMainlineConfig",
    "LocalMainlinePipeline",
    "MainlineRunError",
    "MainlineRunErrorCode",
    "PublishedMainlineRun",
]
