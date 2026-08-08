"""Bounded producer/consumer execution for the local Mage perception stream.

The runner never materializes a whole recording up front. It keeps at most two
ordered native-video observations in flight so next-segment materialization and
codec preprocessing can overlap the current serialized model generation.
Deterministic projection, tracking, fusion, and persistence remain ordinal.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Final, TypeVar

from robata.application.canonical.mage_stream import (
    FfmpegCommandResult,
    MageMaterializedReasoningContext,
    MageReasoningContext,
    MageStreamMaterializationError,
    MageStreamMaterializer,
    MageStreamPlan,
    build_perception_context_manifest,
    exact_file_sha256,
)
from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.perception_stream import MageObservation, PerceptionContextManifest
from robata.inference.mage_video_adapter import (
    MageVideoDurableCameraSegment,
    MageVideoDurableSegmentResolver,
)
from robata.perception.durable_scheduler import (
    DurablePerceptionRun,
    DurablePerceptionRunSnapshot,
    DurablePerceptionWorkClaim,
    DurablePerceptionWorkItem,
    DurablePerceptionWorkState,
    SQLitePerceptionWorkScheduler,
)
from robata.perception.pipeline import (
    LocalPerceptionArtifactReference,
    LocalPerceptionArtifactStore,
    PerceptionContextOutcome,
    PerceptionProjectedContext,
    PerceptionStage,
    StreamPerceptionPipeline,
    StreamPerceptionRunResult,
)
from robata.perception.projectors import (
    MediaHealthCameraFact,
    MediaHealthDisposition,
    MediaHealthReport,
    create_media_health_report,
)

LOCAL_MAGE_STREAM_EXECUTION_POLICY_VERSION: Final = "local-mage-stream-execution-v2"
LOCAL_MAGE_MEDIA_HEALTH_POLICY_VERSION: Final = "local-mage-media-health-v1"
LOCAL_MAGE_STREAM_CONTEXT_REPORT_NAMESPACE: Final = "local-mage-stream-context-report-v1"
LOCAL_MAGE_STREAM_RUN_MANIFEST_VERSION: Final = "local-mage-stream-run-manifest-v1"
LOCAL_MAGE_STREAM_RUN_MANIFEST_NAMESPACE: Final = "local-mage-stream-run-manifest-v1"

_T = TypeVar("_T")


class LocalMageStreamExecutionError(RuntimeError):
    """A single-worker local Mage stream cannot preserve its declared bindings."""


@dataclass(frozen=True, slots=True)
class LocalMageStreamContextExecution:
    """Measured evidence for one just-in-time focus context."""

    focus_segment_ordinal: int
    context_manifest_key: str
    context_manifest_semantic_sha256: str
    durable_path: Path
    materialization_seconds: float
    media_scan_seconds: float
    observation_seconds: float
    projection_seconds: float
    temporal_reconcile_seconds: float
    fusion_seconds: float
    normal_model_call_count: int
    refinement_model_call_count: int
    persisted_report_exact_sha256: str | None

    def as_projection(self) -> dict[str, object]:
        return {
            "execution_policy_version": LOCAL_MAGE_STREAM_EXECUTION_POLICY_VERSION,
            "focus_segment_ordinal": self.focus_segment_ordinal,
            "context_manifest_key": self.context_manifest_key,
            "context_manifest_semantic_sha256": self.context_manifest_semantic_sha256,
            "durable_path": str(self.durable_path),
            "materialization_seconds": self.materialization_seconds,
            "media_scan_seconds": self.media_scan_seconds,
            "observation_seconds": self.observation_seconds,
            "projection_seconds": self.projection_seconds,
            "temporal_reconcile_seconds": self.temporal_reconcile_seconds,
            "fusion_seconds": self.fusion_seconds,
            "normal_model_call_count": self.normal_model_call_count,
            "refinement_model_call_count": self.refinement_model_call_count,
        }


@dataclass(frozen=True, slots=True)
class LocalMageStreamDurableExecution:
    """Authoritative local scheduler state observed after one stream attempt."""

    run: DurablePerceptionRun
    snapshot: DurablePerceptionRunSnapshot
    finalization_state: str
    fusion_work_item_ids: tuple[str, ...]
    pending_refinement_work_item_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalMageStreamExecutionResult:
    """Whole-stream output plus per-context measured execution evidence."""

    pipeline_result: StreamPerceptionRunResult
    contexts: tuple[LocalMageStreamContextExecution, ...]
    queue_depth: int
    run_manifest: LocalPerceptionArtifactReference | None = None
    durable_execution: LocalMageStreamDurableExecution | None = None


class _LocalDurableSchedulerBridge:
    """Execute local work under durable claims without coupling the pipeline to SQLite.

    Normal work is claimed around the operation that actually performs it.  Fusion
    and refinement are written after the deterministic pipeline has emitted their
    immutable outputs; refinement remains request-only and is intentionally not
    marked successful without a real targeted-model result.
    """

    def __init__(
        self,
        *,
        scheduler: SQLitePerceptionWorkScheduler,
        plan: MageStreamPlan,
        codec_policy_version: str,
        worker_id: str,
        lease_duration_seconds: int,
    ) -> None:
        if not isinstance(scheduler, SQLitePerceptionWorkScheduler):
            raise TypeError("durable_scheduler must be SQLitePerceptionWorkScheduler")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("durable_worker_id must be a nonempty string")
        if (
            isinstance(lease_duration_seconds, bool)
            or not isinstance(lease_duration_seconds, int)
            or lease_duration_seconds <= 0
        ):
            raise ValueError("durable_lease_duration_seconds must be a positive integer")
        self._scheduler = scheduler
        self._worker_id = worker_id
        self._lease_duration_seconds = lease_duration_seconds
        self._run = scheduler.register_plan(
            plan,
            codec_policy_version=codec_policy_version,
        )

    @property
    def run(self) -> DurablePerceptionRun:
        return self._run

    def run_context_stage(
        self,
        *,
        focus_segment_ordinal: int,
        stage: PerceptionStage,
        operation: Callable[[], _T],
        result_binding: Callable[[_T], tuple[str, str]],
    ) -> _T:
        claim = self._claim_context_stage(
            focus_segment_ordinal=focus_segment_ordinal,
            stage=stage,
        )
        try:
            value = operation()
        except BaseException as error:
            self._fail(claim, stage=stage, error=error)
            raise
        result_reference, result_sha256 = result_binding(value)
        self._succeed(claim, result_reference=result_reference, result_sha256=result_sha256)
        return value

    def begin_context_stage(
        self, *, focus_segment_ordinal: int, stage: PerceptionStage
    ) -> DurablePerceptionWorkClaim:
        return self._claim_context_stage(
            focus_segment_ordinal=focus_segment_ordinal,
            stage=stage,
        )

    def complete(
        self,
        claim: DurablePerceptionWorkClaim,
        *,
        result_reference: str,
        result_sha256: str,
    ) -> None:
        self._succeed(claim, result_reference=result_reference, result_sha256=result_sha256)

    def fail(
        self, claim: DurablePerceptionWorkClaim, *, stage: PerceptionStage, error: BaseException
    ) -> None:
        self._fail(claim, stage=stage, error=error)

    def record_terminal_state(
        self,
        result: StreamPerceptionRunResult,
    ) -> LocalMageStreamDurableExecution:
        """Persist derived deterministic products, then seal/finalize only if complete."""

        observation_ordinals = {
            outcome.observation.observation_logical_key: outcome.context.focus_segment_ordinal
            for outcome in result.contexts
        }
        tracks_by_key = {track.event_track_key: track for track in result.event_tracks}
        fusion_by_track_key: dict[str, DurablePerceptionWorkItem] = {}
        fusion_work_item_ids: list[str] = []
        for decision in sorted(result.fusion_decisions, key=lambda item: item.fusion_key):
            track = tracks_by_key.get(decision.source_event_track_key)
            if track is None:
                raise LocalMageStreamExecutionError(
                    "fusion decision references an event track absent from final stream result"
                )
            source = track.source_hypotheses[-1]
            try:
                ordinal = observation_ordinals[source.source_observation_logical_key]
            except KeyError as error:
                raise LocalMageStreamExecutionError(
                    "fusion track source observation is absent from final stream contexts"
                ) from error
            item = self._scheduler.schedule_derived(
                run_key=self._run.run_key,
                focus_segment_ordinal=ordinal,
                stage=PerceptionStage.FUSION,
                input_sha256=decision.source_event_track_revision_semantic_sha256,
                config_sha256=semantic_sha256(
                    {
                        "durable_run_config_sha256": self._run.config_sha256,
                        "stage": PerceptionStage.FUSION.value,
                        "fusion_policy_version": decision.policy_version,
                    }
                ),
            )
            self._complete_item(
                item,
                result_reference=f"fusion:{decision.fusion_key}",
                result_sha256=decision.fusion_semantic_sha256,
            )
            fusion_by_track_key[decision.source_event_track_key] = item
            fusion_work_item_ids.append(item.work_item_id)

        track_by_latest_hypothesis = {
            track.source_hypotheses[-1].hypothesis_logical_key: track
            for track in result.event_tracks
        }
        pending_refinement_work_item_ids: list[str] = []
        for request in sorted(result.refine_requests, key=lambda item: item.refine_request_key):
            track = track_by_latest_hypothesis.get(request.target_hypothesis_logical_key)
            if track is None:
                raise LocalMageStreamExecutionError(
                    "refine request target hypothesis is absent from final event tracks"
                )
            fusion = fusion_by_track_key.get(track.event_track_key)
            if fusion is None:
                raise LocalMageStreamExecutionError(
                    "refine request has no durable upstream FUSION work item"
                )
            try:
                ordinal = observation_ordinals[request.source_observation_logical_key]
            except KeyError as error:
                raise LocalMageStreamExecutionError(
                    "refine request source observation is absent from final stream contexts"
                ) from error
            item = self._scheduler.schedule_derived(
                run_key=self._run.run_key,
                focus_segment_ordinal=ordinal,
                stage=PerceptionStage.PERCEPTION_REFINE,
                input_sha256=request.refine_request_semantic_sha256,
                config_sha256=semantic_sha256(
                    {
                        "durable_run_config_sha256": self._run.config_sha256,
                        "stage": PerceptionStage.PERCEPTION_REFINE.value,
                        "refine_policy_version": request.refine_policy_version,
                        "prompt_version": request.prompt_version,
                    }
                ),
                upstream_work_item_id=fusion.work_item_id,
            )
            pending_refinement_work_item_ids.append(item.work_item_id)

        if pending_refinement_work_item_ids:
            snapshot = self._scheduler.snapshot(self._run.run_key)
            return LocalMageStreamDurableExecution(
                run=snapshot.run,
                snapshot=snapshot,
                finalization_state="PENDING_REFINEMENT",
                fusion_work_item_ids=tuple(fusion_work_item_ids),
                pending_refinement_work_item_ids=tuple(pending_refinement_work_item_ids),
            )

        sealed = self._scheduler.seal_derived_work(self._run.run_key)
        finalization = next(
            item
            for item in self._scheduler.items_for_run(self._run.run_key)
            if item.stage is PerceptionStage.FINALIZE
        )
        result_reference, result_sha256 = _terminal_result_binding(result, sealed.run_key)
        self._complete_item(
            finalization,
            result_reference=result_reference,
            result_sha256=result_sha256,
        )
        snapshot = self._scheduler.snapshot(self._run.run_key)
        return LocalMageStreamDurableExecution(
            run=snapshot.run,
            snapshot=snapshot,
            finalization_state="SUCCEEDED",
            fusion_work_item_ids=tuple(fusion_work_item_ids),
            pending_refinement_work_item_ids=(),
        )

    def _claim_context_stage(
        self, *, focus_segment_ordinal: int, stage: PerceptionStage
    ) -> DurablePerceptionWorkClaim:
        item = next(
            item
            for item in self._scheduler.context_work(self._run.run_key, focus_segment_ordinal)
            if item.stage is stage
        )
        return self._claim_item(item)

    def _claim_item(self, item: DurablePerceptionWorkItem) -> DurablePerceptionWorkClaim:
        current = self._scheduler.get(item.work_item_id)
        if current.state is DurablePerceptionWorkState.SUCCEEDED:
            raise LocalMageStreamExecutionError(
                "local Mage executor will not recompute an already-succeeded durable work item"
            )
        claim = self._scheduler.claim_and_start(
            self._worker_id,
            self._lease_duration_seconds,
            run_key=self._run.run_key,
            work_item_id=item.work_item_id,
        )
        if claim is None:
            raise LocalMageStreamExecutionError(
                f"durable {item.stage.value} work is not ready for claim"
            )
        return claim

    def _complete_item(
        self,
        item: DurablePerceptionWorkItem,
        *,
        result_reference: str,
        result_sha256: str,
    ) -> None:
        claim = self._claim_item(item)
        self._succeed(claim, result_reference=result_reference, result_sha256=result_sha256)

    def _succeed(
        self,
        claim: DurablePerceptionWorkClaim,
        *,
        result_reference: str,
        result_sha256: str,
    ) -> None:
        self._scheduler.succeed(
            claim.lease,
            result_reference=result_reference,
            result_sha256=result_sha256,
        )

    def _fail(
        self,
        claim: DurablePerceptionWorkClaim,
        *,
        stage: PerceptionStage,
        error: BaseException,
    ) -> None:
        detail = f"{type(error).__name__}: {error}"
        self._scheduler.fail(
            claim.lease,
            error_code=f"{stage.value}_FAILED",
            retryable=False,
            error_detail=detail or type(error).__name__,
        )


def _terminal_result_binding(result: StreamPerceptionRunResult, run_key: str) -> tuple[str, str]:
    terminal_artifacts = result.terminal_artifacts
    if terminal_artifacts is not None:
        artifact = terminal_artifacts.terminal_manifest
        return artifact.logical_key, artifact.exact_sha256
    return (
        f"local-stream-finalize:{run_key}",
        semantic_sha256(
            {
                "event_track_revision_sha256_values": [
                    item.revision_semantic_sha256 for item in result.event_tracks
                ],
                "fusion_sha256_values": [
                    item.fusion_semantic_sha256 for item in result.fusion_decisions
                ],
                "refine_request_sha256_values": [
                    item.refine_request_semantic_sha256 for item in result.refine_requests
                ],
            }
        ),
    )


def _media_health_result_binding(value: MediaHealthReport) -> tuple[str, str]:
    return value.media_health_key, value.media_health_semantic_sha256


def _observation_result_binding(value: MageObservation) -> tuple[str, str]:
    return value.observation_logical_key, value.observation_semantic_sha256


def _projected_result_binding(value: PerceptionProjectedContext) -> tuple[str, str]:
    return (
        value.event_projection.event_projection_key,
        semantic_sha256(
            {
                "qa_projection_semantic_sha256": (
                    value.qa_projection.qa_projection_semantic_sha256
                ),
                "event_projection_semantic_sha256": (
                    value.event_projection.event_projection_semantic_sha256
                ),
                "evidence_projection_semantic_sha256": (
                    value.evidence_projection.evidence_projection_semantic_sha256
                ),
            }
        ),
    )


def _temporal_result_binding(value: PerceptionContextOutcome) -> tuple[str, str]:
    return (
        value.temporal_reconcile.reconcile_key,
        value.temporal_reconcile.reconcile_semantic_sha256,
    )


@dataclass(frozen=True, slots=True)
class _PreparedStreamContext:
    focus_segment_ordinal: int
    context: PerceptionContextManifest
    materialized_context: MageMaterializedReasoningContext
    media_health: MediaHealthReport
    materialization_seconds: float
    media_scan_seconds: float


class LocalMaterializedSegmentResolver(MageVideoDurableSegmentResolver):
    """Bind each context to the one just-materialized selected-camera segment."""

    def __init__(self) -> None:
        self._by_context_digest: dict[str, MageVideoDurableCameraSegment] = {}
        self._lock = RLock()

    def register(
        self,
        *,
        context: PerceptionContextManifest,
        materialized_context: MageMaterializedReasoningContext,
    ) -> None:
        if materialized_context.context.focus_segment_ordinal != context.focus_segment_ordinal:
            raise LocalMageStreamExecutionError(
                "materialized context focus segment does not match perception context"
            )
        context_lineage = tuple(item.segment_semantic_sha256 for item in context.ordered_segments)
        materialized_lineage = tuple(
            item.segment_semantic_sha256 for item in materialized_context.context.ordered_segments
        )
        if context_lineage != materialized_lineage:
            raise LocalMageStreamExecutionError(
                "materialized context storage lineage does not match perception context"
            )
        if len(context.ordered_segments) != 1:
            raise LocalMageStreamExecutionError(
                "local v1 resolver accepts one non-overlapping focus segment only"
            )
        if len(materialized_context.component_segment_exact_sha256_values) != 1:
            raise LocalMageStreamExecutionError(
                "local v1 resolver accepts one materialized component segment only"
            )
        camera_id = materialized_context.camera_id
        binding = context.cameras[camera_id]
        if not binding.available or not binding.selected_for_inference:
            raise LocalMageStreamExecutionError(
                "materialized camera is not selected and available in its perception context"
            )
        if binding.codec_stream_exact_sha256 != materialized_context.content_exact_sha256:
            raise LocalMageStreamExecutionError(
                "materialized content hash does not match perception context codec hash"
            )
        segment = MageVideoDurableCameraSegment(
            camera_id=camera_id,
            segment_semantic_sha256_values=tuple(
                item.segment_semantic_sha256 for item in context.ordered_segments
            ),
            codec_stream_exact_sha256=materialized_context.content_exact_sha256,
            durable_path=str(materialized_context.durable_path),
            content_sha256=materialized_context.content_exact_sha256,
            byte_count=materialized_context.byte_count,
        )
        context_digest = context.context_manifest_semantic_sha256
        with self._lock:
            existing = self._by_context_digest.get(context_digest)
            if existing is not None and existing != segment:
                raise LocalMageStreamExecutionError(
                    "a context was registered with conflicting durable native-video bytes"
                )
            self._by_context_digest[context_digest] = segment

    def resolve(
        self,
        *,
        context: PerceptionContextManifest,
        camera_id: CameraId,
    ) -> tuple[MageVideoDurableCameraSegment, ...]:
        with self._lock:
            registered = self._by_context_digest.get(context.context_manifest_semantic_sha256)
        if registered is None:
            raise LocalMageStreamExecutionError(
                "context has not been materialized and registered for native-video inference"
            )
        if registered.camera_id is not camera_id:
            raise LocalMageStreamExecutionError(
                "requested camera does not match the registered native-video segment"
            )
        return (registered,)


FfprobeCommandRunner = Callable[[tuple[str, ...]], FfmpegCommandResult]


class LocalMediaHealthScanner:
    """Measure only local file/hash/ffprobe facts; never invent semantic QA."""

    def __init__(
        self,
        *,
        ffprobe_binary: str = "ffprobe",
        command_runner: FfprobeCommandRunner | None = None,
        policy_version: str = LOCAL_MAGE_MEDIA_HEALTH_POLICY_VERSION,
    ) -> None:
        if not isinstance(ffprobe_binary, str) or not ffprobe_binary.strip():
            raise ValueError("ffprobe_binary must be a nonempty string")
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version must be a nonempty string")
        self._ffprobe_binary = ffprobe_binary
        self._command_runner = command_runner or _subprocess_ffprobe_runner
        self._policy_version = policy_version

    def scan(
        self,
        *,
        context: PerceptionContextManifest,
        materialized_context: MageMaterializedReasoningContext,
    ) -> MediaHealthReport:
        if len(context.ordered_segments) != 1:
            raise LocalMageStreamExecutionError(
                "local media-health v1 accepts one non-overlapping focus segment only"
            )
        selected_camera = materialized_context.camera_id
        selected_fact = self._scan_selected_camera(
            context=context,
            materialized_context=materialized_context,
        )
        facts: dict[CameraId, MediaHealthCameraFact] = {}
        for camera_id in CAMERA_IDS:
            if camera_id is selected_camera:
                facts[camera_id] = selected_fact
            else:
                facts[camera_id] = MediaHealthCameraFact(
                    camera_id=camera_id,
                    disposition=MediaHealthDisposition.UNAVAILABLE,
                    issue_codes=("CAMERA_UNAVAILABLE",),
                )
        return create_media_health_report(
            context_manifest_semantic_sha256=context.context_manifest_semantic_sha256,
            policy_version=self._policy_version,
            cameras=SixCameraMap[MediaHealthCameraFact](facts),
        )

    def _scan_selected_camera(
        self,
        *,
        context: PerceptionContextManifest,
        materialized_context: MageMaterializedReasoningContext,
    ) -> MediaHealthCameraFact:
        camera_id = materialized_context.camera_id
        binding = context.cameras[camera_id]
        issues: set[str] = set()
        path = materialized_context.durable_path.expanduser().resolve()
        if not binding.available or not binding.selected_for_inference:
            issues.add("SELECTED_CAMERA_CONTEXT_MISMATCH")
        if not path.is_file():
            issues.add("MATERIALIZED_SEGMENT_MISSING")
        else:
            try:
                digest, byte_count = exact_file_sha256(path)
            except MageStreamMaterializationError:
                issues.add("MATERIALIZED_SEGMENT_UNREADABLE")
            else:
                if byte_count != materialized_context.byte_count:
                    issues.add("MATERIALIZED_SEGMENT_BYTE_COUNT_MISMATCH")
                if digest != materialized_context.content_exact_sha256:
                    issues.add("MATERIALIZED_SEGMENT_HASH_MISMATCH")
                if digest != binding.codec_stream_exact_sha256:
                    issues.add("CONTEXT_CODEC_HASH_MISMATCH")
                if not issues:
                    ffprobe_issue = self._ffprobe_issue(path)
                    if ffprobe_issue is not None:
                        issues.add(ffprobe_issue)
        if not issues:
            disposition = MediaHealthDisposition.HEALTHY
        elif issues == {"FFPROBE_UNAVAILABLE"}:
            disposition = MediaHealthDisposition.DEGRADED
        else:
            disposition = MediaHealthDisposition.UNUSABLE
        return MediaHealthCameraFact(
            camera_id=camera_id,
            disposition=disposition,
            issue_codes=tuple(sorted(issues)),
            observed_interval=context.context_interval,
        )

    def _ffprobe_issue(self, path: Path) -> str | None:
        command = (
            self._ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        )
        try:
            result = self._command_runner(command)
        except FileNotFoundError:
            return "FFPROBE_UNAVAILABLE"
        except OSError:
            return "FFPROBE_UNAVAILABLE"
        if result.returncode != 0:
            return "FFPROBE_DECODE_FAILED"
        try:
            document = json.loads(result.stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            return "FFPROBE_OUTPUT_INVALID"
        streams = document.get("streams") if isinstance(document, dict) else None
        if not isinstance(streams, list) or not any(
            isinstance(stream, dict) and stream.get("codec_type") == "video" for stream in streams
        ):
            return "FFPROBE_NO_VIDEO_STREAM"
        # ``ffprobe`` measured stream readability.  We deliberately do not
        # label black/frozen/exposure conditions without a separate probe.
        return None


def execute_local_mage_stream(
    *,
    plan: MageStreamPlan,
    source_path: Path,
    selected_camera: CameraId,
    materializer: MageStreamMaterializer,
    codec_policy_version: str,
    resolver: LocalMaterializedSegmentResolver,
    pipeline: StreamPerceptionPipeline,
    artifact_store: LocalPerceptionArtifactStore | None = None,
    media_health_scanner: LocalMediaHealthScanner | None = None,
    materialization_root: Path,
    max_inflight_observations: int = 2,
    durable_scheduler: SQLitePerceptionWorkScheduler | None = None,
    durable_worker_id: str = "local-mage-stream",
    durable_lease_duration_seconds: int = 14_520,
) -> LocalMageStreamExecutionResult:
    """Run a bounded producer/consumer stream with ordered deterministic reduction.

    At most ``max_inflight_observations`` native-video requests are in flight.
    Materialization and media health for the next focus segment proceed while the
    current Mage request is in preprocessing/generation; observations are still
    consumed strictly by segment ordinal so tracks and durable projections remain
    deterministic. A value of one restores the serial backpressure profile.
    """

    _validate_local_v1_plan(plan)
    if not isinstance(selected_camera, CameraId):
        raise TypeError("selected_camera must be CameraId")
    if (
        isinstance(max_inflight_observations, bool)
        or not isinstance(max_inflight_observations, int)
        or max_inflight_observations < 1
        or max_inflight_observations > 2
    ):
        raise ValueError("max_inflight_observations must be an integer in [1, 2]")
    scanner = media_health_scanner or LocalMediaHealthScanner()
    durable_bridge = (
        None
        if durable_scheduler is None
        else _LocalDurableSchedulerBridge(
            scheduler=durable_scheduler,
            plan=plan,
            codec_policy_version=codec_policy_version,
            worker_id=durable_worker_id,
            lease_duration_seconds=durable_lease_duration_seconds,
        )
    )
    session = pipeline.open_session()
    reports: list[LocalMageStreamContextExecution] = []
    context_iterator = iter(plan.reasoning_contexts)

    def prepare(stream_context: MageReasoningContext) -> _PreparedStreamContext:
        focus_segment = stream_context.ordered_segments[-1]
        materialization_started = time.perf_counter()
        materialized_storage = materializer.materialize_storage_segment(
            plan=plan,
            source_path=source_path,
            segment=focus_segment,
            output_root=materialization_root,
        )
        materialized_context = materializer.materialize_reasoning_context(
            context=stream_context,
            camera_id=selected_camera,
            storage_segments=(materialized_storage,),
            output_root=materialization_root,
        )
        materialization_seconds = time.perf_counter() - materialization_started
        context = build_perception_context_manifest(
            plan=plan,
            context=stream_context,
            materialized_context=materialized_context,
            codec_policy_version=codec_policy_version,
        )
        resolver.register(context=context, materialized_context=materialized_context)
        scan_started = time.perf_counter()

        def scan_current_context(
            context: PerceptionContextManifest = context,
            materialized_context: MageMaterializedReasoningContext = materialized_context,
        ) -> MediaHealthReport:
            return scanner.scan(context=context, materialized_context=materialized_context)

        if durable_bridge is None:
            health = session.scan_media(scan_current_context)
        else:
            health = durable_bridge.run_context_stage(
                focus_segment_ordinal=stream_context.focus_segment_ordinal,
                stage=PerceptionStage.MEDIA_SCAN,
                operation=lambda: session.scan_media(scan_current_context),
                result_binding=_media_health_result_binding,
            )
        return _PreparedStreamContext(
            focus_segment_ordinal=stream_context.focus_segment_ordinal,
            context=context,
            materialized_context=materialized_context,
            media_health=health,
            materialization_seconds=materialization_seconds,
            media_scan_seconds=time.perf_counter() - scan_started,
        )

    def append_report(
        prepared: _PreparedStreamContext,
        before: Mapping[PerceptionStage, tuple[int, float]],
        after: Mapping[PerceptionStage, tuple[int, float]],
    ) -> None:
        report = LocalMageStreamContextExecution(
            focus_segment_ordinal=prepared.focus_segment_ordinal,
            context_manifest_key=prepared.context.context_manifest_key,
            context_manifest_semantic_sha256=prepared.context.context_manifest_semantic_sha256,
            durable_path=prepared.materialized_context.durable_path,
            materialization_seconds=prepared.materialization_seconds,
            media_scan_seconds=prepared.media_scan_seconds,
            observation_seconds=_stage_elapsed_delta(
                before, after, PerceptionStage.PERCEPTION_OBSERVE
            ),
            projection_seconds=_stage_elapsed_delta(
                before, after, PerceptionStage.OBSERVATION_PROJECT
            ),
            temporal_reconcile_seconds=_stage_elapsed_delta(
                before, after, PerceptionStage.TEMPORAL_RECONCILE
            ),
            fusion_seconds=_stage_elapsed_delta(before, after, PerceptionStage.FUSION),
            normal_model_call_count=_stage_count_delta(
                before, after, PerceptionStage.PERCEPTION_OBSERVE
            ),
            refinement_model_call_count=0,
            persisted_report_exact_sha256=None,
        )
        if artifact_store is not None:
            report_digest = artifact_store.put(
                kind="local-stream-context-report",
                logical_key=(
                    f"{LOCAL_MAGE_STREAM_CONTEXT_REPORT_NAMESPACE}:"
                    f"{prepared.context.context_manifest_semantic_sha256}"
                ),
                payload=canonical_json_bytes(report.as_projection()),
            )
            report = replace(report, persisted_report_exact_sha256=report_digest)
        reports.append(report)

    pending: deque[
        tuple[
            _PreparedStreamContext,
            Future[tuple[MageObservation, float]],
            DurablePerceptionWorkClaim | None,
        ]
    ] = deque()
    with ThreadPoolExecutor(
        max_workers=max_inflight_observations,
        thread_name_prefix="robata-mage-observe",
    ) as executor:

        def submit_next() -> bool:
            try:
                stream_context = next(context_iterator)
            except StopIteration:
                return False
            prepared = prepare(stream_context)
            observation_claim: DurablePerceptionWorkClaim | None = None
            if durable_bridge is not None:
                observation_claim = durable_bridge.begin_context_stage(
                    focus_segment_ordinal=prepared.focus_segment_ordinal,
                    stage=PerceptionStage.PERCEPTION_OBSERVE,
                )
            try:
                future = executor.submit(pipeline.observe_context, prepared.context)
            except BaseException as error:
                if durable_bridge is not None and observation_claim is not None:
                    durable_bridge.fail(
                        observation_claim,
                        stage=PerceptionStage.PERCEPTION_OBSERVE,
                        error=error,
                    )
                raise
            pending.append((prepared, future, observation_claim))
            return True

        while len(pending) < max_inflight_observations and submit_next():
            pass
        while pending:
            prepared, future, observation_claim = pending.popleft()
            try:
                observation, _provider_elapsed = future.result()
            except BaseException as error:
                if durable_bridge is not None and observation_claim is not None:
                    durable_bridge.fail(
                        observation_claim,
                        stage=PerceptionStage.PERCEPTION_OBSERVE,
                        error=error,
                    )
                raise
            if durable_bridge is not None and observation_claim is not None:
                observation_reference, observation_sha256 = _observation_result_binding(observation)
                durable_bridge.complete(
                    observation_claim,
                    result_reference=observation_reference,
                    result_sha256=observation_sha256,
                )
            before = _measurement_map(session.stage_measurements())
            if durable_bridge is None:
                session.consume_precomputed(
                    context=prepared.context,
                    media_health=prepared.media_health,
                    observation=observation,
                    observation_elapsed_seconds=_provider_elapsed,
                )
            else:

                def project_current(
                    context: PerceptionContextManifest = prepared.context,
                    media_health: MediaHealthReport = prepared.media_health,
                    current_observation: MageObservation = observation,
                    elapsed: float = _provider_elapsed,
                ) -> PerceptionProjectedContext:
                    return session.project_precomputed(
                        context=context,
                        media_health=media_health,
                        observation=current_observation,
                        observation_elapsed_seconds=elapsed,
                    )

                projected = durable_bridge.run_context_stage(
                    focus_segment_ordinal=prepared.focus_segment_ordinal,
                    stage=PerceptionStage.OBSERVATION_PROJECT,
                    operation=project_current,
                    result_binding=_projected_result_binding,
                )

                def reconcile_current(
                    current_projection: PerceptionProjectedContext = projected,
                ) -> PerceptionContextOutcome:
                    return session.reconcile_projected(current_projection)

                durable_bridge.run_context_stage(
                    focus_segment_ordinal=prepared.focus_segment_ordinal,
                    stage=PerceptionStage.TEMPORAL_RECONCILE,
                    operation=reconcile_current,
                    result_binding=_temporal_result_binding,
                )
            after = _measurement_map(session.stage_measurements())
            append_report(prepared, before, after)
            while len(pending) < max_inflight_observations and submit_next():
                pass

    result = session.finalize()
    durable_execution = (
        None if durable_bridge is None else durable_bridge.record_terminal_state(result)
    )
    context_reports = tuple(reports)
    run_manifest: LocalPerceptionArtifactReference | None = None
    if artifact_store is not None:
        run_projection = _build_run_manifest_projection(
            plan=plan,
            selected_camera=selected_camera,
            codec_policy_version=codec_policy_version,
            queue_depth=max_inflight_observations,
            contexts=context_reports,
            pipeline_result=result,
            durable_execution=durable_execution,
            artifact_store=artifact_store,
        )
        run_identity = semantic_sha256(run_projection)
        run_logical_key = f"{LOCAL_MAGE_STREAM_RUN_MANIFEST_NAMESPACE}:{run_identity}"
        run_digest = artifact_store.put(
            kind="local-stream-run-manifest",
            logical_key=run_logical_key,
            payload=canonical_json_bytes(run_projection),
        )
        run_manifest = artifact_store.reference(
            kind="local-stream-run-manifest", logical_key=run_logical_key
        )
        if run_manifest.exact_sha256 != run_digest:
            raise LocalMageStreamExecutionError(
                "run manifest CAS reference changed during persistence"
            )
    return LocalMageStreamExecutionResult(
        pipeline_result=result,
        contexts=context_reports,
        queue_depth=max_inflight_observations,
        run_manifest=run_manifest,
        durable_execution=durable_execution,
    )


def _build_run_manifest_projection(
    *,
    plan: MageStreamPlan,
    selected_camera: CameraId,
    codec_policy_version: str,
    queue_depth: int,
    contexts: tuple[LocalMageStreamContextExecution, ...],
    pipeline_result: StreamPerceptionRunResult,
    durable_execution: LocalMageStreamDurableExecution | None,
    artifact_store: LocalPerceptionArtifactStore,
) -> dict[str, object]:
    """Build a deterministic durable root for one local Mage stream execution.

    Timing samples remain in per-context reports for hotspot analysis, but are not
    part of this root's identity.  The root instead binds the immutable plan,
    terminal closure, context reports, and accepted raw request bindings so a
    restarted process can discover the exact accepted artifacts without rerunning
    media preparation or model generation.
    """
    terminal = pipeline_result.terminal_artifacts
    terminal_projection: dict[str, object] | None = None
    if terminal is not None:
        terminal_projection = {
            "event_tracks": [
                _artifact_reference_projection(item) for item in terminal.event_tracks
            ],
            "fusion_decisions": [
                _artifact_reference_projection(item) for item in terminal.fusion_decisions
            ],
            "refine_requests": [
                _artifact_reference_projection(item) for item in terminal.refine_requests
            ],
            "terminal_manifest": _artifact_reference_projection(terminal.terminal_manifest),
        }
    context_hashes = {item.context_manifest_semantic_sha256 for item in contexts}
    accepted_bindings: list[dict[str, object]] = []
    for reference in artifact_store.references(kind="accepted-inference-binding"):
        try:
            document = json.loads(
                artifact_store.read(kind=reference.kind, logical_key=reference.logical_key)
            )
        except (KeyError, RuntimeError, json.JSONDecodeError):
            continue
        context = document.get("context") if isinstance(document, dict) else None
        if (
            isinstance(context, dict)
            and context.get("context_manifest_semantic_sha256") in context_hashes
        ):
            accepted_bindings.append(
                {key: value for key, value in _artifact_reference_projection(reference).items()}
            )
    accepted_bindings.sort(key=lambda item: str(item["logical_key"]))
    context_projection = [
        {
            "focus_segment_ordinal": item.focus_segment_ordinal,
            "context_manifest_key": item.context_manifest_key,
            "context_manifest_semantic_sha256": item.context_manifest_semantic_sha256,
            "persisted_report_exact_sha256": item.persisted_report_exact_sha256,
        }
        for item in sorted(contexts, key=lambda value: value.focus_segment_ordinal)
    ]
    stage_projection = [
        {
            "stage": measurement.stage.value,
            "invocation_count": measurement.invocation_count,
        }
        for measurement in pipeline_result.stage_measurements
    ]
    durable_projection: dict[str, object] | None = None
    if durable_execution is not None:
        durable_projection = {
            "run_key": durable_execution.run.run_key,
            "scheduler_policy_version": durable_execution.run.scheduler_policy_version,
            "config_sha256": durable_execution.run.config_sha256,
            "derived_work_sealed": durable_execution.run.derived_work_sealed,
            "finalization_state": durable_execution.finalization_state,
            "fusion_work_item_ids": list(durable_execution.fusion_work_item_ids),
            "pending_refinement_work_item_ids": list(
                durable_execution.pending_refinement_work_item_ids
            ),
            "stage_counts": [
                {
                    "stage": item.stage.value,
                    "planned": item.planned,
                    "ready": item.ready,
                    "leased": item.leased,
                    "running": item.running,
                    "retry_wait": item.retry_wait,
                    "succeeded": item.succeeded,
                    "failed_permanent": item.failed_permanent,
                }
                for item in durable_execution.snapshot.stage_counts
            ],
        }
    return {
        "manifest_version": LOCAL_MAGE_STREAM_RUN_MANIFEST_VERSION,
        "plan_key": plan.plan_key,
        "plan_semantic_sha256": plan.plan_semantic_sha256,
        "recording_key": plan.recording.recording_key,
        "recording_exact_sha256": plan.recording.recording_exact_sha256,
        "selected_camera": selected_camera.value,
        "codec_policy_version": codec_policy_version,
        "execution_policy_version": LOCAL_MAGE_STREAM_EXECUTION_POLICY_VERSION,
        "queue_depth": queue_depth,
        "contexts": context_projection,
        "terminal_artifacts": terminal_projection,
        "accepted_inference_bindings": accepted_bindings,
        "durable_execution": durable_projection,
        "stage_invocations": stage_projection,
        "normal_model_call_count": pipeline_result.normal_model_call_count,
        "refinement_model_call_count": pipeline_result.refinement_model_call_count,
    }


def _artifact_reference_projection(reference: LocalPerceptionArtifactReference) -> dict[str, str]:
    return {
        "kind": reference.kind,
        "logical_key": reference.logical_key,
        "exact_sha256": reference.exact_sha256,
    }


def _validate_local_v1_plan(plan: MageStreamPlan) -> None:
    if plan.policy.reasoning_horizon_duration_ns != plan.policy.scan_segment_duration_ns:
        raise LocalMageStreamExecutionError(
            "local Mage stream v1 requires reasoning_horizon_duration_ns to equal "
            "scan_segment_duration_ns; longer overlapping context replay is not enabled"
        )
    for storage_segment, context in zip(
        plan.storage_segments, plan.reasoning_contexts, strict=True
    ):
        if context.ordered_segments != (storage_segment,):
            raise LocalMageStreamExecutionError(
                "local Mage stream v1 requires each context to contain only its focus segment"
            )


def _measurement_map(
    measurements: tuple[object, ...],
) -> Mapping[PerceptionStage, tuple[int, float]]:
    values: dict[PerceptionStage, tuple[int, float]] = {}
    for measurement in measurements:
        stage = getattr(measurement, "stage", None)
        count = getattr(measurement, "invocation_count", None)
        elapsed = getattr(measurement, "elapsed_seconds", None)
        if not isinstance(stage, PerceptionStage):
            raise TypeError("session returned an invalid perception stage measurement")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TypeError("session returned an invalid perception stage count")
        if not isinstance(elapsed, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise TypeError("session returned an invalid perception stage elapsed time")
        values[stage] = (count, elapsed)
    return values


def _stage_count_delta(
    before: Mapping[PerceptionStage, tuple[int, float]],
    after: Mapping[PerceptionStage, tuple[int, float]],
    stage: PerceptionStage,
) -> int:
    return after[stage][0] - before[stage][0]


def _stage_elapsed_delta(
    before: Mapping[PerceptionStage, tuple[int, float]],
    after: Mapping[PerceptionStage, tuple[int, float]],
    stage: PerceptionStage,
) -> float:
    return after[stage][1] - before[stage][1]


def _subprocess_ffprobe_runner(command: tuple[str, ...]) -> FfmpegCommandResult:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return FfmpegCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


__all__ = [
    "LOCAL_MAGE_MEDIA_HEALTH_POLICY_VERSION",
    "LOCAL_MAGE_STREAM_CONTEXT_REPORT_NAMESPACE",
    "LOCAL_MAGE_STREAM_EXECUTION_POLICY_VERSION",
    "FfprobeCommandRunner",
    "LocalMageStreamContextExecution",
    "LocalMageStreamDurableExecution",
    "LocalMageStreamExecutionError",
    "LocalMageStreamExecutionResult",
    "LocalMaterializedSegmentResolver",
    "LocalMediaHealthScanner",
    "execute_local_mage_stream",
]
