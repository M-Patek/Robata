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
from typing import Final

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
from robata.perception.pipeline import (
    LocalPerceptionArtifactReference,
    LocalPerceptionArtifactStore,
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
class LocalMageStreamExecutionResult:
    """Whole-stream output plus per-context measured execution evidence."""

    pipeline_result: StreamPerceptionRunResult
    contexts: tuple[LocalMageStreamContextExecution, ...]
    queue_depth: int
    run_manifest: LocalPerceptionArtifactReference | None = None


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

        health = session.scan_media(scan_current_context)
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

    pending: deque[tuple[_PreparedStreamContext, Future[tuple[MageObservation, float]]]] = deque()
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
            pending.append((prepared, executor.submit(pipeline.observe_context, prepared.context)))
            return True

        while len(pending) < max_inflight_observations and submit_next():
            pass
        while pending:
            prepared, future = pending.popleft()
            observation, _provider_elapsed = future.result()
            before = _measurement_map(session.stage_measurements())
            session.consume_precomputed(
                context=prepared.context,
                media_health=prepared.media_health,
                observation=observation,
                observation_elapsed_seconds=_provider_elapsed,
            )
            after = _measurement_map(session.stage_measurements())
            append_report(prepared, before, after)
            while len(pending) < max_inflight_observations and submit_next():
                pass

    result = session.finalize()
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
    )


def _build_run_manifest_projection(
    *,
    plan: MageStreamPlan,
    selected_camera: CameraId,
    codec_policy_version: str,
    queue_depth: int,
    contexts: tuple[LocalMageStreamContextExecution, ...],
    pipeline_result: StreamPerceptionRunResult,
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
    "LocalMageStreamExecutionError",
    "LocalMageStreamExecutionResult",
    "LocalMaterializedSegmentResolver",
    "LocalMediaHealthScanner",
    "execute_local_mage_stream",
]
