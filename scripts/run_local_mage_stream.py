"""Plan or execute the additive Mage stream-oriented local path.

``--execute`` is a concrete in-repository single-worker producer/consumer path:
for each immutable focus segment it stream-copies one native-video file, measures
local media health, and sends one Mage observation request to the declared endpoint.
The default sustained qualification profile keeps two bounded observations in flight: one
active request and one preparation slot, while endpoint generation remains single-flight.
Results are consumed in ordinal order. No Qwen model is loaded, deleted, or selected
unless a caller uses the explicit legacy route.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.application.canonical.mage_stream import (  # noqa: E402
    DEFAULT_KEYFRAME_ALIGNMENT_TOLERANCE_NS,
    DEFAULT_REASONING_HORIZON_DURATION_NS,
    DEFAULT_SCAN_SEGMENT_DURATION_NS,
    AbsoluteNanosecondInterval,
    MageStreamMaterializationError,
    MageStreamMaterializer,
    MageStreamPlan,
    MageStreamPlanningError,
    MageStreamPolicy,
    MageStreamRecording,
    MageStreamSegmentationMode,
    exact_file_sha256,
    plan_keyframe_aligned_mage_stream,
    plan_mage_stream,
    probe_video_keyframe_offsets_ns,
)
from robata.application.canonical.mage_stream_execution import (  # noqa: E402
    LOCAL_MAGE_STREAM_EXECUTION_POLICY_VERSION,
    LocalMageStreamExecutionError,
    LocalMaterializedSegmentResolver,
    LocalMediaHealthScanner,
    execute_local_mage_stream,
)
from robata.application.canonical.perception_composition import (  # noqa: E402
    create_default_vnext_perception_scheduler,
)
from robata.application.canonical.perception_routing import (  # noqa: E402
    LEGACY_QWEN_WINDOW_PROFILE,
    MAGE_STREAM_VNEXT_PROFILE,
    resolve_perception_route,
)
from robata.benchmark.gpu_telemetry import NvidiaSmiGpuSampler  # noqa: E402
from robata.contracts.cameras import CAMERA_ID_VALUES, CameraId  # noqa: E402
from robata.inference.mage_video_adapter import (  # noqa: E402
    FileMageVideoResultArtifactReader,
    MageVideoObservationAdapter,
    MageVideoObservationAdapterConfig,
)
from robata.inference.mage_video_endpoint import (  # noqa: E402
    MageVideoCodecPolicy,
    MageVideoNeuralCodecParameters,
)
from robata.inference.mage_video_http_transport import (  # noqa: E402
    MageVideoHttpTransport,
    MageVideoHttpTransportError,
    fetch_mage_video_endpoint_health,
)
from robata.perception.fusion import (  # noqa: E402
    PerceptionFusionEngine,
    PerceptionFusionPolicy,
)
from robata.perception.pipeline import (  # noqa: E402
    LocalPerceptionArtifactStore,
    StreamPerceptionPipeline,
)
from robata.perception.projectors import (  # noqa: E402
    EventProjector,
    EvidenceProjector,
    QaProjector,
)
from robata.perception.single_route import (  # noqa: E402
    SingleCameraAuthority,
    SingleCameraAuthorityPolicy,
)
from robata.perception.tracking import EventTrackPolicy, EventTrackReconciler  # noqa: E402

MAGE_STREAM_FUSION_POLICY_VERSION = "mage-stream-fusion-v1"
MAGE_STREAM_TRACK_POLICY_VERSION = "mage-stream-track-v1"
MAGE_STREAM_REFINE_POLICY_VERSION = "mage-stream-refine-v1"
MAGE_STREAM_REFINE_PROMPT_VERSION = "mage-stream-refine-prompt-v1"
V1_LIMITATION = (
    "v1 uses exactly one selected native-video camera and one non-overlapping focus segment "
    "per normal Mage observation; it does not mosaic cameras, replay overlapping reasoning "
    "contexts, or make one business call per camera"
)


class LocalMageStreamRunnerError(RuntimeError):
    """The local runner lacks a requested input or cannot preserve stream semantics."""


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_seconds_to_ns(value: str) -> int:
    try:
        seconds = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a positive decimal number of seconds") from error
    if not seconds.is_finite() or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive decimal number of seconds")
    nanoseconds = seconds * Decimal(1_000_000_000)
    if nanoseconds != nanoseconds.to_integral_value():
        raise argparse.ArgumentTypeError("must resolve to a whole number of nanoseconds")
    return int(nanoseconds)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="one selected camera native video source")
    parser.add_argument(
        "--recording-key",
        default=None,
        help="stable source recording key; defaults to source filename stem",
    )
    parser.add_argument(
        "--recording-start-ns",
        "--start-ns",
        dest="recording_start_ns",
        type=_nonnegative_int,
        required=True,
        help="absolute source-clock start timestamp in nanoseconds",
    )
    parser.add_argument(
        "--recording-end-ns",
        "--end-ns",
        dest="recording_end_ns",
        type=_nonnegative_int,
        required=True,
        help="absolute source-clock end timestamp in nanoseconds",
    )
    parser.add_argument(
        "--scan-segment-seconds",
        "--segment-seconds",
        dest="scan_segment_duration_ns",
        type=_positive_seconds_to_ns,
        default=DEFAULT_SCAN_SEGMENT_DURATION_NS,
        help="non-overlapping immutable focus segment duration; default 8 seconds",
    )
    parser.add_argument(
        "--reasoning-horizon-seconds",
        dest="reasoning_horizon_duration_ns",
        type=_positive_seconds_to_ns,
        default=DEFAULT_REASONING_HORIZON_DURATION_NS,
        help=(
            "causal reasoning horizon; --execute v1 requires it to equal the scan segment "
            "duration to avoid replaying visual input"
        ),
    )
    parser.add_argument("--policy-version", default="mage-stream-planner-v1")
    parser.add_argument(
        "--segment-boundary-mode",
        choices=("auto", "fixed", "keyframe_aligned"),
        default="auto",
        help=(
            "auto uses keyframe-aligned boundaries for materialize/execute and fixed "
            "boundaries for plan-only output"
        ),
    )
    parser.add_argument(
        "--keyframe-alignment-tolerance-ns",
        type=_positive_int,
        default=DEFAULT_KEYFRAME_ALIGNMENT_TOLERANCE_NS,
        help="maximum absolute slip from each nominal segment boundary",
    )
    parser.add_argument("--codec-policy-version", default="mage-video-codec-policy-v2")
    parser.add_argument("--camera", choices=CAMERA_ID_VALUES, default=CameraId.CAM_01.value)
    parser.add_argument(
        "--profile",
        choices=(MAGE_STREAM_VNEXT_PROFILE, LEGACY_QWEN_WINDOW_PROFILE),
        default=MAGE_STREAM_VNEXT_PROFILE,
        help="Mage vNext is the default; Qwen is retained as an explicit legacy route",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", "--plan-only", dest="dry_run", action="store_true")
    mode.add_argument(
        "--materialize",
        action="store_true",
        help="stream-copy planned native segments without endpoint inference",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run the in-repository single-worker Mage producer/consumer composition",
    )
    parser.add_argument(
        "--materialization-dir",
        type=Path,
        default=REPOSITORY_ROOT / ".local" / "mage-stream-materialization",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPOSITORY_ROOT / ".local" / "mage-stream-artifacts",
        help="local exact-byte CAS for observation/projection/report artifacts",
    )
    parser.add_argument(
        "--scheduler-db",
        type=Path,
        default=None,
        help="SQLite path for the durable Mage vNext scheduler; defaults inside --artifact-dir",
    )
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--ffprobe-binary", default="ffprobe")
    parser.add_argument(
        "--endpoint-url",
        default="http://127.0.0.1:8102",
        help="declared Mage endpoint base URL; endpoint health supplies its model identity",
    )
    parser.add_argument(
        "--endpoint-timeout-seconds",
        type=_positive_float,
        default=7_260.0,
        help="HTTP endpoint timeout for one idempotent native-video request",
    )
    parser.add_argument(
        "--gpu-sample-interval-seconds",
        type=_positive_float,
        default=0.25,
        help="nvidia-smi sampling interval for full-wall local qualification telemetry",
    )
    parser.add_argument(
        "--gpu-telemetry-output",
        type=Path,
        default=None,
        help="GPU telemetry JSON path; defaults to <artifact-dir>/gpu-telemetry.json",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=256,
        help=(
            "compact Mage decoder output budget for sustained native execution; "
            "identity-bound and qualification-gated"
        ),
    )
    parser.add_argument(
        "--max-inflight-observations",
        type=_positive_int,
        choices=(1, 2),
        default=2,
        help=(
            "bounded native-video requests in flight; default sustained mode is 2 "
            "(one active generation plus one preparation slot); model generation "
            "remains provider-serialized"
        ),
    )
    parser.add_argument("--codec-mode", choices=("traditional", "neural"), default="traditional")
    parser.add_argument("--codec-target-canvas", type=_positive_int, default=32)
    parser.add_argument("--codec-group-size", type=_positive_int, default=32)
    parser.add_argument("--codec-images-per-group", type=_positive_int, default=4)
    parser.add_argument("--codec-patch-size", type=_positive_int, default=16)
    parser.add_argument("--codec-max-pixels", type=_positive_int, default=150_000)
    parser.add_argument("--codec-min-group-frames", type=_positive_int, default=8)
    parser.add_argument("--codec-max-group-frames", type=_positive_int, default=64)
    parser.add_argument("--codec-timeout-seconds", type=_positive_int, default=7_200)
    parser.add_argument("--preprocess-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--neural-qp", type=_nonnegative_int, default=42)
    parser.add_argument("--neural-reset-interval", type=_positive_int, default=64)
    parser.add_argument("--neural-intra-period", type=int, default=-1)
    parser.add_argument("--neural-max-side", type=_nonnegative_int, default=0)
    parser.add_argument("--neural-sequence-length-frames", type=_nonnegative_int, default=0)
    parser.add_argument("--neural-canvas-token-side", type=_positive_int, default=None)
    parser.add_argument("--neural-readiness-coverage-bins", type=_positive_int, default=3)
    parser.add_argument("--neural-readiness-delta-ratio", type=_positive_float, default=0.05)
    parser.add_argument("--neural-bitcost-percentile", type=_positive_int, default=99)
    parser.add_argument("--neural-decode-backsearch-max", type=_positive_int, default=16)
    parser.add_argument("--output", type=Path, default=None, help="write JSON report to this path")
    return parser


def _stream_plan(
    *,
    arguments: argparse.Namespace,
    recording: MageStreamRecording,
    source: Path,
) -> MageStreamPlan:
    requested_mode = arguments.segment_boundary_mode
    if requested_mode == "auto":
        requested_mode = (
            "keyframe_aligned" if arguments.execute or arguments.materialize else "fixed"
        )
    if requested_mode == "fixed":
        if arguments.execute or arguments.materialize:
            raise LocalMageStreamRunnerError(
                "native codec materialization requires keyframe_aligned segment boundaries"
            )
        return plan_mage_stream(
            recording=recording,
            policy=MageStreamPolicy(
                scan_segment_duration_ns=arguments.scan_segment_duration_ns,
                reasoning_horizon_duration_ns=arguments.reasoning_horizon_duration_ns,
                policy_version=arguments.policy_version,
            ),
        )
    policy = MageStreamPolicy(
        scan_segment_duration_ns=arguments.scan_segment_duration_ns,
        reasoning_horizon_duration_ns=arguments.reasoning_horizon_duration_ns,
        policy_version=arguments.policy_version,
        segmentation_mode=MageStreamSegmentationMode.KEYFRAME_ALIGNED,
        keyframe_alignment_tolerance_ns=arguments.keyframe_alignment_tolerance_ns,
    )
    keyframe_offsets_ns = probe_video_keyframe_offsets_ns(
        source,
        ffprobe_binary=arguments.ffprobe_binary,
    )
    return plan_keyframe_aligned_mage_stream(
        recording=recording,
        policy=policy,
        keyframe_offsets_ns=keyframe_offsets_ns,
    )


def _plan_document(plan: MageStreamPlan) -> dict[str, object]:
    return {
        "plan_key": plan.plan_key,
        "plan_semantic_sha256": plan.plan_semantic_sha256,
        "recording": {
            "recording_key": plan.recording.recording_key,
            "recording_exact_sha256": plan.recording.recording_exact_sha256,
            "interval": plan.recording.interval.as_projection(),
        },
        "policy": {
            "policy_version": plan.policy.policy_version,
            "segmentation_mode": plan.policy.segmentation_mode.value,
            "keyframe_alignment_tolerance_ns": (plan.policy.keyframe_alignment_tolerance_ns),
            "scan_segment_duration_ns": plan.policy.scan_segment_duration_ns,
            "reasoning_horizon_duration_ns": plan.policy.reasoning_horizon_duration_ns,
        },
        "storage_segments": [
            {
                "ordinal": segment.ordinal,
                "segment_key": segment.segment_key,
                "segment_semantic_sha256": segment.segment_semantic_sha256,
                "interval": segment.interval.as_projection(),
            }
            for segment in plan.storage_segments
        ],
        "reasoning_contexts": [
            {
                "focus_segment_ordinal": context.focus_segment_ordinal,
                "context_key": context.context_key,
                "context_semantic_sha256": context.context_semantic_sha256,
                "reasoning_horizon": context.reasoning_horizon.as_projection(),
                "materialized_interval": context.materialized_interval.as_projection(),
                "storage_segment_ordinals": [item.ordinal for item in context.ordered_segments],
            }
            for context in plan.reasoning_contexts
        ],
    }


def _codec_policy(arguments: argparse.Namespace) -> MageVideoCodecPolicy:
    neural_parameters = None
    if arguments.codec_mode == "neural":
        neural_parameters = MageVideoNeuralCodecParameters(
            quantization_parameter=arguments.neural_qp,
            reset_interval=arguments.neural_reset_interval,
            intra_period=arguments.neural_intra_period,
            max_side=arguments.neural_max_side,
            sequence_length_frames=arguments.neural_sequence_length_frames,
            canvas_token_side=arguments.neural_canvas_token_side,
            readiness_coverage_bins=arguments.neural_readiness_coverage_bins,
            readiness_delta_ratio=arguments.neural_readiness_delta_ratio,
            bitcost_percentile=arguments.neural_bitcost_percentile,
            decode_backsearch_max=arguments.neural_decode_backsearch_max,
        )
    return MageVideoCodecPolicy(
        policy_version=arguments.codec_policy_version,
        codec_mode=arguments.codec_mode,
        target_canvas=arguments.codec_target_canvas,
        group_size=arguments.codec_group_size,
        images_per_group=arguments.codec_images_per_group,
        patch_size=arguments.codec_patch_size,
        max_pixels=arguments.codec_max_pixels,
        min_group_frames=arguments.codec_min_group_frames,
        max_group_frames=arguments.codec_max_group_frames,
        timeout_seconds=arguments.codec_timeout_seconds,
        preprocess_device=arguments.preprocess_device,
        neural_parameters=neural_parameters,
    )


def _single_route_authority(arguments: argparse.Namespace) -> SingleCameraAuthority:
    return SingleCameraAuthority(
        SingleCameraAuthorityPolicy(
            camera_id=CameraId(arguments.camera),
            max_inflight_observations=arguments.max_inflight_observations,
        )
    )


def _execute(
    *,
    arguments: argparse.Namespace,
    plan: MageStreamPlan,
    source: Path,
) -> dict[str, object]:
    _validate_v1_execution_policy(plan)
    endpoint_health = fetch_mage_video_endpoint_health(
        endpoint_url=arguments.endpoint_url,
        timeout_seconds=arguments.endpoint_timeout_seconds,
    )
    codec_policy = _codec_policy(arguments)
    authority = _single_route_authority(arguments)
    resolver = LocalMaterializedSegmentResolver()
    artifact_store = LocalPerceptionArtifactStore(arguments.artifact_dir)
    scheduler_path = (
        arguments.scheduler_db
        if arguments.scheduler_db is not None
        else arguments.artifact_dir / "perception-vnext.sqlite3"
    )
    durable_scheduler = create_default_vnext_perception_scheduler(
        scheduler_path,
        profile=arguments.profile,
    )
    adapter = MageVideoObservationAdapter(
        model_identity=endpoint_health.model_identity,
        codec_policy=codec_policy,
        segment_resolver=resolver,
        transport=MageVideoHttpTransport(
            endpoint_url=arguments.endpoint_url,
            timeout_seconds=arguments.endpoint_timeout_seconds,
        ),
        artifact_reader=FileMageVideoResultArtifactReader(),
        config=MageVideoObservationAdapterConfig(max_new_tokens=arguments.max_new_tokens),
        accepted_binding_sink=artifact_store,
    )
    pipeline = StreamPerceptionPipeline(
        provider=adapter,
        qa_projector=QaProjector(),
        event_projector=EventProjector(),
        evidence_projector=EvidenceProjector(),
        reconciler=EventTrackReconciler(EventTrackPolicy(version=MAGE_STREAM_TRACK_POLICY_VERSION)),
        fusion_engine=PerceptionFusionEngine(
            PerceptionFusionPolicy(version=MAGE_STREAM_FUSION_POLICY_VERSION)
        ),
        refine_policy_version=MAGE_STREAM_REFINE_POLICY_VERSION,
        refine_prompt_version=MAGE_STREAM_REFINE_PROMPT_VERSION,
        artifact_sink=artifact_store,
    )
    gpu_sampler = NvidiaSmiGpuSampler(interval_seconds=arguments.gpu_sample_interval_seconds)
    gpu_sampler.start()
    try:
        result = execute_local_mage_stream(
            plan=plan,
            source_path=source,
            selected_camera=CameraId(arguments.camera),
            materializer=MageStreamMaterializer(
                ffmpeg_binary=arguments.ffmpeg_binary,
                ffprobe_binary=arguments.ffprobe_binary,
                verify_packet_boundaries=True,
            ),
            codec_policy_version=codec_policy.policy_version,
            resolver=resolver,
            pipeline=pipeline,
            artifact_store=artifact_store,
            media_health_scanner=LocalMediaHealthScanner(ffprobe_binary=arguments.ffprobe_binary),
            materialization_root=arguments.materialization_dir,
            max_inflight_observations=arguments.max_inflight_observations,
            durable_scheduler=durable_scheduler,
        )
    finally:
        gpu_telemetry = gpu_sampler.stop()
        gpu_telemetry_path = (
            arguments.gpu_telemetry_output
            if arguments.gpu_telemetry_output is not None
            else arguments.artifact_dir / "gpu-telemetry.json"
        )
        _write_output(gpu_telemetry_path, gpu_telemetry.to_payload())
    durable_execution = result.durable_execution
    if durable_execution is None:
        raise LocalMageStreamRunnerError(
            "Mage vNext --execute did not receive durable scheduler execution state"
        )
    return {
        "performed": True,
        "execution_policy_version": LOCAL_MAGE_STREAM_EXECUTION_POLICY_VERSION,
        "single_route": authority.as_projection(),
        "decoder": {"max_new_tokens": arguments.max_new_tokens},
        "queue_depth": result.queue_depth,
        "execution_profile": result.execution_profile.value,
        "execution_timing": result.timing.as_projection(),
        "gpu_telemetry": {
            "path": str(gpu_telemetry_path.expanduser().resolve()),
            "measurement_status": gpu_telemetry.measurement_status.value,
            "sample_count": len(gpu_telemetry.samples),
            "device_summaries": [item.to_payload() for item in gpu_telemetry.summary],
            "errors": list(gpu_telemetry.errors),
        },
        "durable_execution": {
            "database_path": str(durable_scheduler.database_path),
            "run": {
                "run_key": durable_execution.run.run_key,
                "plan_key": durable_execution.run.plan_key,
                "plan_semantic_sha256": durable_execution.run.plan_semantic_sha256,
                "codec_policy_version": durable_execution.run.codec_policy_version,
                "scheduler_policy_version": durable_execution.run.scheduler_policy_version,
                "config_sha256": durable_execution.run.config_sha256,
                "derived_work_sealed": durable_execution.run.derived_work_sealed,
            },
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
        },
        "run_manifest": (
            None
            if result.run_manifest is None
            else {
                "kind": result.run_manifest.kind,
                "logical_key": result.run_manifest.logical_key,
                "exact_sha256": result.run_manifest.exact_sha256,
            }
        ),
        "endpoint": {
            "url": arguments.endpoint_url,
            "model_identity": endpoint_health.model_identity.model_dump(mode="json"),
            "codec_policy": codec_policy.model_dump(mode="json"),
        },
        "normal_model_call_count": result.pipeline_result.normal_model_call_count,
        "refinement_model_call_count": result.pipeline_result.refinement_model_call_count,
        "total_model_call_count": result.pipeline_result.total_model_call_count,
        "stage_measurements": [
            {
                "stage": item.stage.value,
                "invocation_count": item.invocation_count,
                "elapsed_seconds": item.elapsed_seconds,
            }
            for item in result.pipeline_result.stage_measurements
        ],
        "contexts": [
            {
                **item.as_projection(),
                "persisted_report_exact_sha256": item.persisted_report_exact_sha256,
            }
            for item in result.contexts
        ],
        "event_tracks": [
            item.model_dump(mode="json") for item in result.pipeline_result.event_tracks
        ],
        "fusion_decisions": [
            item.model_dump(mode="json") for item in result.pipeline_result.fusion_decisions
        ],
        "refine_requests": [
            item.model_dump(mode="json") for item in result.pipeline_result.refine_requests
        ],
    }


def _validate_v1_execution_policy(plan: MageStreamPlan) -> None:
    if plan.policy.reasoning_horizon_duration_ns != plan.policy.scan_segment_duration_ns:
        raise LocalMageStreamRunnerError(
            "--execute requires --reasoning-horizon-seconds to equal --scan-segment-seconds "
            "for Mage stream v1; overlapping visual replay is deliberately disabled"
        )
    if any(len(context.ordered_segments) != 1 for context in plan.reasoning_contexts):
        raise LocalMageStreamRunnerError(
            "--execute requires exactly one non-overlapping focus segment per context"
        )


def _materialize_only(
    *,
    arguments: argparse.Namespace,
    plan: MageStreamPlan,
    source: Path,
) -> list[dict[str, object]]:
    materializer = MageStreamMaterializer(
        ffmpeg_binary=arguments.ffmpeg_binary,
        ffprobe_binary=arguments.ffprobe_binary,
        verify_packet_boundaries=True,
    )
    return [
        {
            "ordinal": segment.ordinal,
            "segment_semantic_sha256": segment.segment_semantic_sha256,
            "durable_path": str(
                materializer.materialize_storage_segment(
                    plan=plan,
                    source_path=source,
                    segment=segment,
                    output_root=arguments.materialization_dir,
                ).durable_path
            ),
        }
        for segment in plan.storage_segments
    ]


def _write_output(path: Path, payload: dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as error:
        raise LocalMageStreamRunnerError("could not write local Mage stream output") from error


def _report_error(arguments: argparse.Namespace, error: BaseException) -> int:
    payload = {
        "ok": False,
        "code": "LOCAL_MAGE_STREAM_FAILED",
        "detail": str(error),
        "profile": arguments.profile,
        "mage_default_profile": MAGE_STREAM_VNEXT_PROFILE,
        "qwen_weights_preserved": True,
        "v1_limitation": V1_LIMITATION,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        route = resolve_perception_route(
            arguments.profile,
            allow_explicit_legacy_qwen=(arguments.profile == LEGACY_QWEN_WINDOW_PROFILE),
        )
        source = arguments.source.expanduser().resolve()
        source_sha256, source_byte_count = exact_file_sha256(source)
        recording = MageStreamRecording(
            recording_key=arguments.recording_key or source.stem,
            recording_exact_sha256=source_sha256,
            interval=AbsoluteNanosecondInterval(
                arguments.recording_start_ns,
                arguments.recording_end_ns,
            ),
        )
        if arguments.profile == LEGACY_QWEN_WINDOW_PROFILE and arguments.execute:
            raise LocalMageStreamRunnerError(
                "legacy Qwen execution is deliberately not performed by the Mage stream runner"
            )
        plan = _stream_plan(arguments=arguments, recording=recording, source=source)
        report: dict[str, object] = {
            "ok": True,
            "profile": route.profile.value,
            "mage_default_profile": MAGE_STREAM_VNEXT_PROFILE,
            "composition": {
                "mode": route.execution_mode.value,
                "version": route.composition_version,
                "scheduler_policy_version": route.scheduler_policy_version,
                "qwen_autoload": False,
            },
            "route": {
                "native_media_type": route.native_media_type,
                "normal_model_stage": route.normal_model_stage.value,
                "refinement_model_stage": route.refinement_model_stage.value,
                "explicit_legacy_only": route.explicit_legacy_only,
            },
            "qwen_weights_preserved": True,
            "v1_limitation": V1_LIMITATION,
            "selected_camera": arguments.camera,
            "single_route": _single_route_authority(arguments).as_projection(),
            "decoder": {"max_new_tokens": arguments.max_new_tokens},
            "segment_boundary_mode": plan.policy.segmentation_mode.value,
            "source_path": str(source),
            "source_byte_count": source_byte_count,
            "plan": _plan_document(plan),
            "execution": {"requested": bool(arguments.execute), "performed": False},
        }
        if arguments.profile == LEGACY_QWEN_WINDOW_PROFILE:
            report["legacy_route"] = {
                "status": "EXPLICIT_LEGACY_ONLY",
                "message": (
                    "Qwen weights and legacy scripts are retained; this Mage stream runner "
                    "does not autoload, delete, or route to Qwen. Use "
                    "scripts/run_local_qwen_canonical.py explicitly."
                ),
            }
        elif arguments.execute:
            report["execution"] = _execute(arguments=arguments, plan=plan, source=source)
        elif arguments.materialize:
            report["materialized_storage_segments"] = _materialize_only(
                arguments=arguments,
                plan=plan,
                source=source,
            )
        if arguments.output is not None:
            _write_output(arguments.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    except (
        LocalMageStreamExecutionError,
        LocalMageStreamRunnerError,
        MageStreamMaterializationError,
        MageStreamPlanningError,
        MageVideoHttpTransportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return _report_error(arguments, error)


if __name__ == "__main__":
    raise SystemExit(main())
