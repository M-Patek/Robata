"""Compare frozen Mage native-video evidence with Qwen on aligned cam_01 frames.

Every real model invocation must be owned by ``run_bounded_qwen_batch_benchmark.py``.
This child process deliberately makes no production claim: Mage sees the frozen native
video derivative while Qwen sees six exact r12 PNG derivatives per aligned segment.
Agreement is therefore model-to-model agreement on aligned evidence, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robata.benchmark.gpu_telemetry import NvidiaSmiGpuSampler  # noqa: E402
from robata.benchmark.qwen_mage_common_projection import (  # noqa: E402
    COMMON_FRAMES_PER_SEGMENT,
    COMMON_QWEN_MODEL_FAMILY,
    CommonProjectionCase,
    CommonProjectionError,
    CommonProjectionFixture,
    CommonQwenProjection,
    action_token_f1,
    build_qwen_common_prompt,
    compare_observations,
    downstream_projection,
    interval_iou,
    load_common_projection_fixture,
    load_selected_frame_payloads,
    project_qwen_compact_output,
    run_common_downstream,
)
from robata.benchmark.qwen_r12_request_corpus import (  # noqa: E402
    QWEN_R12_20260806_EXPECTED,
    load_qwen_request_corpus,
)
from robata.contracts.cameras import CameraId  # noqa: E402
from robata.contracts.hashing import exact_bytes_sha256  # noqa: E402
from robata.inference.local_hf_runtime import (  # noqa: E402
    LocalHfBatchGenerationRequest,
    LocalHfBatchMemberObservation,
    LocalHuggingFaceVisionRuntime,
)

REPORT_VERSION: Final = "qwen-mage-common-cam01-five-segment-qualification-v1"
DEFAULT_CHECKPOINT_MANIFEST_SHA256: Final = (
    "1f7293b2629473f0240c8675025e1402da4306f05cc9026adf4c801f20f99f10"
)
TARGET_AGGREGATE_RTF: Final = 25.0
EXIT_FAILURE: Final = 2


class CommonProjectionBenchmarkError(RuntimeError):
    """The bounded common-projection qualification could not be completed honestly."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _write_json(path: Path, payload: object) -> str:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    body = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = resolved.with_name(f".{resolved.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(body)
        os.replace(temporary, resolved)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return exact_bytes_sha256(body)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--corpus-db", type=Path, required=True)
    parser.add_argument("--mage-stream-artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("serial", "batch4"), default="serial")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-image-side", type=_positive_int, default=448)
    parser.add_argument("--gpu-weight-memory-gib", type=_positive_int, default=7)
    parser.add_argument("--cpu-weight-memory-gib", type=_positive_int, default=1)
    parser.add_argument("--gpu-sample-interval-seconds", type=_positive_float, default=0.25)
    parser.add_argument(
        "--expected-checkpoint-manifest-sha256",
        default=DEFAULT_CHECKPOINT_MANIFEST_SHA256,
    )
    return parser


def _case_max_new_tokens(case: CommonProjectionCase) -> int:
    value = int(case.binding.endpoint_request.decoder.max_new_tokens)
    if value <= 0:
        raise CommonProjectionBenchmarkError("frozen Mage decoder output budget is invalid")
    return value


def _load_frame_payloads(
    fixture: CommonProjectionFixture,
) -> tuple[tuple[bytes, ...], ...]:
    payload_by_sha: dict[str, bytes] = {}
    result: list[tuple[bytes, ...]] = []
    for case in fixture.cases:
        payloads: list[bytes] = []
        for frame, payload in zip(
            case.selected_frames,
            load_selected_frame_payloads(case),
            strict=True,
        ):
            prior = payload_by_sha.setdefault(frame.sha256, payload)
            if prior != payload:
                raise CommonProjectionBenchmarkError(
                    "same frame digest resolved to different bytes"
                )
            payloads.append(prior)
        result.append(tuple(payloads))
    return tuple(result)


def _observation_projection(projection: CommonQwenProjection) -> dict[str, object]:
    observation = projection.observation
    qa = observation.semantic_qa.root[CameraId.CAM_01]
    return {
        "observation_semantic_sha256": observation.observation_semantic_sha256,
        "raw_output_exact_sha256": projection.raw_output_exact_sha256,
        "inference_artifact_exact_sha256": projection.inference_artifact_exact_sha256,
        "selected_camera_qa": {
            "disposition": qa.disposition.value,
            "issue_codes": [issue.code for issue in qa.issues],
            "confidence": qa.confidence,
        },
        "observations": [
            {
                "action": item.action,
                "start_ns": str(item.interval.start_ns),
                "end_ns": str(item.interval.end_ns),
                "confidence": item.confidence,
            }
            for item in observation.observations
        ],
        "diagnostics": list(projection.diagnostics),
    }


def _mage_observation_projection(case: CommonProjectionCase) -> dict[str, object]:
    observation = case.mage_observation
    qa = observation.semantic_qa.root[CameraId.CAM_01]
    return {
        "observation_semantic_sha256": observation.observation_semantic_sha256,
        "selected_camera_qa": {
            "disposition": qa.disposition.value,
            "issue_codes": [issue.code for issue in qa.issues],
            "confidence": qa.confidence,
        },
        "observations": [
            {
                "action": item.action,
                "start_ns": str(item.interval.start_ns),
                "end_ns": str(item.interval.end_ns),
                "confidence": item.confidence,
            }
            for item in observation.observations
        ],
    }


def _member_projection(member: LocalHfBatchMemberObservation) -> dict[str, object]:
    return {
        "rendered_image_sizes": [list(size) for size in member.rendered_image_sizes],
        "prompt_tokens": member.prompt_tokens,
        "output_tokens": member.output_tokens,
        "output_text": member.output_text,
        "output_exact_sha256": exact_bytes_sha256(member.output_text.encode("utf-8")),
    }


def _raw_output_diagnostic(output_text: str) -> dict[str, object]:
    try:
        value = json.loads(output_text)
    except json.JSONDecodeError as error:
        return {
            "valid_json": False,
            "root_type": None,
            "error": str(error),
        }
    diagnostic: dict[str, object] = {
        "valid_json": True,
        "root_type": type(value).__name__,
    }
    if isinstance(value, dict):
        diagnostic["root_keys"] = sorted(str(key) for key in value)
    elif isinstance(value, list):
        diagnostic["array_length"] = len(value)
        diagnostic["member_key_sets"] = [
            sorted(str(key) for key in item) if isinstance(item, dict) else None for item in value
        ]
    return diagnostic


def _chunks(count: int, *, mode: str) -> tuple[tuple[int, ...], ...]:
    if count <= 0:
        raise CommonProjectionBenchmarkError("common fixture is empty")
    size = 1 if mode == "serial" else 4
    return tuple(tuple(range(start, min(start + size, count))) for start in range(0, count, size))


def _compare_projected_actions(
    mage_records: list[dict[str, object]],
    qwen_records: list[dict[str, object]],
) -> dict[str, object]:
    remaining = set(range(len(qwen_records)))
    matches: list[dict[str, object]] = []
    for mage_index, expected in enumerate(mage_records):
        candidates: list[tuple[float, float, float, int]] = []
        for qwen_index in sorted(remaining):
            actual = qwen_records[qwen_index]
            label = action_token_f1(str(expected["action"]), str(actual["action"]))
            temporal = interval_iou(
                int(str(expected["start_ns"])),
                int(str(expected["end_ns"])),
                int(str(actual["start_ns"])),
                int(str(actual["end_ns"])),
            )
            candidates.append(((label + temporal) / 2.0, label, temporal, qwen_index))
        if not candidates:
            matches.append(
                {
                    "mage_index": mage_index,
                    "qwen_index": None,
                    "mage_action": expected["action"],
                    "qwen_action": None,
                    "label_token_f1": 0.0,
                    "temporal_iou": 0.0,
                }
            )
            continue
        _, label, temporal, qwen_index = max(
            candidates, key=lambda item: (item[0], item[1], item[2], -item[3])
        )
        remaining.remove(qwen_index)
        matches.append(
            {
                "mage_index": mage_index,
                "qwen_index": qwen_index,
                "mage_action": expected["action"],
                "qwen_action": qwen_records[qwen_index]["action"],
                "label_token_f1": label,
                "temporal_iou": temporal,
            }
        )
    label_values = [float(item["label_token_f1"]) for item in matches]
    temporal_values = [float(item["temporal_iou"]) for item in matches]
    return {
        "mage_count": len(mage_records),
        "qwen_count": len(qwen_records),
        "unmatched_qwen_count": len(remaining),
        "mean_label_token_f1": sum(label_values) / len(label_values) if label_values else None,
        "mean_temporal_iou": (
            sum(temporal_values) / len(temporal_values) if temporal_values else None
        ),
        "matches": matches,
        "unmatched_qwen_actions": [qwen_records[index]["action"] for index in sorted(remaining)],
    }


def _downstream_agreement(
    mage_projection: dict[str, object], qwen_projection: dict[str, object]
) -> dict[str, object]:
    mage_tracks = list(mage_projection["event_tracks"])  # type: ignore[arg-type]
    qwen_tracks = list(qwen_projection["event_tracks"])  # type: ignore[arg-type]
    mage_fusion = list(mage_projection["fusion_decisions"])  # type: ignore[arg-type]
    qwen_fusion = list(qwen_projection["fusion_decisions"])  # type: ignore[arg-type]
    return {
        "authority": "UNLABELED_MODEL_AGREEMENT_ONLY",
        "is_ground_truth_accuracy": False,
        "event_tracks": _compare_projected_actions(mage_tracks, qwen_tracks),
        "fusion_decisions": _compare_projected_actions(mage_fusion, qwen_fusion),
        "mage_refine_request_count": mage_projection["refine_request_count"],
        "qwen_refine_request_count": qwen_projection["refine_request_count"],
    }


def _capacity_projection(
    *, duration_seconds: float, recurring_wall_seconds: float
) -> dict[str, object]:
    if duration_seconds <= 0.0 or recurring_wall_seconds <= 0.0:
        raise CommonProjectionBenchmarkError("capacity inputs must be positive")
    rtf = duration_seconds / recurring_wall_seconds
    return {
        "source_recording_seconds": duration_seconds,
        "source_camera_seconds": duration_seconds,
        "recurring_wall_seconds": recurring_wall_seconds,
        "recording_realtime_factor": rtf,
        "camera_realtime_factor": rtf,
        "target_aggregate_camera_realtime_factor": TARGET_AGGREGATE_RTF,
        "local_equivalent_lanes_for_25x": math.ceil(TARGET_AGGREGATE_RTF / rtf),
        "production_qualification": "NOT_CLAIMED",
    }


def _load_frozen_mage_receipt(mage_root: Path) -> dict[str, object] | None:
    path = mage_root.expanduser().resolve().parent / "stream-report.json"
    if not path.is_file():
        return None
    body = path.read_bytes()
    try:
        document = json.loads(body)
        timing = document["execution"]["execution_timing"]
        stage_measurements = document["execution"]["stage_measurements"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise CommonProjectionBenchmarkError(
            f"invalid frozen Mage stream receipt: {path}"
        ) from error
    return {
        "path": str(path),
        "exact_sha256": exact_bytes_sha256(body),
        "execution_timing": timing,
        "stage_measurements": stage_measurements,
    }


def _run_candidate(
    *,
    runtime: LocalHuggingFaceVisionRuntime,
    fixture: CommonProjectionFixture,
    frame_payloads: tuple[tuple[bytes, ...], ...],
    mode: str,
    checkpoint_manifest_sha256: str,
) -> tuple[dict[str, object], tuple[CommonQwenProjection, ...] | None]:
    if len(frame_payloads) != len(fixture.cases):
        raise CommonProjectionBenchmarkError("frame payload/case count differs")
    physical_reports: list[dict[str, object]] = []
    case_reports: list[dict[str, object]] = []
    parsed: list[CommonQwenProjection | None] = [None] * len(fixture.cases)
    observation_elapsed: list[float] = [0.0] * len(fixture.cases)
    generation_sum = 0.0
    physical_wall_sum = 0.0
    parse_sum = 0.0
    recurring_started = time.perf_counter()

    for physical_ordinal, indices in enumerate(_chunks(len(fixture.cases), mode=mode)):
        call_started = time.perf_counter()
        if mode == "serial":
            index = indices[0]
            case = fixture.cases[index]
            generated = runtime.generate(
                image_payloads=frame_payloads[index],
                prompt=build_qwen_common_prompt(case),
                max_new_tokens=_case_max_new_tokens(case),
            )
            members = (
                LocalHfBatchMemberObservation(
                    rendered_image_sizes=generated.rendered_image_sizes,
                    prompt_tokens=generated.prompt_tokens,
                    output_tokens=generated.output_tokens,
                    output_text=generated.output_text,
                ),
            )
            generation_seconds = generated.generation_seconds
            peak_bytes = generated.gpu_peak_allocated_bytes
            physical_mode = "SERIAL_GENERATE_V1"
        else:
            budgets = {_case_max_new_tokens(fixture.cases[index]) for index in indices}
            if len(budgets) != 1:
                raise CommonProjectionBenchmarkError("batch members have different output budgets")
            batch = runtime.generate_batch(
                requests=tuple(
                    LocalHfBatchGenerationRequest(
                        image_payloads=frame_payloads[index],
                        prompt=build_qwen_common_prompt(fixture.cases[index]),
                        max_new_tokens=_case_max_new_tokens(fixture.cases[index]),
                    )
                    for index in indices
                )
            )
            members = batch.members
            generation_seconds = batch.physical_generation_seconds
            peak_bytes = batch.physical_gpu_peak_allocated_bytes
            physical_mode = "NATIVE_BATCH_GENERATE_V1"
        call_wall = time.perf_counter() - call_started
        if len(members) != len(indices):
            raise CommonProjectionBenchmarkError("runtime member count differs from physical batch")
        generation_sum += generation_seconds
        physical_wall_sum += call_wall
        equal_share = generation_seconds / len(indices)
        for index in indices:
            observation_elapsed[index] = equal_share
        physical_reports.append(
            {
                "physical_ordinal": physical_ordinal,
                "physical_mode": physical_mode,
                "case_ordinals": list(indices),
                "member_count": len(indices),
                "physical_call_wall_seconds": call_wall,
                "physical_generation_seconds": generation_seconds,
                "processor_decode_handoff_seconds": max(0.0, call_wall - generation_seconds),
                "physical_gpu_peak_allocated_bytes": peak_bytes,
                "prompt_tokens": sum(member.prompt_tokens for member in members),
                "output_tokens": sum(member.output_tokens for member in members),
            }
        )
        for index, member in zip(indices, members, strict=True):
            case = fixture.cases[index]
            parse_started = time.perf_counter()
            parse_error: dict[str, str] | None = None
            candidate: CommonQwenProjection | None = None
            try:
                candidate = project_qwen_compact_output(
                    case=case,
                    checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                    output_text=member.output_text,
                )
            except CommonProjectionError as error:
                parse_error = {"type": type(error).__name__, "message": str(error)}
            parse_seconds = time.perf_counter() - parse_started
            parse_sum += parse_seconds
            parsed[index] = candidate
            case_reports.append(
                {
                    "case_ordinal": case.ordinal,
                    "context_manifest_semantic_sha256": (
                        case.context.context_manifest_semantic_sha256
                    ),
                    "qwen_prompt_exact_sha256": exact_bytes_sha256(
                        build_qwen_common_prompt(case).encode("utf-8")
                    ),
                    "source_mage_prompt_exact_sha256": exact_bytes_sha256(
                        case.binding.prompt.encode("utf-8")
                    ),
                    "selected_frame_sha256_values": [
                        frame.sha256 for frame in case.selected_frames
                    ],
                    "runtime_member": _member_projection(member),
                    "raw_output_diagnostic": _raw_output_diagnostic(member.output_text),
                    "parse_seconds": parse_seconds,
                    "parse_error": parse_error,
                    "projection": _observation_projection(candidate)
                    if candidate is not None
                    else None,
                }
            )

    inference_parse_wall = time.perf_counter() - recurring_started
    parsed_tuple = tuple(item for item in parsed if item is not None)
    downstream_wall = 0.0
    downstream = None
    if len(parsed_tuple) == len(fixture.cases):
        downstream_started = time.perf_counter()
        downstream_result = run_common_downstream(
            cases=fixture.cases,
            observations=tuple(item.observation for item in parsed_tuple),
            observation_elapsed_seconds=observation_elapsed,
        )
        downstream_wall = time.perf_counter() - downstream_started
        downstream = downstream_projection(downstream_result)
    recurring_wall = time.perf_counter() - recurring_started
    report = {
        "mode": mode,
        "batch_policy": (
            "SERIAL_ONE_REQUEST_PER_CALL_V1"
            if mode == "serial"
            else "CONTIGUOUS_FOUR_THEN_NATIVE_BATCH_TAIL_V1"
        ),
        "batch_time_attribution": "EQUAL_SHARE_FOR_PIPELINE_STAGE_SUM_ONLY_V1",
        "physical_calls": physical_reports,
        "cases": sorted(case_reports, key=lambda item: int(item["case_ordinal"])),
        "physical_call_count": len(physical_reports),
        "parse_success_count": len(parsed_tuple),
        "parse_failure_count": len(fixture.cases) - len(parsed_tuple),
        "quality_gate": {
            "strict_compact_observation_expansion_required": True,
            "passed": len(parsed_tuple) == len(fixture.cases),
            "downstream_recomputed": downstream is not None,
            "capacity_decision_eligible": downstream is not None,
            "failure_disposition": (
                None if downstream is not None else "REJECT_CANDIDATE_OUTPUT_BEFORE_DOWNSTREAM_V1"
            ),
        },
        "inference_and_parse_wall_seconds": inference_parse_wall,
        "physical_generation_seconds": generation_sum,
        "physical_call_wall_seconds": physical_wall_sum,
        "processor_decode_handoff_seconds": max(0.0, physical_wall_sum - generation_sum),
        "strict_parse_seconds": parse_sum,
        "downstream_projection_wall_seconds": downstream_wall,
        "recurring_wall_seconds": recurring_wall,
        "orchestration_residual_seconds": max(
            0.0, recurring_wall - physical_wall_sum - parse_sum - downstream_wall
        ),
        "downstream": downstream,
    }
    return report, parsed_tuple if len(parsed_tuple) == len(fixture.cases) else None


def run(arguments: argparse.Namespace) -> tuple[int, dict[str, object]]:
    started_at = _utc_now()
    started = time.perf_counter()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    gpu_path = output_dir / "gpu-telemetry.json"
    runtime: LocalHuggingFaceVisionRuntime | None = None
    sampler: NvidiaSmiGpuSampler | None = None
    gpu_report: dict[str, Any] | None = None
    report: dict[str, object]
    exit_code = EXIT_FAILURE
    try:
        fixture_started = time.perf_counter()
        corpus = load_qwen_request_corpus(
            arguments.corpus_db,
            expected=QWEN_R12_20260806_EXPECTED,
        )
        fixture = load_common_projection_fixture(
            corpus=corpus,
            mage_stream_artifact_root=arguments.mage_stream_artifact_root,
        )
        frame_payloads = _load_frame_payloads(fixture)
        fixture_wall = time.perf_counter() - fixture_started
        budgets = {_case_max_new_tokens(case) for case in fixture.cases}
        if len(budgets) != 1:
            raise CommonProjectionBenchmarkError("frozen Mage prompts have unequal output budgets")

        mage_downstream_started = time.perf_counter()
        mage_downstream_result = run_common_downstream(
            cases=fixture.cases,
            observations=tuple(case.mage_observation for case in fixture.cases),
            observation_elapsed_seconds=tuple(
                case.binding.endpoint_response.generation_seconds for case in fixture.cases
            ),
        )
        mage_downstream_wall = time.perf_counter() - mage_downstream_started
        mage_downstream = downstream_projection(mage_downstream_result)
        mage_generation_seconds = sum(
            case.binding.endpoint_response.generation_seconds for case in fixture.cases
        )
        common: dict[str, object] = {
            "report_version": REPORT_VERSION,
            "authority": "LOCAL_NONPRODUCTION_ONLY",
            "production_eligible": False,
            "started_at": started_at,
            "configuration": {
                "mode": arguments.mode,
                "verify_only": arguments.verify_only,
                "model_directory": (
                    str(arguments.model_dir.expanduser().resolve())
                    if arguments.model_dir is not None
                    else None
                ),
                "max_image_side": arguments.max_image_side,
                "gpu_weight_memory_gib": arguments.gpu_weight_memory_gib,
                "cpu_weight_memory_gib": arguments.cpu_weight_memory_gib,
                "checkpoint_manifest_sha256": (arguments.expected_checkpoint_manifest_sha256),
            },
            "input_equivalence": {
                "camera_id": CameraId.CAM_01.value,
                "aligned_interval_count": len(fixture.cases),
                "source_duration_seconds": fixture.duration_seconds,
                "frames_per_qwen_segment": COMMON_FRAMES_PER_SEGMENT,
                "mage_modality": "FROZEN_NATIVE_VIDEO_DERIVATIVE",
                "qwen_modality": "SIX_EXACT_R12_PNG_DERIVATIVES_PER_INTERVAL",
                "same_camera": True,
                "same_aligned_intervals": True,
                "byte_identical_inputs": False,
                "agreement_authority": "UNLABELED_MODEL_AGREEMENT_ONLY",
                "ground_truth_accuracy": False,
            },
            "fixture": {
                "semantic_sha256": fixture.semantic_sha256,
                "corpus_database_path": str(corpus.database_path),
                "corpus_database_exact_sha256": corpus.database_sha256,
                "corpus_semantic_sha256": corpus.semantic_sha256,
                "fixture_load_and_frame_verify_seconds": fixture_wall,
                "cases": [case.projection() for case in fixture.cases],
            },
            "mage": {
                "model_family": fixture.cases[0].mage_observation.model_family,
                "frozen_generation_seconds": mage_generation_seconds,
                "frozen_generation_only_capacity": _capacity_projection(
                    duration_seconds=fixture.duration_seconds,
                    recurring_wall_seconds=mage_generation_seconds,
                ),
                "observations": [_mage_observation_projection(case) for case in fixture.cases],
                "downstream_recompute_wall_seconds": mage_downstream_wall,
                "downstream": mage_downstream,
                "frozen_stream_receipt": _load_frozen_mage_receipt(
                    arguments.mage_stream_artifact_root
                ),
            },
        }
        if arguments.verify_only:
            report = {
                **common,
                "status": "VERIFIED",
                "finished_at": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "qwen": None,
                "agreement": None,
                "error": None,
            }
            exit_code = 0
        else:
            if arguments.model_dir is None:
                raise CommonProjectionBenchmarkError("--model-dir is required for a real run")
            sampler = NvidiaSmiGpuSampler(interval_seconds=arguments.gpu_sample_interval_seconds)
            sampler.start()
            runtime = LocalHuggingFaceVisionRuntime(
                model_directory=arguments.model_dir,
                offload_directory=output_dir / "offload",
                max_image_side=arguments.max_image_side,
                gpu_weight_memory_gib=arguments.gpu_weight_memory_gib,
                cpu_weight_memory_gib=arguments.cpu_weight_memory_gib,
            )
            load = runtime.load()
            qwen, projections = _run_candidate(
                runtime=runtime,
                fixture=fixture,
                frame_payloads=frame_payloads,
                mode=arguments.mode,
                checkpoint_manifest_sha256=arguments.expected_checkpoint_manifest_sha256,
            )
            agreement = None
            if projections is not None:
                observation_agreement = compare_observations(
                    mage=tuple(case.mage_observation for case in fixture.cases),
                    qwen=tuple(item.observation for item in projections),
                )
                qwen_downstream = qwen["downstream"]
                if not isinstance(qwen_downstream, dict):
                    raise CommonProjectionBenchmarkError("Qwen downstream projection is missing")
                agreement = {
                    "authority": "UNLABELED_MODEL_AGREEMENT_ONLY",
                    "is_ground_truth_accuracy": False,
                    "observation": observation_agreement,
                    "downstream": _downstream_agreement(mage_downstream, qwen_downstream),
                }
            recurring = float(qwen["recurring_wall_seconds"])
            capacity = _capacity_projection(
                duration_seconds=fixture.duration_seconds,
                recurring_wall_seconds=recurring,
            )
            capacity["quality_qualified"] = projections is not None
            capacity["decision_eligible"] = projections is not None
            qwen["capacity"] = capacity
            qwen["model_family"] = COMMON_QWEN_MODEL_FAMILY
            qwen["load"] = {
                "load_seconds": load.load_seconds,
                "gpu_name": load.gpu_name,
                "gpu_total_bytes": load.gpu_total_bytes,
                "gpu_free_before_bytes": load.gpu_free_before_bytes,
                "gpu_allocated_after_load_bytes": load.gpu_allocated_after_load_bytes,
            }
            report = {
                **common,
                "status": "SUCCEEDED" if projections is not None else "FAILED_QUALITY_GATE",
                "finished_at": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "qwen": qwen,
                "agreement": agreement,
                "gpu_telemetry": None,
                "error": None,
            }
            exit_code = 0
    except Exception as error:
        is_oom = type(error).__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or (
            "out of memory" in str(error).casefold()
        )
        report = {
            "report_version": REPORT_VERSION,
            "authority": "LOCAL_NONPRODUCTION_ONLY",
            "production_eligible": False,
            "status": "FAILED_OOM" if is_oom else "FAILED",
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "configuration": {
                "mode": arguments.mode,
                "verify_only": arguments.verify_only,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        exit_code = EXIT_FAILURE
    finally:
        if runtime is not None:
            with suppress(Exception):
                runtime.close()
        if sampler is not None:
            with suppress(Exception):
                gpu_report = sampler.stop().to_payload()
        if gpu_report is not None:
            gpu_sha = _write_json(gpu_path, gpu_report)
            report["gpu_telemetry"] = {
                "path": str(gpu_path),
                "exact_sha256": gpu_sha,
                "summary": gpu_report.get("summary"),
                "measurement_status": gpu_report.get("measurement_status"),
                "sample_count": gpu_report.get("sample_count"),
            }
        report["finished_at"] = _utc_now()
        report["wall_seconds"] = time.perf_counter() - started
        _write_json(report_path, report)
    return exit_code, report


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    exit_code, report = run(arguments)
    report_path = arguments.output_dir.expanduser().resolve() / "report.json"
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "report_exact_sha256": exact_bytes_sha256(report_path.read_bytes()),
                "wall_seconds": report["wall_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
