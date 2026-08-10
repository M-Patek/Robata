"""Run a direct, bounded-input Qwen serial/native-batch viability benchmark.

The workload is reconstructed from the immutable r12 inference-evidence database,
not from MCAP export.  This avoids repeating source materialization while preserving
exact request, prompt, image and baseline-output identity.  Use the separate
``run_bounded_qwen_batch_benchmark.py`` parent for every real model invocation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
import traceback
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robata.application.canonical.local_real_model import (  # noqa: E402
    LOCAL_QWEN_MODEL_NAME,
    LOCAL_QWEN_MODEL_VERSION,
    LOCAL_QWEN_PROVIDER,
    build_local_qwen_capabilities,
)
from robata.benchmark.gpu_telemetry import NvidiaSmiGpuSampler  # noqa: E402
from robata.benchmark.qwen_r12_request_corpus import (  # noqa: E402
    QWEN_R12_20260806_EXPECTED,
    QwenRequestCase,
    QwenRequestCorpus,
    load_qwen_request_corpus,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256  # noqa: E402
from robata.contracts.schema_registry import SchemaRegistry  # noqa: E402
from robata.inference.local_hf_adapter import (  # noqa: E402
    LocalHfLoopbackAdapterConfig,
    LocalHfLoopbackVisionAdapter,
)
from robata.inference.local_hf_runtime import (  # noqa: E402
    LocalHfBatchGenerationRequest,
    LocalHfBatchMemberObservation,
    LocalHuggingFaceVisionRuntime,
)
from robata.inference.offline_fixture import (  # noqa: E402
    InMemoryRawProviderBytesStore,
    StrictProviderClaimParser,
)

REPORT_VERSION = "qwen-r12-native-batch-viability-v1"
DEFAULT_CHECKPOINT_MANIFEST_SHA256 = (
    "1f7293b2629473f0240c8675025e1402da4306f05cc9026adf4c801f20f99f10"
)
EXIT_FAILURE = 2


class QwenBatchHedgeBenchmarkError(RuntimeError):
    """The frozen Qwen request corpus could not be benchmarked honestly."""


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


def _sha256(value: bytes) -> str:
    return exact_bytes_sha256(value)


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
    return _sha256(body)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=_positive_int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--max-cases", type=_positive_int, default=None)
    parser.add_argument(
        "--task",
        choices=("all", "QA_COARSE", "QA_DENSE"),
        default="all",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--batch-packing-policy",
        choices=("contiguous-task-v1", "task-claim-group-v1"),
        default="task-claim-group-v1",
    )
    parser.add_argument(
        "--multi-claim-policy",
        choices=("batch-v1", "serial-v1"),
        default="batch-v1",
    )
    parser.add_argument("--max-image-side", type=_positive_int, default=448)
    parser.add_argument("--gpu-weight-memory-gib", type=_positive_int, default=7)
    parser.add_argument("--cpu-weight-memory-gib", type=_positive_int, default=1)
    parser.add_argument("--gpu-sample-interval-seconds", type=_positive_float, default=0.25)
    parser.add_argument(
        "--expected-checkpoint-manifest-sha256",
        default=DEFAULT_CHECKPOINT_MANIFEST_SHA256,
    )
    return parser


def _build_adapter(*, checkpoint_manifest_sha256: str) -> LocalHfLoopbackVisionAdapter:
    capabilities = build_local_qwen_capabilities(
        checkpoint_manifest_sha256=checkpoint_manifest_sha256
    )
    return LocalHfLoopbackVisionAdapter(
        capabilities=capabilities,
        parser=StrictProviderClaimParser(
            SchemaRegistry(),
            parser_version="qwen-r12-native-batch-benchmark-parser-v1",
        ),
        evidence_ledger=InMemoryRawProviderBytesStore(),
        config=LocalHfLoopbackAdapterConfig(
            provider=LOCAL_QWEN_PROVIDER,
            default_max_new_tokens=128,
            request_timeout_cap_ms=300_000,
        ),
    )


def _selected_cases(
    corpus: QwenRequestCorpus,
    *,
    task: str,
    max_cases: int | None,
) -> tuple[QwenRequestCase, ...]:
    selected = tuple(
        case for case in corpus.cases if task == "all" or case.request.task.value == task
    )
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise QwenBatchHedgeBenchmarkError("selected corpus contains no requests")
    return selected


def _load_image_bytes(
    cases: tuple[QwenRequestCase, ...],
) -> tuple[dict[str, bytes], dict[str, int]]:
    cache: dict[str, bytes] = {}
    reference_count = 0
    total_unique_bytes = 0
    for case in cases:
        for image in case.selected_images:
            reference_count += 1
            payload = cache.get(image.sha256)
            if payload is None:
                payload = image.path.read_bytes()
                if len(payload) != image.byte_count or exact_bytes_sha256(payload) != image.sha256:
                    raise QwenBatchHedgeBenchmarkError(
                        f"selected image changed after corpus verification: {image.path}"
                    )
                cache[image.sha256] = payload
                total_unique_bytes += len(payload)
    return cache, {
        "references": reference_count,
        "unique_images": len(cache),
        "cache_hits": reference_count - len(cache),
        "unique_bytes": total_unique_bytes,
    }


def _prepare_cases(
    *,
    adapter: LocalHfLoopbackVisionAdapter,
    cases: tuple[QwenRequestCase, ...],
    image_bytes: dict[str, bytes],
) -> tuple[dict[str, Any], ...]:
    prepared: list[dict[str, Any]] = []
    for case in cases:
        selected_items = adapter._validate_request(case.request)
        selected_digests = tuple(item.artifact.sha256 for item in selected_items)
        corpus_digests = tuple(item.sha256 for item in case.selected_images)
        if selected_digests != corpus_digests:
            raise QwenBatchHedgeBenchmarkError(
                f"adapter/corpus selected-image drift for request {case.request.request_id}"
            )
        prompt = adapter._canonical_claim_prompt(case.request, selected_items)
        try:
            prompt_document = json.loads(prompt)
            claim_group_count = int(prompt_document["compact_output_contract"]["claim_group_count"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise QwenBatchHedgeBenchmarkError(
                f"prompt claim-group projection is invalid for {case.request.request_id}"
            ) from error
        if claim_group_count < 0:
            raise QwenBatchHedgeBenchmarkError("prompt claim-group count must be nonnegative")
        max_new_tokens = adapter._max_new_tokens(case.request)
        prepared.append(
            {
                "case": case,
                "selected_items": selected_items,
                "image_payloads": tuple(image_bytes[digest] for digest in selected_digests),
                "prompt": prompt,
                "prompt_exact_sha256": exact_bytes_sha256(prompt.encode("utf-8")),
                "claim_group_count": claim_group_count,
                "max_new_tokens": max_new_tokens,
            }
        )
    return tuple(prepared)


def _load_baseline_evidence(
    database_path: Path,
    cases: tuple[QwenRequestCase, ...],
) -> dict[str, dict[str, Any]]:
    state_directory = database_path.parent
    requested_ids = {case.intent.inference_id for case in cases}
    uri = f"{database_path.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    try:
        terminals = {
            row[0]: (bytes(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT inference_id, payload_json, payload_sha256 FROM model_inference_terminals"
            )
            if row[0] in requested_ids
        }
        responses = {
            row[0]: (str(row[1]), int(row[2]))
            for row in connection.execute(
                "SELECT inference_id, exact_bytes_sha256, byte_count FROM raw_provider_responses"
            )
            if row[0] in requested_ids
        }
    finally:
        connection.close()
    if set(terminals) != requested_ids or set(responses) != requested_ids:
        raise QwenBatchHedgeBenchmarkError("baseline terminal/raw evidence is incomplete")
    evidence: dict[str, dict[str, Any]] = {}
    for case in cases:
        inference_id = case.intent.inference_id
        terminal_bytes, terminal_sha = terminals[inference_id]
        if exact_bytes_sha256(terminal_bytes) != terminal_sha:
            raise QwenBatchHedgeBenchmarkError("baseline terminal payload digest mismatch")
        terminal = json.loads(terminal_bytes)
        if canonical_json_bytes(terminal) != terminal_bytes:
            raise QwenBatchHedgeBenchmarkError("baseline terminal payload is not canonical JSON")
        raw_sha, raw_byte_count = responses[inference_id]
        raw_path = state_directory / "raw-provider-cas" / raw_sha[:2] / raw_sha
        raw_bytes = raw_path.read_bytes()
        if len(raw_bytes) != raw_byte_count or exact_bytes_sha256(raw_bytes) != raw_sha:
            raise QwenBatchHedgeBenchmarkError("baseline raw provider CAS mismatch")
        evidence[inference_id] = {
            "terminal_payload_sha256": terminal_sha,
            "normalized_output": terminal["normalized_output"],
            "latency_ms": terminal["latency_ms"],
            "usage": terminal["usage"],
            "raw_output": raw_bytes.decode("utf-8"),
            "raw_output_exact_sha256": raw_sha,
        }
    return evidence


def _chunks_without_crossing_task(
    prepared: tuple[dict[str, Any], ...],
    *,
    batch_size: int,
    packing_policy: str,
    multi_claim_policy: str,
) -> tuple[tuple[dict[str, Any], ...], ...]:
    if packing_policy == "contiguous-task-v1":
        chunks: list[tuple[dict[str, Any], ...]] = []
        start = 0
        while start < len(prepared):
            task = prepared[start]["case"].request.task
            end = start
            while end < len(prepared) and prepared[end]["case"].request.task is task:
                end += 1
            for offset in range(start, end, batch_size):
                chunks.append(prepared[offset : min(offset + batch_size, end)])
            start = end
        return tuple(chunks)
    if packing_policy != "task-claim-group-v1":
        raise QwenBatchHedgeBenchmarkError(f"unsupported batch packing policy: {packing_policy}")
    ordered_groups: dict[tuple[object, int], list[dict[str, Any]]] = {}
    for item in prepared:
        key = (item["case"].request.task, int(item["claim_group_count"]))
        ordered_groups.setdefault(key, []).append(item)
    chunks: list[tuple[dict[str, Any], ...]] = []
    for (_task, claim_group_count), group in ordered_groups.items():
        effective_batch_size = (
            1 if multi_claim_policy == "serial-v1" and claim_group_count > 1 else batch_size
        )
        if multi_claim_policy not in {"batch-v1", "serial-v1"}:
            raise QwenBatchHedgeBenchmarkError(
                f"unsupported multi-claim policy: {multi_claim_policy}"
            )
        chunks.extend(
            tuple(group[offset : offset + effective_batch_size])
            for offset in range(0, len(group), effective_batch_size)
        )
    return tuple(chunks)


def _parse_candidate(
    *,
    adapter: LocalHfLoopbackVisionAdapter,
    prepared: dict[str, Any],
    output_text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = adapter._decode_compact_payload(
            data=output_text.encode("utf-8"),
            request=prepared["case"].request,
            items=prepared["selected_items"],
        )
        return payload.model_dump(mode="json"), None
    except (TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"


def _member_projection(member: LocalHfBatchMemberObservation) -> dict[str, Any]:
    return {
        "rendered_image_sizes": [list(size) for size in member.rendered_image_sizes],
        "prompt_tokens": member.prompt_tokens,
        "output_tokens": member.output_tokens,
        "output_text": member.output_text,
    }


def _source_duration_seconds(cases: tuple[QwenRequestCase, ...]) -> float:
    start_ns = min(case.intent.start_ns for case in cases)
    end_ns = max(case.intent.end_ns for case in cases)
    return (end_ns - start_ns) / 1_000_000_000


def _run_model(
    *,
    runtime: LocalHuggingFaceVisionRuntime,
    adapter: LocalHfLoopbackVisionAdapter,
    prepared: tuple[dict[str, Any], ...],
    baseline: dict[str, dict[str, Any]],
    batch_size: int,
    packing_policy: str,
    multi_claim_policy: str,
) -> dict[str, Any]:
    batches = _chunks_without_crossing_task(
        prepared,
        batch_size=batch_size,
        packing_policy=packing_policy,
        multi_claim_policy=multi_claim_policy,
    )
    batch_reports: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    physical_generation_sum = 0.0
    physical_call_wall_sum = 0.0
    started = time.perf_counter()
    for batch_ordinal, batch in enumerate(batches):
        call_started = time.perf_counter()
        force_serial = batch_size == 1 or (
            multi_claim_policy == "serial-v1"
            and len(batch) == 1
            and int(batch[0]["claim_group_count"]) > 1
        )
        if force_serial:
            only = batch[0]
            serial = runtime.generate(
                image_payloads=only["image_payloads"],
                prompt=only["prompt"],
                max_new_tokens=only["max_new_tokens"],
            )
            members = (
                LocalHfBatchMemberObservation(
                    rendered_image_sizes=serial.rendered_image_sizes,
                    prompt_tokens=serial.prompt_tokens,
                    output_tokens=serial.output_tokens,
                    output_text=serial.output_text,
                ),
            )
            generation_seconds = serial.generation_seconds
            peak_bytes = serial.gpu_peak_allocated_bytes
            physical_mode = "SERIAL_GENERATE_V1"
        else:
            batch_observation = runtime.generate_batch(
                requests=tuple(
                    LocalHfBatchGenerationRequest(
                        image_payloads=item["image_payloads"],
                        prompt=item["prompt"],
                        max_new_tokens=item["max_new_tokens"],
                    )
                    for item in batch
                )
            )
            members = batch_observation.members
            generation_seconds = batch_observation.physical_generation_seconds
            peak_bytes = batch_observation.physical_gpu_peak_allocated_bytes
            physical_mode = "NATIVE_BATCH_GENERATE_V1"
        call_wall = time.perf_counter() - call_started
        if len(members) != len(batch):
            raise QwenBatchHedgeBenchmarkError("runtime member count differs from physical batch")
        physical_generation_sum += generation_seconds
        physical_call_wall_sum += call_wall
        batch_reports.append(
            {
                "batch_ordinal": batch_ordinal,
                "task": batch[0]["case"].request.task.value,
                "member_count": len(batch),
                "claim_group_count": int(batch[0]["claim_group_count"]),
                "physical_mode": physical_mode,
                "physical_call_wall_seconds": call_wall,
                "physical_generation_seconds": generation_seconds,
                "processor_handoff_seconds": max(0.0, call_wall - generation_seconds),
                "physical_gpu_peak_allocated_bytes": peak_bytes,
                "prompt_tokens": sum(member.prompt_tokens for member in members),
                "output_tokens": sum(member.output_tokens for member in members),
            }
        )
        for prepared_item, member in zip(batch, members, strict=True):
            case = prepared_item["case"]
            historical = baseline[case.intent.inference_id]
            normalized, parse_error = _parse_candidate(
                adapter=adapter,
                prepared=prepared_item,
                output_text=member.output_text,
            )
            case_reports.append(
                {
                    "case_ordinal": case.ordinal,
                    "inference_id": case.intent.inference_id,
                    "request_id": case.request.request_id,
                    "task": case.request.task.value,
                    "prompt_exact_sha256": prepared_item["prompt_exact_sha256"],
                    "candidate": _member_projection(member),
                    "candidate_output_exact_sha256": exact_bytes_sha256(
                        member.output_text.encode("utf-8")
                    ),
                    "parse_error": parse_error,
                    "normalized_output": normalized,
                    "baseline": historical,
                    "raw_exact_match": member.output_text == historical["raw_output"],
                    "normalized_exact_match": normalized == historical["normalized_output"],
                    "output_exhausted": member.output_tokens >= prepared_item["max_new_tokens"],
                }
            )
    execution_wall = time.perf_counter() - started
    return {
        "wall_seconds": execution_wall,
        "physical_call_wall_seconds_sum": physical_call_wall_sum,
        "physical_generation_seconds_sum": physical_generation_sum,
        "processor_handoff_seconds_sum": max(0.0, physical_call_wall_sum - physical_generation_sum),
        "batch_count": len(batch_reports),
        "case_count": len(case_reports),
        "batches": batch_reports,
        "cases": case_reports,
    }


def _quality_projection(execution: dict[str, Any]) -> dict[str, Any]:
    cases = execution["cases"]
    return {
        "case_count": len(cases),
        "parse_valid_count": sum(case["parse_error"] is None for case in cases),
        "raw_exact_match_count": sum(case["raw_exact_match"] for case in cases),
        "normalized_exact_match_count": sum(case["normalized_exact_match"] for case in cases),
        "output_exhaustion_count": sum(case["output_exhausted"] for case in cases),
        "quality_gate_pass": all(
            case["parse_error"] is None
            and case["normalized_exact_match"]
            and not case["output_exhausted"]
            for case in cases
        ),
        "comparison_scope": "EXACT_R12_NORMALIZED_QA_OUTPUT_NOT_LABELED_GROUND_TRUTH",
    }


def _capacity_projection(
    *,
    cases: tuple[QwenRequestCase, ...],
    execution_wall_seconds: float,
    complete_corpus: bool,
) -> dict[str, Any]:
    camera_count = len({image.camera_id for case in cases for image in case.selected_images})
    if not complete_corpus:
        return {
            "scope": "PARTIAL_CORPUS_SMOKE_NOT_CAPACITY_EVIDENCE",
            "source_recording_seconds": None,
            "distinct_camera_count": camera_count,
            "source_camera_seconds": None,
            "recording_real_time_multiple": None,
            "camera_real_time_multiple": None,
            "local_equivalent_lanes_for_25x_camera_hours": None,
            "production_eligible": False,
        }
    source_seconds = _source_duration_seconds(cases)
    recording_rtf = source_seconds / execution_wall_seconds
    camera_rtf = source_seconds * camera_count / execution_wall_seconds
    return {
        "scope": "QWEN_R12_QA_ONLY_NOT_FULL_PIPELINE",
        "source_recording_seconds": source_seconds,
        "distinct_camera_count": camera_count,
        "source_camera_seconds": source_seconds * camera_count,
        "recording_real_time_multiple": recording_rtf,
        "camera_real_time_multiple": camera_rtf,
        "local_equivalent_lanes_for_25x_camera_hours": math.ceil(25.0 / camera_rtf),
        "production_eligible": False,
    }


def run(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started_at = _utc_now()
    started = time.perf_counter()
    output_dir = arguments.output_dir.expanduser().resolve()
    report_path = output_dir / "report.json"
    gpu_path = output_dir / "gpu-telemetry.json"
    runtime: LocalHuggingFaceVisionRuntime | None = None
    sampler: NvidiaSmiGpuSampler | None = None
    gpu_report: dict[str, Any] | None = None
    report: dict[str, Any]
    exit_code = EXIT_FAILURE
    try:
        corpus = load_qwen_request_corpus(
            arguments.corpus_db,
            expected=QWEN_R12_20260806_EXPECTED,
        )
        cases = _selected_cases(corpus, task=arguments.task, max_cases=arguments.max_cases)
        adapter = _build_adapter(
            checkpoint_manifest_sha256=arguments.expected_checkpoint_manifest_sha256
        )
        image_cache_started = time.perf_counter()
        image_bytes, cache_metrics = _load_image_bytes(cases)
        image_cache_seconds = time.perf_counter() - image_cache_started
        prepare_started = time.perf_counter()
        prepared = _prepare_cases(adapter=adapter, cases=cases, image_bytes=image_bytes)
        baseline = _load_baseline_evidence(corpus.database_path, cases)
        prepare_seconds = time.perf_counter() - prepare_started
        common: dict[str, Any] = {
            "report_version": REPORT_VERSION,
            "authority": "LOCAL_NONPRODUCTION_ONLY",
            "production_eligible": False,
            "started_at": started_at,
            "configuration": {
                "model_directory": str(arguments.model_dir.expanduser().resolve()),
                "model_identifier": LOCAL_QWEN_MODEL_NAME,
                "model_version": LOCAL_QWEN_MODEL_VERSION,
                "checkpoint_manifest_sha256": arguments.expected_checkpoint_manifest_sha256,
                "batch_size": arguments.batch_size,
                "batch_packing_policy": arguments.batch_packing_policy,
                "multi_claim_policy": arguments.multi_claim_policy,
                "execution_mode": (
                    "SERIAL_CONTROL_V1" if arguments.batch_size == 1 else "NATIVE_BATCH_V1"
                ),
                "task": arguments.task,
                "max_cases": arguments.max_cases,
                "max_image_side": arguments.max_image_side,
                "gpu_weight_memory_gib": arguments.gpu_weight_memory_gib,
                "cpu_weight_memory_gib": arguments.cpu_weight_memory_gib,
            },
            "corpus": {
                "manifest": corpus.manifest_projection(),
                "semantic_sha256": corpus.semantic_sha256,
                "selected_case_ordinals": [case.ordinal for case in cases],
                "selected_case_count": len(cases),
            },
            "preparation": {
                "verified_image_cache_seconds": image_cache_seconds,
                "request_prompt_preparation_seconds": prepare_seconds,
                "image_cache": cache_metrics,
            },
        }
        if arguments.verify_only:
            report = {
                **common,
                "status": "CORPUS_VERIFIED",
                "finished_at": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "load": None,
                "execution": None,
                "quality": None,
                "capacity": None,
                "gpu_telemetry": None,
                "error": None,
            }
            exit_code = 0
        else:
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
            execution = _run_model(
                runtime=runtime,
                adapter=adapter,
                prepared=prepared,
                baseline=baseline,
                batch_size=arguments.batch_size,
                packing_policy=arguments.batch_packing_policy,
                multi_claim_policy=arguments.multi_claim_policy,
            )
            quality = _quality_projection(execution)
            capacity = _capacity_projection(
                cases=cases,
                execution_wall_seconds=execution["wall_seconds"],
                complete_corpus=(
                    arguments.task == "all"
                    and arguments.max_cases is None
                    and len(cases) == len(corpus.cases)
                ),
            )
            report = {
                **common,
                "status": "SUCCEEDED" if quality["quality_gate_pass"] else "FAILED_QUALITY_GATE",
                "finished_at": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "load": {
                    "load_seconds": load.load_seconds,
                    "gpu_name": load.gpu_name,
                    "gpu_total_bytes": load.gpu_total_bytes,
                    "gpu_free_before_bytes": load.gpu_free_before_bytes,
                    "gpu_allocated_after_load_bytes": load.gpu_allocated_after_load_bytes,
                },
                "execution": execution,
                "quality": quality,
                "capacity": capacity,
                "gpu_telemetry": None,
                "error": None,
            }
            # A valid but changed output is a qualification result, not a process failure.
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
                "model_directory": str(arguments.model_dir.expanduser().resolve()),
                "batch_size": arguments.batch_size,
                "batch_packing_policy": arguments.batch_packing_policy,
                "multi_claim_policy": arguments.multi_claim_policy,
                "task": arguments.task,
                "max_cases": arguments.max_cases,
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
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(arguments.output_dir.expanduser().resolve() / "report.json"),
                "wall_seconds": report["wall_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
