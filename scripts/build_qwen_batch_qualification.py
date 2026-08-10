"""Build the frozen local Qwen native-batch qualification report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256  # noqa: E402

REPORT_VERSION = "qwen-native-batch-qualification-v1"
CORPUS_SEMANTIC_SHA256 = "d4bd44f5e573b2abc13000cf9421134ac0e8d00fe92890fc6a7fa265c84425ed"


class QwenBatchQualificationError(ValueError):
    """The local Qwen batch evidence is incomplete or internally inconsistent."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-report", type=Path, required=True)
    parser.add_argument("--serial-lifecycle", type=Path, required=True)
    parser.add_argument("--batch2-report", type=Path, required=True)
    parser.add_argument("--batch2-lifecycle", type=Path, required=True)
    parser.add_argument("--batch4-report", type=Path, required=True)
    parser.add_argument("--batch4-lifecycle", type=Path, required=True)
    parser.add_argument("--hybrid-report", type=Path, action="append", required=True)
    parser.add_argument("--hybrid-lifecycle", type=Path, action="append", required=True)
    parser.add_argument("--batch8-report", type=Path, required=True)
    parser.add_argument("--batch8-lifecycle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> tuple[bytes, dict[str, Any]]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise QwenBatchQualificationError(f"invalid JSON: {resolved}") from error
    if not isinstance(value, dict):
        raise QwenBatchQualificationError(f"JSON root must be an object: {resolved}")
    return raw, value


def _reference(path: Path, raw: bytes) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "byte_count": len(raw),
        "exact_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_lifecycle(payload: dict[str, Any]) -> None:
    if payload.get("report_version") != "bounded-qwen-batch-benchmark-lifecycle-v1":
        raise QwenBatchQualificationError("lifecycle version differs")
    if payload.get("outcome") != "SUCCEEDED" or payload.get("exit_code") != 0:
        raise QwenBatchQualificationError("benchmark lifecycle did not succeed")
    ownership = payload["process_ownership"]
    if (
        ownership.get("shell_used") is not False
        or ownership.get("direct_popen_ownership") is not True
    ):
        raise QwenBatchQualificationError("benchmark lifecycle did not own a direct child")
    if payload["benchmark"]["cleanup"].get("reaped") is not True:
        raise QwenBatchQualificationError("benchmark child was not reaped")


def _validate_run(
    payload: dict[str, Any],
    *,
    batch_size: int,
    multi_claim_policy: str,
    expected_quality_pass: bool,
) -> None:
    if payload.get("report_version") != "qwen-r12-native-batch-viability-v1":
        raise QwenBatchQualificationError("benchmark report version differs")
    if payload.get("production_eligible") is not False:
        raise QwenBatchQualificationError("local Qwen report must remain non-production")
    config = payload["configuration"]
    if config["batch_size"] != batch_size:
        raise QwenBatchQualificationError("batch size differs")
    packing_policy = config.get("batch_packing_policy")
    observed_multi_claim_policy = config.get("multi_claim_policy")
    if batch_size == 1:
        if packing_policy not in {None, "task-claim-group-v1"}:
            raise QwenBatchQualificationError("serial packing policy differs")
        if observed_multi_claim_policy not in {None, multi_claim_policy}:
            raise QwenBatchQualificationError("serial multi-claim policy differs")
    else:
        if packing_policy != "task-claim-group-v1":
            raise QwenBatchQualificationError("packing policy differs")
        if (observed_multi_claim_policy or "batch-v1") != multi_claim_policy:
            raise QwenBatchQualificationError("multi-claim policy differs")
    corpus = payload["corpus"]
    if corpus["semantic_sha256"] != CORPUS_SEMANTIC_SHA256:
        raise QwenBatchQualificationError("corpus identity differs")
    if corpus["selected_case_count"] != 51:
        raise QwenBatchQualificationError("full 51-request corpus is required")
    execution = payload["execution"]
    quality = payload["quality"]
    if execution["case_count"] != 51 or quality["case_count"] != 51:
        raise QwenBatchQualificationError("execution did not cover 51 requests")
    if quality["parse_valid_count"] != 51 or quality["output_exhaustion_count"] != 0:
        raise QwenBatchQualificationError("candidate output validity differs")
    if quality["quality_gate_pass"] is not expected_quality_pass:
        raise QwenBatchQualificationError("quality-gate state differs")
    if expected_quality_pass:
        if quality["normalized_exact_match_count"] != 51 or quality["raw_exact_match_count"] != 51:
            raise QwenBatchQualificationError("accepted candidate is not exact to serial control")
        if payload["status"] != "SUCCEEDED":
            raise QwenBatchQualificationError("accepted run status differs")
    elif payload["status"] != "FAILED_QUALITY_GATE":
        raise QwenBatchQualificationError("non-parity candidate must remain quality-gated")
    capacity = payload["capacity"]
    if capacity["scope"] != "QWEN_R12_QA_ONLY_NOT_FULL_PIPELINE":
        raise QwenBatchQualificationError("capacity scope differs")
    if capacity["production_eligible"] is not False:
        raise QwenBatchQualificationError("capacity must remain non-production")


def _run_projection(
    report_path: Path,
    report_raw: bytes,
    report: dict[str, Any],
    lifecycle_path: Path,
    lifecycle_raw: bytes,
    lifecycle: dict[str, Any],
) -> dict[str, Any]:
    execution = report["execution"]
    quality = report["quality"]
    capacity = report["capacity"]
    gpu = report["gpu_telemetry"]["summary"][0]
    return {
        "report": _reference(report_path, report_raw),
        "lifecycle": _reference(lifecycle_path, lifecycle_raw),
        "lifecycle_wall_seconds": lifecycle["wall_seconds"],
        "configuration": {
            "batch_size": report["configuration"]["batch_size"],
            "batch_packing_policy": report["configuration"].get(
                "batch_packing_policy", "not-applicable-serial-v1"
            ),
            "multi_claim_policy": report["configuration"].get(
                "multi_claim_policy",
                (
                    "not-applicable-serial-v1"
                    if report["configuration"]["batch_size"] == 1
                    else "batch-v1"
                ),
            ),
            "execution_mode": report["configuration"]["execution_mode"],
            "max_image_side": report["configuration"]["max_image_side"],
            "gpu_weight_memory_gib": report["configuration"]["gpu_weight_memory_gib"],
            "cpu_weight_memory_gib": report["configuration"]["cpu_weight_memory_gib"],
        },
        "load_seconds": report["load"]["load_seconds"],
        "execution": {
            key: execution[key]
            for key in (
                "wall_seconds",
                "physical_call_wall_seconds_sum",
                "physical_generation_seconds_sum",
                "processor_handoff_seconds_sum",
                "batch_count",
                "case_count",
            )
        },
        "quality": quality,
        "capacity": capacity,
        "gpu": {
            key: gpu[key]
            for key in (
                "gpu_name",
                "sample_count",
                "memory_total_mib",
                "memory_used_mib_max",
                "utilization_gpu_percent_mean",
                "utilization_gpu_percent_max",
                "power_draw_watts_mean",
                "power_draw_watts_max",
                "temperature_celsius_mean",
                "temperature_celsius_max",
            )
        },
    }


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    if len(arguments.hybrid_report) != 2 or len(arguments.hybrid_lifecycle) != 2:
        raise QwenBatchQualificationError("exactly two hybrid recomputations are required")
    specs = {
        "serial_control": (
            arguments.serial_report,
            arguments.serial_lifecycle,
            1,
            "batch-v1",
            True,
        ),
        "naive_batch2": (
            arguments.batch2_report,
            arguments.batch2_lifecycle,
            2,
            "batch-v1",
            False,
        ),
        "grouped_batch4": (
            arguments.batch4_report,
            arguments.batch4_lifecycle,
            4,
            "batch-v1",
            False,
        ),
        "grouped_batch8": (
            arguments.batch8_report,
            arguments.batch8_lifecycle,
            8,
            "batch-v1",
            False,
        ),
    }
    runs: dict[str, dict[str, Any]] = {}
    for name, (report_path, lifecycle_path, size, multi_policy, quality_pass) in specs.items():
        report_raw, report = _load(report_path)
        lifecycle_raw, lifecycle = _load(lifecycle_path)
        _validate_run(
            report,
            batch_size=size,
            multi_claim_policy=multi_policy,
            expected_quality_pass=quality_pass,
        )
        _validate_lifecycle(lifecycle)
        runs[name] = _run_projection(
            report_path,
            report_raw,
            report,
            lifecycle_path,
            lifecycle_raw,
            lifecycle,
        )

    hybrid_runs: list[dict[str, Any]] = []
    for report_path, lifecycle_path in zip(
        arguments.hybrid_report, arguments.hybrid_lifecycle, strict=True
    ):
        report_raw, report = _load(report_path)
        lifecycle_raw, lifecycle = _load(lifecycle_path)
        _validate_run(
            report,
            batch_size=4,
            multi_claim_policy="serial-v1",
            expected_quality_pass=True,
        )
        _validate_lifecycle(lifecycle)
        hybrid_runs.append(
            _run_projection(
                report_path,
                report_raw,
                report,
                lifecycle_path,
                lifecycle_raw,
                lifecycle,
            )
        )

    serial_wall = runs["serial_control"]["execution"]["wall_seconds"]
    hybrid_walls = [run["execution"]["wall_seconds"] for run in hybrid_runs]
    hybrid_rtfs = [run["capacity"]["camera_real_time_multiple"] for run in hybrid_runs]
    conservative_rtf = min(hybrid_rtfs)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "corpus": {
            "semantic_sha256": CORPUS_SEMANTIC_SHA256,
            "request_count": 51,
            "task_counts": {"QA_COARSE": 41, "QA_DENSE": 10},
            "image_reference_count": 306,
            "unique_image_count": 276,
            "camera_count": 6,
            "source_recording_seconds": 40.833513001,
            "scope": "QWEN_R12_QA_ONLY_NOT_FULL_PIPELINE",
        },
        "runs": {
            **runs,
            "batch4_hybrid_recomputations": hybrid_runs,
        },
        "comparison": {
            "serial_execution_wall_seconds": serial_wall,
            "hybrid_execution_wall_seconds_median": statistics.median(hybrid_walls),
            "hybrid_execution_wall_seconds_range": [min(hybrid_walls), max(hybrid_walls)],
            "hybrid_speedup_median": serial_wall / statistics.median(hybrid_walls),
            "hybrid_wall_reduction_fraction": 1.0 - statistics.median(hybrid_walls) / serial_wall,
            "hybrid_conservative_camera_real_time_multiple": conservative_rtf,
            "hybrid_local_equivalent_lanes_for_25x_camera_hours": math.ceil(
                25.0 / conservative_rtf
            ),
            "hybrid_normalized_exact_match_count": 51,
            "hybrid_raw_exact_match_count": 51,
            "batch4_all_native_normalized_match_count": runs["grouped_batch4"]["quality"][
                "normalized_exact_match_count"
            ],
            "batch8_normalized_match_count": runs["grouped_batch8"]["quality"][
                "normalized_exact_match_count"
            ],
        },
        "decision": {
            "batch_candidate": "ACCEPT_BATCH4_HYBRID_FOR_VERSIONED_LOCAL_ENDPOINT_INTEGRATION",
            "selected_policy": {
                "batch_size": 4,
                "batch_packing_policy": "task-claim-group-v1",
                "single_claim_physical_path": "NATIVE_BATCH_GENERATE_V1",
                "multi_claim_physical_path": "SERIAL_GENERATE_V1",
                "multi_claim_policy": "serial-v1",
            },
            "qwen_route_state": "HOLD_FULL_PRODUCTION_QUALIFICATION",
            "mage_route_state": "HOLD",
            "rejected": {
                "all_native_batch2": "one normalized QA decision differs from serial control",
                "all_native_batch4": "one normalized QA decision differs from serial control",
                "all_native_batch8": "two normalized QA decisions differ and throughput regresses",
            },
            "rollback": "select unchanged serial Qwen binding; do not rewrite artifacts or cache",
            "remaining_gates": [
                "same cam_01 5x8s common QA/event/evidence/track/fusion comparison",
                "representative labeled quality data",
                "versioned batch endpoint/adapter/idempotency integration",
                "Linux H100 BF16 and sustained multi-worker qualification",
                "full production pipeline rather than QA-only workload",
            ],
        },
        "semantic_sha256": "0" * 64,
    }
    report["semantic_sha256"] = semantic_sha256(
        {key: value for key, value in report.items() if key != "semantic_sha256"}
    )
    validate(report)
    return report


def validate(report: dict[str, Any]) -> None:
    if report.get("report_version") != REPORT_VERSION:
        raise QwenBatchQualificationError("report version differs")
    if (
        report.get("authority") != "LOCAL_NONPRODUCTION_ONLY"
        or report.get("production_eligible") is not False
    ):
        raise QwenBatchQualificationError("local authority must remain non-production")
    if report["corpus"]["semantic_sha256"] != CORPUS_SEMANTIC_SHA256:
        raise QwenBatchQualificationError("corpus semantic identity differs")
    comparison = report["comparison"]
    runs = report["runs"]["batch4_hybrid_recomputations"]
    if len(runs) != 2:
        raise QwenBatchQualificationError("two hybrid runs are required")
    if any(run["quality"]["quality_gate_pass"] is not True for run in runs):
        raise QwenBatchQualificationError("hybrid recomputation must pass exact quality parity")
    conservative = min(run["capacity"]["camera_real_time_multiple"] for run in runs)
    if comparison["hybrid_conservative_camera_real_time_multiple"] != conservative:
        raise QwenBatchQualificationError("conservative hybrid capacity differs")
    if comparison["hybrid_local_equivalent_lanes_for_25x_camera_hours"] != math.ceil(
        25.0 / conservative
    ):
        raise QwenBatchQualificationError("25x lane formula differs")
    if report["decision"]["qwen_route_state"] != "HOLD_FULL_PRODUCTION_QUALIFICATION":
        raise QwenBatchQualificationError("Qwen full route must remain HOLD")
    projection = {key: value for key, value in report.items() if key != "semantic_sha256"}
    if report["semantic_sha256"] != semantic_sha256(projection):
        raise QwenBatchQualificationError("report semantic SHA differs")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build(arguments)
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "exact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "semantic_sha256": report["semantic_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
