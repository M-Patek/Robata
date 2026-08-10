"""Build the local Mage spatial-sampling qualification report.

This is same-sample local evidence only.  It deliberately keeps model lifecycle,
recurring stream service, output length, spatial tokens, quality inspection, and
capacity units separate; it cannot promote a production route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256  # noqa: E402

REPORT_VERSION = "mage-spatial-sampling-qualification-v1"
MEDIA_SECONDS = 40.0
CAMERA_TARGET_RTF = 25.0
RECORDING_TARGET_RTF_IF_SIX_INDEPENDENT_CAMERAS = 150.0


class SpatialEvidenceError(ValueError):
    """The local spatial qualification evidence is incomplete or inconsistent."""


def _load(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.resolve().read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SpatialEvidenceError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise SpatialEvidenceError(f"JSON root must be an object: {path}")
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


def _gpu(stream: dict[str, Any]) -> dict[str, object]:
    telemetry = stream["execution"]["gpu_telemetry"]
    summaries = telemetry["device_summaries"]
    if telemetry["measurement_status"] != "MEASURED" or len(summaries) != 1:
        raise SpatialEvidenceError("stream GPU telemetry must contain one measured device")
    item = summaries[0]
    return {
        key: item[key]
        for key in (
            "gpu_index",
            "gpu_name",
            "sample_count",
            "memory_total_mib",
            "memory_used_mib_mean",
            "memory_used_mib_max",
            "utilization_gpu_percent_mean",
            "utilization_gpu_percent_max",
            "power_draw_watts_mean",
            "power_draw_watts_max",
            "temperature_celsius_mean",
            "temperature_celsius_max",
        )
    }


def _bindings(directory: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for path in directory.rglob("*.json"):
        raw, binding = _load(path)
        ordinal = binding["context"]["focus_segment_ordinal"]
        if not isinstance(ordinal, int) or not 0 <= ordinal < 5 or ordinal in rows:
            raise SpatialEvidenceError("accepted binding ordinals must be unique 0..4")
        response = binding["endpoint_response"]
        result = response["result_artifact"]
        rows[ordinal] = {
            "ordinal": ordinal,
            "binding": _reference(path, raw),
            "request_identity_sha256": binding["request_identity_sha256"],
            "inference_identity": binding["inference_identity"],
            "result_artifact_exact_sha256": binding["result_artifact_exact_sha256"],
            "generation_seconds": response["generation_seconds"],
            "prompt_tokens": response["prompt_tokens"],
            "output_tokens": response["output_tokens"],
            "output_text": response["output_text"],
            "result_artifact_identity": result["artifact_identity"],
            "load_seconds": response["load_seconds"],
        }
    if set(rows) != set(range(5)):
        raise SpatialEvidenceError("exactly five accepted inference bindings are required")
    return [rows[index] for index in range(5)]


def _preparation(receipt_path: Path, cache_path: Path) -> dict[str, object]:
    receipt_raw, receipt = _load(receipt_path)
    cache_raw, cache = _load(cache_path)
    per_job = receipt["measurement"]["per_job_wall_seconds"]
    job_sum = receipt["measurement"]["sum_job_wall_seconds"]
    if len(per_job) != 5 or not math.isclose(sum(per_job), job_sum, abs_tol=1e-9):
        raise SpatialEvidenceError("preparation receipt timing is inconsistent")
    if cache["manifest_semantic_sha256"] != semantic_sha256(
        {key: value for key, value in cache.items() if key != "manifest_semantic_sha256"}
    ):
        raise SpatialEvidenceError("cache manifest semantic hash differs")
    return {
        "receipt": _reference(receipt_path, receipt_raw),
        "cache_manifest": {
            **_reference(cache_path, cache_raw),
            "semantic_sha256": cache["manifest_semantic_sha256"],
        },
        "policy": receipt["policy"],
        "identity": {
            "namespace_identity": cache["namespace_identity"],
            "provider_identity_sha256": cache["provider_identity_sha256"],
            "provider_implementation_sha256": cache["provider_implementation_sha256"],
            "toolchain_identity_sha256": cache["toolchain"]["toolchain_identity_sha256"],
            "container_image_digest": cache["toolchain"]["container_image_digest"],
        },
        "measurement": {
            "per_segment_seconds": per_job,
            "worker_job_sum_seconds": job_sum,
            "workload_wall_seconds": receipt["measurement"]["workload_wall_seconds"],
            "worker_job_sum_realtime_factor": MEDIA_SECONDS / job_sum,
            "workload_realtime_factor": MEDIA_SECONDS
            / receipt["measurement"]["workload_wall_seconds"],
        },
        "canvas_count_per_segment": [job["output"]["canvas_count"] for job in receipt["jobs"]],
        "patch_rows_per_segment": [
            job["output"]["src_positions"]["row_count"] for job in receipt["jobs"]
        ],
    }


def _run(root: Path, *, lifecycle_required: bool) -> dict[str, object]:
    stream_path = root / "stream-report.json"
    stream_raw, stream = _load(stream_path)
    if stream.get("ok") is not True or stream["execution"]["normal_model_call_count"] != 5:
        raise SpatialEvidenceError(f"stream did not complete five model calls: {root}")
    timing = stream["execution"]["execution_timing"]
    rows = _bindings(root / "stream-artifacts" / "accepted-inference-binding")
    generation = [float(row["generation_seconds"]) for row in rows]
    output_tokens = [int(row["output_tokens"]) for row in rows]
    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    output_texts = [str(row["output_text"]) for row in rows]
    result: dict[str, object] = {
        "stream_report": _reference(stream_path, stream_raw),
        "timing": {
            "run_wall_seconds": timing["run_wall_seconds"],
            "end_to_end_realtime_factor": timing["end_to_end_realtime_factor"],
            "observation_interval_union_seconds": timing["observation_interval_union_seconds"],
            "preparation_interval_union_seconds": timing["preparation_interval_union_seconds"],
            "preparation_observation_overlap_seconds": timing[
                "preparation_observation_overlap_seconds"
            ],
        },
        "generation": {
            "sum_seconds": sum(generation),
            "warm_sum_seconds": sum(generation[1:]),
            "per_segment_seconds": generation,
            "prompt_tokens_per_segment": prompt_tokens,
            "prompt_tokens_total": sum(prompt_tokens),
            "output_tokens_per_segment": output_tokens,
            "output_tokens_total": sum(output_tokens),
            "warm_output_tokens": sum(output_tokens[1:]),
            "warm_milliseconds_per_output_token": 1000.0
            * sum(generation[1:])
            / sum(output_tokens[1:]),
            "output_text_sequence_sha256": hashlib.sha256(
                canonical_json_bytes(output_texts)
            ).hexdigest(),
            "strict_json_all_segments": all(
                isinstance(json.loads(text), dict) for text in output_texts
            ),
            "output_budget_exhausted": any(tokens >= 256 for tokens in output_tokens),
            "per_segment": rows,
        },
        "gpu": _gpu(stream),
    }
    if lifecycle_required:
        lifecycle_path = root / "lifecycle-report.json"
        lifecycle_raw, lifecycle = _load(lifecycle_path)
        ownership = lifecycle["process_ownership"]
        if (
            lifecycle["outcome"] != "SUCCEEDED"
            or not ownership["post_shutdown_port_closed"]
            or not lifecycle["endpoint"]["cleanup"]["reaped"]
        ):
            raise SpatialEvidenceError("bounded lifecycle did not cleanly reap the endpoint")
        ready = next(
            item
            for item in lifecycle["events"]
            if item["phase"] == "ENDPOINT" and item["state"] == "READY"
        )
        result["lifecycle"] = {
            "report": _reference(lifecycle_path, lifecycle_raw),
            "wall_seconds": lifecycle["wall_seconds"],
            "endpoint_ready_offset_seconds": ready["elapsed_seconds"],
            "endpoint_cleanup": lifecycle["endpoint"]["cleanup"],
            "post_shutdown_port_closed": ownership["post_shutdown_port_closed"],
            "containment": ownership["containment"],
        }
    return result


def _profile_summary(run: dict[str, Any]) -> dict[str, object]:
    return {
        "end_to_end_realtime_factor": run["timing"]["end_to_end_realtime_factor"],
        "run_wall_seconds": run["timing"]["run_wall_seconds"],
        "prompt_tokens_total": run["generation"]["prompt_tokens_total"],
        "output_tokens_total": run["generation"]["output_tokens_total"],
        "generation_sum_seconds": run["generation"]["sum_seconds"],
        "warm_milliseconds_per_output_token": run["generation"][
            "warm_milliseconds_per_output_token"
        ],
        "memory_used_mib_max": run["gpu"]["memory_used_mib_max"],
    }


def validate(report: dict[str, Any]) -> None:
    if report.get("report_version") != REPORT_VERSION:
        raise SpatialEvidenceError("report version differs")
    if (
        report.get("production_eligible") is not False
        or report.get("authority") != "LOCAL_NONPRODUCTION_ONLY"
    ):
        raise SpatialEvidenceError("local authority must remain non-production")
    candidate = report["profiles"]["traditional_8x131k"]
    runs = candidate["fresh_recomputations"]
    if len(runs) != 2:
        raise SpatialEvidenceError("131K candidate requires two fresh recomputations")
    digests = {run["generation"]["output_text_sequence_sha256"] for run in runs}
    if len(digests) != 1 or not candidate["semantic_output_recomputation_stable"]:
        raise SpatialEvidenceError("131K output recomputation is not stable")
    if report["decision"]["state"] != "HOLD_MAGE_SPATIAL":
        raise SpatialEvidenceError("local spatial decision must remain HOLD")
    conservative = min(run["timing"]["end_to_end_realtime_factor"] for run in runs)
    comparison = report["comparison"]
    if comparison["traditional_8x131k"]["conservative_realtime_factor"] != conservative:
        raise SpatialEvidenceError("conservative 131K realtime factor differs")
    if comparison["traditional_8x131k"]["local_lanes_for_25x"] != math.ceil(
        CAMERA_TARGET_RTF / conservative
    ):
        raise SpatialEvidenceError("25x lane formula differs")
    projection = dict(report)
    claimed = projection.pop("semantic_sha256", None)
    if claimed != semantic_sha256(projection):
        raise SpatialEvidenceError("report semantic hash differs")


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    control_raw, control = _load(arguments.control_report)
    profile_98 = _run(arguments.run_98k, lifecycle_required=False)
    run_131_a = _run(arguments.run_131k_a, lifecycle_required=True)
    run_131_b = _run(arguments.run_131k_b, lifecycle_required=True)
    if (
        run_131_a["generation"]["output_text_sequence_sha256"]
        != run_131_b["generation"]["output_text_sequence_sha256"]
    ):
        raise SpatialEvidenceError("fresh 131K output text sequences differ")
    contact_raw = arguments.contact_sheet.read_bytes()
    conservative = min(
        run_131_a["timing"]["end_to_end_realtime_factor"],
        run_131_b["timing"]["end_to_end_realtime_factor"],
    )
    mean_rtf = mean(
        [
            run_131_a["timing"]["end_to_end_realtime_factor"],
            run_131_b["timing"]["end_to_end_realtime_factor"],
        ]
    )
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "observed_at_date": "2026-08-09",
        "repository": {
            "branch": "codex/mage-25x-convergence-20260809",
            "base_commit": "ff1ee609f9899b296bc89d8aab9a37317ed75f54",
            "worktree_dirty_during_measurement": True,
        },
        "scope": {
            "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "source_duration_seconds": MEDIA_SECONDS,
            "camera": "cam_01",
            "segment_count": 5,
            "segment_duration_seconds": 8,
            "generation_lanes": 1,
            "preparation_workers": 1,
            "model_load_profile": "bitsandbytes_4bit_nf4_v1",
            "decoder_output_profile": "full-v1",
            "max_new_tokens": 256,
            "representative_labeled_quality_set": False,
            "h100_measured": False,
        },
        "sources": {
            "tracked_65k_control": _reference(arguments.control_report, control_raw),
            "visual_contact_sheet_131k": _reference(arguments.contact_sheet, contact_raw),
        },
        "profiles": {
            "traditional_8x65k": {
                "source": "tracked_65k_control",
                "summary": {
                    "end_to_end_realtime_factor": control["traditional_target_canvas_8"][
                        "generation"
                    ]["timing"]["hot_end_to_end_realtime_factor"],
                    "run_wall_seconds": control["traditional_target_canvas_8"]["generation"][
                        "timing"
                    ]["stream_run_wall_seconds"],
                    "prompt_tokens_total": 5 * 767,
                    "output_tokens_total": sum(
                        item["output_tokens"]
                        for item in control["traditional_target_canvas_8"]["generation"][
                            "per_segment"
                        ]
                    ),
                    "generation_sum_seconds": sum(
                        item["generation_seconds"]
                        for item in control["traditional_target_canvas_8"]["generation"][
                            "per_segment"
                        ]
                    ),
                    "memory_used_mib_max": control["traditional_target_canvas_8"]["generation"][
                        "gpu_telemetry"
                    ]["device"]["memory_used_mib_max"],
                },
                "quality": {
                    "state": "HOLD",
                    "finding": (
                        "segment 4 claimed a green book; retained visual evidence "
                        "shows green fabric"
                    ),
                },
            },
            "traditional_8x98k": {
                "preparation": _preparation(arguments.prep_98k, arguments.cache_98k),
                "run": profile_98,
                "summary": _profile_summary(profile_98),
                "quality": {
                    "state": "HOLD",
                    "improvements": ["segment 4 no longer claims a green book"],
                    "failures": [
                        "segment 0 repeats the same look-at-shirt observation four times",
                        "segment 2 changes the same green garment to a pair of pants",
                    ],
                },
            },
            "traditional_8x131k": {
                "preparation": _preparation(arguments.prep_131k, arguments.cache_131k),
                "fresh_recomputations": [run_131_a, run_131_b],
                "semantic_output_recomputation_stable": True,
                "mean_end_to_end_realtime_factor": mean_rtf,
                "conservative_end_to_end_realtime_factor": conservative,
                "quality": {
                    "inspection_authority": (
                        "AGENT_VISUAL_INSPECTION_SINGLE_RECORDING_NOT_LABELED_GROUND_TRUTH"
                    ),
                    "state": "HOLD_PARETO_CANDIDATE",
                    "improvements": [
                        "segment 0 collapses repeated look-at claims into pick-up plus fold",
                        "segment 4 identifies green cloth and removes the unsupported book",
                        "both fresh recomputations produce the exact same ordered output text",
                    ],
                    "remaining_failures": [
                        "segment 2 calls the same green garment a drawstring bag",
                        "segment 3 calls the same green garment a pant",
                        "single-camera confidence and visibility remain overconfident at 1.0",
                    ],
                },
            },
        },
        "comparison": {
            "traditional_8x65k": control["traditional_target_canvas_8"]["generation"]["timing"],
            "traditional_8x98k": _profile_summary(profile_98),
            "traditional_8x131k": {
                "mean_realtime_factor": mean_rtf,
                "conservative_realtime_factor": conservative,
                "local_lanes_for_25x": math.ceil(CAMERA_TARGET_RTF / conservative),
                "local_lanes_for_150x_if_six_independent_cameras": math.ceil(
                    RECORDING_TARGET_RTF_IF_SIX_INDEPENDENT_CAMERAS / conservative
                ),
                "prompt_tokens_total": run_131_a["generation"]["prompt_tokens_total"],
                "output_tokens_total": run_131_a["generation"]["output_tokens_total"],
                "warm_milliseconds_per_output_token": run_131_a["generation"][
                    "warm_milliseconds_per_output_token"
                ],
                "memory_used_mib_max": max(
                    run_131_a["gpu"]["memory_used_mib_max"],
                    run_131_b["gpu"]["memory_used_mib_max"],
                ),
            },
            "interpretation": {
                "higher_resolution_is_not_intrinsically_faster": True,
                "observed_wall_improvement_driver": "shorter deterministic output sequence",
                "warm_per_output_token_cost_increased": True,
                "gpu_utilization_not_used_as_promotion_evidence": True,
            },
        },
        "six_by_131k": {
            "state": "NOT_RUN_NOT_JUSTIFIED_BY_CURRENT_QUALITY_GATE",
            "reason": (
                "reducing temporal canvases does not address the remaining "
                "cross-segment object-identity drift and risks missing short "
                "actions/boundaries"
            ),
            "docker_daemon_observed_unavailable_after_decision": True,
        },
        "capacity_contract": {
            "camera_hours_per_day": 500,
            "headroom": 1.2,
            "required_aggregate_realtime_factor": CAMERA_TARGET_RTF,
            "recording_hours_requirement_interpretation_unresolved": True,
            "six_independent_camera_realtime_factor": (
                RECORDING_TARGET_RTF_IF_SIX_INDEPENDENT_CAMERAS
            ),
        },
        "decision": {
            "state": "HOLD_MAGE_SPATIAL",
            "selected_local_profile": "traditional_8x131k",
            "selected_profile_role": "QUALITY_FIRST_LOCAL_CANDIDATE_NOT_PRODUCTION_DEFAULT",
            "rollback_profile": "traditional_8x98k",
            "reasons": [
                (
                    "131K removes the prominent duplicate/book failures and is "
                    "byte-stable at the semantic output-text level across two "
                    "fresh endpoint states"
                ),
                "remaining garment taxonomy drift fails representative quality promotion",
                (
                    "the local observed speedup is output-length driven and "
                    "cannot be extrapolated to H100"
                ),
                (
                    "target H100 BF16 service and the 25x versus 150x capacity "
                    "unit remain unmeasured/unresolved"
                ),
            ],
            "next_gate": (
                "TEMPORAL_OBJECT_CONSISTENCY_OR_BOUNDED_LABEL_REFINE_PLUS_"
                "REPRESENTATIVE_LABELED_QUALITY"
            ),
        },
    }
    report["semantic_sha256"] = semantic_sha256(report)
    validate(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-report",
        type=Path,
        default=ROOT / "docs" / "mage-traditional-codec-generation-qualification-2026-08-09.json",
    )
    parser.add_argument(
        "--prep-98k", type=Path, default=ROOT / ".tmp" / "spatial-8x98k" / "container-receipt.json"
    )
    parser.add_argument(
        "--cache-98k",
        type=Path,
        default=ROOT / ".tmp" / "spatial-8x98k-cache-v1" / "traditional-cache-manifest-v1.json",
    )
    parser.add_argument("--run-98k", type=Path, default=ROOT / ".tmp" / "spatial-8x98k-generation")
    parser.add_argument(
        "--prep-131k",
        type=Path,
        default=ROOT / ".tmp" / "spatial-8x131k" / "container-receipt.json",
    )
    parser.add_argument(
        "--cache-131k",
        type=Path,
        default=ROOT / ".tmp" / "spatial-8x131k-cache-v1" / "traditional-cache-manifest-v1.json",
    )
    parser.add_argument(
        "--run-131k-a", type=Path, default=ROOT / ".tmp" / "spatial-8x131k-bounded-r1"
    )
    parser.add_argument(
        "--run-131k-b", type=Path, default=ROOT / ".tmp" / "spatial-8x131k-bounded-r2"
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=ROOT / ".tmp" / "spatial-8x131k-bounded-r1" / "131k-contact-sheet.jpg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "mage-spatial-sampling-qualification-2026-08-09.json",
    )
    return parser


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
