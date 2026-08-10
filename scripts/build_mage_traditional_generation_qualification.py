"""Build the additive local Mage traditional-codec generation qualification report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256  # noqa: E402

REPORT_VERSION = "mage-traditional-codec-generation-qualification-v1"
BASELINE_EXACT = "7298d21fb05f0ecbc4bc1e11481f67abf2c82b4b13380227177edfbbbaa24287"
BASELINE_SEMANTIC = "ea659e3e78243e43e4c1f921ff0898c64f18c4e68993c9c219d2425c8a25b0d8"


class EvidenceError(ValueError):
    """Local qualification evidence is missing or inconsistent."""


def load(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.resolve().read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return raw, value


def ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def source(path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "path": ref(path),
        "byte_count": len(raw),
        "exact_sha256": hashlib.sha256(raw).hexdigest(),
    }


def telemetry(path: Path) -> dict[str, Any]:
    raw, data = load(path)
    if data.get("measurement_status") != "MEASURED" or len(data.get("summary", [])) != 1:
        raise EvidenceError(f"GPU telemetry is not a one-device measured artifact: {path}")
    item = data["summary"][0]
    keys = (
        "gpu_index",
        "gpu_name",
        "sample_count",
        "memory_total_mib",
        "memory_used_mib_mean",
        "memory_used_mib_max",
        "memory_used_fraction_mean",
        "memory_used_fraction_max",
        "utilization_gpu_percent_mean",
        "utilization_gpu_percent_max",
        "power_draw_watts_mean",
        "power_draw_watts_max",
        "temperature_celsius_mean",
        "temperature_celsius_max",
    )
    return {
        "measurement_status": "MEASURED",
        "source_artifact": source(path, raw),
        "device": {key: item[key] for key in keys},
        "errors": data.get("errors", []),
    }


def preparation(
    receipt_path: Path, host_path: Path, cache_path: Path, media: float
) -> dict[str, Any]:
    receipt_raw, receipt = load(receipt_path)
    host_raw, host = load(host_path)
    cache_raw, cache = load(cache_path)
    per_segment = receipt["measurement"]["per_job_wall_seconds"]
    job_sum = receipt["measurement"]["sum_job_wall_seconds"]
    if len(per_segment) != 5 or not math.isclose(sum(per_segment), job_sum, abs_tol=1e-9):
        raise EvidenceError("traditional preparation job timing differs")
    semantic = cache["manifest_semantic_sha256"]
    projection = dict(cache)
    projection.pop("manifest_semantic_sha256")
    if semantic_sha256(projection) != semantic:
        raise EvidenceError("traditional cache manifest semantic hash differs")
    host_wall = host["host_wall_seconds"]
    workload = receipt["measurement"]["workload_wall_seconds"]
    toolchain = cache["toolchain"]
    return {
        "source_receipt": {
            **source(receipt_path, receipt_raw),
            "content_sha256": receipt["receipt_content_sha256"],
        },
        "source_host_measurement": source(host_path, host_raw),
        "source_cache_manifest": {**source(cache_path, cache_raw), "semantic_sha256": semantic},
        "policy": receipt["policy"],
        "cache_identity": {
            "namespace_identity": cache["namespace_identity"],
            "provider_identity_sha256": cache["provider_identity_sha256"],
            "codec_policy_sha256": cache["codec_policy_sha256"],
            "codec_config_sha256": cache["codec_config_sha256"],
            "toolchain_identity_sha256": toolchain["toolchain_identity_sha256"],
            "container_image_digest": toolchain["container_image_digest"],
        },
        "measurement": {
            "per_segment_seconds": per_segment,
            "worker_job_sum_seconds": job_sum,
            "workload_wall_seconds": workload,
            "host_wall_seconds": host_wall,
            "worker_job_sum_realtime_factor": media / job_sum,
            "workload_realtime_factor": media / workload,
            "host_envelope_realtime_factor": media / host_wall,
        },
    }


def dcvc_control(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, report = load(path)
    if hashlib.sha256(raw).hexdigest() != BASELINE_EXACT:
        raise EvidenceError("DCVC control exact hash differs")
    if report.get("semantic_sha256") != BASELINE_SEMANTIC:
        raise EvidenceError("DCVC control semantic hash differs")
    variant = report["variants"]["provider_v2_bounded"][0]
    prep = variant["preparation"]
    generation = variant["generation"]
    measurement = generation["measurement"]
    rows = [
        {
            "ordinal": item["ordinal"],
            "generation_seconds": item["generate_seconds"],
            "observation_seconds": item["observation_seconds"],
            "prompt_tokens": item["prompt_tokens"],
            "output_tokens": item["output_tokens"],
            "output_budget_exhausted": item["output_budget_exhausted"],
            "result_artifact_exact_sha256": item["result_artifact_exact_sha256"],
            "normalized_output_semantic_sha256": item["normalized_output_semantic_sha256"],
        }
        for item in measurement["per_segment"]
    ]
    media = generation["sample"]["media_duration_ns"] / 1e9
    generation_sum = sum(item["generation_seconds"] for item in rows)
    warm_sum = sum(item["generation_seconds"] for item in rows[1:])
    warm_media = sum(
        (item["end_ns"] - item["start_ns"]) / 1e9 for item in generation["sample"]["segments"][1:]
    )
    gpu = generation["stream_gpu_telemetry"]
    control = {
        "source_report": {**source(path, raw), "semantic_sha256": BASELINE_SEMANTIC},
        "variant_id": variant["variant_id"],
        "camera_count": 1,
        "segment_count": 5,
        "media_seconds": media,
        "preparation": {
            "wall_seconds": prep["wall_seconds"],
            "worker_job_sum_seconds": sum(
                item["preparation_seconds"] for item in prep["per_segment"]
            ),
            "model_load_seconds": prep["provider_model_load_seconds"],
            "model_load_included_in_wall": prep["startup_load_included_in_wall"],
        },
        "generation": {
            "stream_run_wall_seconds": measurement["stream_run_wall_seconds"],
            "stream_realtime_factor": media / measurement["stream_run_wall_seconds"],
            "generation_sum_seconds": generation_sum,
            "generation_realtime_factor": media / generation_sum,
            "first_generation_seconds": rows[0]["generation_seconds"],
            "warm_generation_mean_seconds": mean(item["generation_seconds"] for item in rows[1:]),
            "warm_generation_realtime_factor": warm_media / warm_sum,
            "model_load_seconds": measurement["model_load_seconds"],
            "model_load_included_in_overall_wall": measurement[
                "model_load_included_in_overall_wall"
            ],
            "model_load_included_in_stream_run_wall": False,
            "per_segment": rows,
        },
        "gpu_telemetry": {
            "measurement_status": gpu["measurement_status"],
            "device": gpu["devices"][0],
            "source_artifact": gpu["source_artifact"],
        },
        "production_eligible": False,
    }
    return control, report


def traditional_generation(
    stream_path: Path,
    gpu_path: Path,
    bindings_dir: Path,
    results_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stream_raw, stream = load(stream_path)
    if stream.get("ok") is not True:
        raise EvidenceError("traditional stream did not succeed")
    timing = stream["execution"]["execution_timing"]
    segments = {item["ordinal"]: item for item in stream["plan"]["storage_segments"]}
    results: dict[str, tuple[Path, bytes, dict[str, Any]]] = {}
    for path in results_dir.glob("*.json"):
        raw, item = load(path)
        results[hashlib.sha256(raw).hexdigest()] = (path, raw, item)
    if len(results) != 5:
        raise EvidenceError("traditional result set must contain five artifacts")
    rows: dict[int, dict[str, Any]] = {}
    binding_sources = []
    load_seconds = set()
    for path in bindings_dir.rglob("*.json"):
        raw, binding = load(path)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != path.stem:
            raise EvidenceError("binding filename is not its exact hash")
        ordinal = binding["context"]["focus_segment_ordinal"]
        response = binding["endpoint_response"]
        result_exact = binding["result_artifact_exact_sha256"]
        result_path, result_raw, result = results[result_exact]
        if result["request_id"] != response["request_id"]:
            raise EvidenceError("binding/result request differs")
        output = json.loads(response["output_text"])
        interval = segments[ordinal]["interval"]
        duration = (interval["end_ns"] - interval["start_ns"]) / 1e9
        rows[ordinal] = {
            "ordinal": ordinal,
            "segment_duration_seconds": duration,
            "segment_semantic_sha256": segments[ordinal]["segment_semantic_sha256"],
            "generation_seconds": response["generation_seconds"],
            "prompt_tokens": response["prompt_tokens"],
            "output_tokens": response["output_tokens"],
            "output_budget_exhausted": response["output_tokens"]
            >= stream["decoder"]["max_new_tokens"],
            "strict_json": True,
            "observation_count": len(output["observations"]),
            "output_text_sha256": hashlib.sha256(response["output_text"].encode()).hexdigest(),
            "request_id": response["request_id"],
            "binding_exact_sha256": digest,
            "result_artifact": source(result_path, result_raw),
            "model_identity_sha256": response["inference_identity"]["model_identity_sha256"],
            "checkpoint_manifest_sha256": response["inference_identity"]["model_identity"][
                "checkpoint_manifest_sha256"
            ],
        }
        load_seconds.add(response["load_seconds"])
        binding_sources.append(source(path, raw))
    if sorted(rows) != list(range(5)) or len(load_seconds) != 1:
        raise EvidenceError("traditional bindings are incomplete or disagree on model load")
    ordered = [rows[index] for index in range(5)]
    media = timing["media_duration_seconds"]
    generation_sum = sum(item["generation_seconds"] for item in ordered)
    warm_sum = sum(item["generation_seconds"] for item in ordered[1:])
    warm_media = sum(item["segment_duration_seconds"] for item in ordered[1:])
    projection = {
        "source_report": source(stream_path, stream_raw),
        "source_bindings": sorted(binding_sources, key=lambda item: item["path"]),
        "selected_camera": stream["selected_camera"],
        "camera_count": 1,
        "segment_count": 5,
        "media_seconds": media,
        "decoder": stream["decoder"],
        "lifecycle": {
            "model_load_seconds": load_seconds.pop(),
            "model_load_included_in_stream_run_wall": False,
            "model_load_included_in_generation_sum": False,
            "basis": (
                "Endpoint health preceded the stream command; load_seconds is resident "
                "endpoint metadata."
            ),
        },
        "timing": {
            "stream_run_wall_seconds": timing["run_wall_seconds"],
            "hot_end_to_end_realtime_factor": media / timing["run_wall_seconds"],
            "observation_worker_sum_seconds": timing["observation_worker_sum_seconds"],
            "observation_realtime_factor": media / timing["observation_worker_sum_seconds"],
            "generation_sum_seconds": generation_sum,
            "generation_realtime_factor": media / generation_sum,
            "first_generation_seconds": ordered[0]["generation_seconds"],
            "warm_generation_mean_seconds": mean(
                item["generation_seconds"] for item in ordered[1:]
            ),
            "warm_generation_sum_seconds": warm_sum,
            "warm_media_seconds": warm_media,
            "warm_generation_realtime_factor": warm_media / warm_sum,
        },
        "per_segment": ordered,
        "gpu_telemetry": telemetry(gpu_path),
        "execution_profile": stream["execution"]["execution_profile"],
        "normal_model_call_count": stream["execution"]["normal_model_call_count"],
        "refinement_model_call_count": stream["execution"]["refinement_model_call_count"],
        "run_manifest": stream["execution"]["run_manifest"],
    }
    return projection, stream


def quality16(
    prep: dict[str, Any],
    binding_path: Path,
    result_path: Path,
    gpu_path: Path,
) -> dict[str, Any]:
    binding_raw, binding = load(binding_path)
    result_raw, result = load(result_path)
    if hashlib.sha256(result_raw).hexdigest() != binding["result_artifact_exact_sha256"]:
        raise EvidenceError("quality16 result exact hash differs")
    output_text = result["output_text"]
    try:
        json.loads(output_text)
    except json.JSONDecodeError:
        strict_json = False
    else:
        strict_json = True
    return {
        "candidate_id": "traditional-target-canvas-16-images-per-group-2",
        "decision": "STOP",
        "production_eligible": False,
        "preparation": prep,
        "generation": {
            "source_binding": source(binding_path, binding_raw),
            "source_result": source(result_path, result_raw),
            "generation_seconds": result["generation_seconds"],
            "model_load_seconds": result["load_seconds"],
            "model_load_included_in_generation_seconds": False,
            "prompt_tokens": result["prompt_tokens"],
            "output_tokens": result["output_tokens"],
            "output_budget": 256,
            "output_budget_exhausted": result["output_tokens"] >= 256,
            "strict_json": strict_json,
            "output_text_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
            "gpu_telemetry": telemetry(gpu_path),
        },
        "quality": {
            "gate": "FAIL",
            "failure_modes": [
                "repetitive unsupported action label: unbutton a shirt",
                "duplicate one-second observations",
                "256-token budget exhausted",
                "truncated non-JSON response",
            ],
            "reason": (
                "Doubling canvas count increased prompt tokens and VRAM while producing "
                "repeated unsupported actions until truncation."
            ),
        },
    }


def inspection(directory: Path) -> dict[str, Any]:
    files = {"t02.jpg": 2.0, "t05_5.jpg": 5.5, "t34.jpg": 34.0, "t38.jpg": 38.0}
    artifacts = []
    for name, offset in files.items():
        path = directory / name
        raw = path.read_bytes()
        artifacts.append({**source(path, raw), "offset_seconds": offset})
    return {
        "method": "manual frame inspection of retained local sample",
        "source_artifacts": artifacts,
        "finding": {
            "segment_ordinal": 4,
            "severity": "OBJECT_CLASS_HALLUCINATION",
            "model_claim": "flipping through a green book",
            "visual_observation": "handling/folding green fabric; no book is visible",
            "gate": "FAIL",
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    dcvc, baseline = dcvc_control(args.baseline)
    traditional, stream = traditional_generation(
        args.traditional_stream_report,
        args.traditional_gpu_telemetry,
        args.traditional_bindings,
        args.traditional_results,
    )
    media = traditional["media_seconds"]
    prep8 = preparation(
        args.traditional_preparation_receipt,
        args.traditional_host_measurement,
        args.traditional_cache_manifest,
        media,
    )
    prep16 = preparation(
        args.quality16_preparation_receipt,
        args.quality16_host_measurement,
        args.quality16_cache_manifest,
        media,
    )
    expected_model = baseline["variants"]["provider_v2_bounded"][0]["generation"]["controls"][
        "model_identity_sha256"
    ]
    if any(item["model_identity_sha256"] != expected_model for item in traditional["per_segment"]):
        raise EvidenceError("traditional and DCVC model identity differs")
    required_camera = args.daily_camera_hours * args.headroom / 24
    required_recording = (
        args.daily_recording_hours * args.recording_camera_count * args.headroom / 24
    )
    hot_rtf = traditional["timing"]["hot_end_to_end_realtime_factor"]
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "report_date": "2026-08-09",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "evidence_class": "RETAINED_LOCAL_MEASUREMENT",
        "production_eligible": False,
        "source_artifact_retention": {
            "status": "LOCAL_ONLY_NOT_TRACKED",
            "note": (
                "Raw .tmp artifacts remain local; this report embeds exact hashes but a clean "
                "checkout cannot replay them."
            ),
        },
        "measurement_scope": {
            "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "camera_count": 1,
            "worker_count": 1,
            "generation_lane_count": 1,
            "media_seconds": media,
            "segment_count": 5,
            "selected_camera": traditional["selected_camera"],
            "recording_key": stream["plan"]["recording"]["recording_key"],
            "recording_exact_sha256": stream["plan"]["recording"]["recording_exact_sha256"],
            "source_byte_count": stream["source_byte_count"],
        },
        "capacity_target": {
            "headroom": args.headroom,
            "camera_hour_interpretation": {
                "daily_camera_hours": args.daily_camera_hours,
                "required_aggregate_realtime_factor": required_camera,
            },
            "recording_hour_interpretation": {
                "daily_recording_hours": args.daily_recording_hours,
                "camera_count_per_recording": args.recording_camera_count,
                "required_aggregate_camera_realtime_factor": required_recording,
            },
            "capacity_unit_conflict": {
                "resolved": False,
                "production_decision_blocked": True,
                "statement": (
                    "The convergence cycle uses 500 camera-hours/day (25x with 20% "
                    "headroom), while governance/REQUIREMENTS.md states 500 "
                    "recording-hours/day; six independent cameras imply 150x aggregate "
                    "camera realtime."
                ),
            },
        },
        "dcvc_control": dcvc,
        "traditional_target_canvas_8": {
            "route": "TRADITIONAL_H264_HEVC",
            "preparation": prep8,
            "generation": traditional,
            "quality": {
                "decision": "HOLD",
                "strict_json_segments": 5,
                "inspection": inspection(args.inspection),
                "reason": (
                    "All calls completed and the hot route exceeded realtime locally, but "
                    "segment 4 asserted a green book where retained frames show green fabric."
                ),
            },
            "production_eligible": False,
        },
        "traditional_target_canvas_16": quality16(
            prep16,
            args.quality16_binding,
            args.quality16_result,
            args.quality16_gpu_telemetry,
        ),
        "comparison": {
            "traditional_preparation_worker_speedup_vs_dcvc": dcvc["preparation"][
                "worker_job_sum_seconds"
            ]
            / prep8["measurement"]["worker_job_sum_seconds"],
            "traditional_host_envelope_speedup_vs_dcvc_full_wall": dcvc["preparation"][
                "wall_seconds"
            ]
            / prep8["measurement"]["host_wall_seconds"],
            "traditional_hot_end_to_end_realtime_factor": hot_rtf,
            "traditional_warm_generation_realtime_factor": traditional["timing"][
                "warm_generation_realtime_factor"
            ],
            "dcvc_stream_realtime_factor": dcvc["generation"]["stream_realtime_factor"],
            "dcvc_warm_generation_realtime_factor": dcvc["generation"][
                "warm_generation_realtime_factor"
            ],
            "local_logical_lanes_for_25x_camera_target": math.ceil(required_camera / hot_rtf),
            "local_logical_lanes_for_150x_recording_target": math.ceil(
                required_recording / hot_rtf
            ),
            "codec_bottleneck_transferred_to_decoder": True,
        },
        "decision": {
            "state": "HOLD_TRADITIONAL",
            "traditional_target_canvas_8": "HOLD",
            "traditional_target_canvas_16": "STOP",
            "dcvc_role": "RETAINED_CONTROL",
            "next_gate": "P20_COMPACT_DECODER_BUDGET_AND_QUALITY_AB",
            "reasons": [
                (
                    "traditional preparation is approximately seven times faster than DCVC "
                    "worker job sum"
                ),
                (
                    "the first eight-canvas generation emitted 203 tokens and dominated hot "
                    "stream wall"
                ),
                "the eight-canvas route has an object-class hallucination on segment 4",
                "the sixteen-canvas candidate exhausted 256 tokens and failed strict JSON",
                "the production capacity unit remains unresolved",
                "the run is single-camera, single-worker, local RTX 4060 evidence",
            ],
        },
    }
    report["semantic_sha256"] = semantic_sha256(report)
    validate(report)
    return report


def validate(report: dict[str, Any]) -> None:
    projection = deepcopy(report)
    semantic = projection.pop("semantic_sha256")
    if semantic_sha256(projection) != semantic:
        raise EvidenceError("report semantic hash differs")
    if (
        report.get("report_version") != REPORT_VERSION
        or report.get("production_eligible") is not False
    ):
        raise EvidenceError("report authority differs")
    capacity = report["capacity_target"]
    conflict = capacity["capacity_unit_conflict"]
    if (
        conflict.get("resolved") is not False
        or conflict.get("production_decision_blocked") is not True
    ):
        raise EvidenceError("capacity unit conflict must remain blocking")
    comparison = report["comparison"]
    hot = comparison["traditional_hot_end_to_end_realtime_factor"]
    camera = capacity["camera_hour_interpretation"]["required_aggregate_realtime_factor"]
    recording = capacity["recording_hour_interpretation"][
        "required_aggregate_camera_realtime_factor"
    ]
    if comparison["local_logical_lanes_for_25x_camera_target"] != math.ceil(camera / hot):
        raise EvidenceError("camera-hour lane formula differs")
    if comparison["local_logical_lanes_for_150x_recording_target"] != math.ceil(recording / hot):
        raise EvidenceError("recording-hour lane formula differs")
    if report["decision"]["state"] != "HOLD_TRADITIONAL":
        raise EvidenceError("eight-canvas decision must remain HOLD")
    candidate = report["traditional_target_canvas_16"]
    if candidate["decision"] != "STOP" or candidate["generation"]["strict_json"] is not False:
        raise EvidenceError("sixteen-canvas decision must remain STOP")
    if candidate["generation"]["output_budget_exhausted"] is not True:
        raise EvidenceError("sixteen-canvas output budget must remain exhausted")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json",
    )
    result.add_argument(
        "--traditional-preparation-receipt",
        type=Path,
        default=ROOT / ".tmp/traditional-atomic-r3/container-receipt.json",
    )
    result.add_argument(
        "--traditional-host-measurement",
        type=Path,
        default=ROOT / ".tmp/traditional-atomic-r3/host-measurement.json",
    )
    result.add_argument(
        "--traditional-cache-manifest",
        type=Path,
        default=ROOT / ".tmp/traditional-cache-v1-r3/traditional-cache-manifest-v1.json",
    )
    result.add_argument(
        "--traditional-stream-report",
        type=Path,
        default=ROOT / ".tmp/traditional-generation-r3/stream-report.json",
    )
    result.add_argument(
        "--traditional-gpu-telemetry",
        type=Path,
        default=ROOT / ".tmp/traditional-generation-r3/stream-gpu-telemetry.json",
    )
    result.add_argument(
        "--traditional-bindings",
        type=Path,
        default=ROOT / ".tmp/traditional-generation-r3/stream-artifacts/accepted-inference-binding",
    )
    result.add_argument(
        "--traditional-results",
        type=Path,
        default=ROOT / ".tmp/traditional-generation-r2/endpoint-results",
    )
    result.add_argument(
        "--inspection", type=Path, default=ROOT / ".tmp/traditional-generation-r3/inspection"
    )
    result.add_argument(
        "--quality16-preparation-receipt",
        type=Path,
        default=ROOT / ".tmp/traditional-quality16-r1/container-receipt.json",
    )
    result.add_argument(
        "--quality16-host-measurement",
        type=Path,
        default=ROOT / ".tmp/traditional-quality16-r1/host-measurement.json",
    )
    result.add_argument(
        "--quality16-cache-manifest",
        type=Path,
        default=ROOT / ".tmp/traditional-quality16-cache-v1-r1/traditional-cache-manifest-v1.json",
    )
    result.add_argument(
        "--quality16-binding",
        type=Path,
        default=ROOT
        / ".tmp/traditional-quality16-generation-r1/stream-artifacts"
        / "accepted-inference-binding/d9"
        / "d9dfaacbc0126cf38ec61fcad22f7e63b0d9686fba9b18ee5a6431091bf37194.json",
    )
    result.add_argument(
        "--quality16-result",
        type=Path,
        default=ROOT
        / ".tmp/traditional-quality16-generation-r1/endpoint-results"
        / "9321c4af201092eda4e6a9ef36f3dc83354f0b259e8b39a5398cf25e5a72b589.json",
    )
    result.add_argument(
        "--quality16-gpu-telemetry",
        type=Path,
        default=ROOT / ".tmp/traditional-quality16-generation-r1/stream-gpu-telemetry.json",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/mage-traditional-codec-generation-qualification-2026-08-09.json",
    )
    result.add_argument("--daily-camera-hours", type=float, default=500.0)
    result.add_argument("--daily-recording-hours", type=float, default=500.0)
    result.add_argument("--recording-camera-count", type=int, default=6)
    result.add_argument("--headroom", type=float, default=1.20)
    return result


def main() -> int:
    args = parser().parse_args()
    report = build(args)
    payload = canonical_json_bytes(report) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "semantic_sha256": report["semantic_sha256"],
                "production_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
