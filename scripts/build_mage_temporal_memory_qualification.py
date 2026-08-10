"""Build the same-sample Mage temporal-memory A/B qualification report.

The report is local, single-camera RTX 4060 evidence. It separates endpoint lifecycle
from recurring stream service and rejects speedups that change the observed action
sequence. It never promotes the candidate to production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.contracts.hashing import canonical_json_bytes, semantic_sha256  # noqa: E402

REPORT_VERSION = "mage-temporal-memory-ab-qualification-v1"
MEDIA_SECONDS = 40.0
CAMERA_TARGET_RTF = 25.0
SIX_CAMERA_INDEPENDENT_TARGET_RTF = 150.0
EXPECTED_TEMPORAL_ACTION = "a person picks up a green garment from a table"


class TemporalMemoryEvidenceError(ValueError):
    """The A/B evidence is missing, inconsistent, or overclaims its authority."""


def _load(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.resolve().read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise TemporalMemoryEvidenceError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TemporalMemoryEvidenceError(f"JSON root must be an object: {path}")
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


def _argument_value(arguments: list[object], flag: str) -> str:
    try:
        index = arguments.index(flag)
    except ValueError as error:
        raise TemporalMemoryEvidenceError(f"missing argument {flag}") from error
    if index + 1 >= len(arguments) or not isinstance(arguments[index + 1], str):
        raise TemporalMemoryEvidenceError(f"argument {flag} lacks a string value")
    return arguments[index + 1]


def _bindings(directory: Path) -> list[dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for path in directory.rglob("*.json"):
        raw, binding = _load(path)
        context = binding.get("context")
        response = binding.get("endpoint_response")
        if not isinstance(context, dict) or not isinstance(response, dict):
            raise TemporalMemoryEvidenceError("accepted binding lacks context or response")
        ordinal = context.get("focus_segment_ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal < 5:
            raise TemporalMemoryEvidenceError("accepted binding ordinal must be 0..4")
        if ordinal in rows:
            raise TemporalMemoryEvidenceError("accepted binding ordinals must be unique")
        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise TemporalMemoryEvidenceError("accepted binding output_text must be a string")
        try:
            output = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise TemporalMemoryEvidenceError("accepted binding output is not JSON") from error
        observations = output.get("observations") if isinstance(output, dict) else None
        if not isinstance(observations, list):
            raise TemporalMemoryEvidenceError("accepted binding lacks observations")
        actions = []
        for observation in observations:
            action = observation.get("action") if isinstance(observation, dict) else None
            if not isinstance(action, str):
                raise TemporalMemoryEvidenceError("observation action must be a string")
            actions.append(action)
        rows[ordinal] = {
            "ordinal": ordinal,
            "binding": _reference(path, raw),
            "request_identity_sha256": binding["request_identity_sha256"],
            "inference_identity": binding["inference_identity"],
            "result_artifact_exact_sha256": binding["result_artifact_exact_sha256"],
            "generation_seconds": response["generation_seconds"],
            "load_seconds": response["load_seconds"],
            "prompt_tokens": response["prompt_tokens"],
            "output_tokens": response["output_tokens"],
            "output_text": output_text,
            "actions": actions,
        }
    if set(rows) != set(range(5)):
        raise TemporalMemoryEvidenceError("exactly five accepted bindings are required")
    return [rows[index] for index in range(5)]


def _gpu(execution: dict[str, Any]) -> dict[str, object]:
    telemetry = execution["gpu_telemetry"]
    summaries = telemetry["device_summaries"]
    if telemetry["measurement_status"] != "MEASURED" or len(summaries) != 1:
        raise TemporalMemoryEvidenceError("one measured GPU summary is required")
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


def _artifact_count(root: Path, kind: str) -> int:
    directory = root / "stream-artifacts" / "_logical" / kind
    return len(tuple(directory.glob("*.ref"))) if directory.is_dir() else 0


def _verify_cas(root: Path) -> int:
    checked = 0
    artifact_root = root / "stream-artifacts"
    for path in artifact_root.rglob("*.json"):
        if "_logical" in path.parts:
            continue
        digest = path.stem
        if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise TemporalMemoryEvidenceError(f"CAS filename/content mismatch: {path}")
            checked += 1
    return checked


def _scheduler(root: Path) -> dict[str, object]:
    path = root / "stream-scheduler.sqlite3"
    connection = sqlite3.connect(path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        states = connection.execute(
            "SELECT stage, state, COUNT(*) FROM perception_vnext_work_items "
            "GROUP BY stage, state ORDER BY stage, state"
        ).fetchall()
    finally:
        connection.close()
    if integrity != ("ok",) or foreign_keys:
        raise TemporalMemoryEvidenceError("scheduler SQLite integrity failed")
    return {
        "database": _reference(path, path.read_bytes()),
        "integrity_check": "ok",
        "foreign_key_violation_count": 0,
        "stage_states": [
            {"stage": stage, "state": state, "count": count} for stage, state, count in states
        ],
    }


def _run(root: Path, *, expected_profile: str) -> dict[str, object]:
    lifecycle_raw, lifecycle = _load(root / "lifecycle-report.json")
    stream_raw, stream = _load(root / "stream-report.json")
    gpu_raw, _ = _load(root / "stream-gpu-telemetry.json")
    endpoint_args_raw = (root / "endpoint-args.json").read_bytes()
    endpoint_args = json.loads(endpoint_args_raw)
    if not isinstance(endpoint_args, list):
        raise TemporalMemoryEvidenceError("endpoint args must be an array")
    execution = stream["execution"]
    if lifecycle.get("outcome") != "SUCCEEDED" or lifecycle.get("exit_code") != 0:
        raise TemporalMemoryEvidenceError("bounded lifecycle did not succeed")
    ownership = lifecycle["process_ownership"]
    if (
        not ownership["post_shutdown_port_closed"]
        or lifecycle["endpoint"]["cleanup"]["reaped"] is not True
    ):
        raise TemporalMemoryEvidenceError("bounded lifecycle did not prove endpoint cleanup")
    if execution["decoder_output_profile"] != expected_profile:
        raise TemporalMemoryEvidenceError("decoder profile differs")
    if execution["normal_model_call_count"] != 5 or execution["refinement_model_call_count"] != 0:
        raise TemporalMemoryEvidenceError(
            "A/B requires exactly five normal calls and zero refine calls"
        )
    bindings = _bindings(root / "stream-artifacts" / "accepted-inference-binding")
    generations = [float(item["generation_seconds"]) for item in bindings]
    warm = generations[1:]
    output_sequence = [str(item["output_text"]) for item in bindings]
    ready_event = next(
        event
        for event in lifecycle["events"]
        if event["phase"] == "ENDPOINT" and event["state"] == "READY"
    )
    cache_path = Path(_argument_value(endpoint_args, "--codec-cache-manifest"))
    cache_raw, cache = _load(cache_path)
    if execution["execution_timing"]["media_duration_seconds"] != MEDIA_SECONDS:
        raise TemporalMemoryEvidenceError("stream media duration differs from 40 seconds")
    return {
        "root": str(root.resolve()),
        "decoder_output_profile": expected_profile,
        "references": {
            "lifecycle": _reference(root / "lifecycle-report.json", lifecycle_raw),
            "stream": _reference(root / "stream-report.json", stream_raw),
            "gpu": _reference(root / "stream-gpu-telemetry.json", gpu_raw),
            "endpoint_args": _reference(root / "endpoint-args.json", endpoint_args_raw),
        },
        "shared_identity": {
            "source_path": stream["source_path"],
            "source_byte_count": stream["source_byte_count"],
            "plan_semantic_sha256": stream["plan"]["plan_semantic_sha256"],
            "selected_camera": stream["selected_camera"],
            "model_identity": lifecycle["endpoint"]["health_response"]["model_identity"],
            "codec_cache": {
                **_reference(cache_path, cache_raw),
                "manifest_semantic_sha256": cache["manifest_semantic_sha256"],
            },
        },
        "lifecycle": {
            "wall_seconds": lifecycle["wall_seconds"],
            "endpoint_ready_seconds": ready_event["elapsed_seconds"],
            "model_load_seconds": bindings[0]["load_seconds"],
            "post_shutdown_port_closed": True,
            "endpoint_reaped": True,
        },
        "recurring": {
            "stream_wall_seconds": execution["execution_timing"]["run_wall_seconds"],
            "end_to_end_realtime_factor": execution["execution_timing"][
                "end_to_end_realtime_factor"
            ],
            "generation_seconds": generations,
            "generation_sum_seconds": sum(generations),
            "warm_generation_mean_seconds": mean(warm),
            "warm_generation_median_seconds": median(warm),
            "warm_generation_p95_seconds": quantiles(warm, n=20, method="inclusive")[18],
            "prompt_tokens": [item["prompt_tokens"] for item in bindings],
            "prompt_tokens_total": sum(int(item["prompt_tokens"]) for item in bindings),
            "output_tokens": [item["output_tokens"] for item in bindings],
            "output_tokens_total": sum(int(item["output_tokens"]) for item in bindings),
            "output_text_sequence_sha256": hashlib.sha256(
                canonical_json_bytes(output_sequence)
            ).hexdigest(),
            "bindings": bindings,
        },
        "gpu": _gpu(execution),
        "temporal_memory": execution["temporal_memory"],
        "durability": {
            "memory_artifact_count": _artifact_count(
                root, "mage-video-temporal-observation-memory"
            ),
            "memory_link_artifact_count": _artifact_count(root, "mage-video-temporal-memory-link"),
            "observation_artifact_count": _artifact_count(root, "observation"),
            "accepted_binding_count": _artifact_count(root, "accepted-inference-binding"),
            "cas_objects_verified": _verify_cas(root),
            "scheduler": _scheduler(root),
        },
        "event_tracks": execution["event_tracks"],
        "refine_requests": execution["refine_requests"],
    }


def build(*, control_root: Path, candidate_root: Path) -> dict[str, object]:
    control = _run(control_root.resolve(), expected_profile="full-v1")
    candidate = _run(candidate_root.resolve(), expected_profile="temporal-memory-v1")
    if control["shared_identity"] != candidate["shared_identity"]:
        raise TemporalMemoryEvidenceError("control and candidate identities differ")
    temporal = candidate["temporal_memory"]
    if temporal.get("enabled") is not True or temporal.get("persisted_memory_count") != 5:
        raise TemporalMemoryEvidenceError("candidate temporal memory execution is incomplete")
    if (
        candidate["durability"]["memory_artifact_count"] != 5
        or candidate["durability"]["memory_link_artifact_count"] != 5
    ):
        raise TemporalMemoryEvidenceError("candidate lacks five durable memory/link artifacts")
    candidate_actions = [
        action for binding in candidate["recurring"]["bindings"] for action in binding["actions"]
    ]
    if candidate_actions != [EXPECTED_TEMPORAL_ACTION] * 5:
        raise TemporalMemoryEvidenceError(
            "candidate output no longer matches inspected anchoring result"
        )
    control_rtf = control["recurring"]["end_to_end_realtime_factor"]
    candidate_rtf = candidate["recurring"]["end_to_end_realtime_factor"]
    comparison = {
        "stream_wall_reduction_percent": 100.0
        * (
            control["recurring"]["stream_wall_seconds"]
            - candidate["recurring"]["stream_wall_seconds"]
        )
        / control["recurring"]["stream_wall_seconds"],
        "generation_sum_reduction_percent": 100.0
        * (
            control["recurring"]["generation_sum_seconds"]
            - candidate["recurring"]["generation_sum_seconds"]
        )
        / control["recurring"]["generation_sum_seconds"],
        "warm_generation_mean_reduction_percent": 100.0
        * (
            control["recurring"]["warm_generation_mean_seconds"]
            - candidate["recurring"]["warm_generation_mean_seconds"]
        )
        / control["recurring"]["warm_generation_mean_seconds"],
        "realtime_factor_gain_percent": 100.0 * (candidate_rtf / control_rtf - 1.0),
        "prompt_token_increase_percent": 100.0
        * (
            candidate["recurring"]["prompt_tokens_total"]
            / control["recurring"]["prompt_tokens_total"]
            - 1.0
        ),
        "output_token_reduction_percent": 100.0
        * (
            1.0
            - candidate["recurring"]["output_tokens_total"]
            / control["recurring"]["output_tokens_total"]
        ),
        "peak_vram_increase_mib": (
            candidate["gpu"]["memory_used_mib_max"] - control["gpu"]["memory_used_mib_max"]
        ),
        "control_local_lanes_for_25x": math.ceil(CAMERA_TARGET_RTF / control_rtf),
        "candidate_unaccepted_lanes_for_25x": math.ceil(CAMERA_TARGET_RTF / candidate_rtf),
        "control_local_lanes_for_150x_if_six_independent_cameras": math.ceil(
            SIX_CAMERA_INDEPENDENT_TARGET_RTF / control_rtf
        ),
        "candidate_unaccepted_lanes_for_150x_if_six_independent_cameras": math.ceil(
            SIX_CAMERA_INDEPENDENT_TARGET_RTF / candidate_rtf
        ),
    }
    report: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "hardware_scope": "NVIDIA GeForce RTX 4060 Laptop GPU; one camera; one generation lane",
        "quality_basis": "AGENT_VISUAL_INSPECTION_SINGLE_RECORDING_NOT_LABELED_GROUND_TRUTH",
        "shared_identity": control["shared_identity"],
        "control": control,
        "candidate": candidate,
        "comparison": comparison,
        "quality": {
            "control_findings": [
                (
                    "The same green garment drifts across shirt, drawstring bag, "
                    "pant, and cloth labels."
                ),
                "The control retains distinct pick-up, fold, hold, and fold-like phases.",
            ],
            "candidate_findings": [
                "All five segments collapse to the identical pick-up-a-green-garment action.",
                "Four later segments repeat the predecessor label despite distinct control phases.",
                "The first segment also loses the control's separate folding action.",
                "The speed gain is therefore coupled to semantic collapse and fewer output tokens.",
            ],
            "candidate_action_repeat_count": 5,
            "candidate_distinct_action_count": 1,
            "disposition": "REJECT_PROMPT_ANCHORING_AND_ACTION_COLLAPSE",
        },
        "decision": {
            "temporal_memory_v1": "REJECT",
            "mage_route": "HOLD",
            "qwen_batch_hedge": "ACTIVATE",
            "reason": (
                "Temporal memory improves local wall time only by collapsing the action sequence; "
                "it fails semantic non-inferiority and cannot contribute accepted capacity."
            ),
            "rollback": "Use full-v1 8x131K control; do not select temporal-memory-v1.",
        },
    }
    report["semantic_sha256"] = semantic_sha256(report)
    validate(report)
    return report


def validate(report: dict[str, Any]) -> None:
    if report.get("report_version") != REPORT_VERSION:
        raise TemporalMemoryEvidenceError("report version differs")
    if (
        report.get("authority") != "LOCAL_NONPRODUCTION_ONLY"
        or report.get("production_eligible") is not False
    ):
        raise TemporalMemoryEvidenceError("report must remain local and non-production")
    if report["decision"]["temporal_memory_v1"] != "REJECT":
        raise TemporalMemoryEvidenceError("temporal memory candidate must remain rejected")
    if report["quality"]["disposition"] != "REJECT_PROMPT_ANCHORING_AND_ACTION_COLLAPSE":
        raise TemporalMemoryEvidenceError("quality disposition differs")
    if report["candidate"]["temporal_memory"]["persisted_memory_count"] != 5:
        raise TemporalMemoryEvidenceError("candidate memory count differs")
    if report["comparison"]["candidate_unaccepted_lanes_for_25x"] != math.ceil(
        CAMERA_TARGET_RTF / report["candidate"]["recurring"]["end_to_end_realtime_factor"]
    ):
        raise TemporalMemoryEvidenceError("candidate lane formula differs")
    semantic = report.get("semantic_sha256")
    projection = dict(report)
    projection.pop("semantic_sha256", None)
    if semantic != semantic_sha256(projection):
        raise TemporalMemoryEvidenceError("report semantic hash differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build(
        control_root=arguments.control_root,
        candidate_root=arguments.candidate_root,
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "exact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "semantic_sha256": report["semantic_sha256"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
