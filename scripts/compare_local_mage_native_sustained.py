"""Build a machine-verifiable serial-versus-prefetch Mage qualification report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.mage_native_sustained import (  # noqa: E402
    MageNativeGenerationTelemetrySample,
    MageNativeRunIdentity,
    MageNativeRunMeasurement,
    MageNativeTelemetryDisposition,
    MageNativeTimeInterval,
    build_mage_native_sustained_comparison_report,
)
from robata.contracts.hashing import (  # noqa: E402
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)


class MageNativeComparisonInputError(ValueError):
    """A retained runner report or endpoint sidecar is incomplete or inconsistent."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-report", type=Path, required=True)
    parser.add_argument("--serial-telemetry", type=Path, required=True)
    parser.add_argument(
        "--serial-result-dir",
        type=Path,
        default=None,
        help="serial result artifact directory; defaults beside telemetry",
    )
    parser.add_argument("--prefetch-report", type=Path, required=True)
    parser.add_argument("--prefetch-telemetry", type=Path, required=True)
    parser.add_argument(
        "--prefetch-result-dir",
        type=Path,
        default=None,
        help="prefetch result artifact directory; defaults beside telemetry",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MageNativeComparisonInputError(f"could not read JSON document: {path}") from error
    if not isinstance(value, dict):
        raise MageNativeComparisonInputError(f"JSON document must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MageNativeComparisonInputError(f"could not read telemetry JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise MageNativeComparisonInputError(
                f"telemetry line {line_number} is not valid JSON: {path}"
            ) from error
        if not isinstance(value, dict):
            raise MageNativeComparisonInputError(
                f"telemetry line {line_number} must be an object: {path}"
            )
        rows.append(value)
    return tuple(rows)


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MageNativeComparisonInputError(f"{name} must be an object")
    return value


def _seconds_interval(
    runtime: Mapping[str, Any],
    *,
    start_field: str,
    end_field: str,
    origin: float,
) -> MageNativeTimeInterval:
    start = float(runtime[start_field]) - origin
    end = float(runtime[end_field]) - origin
    return MageNativeTimeInterval(start_seconds=max(0.0, start), end_seconds=end)


def _output_text_sha256(
    *,
    event: Mapping[str, Any],
    result_artifact_dir: Path,
) -> str:
    artifact_identity = str(event["result_artifact_identity"])
    artifact_path = result_artifact_dir.expanduser().resolve() / f"{artifact_identity}.json"
    try:
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise MageNativeComparisonInputError(
            f"could not read result artifact: {artifact_path}"
        ) from error
    if not isinstance(artifact, Mapping):
        raise MageNativeComparisonInputError(f"result artifact must be an object: {artifact_path}")
    expected_exact_sha256 = str(event["result_artifact_exact_sha256"])
    actual_exact_sha256 = exact_bytes_sha256(artifact_bytes)
    if actual_exact_sha256 != expected_exact_sha256:
        raise MageNativeComparisonInputError(f"result artifact exact SHA mismatch: {artifact_path}")
    if str(artifact.get("artifact_identity")) != artifact_identity:
        raise MageNativeComparisonInputError(f"result artifact identity mismatch: {artifact_path}")
    if str(artifact.get("request_id")) != str(event["request_id"]):
        raise MageNativeComparisonInputError(f"result artifact request mismatch: {artifact_path}")
    if int(artifact.get("output_tokens", -1)) != int(event["output_tokens"]):
        raise MageNativeComparisonInputError(
            f"result artifact output token mismatch: {artifact_path}"
        )
    output_text = artifact.get("output_text")
    if not isinstance(output_text, str) or not output_text:
        raise MageNativeComparisonInputError(
            f"result artifact output_text must be nonempty: {artifact_path}"
        )
    return exact_bytes_sha256(output_text.encode("utf-8"))


def measurement_from_files(
    *,
    report_path: Path,
    telemetry_path: Path,
    result_artifact_dir: Path | None = None,
) -> MageNativeRunMeasurement:
    """Hydrate one fresh local run from its runner report and endpoint JSONL."""

    report = _read_json(report_path)
    execution = _object(report.get("execution"), "execution")
    timing = _object(execution.get("execution_timing"), "execution.execution_timing")
    endpoint = _object(execution.get("endpoint"), "execution.endpoint")
    model_identity = _object(endpoint.get("model_identity"), "execution.endpoint.model_identity")
    plan = _object(report.get("plan"), "plan")
    recording = _object(plan.get("recording"), "plan.recording")
    recording_interval = _object(recording.get("interval"), "plan.recording.interval")
    contexts = execution.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        raise MageNativeComparisonInputError("execution.contexts must be a nonempty list")
    context_ordinals: dict[str, int] = {}
    for context in contexts:
        value = _object(context, "execution.contexts[]")
        context_ordinals[str(value["context_manifest_key"])] = int(value["focus_segment_ordinal"])

    events = _read_jsonl(telemetry_path)
    if not events:
        raise MageNativeComparisonInputError("telemetry JSONL contains no fresh generation rows")
    resolved_result_artifact_dir = (
        result_artifact_dir
        if result_artifact_dir is not None
        else telemetry_path.expanduser().resolve().parent / "endpoint-results"
    )
    runtime_rows = [
        _object(event.get("runtime_telemetry"), "runtime_telemetry") for event in events
    ]
    origin = min(float(runtime["request_started_monotonic_seconds"]) for runtime in runtime_rows)

    samples: list[MageNativeGenerationTelemetrySample] = []
    for event, runtime in zip(events, runtime_rows, strict=True):
        context_id = str(event["context_id"])
        if context_id not in context_ordinals:
            raise MageNativeComparisonInputError(
                f"telemetry context_id is absent from runner report: {context_id}"
            )
        samples.append(
            MageNativeGenerationTelemetrySample(
                telemetry_event_version=str(event["event_version"]),
                segment_ordinal=context_ordinals[context_id],
                request_id=str(event["request_id"]),
                inference_identity_sha256=str(event["inference_identity_sha256"]),
                result_artifact_identity_sha256=str(event["result_artifact_identity"]),
                output_text_sha256=_output_text_sha256(
                    event=event,
                    result_artifact_dir=resolved_result_artifact_dir,
                ),
                disposition=MageNativeTelemetryDisposition.FRESH_GENERATION,
                request_interval=_seconds_interval(
                    runtime,
                    start_field="request_started_monotonic_seconds",
                    end_field="request_completed_monotonic_seconds",
                    origin=origin,
                ),
                processor_interval=_seconds_interval(
                    runtime,
                    start_field="processor_started_monotonic_seconds",
                    end_field="processor_completed_monotonic_seconds",
                    origin=origin,
                ),
                generation_interval=_seconds_interval(
                    runtime,
                    start_field="generation_started_monotonic_seconds",
                    end_field="generation_completed_monotonic_seconds",
                    origin=origin,
                ),
                decode_interval=_seconds_interval(
                    runtime,
                    start_field="decode_started_monotonic_seconds",
                    end_field="decode_completed_monotonic_seconds",
                    origin=origin,
                ),
                prompt_tokens=int(event["prompt_tokens"]),
                output_tokens=int(event["output_tokens"]),
                max_new_tokens=int(event["max_new_tokens"]),
                output_valid=True,
                time_to_first_token_seconds=(
                    None
                    if runtime.get("time_to_first_token_seconds") is None
                    else float(runtime["time_to_first_token_seconds"])
                ),
            )
        )

    first = events[0]
    durable = _object(execution.get("durable_execution"), "execution.durable_execution")
    durable_run = _object(durable.get("run"), "execution.durable_execution.run")
    run_manifest = execution.get("run_manifest")
    run_id = (
        str(_object(run_manifest, "execution.run_manifest")["logical_key"])
        if isinstance(run_manifest, Mapping)
        else semantic_sha256(
            {
                "durable_run_key": durable_run["run_key"],
                "execution_profile": execution["execution_profile"],
                "report_path": str(report_path.expanduser().resolve()),
            }
        )
    )
    media_duration_seconds = (
        int(recording_interval["end_ns"]) - int(recording_interval["start_ns"])
    ) / 1_000_000_000
    return MageNativeRunMeasurement(
        run_id=run_id,
        execution_profile=str(execution["execution_profile"]),
        telemetry_event_version=str(first["event_version"]),
        identity=MageNativeRunIdentity(
            model_identity_sha256=str(first["model_identity_sha256"]),
            checkpoint_sha256=str(model_identity["checkpoint_manifest_sha256"]),
            source_media_sha256=str(recording["recording_exact_sha256"]),
            segment_manifest_sha256=str(plan["plan_semantic_sha256"]),
            prompt_sha256=str(first["decoder_identity_sha256"]),
            codec_policy_sha256=str(first["codec_policy_sha256"]),
            camera_id=str(report["selected_camera"]),
        ),
        expected_segment_count=len(contexts),
        media_duration_seconds=media_duration_seconds,
        wall_seconds=float(timing["run_wall_seconds"]),
        model_load_seconds=float(first.get("model_load_seconds", 0.0)),
        model_load_included_in_wall=bool(first.get("model_load_included_in_run_wall", False)),
        telemetry=tuple(samples),
    )


def build_comparison_payload(
    *,
    serial_report: Path,
    serial_telemetry: Path,
    prefetch_report: Path,
    prefetch_telemetry: Path,
    serial_result_dir: Path | None = None,
    prefetch_result_dir: Path | None = None,
) -> dict[str, object]:
    serial = measurement_from_files(
        report_path=serial_report,
        telemetry_path=serial_telemetry,
        result_artifact_dir=serial_result_dir,
    )
    prefetch = measurement_from_files(
        report_path=prefetch_report,
        telemetry_path=prefetch_telemetry,
        result_artifact_dir=prefetch_result_dir,
    )
    comparison = build_mage_native_sustained_comparison_report(
        serial=serial,
        prefetch=prefetch,
    )
    serial_document = _read_json(serial_report)
    prefetch_document = _read_json(prefetch_report)
    payload: dict[str, object] = {
        "format_version": "local-mage-native-sustained-qualification-v1",
        "evidence_class": "LOCAL_CONFORMANCE",
        "production_eligible": False,
        "serial_measurement": serial.model_dump(mode="json"),
        "prefetch_measurement": prefetch.model_dump(mode="json"),
        "comparison": comparison.as_dict(),
        "gpu_telemetry": {
            "serial": _object(serial_document["execution"], "execution").get("gpu_telemetry"),
            "prefetch": _object(prefetch_document["execution"], "execution").get("gpu_telemetry"),
        },
    }
    payload["semantic_sha256"] = semantic_sha256(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = build_comparison_payload(
            serial_report=arguments.serial_report,
            serial_telemetry=arguments.serial_telemetry,
            prefetch_report=arguments.prefetch_report,
            prefetch_telemetry=arguments.prefetch_telemetry,
            serial_result_dir=arguments.serial_result_dir,
            prefetch_result_dir=arguments.prefetch_result_dir,
        )
        destination = arguments.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical_json_bytes(payload) + b"\n")
    except (MageNativeComparisonInputError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "detail": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
