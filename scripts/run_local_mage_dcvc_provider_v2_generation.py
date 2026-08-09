"""Run one isolated Provider V2 Mage generation qualification lane.

The harness owns only the endpoint/decoder phase. Provider V2 assets must already
exist in a verified cache manifest produced by the persistent prewarm worker. It
starts one Mage endpoint, waits for health, executes the exact local single-camera
40-second stream with one observation in flight, captures full-wall GPU telemetry,
and stops the endpoint. It never starts DCVC preparation, so the local RTX lane
cannot overlap codec preparation with generation.

The generated plan/result is local qualification evidence and is never production
eligible. ``sequence_length_frames`` is recorded as an identity field only; this
harness rejects any claim that it caps DCVC recurrent work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.gpu_telemetry import NvidiaSmiGpuSampler  # noqa: E402

HARNESS_VERSION = "mage-dcvc-provider-v2-generation-harness-v1"
QUALIFIED_MANIFEST_VERSION = "mage-dcvc-qualified-provider-manifest-v2"
CACHE_MANIFEST_VERSION = "mage-codec-cache-manifest-v2"
RECIPE_VERSION = "mage-dcvc-readiness-explicit-v2"
DEVICE_POLICY = "exclusive-shared-device-v1"
MEDIA_DURATION_NS = 40_000_000_000
SEGMENT_COUNT = 5


class ProviderV2GenerationHarnessError(RuntimeError):
    """The requested Provider V2 generation lane is unsafe or failed."""


@dataclass(frozen=True, slots=True)
class ProviderV2GenerationInputs:
    model_directory: Path
    qualified_provider_manifest_path: Path
    checkpoint_manifest_path: Path
    cache_manifest_path: Path
    shared_device_guard_file: Path
    materialization_directory: Path
    source_path: Path
    model_identifier: str
    model_revision: str
    checkpoint_manifest_sha256: str
    cache_manifest_semantic_sha256: str
    cache_namespace_identity: str
    provider_implementation_sha256: str
    effective_config_sha256: str
    effective_config: Mapping[str, Any]
    admitted_segment_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ProviderV2GenerationPaths:
    root: Path
    state_directory: Path
    result_artifact_directory: Path
    stream_artifact_directory: Path
    scheduler_database: Path
    endpoint_stdout: Path
    endpoint_stderr: Path
    stream_stdout: Path
    stream_stderr: Path
    endpoint_generation_telemetry: Path
    stream_gpu_telemetry: Path
    full_wall_gpu_telemetry: Path
    stream_report: Path
    harness_result: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--qualified-provider-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest-path", required=True, type=Path)
    parser.add_argument("--codec-cache-manifest", required=True, type=Path)
    parser.add_argument("--shared-device-guard-file", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--materialization-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--ffprobe-binary", default="ffprobe")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8102)
    parser.add_argument(
        "--recording-key",
        required=True,
        help="recording identity must reproduce the exact prewarmed segment paths",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=256)
    parser.add_argument("--codec-timeout-seconds", type=_positive_int, default=7_200)
    parser.add_argument("--endpoint-timeout-seconds", type=_positive_float, default=7_260.0)
    parser.add_argument("--startup-timeout-seconds", type=_positive_float, default=300.0)
    parser.add_argument("--gpu-sample-interval-seconds", type=_positive_float, default=0.5)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate identities and write commands without starting GPU work",
    )
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not parsed > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65_535:
        raise argparse.ArgumentTypeError("must be a TCP port")
    return parsed


def _read_object(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProviderV2GenerationHarnessError(f"could not read {label}: {resolved}") from error
    if not isinstance(value, dict):
        raise ProviderV2GenerationHarnessError(f"{label} must contain a JSON object")
    return value


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderV2GenerationHarnessError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ProviderV2GenerationHarnessError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProviderV2GenerationHarnessError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProviderV2GenerationHarnessError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ProviderV2GenerationHarnessError(f"{label} must be a lowercase SHA-256")
    return digest


def _resolve_existing_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ProviderV2GenerationHarnessError(f"{label} is unavailable: {path}") from error
    if not resolved.is_file():
        raise ProviderV2GenerationHarnessError(f"{label} is not a file: {resolved}")
    return resolved


def _resolve_existing_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ProviderV2GenerationHarnessError(f"{label} is unavailable: {path}") from error
    if not resolved.is_dir():
        raise ProviderV2GenerationHarnessError(f"{label} is not a directory: {resolved}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_provider_v2_generation_inputs(arguments: argparse.Namespace) -> ProviderV2GenerationInputs:
    model_directory = _resolve_existing_directory(arguments.model_dir, "qualified model directory")
    qualified_path = _resolve_existing_file(
        arguments.qualified_provider_manifest, "qualified provider manifest"
    )
    checkpoint_path = _resolve_existing_file(
        arguments.checkpoint_manifest_path, "checkpoint manifest"
    )
    cache_path = _resolve_existing_file(arguments.codec_cache_manifest, "codec cache manifest")
    source_path = _resolve_existing_file(arguments.source, "source recording")
    materialization = _resolve_existing_directory(
        arguments.materialization_dir, "materialization directory"
    )
    guard = arguments.shared_device_guard_file.expanduser().resolve()
    guard.parent.mkdir(parents=True, exist_ok=True)

    qualified = _read_object(qualified_path, "qualified provider manifest")
    checkpoint = _read_object(checkpoint_path, "checkpoint manifest")
    cache = _read_object(cache_path, "codec cache manifest")

    if qualified.get("manifest_version") != QUALIFIED_MANIFEST_VERSION:
        raise ProviderV2GenerationHarnessError("qualified provider manifest version is unsupported")
    if cache.get("manifest_version") != CACHE_MANIFEST_VERSION:
        raise ProviderV2GenerationHarnessError("cache manifest is not Provider V2")
    if cache.get("recipe_version") != RECIPE_VERSION:
        raise ProviderV2GenerationHarnessError("cache manifest does not use the explicit V2 recipe")

    declared_model = Path(_text(qualified.get("qualified_model_directory"), "qualified model path"))
    if declared_model.expanduser().resolve() != model_directory:
        raise ProviderV2GenerationHarnessError("qualified manifest names another model directory")

    bundle = _object(qualified.get("bundle"), "qualified bundle")
    qualified_checkpoint = _object(
        qualified.get("qualified_checkpoint_manifest"), "qualified checkpoint manifest"
    )
    model_identifier = _text(bundle.get("qualified_model_identifier"), "qualified model identifier")
    model_revision = _text(bundle.get("qualified_model_revision"), "qualified model revision")
    checkpoint_sha256 = _sha256(
        qualified_checkpoint.get("manifest_sha256"), "qualified checkpoint SHA-256"
    )
    observed_checkpoint_sha256 = _sha256(
        checkpoint.get("manifest_sha256"), "checkpoint manifest SHA-256"
    )
    if observed_checkpoint_sha256 != checkpoint_sha256:
        raise ProviderV2GenerationHarnessError("checkpoint file does not match qualified manifest")
    if (
        checkpoint.get("model_identifier") != model_identifier
        or checkpoint.get("model_revision") != model_revision
    ):
        raise ProviderV2GenerationHarnessError("checkpoint model identity does not match bundle")
    cache_checkpoint_sha256 = _sha256(
        cache.get("checkpoint_manifest_sha256"), "cache checkpoint SHA-256"
    )
    if cache_checkpoint_sha256 != checkpoint_sha256:
        raise ProviderV2GenerationHarnessError("cache belongs to another checkpoint")

    effective = _object(cache.get("effective_config"), "effective config")
    if effective.get("provider_version") != "robata-mage-dcvc-provider-v2":
        raise ProviderV2GenerationHarnessError("effective config is not Provider V2")
    if effective.get("recipe_version") != RECIPE_VERSION:
        raise ProviderV2GenerationHarnessError("effective config recipe is not explicit V2")
    if effective.get("device_concurrency_policy") != DEVICE_POLICY:
        raise ProviderV2GenerationHarnessError(
            "local generation requires exclusive shared-device policy"
        )
    if _integer(effective.get("sequence_length_frames"), "sequence_length_frames") != 0:
        raise ProviderV2GenerationHarnessError(
            "Provider V2 qualification requires sequence_length_frames=0; it is not a compute cap"
        )
    if effective.get("canvas_token_side") is not None:
        raise ProviderV2GenerationHarnessError(
            "Provider V2 qualification requires canvas_token_side=null"
        )
    if effective.get("encoded_frame_extent") != "through-last-sampled-frame":
        raise ProviderV2GenerationHarnessError(
            "effective config understates recurrent frame extent"
        )
    if effective.get("preparation_device") != "cuda":
        raise ProviderV2GenerationHarnessError("local Provider V2 generation expects CUDA assets")

    provider_sha256 = _sha256(
        cache.get("provider_implementation_sha256"), "provider implementation SHA-256"
    )
    if (
        _sha256(
            effective.get("provider_implementation_sha256"),
            "effective provider implementation SHA-256",
        )
        != provider_sha256
    ):
        raise ProviderV2GenerationHarnessError("effective config/provider identity mismatch")
    effective_sha256 = _sha256(effective.get("effective_config_sha256"), "effective config SHA-256")
    namespace_identity = _sha256(cache.get("namespace_identity"), "cache namespace identity")
    cache_semantic = _sha256(
        cache.get("manifest_semantic_sha256"), "cache manifest semantic SHA-256"
    )

    entries = _array(cache.get("entries"), "cache entries")
    if len(entries) != SEGMENT_COUNT or cache.get("entry_count") != SEGMENT_COUNT:
        raise ProviderV2GenerationHarnessError(
            "generation qualification requires exactly five entries"
        )
    admitted: list[Path] = []
    for ordinal, raw in enumerate(entries):
        entry = _object(raw, f"cache entry {ordinal}")
        source = _resolve_existing_file(
            Path(_text(entry.get("source_path"), f"cache entry {ordinal} source")),
            f"cache entry {ordinal} source",
        )
        if not _is_within(source, materialization):
            raise ProviderV2GenerationHarnessError(
                f"cache entry {ordinal} source is outside materialization directory"
            )
        admitted.append(source)

    return ProviderV2GenerationInputs(
        model_directory=model_directory,
        qualified_provider_manifest_path=qualified_path,
        checkpoint_manifest_path=checkpoint_path,
        cache_manifest_path=cache_path,
        shared_device_guard_file=guard,
        materialization_directory=materialization,
        source_path=source_path,
        model_identifier=model_identifier,
        model_revision=model_revision,
        checkpoint_manifest_sha256=checkpoint_sha256,
        cache_manifest_semantic_sha256=cache_semantic,
        cache_namespace_identity=namespace_identity,
        provider_implementation_sha256=provider_sha256,
        effective_config_sha256=effective_sha256,
        effective_config=effective,
        admitted_segment_paths=tuple(admitted),
    )


def generation_paths(output_root: Path) -> ProviderV2GenerationPaths:
    root = output_root.expanduser().resolve()
    return ProviderV2GenerationPaths(
        root=root,
        state_directory=root / "endpoint-state",
        result_artifact_directory=root / "endpoint-results",
        stream_artifact_directory=root / "stream-artifacts",
        scheduler_database=root / "stream-scheduler.sqlite3",
        endpoint_stdout=root / "endpoint.stdout.log",
        endpoint_stderr=root / "endpoint.stderr.log",
        stream_stdout=root / "stream.stdout.log",
        stream_stderr=root / "stream.stderr.log",
        endpoint_generation_telemetry=root / "endpoint-generation-telemetry.jsonl",
        stream_gpu_telemetry=root / "stream-gpu-telemetry.json",
        full_wall_gpu_telemetry=root / "full-wall-gpu-telemetry.json",
        stream_report=root / "stream-report.json",
        harness_result=root / "harness-result.json",
    )


def _config_cli(effective: Mapping[str, Any], *, timeout_seconds: int) -> list[str]:
    values = [
        "--codec-mode",
        "neural",
        "--codec-target-canvas",
        str(_integer(effective.get("target_canvas"), "target_canvas", minimum=1)),
        "--codec-group-size",
        str(_integer(effective.get("group_size"), "group_size", minimum=1)),
        "--codec-images-per-group",
        str(_integer(effective.get("images_per_group"), "images_per_group", minimum=1)),
        "--codec-patch-size",
        str(_integer(effective.get("patch"), "patch", minimum=1)),
        "--codec-max-pixels",
        str(_integer(effective.get("max_pixels"), "max_pixels", minimum=1)),
        "--codec-min-group-frames",
        str(_integer(effective.get("min_group_frames"), "min_group_frames", minimum=1)),
        "--codec-max-group-frames",
        str(_integer(effective.get("max_group_frames"), "max_group_frames", minimum=1)),
        "--codec-timeout-seconds",
        str(timeout_seconds),
        "--preprocess-device",
        _text(effective.get("preparation_device"), "preparation_device"),
        "--neural-qp",
        str(_integer(effective.get("qp"), "qp")),
        "--neural-reset-interval",
        str(_integer(effective.get("reset_interval"), "reset_interval", minimum=1)),
        "--neural-intra-period",
        str(int(effective.get("intra_period"))),
        "--neural-max-side",
        str(_integer(effective.get("max_side"), "max_side")),
        "--neural-sequence-length-frames",
        "0",
        "--neural-readiness-coverage-bins",
        str(
            _integer(
                effective.get("readiness_coverage_bins"),
                "readiness_coverage_bins",
                minimum=1,
            )
        ),
        "--neural-readiness-delta-ratio",
        str(float(effective.get("readiness_delta_ratio"))),
        "--neural-bitcost-percentile",
        str(_integer(effective.get("bitcost_percentile"), "bitcost_percentile", minimum=1)),
        "--neural-decode-backsearch-max",
        str(
            _integer(
                effective.get("decode_backsearch_max"),
                "decode_backsearch_max",
                minimum=1,
            )
        ),
    ]
    return values


def build_endpoint_command(
    *,
    arguments: argparse.Namespace,
    inputs: ProviderV2GenerationInputs,
    paths: ProviderV2GenerationPaths,
) -> list[str]:
    return [
        str(arguments.python.expanduser().resolve()),
        "-u",
        str(REPOSITORY_ROOT / "scripts" / "run_mage_video_endpoint.py"),
        "--model-dir",
        str(inputs.model_directory),
        "--model-identifier",
        inputs.model_identifier,
        "--model-revision",
        inputs.model_revision,
        "--checkpoint-manifest-path",
        str(inputs.checkpoint_manifest_path),
        "--checkpoint-manifest-sha256",
        inputs.checkpoint_manifest_sha256,
        "--load-profile",
        "local-4bit-nf4",
        *_config_cli(inputs.effective_config, timeout_seconds=arguments.codec_timeout_seconds),
        "--state-dir",
        str(paths.state_directory),
        "--result-artifact-dir",
        str(paths.result_artifact_directory),
        "--generation-telemetry-jsonl",
        str(paths.endpoint_generation_telemetry),
        "--codec-cache-manifest",
        str(inputs.cache_manifest_path),
        "--qualified-provider-manifest",
        str(inputs.qualified_provider_manifest_path),
        "--shared-device-guard-file",
        str(inputs.shared_device_guard_file),
        "--require-verified-codec-cache",
        "--require-provider-v2-cache",
        "--durable-input-root",
        str(inputs.materialization_directory),
        "--host",
        arguments.host,
        "--port",
        str(arguments.port),
        "--log-level",
        "warning",
    ]


def build_stream_command(
    *,
    arguments: argparse.Namespace,
    inputs: ProviderV2GenerationInputs,
    paths: ProviderV2GenerationPaths,
) -> list[str]:
    return [
        str(arguments.python.expanduser().resolve()),
        str(REPOSITORY_ROOT / "scripts" / "run_local_mage_stream.py"),
        str(inputs.source_path),
        "--recording-key",
        arguments.recording_key,
        "--recording-start-ns",
        "0",
        "--recording-end-ns",
        str(MEDIA_DURATION_NS),
        "--scan-segment-seconds",
        "8",
        "--reasoning-horizon-seconds",
        "8",
        "--segment-boundary-mode",
        "keyframe_aligned",
        "--camera",
        "cam_01",
        "--execute",
        "--materialization-dir",
        str(inputs.materialization_directory),
        "--ffmpeg-binary",
        str(arguments.ffmpeg_binary),
        "--ffprobe-binary",
        str(arguments.ffprobe_binary),
        "--artifact-dir",
        str(paths.stream_artifact_directory),
        "--scheduler-db",
        str(paths.scheduler_database),
        "--endpoint-url",
        f"http://{arguments.host}:{arguments.port}",
        "--endpoint-timeout-seconds",
        str(arguments.endpoint_timeout_seconds),
        "--gpu-sample-interval-seconds",
        str(arguments.gpu_sample_interval_seconds),
        "--gpu-telemetry-output",
        str(paths.stream_gpu_telemetry),
        "--max-new-tokens",
        str(arguments.max_new_tokens),
        "--max-inflight-observations",
        "1",
        *_config_cli(inputs.effective_config, timeout_seconds=arguments.codec_timeout_seconds),
        "--output",
        str(paths.stream_report),
    ]


def build_plan_document(
    *,
    arguments: argparse.Namespace,
    inputs: ProviderV2GenerationInputs,
    paths: ProviderV2GenerationPaths,
) -> dict[str, object]:
    return {
        "harness_version": HARNESS_VERSION,
        "authority": "LOCAL_QUALIFICATION_NON_PRODUCTION",
        "production_eligible": False,
        "execution": {
            "camera_count": 1,
            "generation_inflight_limit": 1,
            "provider_preparation_started_by_harness": False,
            "codec_generation_overlap_allowed": False,
            "shared_device_guard_file": str(inputs.shared_device_guard_file),
        },
        "recurrent_work": {
            "semantics": "FULL_RECURRENT_CHAIN_THROUGH_LAST_SAMPLED_FRAME",
            "sequence_length_frames": 0,
            "sequence_length_is_compute_cap": False,
        },
        "identities": {
            "model_identifier": inputs.model_identifier,
            "model_revision": inputs.model_revision,
            "checkpoint_manifest_sha256": inputs.checkpoint_manifest_sha256,
            "provider_implementation_sha256": inputs.provider_implementation_sha256,
            "effective_config_sha256": inputs.effective_config_sha256,
            "cache_namespace_identity": inputs.cache_namespace_identity,
            "cache_manifest_semantic_sha256": inputs.cache_manifest_semantic_sha256,
        },
        "sample": {
            "source_path": str(inputs.source_path),
            "duration_ns": MEDIA_DURATION_NS,
            "segment_count": SEGMENT_COUNT,
            "admitted_segment_paths": [str(item) for item in inputs.admitted_segment_paths],
        },
        "paths": {
            "output_root": str(paths.root),
            "stream_report": str(paths.stream_report),
            "endpoint_generation_telemetry": str(paths.endpoint_generation_telemetry),
            "full_wall_gpu_telemetry": str(paths.full_wall_gpu_telemetry),
            "hf_home": str(paths.root / "hf-home"),
            "hf_modules_cache": str(paths.root / "hf-modules"),
        },
        "commands": {
            "endpoint": build_endpoint_command(arguments=arguments, inputs=inputs, paths=paths),
            "stream": build_stream_command(arguments=arguments, inputs=inputs, paths=paths),
        },
    }


def _lossless_json_bytes(payload: object) -> bytes:
    """Encode operational telemetry without coercing nanosecond integers to floats."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_document(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(_lossless_json_bytes(payload))
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ProviderV2GenerationHarnessError(f"could not write {path}") from error


def _prepare_actual_root(paths: ProviderV2GenerationPaths) -> None:
    if paths.root.exists():
        try:
            occupied = next(paths.root.iterdir(), None)
        except OSError as error:
            raise ProviderV2GenerationHarnessError("could not inspect output root") from error
        if occupied is not None:
            raise ProviderV2GenerationHarnessError(
                "output root must be absent or empty so replay cannot contaminate timing"
            )
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.state_directory.mkdir(parents=True, exist_ok=False)
    paths.result_artifact_directory.mkdir(parents=True, exist_ok=False)
    paths.stream_artifact_directory.mkdir(parents=True, exist_ok=False)


def _health_document(url: str, timeout_seconds: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as error:
        raise ProviderV2GenerationHarnessError("endpoint health request failed") from error
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProviderV2GenerationHarnessError("endpoint health response is not JSON") from error
    return _require_ready_health(value)


def _require_ready_health(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or str(value.get("status", "")).strip().upper() != "READY":
        raise ProviderV2GenerationHarnessError("endpoint health response is not READY")
    return value


def _wait_for_health(
    *,
    process: subprocess.Popen[bytes],
    url: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    last_error: BaseException | None = None
    while time.perf_counter() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise ProviderV2GenerationHarnessError(
                f"endpoint exited before readiness with code {exit_code}"
            )
        try:
            health = _health_document(url, min(2.0, timeout_seconds))
        except ProviderV2GenerationHarnessError as error:
            last_error = error
            time.sleep(0.25)
            continue
        return health, time.perf_counter() - started
    raise ProviderV2GenerationHarnessError("endpoint readiness timed out") from last_error


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def _open_log(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8", newline="\n")


def _execute(
    *,
    arguments: argparse.Namespace,
    inputs: ProviderV2GenerationInputs,
    paths: ProviderV2GenerationPaths,
    plan: Mapping[str, object],
) -> dict[str, object]:
    _prepare_actual_root(paths)
    endpoint_command = build_endpoint_command(arguments=arguments, inputs=inputs, paths=paths)
    stream_command = build_stream_command(arguments=arguments, inputs=inputs, paths=paths)
    sampler = NvidiaSmiGpuSampler(interval_seconds=arguments.gpu_sample_interval_seconds)
    endpoint_process: subprocess.Popen[bytes] | None = None
    endpoint_stdout: TextIO | None = None
    endpoint_stderr: TextIO | None = None
    stream_started: float | None = None
    overall_started = time.perf_counter()
    health: dict[str, Any] | None = None
    startup_seconds: float | None = None
    stream_exit_code: int | None = None
    failure: str | None = None
    telemetry_payload: dict[str, object] | None = None
    try:
        sampler.start()
        endpoint_stdout = _open_log(paths.endpoint_stdout)
        endpoint_stderr = _open_log(paths.endpoint_stderr)
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["HF_HOME"] = str(paths.root / "hf-home")
        environment["HF_MODULES_CACHE"] = str(paths.root / "hf-modules")
        endpoint_process = subprocess.Popen(
            endpoint_command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=endpoint_stdout,
            stderr=endpoint_stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        health, startup_seconds = _wait_for_health(
            process=endpoint_process,
            url=f"http://{arguments.host}:{arguments.port}/healthz",
            timeout_seconds=arguments.startup_timeout_seconds,
        )
        stream_started = time.perf_counter()
        with (
            _open_log(paths.stream_stdout) as stream_stdout,
            _open_log(paths.stream_stderr) as stream_stderr,
        ):
            completed = subprocess.run(
                stream_command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=stream_stdout,
                stderr=stream_stderr,
                timeout=(arguments.endpoint_timeout_seconds * SEGMENT_COUNT) + 120.0,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        stream_exit_code = completed.returncode
        if stream_exit_code != 0:
            raise ProviderV2GenerationHarnessError(
                f"stream qualification exited with code {stream_exit_code}"
            )
        if not paths.stream_report.is_file():
            raise ProviderV2GenerationHarnessError("stream qualification produced no report")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        _stop_process(endpoint_process)
        if endpoint_stdout is not None:
            endpoint_stdout.close()
        if endpoint_stderr is not None:
            endpoint_stderr.close()
        telemetry = sampler.stop()
        telemetry_payload = telemetry.to_payload()
        _write_document(paths.full_wall_gpu_telemetry, telemetry_payload)
        stream_seconds = None if stream_started is None else time.perf_counter() - stream_started
        result: dict[str, object] = {
            **dict(plan),
            "status": "FAILED" if failure is not None else "SUCCEEDED",
            "failure": failure,
            "measurement": {
                "overall_wall_seconds": time.perf_counter() - overall_started,
                "endpoint_startup_to_health_seconds": startup_seconds,
                "stream_wall_seconds": stream_seconds,
                "stream_exit_code": stream_exit_code,
                "model_load_included_in_overall_wall": True,
                "codec_preparation_included_in_overall_wall": False,
                "gpu_telemetry": telemetry_payload,
            },
            "endpoint_health": health,
        }
        _write_document(paths.harness_result, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        inputs = load_provider_v2_generation_inputs(arguments)
        paths = generation_paths(arguments.output_root)
        plan = build_plan_document(arguments=arguments, inputs=inputs, paths=paths)
        if arguments.plan_only:
            _write_document(paths.harness_result, {**plan, "status": "PLANNED"})
            print(json.dumps({"ok": True, "result": str(paths.harness_result)}, sort_keys=True))
            return 0
        result = _execute(arguments=arguments, inputs=inputs, paths=paths, plan=plan)
    except (
        OSError,
        ValueError,
        ProviderV2GenerationHarnessError,
        subprocess.SubprocessError,
    ) as error:
        print(
            json.dumps(
                {"ok": False, "error": type(error).__name__, "message": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"ok": True, "status": result["status"], "result": str(paths.harness_result)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
