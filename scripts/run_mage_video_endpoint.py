"""Launch the native Mage video endpoint with explicit load-profile diagnostics.

The local profile requests bitsandbytes 4-bit NF4 for a constrained RTX host.
The production profile requests the runtime native BF16 profile.  Model loading
is performed only after parsing arguments and completing dependency checks.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.inference.mage_checkpoint_identity import (  # noqa: E402
    MageCheckpointIdentityError,
    MageCheckpointManifest,
    build_mage_checkpoint_manifest,
    load_mage_checkpoint_manifest,
    verify_mage_checkpoint_manifest,
    write_mage_checkpoint_manifest,
)

LOCAL_4BIT_PROFILE = "local-4bit-nf4"
PRODUCTION_NATIVE_PROFILE = "production-native-bf16"


class MageVideoEndpointLaunchError(RuntimeError):
    """The native Mage endpoint is not ready to launch."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _port(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > 65_535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return parsed


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("must be a lowercase hexadecimal SHA-256 digest")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", required=True, type=Path, help="local Mage checkpoint directory"
    )
    parser.add_argument("--model-identifier", default="Mage-VL")
    parser.add_argument("--model-revision", default="local")
    parser.add_argument(
        "--checkpoint-manifest-sha256",
        type=_sha256,
        default=None,
        help=(
            "optional expected digest pin; the launcher still loads/builds and verifies the full "
            "checkpoint manifest against --model-dir"
        ),
    )
    parser.add_argument(
        "--checkpoint-manifest-path",
        type=Path,
        default=None,
        help=(
            "canonical mage-checkpoint-manifest-v2 path; defaults under --state-dir and is always "
            "verified against --model-dir"
        ),
    )
    parser.add_argument(
        "--refresh-checkpoint-manifest",
        action="store_true",
        help="rehash --model-dir and overwrite --checkpoint-manifest-path",
    )
    parser.add_argument(
        "--load-profile",
        choices=(LOCAL_4BIT_PROFILE, PRODUCTION_NATIVE_PROFILE),
        default=LOCAL_4BIT_PROFILE,
        help="explicit runtime load profile; local 4-bit is the local default",
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
    parser.add_argument(
        "--preprocess-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="explicit native codec preprocessing device bound into policy v2",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=REPOSITORY_ROOT / ".local" / "mage-video-endpoints" / "mage",
        help="durable endpoint idempotency, result, and offload root",
    )
    parser.add_argument("--idempotency-state-path", type=Path, default=None)
    parser.add_argument("--result-artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--generation-telemetry-jsonl",
        type=Path,
        default=None,
        help=(
            "optional non-wire JSONL sink for versioned generation timing, TTFT, token "
            "throughput, and monotonic generation intervals"
        ),
    )
    parser.add_argument(
        "--durable-input-root",
        type=Path,
        action="append",
        default=None,
        help="approved root for materialized native video inputs; repeatable",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8102)
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    parser.add_argument(
        "--readiness-only",
        action="store_true",
        help="load, validate health, print diagnostics, release the model, and exit",
    )
    return parser


def _checkpoint_manifest(
    arguments: argparse.Namespace,
    *,
    state_root: Path,
) -> tuple[MageCheckpointManifest, Path]:
    model_directory = arguments.model_dir.expanduser().resolve()
    manifest_path = (
        arguments.checkpoint_manifest_path.expanduser().resolve()
        if arguments.checkpoint_manifest_path is not None
        else (state_root / "checkpoint-manifest-v2.json").resolve()
    )
    try:
        if arguments.refresh_checkpoint_manifest or not manifest_path.is_file():
            manifest = build_mage_checkpoint_manifest(
                model_directory=model_directory,
                model_identifier=arguments.model_identifier,
                model_revision=arguments.model_revision,
            )
            write_mage_checkpoint_manifest(
                manifest=manifest,
                manifest_path=manifest_path,
            )
        else:
            manifest = load_mage_checkpoint_manifest(manifest_path=manifest_path)
            if manifest.model_identifier != arguments.model_identifier:
                raise MageVideoEndpointLaunchError(
                    "checkpoint manifest model_identifier does not match the launcher"
                )
            if manifest.model_revision != arguments.model_revision:
                raise MageVideoEndpointLaunchError(
                    "checkpoint manifest model_revision does not match the launcher"
                )
            verify_mage_checkpoint_manifest(
                manifest=manifest,
                model_directory=model_directory,
            )
    except MageCheckpointIdentityError as error:
        raise MageVideoEndpointLaunchError(str(error)) from error

    expected = arguments.checkpoint_manifest_sha256
    if expected is not None and manifest.manifest_sha256 != expected:
        raise MageVideoEndpointLaunchError(
            "verified checkpoint manifest does not match --checkpoint-manifest-sha256"
        )
    return manifest, manifest_path


def _runtime_profile(runtime_module: Any, requested_profile: str) -> Any:
    profile_type = getattr(runtime_module, "MageVideoLoadProfile", None)
    if profile_type is None:
        raise MageVideoEndpointLaunchError(
            "Mage runtime lacks MageVideoLoadProfile; update the Mage runtime before launching"
        )
    if requested_profile == LOCAL_4BIT_PROFILE:
        for name in ("BITSANDBYTES_4BIT_NF4", "BNB_4BIT_NF4"):
            value = getattr(profile_type, name, None)
            if value is not None:
                return value
        raise MageVideoEndpointLaunchError("Mage runtime lacks the bitsandbytes 4-bit NF4 profile")
    if requested_profile == PRODUCTION_NATIVE_PROFILE:
        value = getattr(profile_type, "NATIVE_BF16", None)
        if value is None:
            raise MageVideoEndpointLaunchError(
                "Mage runtime lacks the native BF16 production profile"
            )
        return value
    raise MageVideoEndpointLaunchError(f"unknown Mage load profile: {requested_profile}")


def _create_runtime(
    *,
    runtime_module: Any,
    model_directory: Path,
    offload_directory: Path,
    requested_profile: str,
) -> tuple[Any, str]:
    runtime_type = getattr(runtime_module, "MageVideoRuntime", None)
    identity_type = getattr(runtime_module, "MageVideoRuntimeIdentity", None)
    if runtime_type is None or identity_type is None:
        raise MageVideoEndpointLaunchError(
            "Mage runtime does not expose explicit profile identity support"
        )
    profile = _runtime_profile(runtime_module, requested_profile)
    try:
        parameters = inspect.signature(runtime_type).parameters
    except (TypeError, ValueError) as error:
        raise MageVideoEndpointLaunchError(
            "could not inspect Mage runtime profile support"
        ) from error
    if "load_profile" not in parameters or "runtime_identity" not in parameters:
        raise MageVideoEndpointLaunchError(
            "Mage runtime does not fail closed on declared load profile"
        )
    runtime_identity = identity_type(load_profile=profile)
    runtime = runtime_type(
        model_directory=model_directory,
        offload_directory=offload_directory,
        load_profile=profile,
        runtime_identity=runtime_identity,
    )
    profile_value = getattr(profile, "value", profile)
    if not isinstance(profile_value, str):
        raise MageVideoEndpointLaunchError("Mage runtime profile has no canonical string value")
    return runtime, profile_value


def _codec_policy(endpoint_module: Any, arguments: argparse.Namespace) -> Any:
    policy_type = getattr(endpoint_module, "MageVideoCodecPolicy", None)
    if policy_type is None:
        raise MageVideoEndpointLaunchError("Mage endpoint lacks MageVideoCodecPolicy")
    return policy_type(
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
    )


def _generation_telemetry_sink(
    endpoint_module: Any,
    arguments: argparse.Namespace,
) -> tuple[Any | None, Path | None]:
    requested_path = arguments.generation_telemetry_jsonl
    if requested_path is None:
        return None, None
    sink_type = getattr(endpoint_module, "MageVideoGenerationJsonlSink", None)
    if sink_type is None:
        raise MageVideoEndpointLaunchError(
            "Mage endpoint does not expose the generation telemetry JSONL sink"
        )
    resolved_path = requested_path.expanduser().resolve()
    return sink_type(resolved_path), resolved_path


def _state_paths(arguments: argparse.Namespace) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    state_root = arguments.state_dir.expanduser().resolve()
    try:
        state_root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MageVideoEndpointLaunchError("could not create state directory") from error
    idempotency_path = (
        arguments.idempotency_state_path.expanduser().resolve()
        if arguments.idempotency_state_path is not None
        else state_root / "endpoint-idempotency.sqlite3"
    )
    result_root = (
        arguments.result_artifact_dir.expanduser().resolve()
        if arguments.result_artifact_dir is not None
        else state_root / "result-artifacts"
    )
    raw_roots = arguments.durable_input_root or [state_root / "inputs"]
    input_roots = tuple(Path(item).expanduser().resolve() for item in raw_roots)
    try:
        for root in input_roots:
            root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MageVideoEndpointLaunchError("could not create durable input root") from error
    return state_root, idempotency_path, result_root, input_roots


def _print_json(payload: dict[str, object], *, error: bool = False) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        file=sys.stderr if error else sys.stdout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    runtime: Any | None = None
    service: Any | None = None
    try:
        endpoint_module = importlib.import_module("robata.inference.mage_video_endpoint")
        runtime_module = importlib.import_module("robata.inference.mage_video_runtime")
        uvicorn = importlib.import_module("uvicorn")
        state_root, idempotency_path, result_root, input_roots = _state_paths(arguments)
        checkpoint_manifest, checkpoint_manifest_path = _checkpoint_manifest(
            arguments,
            state_root=state_root,
        )
        checkpoint_sha256 = checkpoint_manifest.manifest_sha256
        policy = _codec_policy(endpoint_module, arguments)
        generation_telemetry_sink, generation_telemetry_path = _generation_telemetry_sink(
            endpoint_module,
            arguments,
        )
        codec_checker = getattr(runtime_module, "require_mage_video_codec_dependencies", None)
        if not callable(codec_checker):
            raise MageVideoEndpointLaunchError(
                "Mage runtime does not expose native codec diagnostics"
            )
        codec_checker(policy.native_codec_config(), arguments.model_dir.expanduser().resolve())
        runtime, canonical_profile = _create_runtime(
            runtime_module=runtime_module,
            model_directory=arguments.model_dir.expanduser().resolve(),
            offload_directory=state_root / "model-offload",
            requested_profile=arguments.load_profile,
        )
        runtime_identity = getattr(runtime, "runtime_identity", None)
        if runtime_identity is None:
            raise MageVideoEndpointLaunchError("Mage runtime exposes no resident runtime identity")
        model_identity = endpoint_module.MageVideoModelIdentity(
            model_identifier=arguments.model_identifier,
            model_revision=arguments.model_revision,
            checkpoint_manifest_sha256=checkpoint_sha256,
            runtime_identity=runtime_identity,
        )
        service = endpoint_module.MageVideoEndpointService(
            runtime=runtime,
            model_identity=model_identity,
            idempotency_state_path=idempotency_path,
            result_artifact_directory=result_root,
            durable_input_roots=input_roots,
            generation_telemetry_sink=generation_telemetry_sink,
        )
        service.start()
        health = service.health()
        _print_json(
            {
                "ok": True,
                "status": health.status,
                "endpoint": {"host": arguments.host, "port": arguments.port},
                "load_profile": canonical_profile,
                "load_profile_request": arguments.load_profile,
                "runtime_identity": model_identity.runtime_identity.model_dump(mode="json"),
                "model_identifier": arguments.model_identifier,
                "model_revision": arguments.model_revision,
                "checkpoint_manifest_sha256": checkpoint_sha256,
                "checkpoint_manifest_path": str(checkpoint_manifest_path),
                "checkpoint_manifest_version": checkpoint_manifest.manifest_version,
                "checkpoint_inclusion_policy_version": (
                    checkpoint_manifest.inclusion_policy_version
                ),
                "checkpoint_included_file_count": checkpoint_manifest.included_file_count,
                "checkpoint_total_byte_count": checkpoint_manifest.total_byte_count,
                "codec_policy": policy.model_dump(mode="json"),
                "state_dir": str(state_root),
                "idempotency_state_path": str(idempotency_path),
                "result_artifact_dir": str(result_root),
                "generation_telemetry_jsonl": (
                    str(generation_telemetry_path)
                    if generation_telemetry_path is not None
                    else None
                ),
                "durable_input_roots": [str(item) for item in input_roots],
                "limitation": "v2 accepts one native video input and one decoder only",
            }
        )
        if arguments.readiness_only:
            service.stop()
            return 0
        application = endpoint_module.create_mage_video_endpoint_app(service)
        uvicorn.run(
            application,
            host=arguments.host,
            port=arguments.port,
            workers=1,
            log_level=arguments.log_level,
            access_log=True,
        )
    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        MageVideoEndpointLaunchError,
    ) as error:
        _print_json(
            {
                "ok": False,
                "code": "MAGE_VIDEO_ENDPOINT_FAILED",
                "detail": str(error),
                "load_profile_request": arguments.load_profile,
            },
            error=True,
        )
        return 2
    finally:
        if (
            arguments.readiness_only
            and service is not None
            and runtime is not None
            and getattr(runtime, "loaded", False)
        ):
            service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
