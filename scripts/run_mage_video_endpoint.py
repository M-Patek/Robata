"""Launch the native Mage video endpoint with explicit load-profile diagnostics.

The local profile requests bitsandbytes 4-bit NF4 for a constrained RTX host.
The production profile requests the runtime native BF16 profile.  Model loading
is performed only after parsing arguments and completing dependency checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import ipaddress
import json
import sys
import time
import uuid
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.hashing import (  # noqa: E402
    canonical_json_bytes,
    exact_bytes_sha256,
    semantic_sha256,
)
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
OBSERVED_V1_CACHE_FAMILY = "observed-v1"
PROVIDER_V2_CACHE_FAMILY = "provider-v2"
TRADITIONAL_V1_CACHE_FAMILY = "traditional-v1"
OBSERVED_V1_CACHE_MANIFEST_VERSION = "mage-codec-cache-manifest-v1"
PROVIDER_V2_CACHE_MANIFEST_VERSION = "mage-codec-cache-manifest-v2"
TRADITIONAL_V1_CACHE_MANIFEST_VERSION = "mage-traditional-codec-cache-manifest-v1"
CONTROLLED_PRIVATE_NETWORK = "controlled-private-network"
AUTHENTICATED_REVERSE_PROXY = "authenticated-reverse-proxy"
_DECLARED_NETWORK_BOUNDARIES = (
    CONTROLLED_PRIVATE_NETWORK,
    AUTHENTICATED_REVERSE_PROXY,
)


class MageCodecCacheLaunchConfiguration:
    """Verified launcher-only cache state without changing endpoint wire contracts."""

    __slots__ = (
        "admission",
        "cache_family",
        "cache_root",
        "manifest",
        "manifest_path",
        "qualified_provider_manifest",
        "qualified_provider_manifest_path",
    )

    def __init__(
        self,
        *,
        cache_root: Path | None,
        admission: Any | None,
        manifest: Any | None,
        manifest_path: Path | None,
        cache_family: str | None,
        qualified_provider_manifest: Any | None = None,
        qualified_provider_manifest_path: Path | None = None,
    ) -> None:
        self.cache_root = cache_root
        self.admission = admission
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.cache_family = cache_family
        self.qualified_provider_manifest = qualified_provider_manifest
        self.qualified_provider_manifest_path = qualified_provider_manifest_path


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
    parser.add_argument("--neural-qp", type=int, default=42)
    parser.add_argument("--neural-reset-interval", type=_positive_int, default=64)
    parser.add_argument("--neural-intra-period", type=int, default=-1)
    parser.add_argument("--neural-max-side", type=int, default=0)
    parser.add_argument("--neural-sequence-length-frames", type=int, default=0)
    parser.add_argument("--neural-canvas-token-side", type=_positive_int, default=None)
    parser.add_argument("--neural-readiness-coverage-bins", type=_positive_int, default=3)
    parser.add_argument("--neural-readiness-delta-ratio", type=float, default=0.05)
    parser.add_argument("--neural-bitcost-percentile", type=_positive_int, default=99)
    parser.add_argument("--neural-decode-backsearch-max", type=_positive_int, default=16)
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
        "--codec-cache-manifest",
        type=Path,
        default=None,
        help=(
            "verified observed-v1, explicit Provider V2, or deployment-pinned traditional "
            "H.264/HEVC cache manifest used as the only codec cache root"
        ),
    )
    parser.add_argument(
        "--qualified-provider-manifest",
        type=Path,
        default=None,
        help=(
            "required mage-dcvc-qualified-provider-manifest-v2 when the cache manifest is "
            "Provider V2"
        ),
    )
    parser.add_argument(
        "--shared-device-guard-file",
        type=Path,
        default=None,
        help=(
            "operational cross-process lock shared with the DCVC preparation worker; "
            "required by Provider V2 exclusive-shared-device-v1 configurations"
        ),
    )
    parser.add_argument(
        "--require-verified-codec-cache",
        action="store_true",
        help="reject any request whose exact source/policy/checkpoint is absent from the manifest",
    )
    cache_family_gate = parser.add_mutually_exclusive_group()
    cache_family_gate.add_argument(
        "--require-provider-v2-cache",
        action="store_true",
        help=(
            "fail closed unless --codec-cache-manifest is the explicit Provider V2 family; "
            "observed-v1 remains available only as an intentional rollback path"
        ),
    )
    cache_family_gate.add_argument(
        "--require-traditional-codec-cache",
        action="store_true",
        help=(
            "fail closed unless --codec-cache-manifest is the additive, deployment-pinned "
            "traditional H.264/HEVC cache family"
        ),
    )
    parser.add_argument(
        "--traditional-provider-identity-sha256",
        type=_sha256,
        default=None,
        help="required deployment pin for a traditional codec cache manifest",
    )
    parser.add_argument(
        "--traditional-toolchain-identity-sha256",
        type=_sha256,
        default=None,
        help="required codec-video-prep/toolchain identity pin for traditional cache replay",
    )
    parser.add_argument(
        "--traditional-container-image-digest",
        type=_sha256,
        default=None,
        help="required lowercase SHA-256 digest of the qualified traditional codec image",
    )
    parser.add_argument("--warmup-video", type=Path, default=None)
    parser.add_argument("--warmup-video-sha256", type=_sha256, default=None)
    parser.add_argument("--warmup-prompt-file", type=Path, default=None)
    parser.add_argument("--warmup-max-new-tokens", type=_positive_int, default=32)
    parser.add_argument("--warmup-report-json", type=Path, default=None)
    parser.add_argument(
        "--durable-input-root",
        type=Path,
        action="append",
        default=None,
        help="approved root for materialized native video inputs; repeatable",
    )
    parser.add_argument("--host", default="127.0.0.1")
    bind_group = parser.add_mutually_exclusive_group()
    bind_group.add_argument(
        "--network-boundary",
        choices=_DECLARED_NETWORK_BOUNDARIES,
        default=None,
        help=(
            "operator declaration that a wildcard bind is confined to a controlled private "
            "network or is reachable only through an authenticated reverse proxy; this "
            "launcher does not provide endpoint authentication"
        ),
    )
    bind_group.add_argument(
        "--allow-unauthenticated-public-bind",
        action="store_true",
        help=(
            "HIGH RISK: explicitly acknowledge that a wildcard bind may expose this "
            "unauthenticated endpoint publicly"
        ),
    )
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


def _is_wildcard_bind_host(host: str) -> bool:
    normalized = host.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized == "*":
        return True
    try:
        return ipaddress.ip_address(normalized).is_unspecified
    except ValueError:
        return False


def _validate_bind_security(arguments: argparse.Namespace) -> dict[str, object]:
    host = str(arguments.host).strip()
    if not host:
        raise MageVideoEndpointLaunchError("--host must not be empty")

    network_boundary = arguments.network_boundary
    public_bind_acknowledged = bool(arguments.allow_unauthenticated_public_bind)
    wildcard = _is_wildcard_bind_host(host)
    if wildcard and network_boundary is None and not public_bind_acknowledged:
        raise MageVideoEndpointLaunchError(
            "wildcard bind requires --network-boundary controlled-private-network, "
            "--network-boundary authenticated-reverse-proxy, or the HIGH-RISK "
            "--allow-unauthenticated-public-bind acknowledgement"
        )

    return {
        "host": host,
        "wildcard": wildcard,
        "endpoint_authentication": "NOT_PROVIDED_BY_LAUNCHER",
        "network_boundary": network_boundary,
        "unauthenticated_public_bind_acknowledged": public_bind_acknowledged,
    }


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
    codec_cache_root: Path | None = None,
    shared_device_guard_file: Path | None = None,
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
    runtime_kwargs: dict[str, object] = {
        "model_directory": model_directory,
        "offload_directory": offload_directory,
        "load_profile": profile,
        "runtime_identity": runtime_identity,
        "codec_cache_root": codec_cache_root,
    }
    if shared_device_guard_file is not None:
        if "shared_device_guard_file" not in parameters:
            raise MageVideoEndpointLaunchError(
                "Mage runtime does not support the required shared device guard"
            )
        runtime_kwargs["shared_device_guard_file"] = shared_device_guard_file
    runtime = runtime_type(**runtime_kwargs)
    profile_value = getattr(profile, "value", profile)
    if not isinstance(profile_value, str):
        raise MageVideoEndpointLaunchError("Mage runtime profile has no canonical string value")
    return runtime, profile_value


def _codec_policy(endpoint_module: Any, arguments: argparse.Namespace) -> Any:
    policy_type = getattr(endpoint_module, "MageVideoCodecPolicy", None)
    neural_type = getattr(endpoint_module, "MageVideoNeuralCodecParameters", None)
    if policy_type is None or neural_type is None:
        raise MageVideoEndpointLaunchError("Mage endpoint lacks codec policy contracts")
    neural_parameters = None
    if arguments.codec_mode == "neural":
        if not 0 <= arguments.neural_qp <= 63:
            raise MageVideoEndpointLaunchError("--neural-qp must be between 0 and 63")
        if arguments.neural_intra_period < -1:
            raise MageVideoEndpointLaunchError("--neural-intra-period must be at least -1")
        if arguments.neural_max_side < 0 or arguments.neural_sequence_length_frames < 0:
            raise MageVideoEndpointLaunchError(
                "neural size and sequence controls must be nonnegative"
            )
        if arguments.neural_readiness_delta_ratio <= 0.0:
            raise MageVideoEndpointLaunchError("--neural-readiness-delta-ratio must be positive")
        neural_parameters = neural_type(
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
        neural_parameters=neural_parameters,
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


def _require_launch_codec_dependencies(
    *,
    runtime_module: Any,
    codec_policy: Any,
    model_directory: Path,
    cache_family: str | None,
) -> dict[str, object]:
    # Exact traditional replay loads already-qualified assets through Mage's own
    # result loader. Requiring cv-preinfer here would make the replay endpoint
    # depend on a tool it is specifically forbidden to execute.
    if cache_family == TRADITIONAL_V1_CACHE_FAMILY:
        return {
            "report_version": "mage-native-codec-launch-v1",
            "ready": True,
            "mode": "TRADITIONAL_EXACT_CACHE_REPLAY",
            "blocker_code": None,
            "missing_assets": [],
            "detail": "qualified traditional codec assets are replayed through Mage's loader",
        }
    native_config = codec_policy.native_codec_config()
    inspector = getattr(runtime_module, "inspect_mage_video_codec_dependencies", None)
    report: dict[str, object] | None = None
    if callable(inspector):
        observed = inspector(native_config, model_directory)
        as_dict = getattr(observed, "as_dict", None)
        if not callable(as_dict):
            raise MageVideoEndpointLaunchError(
                "Mage runtime native codec diagnostics returned no JSON projection"
            )
        raw_report = as_dict()
        if not isinstance(raw_report, dict):
            raise MageVideoEndpointLaunchError(
                "Mage runtime native codec diagnostics returned an invalid projection"
            )
        report = dict(raw_report)
    codec_checker = getattr(runtime_module, "require_mage_video_codec_dependencies", None)
    if not callable(codec_checker):
        raise MageVideoEndpointLaunchError("Mage runtime does not expose native codec diagnostics")
    codec_checker(native_config, model_directory)
    return report or {
        "report_version": "mage-native-codec-launch-v1",
        "ready": True,
        "mode": "LEGACY_RUNTIME_CHECKER",
        "blocker_code": None,
        "missing_assets": [],
        "detail": "native codec dependency checker completed",
    }


def _codec_cache_manifest_version(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise MageVideoEndpointLaunchError("codec cache manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise MageVideoEndpointLaunchError("codec cache manifest must be a JSON object")
    version = payload.get("manifest_version")
    if not isinstance(version, str) or not version:
        raise MageVideoEndpointLaunchError("codec cache manifest has no manifest_version")
    return version


def _traditional_identity_pins(arguments: argparse.Namespace) -> tuple[str, str, str]:
    pins = {
        "--traditional-provider-identity-sha256": (arguments.traditional_provider_identity_sha256),
        "--traditional-toolchain-identity-sha256": (
            arguments.traditional_toolchain_identity_sha256
        ),
        "--traditional-container-image-digest": arguments.traditional_container_image_digest,
    }
    missing = [name for name, value in pins.items() if value is None]
    if missing:
        raise MageVideoEndpointLaunchError(
            "traditional codec cache launch requires deployment identity pins: "
            + ", ".join(missing)
        )
    provider, toolchain, image = pins.values()
    assert isinstance(provider, str)
    assert isinstance(toolchain, str)
    assert isinstance(image, str)
    return provider, toolchain, image


def _traditional_identity_flags_present(arguments: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            arguments.traditional_provider_identity_sha256,
            arguments.traditional_toolchain_identity_sha256,
            arguments.traditional_container_image_digest,
        )
    )


def _codec_cache_configuration(
    *,
    cache_module: Any,
    endpoint_module: Any,
    arguments: argparse.Namespace,
    checkpoint_sha256: str,
    codec_policy: Any,
    checkpoint_manifest: Any | None = None,
    cache_v2_module: Any | None = None,
    qualified_provider_module: Any | None = None,
    preparation_worker_module: Any | None = None,
    traditional_cache_module: Any | None = None,
) -> MageCodecCacheLaunchConfiguration:
    requested = arguments.codec_cache_manifest
    if requested is None:
        required_gates = [
            name
            for name, enabled in (
                ("--require-verified-codec-cache", arguments.require_verified_codec_cache),
                ("--require-provider-v2-cache", arguments.require_provider_v2_cache),
                (
                    "--require-traditional-codec-cache",
                    arguments.require_traditional_codec_cache,
                ),
            )
            if enabled
        ]
        if required_gates:
            raise MageVideoEndpointLaunchError(
                f"{required_gates[0]} requires --codec-cache-manifest"
            )
        if arguments.qualified_provider_manifest is not None:
            raise MageVideoEndpointLaunchError(
                "--qualified-provider-manifest requires a Provider V2 cache manifest"
            )
        if _traditional_identity_flags_present(arguments):
            raise MageVideoEndpointLaunchError(
                "traditional deployment identity pins require a traditional codec cache manifest"
            )
        return MageCodecCacheLaunchConfiguration(
            cache_root=None,
            admission=None,
            manifest=None,
            manifest_path=None,
            cache_family=None,
        )

    manifest_path = requested.expanduser().resolve()
    version = _codec_cache_manifest_version(manifest_path)
    if version == OBSERVED_V1_CACHE_MANIFEST_VERSION:
        return _observed_v1_cache_configuration(
            cache_module=cache_module,
            endpoint_module=endpoint_module,
            arguments=arguments,
            manifest_path=manifest_path,
            checkpoint_sha256=checkpoint_sha256,
            codec_policy=codec_policy,
        )
    if version == PROVIDER_V2_CACHE_MANIFEST_VERSION:
        return _provider_v2_cache_configuration(
            cache_v2_module=cache_v2_module,
            qualified_provider_module=qualified_provider_module,
            preparation_worker_module=preparation_worker_module,
            endpoint_module=endpoint_module,
            arguments=arguments,
            manifest_path=manifest_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_manifest=checkpoint_manifest,
            codec_policy=codec_policy,
        )
    if version == TRADITIONAL_V1_CACHE_MANIFEST_VERSION:
        return _traditional_v1_cache_configuration(
            traditional_cache_module=traditional_cache_module,
            arguments=arguments,
            manifest_path=manifest_path,
            checkpoint_sha256=checkpoint_sha256,
            codec_policy=codec_policy,
        )
    raise MageVideoEndpointLaunchError(f"unsupported codec cache manifest_version: {version}")


def _observed_v1_cache_configuration(
    *,
    cache_module: Any,
    endpoint_module: Any,
    arguments: argparse.Namespace,
    manifest_path: Path,
    checkpoint_sha256: str,
    codec_policy: Any,
) -> MageCodecCacheLaunchConfiguration:
    if arguments.require_provider_v2_cache:
        raise MageVideoEndpointLaunchError(
            "--require-provider-v2-cache rejects the observed-v1 rollback cache family"
        )
    if arguments.require_traditional_codec_cache:
        raise MageVideoEndpointLaunchError(
            "--require-traditional-codec-cache rejects the observed-v1 rollback cache family"
        )
    if _traditional_identity_flags_present(arguments):
        raise MageVideoEndpointLaunchError(
            "traditional deployment identity pins are invalid for observed-v1"
        )
    if arguments.qualified_provider_manifest is not None:
        raise MageVideoEndpointLaunchError(
            "--qualified-provider-manifest is valid only with a Provider V2 cache manifest"
        )
    manifest = cache_module.load_mage_codec_cache_manifest(path=manifest_path)
    if manifest.manifest_version != OBSERVED_V1_CACHE_MANIFEST_VERSION:
        raise MageVideoEndpointLaunchError("observed-v1 loader returned another cache family")
    verified_entries = cache_module.verify_mage_codec_cache_manifest(manifest=manifest)
    _verify_cache_manifest_launcher_identity(
        manifest=manifest,
        endpoint_module=endpoint_module,
        checkpoint_sha256=checkpoint_sha256,
        codec_policy=codec_policy,
    )
    if len(verified_entries) != len(manifest.entries):
        raise MageVideoEndpointLaunchError("codec cache manifest verification count mismatch")
    entries_by_source = {
        entry.source_path: (manifest_entry, entry)
        for manifest_entry, entry in zip(manifest.entries, verified_entries, strict=True)
    }

    def admit(request: Any, paths: Sequence[Path]) -> None:
        source_path, entry_pair = _admitted_source_entry(
            paths=paths,
            entries_by_source=entries_by_source,
        )
        manifest_entry, entry = entry_pair
        _verify_cache_request_identity(
            request=request,
            source_path=source_path,
            entry=entry,
            endpoint_module=endpoint_module,
            checkpoint_sha256=manifest.checkpoint_manifest_sha256,
            codec_policy_sha256=manifest.codec_policy_sha256,
        )
        cache_module.verify_mage_codec_cache_entry(
            cache_directory=Path(manifest_entry.provider_cache_directory),
            expected_source_path=Path(source_path),
            expected_checkpoint_manifest_sha256=manifest.checkpoint_manifest_sha256,
            expected_codec_policy_sha256=manifest.codec_policy_sha256,
            expected_recipe_sha256=manifest.recipe.semantic_sha256,
            expected_namespace_identity=manifest.namespace_identity,
        )

    return MageCodecCacheLaunchConfiguration(
        cache_root=Path(manifest.qualified_cache_root),
        admission=admit,
        manifest=manifest,
        manifest_path=manifest_path,
        cache_family=OBSERVED_V1_CACHE_FAMILY,
    )


def _provider_v2_cache_configuration(
    *,
    cache_v2_module: Any | None,
    qualified_provider_module: Any | None,
    preparation_worker_module: Any | None,
    endpoint_module: Any,
    arguments: argparse.Namespace,
    manifest_path: Path,
    checkpoint_sha256: str,
    checkpoint_manifest: Any | None,
    codec_policy: Any,
) -> MageCodecCacheLaunchConfiguration:
    if arguments.require_traditional_codec_cache:
        raise MageVideoEndpointLaunchError(
            "--require-traditional-codec-cache rejects the Provider V2 cache family"
        )
    if _traditional_identity_flags_present(arguments):
        raise MageVideoEndpointLaunchError(
            "traditional deployment identity pins are invalid for Provider V2"
        )
    if (
        cache_v2_module is None
        or qualified_provider_module is None
        or preparation_worker_module is None
    ):
        raise MageVideoEndpointLaunchError("Provider V2 cache support modules are unavailable")
    if checkpoint_manifest is None:
        raise MageVideoEndpointLaunchError("Provider V2 requires the verified checkpoint manifest")
    requested_qualified = arguments.qualified_provider_manifest
    if requested_qualified is None:
        raise MageVideoEndpointLaunchError(
            "Provider V2 cache launch requires --qualified-provider-manifest"
        )

    manifest = cache_v2_module.load_mage_codec_cache_manifest_v2(path=manifest_path)
    if manifest.manifest_version != PROVIDER_V2_CACHE_MANIFEST_VERSION:
        raise MageVideoEndpointLaunchError("Provider V2 loader returned another cache family")
    verified_entries = cache_v2_module.verify_mage_codec_cache_manifest_v2(manifest=manifest)
    _verify_cache_manifest_launcher_identity(
        manifest=manifest,
        endpoint_module=endpoint_module,
        checkpoint_sha256=checkpoint_sha256,
        codec_policy=codec_policy,
    )
    if len(verified_entries) != len(manifest.entries):
        raise MageVideoEndpointLaunchError("Provider V2 cache verification count mismatch")

    qualified_path = requested_qualified.expanduser().resolve()
    qualified_manifest = qualified_provider_module.load_mage_dcvc_qualified_provider_manifest(
        manifest_path=qualified_path
    )
    qualified_provider_module.verify_mage_dcvc_qualified_provider(manifest=qualified_manifest)
    _verify_provider_v2_launch_identity(
        cache_v2_module=cache_v2_module,
        preparation_worker_module=preparation_worker_module,
        arguments=arguments,
        checkpoint_manifest=checkpoint_manifest,
        cache_manifest=manifest,
        qualified_manifest=qualified_manifest,
        codec_policy=codec_policy,
    )

    binding_type = getattr(endpoint_module, "MageVideoCodecCacheBinding", None)
    if binding_type is None:
        raise MageVideoEndpointLaunchError(
            "Mage endpoint lacks the internal exact codec cache binding"
        )
    entries_by_source = {
        entry.source_path: (manifest_entry, entry)
        for manifest_entry, entry in zip(manifest.entries, verified_entries, strict=True)
    }

    def admit(request: Any, paths: Sequence[Path]) -> Any:
        source_path, entry_pair = _admitted_source_entry(
            paths=paths,
            entries_by_source=entries_by_source,
        )
        manifest_entry, expected_entry = entry_pair
        _verify_cache_request_identity(
            request=request,
            source_path=source_path,
            entry=expected_entry,
            endpoint_module=endpoint_module,
            checkpoint_sha256=manifest.checkpoint_manifest_sha256,
            codec_policy_sha256=manifest.codec_policy_sha256,
        )
        verified_entry = cache_v2_module.verify_mage_codec_cache_entry_v2(
            cache_directory=Path(manifest_entry.provider_cache_directory),
            expected_entry=expected_entry,
            effective_config=manifest.effective_config,
        )
        if (
            verified_entry.source_path != source_path
            or verified_entry.checkpoint_manifest_sha256 != manifest.checkpoint_manifest_sha256
            or verified_entry.codec_policy_sha256 != manifest.codec_policy_sha256
            or verified_entry.namespace_identity != manifest.namespace_identity
            or verified_entry.provider_implementation_sha256
            != manifest.provider_implementation_sha256
            or verified_entry.effective_config_sha256
            != manifest.effective_config.effective_config_sha256
            or verified_entry.recipe_version != manifest.recipe_version
        ):
            raise RuntimeError("Provider V2 cache entry no longer matches its admitted namespace")
        return binding_type(
            source_path=Path(source_path),
            provider_cache_directory=Path(manifest_entry.provider_cache_directory),
        )

    return MageCodecCacheLaunchConfiguration(
        cache_root=Path(manifest.qualified_cache_root),
        admission=admit,
        manifest=manifest,
        manifest_path=manifest_path,
        cache_family=PROVIDER_V2_CACHE_FAMILY,
        qualified_provider_manifest=qualified_manifest,
        qualified_provider_manifest_path=qualified_path,
    )


def _traditional_v1_cache_configuration(
    *,
    traditional_cache_module: Any | None,
    arguments: argparse.Namespace,
    manifest_path: Path,
    checkpoint_sha256: str,
    codec_policy: Any,
) -> MageCodecCacheLaunchConfiguration:
    if arguments.require_provider_v2_cache:
        raise MageVideoEndpointLaunchError(
            "--require-provider-v2-cache rejects the traditional cache family"
        )
    if arguments.qualified_provider_manifest is not None:
        raise MageVideoEndpointLaunchError(
            "--qualified-provider-manifest is valid only with a Provider V2 cache manifest"
        )
    if arguments.shared_device_guard_file is not None:
        raise MageVideoEndpointLaunchError(
            "--shared-device-guard-file is a DCVC Provider V2 control and is invalid for "
            "traditional cache replay"
        )
    if arguments.codec_mode != "traditional" or getattr(codec_policy, "codec_mode", None) != (
        "traditional"
    ):
        raise MageVideoEndpointLaunchError(
            "traditional cache launch requires --codec-mode traditional"
        )
    if (
        arguments.preprocess_device != "cpu"
        or getattr(codec_policy, "preprocess_device", None) != "cpu"
    ):
        raise MageVideoEndpointLaunchError(
            "traditional cache launch requires --preprocess-device cpu"
        )
    if traditional_cache_module is None:
        raise MageVideoEndpointLaunchError("traditional codec cache support module is unavailable")
    admission_type = getattr(traditional_cache_module, "MageTraditionalCodecCacheAdmission", None)
    if admission_type is None or not callable(getattr(admission_type, "from_manifest_path", None)):
        raise MageVideoEndpointLaunchError(
            "traditional codec cache module lacks strict admission support"
        )
    provider_pin, toolchain_pin, image_pin = _traditional_identity_pins(arguments)
    try:
        admission = admission_type.from_manifest_path(
            manifest_path=manifest_path,
            expected_checkpoint_manifest_sha256=checkpoint_sha256,
            expected_codec_policy=codec_policy,
            expected_provider_identity_sha256=provider_pin,
            expected_toolchain_identity_sha256=toolchain_pin,
            expected_container_image_digest=image_pin,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise MageVideoEndpointLaunchError(
            f"traditional codec cache admission failed: {error}"
        ) from error
    manifest = admission.manifest
    if manifest.manifest_version != TRADITIONAL_V1_CACHE_MANIFEST_VERSION:
        raise MageVideoEndpointLaunchError("traditional cache loader returned another cache family")
    if manifest.effective_config.engine not in {"hevc", "cv-preinfer"}:
        raise MageVideoEndpointLaunchError(
            "traditional cache manifest selected a non-traditional engine"
        )
    if manifest.effective_config.native_codec_config.get("preprocess_device") != "cpu":
        raise MageVideoEndpointLaunchError(
            "traditional cache manifest was not qualified with CPU preprocessing"
        )
    return MageCodecCacheLaunchConfiguration(
        cache_root=admission.cache_root,
        admission=admission,
        manifest=manifest,
        manifest_path=manifest_path,
        cache_family=TRADITIONAL_V1_CACHE_FAMILY,
    )


def _verify_cache_manifest_launcher_identity(
    *,
    manifest: Any,
    endpoint_module: Any,
    checkpoint_sha256: str,
    codec_policy: Any,
) -> None:
    if manifest.checkpoint_manifest_sha256 != checkpoint_sha256:
        raise MageVideoEndpointLaunchError("codec cache checkpoint identity does not match")
    policy_identity = endpoint_module.build_mage_video_codec_policy_identity(codec_policy)
    if manifest.codec_policy_sha256 != policy_identity.policy_sha256:
        raise MageVideoEndpointLaunchError("codec cache policy identity does not match launcher")


def _admitted_source_entry(
    *,
    paths: Sequence[Path],
    entries_by_source: dict[str, tuple[Any, Any]],
) -> tuple[str, tuple[Any, Any]]:
    if len(paths) != 1:
        raise RuntimeError("verified Mage codec cache admits exactly one video path")
    source_path = str(Path(paths[0]).expanduser().resolve())
    pair = entries_by_source.get(source_path)
    if pair is None:
        raise RuntimeError("request source is absent from the verified Mage codec cache")
    return source_path, pair


def _verify_cache_request_identity(
    *,
    request: Any,
    source_path: str,
    entry: Any,
    endpoint_module: Any,
    checkpoint_sha256: str,
    codec_policy_sha256: str,
) -> None:
    segment = request.camera_encodings[0].segment_manifest
    if segment.content_sha256 != entry.source_content_sha256:
        raise RuntimeError("request source digest does not match verified Mage codec cache")
    if segment.byte_count != entry.source_byte_count:
        raise RuntimeError("request source byte count does not match verified Mage codec cache")
    if entry.source_path != source_path:
        raise RuntimeError("verified Mage codec cache source binding changed")
    if request.model_identity.checkpoint_manifest_sha256 != checkpoint_sha256:
        raise RuntimeError("request checkpoint does not match verified Mage codec cache")
    request_policy = endpoint_module.build_mage_video_codec_policy_identity(request.codec_policy)
    if request_policy.policy_sha256 != codec_policy_sha256:
        raise RuntimeError("request codec policy does not match verified Mage codec cache")


def _verify_provider_v2_launch_identity(
    *,
    cache_v2_module: Any,
    preparation_worker_module: Any,
    arguments: argparse.Namespace,
    checkpoint_manifest: Any,
    cache_manifest: Any,
    qualified_manifest: Any,
    codec_policy: Any,
) -> None:
    model_root = arguments.model_dir.expanduser().resolve()
    qualified_root = Path(qualified_manifest.qualified_model_directory).expanduser().resolve()
    if qualified_root != model_root:
        raise MageVideoEndpointLaunchError(
            "qualified provider model directory does not match --model-dir"
        )
    qualified_checkpoint = qualified_manifest.qualified_checkpoint_manifest
    if qualified_checkpoint != checkpoint_manifest:
        raise MageVideoEndpointLaunchError(
            "qualified provider checkpoint manifest does not match the current checkpoint"
        )
    if (
        qualified_checkpoint.manifest_sha256 != cache_manifest.checkpoint_manifest_sha256
        or qualified_checkpoint.model_identifier != arguments.model_identifier
        or qualified_checkpoint.model_revision != arguments.model_revision
    ):
        raise MageVideoEndpointLaunchError(
            "qualified provider checkpoint identity/revision does not match the launcher"
        )
    bundle = qualified_manifest.bundle
    if (
        bundle.qualified_model_identifier != arguments.model_identifier
        or bundle.qualified_model_revision != arguments.model_revision
        or qualified_manifest.provider_version != cache_manifest.provider_version
    ):
        raise MageVideoEndpointLaunchError(
            "qualified provider bundle identity does not match the launcher/cache"
        )

    effective_config = cache_manifest.effective_config
    cache_v2_module.validate_mage_dcvc_effective_config_for_policy(
        effective_config=effective_config,
        codec_policy=codec_policy,
    )
    concurrency_policy = getattr(effective_config, "device_concurrency_policy", None)
    preparation_device = getattr(effective_config, "preparation_device", None)
    if concurrency_policy == "exclusive-shared-device-v1":
        if not isinstance(preparation_device, str) or not preparation_device.startswith("cuda"):
            raise MageVideoEndpointLaunchError(
                "Provider V2 shared-device policy must identify a CUDA preparation device"
            )
        if arguments.shared_device_guard_file is None:
            raise MageVideoEndpointLaunchError(
                "Provider V2 shared local GPU requires --shared-device-guard-file"
            )
        guard_path = arguments.shared_device_guard_file.expanduser().resolve()
        if guard_path.exists() and guard_path.is_dir():
            raise MageVideoEndpointLaunchError("--shared-device-guard-file must not be a directory")
    elif concurrency_policy != "separate-device-v1":
        raise MageVideoEndpointLaunchError(
            "Provider V2 effective config has no supported device concurrency policy"
        )
    _verify_current_provider_sources_match_qualified_bundle(
        preparation_worker_module=preparation_worker_module,
        qualified_manifest=qualified_manifest,
    )
    observed_implementation = (
        preparation_worker_module.build_mage_dcvc_provider_implementation_sha256(model_root)
    )
    if (
        observed_implementation != cache_manifest.provider_implementation_sha256
        or observed_implementation != effective_config.provider_implementation_sha256
    ):
        raise MageVideoEndpointLaunchError(
            "current Provider V2 implementation does not match the admitted cache"
        )
    for checkpoint_name, expected_sha256 in (
        ("dcvc_rt_intra.tar", effective_config.intra_checkpoint_sha256),
        ("dcvc_rt_inter.tar", effective_config.inter_checkpoint_sha256),
    ):
        observed_sha256, _ = _exact_file_sha256(
            model_root / "neural_codec" / checkpoint_name,
            description=f"Provider V2 {checkpoint_name}",
        )
        if observed_sha256 != expected_sha256:
            raise MageVideoEndpointLaunchError(
                f"current Provider V2 {checkpoint_name} does not match effective config"
            )


def _verify_current_provider_sources_match_qualified_bundle(
    *,
    preparation_worker_module: Any,
    qualified_manifest: Any,
) -> None:
    """Fail closed unless the exact executing Robata sources are in the qualified tree."""

    protocol_module = getattr(preparation_worker_module, "_protocol", None)
    device_guard_module = getattr(preparation_worker_module, "_device_guard", None)
    runtime_modules = {
        "device_execution_guard.py": device_guard_module,
        "mage_dcvc_preparation_protocol.py": protocol_module,
        "mage_dcvc_preparation_worker.py": preparation_worker_module,
    }
    runtime_sources: dict[str, Path] = {}
    for expected_name, module in runtime_modules.items():
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            raise MageVideoEndpointLaunchError(
                f"current Provider V2 source is not inspectable: {expected_name}"
            )
        unresolved = Path(module_file).expanduser()
        if unresolved.is_symlink():
            raise MageVideoEndpointLaunchError(
                f"current Provider V2 source must not be a symlink: {expected_name}"
            )
        source = unresolved.resolve()
        if source.name != expected_name or not source.is_file():
            raise MageVideoEndpointLaunchError(
                f"current Provider V2 source path is invalid: {expected_name}"
            )
        runtime_sources[expected_name] = source

    provider_root = PurePosixPath("neural_codec") / "robata_provider_v2"
    qualified_sources: dict[str, Any] = {}
    for provider_file in qualified_manifest.bundle.provider_files:
        relative = PurePosixPath(provider_file.relative_path)
        if relative.parent != provider_root or relative.name in qualified_sources:
            raise MageVideoEndpointLaunchError(
                "qualified provider bundle has an invalid runtime source set"
            )
        qualified_sources[relative.name] = provider_file
    if set(qualified_sources) != set(runtime_sources):
        raise MageVideoEndpointLaunchError(
            "qualified provider bundle does not bind the exact executing source set"
        )

    for source_name, source in sorted(runtime_sources.items()):
        observed_sha256, observed_byte_count = _exact_file_sha256(
            source,
            description=f"current Provider V2 source {source_name}",
        )
        qualified_source = qualified_sources[source_name]
        if (
            observed_sha256 != qualified_source.sha256
            or observed_byte_count != qualified_source.byte_count
        ):
            raise MageVideoEndpointLaunchError(
                f"current Provider V2 source bytes differ from qualified bundle: {source_name}"
            )


def _codec_cache_startup_report(
    configuration: MageCodecCacheLaunchConfiguration,
    *,
    arguments: argparse.Namespace,
) -> dict[str, object] | None:
    manifest = configuration.manifest
    manifest_path = configuration.manifest_path
    if manifest is None or manifest_path is None:
        return None
    manifest_exact_sha256, manifest_byte_count = _exact_file_sha256(
        manifest_path, description="verified codec cache manifest"
    )
    common: dict[str, object] = {
        "cache_family": configuration.cache_family,
        "manifest_path": str(manifest_path),
        "manifest_version": manifest.manifest_version,
        "manifest_exact_sha256": manifest_exact_sha256,
        "manifest_byte_count": manifest_byte_count,
        "manifest_semantic_sha256": manifest.manifest_semantic_sha256,
        "namespace_identity": manifest.namespace_identity,
        "qualified_cache_root": manifest.qualified_cache_root,
        "entry_count": getattr(manifest, "entry_count", len(manifest.entries)),
        "required": arguments.require_verified_codec_cache,
        "provider_v2_required": arguments.require_provider_v2_cache,
        "traditional_required": arguments.require_traditional_codec_cache,
        "shared_device_guard": {
            "required": (
                configuration.cache_family == PROVIDER_V2_CACHE_FAMILY
                and getattr(manifest.effective_config, "device_concurrency_policy", None)
                == "exclusive-shared-device-v1"
            ),
            "path": (
                str(arguments.shared_device_guard_file.expanduser().resolve())
                if arguments.shared_device_guard_file is not None
                else None
            ),
            "identity_authoritative": False,
        },
    }
    if configuration.cache_family == OBSERVED_V1_CACHE_FAMILY:
        common.update(
            {
                "provider_identity": None,
                "effective_config_identity": None,
                "recipe_identity": {
                    "recipe_version": manifest.recipe.recipe_version,
                    "recipe_semantic_sha256": manifest.recipe.semantic_sha256,
                },
                "qualified_provider_identity": None,
            }
        )
        return common

    if configuration.cache_family == TRADITIONAL_V1_CACHE_FAMILY:
        effective_config = manifest.effective_config
        toolchain = manifest.toolchain
        common.update(
            {
                "provider_identity": {
                    "provider_version": manifest.provider_version,
                    "provider_implementation_sha256": manifest.provider_implementation_sha256,
                    "provider_identity_sha256": manifest.provider_identity_sha256,
                },
                "effective_config_identity": {
                    "effective_config_version": effective_config.effective_config_version,
                    "effective_config_sha256": effective_config.effective_config_sha256,
                    "codec_config_sha256": effective_config.codec_config_sha256,
                    "engine": effective_config.engine,
                    "preprocess_device": effective_config.native_codec_config.get(
                        "preprocess_device"
                    ),
                },
                "recipe_identity": None,
                "qualified_provider_identity": None,
                "toolchain_identity": {
                    "toolchain_version": toolchain.toolchain_version,
                    "toolchain_identity_sha256": toolchain.toolchain_identity_sha256,
                    "package_name": toolchain.package_name,
                    "package_version": toolchain.package_version,
                    "package_artifact_sha256": toolchain.package_artifact_sha256,
                    "executable_sha256": toolchain.executable_sha256,
                    "provider_command_contract_sha256": (
                        toolchain.provider_command_contract_sha256
                    ),
                    "container_platform": toolchain.container_platform,
                },
                "container_image_identity": {
                    "reference": toolchain.container_image_reference,
                    "digest": toolchain.container_image_digest,
                },
            }
        )
        return common

    qualified = configuration.qualified_provider_manifest
    qualified_path = configuration.qualified_provider_manifest_path
    if qualified is None or qualified_path is None:
        raise MageVideoEndpointLaunchError("Provider V2 startup identity is incomplete")
    effective_config = manifest.effective_config
    common.update(
        {
            "provider_identity": {
                "provider_version": manifest.provider_version,
                "provider_implementation_sha256": manifest.provider_implementation_sha256,
            },
            "effective_config_identity": {
                "effective_config_version": effective_config.effective_config_version,
                "effective_config_sha256": effective_config.effective_config_sha256,
            },
            "recipe_identity": {"recipe_version": manifest.recipe_version},
            "qualified_provider_identity": {
                "manifest_path": str(qualified_path),
                "manifest_version": qualified.manifest_version,
                "manifest_semantic_sha256": qualified.manifest_semantic_sha256,
                "bundle_semantic_sha256": qualified.bundle.bundle_semantic_sha256,
                "qualified_checkpoint_manifest_sha256": (
                    qualified.qualified_checkpoint_manifest.manifest_sha256
                ),
                "qualified_model_identifier": qualified.bundle.qualified_model_identifier,
                "qualified_model_revision": qualified.bundle.qualified_model_revision,
                "provider_source_files": [
                    item.model_dump(mode="json") for item in qualified.bundle.provider_files
                ],
            },
        }
    )
    return common


def _exact_file_sha256(
    path: Path,
    *,
    description: str = "warm-up input",
) -> tuple[str, int]:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MageVideoEndpointLaunchError(f"could not read {description}: {resolved}") from error
    if byte_count <= 0:
        raise MageVideoEndpointLaunchError(f"{description} must be nonempty")
    return digest.hexdigest(), byte_count


def _is_within_any_root(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _write_canonical_report(path: Path, payload: object) -> str:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        temporary.replace(resolved)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise MageVideoEndpointLaunchError("could not persist warm-up report") from error
    return exact_bytes_sha256(resolved.read_bytes())


def _run_non_authoritative_warmup(
    *,
    runtime: Any,
    model_identity: Any,
    codec_policy: Any,
    arguments: argparse.Namespace,
    durable_input_roots: Sequence[Path],
    codec_cache_manifest: Any | None,
    codec_cache_binding_type: Any | None = None,
    traditional_codec_cache_binding_type: Any | None = None,
    exact_codec_cache_asset_type: Any | None = None,
) -> dict[str, object] | None:
    values = (
        arguments.warmup_video,
        arguments.warmup_video_sha256,
        arguments.warmup_prompt_file,
        arguments.warmup_report_json,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise MageVideoEndpointLaunchError(
            "warm-up requires --warmup-video, --warmup-video-sha256, "
            "--warmup-prompt-file, and --warmup-report-json together"
        )
    video = arguments.warmup_video.expanduser().resolve()
    if not _is_within_any_root(video, durable_input_roots):
        raise MageVideoEndpointLaunchError("warm-up video is outside durable input roots")
    video_sha256, video_byte_count = _exact_file_sha256(video)
    if video_sha256 != arguments.warmup_video_sha256:
        raise MageVideoEndpointLaunchError("warm-up video SHA-256 does not match")
    prompt_path = arguments.warmup_prompt_file.expanduser().resolve()
    try:
        prompt_bytes = prompt_path.read_bytes()
        prompt = prompt_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise MageVideoEndpointLaunchError("could not read UTF-8 warm-up prompt") from error
    if not prompt.strip():
        raise MageVideoEndpointLaunchError("warm-up prompt must be nonempty")
    codec_cache_binding = None
    if codec_cache_manifest is not None:
        manifest_version = codec_cache_manifest.manifest_version
        if manifest_version == TRADITIONAL_V1_CACHE_MANIFEST_VERSION:
            matching = [
                item
                for item in codec_cache_manifest.entries
                if item.entry.source_path == str(video)
            ]
        else:
            matching = [
                item for item in codec_cache_manifest.entries if item.source_path == str(video)
            ]
        if len(matching) != 1:
            raise MageVideoEndpointLaunchError(
                "warm-up video is absent from the verified codec cache manifest"
            )
        if manifest_version == PROVIDER_V2_CACHE_MANIFEST_VERSION:
            if codec_cache_binding_type is None:
                raise MageVideoEndpointLaunchError(
                    "Provider V2 warm-up requires the internal exact cache binding"
                )
            codec_cache_binding = codec_cache_binding_type(
                source_path=video,
                provider_cache_directory=Path(matching[0].provider_cache_directory),
            )
        elif manifest_version == TRADITIONAL_V1_CACHE_MANIFEST_VERSION:
            if traditional_codec_cache_binding_type is None or exact_codec_cache_asset_type is None:
                raise MageVideoEndpointLaunchError(
                    "traditional warm-up requires the internal exact cache binding types"
                )
            item = matching[0]
            entry = item.entry
            if (
                video_sha256 != entry.source_content_sha256
                or video_byte_count != entry.source_byte_count
            ):
                raise MageVideoEndpointLaunchError(
                    "warm-up source bytes changed after traditional cache admission"
                )
            assets = tuple(
                exact_codec_cache_asset_type(
                    relative_path=asset.relative_path,
                    byte_count=asset.byte_count,
                    sha256=asset.sha256,
                )
                for asset in entry.assets
            )
            codec_cache_binding = traditional_codec_cache_binding_type(
                source_path=video,
                provider_cache_directory=Path(item.provider_cache_directory),
                codec_engine=codec_cache_manifest.effective_config.engine,
                codec_config_sha256=entry.codec_config_sha256,
                checkpoint_manifest_sha256=entry.checkpoint_manifest_sha256,
                codec_policy_sha256=entry.codec_policy_sha256,
                provider_identity_sha256=entry.provider_identity_sha256,
                toolchain_identity_sha256=entry.toolchain_identity_sha256,
                effective_config_sha256=entry.effective_config_sha256,
                entry_semantic_sha256=entry.entry_semantic_sha256,
                asset_set_sha256=entry.asset_set_sha256,
                assets=assets,
            )
    identity_before = runtime.runtime_identity
    load_observation = runtime.load_observation
    started = time.perf_counter()
    generation_kwargs: dict[str, object] = {
        "video_paths": [video],
        "prompt": prompt,
        "max_new_tokens": arguments.warmup_max_new_tokens,
        "codec_config": codec_policy.native_codec_config(),
    }
    if codec_cache_binding is not None:
        generation_kwargs["codec_cache_binding"] = codec_cache_binding
    generated = runtime.generate(**generation_kwargs)
    completed = time.perf_counter()
    if runtime.runtime_identity != identity_before:
        raise MageVideoEndpointLaunchError("runtime identity changed during warm-up")
    output_text = str(generated.output_text)
    if not output_text:
        raise MageVideoEndpointLaunchError("warm-up produced empty output")
    telemetry = getattr(generated, "telemetry", None)
    runtime_telemetry = None
    if telemetry is not None:
        runtime_telemetry = {
            "telemetry_version": telemetry.telemetry_version,
            "processor_seconds": telemetry.processor_seconds,
            "input_materialization_seconds": telemetry.input_materialization_seconds,
            "generate_seconds": telemetry.generate_seconds,
            "decode_seconds": telemetry.decode_seconds,
            "total_request_seconds": telemetry.total_request_seconds,
            "time_to_first_token_seconds": telemetry.time_to_first_token_seconds,
            "output_tokens_per_second": telemetry.output_tokens_per_second,
        }
    report = {
        "report_version": "mage-video-non-authoritative-warmup-v1",
        "authority": "NON_AUTHORITATIVE_DISCARDED",
        "model_identity": model_identity.model_dump(mode="json"),
        "codec_policy_sha256": semantic_sha256(codec_policy.model_dump(mode="json")),
        "video_path": str(video),
        "video_sha256": video_sha256,
        "video_byte_count": video_byte_count,
        "prompt_sha256": exact_bytes_sha256(prompt_bytes),
        "max_new_tokens": arguments.warmup_max_new_tokens,
        "actual_output_tokens": int(generated.output_tokens),
        "output_text_sha256": exact_bytes_sha256(output_text.encode("utf-8")),
        "model_load_seconds": float(load_observation.load_seconds),
        "model_load_included_in_warmup_wall": False,
        "warmup_wall_seconds": float(completed - started),
        "runtime_telemetry": runtime_telemetry,
        "codec_cache_manifest_semantic_sha256": (
            None if codec_cache_manifest is None else codec_cache_manifest.manifest_semantic_sha256
        ),
        "codec_cache_family": (
            None
            if codec_cache_manifest is None
            else {
                OBSERVED_V1_CACHE_MANIFEST_VERSION: OBSERVED_V1_CACHE_FAMILY,
                PROVIDER_V2_CACHE_MANIFEST_VERSION: PROVIDER_V2_CACHE_FAMILY,
                TRADITIONAL_V1_CACHE_MANIFEST_VERSION: TRADITIONAL_V1_CACHE_FAMILY,
            }.get(codec_cache_manifest.manifest_version)
        ),
    }
    report_path = arguments.warmup_report_json.expanduser().resolve()
    report_exact_sha256 = _write_canonical_report(report_path, report)
    return {
        "performed": True,
        "authority": report["authority"],
        "report_path": str(report_path),
        "report_exact_sha256": report_exact_sha256,
        "warmup_wall_seconds": report["warmup_wall_seconds"],
        "model_load_seconds": report["model_load_seconds"],
        "model_load_included_in_warmup_wall": False,
        "actual_output_tokens": report["actual_output_tokens"],
        "output_text_sha256": report["output_text_sha256"],
    }


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
        bind_security = _validate_bind_security(arguments)
        endpoint_module = importlib.import_module("robata.inference.mage_video_endpoint")
        cache_module = importlib.import_module("robata.inference.mage_codec_cache")
        cache_v2_module = importlib.import_module("robata.inference.mage_codec_cache_v2")
        traditional_cache_module = importlib.import_module(
            "robata.inference.mage_traditional_codec_cache"
        )
        qualified_provider_module = importlib.import_module(
            "robata.inference.mage_dcvc_qualified_provider"
        )
        preparation_worker_module = importlib.import_module(
            "robata.inference.mage_dcvc_preparation_worker"
        )
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
        codec_cache_configuration = _codec_cache_configuration(
            cache_module=cache_module,
            cache_v2_module=cache_v2_module,
            qualified_provider_module=qualified_provider_module,
            preparation_worker_module=preparation_worker_module,
            traditional_cache_module=traditional_cache_module,
            endpoint_module=endpoint_module,
            arguments=arguments,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_manifest=checkpoint_manifest,
            codec_policy=policy,
        )
        codec_cache_root = codec_cache_configuration.cache_root
        codec_cache_admission = codec_cache_configuration.admission
        codec_cache_manifest = codec_cache_configuration.manifest
        codec_dependency_report = _require_launch_codec_dependencies(
            runtime_module=runtime_module,
            codec_policy=policy,
            model_directory=arguments.model_dir.expanduser().resolve(),
            cache_family=codec_cache_configuration.cache_family,
        )
        runtime, canonical_profile = _create_runtime(
            runtime_module=runtime_module,
            model_directory=arguments.model_dir.expanduser().resolve(),
            offload_directory=state_root / "model-offload",
            requested_profile=arguments.load_profile,
            codec_cache_root=codec_cache_root,
            shared_device_guard_file=(
                arguments.shared_device_guard_file.expanduser().resolve()
                if arguments.shared_device_guard_file is not None
                else None
            ),
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
            codec_cache_admission=codec_cache_admission,
        )
        service.start()
        warmup = _run_non_authoritative_warmup(
            runtime=runtime,
            model_identity=model_identity,
            codec_policy=policy,
            arguments=arguments,
            durable_input_roots=input_roots,
            codec_cache_manifest=codec_cache_manifest,
            codec_cache_binding_type=getattr(endpoint_module, "MageVideoCodecCacheBinding", None),
            traditional_codec_cache_binding_type=getattr(
                runtime_module, "MageVideoTraditionalCodecCacheBinding", None
            ),
            exact_codec_cache_asset_type=getattr(
                runtime_module, "MageVideoExactCodecCacheAsset", None
            ),
        )
        health = service.health()
        _print_json(
            {
                "ok": True,
                "status": health.status,
                "endpoint": {"host": arguments.host, "port": arguments.port},
                "bind_security": bind_security,
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
                "codec_dependency_report": codec_dependency_report,
                "state_dir": str(state_root),
                "idempotency_state_path": str(idempotency_path),
                "result_artifact_dir": str(result_root),
                "generation_telemetry_jsonl": (
                    str(generation_telemetry_path)
                    if generation_telemetry_path is not None
                    else None
                ),
                "durable_input_roots": [str(item) for item in input_roots],
                "shared_device_guard_file": (
                    str(runtime.shared_device_guard_file)
                    if getattr(runtime, "shared_device_guard_file", None) is not None
                    else None
                ),
                "warmup": warmup,
                "verified_codec_cache": _codec_cache_startup_report(
                    codec_cache_configuration,
                    arguments=arguments,
                ),
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
