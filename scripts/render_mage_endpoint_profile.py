"""Validate and render the target-only dual-H100 Mage Provider V2 deployment bundle.

The renderer never starts CUDA, Mage, or DCVC and never upgrades target configuration
into qualification evidence. It emits one Provider V2 pre-admission command and one
single-generation endpoint command with explicit container-visible CUDA-role isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

PROFILE_VERSION = "mage-endpoint-deployment-profile-v2"
COMMAND_VERSION = "mage-endpoint-rendered-command-bundle-v2"
TARGET_PROFILE_ID = "mage-h100-bf16-single-worker-v1"
TARGET_EVIDENCE_CLASS = "TARGET_CONFIGURATION_UNVALIDATED"

_REQUIRED_BINDINGS = (
    "codec_cuda_selector",
    "decoder_cuda_selector",
    "qualified_model_dir",
    "qualified_model_revision",
    "qualified_provider_manifest",
    "qualification_manifest_sha256",
    "checkpoint_manifest_path",
    "checkpoint_manifest_sha256",
    "provider_state_root",
    "cache_base_root",
    "codec_cache_manifest",
    "provider_prewarm_report_json",
    "prewarm_video",
    "state_dir",
    "durable_input_root",
    "generation_telemetry_jsonl",
    "warmup_video",
    "warmup_video_sha256",
    "warmup_prompt_file",
    "warmup_report_json",
)

_EXPECTED_TARGET: dict[str, object] = {
    "operating_system": "linux",
    "accelerator": "NVIDIA H100",
    "container_visible_accelerator_count": 2,
    "codec_provider_outer_cuda_selector": "0",
    "mage_decoder_outer_cuda_selector": "1",
    "device_uuid_attestation_required": True,
}

_EXPECTED_PREWARM: dict[str, object] = {
    "preparation_device": "cuda",
    "generation_device": "cuda:1",
    "device_concurrency_policy": "separate-device-v1",
    "resolution_policy": "max-side-448-target-candidate",
    "bounded_resolution_selected": True,
    "bounded_resolution_production_qualified": False,
    "max_side": 448,
    "target_canvas": 8,
    "group_size": 8,
    "images_per_group": 1,
    "patch_size": 16,
    "max_pixels": 65_536,
    "min_group_frames": 8,
    "max_group_frames": 128,
    "sampled_frame_count": 64,
    "sequence_length_frames": 0,
    "sequence_length_frames_is_compute_cap": False,
    "canvas_token_side": None,
    "encoded_frame_extent": "through-last-sampled-frame",
    "neural_qp": 42,
    "neural_reset_interval": 64,
    "neural_intra_period": -1,
    "readiness_coverage_bins": 3,
    "readiness_delta_ratio": 0.05,
    "bitcost_percentile": 99,
    "decode_backsearch_max": 16,
    "timeout_seconds": 7_200,
    "worker_process_count": 1,
    "worker_lifetime": "one-resident-worker-per-prewarm-invocation",
    "replay_authority": "AUTHORITATIVE_EXACT_CACHE",
}

_EXPECTED_ENDPOINT: dict[str, object] = {
    "load_profile": "production-native-bf16",
    "model_identifier": "Mage-VL-Robata-DCVC-V2",
    "codec_mode": "neural",
    "codec_target_canvas": 8,
    "codec_group_size": 8,
    "codec_images_per_group": 1,
    "codec_patch_size": 16,
    "codec_max_pixels": 65_536,
    "codec_min_group_frames": 8,
    "codec_max_group_frames": 128,
    "codec_timeout_seconds": 7_200,
    "preprocess_device": "cuda",
    "neural_qp": 42,
    "neural_reset_interval": 64,
    "neural_intra_period": -1,
    "neural_max_side": 448,
    "neural_sequence_length_frames": 0,
    "neural_canvas_token_side": None,
    "neural_readiness_coverage_bins": 3,
    "neural_readiness_delta_ratio": 0.05,
    "neural_bitcost_percentile": 99,
    "neural_decode_backsearch_max": 16,
    "warmup_max_new_tokens": 32,
    "host": "0.0.0.0",
    "network_boundary": "controlled-private-network",
    "endpoint_authentication": "NOT_PROVIDED_BY_LAUNCHER",
    "unauthenticated_public_bind_acknowledged": False,
    "port": 8_102,
    "log_level": "info",
    "require_verified_codec_cache": True,
    "require_provider_v2_cache": True,
    "qualified_provider_manifest_required": True,
    "warmup_authority": "NON_AUTHORITATIVE_DISCARDED",
}

_EXPECTED_RUNTIME_CONSTRAINTS: dict[str, object] = {
    "uvicorn_workers": 1,
    "provider_worker_processes": 1,
    "generation_concurrency": 1,
    "native_batch_enabled": False,
    "native_batch_max_size": 1,
    "camera_count_per_request": 1,
    "generation_lane_count": 1,
    "device_concurrency_policy": "separate-device-v1",
    "attention_implementation": "runtime-default",
}

_REQUIRED_EXTERNAL_EVIDENCE = (
    "linux_container_dependency_and_provider_v2_readiness",
    "dual_h100_role_mapping_and_process_isolation",
    "qualified_provider_source_checkpoint_and_manifest_identity",
    "native_bf16_decoder_residency_without_cpu_or_disk_offload",
    "provider_v2_cache_rebuilt_on_target_environment",
    "one_resident_codec_worker_and_exact_hit_replay",
    "warm_and_cold_40_second_five_segment_benchmark",
    "representative_output_and_downstream_semantic_quality",
    "restart_cache_loss_and_authoritative_replay_recovery",
    "soak_capacity_thermal_and_cost_qualification",
)

_PREWARM_ARGUMENTS = (
    ("max_side", "--max-side"),
    ("target_canvas", "--target-canvas"),
    ("group_size", "--group-size"),
    ("images_per_group", "--images-per-group"),
    ("max_pixels", "--max-pixels"),
    ("min_group_frames", "--min-group-frames"),
    ("max_group_frames", "--max-group-frames"),
    ("neural_qp", "--neural-qp"),
    ("neural_reset_interval", "--neural-reset-interval"),
    ("neural_intra_period", "--neural-intra-period"),
    ("readiness_coverage_bins", "--readiness-coverage-bins"),
    ("readiness_delta_ratio", "--readiness-delta-ratio"),
    ("bitcost_percentile", "--bitcost-percentile"),
    ("decode_backsearch_max", "--decode-backsearch-max"),
    ("timeout_seconds", "--timeout-seconds"),
)

_ENDPOINT_ARGUMENTS = (
    ("load_profile", "--load-profile"),
    ("model_identifier", "--model-identifier"),
    ("codec_mode", "--codec-mode"),
    ("codec_target_canvas", "--codec-target-canvas"),
    ("codec_group_size", "--codec-group-size"),
    ("codec_images_per_group", "--codec-images-per-group"),
    ("codec_patch_size", "--codec-patch-size"),
    ("codec_max_pixels", "--codec-max-pixels"),
    ("codec_min_group_frames", "--codec-min-group-frames"),
    ("codec_max_group_frames", "--codec-max-group-frames"),
    ("codec_timeout_seconds", "--codec-timeout-seconds"),
    ("preprocess_device", "--preprocess-device"),
    ("neural_qp", "--neural-qp"),
    ("neural_reset_interval", "--neural-reset-interval"),
    ("neural_intra_period", "--neural-intra-period"),
    ("neural_max_side", "--neural-max-side"),
    ("neural_sequence_length_frames", "--neural-sequence-length-frames"),
    ("neural_readiness_coverage_bins", "--neural-readiness-coverage-bins"),
    ("neural_readiness_delta_ratio", "--neural-readiness-delta-ratio"),
    ("neural_bitcost_percentile", "--neural-bitcost-percentile"),
    ("neural_decode_backsearch_max", "--neural-decode-backsearch-max"),
    ("warmup_max_new_tokens", "--warmup-max-new-tokens"),
    ("host", "--host"),
    ("network_boundary", "--network-boundary"),
    ("port", "--port"),
    ("log_level", "--log-level"),
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CUDA_DEVICE = re.compile(r"[0-9]+\Z")


class MageEndpointProfileError(ValueError):
    """A deployment profile or binding is unsafe or ambiguous."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MageEndpointProfileError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], *, expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise MageEndpointProfileError(
            f"{label} keys do not match profile v2; missing={missing}, unknown={unknown}"
        )


def _require_exact(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise MageEndpointProfileError(f"{label} must be {expected!r}")


def _validate_exact_mapping(
    value: object,
    *,
    expected: Mapping[str, object],
    label: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, label=label)
    _exact_keys(mapping, expected=set(expected), label=label)
    for key, expected_value in expected.items():
        _require_exact(mapping[key], expected_value, label=f"{label}.{key}")
    return mapping


def load_and_validate_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile_path = path.expanduser().resolve()
    try:
        raw = profile_path.read_bytes()
    except OSError as error:
        raise MageEndpointProfileError(f"could not read profile: {error}") from error
    try:
        document: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MageEndpointProfileError(f"profile is not valid UTF-8 JSON: {error}") from error
    profile = dict(_mapping(document, label="profile"))
    _exact_keys(
        profile,
        expected={
            "profile_version",
            "profile_id",
            "as_of_date",
            "evidence_class",
            "target",
            "provider_v2_prewarm",
            "endpoint_launcher",
            "runtime_constraints",
            "required_bindings",
            "qualification",
        },
        label="profile",
    )
    _require_exact(profile["profile_version"], PROFILE_VERSION, label="profile_version")
    _require_exact(profile["profile_id"], TARGET_PROFILE_ID, label="profile_id")
    _require_exact(profile["as_of_date"], "2026-08-09", label="as_of_date")
    _require_exact(profile["evidence_class"], TARGET_EVIDENCE_CLASS, label="evidence_class")

    target = _validate_exact_mapping(profile["target"], expected=_EXPECTED_TARGET, label="target")
    prewarm = _validate_exact_mapping(
        profile["provider_v2_prewarm"], expected=_EXPECTED_PREWARM, label="provider_v2_prewarm"
    )
    endpoint = _validate_exact_mapping(
        profile["endpoint_launcher"], expected=_EXPECTED_ENDPOINT, label="endpoint_launcher"
    )
    runtime = _validate_exact_mapping(
        profile["runtime_constraints"],
        expected=_EXPECTED_RUNTIME_CONSTRAINTS,
        label="runtime_constraints",
    )

    if target["codec_provider_outer_cuda_selector"] == target["mage_decoder_outer_cuda_selector"]:
        raise MageEndpointProfileError("codec and decoder outer CUDA selectors must be distinct")
    if prewarm["device_concurrency_policy"] != runtime["device_concurrency_policy"]:
        raise MageEndpointProfileError(
            "device concurrency policy must match across profile sections"
        )
    if prewarm["max_side"] != endpoint["neural_max_side"]:
        raise MageEndpointProfileError("Provider V2 max_side must match endpoint neural_max_side")
    if (
        prewarm["bounded_resolution_selected"] is not True
        or prewarm["bounded_resolution_production_qualified"] is not False
        or prewarm["max_side"] != 448
    ):
        raise MageEndpointProfileError(
            "the H100 target must remain the unqualified max_side=448 candidate"
        )
    if prewarm["sequence_length_frames"] != endpoint["neural_sequence_length_frames"]:
        raise MageEndpointProfileError("sequence_length_frames must match across profile sections")
    if prewarm["sequence_length_frames_is_compute_cap"] is not False:
        raise MageEndpointProfileError(
            "sequence_length_frames must not be represented as a compute cap"
        )
    if prewarm["canvas_token_side"] is not None or endpoint["neural_canvas_token_side"] is not None:
        raise MageEndpointProfileError("Provider V2 requires canvas_token_side=null")
    if prewarm["encoded_frame_extent"] != "through-last-sampled-frame":
        raise MageEndpointProfileError("Provider V2 must encode through the last sampled frame")

    required_bindings = profile["required_bindings"]
    if not isinstance(required_bindings, list) or tuple(required_bindings) != _REQUIRED_BINDINGS:
        raise MageEndpointProfileError(f"required_bindings must equal {list(_REQUIRED_BINDINGS)!r}")

    qualification = _mapping(profile["qualification"], label="qualification")
    _exact_keys(
        qualification,
        expected={"locally_validated", "production_eligible", "required_external_evidence"},
        label="qualification",
    )
    _require_exact(
        qualification["locally_validated"], False, label="qualification.locally_validated"
    )
    _require_exact(
        qualification["production_eligible"], False, label="qualification.production_eligible"
    )
    evidence = qualification["required_external_evidence"]
    if not isinstance(evidence, list) or tuple(evidence) != _REQUIRED_EXTERNAL_EVIDENCE:
        raise MageEndpointProfileError(
            "qualification.required_external_evidence does not match profile v2"
        )

    semantic_sha256 = hashlib.sha256(_canonical_json_bytes(profile)).hexdigest()
    return profile, semantic_sha256


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _absolute_posix_path(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or _contains_control_character(value)
    ):
        raise MageEndpointProfileError(f"{label} must be an absolute Linux path")
    normalized = str(PurePosixPath(value))
    if (
        "\\" in value
        or ".." in PurePosixPath(value).parts
        or value.startswith("//")
        or normalized != value
    ):
        raise MageEndpointProfileError(f"{label} must be a normalized absolute Linux path")
    return value


def _nonempty(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or _contains_control_character(value):
        raise MageEndpointProfileError(f"{label} must be a non-empty string")
    return value


def _outer_cuda_selector(value: str, *, label: str) -> str:
    parsed = _nonempty(value, label=label)
    if _CUDA_DEVICE.fullmatch(parsed) is None:
        raise MageEndpointProfileError(f"{label} must name one numeric outer CUDA-visible ordinal")
    return parsed


def _sha256(value: str, *, label: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise MageEndpointProfileError(f"{label} is not a lowercase SHA-256")
    return value


def _is_within(path: str, roots: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    return any(
        candidate == PurePosixPath(root) or PurePosixPath(root) in candidate.parents
        for root in roots
    )


def _unique_paths(values: Sequence[str], *, label: str) -> list[str]:
    paths = [_absolute_posix_path(value, label=label) for value in values]
    if not paths:
        raise MageEndpointProfileError(f"at least one {label} is required")
    if len(set(paths)) != len(paths):
        raise MageEndpointProfileError(f"{label} values must be unique")
    return paths


def _strictly_within(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return candidate != parent and parent in candidate.parents


def _paths_overlap(left: str, right: str) -> bool:
    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def _require_path_disjoint(
    left_label: str,
    left: str,
    right_label: str,
    right: str,
) -> None:
    if _paths_overlap(left, right):
        raise MageEndpointProfileError(
            f"{left_label} ({left}) and {right_label} ({right}) must be path-disjoint"
        )


def _require_pairwise_path_disjoint(paths: Mapping[str, str], *, category: str) -> None:
    items = list(paths.items())
    for index, (left_label, left) in enumerate(items):
        for right_label, right in items[index + 1 :]:
            if _paths_overlap(left, right):
                raise MageEndpointProfileError(
                    f"{category} {left_label} ({left}) and {right_label} ({right}) "
                    "must be path-disjoint"
                )


def _validate_output_path_topology(
    bound_paths: Mapping[str, str],
    *,
    durable_roots: Sequence[str],
    prewarm_videos: Sequence[str],
) -> None:
    output_roots = {
        "provider_state_root": bound_paths["provider_state_root"],
        "cache_base_root": bound_paths["cache_base_root"],
        "state_dir": bound_paths["state_dir"],
    }
    output_files = {
        "codec_cache_manifest": bound_paths["codec_cache_manifest"],
        "provider_prewarm_report_json": bound_paths["provider_prewarm_report_json"],
        "generation_telemetry_jsonl": bound_paths["generation_telemetry_jsonl"],
        "warmup_report_json": bound_paths["warmup_report_json"],
    }
    required_output_parents = {
        "codec_cache_manifest": "provider_state_root",
        "provider_prewarm_report_json": "provider_state_root",
        "generation_telemetry_jsonl": "state_dir",
        "warmup_report_json": "state_dir",
    }
    for output_label, root_label in required_output_parents.items():
        output_path = output_files[output_label]
        root_path = output_roots[root_label]
        if not _strictly_within(output_path, root_path):
            raise MageEndpointProfileError(
                f"{output_label} ({output_path}) must be strictly beneath "
                f"{root_label} ({root_path})"
            )

    _require_pairwise_path_disjoint(output_roots, category="output roots")
    _require_pairwise_path_disjoint(output_files, category="output files")

    protected_input_roots = {
        "qualified_model_dir": bound_paths["qualified_model_dir"],
        **{f"durable_input_root[{index}]": root for index, root in enumerate(durable_roots)},
    }
    protected_input_files = {
        "qualified_provider_manifest": bound_paths["qualified_provider_manifest"],
        "checkpoint_manifest_path": bound_paths["checkpoint_manifest_path"],
        "warmup_prompt_file": bound_paths["warmup_prompt_file"],
        **{f"prewarm_video[{index}]": video for index, video in enumerate(prewarm_videos)},
    }
    for output_label, output_file in output_files.items():
        for input_label, input_path in protected_input_files.items():
            _require_path_disjoint(
                f"output file {output_label}",
                output_file,
                f"protected input {input_label}",
                input_path,
            )
    protected_inputs = {**protected_input_roots, **protected_input_files}
    for output_label, output_root in output_roots.items():
        for input_label, input_path in protected_inputs.items():
            _require_path_disjoint(
                f"output root {output_label}",
                output_root,
                f"protected input {input_label}",
                input_path,
            )
    for output_label, output_file in output_files.items():
        for input_label, input_path in protected_input_roots.items():
            _require_path_disjoint(
                f"output file {output_label}",
                output_file,
                f"protected input {input_label}",
                input_path,
            )


def render_command(profile: Mapping[str, Any], arguments: argparse.Namespace) -> dict[str, Any]:
    """Render the pre-admission and endpoint commands without running either process."""

    target = _mapping(profile["target"], label="target")
    codec_selector = _outer_cuda_selector(
        arguments.codec_cuda_selector, label="codec_cuda_selector"
    )
    decoder_selector = _outer_cuda_selector(
        arguments.decoder_cuda_selector, label="decoder_cuda_selector"
    )
    if codec_selector == decoder_selector:
        raise MageEndpointProfileError(
            "codec_cuda_selector and decoder_cuda_selector must be distinct"
        )
    _require_exact(
        codec_selector,
        target["codec_provider_outer_cuda_selector"],
        label="codec_cuda_selector",
    )
    _require_exact(
        decoder_selector,
        target["mage_decoder_outer_cuda_selector"],
        label="decoder_cuda_selector",
    )

    qualification_sha256 = _sha256(
        arguments.qualification_manifest_sha256,
        label="qualification_manifest_sha256",
    )
    checkpoint_sha256 = _sha256(
        arguments.checkpoint_manifest_sha256,
        label="checkpoint_manifest_sha256",
    )
    warmup_sha256 = _sha256(arguments.warmup_video_sha256, label="warmup_video_sha256")

    bound_paths = {
        "qualified_model_dir": _absolute_posix_path(
            arguments.qualified_model_dir, label="qualified_model_dir"
        ),
        "qualified_provider_manifest": _absolute_posix_path(
            arguments.qualified_provider_manifest, label="qualified_provider_manifest"
        ),
        "checkpoint_manifest_path": _absolute_posix_path(
            arguments.checkpoint_manifest_path, label="checkpoint_manifest_path"
        ),
        "provider_state_root": _absolute_posix_path(
            arguments.provider_state_root, label="provider_state_root"
        ),
        "cache_base_root": _absolute_posix_path(arguments.cache_base_root, label="cache_base_root"),
        "codec_cache_manifest": _absolute_posix_path(
            arguments.codec_cache_manifest, label="codec_cache_manifest"
        ),
        "provider_prewarm_report_json": _absolute_posix_path(
            arguments.provider_prewarm_report_json, label="provider_prewarm_report_json"
        ),
        "state_dir": _absolute_posix_path(arguments.state_dir, label="state_dir"),
        "generation_telemetry_jsonl": _absolute_posix_path(
            arguments.generation_telemetry_jsonl, label="generation_telemetry_jsonl"
        ),
        "warmup_video": _absolute_posix_path(arguments.warmup_video, label="warmup_video"),
        "warmup_prompt_file": _absolute_posix_path(
            arguments.warmup_prompt_file, label="warmup_prompt_file"
        ),
        "warmup_report_json": _absolute_posix_path(
            arguments.warmup_report_json, label="warmup_report_json"
        ),
    }
    durable_roots = _unique_paths(arguments.durable_input_root, label="durable_input_root")
    prewarm_videos = _unique_paths(arguments.prewarm_video, label="prewarm_video")
    if bound_paths["warmup_video"] not in prewarm_videos:
        raise MageEndpointProfileError("warmup_video must also be listed as a prewarm_video")
    if any(not _is_within(video, durable_roots) for video in prewarm_videos):
        raise MageEndpointProfileError("every prewarm_video must be beneath a durable_input_root")
    _validate_output_path_topology(
        bound_paths,
        durable_roots=durable_roots,
        prewarm_videos=prewarm_videos,
    )

    qualified_model_revision = _nonempty(
        arguments.qualified_model_revision, label="qualified_model_revision"
    )
    prewarm_profile = _mapping(profile["provider_v2_prewarm"], label="provider_v2_prewarm")
    endpoint_profile = _mapping(profile["endpoint_launcher"], label="endpoint_launcher")

    prewarm_argv = [
        "python",
        "scripts/prewarm_local_mage_dcvc_provider_v2.py",
        "--model-dir",
        bound_paths["qualified_model_dir"],
        "--qualified-provider-manifest",
        bound_paths["qualified_provider_manifest"],
        "--qualification-manifest-sha256",
        qualification_sha256,
        "--checkpoint-manifest-path",
        bound_paths["checkpoint_manifest_path"],
        "--checkpoint-manifest-sha256",
        checkpoint_sha256,
        "--provider-state-root",
        bound_paths["provider_state_root"],
        "--cache-base-root",
        bound_paths["cache_base_root"],
        "--manifest-output",
        bound_paths["codec_cache_manifest"],
        "--report-output",
        bound_paths["provider_prewarm_report_json"],
        "--generation-device",
        str(prewarm_profile["generation_device"]),
        "--preparation-device",
        str(prewarm_profile["preparation_device"]),
    ]
    for key, flag in _PREWARM_ARGUMENTS:
        prewarm_argv.extend((flag, str(prewarm_profile[key])))
    for video in prewarm_videos:
        prewarm_argv.extend(("--video", video))

    effective_host = arguments.host if arguments.host is not None else str(endpoint_profile["host"])
    _nonempty(effective_host, label="host")
    effective_port = arguments.port if arguments.port is not None else endpoint_profile["port"]
    if type(effective_port) is not int or not 1 <= effective_port <= 65_535:
        raise MageEndpointProfileError("port must be an integer from 1 through 65535")

    endpoint_argv = [
        "python",
        "scripts/run_mage_video_endpoint.py",
        "--model-dir",
        bound_paths["qualified_model_dir"],
        "--model-revision",
        qualified_model_revision,
        "--checkpoint-manifest-path",
        bound_paths["checkpoint_manifest_path"],
        "--checkpoint-manifest-sha256",
        checkpoint_sha256,
    ]
    for key, flag in _ENDPOINT_ARGUMENTS:
        value: object = endpoint_profile[key]
        if key == "host":
            value = effective_host
        elif key == "port":
            value = effective_port
        endpoint_argv.extend((flag, str(value)))
    endpoint_argv.extend(
        (
            "--state-dir",
            bound_paths["state_dir"],
            "--generation-telemetry-jsonl",
            bound_paths["generation_telemetry_jsonl"],
            "--codec-cache-manifest",
            bound_paths["codec_cache_manifest"],
            "--qualified-provider-manifest",
            bound_paths["qualified_provider_manifest"],
            "--require-verified-codec-cache",
            "--require-provider-v2-cache",
            "--warmup-video",
            bound_paths["warmup_video"],
            "--warmup-video-sha256",
            warmup_sha256,
            "--warmup-prompt-file",
            bound_paths["warmup_prompt_file"],
            "--warmup-report-json",
            bound_paths["warmup_report_json"],
        )
    )
    for root in durable_roots:
        endpoint_argv.extend(("--durable-input-root", root))
    if arguments.readiness_only:
        endpoint_argv.append("--readiness-only")

    return {
        "command_version": COMMAND_VERSION,
        "profile_version": profile["profile_version"],
        "profile_id": profile["profile_id"],
        "profile_semantic_sha256": hashlib.sha256(_canonical_json_bytes(profile)).hexdigest(),
        "evidence_class": profile["evidence_class"],
        "production_eligible": False,
        "device_concurrency_policy": "separate-device-v1",
        "device_mapping_authority": "OUTER_SELECTORS_UNATTESTED",
        "device_uuid_attestation_required": True,
        "bind_security": {
            "host": effective_host,
            "network_boundary": endpoint_profile["network_boundary"],
            "endpoint_authentication": endpoint_profile["endpoint_authentication"],
            "unauthenticated_public_bind_acknowledged": endpoint_profile[
                "unauthenticated_public_bind_acknowledged"
            ],
        },
        "resolution_policy": "max-side-448-target-candidate",
        "bounded_resolution_selected": True,
        "bounded_resolution_production_qualified": False,
        "sequence_length_frames": 0,
        "sequence_length_frames_is_compute_cap": False,
        "canvas_token_side": None,
        "encoded_frame_extent": "through-last-sampled-frame",
        "commands": {
            "provider_v2_prewarm": {
                "environment": {"CUDA_VISIBLE_DEVICES": f"{codec_selector},{decoder_selector}"},
                "codec_outer_cuda_selector": codec_selector,
                "codec_process_logical_device": "cuda:0",
                "decoder_prewarm_logical_device": "cuda:1",
                "argv": prewarm_argv,
            },
            "endpoint": {
                "environment": {"CUDA_VISIBLE_DEVICES": decoder_selector},
                "decoder_outer_cuda_selector": decoder_selector,
                "decoder_process_logical_device": "cuda:0",
                "argv": endpoint_argv,
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--codec-cuda-selector", required=True)
    parser.add_argument("--decoder-cuda-selector", required=True)
    parser.add_argument("--qualified-model-dir", required=True)
    parser.add_argument("--qualified-model-revision", required=True)
    parser.add_argument("--qualified-provider-manifest", required=True)
    parser.add_argument("--qualification-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-manifest-path", required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--provider-state-root", required=True)
    parser.add_argument("--cache-base-root", required=True)
    parser.add_argument("--codec-cache-manifest", required=True)
    parser.add_argument("--provider-prewarm-report-json", required=True)
    parser.add_argument("--prewarm-video", action="append", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--durable-input-root", action="append", required=True)
    parser.add_argument("--generation-telemetry-jsonl", required=True)
    parser.add_argument("--warmup-video", required=True)
    parser.add_argument("--warmup-video-sha256", required=True)
    parser.add_argument("--warmup-prompt-file", required=True)
    parser.add_argument("--warmup-report-json", required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--render-phase", choices=("all", "prewarm", "endpoint"), default="all")
    parser.add_argument("--output-format", choices=("json", "posix-shell"), default="json")
    return parser


def _posix_shell(bundle: Mapping[str, Any], *, phase: str) -> str:
    commands = _mapping(bundle["commands"], label="commands")

    def command_text(name: str, *, replace_process: bool) -> str:
        command = _mapping(commands[name], label=name)
        environment = _mapping(command["environment"], label=f"{name}.environment")
        prefix = "exec " if replace_process else ""
        return (
            f"{prefix}env CUDA_VISIBLE_DEVICES="
            f"{shlex.quote(str(environment['CUDA_VISIBLE_DEVICES']))} "
            f"{shlex.join(command['argv'])}"
        )

    if phase == "prewarm":
        return command_text("provider_v2_prewarm", replace_process=True)
    if phase == "endpoint":
        return command_text("endpoint", replace_process=True)
    return "\n".join(
        (
            "set -euo pipefail",
            command_text("provider_v2_prewarm", replace_process=False),
            command_text("endpoint", replace_process=True),
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        profile, profile_semantic_sha256 = load_and_validate_profile(arguments.profile)
        rendered = render_command(profile, arguments)
        if rendered["profile_semantic_sha256"] != profile_semantic_sha256:
            raise MageEndpointProfileError("profile digest changed during rendering")
        if arguments.output_format == "posix-shell":
            print(_posix_shell(rendered, phase=arguments.render_phase))
        else:
            payload = dict(rendered)
            payload["selected_phase"] = arguments.render_phase
            print(_canonical_json_bytes(payload).decode("utf-8"))
    except MageEndpointProfileError as error:
        print(
            _canonical_json_bytes(
                {"ok": False, "code": "MAGE_ENDPOINT_PROFILE_INVALID", "detail": str(error)}
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
