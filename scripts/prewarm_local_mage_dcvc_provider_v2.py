#!/usr/bin/env python3
"""Prewarm a qualified Mage DCVC Provider V2 cache with one resident worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.inference.mage_checkpoint_identity import load_mage_checkpoint_manifest  # noqa: E402
from robata.inference.mage_dcvc_preparation_worker import (  # noqa: E402
    build_mage_dcvc_effective_config,
)
from robata.inference.mage_dcvc_prewarm import (  # noqa: E402
    MageDcvcPrewarmError,
    prewarm_mage_dcvc_provider_v2,
)
from robata.inference.mage_dcvc_qualified_provider import (  # noqa: E402
    load_mage_dcvc_qualified_provider_manifest,
)
from robata.inference.mage_video_endpoint import (  # noqa: E402
    MageVideoCodecPolicy,
    MageVideoNeuralCodecParameters,
)

_LOCAL_QUALIFIED_MAX_SIDE_DEFAULT = 448

_DEFAULT_PROVIDER_SOURCES = (
    ROOT / "src" / "robata" / "inference" / "device_execution_guard.py",
    ROOT / "src" / "robata" / "inference" / "mage_dcvc_preparation_protocol.py",
    ROOT / "src" / "robata" / "inference" / "mage_dcvc_preparation_worker.py",
)


class MageDcvcPrewarmCliError(RuntimeError):
    """CLI inputs could not be bound to one qualified Provider V2 run."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _device(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "cpu" or lowered == "cuda":
        return lowered
    if lowered.startswith("cuda:") and lowered[5:].isdigit():
        return lowered
    raise argparse.ArgumentTypeError("must be cpu, cuda, or cuda:<index>")


def _preparation_device(value: str) -> str:
    lowered = value.strip().lower()
    if lowered not in {"cpu", "cuda"}:
        raise argparse.ArgumentTypeError("must be cpu or cuda")
    return lowered


def _normalise_device(value: str) -> str:
    value = value.lower()
    return "cuda:0" if value == "cuda" else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qualified-provider-manifest", type=Path, required=True)
    parser.add_argument("--qualification-manifest-sha256", type=_sha256, default=None)
    parser.add_argument("--checkpoint-manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", type=_sha256, default=None)
    parser.add_argument("--provider-state-root", type=Path, required=True)
    parser.add_argument("--cache-base-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--generation-device", type=_device, required=True)
    parser.add_argument("--preparation-device", type=_preparation_device, required=True)
    parser.add_argument("--shared-device-guard-file", type=Path, default=None)
    parser.add_argument(
        "--max-side",
        type=_nonnegative_int,
        default=_LOCAL_QUALIFIED_MAX_SIDE_DEFAULT,
        help=(
            "effective Provider V2 bound; locally qualified default is 448; "
            "pass 0 explicitly for the full-resolution control or rollback profile"
        ),
    )
    parser.add_argument("--target-canvas", type=_positive_int, default=8)
    parser.add_argument("--group-size", type=_positive_int, default=8)
    parser.add_argument("--images-per-group", type=_positive_int, default=1)
    parser.add_argument("--max-pixels", type=_positive_int, default=65_536)
    parser.add_argument("--min-group-frames", type=_positive_int, default=8)
    parser.add_argument("--max-group-frames", type=_positive_int, default=128)
    parser.add_argument("--neural-qp", type=int, default=42)
    parser.add_argument("--neural-reset-interval", type=_positive_int, default=64)
    parser.add_argument("--neural-intra-period", type=int, default=-1)
    parser.add_argument("--readiness-coverage-bins", type=_positive_int, default=3)
    parser.add_argument("--readiness-delta-ratio", type=float, default=0.05)
    parser.add_argument("--bitcost-percentile", type=_positive_int, default=99)
    parser.add_argument("--decode-backsearch-max", type=_positive_int, default=16)
    parser.add_argument("--timeout-seconds", type=_positive_int, default=7_200)
    parser.add_argument("--intra-checkpoint", type=Path, default=None)
    parser.add_argument("--inter-checkpoint", type=Path, default=None)
    parser.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--provider-source-file",
        type=Path,
        action="append",
        default=None,
        help="repeat to override the default executing protocol+worker source bundle",
    )
    return parser


def _codec_policy(arguments: argparse.Namespace) -> MageVideoCodecPolicy:
    return MageVideoCodecPolicy(
        codec_mode="neural",
        preprocess_device=arguments.preparation_device,
        target_canvas=arguments.target_canvas,
        group_size=arguments.group_size,
        images_per_group=arguments.images_per_group,
        patch_size=16,
        max_pixels=arguments.max_pixels,
        min_group_frames=arguments.min_group_frames,
        max_group_frames=arguments.max_group_frames,
        timeout_seconds=arguments.timeout_seconds,
        neural_parameters=MageVideoNeuralCodecParameters(
            quantization_parameter=arguments.neural_qp,
            reset_interval=arguments.neural_reset_interval,
            intra_period=arguments.neural_intra_period,
            max_side=arguments.max_side,
            sequence_length_frames=0,
            canvas_token_side=None,
            readiness_coverage_bins=arguments.readiness_coverage_bins,
            readiness_delta_ratio=arguments.readiness_delta_ratio,
            bitcost_percentile=arguments.bitcost_percentile,
            decode_backsearch_max=arguments.decode_backsearch_max,
        ),
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    model_directory = arguments.model_dir.expanduser().resolve()
    qualification_path = arguments.qualified_provider_manifest.expanduser().resolve()
    qualification_bytes = qualification_path.read_bytes()
    qualification_exact_sha = hashlib.sha256(qualification_bytes).hexdigest()
    if (
        arguments.qualification_manifest_sha256 is not None
        and qualification_exact_sha != arguments.qualification_manifest_sha256
    ):
        raise MageDcvcPrewarmCliError("qualification manifest exact SHA-256 pin does not match")
    qualification = load_mage_dcvc_qualified_provider_manifest(manifest_path=qualification_path)
    checkpoint = load_mage_checkpoint_manifest(
        manifest_path=arguments.checkpoint_manifest_path.expanduser().resolve()
    )
    if (
        arguments.checkpoint_manifest_sha256 is not None
        and checkpoint.manifest_sha256 != arguments.checkpoint_manifest_sha256
    ):
        raise MageDcvcPrewarmCliError("checkpoint manifest SHA-256 pin does not match")

    same_device = _normalise_device(arguments.preparation_device) == _normalise_device(
        arguments.generation_device
    )
    concurrency_policy = "exclusive-shared-device-v1" if same_device else "separate-device-v1"
    if same_device and arguments.shared_device_guard_file is None:
        raise MageDcvcPrewarmCliError(
            "same-device preparation/generation requires --shared-device-guard-file"
        )
    effective_config = build_mage_dcvc_effective_config(
        model_directory=model_directory,
        preparation_device=arguments.preparation_device,
        device_concurrency_policy=concurrency_policy,
        max_side=arguments.max_side,
        target_canvas=arguments.target_canvas,
        group_size=arguments.group_size,
        images_per_group=arguments.images_per_group,
        qp=arguments.neural_qp,
        reset_interval=arguments.neural_reset_interval,
        intra_period=arguments.neural_intra_period,
        max_pixels=arguments.max_pixels,
        min_group_frames=arguments.min_group_frames,
        max_group_frames=arguments.max_group_frames,
        readiness_coverage_bins=arguments.readiness_coverage_bins,
        readiness_delta_ratio=arguments.readiness_delta_ratio,
        bitcost_percentile=arguments.bitcost_percentile,
        decode_backsearch_max=arguments.decode_backsearch_max,
        intra_checkpoint_path=arguments.intra_checkpoint,
        inter_checkpoint_path=arguments.inter_checkpoint,
    )
    report = prewarm_mage_dcvc_provider_v2(
        qualified_provider_manifest=qualification,
        checkpoint_manifest=checkpoint,
        codec_policy=_codec_policy(arguments),
        effective_config=effective_config,
        model_directory=model_directory,
        provider_source_files=tuple(arguments.provider_source_file or _DEFAULT_PROVIDER_SOURCES),
        provider_state_root=arguments.provider_state_root,
        cache_base_root=arguments.cache_base_root,
        source_paths=tuple(arguments.video),
        cache_manifest_output=arguments.manifest_output,
        report_output=arguments.report_output,
        generation_device=arguments.generation_device,
        shared_device_guard_file=arguments.shared_device_guard_file,
        worker_python=arguments.worker_python,
        intra_checkpoint_path=arguments.intra_checkpoint,
        inter_checkpoint_path=arguments.inter_checkpoint,
        response_timeout_seconds=float(arguments.timeout_seconds),
    )
    report_path = arguments.report_output.expanduser().resolve()
    return {
        "ok": True,
        "production_eligible": False,
        "report_version": report.report_version,
        "report_path": str(report_path),
        "report_exact_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "report_semantic_sha256": report.report_semantic_sha256,
        "cache_manifest_path": report.cache_manifest_path,
        "cache_manifest_exact_sha256": report.cache_manifest_exact_sha256,
        "cache_manifest_semantic_sha256": report.cache_manifest_semantic_sha256,
        "qualified_checkpoint_manifest_sha256": report.qualified_checkpoint_manifest_sha256,
        "effective_config_sha256": report.effective_config_sha256,
        "namespace_identity": report.namespace_identity,
        "replay_mode": report.replay_mode,
        "process_start_count": report.worker_process.process_start_count,
        "inferred_process_model_load_count": report.inferred_process_model_load_count,
        "job_count": report.job_count,
        "built_count": report.built_count,
        "verified_hit_count": report.verified_hit_count,
        "prewarm_wall_seconds": report.prewarm_wall_seconds,
        "sequence_length_frames_is_compute_cap": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(_parser().parse_args(argv))
    except (
        MageDcvcPrewarmError,
        MageDcvcPrewarmCliError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {"ok": False, "code": "MAGE_DCVC_PROVIDER_V2_PREWARM_FAILED", "detail": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
