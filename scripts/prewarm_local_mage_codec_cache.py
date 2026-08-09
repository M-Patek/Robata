"""Prewarm Mage's exact native DCVC processor cache with auditable identities."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.hashing import exact_bytes_sha256  # noqa: E402
from robata.inference.mage_checkpoint_identity import (  # noqa: E402
    load_mage_checkpoint_manifest,
    verify_mage_checkpoint_manifest,
)
from robata.inference.mage_codec_cache import (  # noqa: E402
    MageCodecCacheError,
    prewarm_mage_codec_cache,
    verify_mage_codec_cache_manifest,
    write_mage_codec_cache_manifest,
)
from robata.inference.mage_video_endpoint import (  # noqa: E402
    MageVideoCodecPolicy,
    MageVideoNeuralCodecParameters,
)


class MageCodecPrewarmCliError(RuntimeError):
    """The requested local codec prewarm could not be qualified."""


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


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", type=_sha256, default=None)
    parser.add_argument("--cache-base-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--codec-target-canvas", type=_positive_int, default=8)
    parser.add_argument("--codec-group-size", type=_positive_int, default=8)
    parser.add_argument("--codec-images-per-group", type=_positive_int, default=1)
    parser.add_argument("--codec-patch-size", type=_positive_int, default=16)
    parser.add_argument("--codec-max-pixels", type=_positive_int, default=65_536)
    parser.add_argument("--codec-min-group-frames", type=_positive_int, default=8)
    parser.add_argument("--codec-max-group-frames", type=_positive_int, default=8)
    parser.add_argument("--codec-timeout-seconds", type=_positive_int, default=7_200)
    parser.add_argument("--preprocess-device", choices=("cuda",), default="cuda")
    parser.add_argument("--neural-qp", type=_nonnegative_int, default=42)
    parser.add_argument("--neural-reset-interval", type=_positive_int, default=64)
    parser.add_argument("--neural-intra-period", type=int, default=-1)
    parser.add_argument("--neural-max-side", type=_nonnegative_int, default=448)
    parser.add_argument("--neural-sequence-length-frames", type=_nonnegative_int, default=8)
    parser.add_argument("--neural-canvas-token-side", type=_positive_int, default=None)
    parser.add_argument("--neural-readiness-coverage-bins", type=_positive_int, default=3)
    parser.add_argument("--neural-readiness-delta-ratio", type=_positive_float, default=0.05)
    parser.add_argument("--neural-bitcost-percentile", type=_positive_int, default=99)
    parser.add_argument("--neural-decode-backsearch-max", type=_positive_int, default=16)
    return parser


def _policy(arguments: argparse.Namespace) -> MageVideoCodecPolicy:
    return MageVideoCodecPolicy(
        codec_mode="neural",
        preprocess_device=arguments.preprocess_device,
        target_canvas=arguments.codec_target_canvas,
        group_size=arguments.codec_group_size,
        images_per_group=arguments.codec_images_per_group,
        patch_size=arguments.codec_patch_size,
        max_pixels=arguments.codec_max_pixels,
        min_group_frames=arguments.codec_min_group_frames,
        max_group_frames=arguments.codec_max_group_frames,
        timeout_seconds=arguments.codec_timeout_seconds,
        neural_parameters=MageVideoNeuralCodecParameters(
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
        ),
    )


def _prompt(arguments: argparse.Namespace) -> str:
    if arguments.prompt_file is None:
        return "Observe this video segment."
    try:
        prompt = arguments.prompt_file.expanduser().resolve().read_text(encoding="utf-8")
    except OSError as error:
        raise MageCodecPrewarmCliError("could not read --prompt-file") from error
    if not prompt.strip():
        raise MageCodecPrewarmCliError("--prompt-file must be nonempty")
    return prompt


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        model_directory = arguments.model_dir.expanduser().resolve()
        manifest = load_mage_checkpoint_manifest(
            manifest_path=arguments.checkpoint_manifest_path.expanduser().resolve()
        )
        verify_mage_checkpoint_manifest(manifest=manifest, model_directory=model_directory)
        if (
            arguments.checkpoint_manifest_sha256 is not None
            and manifest.manifest_sha256 != arguments.checkpoint_manifest_sha256
        ):
            raise MageCodecPrewarmCliError("checkpoint manifest SHA-256 pin does not match")
        result = prewarm_mage_codec_cache(
            model_directory=model_directory,
            checkpoint_manifest=manifest,
            codec_policy=_policy(arguments),
            cache_base_root=arguments.cache_base_root,
            video_paths=arguments.video,
            prompt=_prompt(arguments),
        )
        verified = verify_mage_codec_cache_manifest(manifest=result)
        output_path = arguments.manifest_output.expanduser().resolve()
        write_mage_codec_cache_manifest(manifest=result, path=output_path)
        exact_sha256 = exact_bytes_sha256(output_path.read_bytes())
        print(
            json.dumps(
                {
                    "ok": True,
                    "manifest_version": result.manifest_version,
                    "manifest_path": str(output_path),
                    "manifest_exact_sha256": exact_sha256,
                    "manifest_semantic_sha256": result.manifest_semantic_sha256,
                    "namespace_identity": result.namespace_identity,
                    "qualified_cache_root": result.qualified_cache_root,
                    "entry_count": result.entry_count,
                    "built_count": result.built_count,
                    "verified_hit_count": result.verified_hit_count,
                    "prewarm_wall_seconds": result.prewarm_wall_seconds,
                    "verified_entry_count": len(verified),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (MageCodecCacheError, MageCodecPrewarmCliError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"ok": False, "code": "MAGE_CODEC_PREWARM_FAILED", "detail": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
