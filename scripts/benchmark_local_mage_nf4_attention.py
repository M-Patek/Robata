"""Run a process-local, non-authoritative NF4 attention-backend benchmark for Mage.

This experiment deliberately monkeypatches a private model-load helper.  It does not
modify the production launcher or the versioned runtime identity.  Its reports can
support a future identity/version decision, but cannot authorize production use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.gpu_telemetry import NvidiaSmiGpuSampler  # noqa: E402
from robata.contracts.hashing import (  # noqa: E402
    canonical_json_bytes,
    exact_bytes_sha256,
)
from robata.inference import mage_video_runtime  # noqa: E402
from robata.inference.mage_checkpoint_identity import (  # noqa: E402
    load_mage_checkpoint_manifest,
    verify_mage_checkpoint_manifest,
)
from robata.inference.mage_codec_cache import (  # noqa: E402
    load_mage_codec_cache_manifest,
    verify_mage_codec_cache_manifest,
)
from robata.inference.mage_video_endpoint import (  # noqa: E402
    MageVideoCodecPolicy,
    MageVideoNeuralCodecParameters,
    build_mage_video_codec_policy_identity,
)

REPORT_VERSION = "mage-nf4-attention-experiment-v1"
REPORT_AUTHORITY = "NON_AUTHORITATIVE_EXPERIMENT"
NF4_LOAD_PROFILE = "bitsandbytes_4bit_nf4_v1"


class MageAttentionBenchmarkError(RuntimeError):
    """A non-authoritative local decoder experiment could not be completed."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
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
    parser.add_argument("--checkpoint-manifest-sha256", type=_sha256, required=True)
    parser.add_argument("--codec-cache-manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--attention", choices=("sdpa", "eager"), required=True)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=256)
    parser.add_argument("--warmup-max-new-tokens", type=_positive_int, default=32)
    parser.add_argument("--gpu-sample-interval-seconds", type=_positive_float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _policy() -> MageVideoCodecPolicy:
    return MageVideoCodecPolicy(
        codec_mode="neural",
        preprocess_device="cuda",
        target_canvas=8,
        group_size=8,
        images_per_group=1,
        patch_size=16,
        max_pixels=65_536,
        min_group_frames=8,
        max_group_frames=8,
        timeout_seconds=7_200,
        neural_parameters=MageVideoNeuralCodecParameters(
            quantization_parameter=42,
            reset_interval=64,
            intra_period=-1,
            max_side=448,
            sequence_length_frames=8,
        ),
    )


def _resolved_attention(model: Any) -> dict[str, str | None]:
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    return {
        "top_level": getattr(config, "_attn_implementation", None),
        "text": getattr(text, "_attn_implementation", None),
        "vision": getattr(vision, "_attn_implementation", None),
    }


def _attention_resolution_matches(*, requested: str, resolved: dict[str, str | None]) -> bool:
    observed = tuple(value for value in resolved.values() if value is not None)
    return bool(observed) and all(value == requested for value in observed)


def _canonical_gpu_telemetry_payload(report: Any) -> dict[str, object]:
    """Project nanosecond counters as decimal strings for canonical JSON safety."""

    payload = report.to_payload()
    for field in (
        "started_wall_clock_unix_ns",
        "stopped_wall_clock_unix_ns",
        "monotonic_duration_ns",
    ):
        value = payload.get(field)
        if isinstance(value, int):
            payload[field] = str(value)
    raw_samples = payload.get("samples")
    if isinstance(raw_samples, list):
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict):
                continue
            for field in (
                "wall_clock_unix_ns",
                "monotonic_offset_ns",
                "query_duration_ns",
            ):
                value = raw_sample.get(field)
                if isinstance(value, int):
                    raw_sample[field] = str(value)
    return payload


def _write_report(path: Path, payload: object) -> str:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".tmp-{uuid.uuid4().hex}")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(resolved)
    return exact_bytes_sha256(resolved.read_bytes())


@contextmanager
def _temporary_attention_backend(attention: str) -> Iterator[None]:
    """Apply a process-local experimental override and always restore the runtime."""

    original = mage_video_runtime._build_model_load_kwargs

    def build_model_load_kwargs(**kwargs: Any) -> dict[str, Any]:
        values = original(**kwargs)
        values["attn_implementation"] = attention
        return values

    mage_video_runtime._build_model_load_kwargs = build_model_load_kwargs
    try:
        yield
    finally:
        mage_video_runtime._build_model_load_kwargs = original


def _validate_inputs(
    arguments: argparse.Namespace,
) -> tuple[Any, Any, tuple[Path, ...], str, tuple[dict[str, object], ...]]:
    model_root = arguments.model_dir.expanduser().resolve()
    checkpoint = load_mage_checkpoint_manifest(
        manifest_path=arguments.checkpoint_manifest_path.expanduser().resolve()
    )
    verify_mage_checkpoint_manifest(manifest=checkpoint, model_directory=model_root)
    if checkpoint.manifest_sha256 != arguments.checkpoint_manifest_sha256:
        raise MageAttentionBenchmarkError("checkpoint manifest pin does not match")

    cache_manifest = load_mage_codec_cache_manifest(
        path=arguments.codec_cache_manifest.expanduser().resolve()
    )
    verified_entries = verify_mage_codec_cache_manifest(manifest=cache_manifest)
    policy = _policy()
    policy_identity = build_mage_video_codec_policy_identity(policy)
    if cache_manifest.checkpoint_manifest_sha256 != checkpoint.manifest_sha256:
        raise MageAttentionBenchmarkError("codec cache checkpoint identity does not match")
    if cache_manifest.codec_policy_sha256 != policy_identity.policy_sha256:
        raise MageAttentionBenchmarkError("codec cache policy identity does not match")

    entries_by_path = {item.source_path: item for item in verified_entries}
    videos = tuple(Path(item).expanduser().resolve() for item in arguments.video)
    if len(set(videos)) != len(videos):
        raise MageAttentionBenchmarkError("video paths must be unique")
    if any(str(video) not in entries_by_path for video in videos):
        raise MageAttentionBenchmarkError("every video must be present in the verified codec cache")

    try:
        prompt_bytes = arguments.prompt_file.expanduser().resolve().read_bytes()
        prompt = prompt_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise MageAttentionBenchmarkError("could not read UTF-8 prompt") from error
    if not prompt.strip():
        raise MageAttentionBenchmarkError("prompt must be nonempty")

    video_inputs = tuple(
        {
            "ordinal": ordinal,
            "source_path": str(video),
            "source_content_sha256": entries_by_path[str(video)].source_content_sha256,
            "source_byte_count": entries_by_path[str(video)].source_byte_count,
            "logical_cache_identity": entries_by_path[str(video)].logical_cache_identity,
        }
        for ordinal, video in enumerate(videos)
    )
    return checkpoint, cache_manifest, videos, prompt, video_inputs


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    runtime: Any | None = None
    sampler: NvidiaSmiGpuSampler | None = None
    try:
        checkpoint, cache_manifest, videos, prompt, video_inputs = _validate_inputs(arguments)
        sampler = NvidiaSmiGpuSampler(interval_seconds=arguments.gpu_sample_interval_seconds)
        sampler.start()
        with _temporary_attention_backend(arguments.attention):
            runtime = mage_video_runtime.MageVideoRuntime(
                model_directory=arguments.model_dir,
                codec_cache_root=Path(cache_manifest.qualified_cache_root),
                load_profile=mage_video_runtime.MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4,
            )
            load_observation = runtime.load()
            model = runtime._model
            if model is None:
                raise MageAttentionBenchmarkError("runtime loaded no model")
            resolved_attention = _resolved_attention(model)
            if not _attention_resolution_matches(
                requested=arguments.attention,
                resolved=resolved_attention,
            ):
                raise MageAttentionBenchmarkError(
                    "requested attention backend was not resolved by the loaded model"
                )

            warmup = runtime.generate(
                video_paths=[videos[0]],
                prompt=prompt,
                max_new_tokens=arguments.warmup_max_new_tokens,
                codec_config=_policy().native_codec_config(),
            )
            timed_started = time.perf_counter()
            results: list[dict[str, object]] = []
            generation_sum = 0.0
            for ordinal, video in enumerate(videos):
                generated = runtime.generate(
                    video_paths=[video],
                    prompt=prompt,
                    max_new_tokens=arguments.max_new_tokens,
                    codec_config=_policy().native_codec_config(),
                )
                telemetry = generated.telemetry
                generation_sum += generated.generation_seconds
                results.append(
                    {
                        "ordinal": ordinal,
                        "video_path": str(video),
                        "output_text_sha256": exact_bytes_sha256(
                            generated.output_text.encode("utf-8")
                        ),
                        "prompt_tokens": generated.prompt_tokens,
                        "output_tokens": generated.output_tokens,
                        "generation_seconds": generated.generation_seconds,
                        "total_request_seconds": (
                            None if telemetry is None else telemetry.total_request_seconds
                        ),
                        "time_to_first_token_seconds": (
                            None if telemetry is None else telemetry.time_to_first_token_seconds
                        ),
                        "output_tokens_per_second": (
                            None if telemetry is None else telemetry.output_tokens_per_second
                        ),
                    }
                )
            timed_wall = time.perf_counter() - timed_started

        gpu = sampler.stop()
        sampler = None
        runtime_identity = runtime.runtime_identity
        if runtime_identity.load_profile.value != NF4_LOAD_PROFILE:
            raise MageAttentionBenchmarkError("benchmark did not use the pinned NF4 load profile")
        report = {
            "report_version": REPORT_VERSION,
            "authority": REPORT_AUTHORITY,
            "production_eligible": False,
            "experimental_scope": {
                "attention_override": "PROCESS_LOCAL_PRIVATE_MONKEYPATCH",
                "runtime_identity_binds_attention_backend": False,
                "production_launcher_modified": False,
                "production_adoption_requires_versioned_identity": True,
            },
            "attention_requested": arguments.attention,
            "attention_resolved": resolved_attention,
            "attention_resolution_verified": True,
            "runtime_identity": {
                "identity_version": runtime_identity.identity_version,
                "load_profile": runtime_identity.load_profile.value,
                "attention_backend_bound": False,
            },
            "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
            "codec_cache_manifest_semantic_sha256": cache_manifest.manifest_semantic_sha256,
            "codec_cache_namespace_identity": cache_manifest.namespace_identity,
            "codec_policy_sha256": cache_manifest.codec_policy_sha256,
            "prompt_sha256": exact_bytes_sha256(prompt.encode("utf-8")),
            "input_videos": list(video_inputs),
            "max_new_tokens": arguments.max_new_tokens,
            "warmup": {
                "max_new_tokens": arguments.warmup_max_new_tokens,
                "actual_output_tokens": warmup.output_tokens,
                "generation_seconds": warmup.generation_seconds,
                "output_text_sha256": exact_bytes_sha256(warmup.output_text.encode("utf-8")),
            },
            "model_load_seconds": load_observation.load_seconds,
            "timed_wall_seconds": timed_wall,
            "generation_sum_seconds": generation_sum,
            "results": results,
            "gpu_telemetry": _canonical_gpu_telemetry_payload(gpu),
        }
        exact_sha256 = _write_report(arguments.output, report)
        print(
            json.dumps(
                {
                    "ok": True,
                    "authority": REPORT_AUTHORITY,
                    "production_eligible": False,
                    "attention": arguments.attention,
                    "report_path": str(arguments.output.expanduser().resolve()),
                    "report_exact_sha256": exact_sha256,
                    "timed_wall_seconds": timed_wall,
                    "generation_sum_seconds": generation_sum,
                    "output_count": len(results),
                },
                sort_keys=True,
            )
        )
        return 0
    except (MageAttentionBenchmarkError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"ok": False, "code": "MAGE_NF4_ATTENTION_EXPERIMENT_FAILED", "detail": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    finally:
        if sampler is not None:
            sampler.stop()
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
