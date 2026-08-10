"""Run the local Mage fixed-frame control on the frozen five-segment cam_01 fixture.

The run deliberately uses the exact six PNG frame bytes from the frozen common fixture.
The default prompt is the exact native Mage binding prompt; ``common_qwen`` is an
explicit cross-model prompt control. Both modes exclude codec preparation, stream
memory, cognition gating, and production admission. The output is therefore a local
visual-frontend diagnostic, not a production qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robata.benchmark.mage_fixed_frame import (  # noqa: E402
    MAGE_FIXED_FRAME_NATIVE_PROMPT_VERSION,
    MAGE_FIXED_FRAME_POLICY_VERSION,
    build_fixed_frame_input_identity,
    close_fixed_frame_images,
    load_verified_fixed_frame_images,
    project_mage_fixed_frame_output,
)
from robata.benchmark.qwen_mage_common_projection import (  # noqa: E402
    COMMON_QWEN_PROMPT_VERSION,
    CommonProjectionError,
    build_qwen_common_prompt,
    compare_observations,
    downstream_projection,
    load_common_projection_fixture,
    run_common_downstream,
)
from robata.benchmark.qwen_r12_request_corpus import (  # noqa: E402
    QWEN_R12_20260806_EXPECTED,
    load_qwen_request_corpus,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256  # noqa: E402
from robata.inference.mage_checkpoint_identity import (  # noqa: E402
    MageCheckpointIdentityError,
    load_mage_checkpoint_manifest,
    verify_mage_checkpoint_manifest,
)
from robata.inference.mage_video_runtime import (  # noqa: E402
    MageVideoLoadProfile,
    MageVideoRuntime,
    MageVideoRuntimeError,
)

REPORT_VERSION: Final = "mage-fixed-frame-control-qualification-v2"
EXIT_FAILURE: Final = 2
DEFAULT_MODEL_DIRECTORY = Path(r"D:\HuggingFace\Mage-VL")
DEFAULT_CORPUS_DATABASE = Path(
    r"D:\tmp\robata-qwen-run-20260806\canonical-qwen-full-r12-20260806"
    r"\inference-evidence.sqlite3"
)
DEFAULT_MAGE_STREAM_ARTIFACT_ROOT = (
    REPOSITORY_ROOT / ".tmp" / "temporal-ab-131k-control-r3" / "stream-artifacts"
)
DEFAULT_CHECKPOINT_MANIFEST: Path | None = None


class MageFixedFrameQualificationError(RuntimeError):
    """The local fixed-frame control could not be completed honestly."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--corpus-db", type=Path, default=DEFAULT_CORPUS_DATABASE)
    parser.add_argument(
        "--mage-stream-artifact-root",
        type=Path,
        default=DEFAULT_MAGE_STREAM_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        default=DEFAULT_CHECKPOINT_MANIFEST,
        required=DEFAULT_CHECKPOINT_MANIFEST is None,
        help="verified local Mage checkpoint manifest (required; never inferred)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=None,
        help="override decoder budget; native mode defaults to the fixture binding budget",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("native_mage", "common_qwen"),
        default="native_mage",
        help="native Mage binding prompt or explicit cross-model Qwen common prompt",
    )
    parser.add_argument(
        "--load-profile",
        choices=tuple(profile.value for profile in MageVideoLoadProfile),
        default=MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4.value,
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def _write_json(path: Path, payload: object) -> str:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload) + b"\n"
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(body)
        os.replace(temporary, target)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return exact_bytes_sha256(body)


def _capacity(*, media_seconds: float, recurring_wall_seconds: float) -> dict[str, object]:
    if recurring_wall_seconds <= 0.0:
        raise MageFixedFrameQualificationError("recurring wall time must be positive")
    realtime_factor = media_seconds / recurring_wall_seconds
    return {
        "camera_media_seconds": media_seconds,
        "recurring_wall_seconds": recurring_wall_seconds,
        "camera_realtime_factor": realtime_factor,
        "local_equivalent_lanes_for_25x": math.ceil(25.0 / realtime_factor),
        "quality_qualified": False,
        "decision_eligible": False,
        "production_qualification": "NOT_CLAIMED",
    }


def _verify_inputs(arguments: argparse.Namespace) -> tuple[Any, Any]:
    model_directory = arguments.model_dir.expanduser().resolve()
    if arguments.checkpoint_manifest is None:
        raise MageFixedFrameQualificationError(
            "--checkpoint-manifest is required; provide a manifest generated for this "
            "exact model tree"
        )
    manifest_path = arguments.checkpoint_manifest.expanduser().resolve()
    if not model_directory.is_dir():
        raise MageFixedFrameQualificationError(
            f"Mage model directory does not exist: {model_directory}"
        )
    try:
        manifest = load_mage_checkpoint_manifest(manifest_path=manifest_path)
        verify_mage_checkpoint_manifest(
            manifest=manifest,
            model_directory=model_directory,
        )
    except MageCheckpointIdentityError as error:
        raise MageFixedFrameQualificationError(
            "Mage checkpoint manifest verification failed"
        ) from error
    corpus = load_qwen_request_corpus(
        arguments.corpus_db.expanduser().resolve(),
        expected=QWEN_R12_20260806_EXPECTED,
    )
    fixture = load_common_projection_fixture(
        corpus=corpus,
        mage_stream_artifact_root=arguments.mage_stream_artifact_root,
    )
    return manifest, fixture


def _prompt_for_case(case: Any, prompt_mode: str) -> tuple[str, str]:
    if prompt_mode == "common_qwen":
        return COMMON_QWEN_PROMPT_VERSION, build_qwen_common_prompt(case)
    if prompt_mode != "native_mage":
        raise MageFixedFrameQualificationError(
            f"unsupported fixed-frame prompt mode: {prompt_mode}"
        )
    prompt = case.binding.endpoint_request.decoder.prompt
    prompt_version = MAGE_FIXED_FRAME_NATIVE_PROMPT_VERSION
    try:
        payload = json.loads(prompt)
        candidate = payload.get("prompt_version") if isinstance(payload, dict) else None
        if isinstance(candidate, str) and candidate.strip():
            prompt_version = candidate
    except (TypeError, ValueError):
        # The exact prompt bytes remain identity-bound even when the prompt is not JSON.
        pass
    return prompt_version, prompt


def _max_new_tokens_for_case(arguments: argparse.Namespace, case: Any, prompt_mode: str) -> int:
    if arguments.max_new_tokens is not None:
        return arguments.max_new_tokens
    if prompt_mode == "native_mage":
        return int(case.binding.endpoint_request.decoder.max_new_tokens)
    return 160


def run(
    arguments: argparse.Namespace,
    *,
    runtime_factory: Callable[..., MageVideoRuntime] = MageVideoRuntime,
) -> tuple[int, dict[str, object]]:
    observed_at = _utc_now()
    try:
        manifest, fixture = _verify_inputs(arguments)
        load_profile = MageVideoLoadProfile(arguments.load_profile)
        effective_budgets = tuple(
            _max_new_tokens_for_case(arguments, case, arguments.prompt_mode)
            for case in fixture.cases
        )
        base: dict[str, object] = {
            "report_version": REPORT_VERSION,
            "observed_at": observed_at,
            "authority": "LOCAL_NONPRODUCTION_ONLY",
            "production_eligible": False,
            "admission": {
                "state": "BASELINE_ONLY_NOT_ADMITTED",
                "reasons": [
                    "FIXED_FRAME_CONTROL_REMOVES_NATIVE_CODEC_AND_STREAMING_SEMANTICS",
                    "UNLABELED_MODEL_AGREEMENT_IS_NOT_GROUND_TRUTH_ACCURACY",
                    "SINGLE_CAMERA_LOCAL_CONTROL_ONLY",
                    "NO_LINUX_H100_SUSTAINED_CAPACITY_EVIDENCE",
                ],
            },
            "configuration": {
                "policy_version": MAGE_FIXED_FRAME_POLICY_VERSION,
                "model_directory": str(arguments.model_dir.expanduser().resolve()),
                "model_identifier": manifest.model_identifier,
                "model_revision": manifest.model_revision,
                "checkpoint_manifest_path": str(
                    arguments.checkpoint_manifest.expanduser().resolve()
                ),
                "checkpoint_manifest_sha256": manifest.manifest_sha256,
                "load_profile": load_profile.value,
                "max_new_tokens_override": arguments.max_new_tokens,
                "effective_max_new_tokens_per_segment": list(effective_budgets),
                "prompt_mode": arguments.prompt_mode,
                "decoder": {
                    "do_sample": False,
                    "use_cache": True,
                },
                "frame_count_per_segment": [len(case.selected_frames) for case in fixture.cases],
                "prompt_equivalence": (
                    "EXACT_MAGE_BINDING_PROMPT_BYTES"
                    if arguments.prompt_mode == "native_mage"
                    else "EXACT_QWEN_COMMON_V2_PROMPT_BYTES"
                ),
                "processor_backend": "frames",
                "codec_preparation": "DISABLED",
                "stream_memory": "DISABLED",
            },
            "fixture": fixture.projection(),
            "fixture_semantic_sha256": fixture.semantic_sha256,
        }
        if arguments.verify_only:
            base["execution"] = {
                "status": "VERIFIED_INPUTS_ONLY",
                "model_loaded": False,
                "segment_count": len(fixture.cases),
            }
            return 0, base

        runtime = runtime_factory(
            model_directory=arguments.model_dir.expanduser().resolve(),
            load_profile=load_profile,
        )
        load_observation = runtime.load()
        run_started = time.perf_counter()
        segment_rows: list[dict[str, object]] = []
        projections = []
        try:
            for case in fixture.cases:
                prompt_version, prompt = _prompt_for_case(case, arguments.prompt_mode)
                max_new_tokens = _max_new_tokens_for_case(arguments, case, arguments.prompt_mode)
                identity = build_fixed_frame_input_identity(
                    case=case,
                    checkpoint_manifest_sha256=manifest.manifest_sha256,
                    model_revision=manifest.model_revision,
                    load_profile=load_profile.value,
                    max_new_tokens=max_new_tokens,
                    prompt=prompt,
                    prompt_version=prompt_version,
                )
                images = load_verified_fixed_frame_images(case)
                try:
                    generation = runtime.generate_fixed_frames(
                        frames=images,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                    )
                finally:
                    close_fixed_frame_images(images)
                row = {
                    "ordinal": case.ordinal,
                    "context_manifest_semantic_sha256": (
                        case.context.context_manifest_semantic_sha256
                    ),
                    "input_identity_semantic_sha256": identity.semantic_sha256,
                    "selected_frame_sha256_values": [
                        frame.sha256 for frame in case.selected_frames
                    ],
                    "prompt_mode": arguments.prompt_mode,
                    "prompt_version": identity.prompt_version,
                    "prompt_exact_sha256": identity.prompt_exact_sha256,
                    "max_new_tokens": max_new_tokens,
                    "raw_output_text": generation.output_text,
                    "raw_output_exact_sha256": exact_bytes_sha256(
                        generation.output_text.encode("utf-8")
                    ),
                    "prompt_tokens": generation.prompt_tokens,
                    "output_tokens": generation.output_tokens,
                    "generation_seconds": generation.generation_seconds,
                    "telemetry": asdict(generation.telemetry),
                }
                try:
                    projection = project_mage_fixed_frame_output(
                        case=case,
                        input_identity=identity,
                        output_text=generation.output_text,
                    )
                except CommonProjectionError as error:
                    row.update(
                        {
                            "projection_status": "FAILED_STRICT_COMPACT_JSON",
                            "projection_error": str(error),
                            "diagnostics": [],
                        }
                    )
                    segment_rows.append(row)
                    continue
                projections.append(projection)
                row.update(
                    {
                        "projection_status": "SUCCEEDED",
                        "inference_artifact_exact_sha256": (
                            projection.inference_artifact_exact_sha256
                        ),
                        "observation_semantic_sha256": (
                            projection.observation.observation_semantic_sha256
                        ),
                        "diagnostics": list(projection.diagnostics),
                    }
                )
                segment_rows.append(row)
        finally:
            runtime.close()
        run_wall_seconds = time.perf_counter() - run_started
        observations = tuple(item.observation for item in projections)
        generation_seconds = tuple(float(row["generation_seconds"]) for row in segment_rows)
        downstream = None
        if len(projections) == len(fixture.cases):
            downstream = run_common_downstream(
                cases=fixture.cases,
                observations=observations,
                observation_elapsed_seconds=generation_seconds,
            )
        projection_failures = len(fixture.cases) - len(projections)
        base["execution"] = {
            "status": ("SUCCEEDED" if projection_failures == 0 else "COMPLETED_WITH_QUALITY_HOLD"),
            "model_loaded": True,
            "load": {
                "load_seconds": load_observation.load_seconds,
                "execution_device": load_observation.execution_device,
                "runtime_identity": {
                    "identity_version": load_observation.runtime_identity.identity_version,
                    "load_profile": load_observation.runtime_identity.load_profile.value,
                },
            },
            "run_wall_seconds": run_wall_seconds,
            "generation_sum_seconds": sum(generation_seconds),
            "cold_total_seconds": load_observation.load_seconds + run_wall_seconds,
            "segments": segment_rows,
            "strict_projection_count": len(projections),
            "strict_projection_failure_count": projection_failures,
            "capacity": _capacity(
                media_seconds=fixture.duration_seconds,
                recurring_wall_seconds=run_wall_seconds,
            ),
        }
        semantic_evidence: dict[str, object] = {
            "authority": "UNLABELED_MODEL_AGREEMENT_ONLY",
            "is_ground_truth_accuracy": False,
            "strict_projection_count": len(projections),
            "expected_projection_count": len(fixture.cases),
            "quality_qualified": False,
            "decision_eligible": False,
        }
        if downstream is None:
            semantic_evidence["hold_reason"] = "HOLD_STRICT_COMPACT_PROJECTION_FAILURE_V1"
            base["downstream"] = {
                "status": "NOT_RUN_DUE_TO_STRICT_PROJECTION_FAILURE",
                "authority": "LOCAL_NONPRODUCTION_ONLY",
            }
        else:
            semantic_evidence["comparison_role_mapping"] = {
                "mage_fields": "fixed_frame_candidate",
                "qwen_fields": "frozen_native_mage_reference",
            }
            semantic_evidence["fixed_frames_vs_native_mage"] = compare_observations(
                mage=observations,
                qwen=tuple(case.mage_observation for case in fixture.cases),
            )
            base["downstream"] = downstream_projection(downstream)
        base["semantic_evidence"] = semantic_evidence
        return 0, base
    except (
        MageFixedFrameQualificationError,
        MageVideoRuntimeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        return EXIT_FAILURE, {
            "report_version": REPORT_VERSION,
            "observed_at": observed_at,
            "authority": "LOCAL_NONPRODUCTION_ONLY",
            "production_eligible": False,
            "execution": {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        }


def main() -> int:
    arguments = _parser().parse_args()
    code, report = run(arguments)
    digest = _write_json(arguments.output, report)
    print(
        json.dumps(
            {
                "exit_code": code,
                "output": str(arguments.output.expanduser().resolve()),
                "report_exact_sha256": digest,
                "status": report.get("execution", {}).get("status"),
            },
            sort_keys=True,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
