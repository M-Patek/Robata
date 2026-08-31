#!/usr/bin/env python3
"""Run the Qwen boundary-only diagnostic on complete native video windows.

This benchmark-local runner deliberately does not replace the canonical
structured annotation route.  Every selected camera/window is materialized as
one complete native ``video`` request.  An optional identity sidecar contributes
only an explicitly marked, untrusted Qwen hypothesis to the text prompt; it is
never treated as gold and is never used to cut the video.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_boundary_probe import (  # noqa: E402
    AUTHORITY,
    EXPECTED_IDENTITY_SIDECAR_FORMAT,
    OFFICIAL_QUALITY_STATUS,
    PRODUCTION_BOUNDARY_FRAME_PROMPT_VERSION,
    PRODUCTION_BOUNDARY_PROBE_VERSION,
    PRODUCTION_BOUNDARY_PROMPT_BOUNDED_V2_VERSION,
    PRODUCTION_BOUNDARY_PROMPT_VERSION,
    QWEN_PRODUCTION_BOUNDARY_ONLY_BLIND_PROMPT,
    QWEN_PRODUCTION_BOUNDARY_ONLY_BOUNDED_V2_PROMPT,
    QWEN_PRODUCTION_BOUNDARY_ONLY_PROMPT,
    TIMESTAMP_BASIS,
    ProductionBoundaryProbeError,
    find_identity_context,
    frame_ordinal_prompt,
    index_identity_sidecar,
    load_json,
    parse_qwen_boundary_frame_output,
    parse_qwen_boundary_only_output,
)

if TYPE_CHECKING:
    from robata.benchmark.qwen_native_video import QwenNativeVideoInput
    from robata.inference.local_hf_runtime import (
        LocalHfVideoGenerationRequest,
        LocalHuggingFaceVisionRuntime,
    )

# Keep the symbol patchable for dry-run tests without importing optional model
# runtime dependencies at module import time.
LocalHuggingFaceVisionRuntime: Any = None

NATIVE_ROUTE = "complete_native_video"
MODEL_IDENTIFIER = "Qwen3-VL-4B-Instruct"
DEFAULT_MANIFEST = ROOT / ".agent_tmp" / "production_sample_cohort_manifest_4s_20260827.json"
DEFAULT_VIDEO_ROOT = ROOT / ".local" / "production-worker-it" / "state2" / "video-view"
DEFAULT_MODEL_DIR = Path(r"D:\HuggingFace\Qwen3-VL-4B-Instruct")
DEFAULT_OFFLOAD_DIR = ROOT / ".local" / "qwen-boundary-only-offload"
DEFAULT_OUTPUT = ROOT / ".agent_tmp" / "production_qwen_boundary_only_4s.json"


class QwenBoundaryOnlyRunnerError(ValueError):
    """Raised when the boundary-only runner input is malformed."""


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QwenBoundaryOnlyRunnerError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise QwenBoundaryOnlyRunnerError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QwenBoundaryOnlyRunnerError(f"{field} must be non-empty text")
    return value.strip()


def _manifest_windows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the small manifest surface needed by the native sampler."""

    raw_windows = _sequence(manifest.get("windows"), field="manifest.windows")
    source = _mapping(manifest.get("source"), field="manifest.source")
    source_cameras = _sequence(source.get("cameras"), field="manifest.source.cameras")
    default_cameras = [
        _text(_mapping(item, field="manifest.source.cameras[]").get("camera_id"), field="camera_id")
        for item in source_cameras
    ]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_windows):
        row = _mapping(raw, field=f"manifest.windows[{index}]")
        window_id = _text(row.get("window_id"), field=f"manifest.windows[{index}].window_id")
        if window_id in seen:
            raise QwenBoundaryOnlyRunnerError(f"duplicate window_id: {window_id}")
        seen.add(window_id)
        start, end = row.get("start_seconds"), row.get("end_seconds")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise QwenBoundaryOnlyRunnerError(f"manifest.windows[{index}].start_seconds invalid")
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            raise QwenBoundaryOnlyRunnerError(f"manifest.windows[{index}].end_seconds invalid")
        start_value, end_value = float(start), float(end)
        if not math.isfinite(start_value) or start_value < 0:
            raise QwenBoundaryOnlyRunnerError(f"manifest.windows[{index}].start_seconds invalid")
        if not math.isfinite(end_value) or end_value <= start_value:
            raise QwenBoundaryOnlyRunnerError(f"manifest.windows[{index}] end must exceed start")
        ordinal = row.get("ordinal", index)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise QwenBoundaryOnlyRunnerError(f"manifest.windows[{index}].ordinal invalid")
        raw_cameras = row.get("camera_ids", default_cameras)
        cameras = [
            _text(camera, field=f"manifest.windows[{index}].camera_ids[]")
            for camera in _sequence(raw_cameras, field=f"manifest.windows[{index}].camera_ids")
        ]
        if not cameras or len(set(cameras)) != len(cameras):
            raise QwenBoundaryOnlyRunnerError(
                f"manifest.windows[{index}].camera_ids must be non-empty and unique"
            )
        result.append(
            {
                "ordinal": ordinal,
                "window_id": window_id,
                "start_seconds": start_value,
                "end_seconds": end_value,
                "camera_ids": cameras,
            }
        )
    if not result:
        raise QwenBoundaryOnlyRunnerError("manifest.windows must be non-empty")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--offload-dir", type=Path, default=DEFAULT_OFFLOAD_DIR)
    parser.add_argument("--identity-sidecar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--camera-id", action="append")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--max-image-side", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--gpu-weight-memory-gib", type=int, default=5)
    parser.add_argument("--cpu-weight-memory-gib", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        choices=("base", "bounded_v2", "frame_ordinal"),
        default="base",
        help=(
            "benchmark prompt arm; bounded_v2 emphasizes the window-relative range; "
            "frame_ordinal asks for sampled frame ordinals to avoid clock confusion"
        ),
    )
    return parser


def _request(
    video: QwenNativeVideoInput,
    *,
    prompt: str,
    max_new_tokens: int,
) -> LocalHfVideoGenerationRequest:
    from robata.inference.local_hf_runtime import LocalHfVideoGenerationRequest

    return LocalHfVideoGenerationRequest(
        video_payloads=video.frame_payloads,
        frame_indices=video.frame_indices,
        frame_timestamps_seconds=video.frame_timestamps_seconds,
        source_fps=video.source_fps,
        total_num_frames=video.total_num_frames,
        width=video.width,
        height=video.height,
        duration_seconds=video.duration_seconds,
        source_window_start_seconds=video.interval_start_seconds,
        source_window_end_seconds=video.interval_end_seconds,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stop_after_first_complete_json_object=True,
    )


def _prompt_for_context(
    context: Mapping[str, Any],
    *,
    prompt_variant: str = "base",
    frame_count: int = 8,
) -> tuple[str, str, bool]:
    if prompt_variant not in {"base", "bounded_v2", "frame_ordinal"}:
        raise QwenBoundaryOnlyRunnerError("unsupported prompt variant")
    frame_prompt = ""
    if prompt_variant == "frame_ordinal":
        try:
            frame_prompt = frame_ordinal_prompt(frame_count)
        except ProductionBoundaryProbeError as exc:
            raise QwenBoundaryOnlyRunnerError(str(exc)) from exc
    base_prompt = (
        frame_prompt
        if prompt_variant == "frame_ordinal"
        else QWEN_PRODUCTION_BOUNDARY_ONLY_BOUNDED_V2_PROMPT
        if prompt_variant == "bounded_v2"
        else QWEN_PRODUCTION_BOUNDARY_ONLY_PROMPT
    )
    base_version = (
        PRODUCTION_BOUNDARY_FRAME_PROMPT_VERSION
        if prompt_variant == "frame_ordinal"
        else PRODUCTION_BOUNDARY_PROMPT_BOUNDED_V2_VERSION
        if prompt_variant == "bounded_v2"
        else PRODUCTION_BOUNDARY_PROMPT_VERSION
    )
    if context.get("status") == "AVAILABLE" and isinstance(context.get("action"), str):
        action = context["action"].strip()
        if action:
            prompt = (
                base_prompt + "\nThe following is an untrusted identity hypothesis from a separate "
                "Qwen observation; "
                + "use it only to locate evidence and do not repeat it in your JSON:\n"
                + f"identity_hypothesis: {action}"
            )
            return (
                prompt,
                base_version + "-identity-conditioned",
                True,
            )
    return (
        (
            frame_prompt
            if prompt_variant == "frame_ordinal"
            else QWEN_PRODUCTION_BOUNDARY_ONLY_BOUNDED_V2_PROMPT
            if prompt_variant == "bounded_v2"
            else QWEN_PRODUCTION_BOUNDARY_ONLY_BLIND_PROMPT
        ),
        base_version + "-blind",
        False,
    )


def _planned_row(
    window: Mapping[str, Any],
    camera_id: str,
    *,
    identity_context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "window_id": window["window_id"],
        "ordinal": window["ordinal"],
        "interval": [window["start_seconds"], window["end_seconds"]],
        "camera_id": camera_id,
        "status": "NOT_RUN",
        "input_mode": NATIVE_ROUTE,
        "timestamp_basis": TIMESTAMP_BASIS,
        "identity_context": dict(identity_context),
        "raw_text": None,
        "parsed_boundary": {
            "parse_status": "INVALID",
            "timestamp_basis": None,
            "boundary_status": "INVALID",
            "start_time_sec": None,
            "end_time_sec": None,
            "confidence": None,
            "evidence": "",
            "errors": ["MODEL_NOT_RUN"],
            "warnings": [],
        },
        "native_video_complete": False,
    }


def _observed_row(
    window: Mapping[str, Any],
    camera_id: str,
    *,
    video: QwenNativeVideoInput,
    observation: Any,
    identity_context: Mapping[str, Any],
    max_new_tokens: int,
    prompt_variant: str = "base",
) -> dict[str, Any]:
    frame_timestamps = list(
        getattr(observation, "frame_timestamps_seconds", video.frame_timestamps_seconds)
    )
    if prompt_variant == "frame_ordinal":
        raw_parsed = parse_qwen_boundary_frame_output(
            observation.output_text, frame_count=len(frame_timestamps)
        )
        parsed = dict(raw_parsed)
        parsed["timestamp_basis"] = TIMESTAMP_BASIS
        parsed["coordinate_mode"] = "sampled_frame_ordinal"
        if parsed.get("boundary_status") == "MEASURED":
            start_ordinal = parsed.get("start_frame_ordinal")
            end_ordinal = parsed.get("end_frame_ordinal")
            if (
                isinstance(start_ordinal, int)
                and isinstance(end_ordinal, int)
                and 0 <= start_ordinal < len(frame_timestamps)
                and 0 <= end_ordinal < len(frame_timestamps)
            ):
                # Use the decoder's actual sampled interval as the local
                # coordinate frame.  MCAP/video timestamps can differ from
                # manifest bounds by a few microseconds (and the first
                # decoded frame may be just after the requested start).  This
                # avoids turning a valid last-frame ordinal into a false
                # out-of-window claim without clipping a substantive error.
                clip_start = float(
                    getattr(video, "interval_start_seconds", window["start_seconds"])
                )
                clip_end = float(getattr(video, "interval_end_seconds", window["end_seconds"]))
                duration = clip_end - clip_start
                start = float(frame_timestamps[start_ordinal]) - clip_start
                end = float(frame_timestamps[end_ordinal]) - clip_start
                # The sampler intentionally keeps the nearest frame before
                # the requested start to preserve boundary evidence.  Allow
                # at most one decoded-frame period of media jitter (plus the
                # existing millisecond timeline tolerance), not an arbitrary
                # clip, and record the tolerance in the parsed provenance.
                source_fps = float(getattr(video, "source_fps", 0.0) or 0.0)
                epsilon = max(1e-3, (1.0 / source_fps) + 1e-3) if source_fps > 0.0 else 1e-3
                if -epsilon <= start < end and end <= duration + epsilon and duration > 0.0:
                    start = max(0.0, min(duration, start))
                    end = max(0.0, min(duration, end))
                    parsed["start_time_sec"] = start
                    parsed["end_time_sec"] = end
                    parsed["mapped_timestamp_basis"] = TIMESTAMP_BASIS
                    parsed["timestamp_mapping_status"] = "MAPPED_FROM_FRAME_ORDINAL"
                    parsed["mapping_tolerance_seconds"] = epsilon
                else:
                    parsed["parse_status"] = "INVALID"
                    parsed["boundary_status"] = "INVALID"
                    parsed["start_time_sec"] = None
                    parsed["end_time_sec"] = None
                    parsed.setdefault("errors", []).append("FRAME_MAPPING_OUT_OF_WINDOW")
            else:
                parsed["parse_status"] = "INVALID"
                parsed["boundary_status"] = "INVALID"
                parsed["start_time_sec"] = None
                parsed["end_time_sec"] = None
                parsed.setdefault("errors", []).append("FRAME_MAPPING_INVALID")
        else:
            parsed["start_time_sec"] = None
            parsed["end_time_sec"] = None
    else:
        parsed = parse_qwen_boundary_only_output(
            observation.output_text,
            window_duration_seconds=float(window["end_seconds"]) - float(window["start_seconds"]),
        )
    warnings = list(parsed.get("warnings", []))
    if getattr(observation, "output_tokens", 0) >= max_new_tokens:
        warnings.append("MAX_NEW_TOKENS_REACHED")
    parsed["warnings"] = list(dict.fromkeys(warnings))
    return {
        "window_id": window["window_id"],
        "ordinal": window["ordinal"],
        "interval": [window["start_seconds"], window["end_seconds"]],
        "camera_id": camera_id,
        "status": "SUCCEEDED",
        "input_mode": getattr(observation, "input_mode", NATIVE_ROUTE),
        "timestamp_basis": TIMESTAMP_BASIS,
        "identity_context": dict(identity_context),
        "prompt_variant": prompt_variant,
        "prompt_frame_count": len(frame_timestamps),
        **(
            {"frame_ordinal_max": len(frame_timestamps) - 1}
            if prompt_variant == "frame_ordinal" and frame_timestamps
            else {}
        ),
        "coordinate_mode": "sampled_frame_ordinal"
        if prompt_variant == "frame_ordinal"
        else TIMESTAMP_BASIS,
        "raw_text": observation.output_text,
        "parsed_boundary": parsed,
        "frame_indices": list(getattr(observation, "frame_indices", video.frame_indices)),
        "frame_timestamps_seconds": frame_timestamps,
        "rendered_frame_sizes": [
            list(size) for size in getattr(observation, "rendered_frame_sizes", ())
        ],
        "prompt_tokens": getattr(observation, "prompt_tokens", None),
        "output_tokens": getattr(observation, "output_tokens", None),
        "generation_seconds": getattr(observation, "generation_seconds", None),
        "gpu_peak_allocated_bytes": getattr(observation, "gpu_peak_allocated_bytes", None),
        "native_video_complete": True,
        "visual_input": (
            observation.visual_input.as_dict()
            if getattr(observation, "visual_input", None) is not None
            else None
        ),
    }


def _failed_row(
    window: Mapping[str, Any],
    camera_id: str,
    error: Exception,
    *,
    identity_context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "window_id": window["window_id"],
        "ordinal": window["ordinal"],
        "interval": [window["start_seconds"], window["end_seconds"]],
        "camera_id": camera_id,
        "status": "FAILED",
        "input_mode": NATIVE_ROUTE,
        "timestamp_basis": TIMESTAMP_BASIS,
        "identity_context": dict(identity_context),
        "error": f"{type(error).__name__}: {error}",
        "raw_text": None,
        "parsed_boundary": {
            "parse_status": "INVALID",
            "timestamp_basis": None,
            "boundary_status": "INVALID",
            "start_time_sec": None,
            "end_time_sec": None,
            "confidence": None,
            "evidence": "",
            "errors": ["RUNTIME_FAILURE"],
            "warnings": [],
        },
        "native_video_complete": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest, field="manifest")
    windows = _manifest_windows(manifest)
    if args.limit is not None:
        if isinstance(args.limit, bool) or args.limit <= 0:
            raise QwenBoundaryOnlyRunnerError("--limit must be a positive integer")
        windows = windows[: args.limit]
    requested_cameras = set(args.camera_id or ())
    all_cameras = {camera for window in windows for camera in window["camera_ids"]}
    unknown = requested_cameras - all_cameras
    if unknown:
        raise QwenBoundaryOnlyRunnerError(f"unknown camera IDs: {', '.join(sorted(unknown))}")
    identity_index: dict[tuple[str, str | None], dict[str, Any]] = {}
    if getattr(args, "identity_sidecar", None) is not None:
        identity_index = index_identity_sidecar(args.identity_sidecar)

    rows: list[dict[str, Any]] = []
    selected_cameras: set[str] = set()
    model_invoked = False
    media_decoded = False
    identity_attached = 0
    started = time.perf_counter()
    runtime: LocalHuggingFaceVisionRuntime | None = None
    load_observation: Any | None = None
    try:
        if not args.dry_run:
            # Native-video decoding is an optional runtime dependency.  Keep
            # dry-run/contract inspection usable in the core environment where
            # the OpenCV-backed Qwen adapter is intentionally absent.
            from robata.benchmark.qwen_native_video import sample_qwen_native_video

            runtime_factory = LocalHuggingFaceVisionRuntime
            if runtime_factory is None:
                from robata.inference.local_hf_runtime import (
                    LocalHuggingFaceVisionRuntime as runtime_factory,
                )

            runtime = runtime_factory(
                model_directory=args.model_dir,
                offload_directory=args.offload_dir,
                max_image_side=args.max_image_side,
                gpu_weight_memory_gib=args.gpu_weight_memory_gib,
                cpu_weight_memory_gib=args.cpu_weight_memory_gib,
            )
            load_observation = runtime.load()
        for window in windows:
            allowed = requested_cameras or set(window["camera_ids"])
            for camera_id in window["camera_ids"]:
                if camera_id not in allowed:
                    continue
                selected_cameras.add(camera_id)
                context = find_identity_context(
                    identity_index, window_id=window["window_id"], camera_id=camera_id
                )
                prompt, prompt_version, attached = _prompt_for_context(
                    context,
                    prompt_variant=getattr(args, "prompt_variant", "base"),
                    frame_count=args.frame_count,
                )
                identity_attached += attached
                if args.dry_run:
                    planned = _planned_row(window, camera_id, identity_context=context)
                    planned["prompt_version"] = prompt_version
                    planned["prompt_frame_count"] = args.frame_count
                    if getattr(args, "prompt_variant", "base") == "frame_ordinal":
                        planned["frame_ordinal_max"] = args.frame_count - 1
                    rows.append(planned)
                    continue
                model_invoked = True
                video_path = args.video_root / f"{camera_id}.mp4"
                try:
                    video = sample_qwen_native_video(
                        video_path,
                        start_seconds=window["start_seconds"],
                        end_seconds=window["end_seconds"],
                        frame_count=args.frame_count,
                        context_before_seconds=0.0,
                        context_after_seconds=0.0,
                        jpeg_quality=args.jpeg_quality,
                    )
                    media_decoded = True
                    assert runtime is not None
                    observation = runtime.generate_video(
                        request=_request(
                            video,
                            prompt=prompt,
                            max_new_tokens=args.max_new_tokens,
                        )
                    )
                    row = _observed_row(
                        window,
                        camera_id,
                        video=video,
                        observation=observation,
                        identity_context=context,
                        max_new_tokens=args.max_new_tokens,
                        prompt_variant=getattr(args, "prompt_variant", "base"),
                    )
                    row["prompt_version"] = prompt_version
                    rows.append(row)
                except Exception as error:  # preserve per-camera evidence and continue
                    failed = _failed_row(
                        window,
                        camera_id,
                        error,
                        identity_context=context,
                    )
                    failed["prompt_version"] = prompt_version
                    failed["prompt_frame_count"] = args.frame_count
                    if getattr(args, "prompt_variant", "base") == "frame_ordinal":
                        failed["frame_ordinal_max"] = args.frame_count - 1
                    rows.append(failed)
    finally:
        if runtime is not None:
            runtime.close()

    source = _mapping(manifest.get("source"), field="manifest.source")
    camera_count = source.get("camera_count")
    if isinstance(camera_count, bool) or not isinstance(camera_count, int):
        camera_count = len(selected_cameras)
    status = (
        "NOT_RUN"
        if args.dry_run
        else "SUCCEEDED"
        if rows and all(row["status"] != "FAILED" for row in rows)
        else "PARTIAL"
    )
    conditioned = identity_attached > 0
    return {
        "format": PRODUCTION_BOUNDARY_PROBE_VERSION,
        "authority": AUTHORITY,
        "production_eligible": False,
        "status": status,
        "source": {
            "manifest": str(args.manifest),
            "video_root": str(args.video_root),
            "identity_sidecar": (
                str(args.identity_sidecar) if getattr(args, "identity_sidecar", None) else None
            ),
            "identity_sidecar_format": EXPECTED_IDENTITY_SIDECAR_FORMAT if identity_index else None,
            "window_count": len(windows),
            "camera_count": camera_count,
            "selected_camera_ids": sorted(selected_cameras),
        },
        "model": {
            "identifier": MODEL_IDENTIFIER,
            "native_route": NATIVE_ROUTE,
            "input_mode": NATIVE_ROUTE,
            "timestamp_basis": TIMESTAMP_BASIS,
            "prompt_version": (
                PRODUCTION_BOUNDARY_FRAME_PROMPT_VERSION
                if getattr(args, "prompt_variant", "base") == "frame_ordinal"
                else PRODUCTION_BOUNDARY_PROMPT_BOUNDED_V2_VERSION
                if getattr(args, "prompt_variant", "base") == "bounded_v2"
                else PRODUCTION_BOUNDARY_PROMPT_VERSION
            ),
            "prompt": (
                frame_ordinal_prompt(args.frame_count)
                if getattr(args, "prompt_variant", "base") == "frame_ordinal"
                else QWEN_PRODUCTION_BOUNDARY_ONLY_BOUNDED_V2_PROMPT
                if getattr(args, "prompt_variant", "base") == "bounded_v2"
                else QWEN_PRODUCTION_BOUNDARY_ONLY_PROMPT
            ),
            "prompt_variant": getattr(args, "prompt_variant", "base"),
            "prompt_frame_count": args.frame_count,
            "frame_ordinal_max": (
                args.frame_count - 1
                if getattr(args, "prompt_variant", "base") == "frame_ordinal"
                else None
            ),
            "identity_conditioned": conditioned,
            "frame_count": args.frame_count,
            "max_image_side": args.max_image_side,
            "max_new_tokens": args.max_new_tokens,
            "stop_after_first_complete_json_object": True,
            "dry_run": args.dry_run,
            "load": (
                {
                    "load_seconds": load_observation.load_seconds,
                    "gpu_name": load_observation.gpu_name,
                }
                if load_observation is not None
                else None
            ),
        },
        "windows": rows,
        "quality": {
            "measurement_status": OFFICIAL_QUALITY_STATUS,
            "quality_claim": False,
            "reason": "boundary-only diagnostic; no source-bound accepted action gold",
        },
        "controls": {
            "model_invoked": model_invoked,
            "source_media_decoded": media_decoded,
            "identity_sidecar_read": bool(identity_index),
            "identity_context_attached_rows": identity_attached,
            "gold_included": False,
            "gold_written": False,
            "predictions_copied_to_gold": False,
            "ontology_modified": False,
            "mapper_modified": False,
            "training_invoked": False,
            "native_video_complete": True,
        },
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        QwenBoundaryOnlyRunnerError,
        ProductionBoundaryProbeError,
    ) as exc:
        print(f"production Qwen boundary-only run failed: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "windows": report["source"]["window_count"],
                "rows": len(report["windows"]),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
