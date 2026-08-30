#!/usr/bin/env python3
"""Smoke/qualification command for bounded Qwen3-VL native-video inputs.

The default ``--dry-run`` validates source decoding, frame timeline, and the exact
processor metadata without loading a model.  Omit it with a local model directory
to exercise ``LocalHuggingFaceVisionRuntime.generate_video``.  Outputs are explicitly
local/non-production evidence and preserve both frame provenance and raw model text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.qwen_native_video import (  # noqa: E402
    QWEN_NATIVE_VIDEO_INPUT_VERSION,
    QwenNativeVideoInput,
    sample_qwen_native_video,
)
from robata.inference.local_hf_runtime import (  # noqa: E402
    LocalHfVideoGenerationRequest,
    LocalHuggingFaceVisionRuntime,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, required=True)
    parser.add_argument("--end-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--offload-dir", type=Path, default=ROOT / ".local" / "qwen-offload")
    parser.add_argument("--prompt", default="Describe the complete action in this video interval.")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--context-before-seconds", type=float, default=0.25)
    parser.add_argument("--context-after-seconds", type=float, default=0.25)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only materialize and validate native-video evidence; do not load a model.",
    )
    return parser


def _input_request(
    video: QwenNativeVideoInput,
    *,
    prompt: str,
    max_new_tokens: int,
) -> LocalHfVideoGenerationRequest:
    return LocalHfVideoGenerationRequest(
        video_payloads=video.frame_payloads,
        frame_indices=video.frame_indices,
        frame_timestamps_seconds=video.frame_timestamps_seconds,
        source_fps=video.source_fps,
        total_num_frames=video.total_num_frames,
        width=video.width,
        height=video.height,
        duration_seconds=video.duration_seconds,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def _report_base(video: QwenNativeVideoInput, *, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "report_version": "qwen-native-video-smoke-v1",
        "status": "SUCCEEDED",
        "authority": "LOCAL_NONPRODUCTION_ONLY",
        "production_eligible": False,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "configuration": {
            "video": str(args.video.resolve()),
            "interval": [args.start_seconds, args.end_seconds],
            "frame_count": args.frame_count,
            "context_before_seconds": args.context_before_seconds,
            "context_after_seconds": args.context_after_seconds,
            "max_image_side": args.max_image_side,
            "max_new_tokens": args.max_new_tokens,
            "adapter_version": QWEN_NATIVE_VIDEO_INPUT_VERSION,
            "input_mode": "native_video",
            "dry_run": args.dry_run,
        },
        "input_evidence": video.evidence(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    video = sample_qwen_native_video(
        args.video,
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        frame_count=args.frame_count,
        context_before_seconds=args.context_before_seconds,
        context_after_seconds=args.context_after_seconds,
    )
    report = _report_base(video, args=args)
    if args.dry_run:
        report["model"] = {"status": "NOT_LOADED"}
        report["raw_model_output"] = None
    else:
        if args.model_dir is None:
            raise SystemExit("--model-dir is required unless --dry-run is set")
        runtime = LocalHuggingFaceVisionRuntime(
            model_directory=args.model_dir,
            offload_directory=args.offload_dir,
            max_image_side=args.max_image_side,
        )
        try:
            load = runtime.load()
            observation = runtime.generate_video(
                request=_input_request(
                    video,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            report["model"] = {
                "load_seconds": load.load_seconds,
                "gpu_name": load.gpu_name,
                "generation_seconds": observation.generation_seconds,
                "prompt_tokens": observation.prompt_tokens,
                "output_tokens": observation.output_tokens,
                "gpu_peak_allocated_bytes": observation.gpu_peak_allocated_bytes,
                "input_mode": observation.input_mode,
                "rendered_frame_sizes": observation.rendered_frame_sizes,
            }
            report["raw_model_output"] = observation.output_text
            report["model_frame_evidence"] = {
                "frame_indices": observation.frame_indices,
                "frame_timestamps_seconds": observation.frame_timestamps_seconds,
                "frame_sha256": observation.frame_sha256,
            }
        finally:
            runtime.close()
    report["elapsed_seconds"] = time.perf_counter() - started
    report["finished_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "dry_run": args.dry_run}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
