#!/usr/bin/env python3
"""Run the bounded WeMM frame/grid quality matrix on the local cohort."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.wemm_frame_grid_quality_matrix import (  # noqa: E402
    DEFAULT_DIMENSION,
    MATRIX_ARM_IDS,
    WemmFrameGridQualityMatrixError,
    run_wemm_frame_grid_quality_matrix,
)


def _pipeline_arm(value: str) -> str:
    if value not in MATRIX_ARM_IDS:
        choices = ", ".join(MATRIX_ARM_IDS)
        raise argparse.ArgumentTypeError(f"pipeline arm must be one of: {choices}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phrase-catalog", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-chunk-size", type=int, default=1)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument(
        "--pipeline-arm",
        type=_pipeline_arm,
        help="optionally run producer/consumer timing for one matrix arm",
    )
    parser.add_argument("--queue-capacity", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_wemm_frame_grid_quality_matrix(
            args.manifest,
            phrase_catalog=args.phrase_catalog,
            model_directory=args.model_dir,
            device=args.device,
            window_chunk_size=args.window_chunk_size,
            max_windows=args.max_windows,
            dimension=args.dimension,
            pipeline_arm=args.pipeline_arm,
            queue_capacity=args.queue_capacity,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, WemmFrameGridQualityMatrixError) as exc:
        print(f"WeMM frame/grid quality matrix failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": report.get("status"),
                "output": str(args.output),
                "arm_count": len(report.get("matrix", {}).get("arms", [])),
                "arms": [
                    {
                        "arm_id": arm.get("arm_id"),
                        "frame_count": arm.get("frame_count"),
                        "total_pixel_budget": arm.get("total_pixel_budget"),
                        "observed_video_grid_thw": arm.get("observed_video_grid_thw"),
                        "inference_seconds": arm.get("inference_seconds"),
                        "estimated_e2e_seconds": arm.get("estimated_e2e_seconds"),
                        "rank_diagnostic": arm.get("rank_diagnostic"),
                    }
                    for arm in report.get("matrix", {}).get("arms", [])
                    if isinstance(arm, dict)
                ],
                "cache": report.get("decode_cache"),
                "pipeline_arm": args.pipeline_arm,
                "official_quality_status": report.get("official_quality_status"),
                "production_eligible": report.get("production_eligible"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
