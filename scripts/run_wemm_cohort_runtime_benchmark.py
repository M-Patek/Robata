#!/usr/bin/env python3
"""Run a bounded WeMM serial/Batch2/Batch4 cohort runtime benchmark.

The command is intentionally scoped to the local ten-window, six-camera
cohort.  It writes a diagnostic artifact only; the cohort has no established
gold and the output is never a production annotation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.wemm_cohort_runtime_benchmark import (  # noqa: E402
    DEFAULT_BATCH_SIZES,
    DEFAULT_DIMENSION,
    DEFAULT_FRAME_COUNT,
    DEFAULT_PIXEL_BUDGET,
    WemmCohortRuntimeBenchmarkError,
    run_wemm_cohort_runtime_benchmark,
)


def _sizes(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            number = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid batch size: {token!r}") from exc
        result.append(number)
    if not result:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return tuple(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phrase-catalog", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--pixel-budget", type=int, default=DEFAULT_PIXEL_BUDGET)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-chunk-size", type=int, default=1)
    parser.add_argument(
        "--batch-sizes",
        type=_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="comma-separated native video batch widths (default: 2,4)",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="also measure bounded decode/consumer overlap",
    )
    parser.add_argument("--queue-capacity", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_wemm_cohort_runtime_benchmark(
            args.manifest,
            phrase_catalog=args.phrase_catalog,
            model_directory=args.model_dir,
            frame_count=args.frame_count,
            pixel_budget=args.pixel_budget,
            dimension=args.dimension,
            device=args.device,
            window_chunk_size=args.window_chunk_size,
            batch_sizes=args.batch_sizes,
            max_windows=args.max_windows,
            include_pipeline=args.pipeline,
            queue_capacity=args.queue_capacity,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, json.JSONDecodeError, WemmCohortRuntimeBenchmarkError) as exc:
        print(f"WeMM cohort benchmark failed: {exc}", file=sys.stderr)
        return 2
    arms = report.get("arms", [])
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "output": str(args.output),
                "arms": [
                    {
                        "arm_id": arm.get("arm_id"),
                        "inference_seconds": arm.get("inference_seconds"),
                        "estimated_e2e_seconds": arm.get("estimated_e2e_seconds"),
                    }
                    for arm in arms
                    if isinstance(arm, dict)
                ],
                "pipeline": bool(report.get("pipeline")),
                "production_eligible": report.get("production_eligible"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
