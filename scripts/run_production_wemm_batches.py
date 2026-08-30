#!/usr/bin/env python3
"""Run resumable WeMM pre-annotation over source-preflight PASS MCAPs.

The command defaults to a dry-run.  Use ``--run`` for actual model inference;
only one archive member is staged at a time and a checkpoint is written after
each recording.  ``--batch B1_stratified_pilot --limit 3`` is the recommended
first validation.  The recording loop remains serial by default; ``--pipeline``
opts into bounded decode/model overlap within each recording.  This route uses
an open provisional phrase catalog and does not load the EPIC ontology, Mapper,
Qwen, Mage, or any gold artifact.
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

from robata.benchmark.production_wemm_batch_runner import (  # noqa: E402
    DEFAULT_DIMENSION,
    DEFAULT_FRAME_COUNT,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_QUEUE_CAPACITY,
    DEFAULT_TOP_K,
    DEFAULT_WINDOW_CHUNK_SIZE,
    DEFAULT_WINDOW_SECONDS,
    ProductionWemmBatchRunnerError,
    run_production_wemm_batch,
)


def _ordinals(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ordinal = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid ordinal: {token!r}") from exc
        if ordinal < 0:
            raise argparse.ArgumentTypeError("ordinals must be non-negative")
        result.append(ordinal)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--phrase-catalog", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        help=(
            "optional stride for overlapping WeMM context windows; omit to keep "
            "non-overlapping compatibility"
        ),
    )
    parser.add_argument(
        "--no-tail",
        action="store_true",
        help="do not add a final short processing window (tail is included by default)",
    )
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument(
        "--window-chunk-size",
        type=int,
        default=DEFAULT_WINDOW_CHUNK_SIZE,
        help="number of processing windows decoded at once (default: 1; lower peak memory)",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help=(
            "opt-in native WeMM video microbatch width (default: 1; preserves "
            "the historical singleton path)"
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=("opt-in bounded producer/consumer decode+inference schedule (serial by default)"),
    )
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=DEFAULT_QUEUE_CAPACITY,
        help="bounded producer/consumer queue capacity (default: 1)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--label-variant",
        choices=("canonical", "verb_noun", "natural"),
        default="canonical",
    )
    parser.add_argument(
        "--fusion",
        choices=("mean", "rank_mean", "rrf", "max", "sum"),
        default="mean",
    )
    parser.add_argument(
        "--score-normalization",
        choices=("unit", "clip", "cosine", "minmax", "rank", "none"),
        default="none",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--ordinals", type=_ordinals)
    parser.add_argument("--batch", help="optional preflight batch name, e.g. B1_stratified_pilot")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--run",
        action="store_true",
        help="invoke WeMM; without this flag only validation/checkpoint planning occurs",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-reuse-backend", action="store_true")
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument(
        "--temporal-mode",
        choices=("none", "dense_score"),
        default="none",
        help="attach model-driven temporal interval proposals to each envelope",
    )
    parser.add_argument("--temporal-start-threshold", type=float, default=0.65)
    parser.add_argument("--temporal-stop-threshold", type=float, default=0.50)
    parser.add_argument("--temporal-merge-gap-seconds", type=float, default=0.25)
    parser.add_argument("--temporal-min-duration-seconds", type=float, default=0.10)
    parser.add_argument("--temporal-min-camera-support", type=int, default=1)
    parser.add_argument(
        "--temporal-boundary-mode",
        choices=("observed_probe", "midpoint"),
        default="midpoint",
    )
    parser.add_argument(
        "--temporal-score-policy",
        choices=("top1", "absolute"),
        default="top1",
        help=(
            "temporal support policy; top1 avoids broad tracks from tightly "
            "clustered raw similarities"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_kwargs = {
            "phrase_catalog": args.phrase_catalog,
            "model_directory": args.model_dir,
            "output_directory": args.output_dir,
            "archive_path": args.archive,
            "window_seconds": args.window_seconds,
            "window_stride_seconds": args.window_stride_seconds,
            "include_tail": not args.no_tail,
            "frame_count": args.frame_count,
            "top_k": args.top_k,
            "dimension": args.dimension,
            "window_chunk_size": args.window_chunk_size,
            "inference_batch_size": args.inference_batch_size,
            "device": args.device,
            "label_variant": args.label_variant,
            "fusion": args.fusion,
            "score_normalization": args.score_normalization,
            "max_windows": args.max_windows,
            "ordinals": args.ordinals,
            "batch": args.batch,
            "limit": args.limit,
            "dry_run": not args.run,
            "resume": not args.no_resume,
            "reuse_backend": not args.no_reuse_backend,
            "staging_directory": args.staging_dir,
            "keep_staging": args.keep_staging,
            "checkpoint_path": args.checkpoint,
            "temporal_mode": args.temporal_mode,
            "temporal_start_threshold": args.temporal_start_threshold,
            "temporal_stop_threshold": args.temporal_stop_threshold,
            "temporal_merge_gap_seconds": args.temporal_merge_gap_seconds,
            "temporal_min_duration_seconds": args.temporal_min_duration_seconds,
            "temporal_min_camera_support": args.temporal_min_camera_support,
            "temporal_boundary_mode": args.temporal_boundary_mode,
            "temporal_score_policy": args.temporal_score_policy,
        }
        if args.pipeline:
            run_kwargs.update(
                {
                    "include_pipeline": True,
                    "queue_capacity": args.queue_capacity,
                }
            )
        report = run_production_wemm_batch(args.source_preflight, **run_kwargs)
    except (OSError, UnicodeError, ProductionWemmBatchRunnerError) as exc:
        print(f"production WeMM batch run failed: {exc}", file=sys.stderr)
        return 2
    summary = report.get("summary", {})
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "selected": summary.get("selected_count"),
                "complete": summary.get("complete_count"),
                "failed": summary.get("failed_count"),
                "skipped": summary.get("skipped_count"),
                "windows": summary.get("window_count"),
                "estimated_windows": summary.get("estimated_window_count"),
                "checkpoint": report.get("checkpoint_path"),
                "production_eligible": report.get("production_eligible"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.get("status") in {"DRY_RUN", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
