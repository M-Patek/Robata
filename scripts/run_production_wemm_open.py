#!/usr/bin/env python3
"""Run review-only WeMM retrieval against a provisional open phrase catalog.

No EPIC action catalog, Mapper, Qwen/Mage route, or gold artifact is loaded.
Use ``--dry-run`` to validate a full manifest/catalog plan without decoding
media or loading the local WeMM checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_open_runner import (  # noqa: E402
    DEFAULT_RELATIVE_MARGIN_SCALE,
    ProductionWemmOpenRunnerError,
    dry_run_open_phrase_plan,
    run_production_wemm_open,
)
from robata.benchmark.production_wemm_preannotation import build_review_pack  # noqa: E402
from robata.benchmark.production_wemm_temporal import (  # noqa: E402
    ProductionWemmTemporalError,
    normalize_score_policy,
)


def _temporal_score_policy(value: str) -> str:
    """Parse a score-policy alias and return its canonical wire spelling."""

    try:
        return normalize_score_policy(value, field="--temporal-score-policy")
    except ProductionWemmTemporalError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phrase-catalog", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--video-min-pixels",
        type=int,
        help="optional shortest-edge pixel bound for direct video resize",
    )
    parser.add_argument(
        "--video-max-pixels",
        type=int,
        help="optional longest-edge pixel bound for direct video resize",
    )
    parser.add_argument(
        "--label-variant",
        choices=("canonical", "verb_noun", "natural"),
        default="canonical",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument(
        "--temporal-mode",
        choices=("none", "dense_score", "adaptive_score"),
        default="none",
        help=(
            "attach temporal interval proposals; adaptive_score adds a second "
            "short-context WeMM review pass"
        ),
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
        type=_temporal_score_policy,
        default="top1",
        help=(
            "temporal support policy (aliases raw, winner, stable, "
            "winner_stability, candidate_relative, relative, and contrast are "
            "normalized to canonical values); top1 avoids broad tracks from "
            "tightly clustered raw similarities; winner_stable additionally "
            "repairs one-context interior winner glitches"
        ),
    )
    parser.add_argument(
        "--temporal-relative-margin-scale",
        type=float,
        default=DEFAULT_RELATIVE_MARGIN_SCALE,
        help=(
            "logistic scale for candidate-vs-runner-up temporal margins "
            "(used by relative_margin policy)"
        ),
    )
    parser.add_argument(
        "--temporal-relative-margin-min-target-score",
        type=float,
        default=0.60,
        help=(
            "minimum target similarity required before a relative-margin "
            "trajectory can support an interval"
        ),
    )
    parser.add_argument(
        "--temporal-refinement-span-seconds",
        type=float,
        default=1.0,
        help="short context width for adaptive_score boundary probes",
    )
    parser.add_argument(
        "--temporal-refinement-min-request-span-seconds",
        type=float,
        default=0.10,
        help="minimum admissible adaptive refinement request width",
    )
    parser.add_argument(
        "--temporal-refinement-max-requests",
        type=int,
        default=128,
        help="maximum onset/offset requests in adaptive_score mode",
    )
    parser.add_argument(
        "--window-chunk-size",
        type=int,
        default=1,
        help="number of processing windows decoded at once (default: 1; lower peak memory)",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help=(
            "opt-in WeMM video microbatch width (default: 1; preserves the "
            "historical singleton path)"
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
        default=1,
        help="bounded producer/consumer queue capacity (default: 1)",
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
    parser.add_argument("--validate-crcs", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate manifest/catalog only; do not decode or invoke WeMM",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.dry_run:
            report = dry_run_open_phrase_plan(
                args.manifest,
                phrase_catalog=args.phrase_catalog,
                max_windows=args.max_windows,
            )
        else:
            run_kwargs = {
                "phrase_catalog": args.phrase_catalog,
                "model_directory": args.model_dir,
                "frame_count": args.frame_count,
                "top_k": args.top_k,
                "dimension": args.dimension,
                "device": args.device,
                "video_min_pixels": args.video_min_pixels,
                "video_max_pixels": args.video_max_pixels,
                "label_variant": args.label_variant,
                "max_windows": args.max_windows,
                "window_chunk_size": args.window_chunk_size,
                "inference_batch_size": args.inference_batch_size,
                "fusion": args.fusion,
                "score_normalization": args.score_normalization,
                "validate_crcs": args.validate_crcs,
                "temporal_mode": args.temporal_mode,
                "temporal_start_threshold": args.temporal_start_threshold,
                "temporal_stop_threshold": args.temporal_stop_threshold,
                "temporal_merge_gap_seconds": args.temporal_merge_gap_seconds,
                "temporal_min_duration_seconds": args.temporal_min_duration_seconds,
                "temporal_min_camera_support": args.temporal_min_camera_support,
                "temporal_boundary_mode": args.temporal_boundary_mode,
                "temporal_score_policy": args.temporal_score_policy,
                "temporal_relative_margin_scale": args.temporal_relative_margin_scale,
                "temporal_relative_margin_min_target_score": (
                    args.temporal_relative_margin_min_target_score
                ),
                "temporal_refinement_span_seconds": args.temporal_refinement_span_seconds,
                "temporal_refinement_min_request_span_seconds": (
                    args.temporal_refinement_min_request_span_seconds
                ),
                "temporal_refinement_max_requests": args.temporal_refinement_max_requests,
            }
            if args.pipeline:
                run_kwargs.update(
                    {
                        "include_pipeline": True,
                        "queue_capacity": args.queue_capacity,
                    }
                )
            report = run_production_wemm_open(args.manifest, **run_kwargs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.review_output is not None and not args.dry_run:
            review = build_review_pack(report)
            args.review_output.parent.mkdir(parents=True, exist_ok=True)
            args.review_output.write_text(
                json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ProductionWemmOpenRunnerError) as exc:
        print(f"production open WeMM run failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report.get("status"),
                "windows": len(
                    report.get("windows", report.get("source", {}).get("window_count", []))
                )
                if isinstance(report.get("windows"), list)
                else report.get("source", {}).get("window_count"),
                "label_space": report.get("label_space", {}).get("kind")
                if isinstance(report.get("label_space"), dict)
                else report.get("catalog", {}).get("format"),
                "production_eligible": report.get("production_eligible"),
                "model_invoked": report.get("controls", {}).get("model_invoked", False),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
