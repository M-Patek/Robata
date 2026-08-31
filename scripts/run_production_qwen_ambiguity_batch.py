#!/usr/bin/env python3
"""Run (or plan) Qwen on selected production WeMM ambiguity windows.

The command is intentionally bounded: ``--limit`` counts recordings, one MCAP
is staged at a time, and each recording receives a complete native six-camera
video root before the existing candidate verifier is called.  This is a
non-production review route; selected eight-second windows remain context and
are not action segments.
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

from robata.benchmark.production_qwen_ambiguity_batch import (  # noqa: E402
    DEFAULT_CPU_WEIGHT_MEMORY_GIB,
    DEFAULT_FRAME_COUNT,
    DEFAULT_GPU_WEIGHT_MEMORY_GIB,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_IMAGE_SIDE,
    DEFAULT_MAX_NEW_TOKENS,
    ProductionQwenAmbiguityBatchError,
    run_production_qwen_ambiguity_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-zip",
        "--archive",
        dest="source_zip",
        type=Path,
        help="optional source archive override; otherwise use selection provenance",
    )
    parser.add_argument(
        "--source-manifest",
        "--source-preflight",
        type=Path,
        dest="source_manifest",
        help="optional source-preflight manifest used to resolve archive/status metadata",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--recording-id", action="append")
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum number of recordings (not windows) to process",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume from output-dir/batch-report.partial.json; completed "
            "recordings are reused and the selection must match"
        ),
    )
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--gpu-weight-memory-gib",
        type=int,
        default=DEFAULT_GPU_WEIGHT_MEMORY_GIB,
    )
    parser.add_argument(
        "--cpu-weight-memory-gib",
        type=int,
        default=DEFAULT_CPU_WEIGHT_MEMORY_GIB,
    )
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--proposal-index", type=int)
    parser.add_argument(
        "--verdict-scope",
        choices=("selected_only", "all_candidates", "pairwise"),
        default="selected_only",
    )
    parser.add_argument("--include-optional-fields", action="store_true")
    parser.add_argument("--mapping-config", type=Path)
    parser.add_argument("--allow-unapproved-profile", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_production_qwen_ambiguity_batch(
            args.selection,
            output_directory=args.output_dir,
            archive_path=args.source_zip,
            source_manifest=args.source_manifest,
            model_directory=args.model_dir,
            offload_directory=args.offload_dir,
            recording_ids=args.recording_id,
            limit=args.limit,
            dry_run=args.dry_run,
            resume=args.resume,
            keep_staging=args.keep_staging,
            window_seconds=args.window_seconds,
            include_tail=not args.no_tail,
            frame_count=args.frame_count,
            max_image_side=args.max_image_side,
            max_new_tokens=args.max_new_tokens,
            gpu_weight_memory_gib=args.gpu_weight_memory_gib,
            cpu_weight_memory_gib=args.cpu_weight_memory_gib,
            jpeg_quality=args.jpeg_quality,
            proposal_index=args.proposal_index,
            verdict_scope=args.verdict_scope,
            include_optional_fields=args.include_optional_fields,
            mapping_config=args.mapping_config,
            allow_unapproved_profile=args.allow_unapproved_profile,
        )
    except (OSError, UnicodeError, ValueError, ProductionQwenAmbiguityBatchError) as exc:
        print(f"production Qwen ambiguity batch failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "recordings": report.get("summary", {}).get("selected_recording_count"),
                "windows": report.get("summary", {}).get("selected_window_count"),
                "verifier_rows": report.get("summary", {}).get("verifier_row_count"),
                "failed_rows": report.get("summary", {}).get("failed_row_count"),
                "skipped_recordings": report.get("summary", {}).get("skipped_recording_count", 0),
                "resumed_from_partial": report.get("resume", {}).get("resumed_from_partial", False),
                "quality_claim": report.get("quality_claim"),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report.get("status") in {"DRY_RUN", "COMPLETE"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
