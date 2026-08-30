#!/usr/bin/env python3
"""Build a review-only annotation projection from recorded model sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_annotation_projection import (  # noqa: E402
    ProductionAnnotationProjectionError,
    load_json,
    project_production_annotations,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", "--qwen-structured", dest="qwen", type=Path, required=True)
    parser.add_argument("--wemm", "--wemm-shadow", dest="wemm", type=Path)
    parser.add_argument("--review", type=Path, help="optional geometry-only review artifact")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-consensus-cameras", type=int, default=2)
    parser.add_argument("--merge-gap-seconds", type=float, default=0.05)
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = project_production_annotations(
            load_json(args.qwen),
            load_json(args.wemm) if args.wemm else None,
            top_k=args.top_k,
            min_consensus_cameras=args.min_consensus_cameras,
            merge_gap_seconds=args.merge_gap_seconds,
            review=load_json(args.review) if args.review else None,
            input_paths={
                "qwen": str(args.qwen),
                "wemm": str(args.wemm) if args.wemm else None,
                "review": str(args.review) if args.review else None,
            },
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_md.write_text(render_markdown(report), encoding="utf-8")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ProductionAnnotationProjectionError,
        ValueError,
    ) as exc:
        print(f"production annotation projection failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "status": report["status"],
                "official_quality_status": report["official_quality_status"],
                "candidate_count": report["metrics"]["candidate_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
