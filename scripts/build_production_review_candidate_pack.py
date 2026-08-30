#!/usr/bin/env python3
"""Build a unified, review-only production candidate pack from frozen sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_review_candidate_pack import (  # noqa: E402
    ProductionReviewCandidatePackError,
    build_production_review_candidate_pack,
    load_json,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--wemm", type=Path, default=None)
    parser.add_argument("--mage", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expected-camera-count", type=int, default=6)
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = build_production_review_candidate_pack(
            load_json(args.identity),
            load_json(args.boundary),
            load_json(args.wemm) if args.wemm is not None else None,
            mage_sidecar=load_json(args.mage) if args.mage is not None else None,
            top_k=args.top_k,
            expected_camera_count=args.expected_camera_count,
            identity_path=str(args.identity),
            boundary_path=str(args.boundary),
            wemm_path=str(args.wemm) if args.wemm is not None else None,
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
        ProductionReviewCandidatePackError,
        ValueError,
    ) as exc:
        print(f"production review candidate pack failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "windows": report["metrics"]["window_count"],
                "candidates": report["metrics"]["candidate_count"],
                "status": report["status"],
                "official_quality_status": report["official_quality_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
