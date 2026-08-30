#!/usr/bin/env python3
"""Analyze how far Terra-surrogate actions fall below WeMM rank 1.

This is a read-only post-hoc diagnostic over an existing production-only WeMM
comparison artifact.  It never invokes a model, decodes media, changes the
Terra vocabulary/Mapper, opens held-out data, or computes identity material.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_rank_error_analysis import (  # noqa: E402
    ProductionWemmRankErrorAnalysisError,
    analyze_production_wemm_rank_errors,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=ROOT / ".agent_tmp" / "wemm_terra_independent_explicit_20260828.json",
        help="existing production WeMM Terra comparison report",
    )
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = analyze_production_wemm_rank_errors(args.comparison)
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
        ProductionWemmRankErrorAnalysisError,
    ) as exc:
        print(f"production WeMM rank-error analysis failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "status": report["status"],
                "official_quality_status": report["official_quality_status"],
                "eligible_windows": report["input"]["eligible_window_count"],
                "routes": {
                    name: {
                        "window_top1": route["window_level"]["top1_rate_any_reference"],
                        "action_top1": route["action_level"]["top1_rate"],
                        "median_rank": route["action_level"]["median_rank"],
                        "rank_buckets": route["action_level"].get("rank_bucket_histogram", {}),
                        "hard_negative_count": route.get("hard_negative_analysis", {}).get(
                            "hard_negative_count", 0
                        ),
                        "confusion_cluster_count": len(
                            route.get("confusion_clusters", {}).get(
                                "single_reference_windows_only", []
                            )
                        ),
                    }
                    for name, route in report["routes"].items()
                },
                "best_variant_by_metric": report.get("prototype_variant_comparison", {}).get(
                    "best_variant_by_metric", {}
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
