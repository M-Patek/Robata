#!/usr/bin/env python3
"""Aggregate completed per-recording WeMM review packs.

This command is read-only.  It does not launch WeMM, open MCAP/media, call
Qwen/Mage, infer boundaries, or write gold annotations.
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

from robata.benchmark.production_wemm_review_pack_aggregate import (  # noqa: E402
    ProductionWemmReviewPackAggregateError,
    aggregate_production_wemm_review_packs,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help=(
            "one or more review-pack JSON files, review directories, or batch "
            "checkpoints (for example B1 and remaining)"
        ),
    )
    parser.add_argument(
        "--expected-camera-count",
        type=int,
        default=6,
        help="expected cameras per review window (default: 6)",
    )
    parser.add_argument("--json-output", type=Path, help="write aggregate JSON to this path")
    parser.add_argument(
        "--markdown-output", type=Path, help="write a compact Markdown report to this path"
    )
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = parser.parse_args(argv)

    try:
        report = aggregate_production_wemm_review_packs(
            args.input[0] if len(args.input) == 1 else args.input,
            expected_camera_count=args.expected_camera_count,
        )
    except (OSError, ProductionWemmReviewPackAggregateError) as exc:
        print(f"production WeMM review-pack aggregation failed: {exc}", file=sys.stderr)
        return 2

    encoded = json.dumps(
        report,
        ensure_ascii=False,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    markdown = render_markdown(report)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
