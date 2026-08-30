#!/usr/bin/env python3
"""Aggregate completed production WeMM pre-annotation sidecars.

This is a read-only post-processing command.  It does not launch WeMM, open
MCAP/media, call Qwen/Mage, read review bridges or evaluate quality against a
gold denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_preannotation_aggregate import (  # noqa: E402
    aggregate_production_wemm_preannotations,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only structural summary of production WeMM pre-annotations."
    )
    parser.add_argument(
        "input", type=Path, help="sidecar, batch-run checkpoint, or batch directory"
    )
    parser.add_argument(
        "--expected-camera-count",
        type=int,
        default=6,
        help="expected cameras per processing window (default: 6)",
    )
    parser.add_argument("--json-output", type=Path, help="write the JSON report to this path")
    parser.add_argument(
        "--markdown-output", type=Path, help="write the Markdown report to this path"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent JSON output (default output is still deterministic JSON)",
    )
    args = parser.parse_args(argv)
    report = aggregate_production_wemm_preannotations(
        args.input, expected_camera_count=args.expected_camera_count
    )
    encoded = json.dumps(
        report, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True
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
