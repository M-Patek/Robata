#!/usr/bin/env python3
"""Rebuild review packs from COMPLETE production WeMM pre-annotations.

The command reads one or more batch roots/checkpoints, follows only
``preannotation_path`` entries whose checkpoint status is ``COMPLETE``, and
writes rebuilt review packs to a separate directory.  It never starts a model,
opens media, reads Qwen/Mage artifacts, infers action boundaries, or writes
gold annotations.  Running/planned/failed entries are reported as rejected.
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

from robata.benchmark.production_wemm_review_pack_rebuild import (  # noqa: E402
    ProductionWemmReviewPackRebuildError,
    rebuild_production_wemm_review_packs,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="one or more batch roots or batch-run*.json checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".agent_tmp" / "production_wemm_review_rebuilt",
        help="independent output directory (default: .agent_tmp/production_wemm_review_rebuilt)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing rebuilt file (never the runner's own output)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="report path (default: <output-dir>/rebuild-report.json)",
    )
    parser.add_argument(
        "--report-markdown",
        type=Path,
        help="Markdown report path (default: <output-dir>/rebuild-report.md)",
    )
    parser.add_argument("--pretty", action="store_true", help="indent report JSON")
    args = parser.parse_args(argv)

    try:
        report = rebuild_production_wemm_review_packs(
            args.inputs,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ProductionWemmReviewPackRebuildError) as exc:
        print(f"production WeMM review-pack rebuild failed: {exc}", file=sys.stderr)
        return 2

    report_json = args.report_json or args.output_dir / "rebuild-report.json"
    report_markdown = args.report_markdown or args.output_dir / "rebuild-report.md"
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(args.output_dir),
                "report_json": str(report_json),
                "report_markdown": str(report_markdown),
                "written_count": report["summary"]["written_count"],
                "rejected_count": report["summary"]["rejected_count"],
                "invalid_count": report["summary"]["invalid_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
