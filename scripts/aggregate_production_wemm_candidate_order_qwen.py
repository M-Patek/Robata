#!/usr/bin/env python3
"""Aggregate a post-hoc WeMM candidate-order Qwen diagnostic sidecar.

No model or media is touched.  The report compares per-window selected rank,
decision, parse status, and evidence wording across as-is/reverse/shuffle
presentation modes.
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

from robata.benchmark.production_wemm_candidate_order_qwen_aggregate import (  # noqa: E402
    ProductionWemmCandidateOrderQwenAggregateError,
    aggregate_candidate_order_qwen_diagnostic,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="candidate-order Qwen aggregate or partial sidecar",
    )
    parser.add_argument("--output", type=Path, required=True, help="aggregate JSON output")
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        help="expected mode (repeatable; defaults to sidecar modes or as_is/reverse/shuffle)",
    )
    args = parser.parse_args(argv)
    try:
        report = aggregate_candidate_order_qwen_diagnostic(
            args.input,
            expected_modes=args.modes,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        output_md = args.output_md or args.output.with_suffix(".md")
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        ProductionWemmCandidateOrderQwenAggregateError,
    ) as exc:
        print(f"candidate-order Qwen aggregate failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "conclusion": report.get("conclusion", {}).get("result"),
                "windows": report.get("metrics", {}).get("window_count"),
                "decision_changed_windows": report.get("metrics", {}).get(
                    "decision_changed_windows"
                ),
                "rank_changed_windows": report.get("metrics", {}).get("rank_changed_windows"),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
