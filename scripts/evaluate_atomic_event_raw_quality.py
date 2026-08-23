#!/usr/bin/env python3
"""Score Qwen/Mage free text before it reaches the semantic Mapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.atomic_event_quality import evaluate_raw_output_records  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="raw-output-first JSONL files")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit per-case projections from the emitted report",
    )
    return parser


def run(inputs: list[Path], *, summary_only: bool) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    for path in inputs:
        evaluation = evaluate_raw_output_records(_load_jsonl(path))
        arm: dict[str, Any] = {
            "name": path.parent.name,
            "source": str(path),
            "summary": evaluation["summary"],
        }
        if not summary_only:
            arm["cases"] = evaluation["cases"]
        arms.append(arm)
    return {"report_version": "atomic-event-raw-quality-comparison-v1", "arms": arms}


def main() -> int:
    args = _parser().parse_args()
    report = run(args.inputs, summary_only=args.summary_only)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
