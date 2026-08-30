#!/usr/bin/env python3
"""Build a provisional review draft from a structured model sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_structured_review_adapter import (  # noqa: E402
    StructuredReviewAdapterError,
    build_structured_review_draft,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structured_input",
        type=Path,
        help="structured annotation envelope or supported model sidecar JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.structured_input.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise StructuredReviewAdapterError("input JSON root must be an object")
        draft = build_structured_review_draft(value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError, StructuredReviewAdapterError) as exc:
        print(f"structured production review draft failed: {exc}", file=sys.stderr)
        return 2

    statuses = {str(item["annotation_draft"]["status"]) for item in draft["windows"]}
    print(
        json.dumps(
            {
                "output": str(args.output),
                "windows": len(draft["windows"]),
                "annotation_statuses": sorted(statuses),
                "gold_written": draft["controls"]["gold_written"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
