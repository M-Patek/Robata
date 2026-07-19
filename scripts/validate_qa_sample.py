"""Validate QA policy against sample-medium.mcap plus the complete 21-issue matrix."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.qa_validation import validate_sample_mcap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default=str(ROOT / "data" / "source" / "sample-medium.mcap"),
    )
    args = parser.parse_args()
    report = validate_sample_mcap(args.source)
    print(
        json.dumps(
            asdict(report),
            default=lambda value: value.value,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
