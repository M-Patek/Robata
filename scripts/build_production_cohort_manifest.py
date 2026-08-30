#!/usr/bin/env python3
"""Build a label-neutral six-camera window manifest from one MCAP source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_cohort import ProductionCohortError, build_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--include-tail", action="store_true")
    args = parser.parse_args()
    try:
        payload = build_manifest(
            args.source,
            window_seconds=args.window_seconds,
            include_tail=args.include_tail,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (ProductionCohortError, OSError) as exc:
        print(f"production cohort manifest failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "windows": len(payload["windows"]),
                "gold_status": payload["gold"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
