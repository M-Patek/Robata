"""Run concurrent SharedFrameCache feed-once stress validation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robata.runtime.integration_validation import run_frame_cache_stress  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, default=8)
    parser.add_argument("--callers", type=int, default=16)
    args = parser.parse_args()
    report = run_frame_cache_stress(video_count=args.videos, callers=args.callers)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
