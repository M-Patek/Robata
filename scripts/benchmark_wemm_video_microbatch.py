"""Plan a bounded WeMM video Batch2/Batch4 experiment without loading a model.

The executable model path is intentionally exposed through
``robata.benchmark.wemm_video_microbatch_benchmark.run_video_microbatch_benchmark``
for callers that already own a resident backend and decoded groups.  This CLI
only validates the 40.8335-second six-camera fixture and writes its plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.wemm_video_microbatch_benchmark import (  # noqa: E402
    DEFAULT_COHORT_DURATION_SECONDS,
    WemmVideoMicrobatchBenchmarkError,
    build_cohort_microbatch_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="bounded production cohort manifest (for example the 4s 10-window fixture)",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_COHORT_DURATION_SECONDS,
        help="maximum cohort duration in seconds (default: 40.8335)",
    )
    args = parser.parse_args()
    try:
        plan = build_cohort_microbatch_plan(
            args.manifest,
            max_duration_seconds=args.max_duration,
        )
    except WemmVideoMicrobatchBenchmarkError as exc:
        parser.error(str(exc))
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
