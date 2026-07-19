"""Emit the explicit H100/model-size capacity assumption matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robata.capacity import calibrate_capacity_scenarios  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h100", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--qa-hours", type=float, default=2.0)
    parser.add_argument("--7b-hours", type=float, default=30.0)
    parser.add_argument("--32b-hours", type=float, default=100.0)
    args = parser.parse_args()
    scenarios = calibrate_capacity_scenarios(
        h100_counts=tuple(args.h100),
        qa_gpu_hours_per_day=args.qa_hours,
        model_annotation_gpu_hours={
            "7B": args.__dict__["7b_hours"],
            "32B": args.__dict__["32b_hours"],
        },
    )
    payload = {
        "measurement_status": "ASSUMPTION",
        "production_eligible": False,
        "scenarios": [scenario.as_dict() for scenario in scenarios],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
