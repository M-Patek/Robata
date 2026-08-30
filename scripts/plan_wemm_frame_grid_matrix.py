#!/usr/bin/env python3
"""Plan a bounded WeMM 4/8-frame by total-pixel-budget matrix.

This command reads only the production-shaped cohort manifest and the existing
four-frame WeMM vocabulary shadow report.  It validates their common six-camera
cohort and writes a plan-only artifact; it never opens an MCAP, decodes media,
loads model weights, or computes an identity.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.wemm_frame_grid_matrix import (  # noqa: E402
    WemmFrameGridMatrixError,
    build_wemm_frame_grid_matrix,
)


def _load(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WemmFrameGridMatrixError(f"could not read {label} JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise WemmFrameGridMatrixError(f"{label} JSON root must be an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="six-camera cohort manifest")
    parser.add_argument(
        "--baseline", type=Path, required=True, help="four-frame WeMM shadow report"
    )
    parser.add_argument("--output", type=Path, required=True, help="plan JSON output path")
    args = parser.parse_args(argv)
    try:
        cohort = _load(args.cohort, label="cohort")
        baseline = _load(args.baseline, label="baseline")
        plan = build_wemm_frame_grid_matrix(cohort, baseline)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, WemmFrameGridMatrixError, TypeError, ValueError) as exc:
        print(f"WeMM frame/grid matrix planning failed: {exc}", file=sys.stderr)
        return 2

    matrix = plan["matrix"]
    arms = matrix["arms"] if isinstance(matrix, Mapping) else []
    print(
        json.dumps(
            {
                "status": plan["status"],
                "output": str(args.output),
                "matrix_id": matrix.get("matrix_id") if isinstance(matrix, Mapping) else None,
                "arms": len(arms) if isinstance(arms, Sequence) else 0,
                "camera_window_slots_per_arm": plan["cohort"]["slot_count"],
                "model_invoked": plan["model_invoked"],
                "media_decoded": plan["media_decoded"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
