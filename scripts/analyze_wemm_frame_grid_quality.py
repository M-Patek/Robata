#!/usr/bin/env python3
"""Compare completed WeMM frame/grid runtime artifacts.

This command is deliberately post-hoc: it reads JSON runtime reports only and
never loads a model, decodes media, reads/writes gold, or derives an identity.
The resulting report is a stability diagnostic.  It does not establish action
accuracy or production eligibility.
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

from robata.benchmark.wemm_frame_grid_quality_probe import (  # noqa: E402
    WemmFrameGridQualityProbeError,
    analyze_wemm_frame_grid_quality,
)


def _input_spec(value: str) -> tuple[str, Path]:
    arm_id, separator, raw_path = value.partition("=")
    arm_id = arm_id.strip()
    raw_path = raw_path.strip()
    if not separator or not arm_id or not raw_path:
        raise argparse.ArgumentTypeError("--input must have ARM_ID=PATH form")
    return arm_id, Path(raw_path).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=_input_spec,
        action="append",
        required=True,
        metavar="ARM_ID=PATH",
        help="completed runtime JSON; repeat once per matrix arm",
    )
    parser.add_argument("--runtime-arm", default="batch4")
    parser.add_argument("--reference", help="ARM_ID used for modal-agreement comparison")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    return parser


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WemmFrameGridQualityProbeError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise WemmFrameGridQualityProbeError(f"{path} must contain a JSON object")
    return value


def _write(payload: Mapping[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if output is None:
        print(rendered, end="")
        return
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    try:
        for arm_id, path in args.input:
            if arm_id in seen:
                raise WemmFrameGridQualityProbeError(f"duplicate input arm: {arm_id!r}")
            seen.add(arm_id)
            entries.append(
                {
                    "arm_id": arm_id,
                    "source_artifact": str(path),
                    "runtime_report": _load(path),
                }
            )
        payload = analyze_wemm_frame_grid_quality(
            entries,
            reference_arm_id=args.reference,
            runtime_arm_id=args.runtime_arm,
        )
        _write(payload, args.output)
    except (OSError, UnicodeError, WemmFrameGridQualityProbeError) as exc:
        print(f"WeMM frame/grid quality probe failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
