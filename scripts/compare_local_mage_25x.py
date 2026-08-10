"""Build the evidence-bound local Mage 25x capacity report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.mage_25x import (  # noqa: E402
    DEFAULT_CAPACITY_HEADROOM,
    DEFAULT_DAILY_CAMERA_HOURS,
    build_mage_25x_capacity_report,
    load_provider_v2_local_baseline,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402

DEFAULT_SOURCE_REPORT = (
    REPOSITORY_ROOT / "docs" / "mage-dcvc-provider-v2-local-qualification-2026-08-09.json"
)
DEFAULT_SOURCE_EXACT_SHA256 = "7298d21fb05f0ecbc4bc1e11481f67abf2c82b4b13380227177edfbbbaa24287"
DEFAULT_SOURCE_SEMANTIC_SHA256 = "ea659e3e78243e43e4c1f921ff0898c64f18c4e68993c9c219d2425c8a25b0d8"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--source-exact-sha256", default=DEFAULT_SOURCE_EXACT_SHA256)
    parser.add_argument("--source-semantic-sha256", default=DEFAULT_SOURCE_SEMANTIC_SHA256)
    parser.add_argument("--daily-camera-hours", type=float, default=DEFAULT_DAILY_CAMERA_HOURS)
    parser.add_argument("--headroom", type=float, default=DEFAULT_CAPACITY_HEADROOM)
    parser.add_argument(
        "--codec-multiplier",
        type=float,
        default=None,
        help="optional unmeasured target codec multiplier; requires --decoder-multiplier",
    )
    parser.add_argument(
        "--decoder-multiplier",
        type=float,
        default=None,
        help="optional unmeasured target decoder multiplier; requires --codec-multiplier",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    baseline = load_provider_v2_local_baseline(
        path=arguments.source_report,
        expected_exact_sha256=arguments.source_exact_sha256,
        expected_semantic_sha256=arguments.source_semantic_sha256,
        source_reference=(
            "docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json"
            if arguments.source_report.resolve() == DEFAULT_SOURCE_REPORT.resolve()
            else str(arguments.source_report)
        ),
    )
    report = build_mage_25x_capacity_report(
        baseline=baseline,
        daily_camera_hours=arguments.daily_camera_hours,
        headroom=arguments.headroom,
        codec_multiplier=arguments.codec_multiplier,
        decoder_multiplier=arguments.decoder_multiplier,
    )
    payload = canonical_json_bytes(report) + b"\n"
    if arguments.output is None:
        sys.stdout.buffer.write(payload)
        return 0
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(output)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
