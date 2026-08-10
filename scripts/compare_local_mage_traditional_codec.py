"""Build the tracked local Mage traditional H.264/HEVC qualification report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.mage_25x import load_provider_v2_local_baseline  # noqa: E402
from robata.benchmark.mage_traditional_codec import (  # noqa: E402
    build_traditional_local_qualification_report,
    load_host_measurement,
    load_traditional_receipt,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402

BASELINE_EXACT_SHA256 = "7298d21fb05f0ecbc4bc1e11481f67abf2c82b4b13380227177edfbbbaa24287"
BASELINE_SEMANTIC_SHA256 = "ea659e3e78243e43e4c1f921ff0898c64f18c4e68993c9c219d2425c8a25b0d8"
DEFAULT_IMAGE_DIGEST = "857ad103f01c1594500f6b6ba300c084d891f9ec6106f7f25de583403ec86cbf"
DEFAULT_BASELINE = (
    REPOSITORY_ROOT / "docs" / "mage-dcvc-provider-v2-local-qualification-2026-08-09.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT / "docs" / "mage-traditional-codec-container-receipt-2026-08-09.json"
)
DEFAULT_SINGLE_RECEIPT = (
    REPOSITORY_ROOT / "docs" / "mage-traditional-codec-single-receipt-2026-08-09.json"
)
DEFAULT_HOST = REPOSITORY_ROOT / "docs" / "mage-traditional-codec-host-measurement-2026-08-09.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "docs" / "mage-traditional-codec-local-qualification-2026-08-09.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--single-receipt", type=Path, default=DEFAULT_SINGLE_RECEIPT)
    parser.add_argument("--host-measurement", type=Path, default=DEFAULT_HOST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-image-digest", default=DEFAULT_IMAGE_DIGEST)
    parser.add_argument("--daily-camera-hours", type=float, default=500.0)
    parser.add_argument("--headroom", type=float, default=1.20)
    return parser


def _reference(path: Path, default: Path) -> str:
    return path.name if path.resolve() == default.resolve() else str(path)


def main() -> int:
    arguments = _parser().parse_args()
    baseline = load_provider_v2_local_baseline(
        path=arguments.baseline,
        expected_exact_sha256=BASELINE_EXACT_SHA256,
        expected_semantic_sha256=BASELINE_SEMANTIC_SHA256,
        source_reference=(
            "docs/mage-dcvc-provider-v2-local-qualification-2026-08-09.json"
            if arguments.baseline.resolve() == DEFAULT_BASELINE.resolve()
            else str(arguments.baseline)
        ),
    )
    receipt = load_traditional_receipt(
        path=arguments.receipt,
        source_reference=(
            "docs/mage-traditional-codec-container-receipt-2026-08-09.json"
            if arguments.receipt.resolve() == DEFAULT_RECEIPT.resolve()
            else str(arguments.receipt)
        ),
    )
    single_receipt = load_traditional_receipt(
        path=arguments.single_receipt,
        source_reference=(
            "docs/mage-traditional-codec-single-receipt-2026-08-09.json"
            if arguments.single_receipt.resolve() == DEFAULT_SINGLE_RECEIPT.resolve()
            else str(arguments.single_receipt)
        ),
    )
    host = load_host_measurement(
        path=arguments.host_measurement,
        expected_image_digest=arguments.expected_image_digest,
        source_reference=(
            "docs/mage-traditional-codec-host-measurement-2026-08-09.json"
            if arguments.host_measurement.resolve() == DEFAULT_HOST.resolve()
            else str(arguments.host_measurement)
        ),
    )
    report = build_traditional_local_qualification_report(
        baseline=baseline,
        baseline_report_path=arguments.baseline,
        receipt=receipt,
        single_receipt=single_receipt,
        host_measurement=host,
        daily_camera_hours=arguments.daily_camera_hours,
        headroom=arguments.headroom,
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report) + b"\n"
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(output)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({"output": str(output), "semantic_sha256": report["semantic_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
