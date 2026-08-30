#!/usr/bin/env python3
"""Boundedly decode one H.264 frame from each mapped production-shaped camera.

This command is a benchmark-local source-media probe.  It reads the selected
``foxglove.CompressedImage`` topics, records the first frame that PyAV can
decode, and writes a JSON diagnostic.  It does not invoke a model, mutate a
production adapter, or make a production-qualification claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_media_decode_probe import (  # noqa: E402
    ProductionMediaDecodeProbeError,
    probe_production_media,
)

DEFAULT_MAPPING_CONFIG = ROOT / "config" / "genrobot-observed-v0.json"
DEFAULT_OUTPUT = ROOT / ".agent_tmp" / "production_media_decode_probe_20260827.json"


def _read_json(path: Path) -> object:
    """Read one JSON mapping while keeping CLI errors in the probe domain."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProductionMediaDecodeProbeError(
            f"could not read mapping config {path}: {exc}"
        ) from exc
    except UnicodeError as exc:
        raise ProductionMediaDecodeProbeError(f"mapping config is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProductionMediaDecodeProbeError(f"invalid mapping JSON {path}: {exc.msg}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ProductionMediaDecodeProbeError(
            f"could not write probe report {path}: {exc}"
        ) from exc


def run(
    *,
    source_path: Path,
    mapping_config_path: Path = DEFAULT_MAPPING_CONFIG,
    output_path: Path | None = DEFAULT_OUTPUT,
    max_messages_per_camera: int = 120,
    validate_crcs: bool = True,
) -> dict[str, Any]:
    """Run the bounded probe and optionally persist its JSON report."""

    mapping = _read_json(mapping_config_path)
    report = probe_production_media(
        source_path,
        camera_topics=mapping,
        max_messages_per_camera=max_messages_per_camera,
        validate_crcs=validate_crcs,
    )
    if output_path is not None:
        _write_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local MCAP source")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        default=DEFAULT_MAPPING_CONFIG,
        help="JSON camera-topic mapping (default: config/genrobot-observed-v0.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: .agent_tmp/production_media_decode_probe_20260827.json)",
    )
    parser.add_argument(
        "--max-messages-per-camera",
        type=int,
        default=120,
        help="maximum mapped messages examined per camera (default: 120)",
    )
    parser.add_argument(
        "--no-validate-crcs",
        action="store_true",
        help="skip MCAP CRC checks for a faster local diagnostic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(
            source_path=args.source,
            mapping_config_path=args.mapping_config,
            output_path=args.output,
            max_messages_per_camera=args.max_messages_per_camera,
            validate_crcs=not args.no_validate_crcs,
        )
    except ProductionMediaDecodeProbeError as exc:
        print(f"production media decode probe failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": None if args.output is None else str(args.output),
                "status": report["status"],
                "decoded_camera_count": report["decoded_camera_count"],
                "camera_count": report["camera_count"],
                "messages_examined": report["messages_examined"],
                "decode_failures": report["decode_failures"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
