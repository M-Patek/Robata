#!/usr/bin/env python3
"""Materialize bounded six-camera visual review surfaces for a cohort manifest.

The command is intentionally a CPU/local review aid.  It decodes only the
mapped H.264 camera topics, writes contact sheets/thumbnails, and records
source-bound paths.  It does not load a model, infer labels, compute a digest,
or modify the human-review gold pack.
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

from robata.benchmark.production_review_surfaces import (  # noqa: E402
    ProductionReviewSurfacesError,
    build_production_review_surfaces,
    write_production_review_surfaces,
)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProductionReviewSurfacesError(f"could not read manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProductionReviewSurfacesError(f"invalid manifest JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProductionReviewSurfacesError("manifest JSON root must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="bundle directory for thumbnails and contact sheets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON review-surface report path",
    )
    parser.add_argument(
        "--frames-per-camera",
        type=int,
        default=4,
        help="evenly spaced frame targets per camera/window (default: 4)",
    )
    parser.add_argument(
        "--thumbnail-max-side",
        type=int,
        default=320,
        help="maximum thumbnail side in pixels (default: 320)",
    )
    parser.add_argument(
        "--max-messages-per-camera",
        type=int,
        default=5000,
        help="bounded source messages examined per camera (default: 5000)",
    )
    parser.add_argument(
        "--no-validate-crcs",
        action="store_true",
        help="skip MCAP CRC checks for a local diagnostic",
    )
    args = parser.parse_args(argv)
    try:
        manifest = _read_manifest(args.manifest)
        report = build_production_review_surfaces(
            manifest,
            args.output_dir,
            frames_per_camera=args.frames_per_camera,
            thumbnail_max_side=args.thumbnail_max_side,
            max_messages_per_camera=args.max_messages_per_camera,
            validate_crcs=not args.no_validate_crcs,
        )
        write_production_review_surfaces(report, args.output)
    except (OSError, ProductionReviewSurfacesError) as exc:
        print(f"production review surfaces failed: {exc}", file=sys.stderr)
        return 2

    camera_surfaces = sum(len(window["camera_surfaces"]) for window in report["windows"])
    ready_surfaces = sum(
        sum(entry["status"] == "READY" for entry in window["camera_surfaces"])
        for window in report["windows"]
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bundle_root": str(args.output_dir),
                "windows": len(report["windows"]),
                "camera_surfaces": camera_surfaces,
                "ready_surfaces": ready_surfaces,
                "model_invoked": report["controls"]["model_invoked"],
                "machine_assisted_draft_generated": report["controls"][
                    "machine_assisted_draft_generated"
                ],
                "sha_or_digest_computed": report["controls"]["sha_or_digest_computed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
