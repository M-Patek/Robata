#!/usr/bin/env python3
"""Merge recorded Qwen identity and boundary sidecars without model work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_identity_boundary_merge import (  # noqa: E402
    ProductionIdentityBoundaryMergeError,
    load_json,
    merge_identity_and_boundaries,
    render_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = merge_identity_and_boundaries(
            load_json(args.identity),
            load_json(args.boundary),
            identity_path=str(args.identity),
            boundary_path=str(args.boundary),
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_md.write_text(render_markdown(report), encoding="utf-8")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ProductionIdentityBoundaryMergeError,
    ) as exc:
        print(f"identity-boundary merge failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "windows": report["metrics"]["window_count"],
                "candidate_windows": report["metrics"]["windows_with_candidates"],
                "measured_camera_boundaries": report["metrics"]["measured_camera_boundary_count"],
                "official_quality_status": report["official_quality_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
