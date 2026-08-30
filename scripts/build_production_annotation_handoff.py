#!/usr/bin/env python3
"""Build a review-only production annotation handoff from a structured envelope."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_annotation_handoff import (  # noqa: E402
    ProductionAnnotationHandoffError,
    build_production_annotation_handoff,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structured_envelope", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_production_annotation_handoff(args.structured_envelope)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, ProductionAnnotationHandoffError, ValueError) as exc:
        print(f"production annotation handoff failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "windows": report["metrics"]["window_count"],
                "annotation_candidates": report["metrics"]["annotation_candidate_count"],
                "rejected_claims": report["metrics"]["rejected_claim_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
