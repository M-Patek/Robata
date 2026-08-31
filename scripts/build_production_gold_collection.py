#!/usr/bin/env python3
"""Build a blank source-bound production gold collection queue.

This command copies only source/window metadata from a production-shaped cohort.
It does not read model/Terra labels, decode media, or establish official gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_gold_collection import (  # noqa: E402
    ProductionGoldCollectionError,
    build_source_bound_gold_collection,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--reviewer-slot",
        action="append",
        dest="reviewer_slots",
        help="Reviewer slot name; may be repeated (default: reviewer_a, reviewer_b).",
    )
    parser.add_argument("--manifest-reference")
    parser.add_argument("--evidence-reference")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        payload = build_source_bound_gold_collection(
            manifest,
            reviewer_slots=args.reviewer_slots,
            manifest_reference=args.manifest_reference or str(args.manifest),
            evidence_reference=args.evidence_reference,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ProductionGoldCollectionError) as exc:
        print(f"production gold collection failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "windows": len(payload["windows"]),
                "reviewer_slots": payload["reviewer_slots"],
                "status": payload["status"],
                "official_gold_status": payload["official_gold_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
