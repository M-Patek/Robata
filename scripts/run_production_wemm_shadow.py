#!/usr/bin/env python3
"""Run the benchmark-local WeMM video-to-ontology shadow on a production cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_shadow import (  # noqa: E402
    ProductionWemmShadowError,
    run_production_wemm_shadow,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--verb-classes", type=Path, required=True)
    parser.add_argument("--noun-classes", type=Path, required=True)
    parser.add_argument("--ontology-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--label-variant",
        choices=("canonical", "verb_noun", "natural"),
        default="canonical",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--validate-crcs", action="store_true")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        payload = run_production_wemm_shadow(
            manifest,
            model_directory=args.model_dir,
            verb_classes=args.verb_classes,
            noun_classes=args.noun_classes,
            ontology_pairs=args.ontology_pairs,
            frame_count=args.frame_count,
            top_k=args.top_k,
            dimension=args.dimension,
            device=args.device,
            label_variant=args.label_variant,
            max_windows=args.max_windows,
            validate_crcs=args.validate_crcs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError, ProductionWemmShadowError) as exc:
        print(f"production WeMM shadow failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "windows": len(payload["windows"]),
                "cameras": payload["source"]["camera_count"],
                "quality": payload["quality"]["measurement_status"],
                "production_eligible": payload["production_eligible"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
