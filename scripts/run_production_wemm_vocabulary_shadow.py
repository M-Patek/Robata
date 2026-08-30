#!/usr/bin/env python3
"""Run native WeMM retrieval in the owner-scoped production vocabulary.

Unlike ``run_production_wemm_shadow.py`` this command does not load EPIC class
tables or an EPIC action-pair catalog.  It is a benchmark-local exploratory
route and always writes a non-gold sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_vocabulary import (  # noqa: E402
    ProductionWemmVocabularyError,
    run_production_wemm_vocabulary_shadow,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
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
    parser.add_argument("--fusion", choices=("mean", "rank", "rrf"), default="mean")
    parser.add_argument(
        "--score-normalization",
        choices=("unit", "clip", "cosine", "minmax", "rank", "none"),
        default="unit",
    )
    parser.add_argument("--validate-crcs", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        report = run_production_wemm_vocabulary_shadow(
            manifest,
            vocabulary=args.vocabulary,
            model_directory=args.model_dir,
            frame_count=args.frame_count,
            top_k=args.top_k,
            dimension=args.dimension,
            device=args.device,
            label_variant=args.label_variant,
            max_windows=args.max_windows,
            fusion=args.fusion,
            score_normalization=args.score_normalization,
            validate_crcs=args.validate_crcs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ProductionWemmVocabularyError) as exc:
        print(f"production vocabulary WeMM shadow failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "windows": len(report["windows"]),
                "cameras": report["source"]["camera_count"],
                "vocabulary_labels": report["vocabulary"]["pair_count"],
                "quality": report["quality"]["measurement_status"],
                "production_eligible": report["production_eligible"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
