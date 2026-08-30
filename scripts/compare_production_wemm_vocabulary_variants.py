#!/usr/bin/env python3
"""Compare completed production-native WeMM vocabulary variants.

This command is post-hoc and read-only.  It does not load a model, decode
media, alter ontology/Mapper/UI, or compute identity/hash material.  The
owner review is a scoped surrogate reference, not official production gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_vocabulary_comparison import (  # noqa: E402
    DEFAULT_KS,
    DEFAULT_OWNER_REVIEW,
    DEFAULT_SIDECARS,
    ProductionWemmVariantComparisonError,
    compare_production_wemm_vocabulary_variants,
    render_markdown,
)


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmVariantComparisonError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProductionWemmVariantComparisonError(f"{path} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-review", type=Path, default=ROOT / DEFAULT_OWNER_REVIEW)
    parser.add_argument("--canonical", type=Path, default=ROOT / DEFAULT_SIDECARS["canonical"])
    parser.add_argument("--verb-noun", type=Path, default=ROOT / DEFAULT_SIDECARS["verb_noun"])
    parser.add_argument("--natural", type=Path, default=ROOT / DEFAULT_SIDECARS["natural"])
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="positive retrieval cutoffs (default: 1 3 5 10)",
    )
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = compare_production_wemm_vocabulary_variants(
            _load(args.owner_review),
            {
                "canonical": _load(args.canonical),
                "verb_noun": _load(args.verb_noun),
                "natural": _load(args.natural),
            },
            ks=args.ks,
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
        ProductionWemmVariantComparisonError,
    ) as exc:
        print(f"production WeMM variant comparison failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "status": report["status"],
                "official_quality_status": report["official_quality_status"],
                "eligible_windows": report["reference"]["eligible_window_count"],
                "routes": {
                    name: {
                        "top1": route["metrics"].get("top1", {}).get("rate", 0.0),
                        "top5": route["metrics"].get("top5", {}).get("rate", 0.0),
                        "mrr": route["metrics"].get("mrr", 0.0),
                    }
                    for name, route in report["routes"].items()
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
