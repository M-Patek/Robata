#!/usr/bin/env python3
"""Compare the two Terra review sidecars as non-gold surrogates.

This command performs a JSON-only comparison.  It does not decode media,
invoke a model, access gold labels, or compute a hash/digest.  The resulting
report is explicitly a consistency diagnostic and never a quality score.
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

from robata.benchmark.terra_surrogate_comparison import (  # noqa: E402
    TerraSurrogateComparisonError,
    compare_terra_surrogate_reviews,
    render_markdown,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TerraSurrogateComparisonError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TerraSurrogateComparisonError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirmed",
        type=Path,
        default=Path(".agent_tmp/terra_confirmed_production_review_20260827.json"),
        help="owner/provisional Terra review JSON",
    )
    parser.add_argument(
        "--independent",
        type=Path,
        default=Path(".agent_tmp/terra_independent_production_review_4s_16f_20260827.json"),
        help="independent Terra review JSON",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        type=Path,
        help="optional JSON report path (``--output`` is a compatibility alias)",
    )
    parser.add_argument("--output-md", type=Path, help="optional Markdown report path")
    args = parser.parse_args(argv)

    try:
        report = compare_terra_surrogate_reviews(
            _load(args.confirmed),
            _load(args.independent),
            confirmed_artifact=str(args.confirmed),
            independent_artifact=str(args.independent),
        )
    except TerraSurrogateComparisonError as exc:
        parser.error(str(exc))

    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_markdown(report)
    if args.output_json is None and args.output_md is None:
        print(json_text, end="")
        return 0
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text, encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "quality_claim": report["quality_claim"],
                "common_windows": report["windows"]["common_count"],
                "output_json": str(args.output_json) if args.output_json else None,
                "output_md": str(args.output_md) if args.output_md else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
