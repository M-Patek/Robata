#!/usr/bin/env python3
"""Evaluate recorded production WeMM retrieval and Qwen verification sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_qwen_candidate_evaluation import (  # noqa: E402
    ProductionWemmQwenEvaluationError,
    evaluate_wemm_qwen_candidate_verifier,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / ".agent_tmp" / "terra_independent_production_review_4s_16f_20260827.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / ".agent_tmp" / "production_review_candidate_pack_4s_terra_20260828.json",
    )
    parser.add_argument(
        "--joined-verifier",
        type=Path,
        default=ROOT / ".agent_tmp" / "p4_wemm_qwen_join_cam01_compact_20260828.json",
    )
    parser.add_argument(
        "--qwen-sidecar",
        type=Path,
        help="optional raw native Qwen sidecar used only for recorded runtime-cost summary",
    )
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = evaluate_wemm_qwen_candidate_verifier(
            args.reference, args.candidates, args.joined_verifier, args.qwen_sidecar
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, ProductionWemmQwenEvaluationError, ValueError) as exc:
        print(f"production WeMM/Qwen evaluation failed: {exc}", file=sys.stderr)
        return 2
    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "status": report["status"],
                "official_quality_status": report["official_quality_status"],
                "denominator_windows": metrics["denominator_windows"],
                "retrieval": metrics["retrieval"],
                "verifier": metrics["verifier"],
                "cost": metrics.get("cost", report.get("cost")),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
