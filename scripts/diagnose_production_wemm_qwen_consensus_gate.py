#!/usr/bin/env python3
"""Run the post-hoc candidate-bound camera consensus/margin diagnostic.

The command consumes an existing WeMM/Qwen join report only.  It never loads
the model, decodes media, or changes the recorded verifier decision.
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

from robata.benchmark.production_wemm_qwen_candidate_verifier import (  # noqa: E402
    ProductionWemmQwenCandidateVerifierError,
    diagnose_candidate_bound_consensus_gate,
    render_candidate_bound_consensus_gate_markdown,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionWemmQwenCandidateVerifierError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProductionWemmQwenCandidateVerifierError(f"{path} must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joined",
        type=Path,
        required=True,
        help="existing non-gold WeMM/Qwen join JSON",
    )
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--expected-camera-count", type=int, default=6)
    parser.add_argument("--min-camera-coverage", type=float, default=0.5)
    parser.add_argument("--min-consensus-fraction", type=float, default=0.5)
    parser.add_argument("--min-retrieval-margin", type=float, default=0.01)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = diagnose_candidate_bound_consensus_gate(
            _load(args.joined),
            expected_camera_count=args.expected_camera_count,
            min_camera_coverage=args.min_camera_coverage,
            min_consensus_fraction=args.min_consensus_fraction,
            min_retrieval_margin=args.min_retrieval_margin,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_md.write_text(
            render_candidate_bound_consensus_gate_markdown(report), encoding="utf-8"
        )
    except (OSError, UnicodeError, ProductionWemmQwenCandidateVerifierError, ValueError) as exc:
        print(f"consensus gate diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "windows": report["summary"]["window_count"],
                "gate_accept_count": report["summary"]["gate_accept_count"],
                "gate_abstain_count": report["summary"]["gate_abstain_count"],
                "accuracy_status": report["accuracy_status"],
                "production_eligible": report["production_eligible"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
