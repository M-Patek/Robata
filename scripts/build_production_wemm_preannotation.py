#!/usr/bin/env python3
"""Build a review-only, open-vocabulary WeMM pre-annotation sidecar.

The command only normalizes an already recorded model observation.  It does
not decode MCAP media, invoke WeMM, read gold, or load the EPIC catalog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_preannotation import (  # noqa: E402
    ProductionWemmPreannotationError,
    build_preannotation_envelope,
    build_review_pack,
    load_json,
    validate_preannotation_envelope,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with source and raw WeMM window proposals")
    parser.add_argument("--output", type=Path, required=True, help="pre-annotation envelope path")
    parser.add_argument("--review-output", type=Path, help="optional human review-pack path")
    parser.add_argument(
        "--candidate-profile",
        default="temporary_phrase_candidates",
        help="provenance label for the temporary phrase candidate source",
    )
    parser.add_argument(
        "--model-invoked",
        action="store_true",
        help="mark that this command consumed a fresh model invocation (default: artifact replay)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = load_json(args.input)
        source = payload.get("source")
        windows = payload.get("windows")
        if not isinstance(source, dict):
            raise ProductionWemmPreannotationError("input.source must be an object")
        if not isinstance(windows, list):
            raise ProductionWemmPreannotationError("input.windows must be an array")
        envelope = build_preannotation_envelope(
            source,
            windows,
            raw_model_output=payload.get("raw_model_output"),
            model=payload.get("model") if isinstance(payload.get("model"), dict) else None,
            candidate_profile=args.candidate_profile,
            model_invoked=args.model_invoked,
        )
        envelope = validate_preannotation_envelope(envelope)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.review_output is not None:
            review = build_review_pack(envelope)
            args.review_output.parent.mkdir(parents=True, exist_ok=True)
            args.review_output.write_text(
                json.dumps(review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ProductionWemmPreannotationError) as exc:
        print(f"production WeMM pre-annotation failed: {exc}", file=sys.stderr)
        return 2
    proposal_count = sum(len(window["proposals"]) for window in envelope["windows"])
    print(
        json.dumps(
            {
                "status": envelope["status"],
                "windows": len(envelope["windows"]),
                "proposals": proposal_count,
                "label_space": envelope["label_space"]["kind"],
                "review_required": True,
                "production_eligible": envelope["production_eligible"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
