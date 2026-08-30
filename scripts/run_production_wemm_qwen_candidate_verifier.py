#!/usr/bin/env python3
"""Replay the WeMM Top-K → Qwen verifier contract on recorded sidecars.

The command is intentionally light: it does not invoke a model or decode
media.  If a Qwen sidecar already contains ``parsed_verification`` it is
reused; otherwise the recorded raw response is parsed against the WeMM Top-K.
The resulting report is non-gold and keeps every raw candidate/verdict for
human review.
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
    render_markdown,
    verify_wemm_qwen_candidate_sidecars,
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
        "--candidates",
        type=Path,
        default=ROOT / ".agent_tmp" / "production_review_candidate_pack_4s_20260828_r2.json",
    )
    parser.add_argument(
        "--qwen",
        type=Path,
        default=ROOT / ".agent_tmp" / "production_qwen_candidate_verification_4s_20260828.json",
    )
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify_wemm_qwen_candidate_sidecars(_load(args.candidates), _load(args.qwen))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        md = args.output_md or args.output_json.with_suffix(".md")
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, ProductionWemmQwenCandidateVerifierError, ValueError) as exc:
        print(f"production candidate verifier failed: {exc}", file=sys.stderr)
        return 2
    decisions: dict[str, int] = {}
    for row in report["windows"]:
        decision = str(row.get("decision", "abstain"))
        decisions[decision] = decisions.get(decision, 0) + 1
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(md),
                "windows": len(report["windows"]),
                "decisions": decisions,
                "official_quality_status": report["official_quality_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
