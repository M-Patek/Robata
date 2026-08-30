#!/usr/bin/env python3
"""Verify a recorded production structured-annotation envelope offline."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_structured_evidence_verifier import (  # noqa: E402
    StructuredEvidenceVerifierError,
    render_markdown,
    verify_production_structured_evidence,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structured_input",
        type=Path,
        help="structured annotation envelope or recorded Qwen sidecar JSON",
    )
    parser.add_argument("--output", type=Path, required=True, help="verification report JSON path")
    parser.add_argument("--output-md", type=Path, help="optional Markdown report path")
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.structured_input.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise StructuredEvidenceVerifierError("structured input JSON root must be an object")
        report = verify_production_structured_evidence(value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_md = args.output_md
        if output_md is not None:
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, StructuredEvidenceVerifierError) as exc:
        print(f"production structured evidence verification failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "windows": report["metrics"]["window_count"],
                "review_required_windows": report["metrics"]["review_required_window_count"],
                "abstained_windows": report["metrics"]["abstained_window_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
