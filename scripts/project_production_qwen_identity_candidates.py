#!/usr/bin/env python3
"""Project clean Qwen identity observations into a pending review queue.

This command is benchmark-local and post-hoc.  It does not invoke a model,
decode media, alter the ontology/Mapper/training, or write gold labels.  The
identity sidecar is retained verbatim in the output; candidates remain
pending because the identity-only arm does not measure action boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_identity_probe import (  # noqa: E402
    ProductionIdentityProbeError,
    project_identity_pending_candidates,
    render_identity_candidate_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", "--input", dest="sidecar", type=Path, required=True)
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = project_identity_pending_candidates(
            args.sidecar,
            input_path=str(args.sidecar),
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        output_md.write_text(render_identity_candidate_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ProductionIdentityProbeError) as exc:
        print(f"production identity candidate projection failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "status": report["status"],
                "windows": report["metrics"]["window_count"],
                "candidates": report["metrics"]["annotation_candidate_count"],
                "production_eligible": report["production_eligible"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
