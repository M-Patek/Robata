#!/usr/bin/env python3
"""Validate a local WeMM annotation draft without invoking any model.

The audit is intentionally lightweight: it checks the editable draft shape,
provenance/QA slots, pending reviewer state, and the invariant that context
windows are not silently converted into action boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_annotation_draft import (  # noqa: E402
    ProductionWemmAnnotationDraftError,
    load_json,
    validate_wemm_annotation_draft,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_wemm_annotation_draft(load_json(args.draft))
    except (OSError, UnicodeError, json.JSONDecodeError, ProductionWemmAnnotationDraftError) as exc:
        print(f"production WeMM annotation draft audit failed: {exc}", file=sys.stderr)
        return 2
    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "status": "VALID",
                "format": report["format"],
                "windows": metrics["window_count"],
                "segments": metrics["segment_count"],
                "qa_status_counts": metrics.get("qa_status_counts", {}),
                "source_preflight_status_counts": metrics.get("source_preflight_status_counts", {}),
                "official_quality_status": report["official_quality_status"],
                "production_eligible": report["production_eligible"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
