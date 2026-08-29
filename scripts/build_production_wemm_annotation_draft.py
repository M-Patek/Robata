#!/usr/bin/env python3
"""Build an editable production annotation draft from a recorded WeMM pack.

This command is inference-free.  It reads an existing WeMM pre-annotation,
review pack, or aggregate review pack and writes a separate review-only draft;
it never opens media, invokes Qwen/Mage, or writes gold annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_annotation_draft import (  # noqa: E402
    ProductionWemmAnnotationDraftError,
    build_wemm_annotation_draft,
    load_json,
    render_markdown,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="existing WeMM pre-annotation, review-pack, or aggregate JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--compact", action="store_true", help="write compact JSON")
    args = parser.parse_args(argv)
    try:
        report = build_wemm_annotation_draft(load_json(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=None if args.compact else 2,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.output_md is not None:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(render_markdown(report), encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ProductionWemmAnnotationDraftError) as exc:
        print(f"production WeMM annotation draft failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "windows": report["metrics"]["window_count"],
                "segments": report["metrics"]["segment_count"],
                "official_quality_status": report["official_quality_status"],
                "gold_written": report["controls"]["gold_written"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
