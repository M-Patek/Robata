#!/usr/bin/env python3
"""Select WeMM processing windows that warrant a later native-video Qwen pass.

This command is a read-only post-processing step.  It does not load a model,
decode media, read gold/Qwen/Mage artifacts, or mutate the input sidecar.
Thresholds control review workload only and are not quality gates.
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

from robata.benchmark.production_wemm_ambiguity_selector import (  # noqa: E402
    ProductionWemmAmbiguitySelectorError,
    render_markdown,
    select_production_wemm_ambiguities,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="WeMM preannotation sidecar, review pack, aggregate, or containing directory",
    )
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--margin-threshold", type=float, default=0.01)
    parser.add_argument("--camera-consensus-threshold", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--top-k-conflict-threshold",
        type=float,
        help="near-tie threshold for Top-K verb/noun conflict (defaults to margin threshold)",
    )
    parser.add_argument("--include-optional-field-gaps", action="store_true")
    parser.add_argument("--include-unmeasured-boundaries", action="store_true")
    edge_group = parser.add_mutually_exclusive_group()
    edge_group.add_argument(
        "--include-recording-edges",
        dest="include_recording_edges",
        action="store_true",
        help="route first/last known context windows solely because of edge position",
    )
    edge_group.add_argument(
        "--exclude-recording-edges",
        dest="exclude_recording_edges",
        action="store_true",
        help="compatibility alias for the default (do not route edge-only windows)",
    )
    transition_group = parser.add_mutually_exclusive_group()
    transition_group.add_argument(
        "--include-adjacent-transitions",
        dest="include_adjacent_transitions",
        action="store_true",
        help="route neighboring context windows solely because Top-1 changes",
    )
    transition_group.add_argument(
        "--exclude-adjacent-transitions",
        dest="exclude_adjacent_transitions",
        action="store_true",
        help="compatibility alias for the default (do not route transition-only windows)",
    )
    parser.add_argument("--max-selected", type=int)
    parser.add_argument("--expected-camera-count", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    try:
        report = select_production_wemm_ambiguities(
            args.input,
            margin_threshold=args.margin_threshold,
            camera_consensus_threshold=args.camera_consensus_threshold,
            top_k_conflict_threshold=args.top_k_conflict_threshold,
            include_optional_field_gaps=args.include_optional_field_gaps,
            include_unmeasured_boundaries=args.include_unmeasured_boundaries,
            # Edge/transition-only routing is intentionally opt-in.  The
            # ``--exclude-*`` switches are retained as no-op compatibility
            # aliases for callers of the earlier CLI; argparse's mutually
            # exclusive groups reject contradictory include/exclude pairs.
            include_recording_edges=args.include_recording_edges,
            include_adjacent_transitions=args.include_adjacent_transitions,
            max_selected=args.max_selected,
            expected_camera_count=args.expected_camera_count,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_md.write_text(render_markdown(report), encoding="utf-8")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ProductionWemmAmbiguitySelectorError,
        ValueError,
    ) as exc:
        print(f"production WeMM ambiguity selection failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "status": report["status"],
                "input_window_count": report["summary"]["input_window_count"],
                "selected_window_count": report["summary"]["selected_window_count"],
                "official_quality_status": report["official_quality_status"],
                "production_eligible": report["production_eligible"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
