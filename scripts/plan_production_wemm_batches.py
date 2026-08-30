#!/usr/bin/env python3
"""Create a label-neutral batch plan for production WeMM pre-annotation.

The command reads ZIP central-directory metadata only.  It does not extract
MCAPs, decode frames, invoke a model, or load an EPIC action catalog.  A QA
status mapping may be supplied separately; without one, recordings remain
pending and no model work is scheduled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_wemm_batch_plan import (  # noqa: E402
    DEFAULT_MAX_BATCH_BYTES,
    ProductionWemmBatchPlanError,
    build_production_wemm_batch_plan,
    load_qa_statuses,
    render_markdown,
)


def _ordinals(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid ordinal: {item!r}") from exc
        if number < 0:
            raise argparse.ArgumentTypeError("ordinals must be non-negative")
        result.append(number)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="production MCAP ZIP archive")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--max-batch-gib",
        type=float,
        default=DEFAULT_MAX_BATCH_BYTES / (1024**3),
        help="maximum planned uncompressed bytes per batch (default: 2 GiB)",
    )
    parser.add_argument(
        "--max-items-per-batch",
        type=int,
        default=1,
        help="maximum recordings per batch (default: serial one-recording batches)",
    )
    parser.add_argument(
        "--priority-ordinals",
        type=_ordinals,
        help="comma-separated recording ordinals to schedule first",
    )
    parser.add_argument(
        "--qa-status-json",
        type=Path,
        help="optional JSON object mapping archive members to PASS/WARNING/FAIL/PENDING",
    )
    parser.add_argument(
        "--include-pending-qa",
        action="store_true",
        help="include pending records in a dry-run schedule (not recommended for execution)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.max_batch_gib <= 0:
            raise ProductionWemmBatchPlanError("--max-batch-gib must be positive")
        qa_statuses = load_qa_statuses(args.qa_status_json) if args.qa_status_json else None
        plan = build_production_wemm_batch_plan(
            args.archive,
            max_batch_bytes=max(1, round(args.max_batch_gib * (1024**3))),
            max_items_per_batch=args.max_items_per_batch,
            priority_ordinals=args.priority_ordinals,
            qa_status_by_member=qa_statuses,
            include_pending_qa=args.include_pending_qa,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.output_md.write_text(render_markdown(plan), encoding="utf-8")
    except (OSError, UnicodeError, ProductionWemmBatchPlanError) as exc:
        print(f"production WeMM batch planning failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": plan["status"],
                "recordings": plan["summary"]["recording_count"],
                "scheduled": plan["summary"]["scheduled_recording_count"],
                "pending_qa": plan["summary"]["pending_qa_count"],
                "batches": plan["summary"]["batch_count"],
                "output": str(args.output_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
