#!/usr/bin/env python3
"""Plan bounded human-review batches from a production WeMM draft.

This is a read-only planner.  It chunks the existing draft (default ten
windows per batch), preserves WeMM Top-K/margin and source provenance, adds a
blank optional Qwen slot, and leaves every reviewer decision and true action
boundary pending.  It never opens media or starts a model.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_review_queue_batches import (  # noqa: E402
    ProductionReviewQueueBatchError,
    build_review_queue_batches,
    render_markdown,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionReviewQueueBatchError(f"could not read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProductionReviewQueueBatchError(f"JSON root must be an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft", type=Path, help="canonical production WeMM annotation draft")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--qwen-results",
        type=Path,
        help="optional JSON result/sidecar keyed by window_id; absent means NOT_RUN slots",
    )
    args = parser.parse_args(argv)
    try:
        draft = _load(args.draft)
        qwen = _load(args.qwen_results) if args.qwen_results is not None else None
        plan = build_review_queue_batches(
            draft,
            batch_size=args.batch_size,
            draft_path=str(args.draft),
            qwen_results=qwen,
        )
        batches_dir = args.output_dir / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)
        batch_paths: dict[str, str] = {}
        for batch in plan["batches"]:
            batch_id = str(batch["batch_id"])
            batch_path = batches_dir / f"{batch_id}.json"
            batch_path.write_text(
                json.dumps(
                    {
                        "format": "robata-production-review-queue-batch-v1",
                        "authority": plan["authority"],
                        "status": plan["status"],
                        "batch": batch,
                        "review_contract": plan["review_contract"],
                        "controls": plan["controls"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            batch_paths[batch_id] = str(batch_path.relative_to(args.output_dir)).replace("\\", "/")
        # Keep the index lightweight: complete items live in their batch JSON
        # files, while the manifest carries only coverage/order metadata.
        manifest_plan = copy.deepcopy(plan)
        manifest_batches = []
        for batch in manifest_plan["batches"]:
            batch_id = str(batch["batch_id"])
            metadata = {key: value for key, value in batch.items() if key != "items"}
            metadata["batch_file"] = batch_paths[batch_id]
            manifest_batches.append(metadata)
        manifest_plan["batches"] = manifest_batches
        manifest_plan["batch_files"] = batch_paths
        manifest_plan["readme"] = "README.md"
        manifest_path = args.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "README.md").write_text(
            render_markdown(manifest_plan, batch_paths=batch_paths), encoding="utf-8"
        )
    except (OSError, UnicodeError, ProductionReviewQueueBatchError, ValueError) as exc:
        print(f"production review queue planning failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": plan["status"],
                "output_dir": str(args.output_dir),
                "manifest": str(manifest_path),
                "recordings": plan["summary"]["recording_count"],
                "windows": plan["summary"]["window_count"],
                "camera_inputs": plan["summary"]["camera_window_input_count"],
                "batches": plan["summary"]["batch_count"],
                "validation": plan["validation"],
                "model_invoked": plan["controls"]["model_invoked"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
