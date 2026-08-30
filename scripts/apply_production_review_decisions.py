#!/usr/bin/env python3
"""Apply explicit reviewer decisions to the production review pack.

The agent segment pack remains a review suggestion.  This command only creates
an ``ACCEPTED`` gold item when a decision file explicitly supplies reviewer
provenance and an ``accept``/``edit``/``split`` decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robata.benchmark.production_review_bridge import (  # noqa: E402
    ProductionReviewBridgeError,
    apply_review_decisions,
    build_decision_template,
    load_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_pack", type=Path)
    parser.add_argument("--blank-pack", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--template", type=Path, help="write an all-pending decision template")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        agent_pack = load_json(str(args.agent_pack))
        if args.template is not None:
            template = build_decision_template(agent_pack)
            args.template.parent.mkdir(parents=True, exist_ok=True)
            args.template.write_text(
                json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        if args.blank_pack is None or args.decisions is None or args.output is None:
            if (
                args.template is not None
                and args.blank_pack is None
                and args.decisions is None
                and args.output is None
            ):
                print(
                    json.dumps(
                        {"template": str(args.template), "status": "PENDING"},
                        ensure_ascii=False,
                    )
                )
                return 0
            parser.error(
                "--blank-pack, --decisions, and --output are required when applying decisions"
            )
        blank_pack = load_json(str(args.blank_pack))
        decisions = load_json(str(args.decisions))
        result = apply_review_decisions(blank_pack, agent_pack, decisions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ProductionReviewBridgeError) as exc:
        print(f"production review decision bridge failed: {exc}", file=sys.stderr)
        return 2
    accepted = sum(
        1 for item in result["items"] if item.get("gold", {}).get("status") == "ACCEPTED"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "items": len(result["items"]),
                "accepted_items": accepted,
                "production_eligible": result.get("production_eligible"),
                "model_predictions_copied": result.get("controls", {}).get(
                    "model_predictions_copied"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
