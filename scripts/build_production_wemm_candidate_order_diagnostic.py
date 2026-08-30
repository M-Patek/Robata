#!/usr/bin/env python3
"""Build a one-window WeMM candidate-order diagnostic artifact.

The command copies one row from a production ambiguity selection and changes
only the presentation order of the selected proposal's ``top_k`` array.  It
does not invoke Qwen or decode media.  Use the resulting ``as_is`` and
``reverse`` files as paired candidates for the existing compact native Qwen
verifier; ``shuffle`` is an optional deterministic third arm.
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

from robata.benchmark.production_wemm_candidate_order_diagnostic import (  # noqa: E402
    ORDER_MODES,
    ProductionWemmCandidateOrderDiagnosticError,
    build_candidate_order_variants,
    render_markdown,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="pilot/full production ambiguity selection JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory receiving one JSON/Markdown file per requested mode",
    )
    parser.add_argument("--window-id", help="window to copy; defaults to the first row")
    parser.add_argument(
        "--proposal-index",
        type=int,
        help="proposal_diagnostics index; required when a row has multiple proposals",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=ORDER_MODES,
        help="presentation mode (repeatable; defaults to as_is, reverse, shuffle)",
    )
    parser.add_argument(
        "--seed",
        default="pilot-order-v1",
        help="local deterministic seed used only by shuffle",
    )
    parser.add_argument(
        "--camera-id",
        help="optional camera scope for the downstream single-camera verifier",
    )
    parser.add_argument(
        "--prefix",
        default="candidate-order",
        help="output filename prefix",
    )
    return parser


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    result = "".join(chars).strip("-")
    return result or "window"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    modes = tuple(args.mode or ORDER_MODES)
    try:
        variants = build_candidate_order_variants(
            args.input,
            modes=modes,
            window_id=args.window_id,
            proposal_index=args.proposal_index,
            seed=args.seed,
            camera_id=args.camera_id,
            source_path=args.input,
        )
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        window_id = str(next(iter(variants.values()))["diagnostic"]["window_id"])
        written: dict[str, str] = {}
        markdown: dict[str, str] = {}
        for mode, report in variants.items():
            stem = f"{args.prefix}-{_slug(window_id)}-{mode}"
            output_json = output_dir / f"{stem}.json"
            output_md = output_dir / f"{stem}.md"
            _write(output_json, report)
            output_md.write_text(render_markdown(report), encoding="utf-8")
            written[mode] = str(output_json)
            markdown[mode] = str(output_md)

        manifest = {
            "format": "robata-production-wemm-candidate-order-diagnostic-pack-v1",
            "authority": "LOCAL_NONPRODUCTION_ONLY",
            "quality_claim": False,
            "official_quality_status": "NOT_MEASURED",
            "model_invoked": False,
            "media_decoded": False,
            "hash_or_digest_computed": False,
            "source_selection": str(args.input),
            "window_id": window_id,
            "camera_id": args.camera_id,
            "seed": args.seed,
            "modes": list(variants),
            "json_artifacts": written,
            "markdown_artifacts": markdown,
            "recommended_pair": {
                "baseline": written.get("as_is"),
                "intervention": written.get("reverse"),
            },
            "notes": [
                "Only proposal_diagnostics[*].top_k array presentation order changes.",
                "Candidate rank, label, and score values remain unchanged.",
                "The selected source interval remains context only, never an action boundary.",
                (
                    "Pass --camera-id or the native verifier's --camera-id for a "
                    "single-camera comparison."
                ),
            ],
        }
        manifest_path = output_dir / f"{args.prefix}-{_slug(window_id)}-manifest.json"
        _write(manifest_path, manifest)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        ProductionWemmCandidateOrderDiagnosticError,
    ) as exc:
        print(f"candidate-order diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "CREATED",
                "window_id": window_id,
                "modes": list(variants),
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "model_invoked": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
