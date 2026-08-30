#!/usr/bin/env python3
"""Build the local structured-annotation sidecar from recorded model outputs.

This command is intentionally sidecar-only.  It reads existing WeMM/Qwen
claims (and an optional Mage artifact), performs no model invocation or media
decode, and writes a non-production envelope.  The default paths are the
current 4-second production shadow artifacts; every path can be overridden for
replay or a fresh local cohort.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Keep the standalone script usable from a clean checkout without requiring an
# editable install or an externally configured PYTHONPATH.  This is a local
# benchmark helper; it does not alter the production runtime import path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.production_structured_annotation import (  # noqa: E402
    build_structured_annotation_envelope,
    load_json,
)

# Keep production annotation assembly on the dedicated Terra vocabulary route.
# The older ``production_wemm_shadow`` artifacts use an EPIC/provisional pair
# catalog and are quarantined diagnostics; callers can still pass one
# explicitly for historical replay.
DEFAULT_WEMM = Path(".agent_tmp/production_wemm_production_vocab_4s_20260827.json")
DEFAULT_QWEN = Path(".agent_tmp/production_qwen_shadow_4s_20260827.json")
DEFAULT_OUTPUT = Path(".agent_tmp/production_structured_annotation_envelope_4s_20260827.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wemm", type=Path, default=DEFAULT_WEMM, help="recorded WeMM sidecar")
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN, help="recorded Qwen sidecar")
    parser.add_argument(
        "--mage",
        type=Path,
        default=None,
        help="optional recorded Mage sidecar; omitted means explicit BLOCKED",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="optional cohort manifest used only for source/window geometry",
    )
    parser.add_argument(
        "--source-path",
        default=None,
        help="explicit source media path (recommended when combining sidecars)",
    )
    parser.add_argument("--camera-count", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact", action="store_true", help="write compact JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sidecars: dict[str, object] = {
        "wemm": load_json(args.wemm),
        "qwen": load_json(args.qwen),
    }
    if args.mage is not None:
        sidecars["mage"] = load_json(args.mage)

    envelope = build_structured_annotation_envelope(
        sidecars,
        source_path=args.source_path,
        source_manifest=args.source_manifest,
        camera_count=args.camera_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            envelope,
            handle,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=False,
        )
        handle.write("\n")
    print(
        f"wrote {args.output} ({len(envelope['windows'])} windows; "
        "quality=NOT_MEASURED; gold_included=false)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
