"""Write a label-blind source-interval projection from a P41 Mage input pack.

This command reads only the bounded-case records needed for source pairing.  It
does not load a model, call a GPU, decode media, inspect cache assets, or run
inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robata.benchmark.mage_source_interval_projection import (  # noqa: E402
    MageSourceIntervalProjectionError,
    build_mage_source_interval_projection,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-pack",
        type=Path,
        required=True,
        help="existing P41 label-blind Mage bounded-case input pack",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new source-interval projection JSON path",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.input_pack.expanduser()
    try:
        document = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MageSourceIntervalProjectionError("could not read input-pack JSON") from error
    if not isinstance(document, Mapping) or not all(isinstance(key, str) for key in document):
        raise MageSourceIntervalProjectionError("input-pack JSON must be an object")
    input_pack: Mapping[str, object] = document
    projection = build_mage_source_interval_projection(input_pack)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise MageSourceIntervalProjectionError(f"refusing to overwrite output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(projection, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return projection


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    projection = run(args)
    readiness = projection["mechanical_overlap_readiness"]
    if not isinstance(readiness, Mapping) or not isinstance(readiness.get("status"), str):
        raise MageSourceIntervalProjectionError("projection readiness is invalid")
    print(
        json.dumps(
            {
                "status": readiness["status"],
                "record_count": projection["record_count"],
                "output": str(args.output.expanduser().resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
