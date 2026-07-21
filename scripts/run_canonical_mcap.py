"""Run the canonical pipeline against one explicitly mapped local MCAP."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.application.canonical.local_composition import (  # noqa: E402
    CanonicalLocalCompositionError,
    run_local_canonical_mcap,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local canonical pipeline from one real six-camera MCAP."
    )
    parser.add_argument("source", metavar="SOURCE", type=Path, help="local MCAP source")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        required=True,
        help="exact six-camera topic mapping profile",
    )
    parser.add_argument(
        "--allow-unapproved-profile",
        action="store_true",
        help="explicitly authorize a development UNAPPROVED mapping profile",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        required=True,
        help="directory for durable local canonical state",
    )
    parser.add_argument(
        "--run-key",
        default="primary",
        help="stable key for replaying one canonical run",
    )
    return parser


def _write_json(payload: object, *, stream: object = sys.stdout) -> None:
    print(canonical_json_bytes(payload).decode("utf-8"), file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_local_canonical_mcap(
            source_path=args.source,
            mapping_config=args.mapping_config,
            state_dir=args.state_dir,
            run_key=args.run_key,
            allow_unapproved_profile=args.allow_unapproved_profile,
        )
    except CanonicalLocalCompositionError as error:
        code = getattr(error.code, "value", error.code)
        _write_json(
            {"ok": False, "code": str(code), "detail": str(error)},
            stream=sys.stderr,
        )
        return 2

    _write_json(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
