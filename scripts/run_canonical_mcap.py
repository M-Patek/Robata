"""Run the retained legacy window canonical composition against one local MCAP.

New Mage-native stream runs use ``scripts/run_local_mage_stream.py``.  This
compatibility entry point requires an explicit ``legacy_window_v1`` profile so
a generic local command cannot silently select the Qwen/window DAG.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.application.canonical.local_composition import (  # noqa: E402
    CanonicalLocalCompositionError,
    run_local_canonical_mcap,
)
from robata.application.canonical.perception_routing import (  # noqa: E402
    LEGACY_QWEN_WINDOW_PROFILE,
    require_explicit_legacy_window_route,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402

DEFAULT_MAX_DURATION_SECONDS = 180


def _positive_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local canonical pipeline from one real six-camera MCAP."
    )
    parser.add_argument("source", metavar="SOURCE", type=Path, help="local MCAP source")
    parser.add_argument(
        "--profile",
        choices=(LEGACY_QWEN_WINDOW_PROFILE,),
        required=True,
        help=("explicit compatibility route; Mage vNext uses scripts/run_local_mage_stream.py"),
    )
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
    parser.add_argument(
        "--max-duration-seconds",
        type=_positive_seconds,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help="analyze at most this many seconds from the recording start",
    )
    return parser


def _write_json(payload: object, *, stream: TextIO = sys.stdout) -> None:
    print(canonical_json_bytes(payload).decode("utf-8"), file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        require_explicit_legacy_window_route(args.profile)
        receipt = run_local_canonical_mcap(
            source_path=args.source,
            mapping_config=args.mapping_config,
            state_dir=args.state_dir,
            run_key=args.run_key,
            allow_unapproved_profile=args.allow_unapproved_profile,
            max_duration_ns=args.max_duration_seconds * 1_000_000_000,
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
