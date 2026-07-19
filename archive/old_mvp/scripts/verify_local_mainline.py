"""Verify a published local fake-model mainline root without model/provider access."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from robata.runtime.verification import LocalMainlineVerificationError, verify_local_mainline_output


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description=(
            "Verify execution evidence, typed run/bundle contracts, and V2 video lineage "
            "for a published local fake-model run."
        )
    )
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = verify_local_mainline_output(Path(args.output))
    except (LocalMainlineVerificationError, OSError, TypeError, ValueError) as error:
        result = {
            "ok": False,
            "error": {"code": "INVALID_OUTPUT", "message": str(error)},
            "provider_requests": 0,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
