"""Offline preflight for a local Robata mainline run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Never

from robata.runtime.preflight import run_preflight

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_CONFIG = REPOSITORY_ROOT / "config" / "genrobot-observed-v0.json"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description=(
            "Check local Python, dependency, mapping, source, and output readiness "
            "without running a model."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mapping-config", type=Path, default=DEFAULT_MAPPING_CONFIG)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--allow-unapproved", action="store_true")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--skip-spec-hash", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result = run_preflight(
            args.source,
            args.output,
            mapping_config=args.mapping_config,
            registry_root=args.registry_root,
            allow_unapproved=args.allow_unapproved,
            spec_path=args.spec,
            verify_spec_hash=not args.skip_spec_hash,
        )
    except (OSError, TypeError, ValueError) as error:
        result = {
            "ok": False,
            "checks": [],
            "warnings": [],
            "error": {"code": "INVALID_ARGUMENT", "message": str(error)},
            "provider_requests": 0,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
