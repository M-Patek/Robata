"""Emit a local-only P15 qualification package from real local artifact files."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.p15_emission import emit_local_p15_qualification_package  # noqa: E402
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a local P0 scope register and explicit local artifact manifest "
            "into a P15 package."
        )
    )
    parser.add_argument("scope", type=Path, help="LOCAL_CONFORMANCE scope evidence register JSON")
    parser.add_argument("manifest", type=Path, help="local P15 qualification manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="P15 package JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local-only P15 package emitter."""

    args = _parser().parse_args(argv)
    try:
        package = emit_local_p15_qualification_package(
            scope_path=args.scope,
            manifest_path=args.manifest,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            canonical_json_bytes(
                {"ok": False, "code": "P15_QUALIFICATION_EMISSION_FAILED", "detail": str(error)}
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    print(canonical_json_bytes(package.as_dict()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
