"""Execute the deterministic, no-network Mage/Qwen local qualification rehearsal."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.benchmark.local_model_comparison_dry_run import (  # noqa: E402
    run_local_model_comparison_dry_run,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path; stdout always receives the same report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local rehearsal and optionally materialize its JSON artifact."""

    args = _parser().parse_args(argv)
    try:
        report = asyncio.run(run_local_model_comparison_dry_run())
        payload = canonical_json_bytes(report.model_dump(mode="json"))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload + b"\n")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(
            canonical_json_bytes(
                {
                    "ok": False,
                    "code": "LOCAL_MODEL_COMPARISON_DRY_RUN_FAILED",
                    "detail": str(error),
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
