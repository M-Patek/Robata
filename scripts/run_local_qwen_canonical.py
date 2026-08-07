"""Run a complete local canonical MCAP with the loopback Qwen adapter."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.application.canonical.local_composition import (  # noqa: E402
    CanonicalLocalCompositionError,
)
from robata.application.canonical.local_real_model import (  # noqa: E402
    LOCAL_QWEN_MODEL_VERSION,
    run_local_qwen_canonical_mcap,
)
from robata.contracts.hashing import canonical_json_bytes  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="real local MCAP source")
    parser.add_argument(
        "--mapping-config",
        type=Path,
        required=True,
        help="exact six-camera topic mapping profile",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=REPOSITORY_ROOT / ".local" / "qwen-canonical",
        help="local SQLite/evidence/outbox state directory",
    )
    parser.add_argument(
        "--run-key",
        default="qwen-local-2026-08-06",
        help="stable key for replaying this local canonical run",
    )
    parser.add_argument(
        "--allow-unapproved-profile",
        action="store_true",
        help="explicitly authorize a development UNAPPROVED mapping profile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional file receiving the same canonical receipt JSON emitted on stdout",
    )
    return parser


def _write_output(path: Path, payload: bytes) -> None:
    destination = path.resolve()
    if destination.exists() and destination.is_dir():
        raise ValueError("receipt output must not be a directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run_local_qwen_canonical_mcap(
            source_path=args.source,
            mapping_config=args.mapping_config,
            state_dir=args.state_dir,
            run_key=args.run_key,
            allow_unapproved_profile=args.allow_unapproved_profile,
        )
        payload = canonical_json_bytes(receipt.model_dump(mode="json"))
        if args.output is not None:
            _write_output(args.output, payload + b"\n")
    except (CanonicalLocalCompositionError, OSError, RuntimeError, TypeError, ValueError) as error:
        code = getattr(error, "code", "LOCAL_QWEN_CANONICAL_FAILED")
        code_text = getattr(code, "value", code)
        print(
            canonical_json_bytes(
                {
                    "ok": False,
                    "code": str(code_text),
                    "detail": str(error),
                    "model_version": LOCAL_QWEN_MODEL_VERSION,
                }
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    print(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
