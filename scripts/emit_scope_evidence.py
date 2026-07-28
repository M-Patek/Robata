"""Emit the internal P0 scope/evidence register for a canonical profile JSON."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from robata.contracts.hashing import canonical_json_bytes  # noqa: E402
from robata.contracts.measurement_truth import EvidenceClass  # noqa: E402
from robata.durability import sync_directory  # noqa: E402
from robata.runtime.measurement_truth import load_profile_evidence_register  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind a canonical profile to code, workload, policy, identity, and evidence facts."
        )
    )
    parser.add_argument("profile", type=Path, help="canonical profile JSON")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository whose current code/config bytes are fingerprinted",
    )
    parser.add_argument(
        "--evidence-class",
        choices=tuple(item.value for item in EvidenceClass),
        default=EvidenceClass.LOCAL_CONFORMANCE.value,
    )
    parser.add_argument("--provider", help="provider label; defaults to profile provider mode")
    parser.add_argument("--hardware", help="hardware label; defaults to profile runtime facts")
    parser.add_argument(
        "--observed-at",
        help="RFC3339 observation timestamp; defaults to current UTC time",
    )
    parser.add_argument("--output", type=Path, required=True, help="register JSON output")
    return parser


def _atomic_write(path: Path, payload: bytes) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        sync_directory(destination.parent)
    except OSError:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        register = load_profile_evidence_register(
            args.profile.resolve(),
            repository_root=args.repository_root.resolve(),
            evidence_class=EvidenceClass(args.evidence_class),
            provider=args.provider,
            hardware=args.hardware,
            observed_at=args.observed_at,
        )
        _atomic_write(
            args.output,
            canonical_json_bytes(register.model_dump(mode="json")) + b"\n",
        )
    except (OSError, TypeError, ValueError) as error:
        print(
            canonical_json_bytes(
                {"ok": False, "code": "SCOPE_EVIDENCE_FAILED", "detail": str(error)}
            ).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    print(canonical_json_bytes(register.model_dump(mode="json")).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
