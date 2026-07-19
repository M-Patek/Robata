"""Windows-compatible local staging helpers with inherited directory ACLs."""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def make_staging_directory(parent: Path, *, prefix: str) -> Path:
    """Create a unique directory without tempfile's restrictive Windows ACL."""

    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(64):
        candidate = parent / f"{prefix}{uuid.uuid4().hex[:16]}"
        try:
            # 0o777 is intentional: Windows uses this to inherit the parent ACL.
            candidate.mkdir(mode=0o777)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not allocate staging directory below {parent}")


def make_temp_file(
    parent: Path,
    *,
    prefix: str,
    suffix: str = "",
) -> tuple[int, Path]:
    """Create an exclusive sibling file and return its open descriptor and path."""

    parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
    for _ in range(64):
        candidate = parent / f"{prefix}{uuid.uuid4().hex[:16]}{suffix}"
        try:
            descriptor = os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        return descriptor, candidate
    raise FileExistsError(f"could not allocate temporary file below {parent}")
