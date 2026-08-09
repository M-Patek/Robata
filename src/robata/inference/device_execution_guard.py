"""Cooperative cross-process execution guard for one accelerator device.

The guard is deliberately operational state: its path and ownership never enter
inference, codec-cache, or result identity.  A one-byte sentinel makes Windows byte
locking deterministic; POSIX uses an advisory exclusive file lock on the same file.
The file is never treated as an owner record, so an abandoned file is harmless and
kernel lock ownership is released automatically when a process exits.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol


class DeviceExecutionGuardError(RuntimeError):
    """The cooperative device-execution guard could not be used safely."""


class DeviceExecutionGuardBusy(DeviceExecutionGuardError):
    """Another process currently owns the guarded accelerator lane."""


class DeviceExecutionGuard(Protocol):
    """Operational exclusion boundary shared by codec preparation and generation."""

    def hold(self) -> AbstractContextManager[None]:
        """Acquire exclusive ownership for one bounded accelerator operation."""


class ExclusiveFileDeviceGuard:
    """Non-blocking advisory lock over a stable file and sentinel byte.

    Every participant must construct this class with the same path.  Acquisition is
    intentionally non-blocking: admission fails closed instead of queueing invisible
    work inside a worker process.  Merely finding the lock file does not indicate an
    owner; only the kernel-held lock does, which makes an old file safe after a crash.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and self._path.is_dir():
            raise DeviceExecutionGuardError("device guard path must not be a directory")

    @property
    def path(self) -> Path:
        """Return the canonical operational lock path."""

        return self._path

    @contextmanager
    def hold(self) -> Iterator[None]:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            stream = self._path.open("a+b")
        except OSError as error:
            raise DeviceExecutionGuardBusy("could not open the device guard file") from error
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            _lock_file_nonblocking(stream)
            try:
                yield
            finally:
                _unlock_file(stream)
        finally:
            stream.close()


def _lock_file_nonblocking(stream: Any) -> None:
    try:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError) as error:
        raise DeviceExecutionGuardBusy("device guard is already held or unavailable") from error


def _unlock_file(stream: Any) -> None:
    try:
        stream.seek(0)
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        # The operation is already complete.  Closing the descriptor below is the
        # final stale-safe release boundary even when explicit unlock reports an
        # operating-system error.
        pass


__all__ = [
    "DeviceExecutionGuard",
    "DeviceExecutionGuardBusy",
    "DeviceExecutionGuardError",
    "ExclusiveFileDeviceGuard",
]
