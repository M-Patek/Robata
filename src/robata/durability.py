"""Small fail-closed filesystem durability primitives."""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


def sync_directory(path: Path) -> None:
    """Synchronize a directory-entry durability barrier or fail closed."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if _is_windows():
        _sync_windows_directory(path)
        return
    _sync_posix_directory(path)


def _is_windows() -> bool:
    return os.name == "nt"


def _sync_posix_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_windows_directory(path: Path) -> None:
    """Flush directory metadata with a write-capable Windows directory handle."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flush_file_buffers = kernel32.FlushFileBuffers
    flush_file_buffers.argtypes = (wintypes.HANDLE,)
    flush_file_buffers.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    ctypes.set_last_error(0)
    handle = create_file(
        str(path),
        _GENERIC_READ | _GENERIC_WRITE,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    if handle is None or handle == _INVALID_HANDLE_VALUE:
        raise _windows_error(f"cannot open directory for durability sync: {path}")

    try:
        ctypes.set_last_error(0)
        if not flush_file_buffers(handle):
            raise _windows_error(f"cannot flush directory durability barrier: {path}")
    finally:
        ctypes.set_last_error(0)
        if not close_handle(handle) and sys.exc_info()[0] is None:
            raise _windows_error(f"cannot close directory durability handle: {path}")


def _windows_error(message: str) -> OSError:
    error = ctypes.get_last_error()
    return OSError(error or 1, message)
