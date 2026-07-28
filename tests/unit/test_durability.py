from __future__ import annotations

import os
from pathlib import Path

import pytest

import robata.durability as durability


class _FakeFunction:
    def __init__(self, result: int) -> None:
        self.result = result
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return self.result


class _FakeKernel32:
    def __init__(self) -> None:
        self.CreateFileW = _FakeFunction(123)
        self.FlushFileBuffers = _FakeFunction(0)
        self.CloseHandle = _FakeFunction(1)


def test_sync_directory_rejects_non_path_values() -> None:
    with pytest.raises(TypeError, match=r"path must be a pathlib\.Path"):
        durability.sync_directory("not-a-path")  # type: ignore[arg-type]


def test_sync_directory_propagates_windows_barrier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(path: Path) -> None:
        assert path == tmp_path
        raise OSError("injected Windows directory flush failure")

    monkeypatch.setattr(durability, "_is_windows", lambda: True)
    monkeypatch.setattr(durability, "_sync_windows_directory", fail)

    with pytest.raises(OSError, match="injected Windows directory flush failure"):
        durability.sync_directory(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="requires Windows last-error APIs")
def test_sync_windows_directory_fails_closed_when_flush_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = _FakeKernel32()
    monkeypatch.setattr(
        durability.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
    )

    with pytest.raises(OSError, match="cannot flush directory durability barrier"):
        durability._sync_windows_directory(tmp_path)

    assert len(kernel32.CreateFileW.calls) == 1
    assert len(kernel32.FlushFileBuffers.calls) == 1


@pytest.mark.skipif(os.name != "nt", reason="requires a Windows directory handle")
def test_sync_directory_flushes_a_real_windows_workspace_directory(tmp_path: Path) -> None:
    durability.sync_directory(tmp_path)
