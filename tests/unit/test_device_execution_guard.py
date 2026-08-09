from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from robata.inference.device_execution_guard import (
    DeviceExecutionGuardBusy,
    ExclusiveFileDeviceGuard,
)


def _hold_until_released(path: str, ready: Any, release: Any) -> None:
    with ExclusiveFileDeviceGuard(Path(path)).hold():
        ready.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("test guard holder timed out")


def _exit_while_holding(path: str, ready: Any) -> None:
    with ExclusiveFileDeviceGuard(Path(path)).hold():
        ready.set()
        os._exit(0)


def test_guard_is_cross_process_nonblocking_and_reusable(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    guard_path = tmp_path / "gpu-0.lock"
    process = context.Process(
        target=_hold_until_released,
        args=(str(guard_path), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=10.0)
        with (
            pytest.raises(DeviceExecutionGuardBusy, match="already held or unavailable"),
            ExclusiveFileDeviceGuard(guard_path).hold(),
        ):
            raise AssertionError("conflicting process acquired the device guard")
    finally:
        release.set()
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    assert process.exitcode == 0

    with ExclusiveFileDeviceGuard(guard_path).hold():
        pass
    assert guard_path.read_bytes().startswith(b"\0")


def test_guard_ignores_stale_file_and_kernel_releases_after_process_exit(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    guard_path = tmp_path / "gpu-0.lock"
    guard_path.write_bytes(b"\0stale-operational-bytes")

    with ExclusiveFileDeviceGuard(guard_path).hold():
        pass

    ready = context.Event()
    process = context.Process(target=_exit_while_holding, args=(str(guard_path), ready))
    process.start()
    assert ready.wait(timeout=10.0)
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
    assert process.exitcode == 0
    assert guard_path.exists()

    # Lock-file existence is not ownership. A kernel-released lock is immediately
    # reusable even when the previous owner did not execute __exit__.
    with ExclusiveFileDeviceGuard(guard_path).hold():
        pass


def test_dcvc_worker_reexports_the_single_shared_guard_implementation() -> None:
    from robata.inference import mage_dcvc_preparation_worker

    assert mage_dcvc_preparation_worker.ExclusiveFileDeviceGuard is ExclusiveFileDeviceGuard


def test_guard_rejects_directory_path(tmp_path: Path) -> None:
    from robata.inference.device_execution_guard import DeviceExecutionGuardError

    with pytest.raises(DeviceExecutionGuardError, match="must not be a directory"):
        ExclusiveFileDeviceGuard(tmp_path)
