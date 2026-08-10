"""Run one finite Qwen batch benchmark under a hard owned-process deadline.

The wrapper never loads the model itself. It owns exactly one benchmark child,
redirects all child output to a durable log, applies a Windows kill-on-close Job
Object when available, enforces benchmark/overall/shutdown deadlines, and always
waits for or kills the child before returning. Argument vectors are JSON arrays;
no shell command string is evaluated.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_local_qwen_batch_hedge.py"
REPORT_VERSION = "bounded-qwen-batch-benchmark-lifecycle-v1"
EXIT_FAILURE = 2
EXIT_TIMEOUT = 124
EXIT_INTERRUPTED = 130
_SECRET_FLAGS = frozenset({"--api-key", "--authorization", "--password", "--secret", "--token"})


class BoundedQwenBatchBenchmarkError(RuntimeError):
    """The finite local Qwen batch benchmark could not be run safely."""


class _DeadlineExceeded(BoundedQwenBatchBenchmarkError):
    """The hard overall deadline expired."""


class _WindowsJob:
    """Kill direct children and descendants when the owning handle closes."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self) -> None:
        self._handle: int | None = None
        self.mode = "process-group-reap-v1"
        if os.name != "nt":
            return

        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class _BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimitInformation),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(handle)
            raise error
        self._handle = int(handle)
        self._kernel32 = kernel32
        self.mode = "windows-job-kill-on-close-v1"

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        if self._handle is None:
            return
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            raise BoundedQwenBatchBenchmarkError("spawned Windows process has no assignable handle")
        if not self._kernel32.AssignProcessToJobObject(self._handle, int(process_handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle is None:
            return
        self._kernel32.CloseHandle(self._handle)
        self._handle = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not parsed > 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-args-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--benchmark-timeout-seconds", type=_positive_float, default=420.0)
    parser.add_argument("--overall-timeout-seconds", type=_positive_float, default=450.0)
    parser.add_argument("--shutdown-timeout-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--poll-seconds", type=_positive_float, default=0.1)
    parser.add_argument("--benchmark-log", type=Path, default=None)
    return parser


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_report(path: Path, payload: dict[str, Any]) -> str:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical_bytes(payload) + b"\n"
    temporary = resolved.with_name(f".{resolved.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(body)
        os.replace(temporary, resolved)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return _sha256(body)


def _load_argv(path: Path) -> list[str]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise BoundedQwenBatchBenchmarkError(
            f"cannot read argument vector {resolved}: {error}"
        ) from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BoundedQwenBatchBenchmarkError(
            f"argument vector must be a JSON array of strings: {resolved}"
        )
    if any("\x00" in item for item in value):
        raise BoundedQwenBatchBenchmarkError(f"NUL byte is not allowed in {resolved}")
    return value


def _redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for item in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        option = item.split("=", 1)[0].lower()
        if option in _SECRET_FLAGS:
            if "=" in item:
                redacted.append(f"{item.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(item)
                redact_next = True
            continue
        redacted.append(item)
    return redacted


def _spawn_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def _spawn_owned(
    command: list[str],
    *,
    log_handle: BinaryIO,
    containment: _WindowsJob,
) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=REPOSITORY_ROOT,
        shell=False,
        **_spawn_kwargs(),
    )
    try:
        containment.assign(process)
    except BaseException:
        with suppress(OSError, subprocess.TimeoutExpired):
            process.terminate()
            process.wait(timeout=2.0)
        with suppress(OSError):
            process.kill()
        raise
    return process


def _terminate_owned_process(
    process: subprocess.Popen[bytes] | None,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    if process is None:
        return {"action": "NOT_STARTED", "exit_code": None, "reaped": True}
    initial_code = process.poll()
    if initial_code is not None:
        process.wait()
        return {"action": "ALREADY_EXITED", "exit_code": initial_code, "reaped": True}
    action = "TERMINATED"
    try:
        if os.name == "nt":
            process.terminate()
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired):
        action = "KILLED"
        if os.name == "nt":
            with suppress(OSError):
                process.kill()
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=2.0)
    return {"action": action, "exit_code": process.poll(), "reaped": process.poll() is not None}


def _record_event(
    events: list[dict[str, Any]],
    *,
    started_monotonic: float,
    phase: str,
    state: str,
    detail: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "at": _utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
        "phase": phase,
        "state": state,
    }
    if detail is not None:
        event["detail"] = detail
    events.append(event)


def _log_evidence(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"exists": False, "byte_count": None, "exact_sha256": None}
    data = path.read_bytes()
    return {"exists": True, "byte_count": len(data), "exact_sha256": _sha256(data)}


def _validate_configuration(
    arguments: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    if arguments.overall_timeout_seconds <= arguments.shutdown_timeout_seconds:
        raise BoundedQwenBatchBenchmarkError(
            "--overall-timeout-seconds must exceed --shutdown-timeout-seconds"
        )
    if arguments.benchmark_timeout_seconds + arguments.shutdown_timeout_seconds > (
        arguments.overall_timeout_seconds
    ):
        raise BoundedQwenBatchBenchmarkError(
            "benchmark plus shutdown deadlines must fit inside the overall deadline"
        )
    python_executable = arguments.python_executable.expanduser().resolve()
    benchmark_script = arguments.benchmark_script.expanduser().resolve()
    for label, path in (
        ("python executable", python_executable),
        ("benchmark script", benchmark_script),
    ):
        if not path.is_file():
            raise BoundedQwenBatchBenchmarkError(f"{label} does not exist: {path}")
    report_path = arguments.report_json.expanduser().resolve()
    benchmark_log = (
        arguments.benchmark_log.expanduser().resolve()
        if arguments.benchmark_log is not None
        else report_path.with_name(f"{report_path.stem}.benchmark.log")
    )
    if report_path == benchmark_log:
        raise BoundedQwenBatchBenchmarkError("report and benchmark log paths must be distinct")
    return python_executable, benchmark_script, report_path, benchmark_log


def run(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    events: list[dict[str, Any]] = []
    outcome = "ORCHESTRATION_ERROR"
    exit_code = EXIT_FAILURE
    error_detail: dict[str, str] | None = None
    process: subprocess.Popen[bytes] | None = None
    containment: _WindowsJob | None = None
    log_handle: BinaryIO | None = None
    cleanup: dict[str, Any] = {"action": "NOT_STARTED", "exit_code": None, "reaped": True}
    child_return_code: int | None = None
    python_executable: Path | None = None
    benchmark_script: Path | None = None
    report_path = arguments.report_json.expanduser().resolve()
    benchmark_log = report_path.with_name(f"{report_path.stem}.benchmark.log")
    command: list[str] = []
    overall_deadline = started_monotonic + arguments.overall_timeout_seconds

    try:
        python_executable, benchmark_script, report_path, benchmark_log = _validate_configuration(
            arguments
        )
        child_args = _load_argv(arguments.benchmark_args_json)
        command = [str(python_executable), str(benchmark_script), *child_args]
        benchmark_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = benchmark_log.open("wb")
        containment = _WindowsJob()
        process = _spawn_owned(command, log_handle=log_handle, containment=containment)
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="BENCHMARK",
            state="STARTED",
            detail=f"pid={process.pid}",
        )
        benchmark_deadline = min(
            started_monotonic + arguments.benchmark_timeout_seconds,
            overall_deadline - arguments.shutdown_timeout_seconds,
        )
        while True:
            child_return_code = process.poll()
            if child_return_code is not None:
                process.wait()
                break
            now = time.monotonic()
            if now >= started_monotonic + arguments.benchmark_timeout_seconds:
                raise TimeoutError("benchmark timed out")
            if now >= overall_deadline - arguments.shutdown_timeout_seconds:
                raise _DeadlineExceeded("hard overall deadline expired during benchmark")
            time.sleep(max(0.001, min(arguments.poll_seconds, benchmark_deadline - now)))
        if child_return_code == 0:
            outcome = "SUCCEEDED"
            exit_code = 0
        else:
            outcome = "BENCHMARK_FAILED"
            exit_code = EXIT_FAILURE
            error_detail = {
                "type": "ChildProcessError",
                "message": f"benchmark exited with code {child_return_code}",
            }
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="BENCHMARK",
            state="COMPLETED",
            detail=f"exit_code={child_return_code}",
        )
    except KeyboardInterrupt:
        outcome = "INTERRUPTED"
        exit_code = EXIT_INTERRUPTED
        error_detail = {"type": "KeyboardInterrupt", "message": "benchmark interrupted"}
    except TimeoutError as error:
        outcome = "BENCHMARK_TIMEOUT"
        exit_code = EXIT_TIMEOUT
        error_detail = {"type": type(error).__name__, "message": str(error)}
    except _DeadlineExceeded as error:
        outcome = "OVERALL_TIMEOUT"
        exit_code = EXIT_TIMEOUT
        error_detail = {"type": type(error).__name__, "message": str(error)}
    except (BoundedQwenBatchBenchmarkError, OSError, ValueError) as error:
        outcome = "ORCHESTRATION_ERROR"
        exit_code = EXIT_FAILURE
        error_detail = {"type": type(error).__name__, "message": str(error)}
    finally:
        cleanup = _terminate_owned_process(
            process,
            timeout_seconds=arguments.shutdown_timeout_seconds,
        )
        if containment is not None:
            containment.close()
        if log_handle is not None:
            log_handle.close()
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="CLEANUP",
            state="COMPLETED",
            detail=f"benchmark={cleanup['action']}",
        )

    finished_monotonic = time.monotonic()
    redacted_command = _redact_command(command)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "authority": "NON_AUTHORITATIVE_LOCAL_QWEN_BATCH_LIFECYCLE_EVIDENCE",
        "production_eligible": False,
        "outcome": outcome,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_seconds": round(finished_monotonic - started_monotonic, 6),
        "deadline": {
            "benchmark_timeout_seconds": arguments.benchmark_timeout_seconds,
            "overall_timeout_seconds": arguments.overall_timeout_seconds,
            "shutdown_timeout_seconds": arguments.shutdown_timeout_seconds,
            "overall_deadline_exceeded": finished_monotonic > overall_deadline,
        },
        "configuration": {
            "python_executable": str(python_executable) if python_executable else None,
            "benchmark_script": str(benchmark_script) if benchmark_script else None,
            "benchmark_args_json": str(arguments.benchmark_args_json.expanduser().resolve()),
            "benchmark_log": str(benchmark_log),
        },
        "process_ownership": {
            "orchestrator_pid": os.getpid(),
            "benchmark_pid": process.pid if process else None,
            "direct_popen_ownership": True,
            "shell_used": False,
            "containment": containment.mode if containment is not None else "NOT_CREATED",
        },
        "benchmark": {
            "command": redacted_command,
            "command_sha256": _sha256(_canonical_bytes(redacted_command)),
            "exit_code": child_return_code,
            "cleanup": cleanup,
            "log": _log_evidence(benchmark_log),
        },
        "events": events,
        "error": error_detail,
    }
    _write_report(report_path, report)
    return exit_code, report


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    exit_code, report = run(arguments)
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "report": str(arguments.report_json.expanduser().resolve()),
                "wall_seconds": report["wall_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
