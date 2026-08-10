"""Run one Mage endpoint plus one finite local stream benchmark under hard bounds.

This orchestration process owns both child processes.  It writes their output to files,
waits for an explicit READY health response, enforces startup/benchmark/overall deadlines,
and always reaps the endpoint before returning.  On Windows, children are additionally
placed in a kill-on-close Job Object so an abnormal orchestrator exit cannot orphan the
resident model process.

Argument vectors are supplied as JSON arrays rather than shell strings.  The orchestrator
owns endpoint bind and benchmark routing flags; callers must not include ``--host``,
``--port``, ``--log-level``, or ``--endpoint-url`` in those arrays.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_mage_video_endpoint.py"
DEFAULT_BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_local_mage_stream.py"
REPORT_VERSION = "bounded-mage-benchmark-lifecycle-v1"
EXIT_FAILURE = 2
EXIT_TIMEOUT = 124
EXIT_INTERRUPTED = 130
_SECRET_FLAGS = frozenset(
    {
        "--api-key",
        "--authorization",
        "--password",
        "--secret",
        "--token",
    }
)


class BoundedMageBenchmarkError(RuntimeError):
    """The bounded benchmark could not be executed safely."""


class _DeadlineExceeded(BoundedMageBenchmarkError):
    """The hard overall deadline expired."""


class _WindowsJob:
    """Kill-on-close Windows Job Object for direct children and descendants."""

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
            raise BoundedMageBenchmarkError("spawned Windows process has no assignable handle")
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


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer port") from error
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-args-json", type=Path, required=True)
    parser.add_argument("--benchmark-args-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--endpoint-script", type=Path, default=DEFAULT_ENDPOINT_SCRIPT)
    parser.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, required=True)
    parser.add_argument("--health-path", default="/healthz")
    parser.add_argument("--startup-timeout-seconds", type=_positive_float, default=90.0)
    parser.add_argument("--benchmark-timeout-seconds", type=_positive_float, default=360.0)
    parser.add_argument("--overall-timeout-seconds", type=_positive_float, default=480.0)
    parser.add_argument("--shutdown-timeout-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--health-poll-seconds", type=_positive_float, default=0.25)
    parser.add_argument("--health-request-timeout-seconds", type=_positive_float, default=1.0)
    parser.add_argument("--endpoint-log", type=Path, default=None)
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
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical_bytes(payload) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(body)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return _sha256(body)


def _load_argv(path: Path, *, forbidden_flags: frozenset[str]) -> list[str]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise BoundedMageBenchmarkError(
            f"cannot read argument vector {resolved}: {error}"
        ) from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BoundedMageBenchmarkError(
            f"argument vector must be a JSON array of strings: {resolved}"
        )
    for item in value:
        option = item.split("=", 1)[0]
        if option in forbidden_flags:
            raise BoundedMageBenchmarkError(
                f"orchestrator-owned option {option!r} is not allowed in {resolved}"
            )
        if "\x00" in item:
            raise BoundedMageBenchmarkError(f"NUL byte is not allowed in {resolved}")
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


def _validate_configuration(arguments: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    try:
        address = ipaddress.ip_address(arguments.host)
    except ValueError as error:
        raise BoundedMageBenchmarkError("--host must be a numeric loopback address") from error
    if not address.is_loopback:
        raise BoundedMageBenchmarkError("bounded local benchmark requires a loopback --host")
    if not arguments.health_path.startswith("/") or "?" in arguments.health_path:
        raise BoundedMageBenchmarkError("--health-path must be an absolute path without a query")
    if arguments.overall_timeout_seconds <= arguments.shutdown_timeout_seconds:
        raise BoundedMageBenchmarkError(
            "--overall-timeout-seconds must exceed --shutdown-timeout-seconds"
        )

    python_executable = arguments.python_executable.expanduser().resolve()
    endpoint_script = arguments.endpoint_script.expanduser().resolve()
    benchmark_script = arguments.benchmark_script.expanduser().resolve()
    for label, path in (
        ("python executable", python_executable),
        ("endpoint script", endpoint_script),
        ("benchmark script", benchmark_script),
    ):
        if not path.is_file():
            raise BoundedMageBenchmarkError(f"{label} does not exist: {path}")

    report_path = arguments.report_json.expanduser().resolve()
    endpoint_log = (
        arguments.endpoint_log.expanduser().resolve()
        if arguments.endpoint_log is not None
        else report_path.with_name(f"{report_path.stem}.endpoint.log")
    )
    benchmark_log = (
        arguments.benchmark_log.expanduser().resolve()
        if arguments.benchmark_log is not None
        else report_path.with_name(f"{report_path.stem}.benchmark.log")
    )
    paths = (report_path, endpoint_log, benchmark_log)
    if len(set(paths)) != len(paths):
        raise BoundedMageBenchmarkError("report and child log paths must be distinct")
    return python_executable, endpoint_script, benchmark_script, endpoint_log, benchmark_log


def _port_is_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    candidate = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            candidate.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        candidate.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        candidate.close()


def _port_accepts_connections(host: str, port: int, timeout: float = 0.1) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port_closed(host: str, port: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _port_accepts_connections(host, port, timeout=min(0.05, timeout_seconds)):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(0.05, remaining))
    return True


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


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise _DeadlineExceeded("hard overall deadline expired")
    return remaining


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


def _fetch_ready_health(
    *,
    url: str,
    process: subprocess.Popen[bytes],
    startup_deadline: float,
    overall_deadline: float,
    poll_seconds: float,
    request_timeout_seconds: float,
) -> tuple[dict[str, Any], int]:
    attempts = 0
    last_error = "no health response"
    while True:
        if process.poll() is not None:
            raise BoundedMageBenchmarkError(
                f"endpoint exited before readiness with code {process.returncode}"
            )
        now = time.monotonic()
        if now >= overall_deadline:
            raise _DeadlineExceeded("hard overall deadline expired during endpoint startup")
        if now >= startup_deadline:
            raise TimeoutError(f"endpoint startup timed out; last health error: {last_error}")
        attempts += 1
        request_timeout = min(
            request_timeout_seconds,
            max(0.01, startup_deadline - now),
            max(0.01, overall_deadline - now),
        )
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                body = response.read(1024 * 1024)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("health body is not an object")
            if payload.get("status") != "READY" or payload.get("loaded") is not True:
                raise ValueError("health body does not declare READY and loaded=true")
            if process.poll() is not None:
                raise BoundedMageBenchmarkError("endpoint exited while readiness was verified")
            return payload, attempts
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError) as error:
            last_error = f"{type(error).__name__}: {error}"
        delay = min(
            poll_seconds,
            startup_deadline - time.monotonic(),
            overall_deadline - time.monotonic(),
        )
        if delay > 0.0:
            time.sleep(delay)


def _wait_for_benchmark(
    *,
    benchmark: subprocess.Popen[bytes],
    endpoint: subprocess.Popen[bytes],
    benchmark_deadline: float,
    overall_deadline: float,
) -> int:
    while True:
        return_code = benchmark.poll()
        if return_code is not None:
            return return_code
        if endpoint.poll() is not None:
            raise BoundedMageBenchmarkError(
                f"endpoint exited during benchmark with code {endpoint.returncode}"
            )
        now = time.monotonic()
        if now >= overall_deadline:
            raise _DeadlineExceeded("hard overall deadline expired during benchmark")
        if now >= benchmark_deadline:
            raise TimeoutError("benchmark timed out")
        time.sleep(min(0.1, benchmark_deadline - now, overall_deadline - now))


def _terminate_owned_process(
    process: subprocess.Popen[bytes] | None,
    *,
    timeout_seconds: float,
    overall_deadline: float,
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
        grace = min(timeout_seconds, max(0.0, overall_deadline - time.monotonic()))
        if grace > 0.0:
            process.wait(timeout=grace)
        else:
            raise subprocess.TimeoutExpired(process.args, grace)
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
    return {
        "action": action,
        "exit_code": process.poll(),
        "reaped": process.poll() is not None,
    }


def run(arguments: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    overall_deadline = started_monotonic + arguments.overall_timeout_seconds
    events: list[dict[str, Any]] = []
    outcome = "ORCHESTRATION_ERROR"
    exit_code = EXIT_FAILURE
    error_detail: dict[str, str] | None = None
    endpoint: subprocess.Popen[bytes] | None = None
    benchmark: subprocess.Popen[bytes] | None = None
    endpoint_log_handle: BinaryIO | None = None
    benchmark_log_handle: BinaryIO | None = None
    containment: _WindowsJob | None = None
    health_payload: dict[str, Any] | None = None
    health_attempts = 0
    endpoint_cleanup: dict[str, Any] = {"action": "NOT_STARTED", "reaped": True}
    benchmark_cleanup: dict[str, Any] = {"action": "NOT_STARTED", "reaped": True}
    port_preflight_free = False
    post_shutdown_port_closed: bool | None = None
    benchmark_return_code: int | None = None
    endpoint_return_code_before_cleanup: int | None = None

    report_path = arguments.report_json.expanduser().resolve()
    endpoint_log = report_path.with_name(f"{report_path.stem}.endpoint.log")
    benchmark_log = report_path.with_name(f"{report_path.stem}.benchmark.log")
    endpoint_command: list[str] = []
    benchmark_command: list[str] = []
    python_executable: Path | None = None
    endpoint_script: Path | None = None
    benchmark_script: Path | None = None

    try:
        (
            python_executable,
            endpoint_script,
            benchmark_script,
            endpoint_log,
            benchmark_log,
        ) = _validate_configuration(arguments)
        endpoint_args = _load_argv(
            arguments.endpoint_args_json,
            forbidden_flags=frozenset({"--host", "--port", "--log-level"}),
        )
        benchmark_args = _load_argv(
            arguments.benchmark_args_json,
            forbidden_flags=frozenset({"--endpoint-url"}),
        )
        endpoint_url = f"http://{arguments.host}:{arguments.port}"
        health_url = f"{endpoint_url}{arguments.health_path}"
        endpoint_command = [
            str(python_executable),
            str(endpoint_script),
            *endpoint_args,
            "--host",
            arguments.host,
            "--port",
            str(arguments.port),
            "--log-level",
            "warning",
        ]
        benchmark_command = [
            str(python_executable),
            str(benchmark_script),
            *benchmark_args,
            "--endpoint-url",
            endpoint_url,
        ]

        port_preflight_free = _port_is_available(arguments.host, arguments.port)
        if not port_preflight_free:
            outcome = "PORT_IN_USE"
            raise BoundedMageBenchmarkError(
                f"refusing to launch: {arguments.host}:{arguments.port} is already bound"
            )
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="PREFLIGHT",
            state="PASSED",
        )

        endpoint_log.parent.mkdir(parents=True, exist_ok=True)
        benchmark_log.parent.mkdir(parents=True, exist_ok=True)
        endpoint_log_handle = endpoint_log.open("wb")
        benchmark_log_handle = benchmark_log.open("wb")
        containment = _WindowsJob()
        endpoint = _spawn_owned(
            endpoint_command,
            log_handle=endpoint_log_handle,
            containment=containment,
        )
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="ENDPOINT",
            state="SPAWNED",
            detail=f"pid={endpoint.pid}",
        )

        startup_deadline = min(
            started_monotonic + arguments.startup_timeout_seconds,
            overall_deadline,
        )
        try:
            health_payload, health_attempts = _fetch_ready_health(
                url=health_url,
                process=endpoint,
                startup_deadline=startup_deadline,
                overall_deadline=overall_deadline,
                poll_seconds=arguments.health_poll_seconds,
                request_timeout_seconds=arguments.health_request_timeout_seconds,
            )
        except TimeoutError as error:
            outcome = "STARTUP_TIMEOUT"
            exit_code = EXIT_TIMEOUT
            raise error
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="ENDPOINT",
            state="READY",
            detail=f"attempts={health_attempts}",
        )

        _remaining(overall_deadline)
        benchmark = _spawn_owned(
            benchmark_command,
            log_handle=benchmark_log_handle,
            containment=containment,
        )
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="BENCHMARK",
            state="SPAWNED",
            detail=f"pid={benchmark.pid}",
        )
        benchmark_deadline = min(
            time.monotonic() + arguments.benchmark_timeout_seconds,
            overall_deadline,
        )
        try:
            benchmark_return_code = _wait_for_benchmark(
                benchmark=benchmark,
                endpoint=endpoint,
                benchmark_deadline=benchmark_deadline,
                overall_deadline=overall_deadline,
            )
        except TimeoutError as error:
            outcome = "BENCHMARK_TIMEOUT"
            exit_code = EXIT_TIMEOUT
            raise error
        if benchmark_return_code != 0:
            outcome = "BENCHMARK_FAILED"
            raise BoundedMageBenchmarkError(f"benchmark exited with code {benchmark_return_code}")
        outcome = "SUCCEEDED"
        exit_code = 0
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="BENCHMARK",
            state="SUCCEEDED",
        )
    except _DeadlineExceeded as error:
        outcome = "OVERALL_TIMEOUT"
        exit_code = EXIT_TIMEOUT
        error_detail = {"type": type(error).__name__, "message": str(error)}
    except KeyboardInterrupt as error:
        outcome = "INTERRUPTED"
        exit_code = EXIT_INTERRUPTED
        error_detail = {"type": type(error).__name__, "message": "keyboard interrupt"}
    except BaseException as error:  # lifecycle evidence must survive all controlled failures
        if outcome == "ORCHESTRATION_ERROR" and isinstance(error, BoundedMageBenchmarkError):
            message = str(error)
            if message.startswith("endpoint exited before readiness"):
                outcome = "ENDPOINT_STARTUP_FAILED"
            elif message.startswith("endpoint exited during benchmark"):
                outcome = "ENDPOINT_FAILED_DURING_BENCHMARK"
        error_detail = {"type": type(error).__name__, "message": str(error)}
    finally:
        if endpoint is not None:
            endpoint_return_code_before_cleanup = endpoint.poll()
        cleanup_deadline = max(overall_deadline, time.monotonic() + 2.0)
        benchmark_cleanup = _terminate_owned_process(
            benchmark,
            timeout_seconds=arguments.shutdown_timeout_seconds,
            overall_deadline=cleanup_deadline,
        )
        endpoint_cleanup = _terminate_owned_process(
            endpoint,
            timeout_seconds=arguments.shutdown_timeout_seconds,
            overall_deadline=cleanup_deadline,
        )
        if containment is not None:
            containment.close()
        for handle in (benchmark_log_handle, endpoint_log_handle):
            if handle is not None:
                handle.close()
        if endpoint is not None:
            post_shutdown_port_closed = _wait_for_port_closed(
                arguments.host,
                arguments.port,
                timeout_seconds=min(2.0, arguments.shutdown_timeout_seconds),
            )
            if not post_shutdown_port_closed and outcome == "SUCCEEDED":
                outcome = "ENDPOINT_CLEANUP_FAILED"
                exit_code = EXIT_FAILURE
                error_detail = {
                    "type": "BoundedMageBenchmarkError",
                    "message": "endpoint port remained reachable after owned process cleanup",
                }
        _record_event(
            events,
            started_monotonic=started_monotonic,
            phase="CLEANUP",
            state="COMPLETED",
            detail=(
                f"benchmark={benchmark_cleanup['action']}; endpoint={endpoint_cleanup['action']}"
            ),
        )

    finished_monotonic = time.monotonic()
    redacted_endpoint_command = _redact_command(endpoint_command)
    redacted_benchmark_command = _redact_command(benchmark_command)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "authority": "NON_AUTHORITATIVE_LOCAL_QUALIFICATION_LIFECYCLE_EVIDENCE",
        "outcome": outcome,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "wall_seconds": round(finished_monotonic - started_monotonic, 6),
        "deadline": {
            "overall_timeout_seconds": arguments.overall_timeout_seconds,
            "startup_timeout_seconds": arguments.startup_timeout_seconds,
            "benchmark_timeout_seconds": arguments.benchmark_timeout_seconds,
            "shutdown_timeout_seconds": arguments.shutdown_timeout_seconds,
            "overall_deadline_exceeded": finished_monotonic > overall_deadline,
        },
        "configuration": {
            "host": arguments.host,
            "port": arguments.port,
            "health_path": arguments.health_path,
            "python_executable": str(python_executable) if python_executable else None,
            "endpoint_script": str(endpoint_script) if endpoint_script else None,
            "benchmark_script": str(benchmark_script) if benchmark_script else None,
            "endpoint_args_json": str(arguments.endpoint_args_json.expanduser().resolve()),
            "benchmark_args_json": str(arguments.benchmark_args_json.expanduser().resolve()),
            "endpoint_log": str(endpoint_log),
            "benchmark_log": str(benchmark_log),
        },
        "process_ownership": {
            "orchestrator_pid": os.getpid(),
            "endpoint_pid": endpoint.pid if endpoint else None,
            "benchmark_pid": benchmark.pid if benchmark else None,
            "direct_popen_ownership": True,
            "shell_used": False,
            "containment": containment.mode if containment is not None else "NOT_CREATED",
            "port_preflight_free": port_preflight_free,
            "endpoint_alive_at_readiness": health_payload is not None,
            "post_shutdown_port_closed": post_shutdown_port_closed,
        },
        "endpoint": {
            "command": redacted_endpoint_command,
            "command_sha256": _sha256(_canonical_bytes(redacted_endpoint_command)),
            "health_attempts": health_attempts,
            "health_response": health_payload,
            "health_response_sha256": (
                _sha256(_canonical_bytes(health_payload)) if health_payload is not None else None
            ),
            "exit_code_before_cleanup": endpoint_return_code_before_cleanup,
            "cleanup": endpoint_cleanup,
        },
        "benchmark": {
            "command": redacted_benchmark_command,
            "command_sha256": _sha256(_canonical_bytes(redacted_benchmark_command)),
            "exit_code": benchmark_return_code,
            "cleanup": benchmark_cleanup,
        },
        "events": events,
        "error": error_detail,
    }
    _write_report(report_path, report)
    return exit_code, report


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    exit_code, report = run(arguments)
    summary = {
        "outcome": report["outcome"],
        "report": str(arguments.report_json.expanduser().resolve()),
        "wall_seconds": report["wall_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
