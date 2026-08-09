"""Reusable best-effort NVIDIA GPU telemetry for local qualifications.

The sampler records wall-clock time for correlation with external logs and monotonic
run-relative offsets for interval analysis.  It is deliberately a sidecar utility: the
measurements are operational evidence, never a canonical pipeline contract.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

NVIDIA_SMI_GPU_TELEMETRY_VERSION: Final[str] = "robata-nvidia-smi-gpu-telemetry-v1"
_NVIDIA_SMI_QUERY_FIELDS: Final[str] = (
    "index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
)
_MAX_RECORDED_ERRORS: Final[int] = 50
_UNAVAILABLE_VALUES: Final[frozenset[str]] = frozenset(
    {
        "",
        "n/a",
        "[n/a]",
        "not supported",
        "[not supported]",
        "not available",
        "[not available]",
    }
)


class GpuTelemetryMeasurementStatus(StrEnum):
    """Whether the sampler obtained complete, partial, or no GPU measurements."""

    MEASURED = "MEASURED"
    PARTIAL = "PARTIAL"
    NOT_MEASURED = "NOT_MEASURED"


@dataclass(frozen=True, slots=True)
class NvidiaSmiCommandResult:
    """Minimal command result used to make the sampler deterministic in tests."""

    returncode: int
    stdout: str
    stderr: str = ""


class NvidiaSmiCommandRunner(Protocol):
    """Callable boundary around the external ``nvidia-smi`` process."""

    def __call__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
    ) -> NvidiaSmiCommandResult: ...


@dataclass(frozen=True, slots=True)
class GpuTelemetrySample:
    """One GPU row returned by a single ``nvidia-smi`` poll."""

    wall_clock_unix_ns: int
    monotonic_offset_ns: int
    query_duration_ns: int
    gpu_index: int
    gpu_name: str
    utilization_gpu_percent: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    power_draw_watts: float | None
    temperature_celsius: float | None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "wall_clock_unix_ns": self.wall_clock_unix_ns,
            "monotonic_offset_ns": self.monotonic_offset_ns,
            "query_duration_ns": self.query_duration_ns,
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "utilization_gpu_percent": self.utilization_gpu_percent,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "power_draw_watts": self.power_draw_watts,
            "temperature_celsius": self.temperature_celsius,
        }


@dataclass(frozen=True, slots=True)
class GpuTelemetryDeviceSummary:
    """Per-device aggregate derived from all available numeric samples."""

    gpu_index: int
    gpu_name: str
    sample_count: int
    first_monotonic_offset_ns: int
    last_monotonic_offset_ns: int
    utilization_gpu_percent_mean: float | None
    utilization_gpu_percent_max: float | None
    memory_used_mib_mean: float | None
    memory_used_mib_max: float | None
    memory_total_mib: float | None
    memory_used_fraction_mean: float | None
    memory_used_fraction_max: float | None
    power_draw_watts_mean: float | None
    power_draw_watts_max: float | None
    temperature_celsius_mean: float | None
    temperature_celsius_max: float | None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "sample_count": self.sample_count,
            "first_monotonic_offset_ns": self.first_monotonic_offset_ns,
            "last_monotonic_offset_ns": self.last_monotonic_offset_ns,
            "utilization_gpu_percent_mean": self.utilization_gpu_percent_mean,
            "utilization_gpu_percent_max": self.utilization_gpu_percent_max,
            "memory_used_mib_mean": self.memory_used_mib_mean,
            "memory_used_mib_max": self.memory_used_mib_max,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_fraction_mean": self.memory_used_fraction_mean,
            "memory_used_fraction_max": self.memory_used_fraction_max,
            "power_draw_watts_mean": self.power_draw_watts_mean,
            "power_draw_watts_max": self.power_draw_watts_max,
            "temperature_celsius_mean": self.temperature_celsius_mean,
            "temperature_celsius_max": self.temperature_celsius_max,
        }


@dataclass(frozen=True, slots=True)
class GpuTelemetryReport:
    """Immutable sampler snapshot suitable for a local JSON qualification sidecar."""

    format_version: str
    measurement_status: GpuTelemetryMeasurementStatus
    nvidia_smi_binary: str | None
    sample_interval_seconds: float
    started_wall_clock_unix_ns: int
    stopped_wall_clock_unix_ns: int
    monotonic_duration_ns: int
    query_count: int
    samples: tuple[GpuTelemetrySample, ...]
    summary: tuple[GpuTelemetryDeviceSummary, ...]
    errors: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        """Return the number of per-device rows collected."""

        return len(self.samples)

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "format_version": self.format_version,
            "measurement_status": self.measurement_status.value,
            "nvidia_smi_binary": self.nvidia_smi_binary,
            "sample_interval_seconds": self.sample_interval_seconds,
            "started_wall_clock_unix_ns": self.started_wall_clock_unix_ns,
            "stopped_wall_clock_unix_ns": self.stopped_wall_clock_unix_ns,
            "monotonic_duration_ns": self.monotonic_duration_ns,
            "query_count": self.query_count,
            "sample_count": self.sample_count,
            "summary": [item.to_payload() for item in self.summary],
            "samples": [item.to_payload() for item in self.samples],
            "errors": list(self.errors),
        }


def _default_command_runner(
    command: Sequence[str],
    timeout_seconds: float,
) -> NvidiaSmiCommandResult:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return NvidiaSmiCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _optional_float(value: str) -> float | None:
    normalized = value.strip()
    if normalized.casefold() in _UNAVAILABLE_VALUES:
        return None
    return float(normalized)


def parse_nvidia_smi_csv(
    payload: str,
    *,
    wall_clock_unix_ns: int,
    monotonic_offset_ns: int,
    query_duration_ns: int,
) -> tuple[tuple[GpuTelemetrySample, ...], tuple[str, ...]]:
    """Parse nounits CSV output without failing the entire poll on one bad row."""

    samples: list[GpuTelemetrySample] = []
    errors: list[str] = []
    rows = csv.reader(io.StringIO(payload), skipinitialspace=True)
    for row_number, row in enumerate(rows, start=1):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 7:
            errors.append(
                f"unexpected nvidia-smi row {row_number}: expected 7 fields, got {len(row)}"
            )
            continue
        try:
            gpu_index = int(row[0].strip())
            gpu_name = row[1].strip()
            if not gpu_name:
                raise ValueError("GPU name is empty")
            samples.append(
                GpuTelemetrySample(
                    wall_clock_unix_ns=wall_clock_unix_ns,
                    monotonic_offset_ns=max(0, monotonic_offset_ns),
                    query_duration_ns=max(0, query_duration_ns),
                    gpu_index=gpu_index,
                    gpu_name=gpu_name,
                    utilization_gpu_percent=_optional_float(row[2]),
                    memory_used_mib=_optional_float(row[3]),
                    memory_total_mib=_optional_float(row[4]),
                    power_draw_watts=_optional_float(row[5]),
                    temperature_celsius=_optional_float(row[6]),
                )
            )
        except ValueError as error:
            errors.append(f"could not parse nvidia-smi row {row_number}: {error}")
    return tuple(samples), tuple(errors)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _maximum(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return max(values)


def summarize_gpu_samples(
    samples: Sequence[GpuTelemetrySample],
) -> tuple[GpuTelemetryDeviceSummary, ...]:
    """Build deterministic per-device summaries from possibly partial samples."""

    by_gpu: dict[int, list[GpuTelemetrySample]] = defaultdict(list)
    for sample in samples:
        by_gpu[sample.gpu_index].append(sample)

    summaries: list[GpuTelemetryDeviceSummary] = []
    for gpu_index, device_samples in sorted(by_gpu.items()):
        ordered = sorted(
            device_samples,
            key=lambda item: (item.monotonic_offset_ns, item.wall_clock_unix_ns),
        )
        utilization = [
            value for item in ordered if (value := item.utilization_gpu_percent) is not None
        ]
        memory_used = [value for item in ordered if (value := item.memory_used_mib) is not None]
        memory_total = [value for item in ordered if (value := item.memory_total_mib) is not None]
        memory_fractions = [
            item.memory_used_mib / item.memory_total_mib
            for item in ordered
            if item.memory_used_mib is not None
            and item.memory_total_mib is not None
            and item.memory_total_mib > 0
        ]
        power = [value for item in ordered if (value := item.power_draw_watts) is not None]
        temperature = [value for item in ordered if (value := item.temperature_celsius) is not None]
        summaries.append(
            GpuTelemetryDeviceSummary(
                gpu_index=gpu_index,
                gpu_name=ordered[-1].gpu_name,
                sample_count=len(ordered),
                first_monotonic_offset_ns=ordered[0].monotonic_offset_ns,
                last_monotonic_offset_ns=ordered[-1].monotonic_offset_ns,
                utilization_gpu_percent_mean=_mean(utilization),
                utilization_gpu_percent_max=_maximum(utilization),
                memory_used_mib_mean=_mean(memory_used),
                memory_used_mib_max=_maximum(memory_used),
                memory_total_mib=_maximum(memory_total),
                memory_used_fraction_mean=_mean(memory_fractions),
                memory_used_fraction_max=_maximum(memory_fractions),
                power_draw_watts_mean=_mean(power),
                power_draw_watts_max=_maximum(power),
                temperature_celsius_mean=_mean(temperature),
                temperature_celsius_max=_maximum(temperature),
            )
        )
    return tuple(summaries)


class NvidiaSmiGpuSampler:
    """Thread-safe, single-use background sampler for all visible NVIDIA GPUs."""

    def __init__(
        self,
        *,
        interval_seconds: float = 0.25,
        command_timeout_seconds: float = 5.0,
        binary_resolver: Callable[[str], str | None] = shutil.which,
        command_runner: NvidiaSmiCommandRunner = _default_command_runner,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        thread_name: str = "robata-nvidia-smi-gpu-telemetry",
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if not thread_name:
            raise ValueError("thread_name must not be empty")

        self._interval_seconds = interval_seconds
        self._command_timeout_seconds = command_timeout_seconds
        self._binary = binary_resolver("nvidia-smi")
        self._command_runner = command_runner
        self._wall_clock_ns = wall_clock_ns
        self._monotonic_ns = monotonic_ns
        self._thread_name = thread_name

        self._lock = threading.RLock()
        self._stop_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._started_wall_clock_unix_ns = 0
        self._started_monotonic_ns = 0
        self._query_count = 0
        self._samples: list[GpuTelemetrySample] = []
        self._errors: list[str] = []
        self._final_report: GpuTelemetryReport | None = None

    @property
    def is_running(self) -> bool:
        """Return whether the background worker is alive and has not been stopped."""

        with self._lock:
            return (
                self._started
                and self._final_report is None
                and self._thread is not None
                and self._thread.is_alive()
            )

    def start(self) -> None:
        """Start sampling immediately; a sampler instance cannot be restarted."""

        with self._lock:
            if self._started:
                raise RuntimeError("GPU telemetry sampler has already been started")
            self._started = True
            self._started_wall_clock_unix_ns = self._wall_clock_ns()
            self._started_monotonic_ns = self._monotonic_ns()
            if self._binary is None:
                self._append_error_locked("nvidia-smi not found")
                return
            self._thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> GpuTelemetryReport:
        """Stop the worker and return an immutable final report; repeated calls are safe."""

        with self._stop_lock:
            with self._lock:
                if not self._started:
                    raise RuntimeError("GPU telemetry sampler has not been started")
                if self._final_report is not None:
                    return self._final_report
                self._stop_event.set()
                thread = self._thread

            if thread is not None and thread is not threading.current_thread():
                join_timeout = max(
                    2.0,
                    self._command_timeout_seconds + self._interval_seconds + 1.0,
                )
                thread.join(timeout=join_timeout)

            with self._lock:
                if thread is not None and thread.is_alive():
                    self._append_error_locked("GPU telemetry thread did not stop before timeout")
                stopped_wall_clock_unix_ns = self._wall_clock_ns()
                stopped_monotonic_ns = self._monotonic_ns()
                self._final_report = self._build_report_locked(
                    stopped_wall_clock_unix_ns=stopped_wall_clock_unix_ns,
                    stopped_monotonic_ns=stopped_monotonic_ns,
                )
                return self._final_report

    def snapshot(self) -> GpuTelemetryReport:
        """Return an immutable point-in-time report without stopping the worker."""

        with self._lock:
            if not self._started:
                raise RuntimeError("GPU telemetry sampler has not been started")
            if self._final_report is not None:
                return self._final_report
            return self._build_report_locked(
                stopped_wall_clock_unix_ns=self._wall_clock_ns(),
                stopped_monotonic_ns=self._monotonic_ns(),
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._collect_once()
            if self._stop_event.wait(self._interval_seconds):
                return

    def _collect_once(self) -> None:
        binary = self._binary
        if binary is None:
            return
        query_started_ns = self._monotonic_ns()
        command = (
            binary,
            f"--query-gpu={_NVIDIA_SMI_QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
        )
        try:
            result = self._command_runner(command, self._command_timeout_seconds)
        except Exception as error:  # Best-effort telemetry must not kill the measured workload.
            self._record_query_error(f"{type(error).__name__}: {error}")
            return

        query_finished_ns = self._monotonic_ns()
        observed_wall_clock_unix_ns = self._wall_clock_ns()
        with self._lock:
            if self._final_report is not None:
                return
            self._query_count += 1
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit {result.returncode}"
            self._append_error(f"nvidia-smi: {detail}")
            return

        parsed, errors = parse_nvidia_smi_csv(
            result.stdout,
            wall_clock_unix_ns=observed_wall_clock_unix_ns,
            monotonic_offset_ns=query_finished_ns - self._started_monotonic_ns,
            query_duration_ns=query_finished_ns - query_started_ns,
        )
        with self._lock:
            if self._final_report is not None:
                return
            self._samples.extend(parsed)
            for parse_error in errors:
                self._append_error_locked(parse_error)

    def _record_query_error(self, message: str) -> None:
        with self._lock:
            if self._final_report is not None:
                return
            self._query_count += 1
            self._append_error_locked(message)

    def _append_error(self, message: str) -> None:
        with self._lock:
            if self._final_report is None:
                self._append_error_locked(message)

    def _append_error_locked(self, message: str) -> None:
        if len(self._errors) < _MAX_RECORDED_ERRORS:
            self._errors.append(message)

    def _build_report_locked(
        self,
        *,
        stopped_wall_clock_unix_ns: int,
        stopped_monotonic_ns: int,
    ) -> GpuTelemetryReport:
        samples = tuple(self._samples)
        errors = tuple(self._errors)
        if samples and errors:
            measurement_status = GpuTelemetryMeasurementStatus.PARTIAL
        elif samples:
            measurement_status = GpuTelemetryMeasurementStatus.MEASURED
        else:
            measurement_status = GpuTelemetryMeasurementStatus.NOT_MEASURED
        return GpuTelemetryReport(
            format_version=NVIDIA_SMI_GPU_TELEMETRY_VERSION,
            measurement_status=measurement_status,
            nvidia_smi_binary=self._binary,
            sample_interval_seconds=self._interval_seconds,
            started_wall_clock_unix_ns=self._started_wall_clock_unix_ns,
            stopped_wall_clock_unix_ns=max(
                self._started_wall_clock_unix_ns,
                stopped_wall_clock_unix_ns,
            ),
            monotonic_duration_ns=max(
                0,
                stopped_monotonic_ns - self._started_monotonic_ns,
            ),
            query_count=self._query_count,
            samples=samples,
            summary=summarize_gpu_samples(samples),
            errors=errors,
        )
