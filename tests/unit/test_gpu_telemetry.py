from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from robata.benchmark.gpu_telemetry import (
    NVIDIA_SMI_GPU_TELEMETRY_VERSION,
    GpuTelemetryMeasurementStatus,
    GpuTelemetrySample,
    NvidiaSmiCommandResult,
    NvidiaSmiGpuSampler,
    parse_nvidia_smi_csv,
    summarize_gpu_samples,
)


def _sample(
    *,
    offset_ns: int,
    utilization: float | None,
    memory_used: float | None,
    memory_total: float | None,
    power: float | None,
    temperature: float | None,
) -> GpuTelemetrySample:
    return GpuTelemetrySample(
        wall_clock_unix_ns=1_700_000_000_000_000_000 + offset_ns,
        monotonic_offset_ns=offset_ns,
        query_duration_ns=2_000_000,
        gpu_index=0,
        gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
        utilization_gpu_percent=utilization,
        memory_used_mib=memory_used,
        memory_total_mib=memory_total,
        power_draw_watts=power,
        temperature_celsius=temperature,
    )


def test_parse_nvidia_smi_csv_records_wall_clock_offset_and_partial_values() -> None:
    parsed, errors = parse_nvidia_smi_csv(
        '0, "NVIDIA RTX, Test", 42, 4096, 8192, 75.5, 62\n'
        "1, NVIDIA Test 2, N/A, 1024, 24576, [Not Supported], 55\n",
        wall_clock_unix_ns=1_700_000_000_000_000_000,
        monotonic_offset_ns=123_000_000,
        query_duration_ns=7_000_000,
    )

    assert errors == ()
    assert len(parsed) == 2
    assert parsed[0].gpu_name == "NVIDIA RTX, Test"
    assert parsed[0].wall_clock_unix_ns == 1_700_000_000_000_000_000
    assert parsed[0].monotonic_offset_ns == 123_000_000
    assert parsed[0].query_duration_ns == 7_000_000
    assert parsed[0].utilization_gpu_percent == 42.0
    assert parsed[0].power_draw_watts == 75.5
    assert parsed[1].utilization_gpu_percent is None
    assert parsed[1].power_draw_watts is None


def test_parse_nvidia_smi_csv_keeps_good_rows_when_one_row_is_malformed() -> None:
    parsed, errors = parse_nvidia_smi_csv(
        "0, GPU 0, 50, 100, 200, 25, 60\nmalformed,row\n",
        wall_clock_unix_ns=10,
        monotonic_offset_ns=20,
        query_duration_ns=30,
    )

    assert len(parsed) == 1
    assert errors == ("unexpected nvidia-smi row 2: expected 7 fields, got 2",)


def test_summarize_gpu_samples_reports_mean_max_memory_fraction_and_bounds() -> None:
    summary = summarize_gpu_samples(
        (
            _sample(
                offset_ns=10,
                utilization=20.0,
                memory_used=4_096.0,
                memory_total=8_192.0,
                power=50.0,
                temperature=60.0,
            ),
            _sample(
                offset_ns=30,
                utilization=80.0,
                memory_used=6_144.0,
                memory_total=8_192.0,
                power=None,
                temperature=70.0,
            ),
        )
    )

    assert len(summary) == 1
    device = summary[0]
    assert device.sample_count == 2
    assert device.first_monotonic_offset_ns == 10
    assert device.last_monotonic_offset_ns == 30
    assert device.utilization_gpu_percent_mean == 50.0
    assert device.utilization_gpu_percent_max == 80.0
    assert device.memory_used_mib_mean == 5_120.0
    assert device.memory_used_mib_max == 6_144.0
    assert device.memory_total_mib == 8_192.0
    assert device.memory_used_fraction_mean == pytest.approx(0.625)
    assert device.memory_used_fraction_max == pytest.approx(0.75)
    assert device.power_draw_watts_mean == 50.0
    assert device.power_draw_watts_max == 50.0
    assert device.temperature_celsius_mean == 65.0
    assert device.temperature_celsius_max == 70.0


def test_sampler_collects_immediately_and_concurrent_stop_is_idempotent() -> None:
    queried = threading.Event()
    received_commands: list[tuple[str, ...]] = []

    def command_runner(
        command: tuple[str, ...] | list[str],
        timeout_seconds: float,
    ) -> NvidiaSmiCommandResult:
        received_commands.append(tuple(command))
        assert timeout_seconds == 1.0
        queried.set()
        return NvidiaSmiCommandResult(
            returncode=0,
            stdout="0, NVIDIA GPU, 75, 6000, 8192, 90.25, 72\n",
        )

    sampler = NvidiaSmiGpuSampler(
        interval_seconds=60.0,
        command_timeout_seconds=1.0,
        binary_resolver=lambda _name: "C:\\Windows\\nvidia-smi.exe",
        command_runner=command_runner,
    )
    sampler.start()
    assert queried.wait(timeout=1.0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = tuple(executor.map(lambda _index: sampler.stop(), range(2)))

    assert reports[0] is reports[1]
    report = reports[0]
    assert report.format_version == NVIDIA_SMI_GPU_TELEMETRY_VERSION
    assert report.measurement_status is GpuTelemetryMeasurementStatus.MEASURED
    assert report.query_count == 1
    assert report.sample_count == 1
    assert report.samples[0].utilization_gpu_percent == 75.0
    assert report.summary[0].memory_used_mib_max == 6000.0
    assert report.started_wall_clock_unix_ns <= report.stopped_wall_clock_unix_ns
    assert report.monotonic_duration_ns >= report.samples[0].monotonic_offset_ns
    assert "--format=csv,noheader,nounits" in received_commands[0]
    assert any(item.startswith("--query-gpu=index,name") for item in received_commands[0])
    assert report.to_payload()["measurement_status"] == "MEASURED"


def test_sampler_reports_missing_binary_without_starting_a_thread() -> None:
    sampler = NvidiaSmiGpuSampler(binary_resolver=lambda _name: None)

    sampler.start()
    assert sampler.is_running is False
    report = sampler.stop()

    assert report.measurement_status is GpuTelemetryMeasurementStatus.NOT_MEASURED
    assert report.query_count == 0
    assert report.sample_count == 0
    assert report.summary == ()
    assert report.errors == ("nvidia-smi not found",)


def test_sampler_records_command_failure_without_raising_in_workload_thread() -> None:
    queried = threading.Event()

    def failing_runner(
        _command: tuple[str, ...] | list[str],
        _timeout_seconds: float,
    ) -> NvidiaSmiCommandResult:
        queried.set()
        return NvidiaSmiCommandResult(returncode=9, stdout="", stderr="driver unavailable")

    sampler = NvidiaSmiGpuSampler(
        interval_seconds=60.0,
        binary_resolver=lambda _name: "nvidia-smi",
        command_runner=failing_runner,
    )
    sampler.start()
    assert queried.wait(timeout=1.0)
    report = sampler.stop()

    assert report.measurement_status is GpuTelemetryMeasurementStatus.NOT_MEASURED
    assert report.query_count == 1
    assert report.errors == ("nvidia-smi: driver unavailable",)


def test_sampler_rejects_invalid_lifecycle_and_configuration() -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        NvidiaSmiGpuSampler(interval_seconds=0)

    sampler = NvidiaSmiGpuSampler(binary_resolver=lambda _name: None)
    with pytest.raises(RuntimeError, match="has not been started"):
        sampler.snapshot()
    with pytest.raises(RuntimeError, match="has not been started"):
        sampler.stop()

    sampler.start()
    with pytest.raises(RuntimeError, match="already been started"):
        sampler.start()
