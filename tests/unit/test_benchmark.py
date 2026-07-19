from __future__ import annotations

import pytest

from robata.runtime.benchmark import (
    BenchmarkSummary,
    ThroughputSample,
    measure_callable,
    run_repeated,
    summarize_samples,
)


def test_throughput_sample_reports_both_units() -> None:
    sample = ThroughputSample(elapsed_ms=120_000, recording_duration_ns=120_000_000_000)

    assert sample.recording_hours_per_wall_hour == pytest.approx(1.0)
    assert sample.camera_video_hours_per_wall_hour == pytest.approx(6.0)
    assert sample.as_dict()["recording_duration_ns"] == "120000000000"


def test_summary_is_non_certifying_by_default_and_uses_nearest_rank() -> None:
    summary = summarize_samples(
        "synthetic-fixture",
        (
            ThroughputSample(elapsed_ms=100, recording_duration_ns=120_000_000_000),
            ThroughputSample(elapsed_ms=200, recording_duration_ns=120_000_000_000),
            ThroughputSample(elapsed_ms=300, recording_duration_ns=120_000_000_000),
        ),
    )

    assert isinstance(summary, BenchmarkSummary)
    assert summary.measurement_status == "NOT_MEASURED"
    assert summary.p50_elapsed_ms == 200
    assert summary.p95_elapsed_ms == 300
    assert summary.as_dict()["certifying"] is False


def test_certifying_summary_requires_corpus_id() -> None:
    sample = ThroughputSample(elapsed_ms=100, recording_duration_ns=1_000_000_000)

    with pytest.raises(ValueError, match="corpus_id"):
        summarize_samples("workload", (sample,), certifying=True)

    summary = summarize_samples(
        "workload",
        (sample,),
        certifying=True,
        corpus_id="approved-corpus-v1",
    )
    assert summary.measurement_status == "CERTIFYING"


def test_measure_callable_uses_injected_clock() -> None:
    ticks = iter((10.0, 10.125))
    sample = measure_callable(
        lambda: None,
        recording_duration_ns=120_000_000_000,
        clock=lambda: next(ticks),
    )

    assert sample.elapsed_ms == 125
    assert sample.recording_hours_per_wall_hour == pytest.approx(960.0)


def test_invalid_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="elapsed_ms"):
        ThroughputSample(elapsed_ms=0, recording_duration_ns=1)


def test_run_repeated_excludes_warmups_and_preserves_sample_count() -> None:
    calls = 0
    ticks = iter(
        (
            0.0,
            0.1,
            1.0,
            1.2,
            2.0,
            2.4,
            3.0,
            3.3,
        )
    )

    def workload() -> None:
        nonlocal calls
        calls += 1

    summary = run_repeated(
        workload,
        workload_id="repeatable",
        recording_duration_ns=120_000_000_000,
        iterations=3,
        warmups=1,
        clock=lambda: next(ticks),
    )

    assert calls == 4
    assert summary.measurement_status == "NOT_MEASURED"
    assert len(summary.samples) == 3
    assert tuple(sample.elapsed_ms for sample in summary.samples) == (100, 200, 400)


def test_run_repeated_rejects_zero_iterations() -> None:
    with pytest.raises(ValueError, match="iterations"):
        run_repeated(
            lambda: None,
            workload_id="invalid",
            recording_duration_ns=1,
            iterations=0,
        )
