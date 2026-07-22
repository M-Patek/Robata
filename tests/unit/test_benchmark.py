from __future__ import annotations

import tracemalloc
from inspect import signature
from uuid import NAMESPACE_URL, uuid5

import pytest

from robata.benchmark import BenchmarkEvidenceContext
from robata.contracts.hashing import semantic_sha256
from robata.runtime.benchmark import (
    BenchmarkSummary,
    ThroughputSample,
    measure_callable,
    measure_callable_with_resources,
    run_repeated,
    summarize_samples,
)


def _digest(label: str) -> str:
    return semantic_sha256({"runtime-benchmark-test": label})


def _context() -> BenchmarkEvidenceContext:
    return BenchmarkEvidenceContext.create(
        benchmark_id=str(uuid5(NAMESPACE_URL, "robata:runtime-benchmark")),
        benchmark_manifest_digest=_digest("benchmark-manifest"),
        governed_corpus_digest=_digest("governed-corpus"),
        ground_truth_manifest_digest=_digest("ground-truth-manifest"),
        grouped_split_manifest_digest=_digest("grouped-split-manifest"),
        data_split="FROZEN_TEST",
        governance_approved=True,
        governance_approval_id="approval-2026-07-21",
        governance_approval_digest=_digest("governance-approval"),
        governance_policy_version="governance-policy-1.0",
    )


def test_throughput_sample_reports_both_units() -> None:
    sample = ThroughputSample(elapsed_ms=120_000, recording_duration_ns=120_000_000_000)

    assert sample.recording_hours_per_wall_hour == pytest.approx(1.0)
    assert sample.camera_video_hours_per_wall_hour == pytest.approx(6.0)
    assert sample.as_dict()["recording_duration_ns"] == "120000000000"


def test_summary_is_local_by_default_and_uses_nearest_rank() -> None:
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
    assert summary.as_dict()["evidence_context"] is None


def test_measured_summary_requires_complete_evidence_context() -> None:
    sample = ThroughputSample(elapsed_ms=100, recording_duration_ns=1_000_000_000)
    context = _context()

    summary = summarize_samples(
        "workload",
        (sample,),
        evidence_context=context,
    )

    assert summary.measurement_status == "MEASURED"
    assert summary.as_dict()["evidence_context"]["context_digest"] == context.context_digest
    assert "certifying" not in signature(summarize_samples).parameters
    assert "corpus_id" not in signature(summarize_samples).parameters

    with pytest.raises(TypeError, match="BenchmarkEvidenceContext"):
        BenchmarkSummary(
            workload_id="workload",
            samples=(sample,),
            evidence_context="approved-corpus",  # type: ignore[arg-type]
        )


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


def test_measure_callable_with_resources_reports_portable_observations() -> None:
    ticks = iter((10.0, 10.125))
    cpu_ticks = iter((20.0, 20.010))
    sample = measure_callable_with_resources(
        lambda: b"result",
        recording_duration_ns=120_000_000_000,
        clock=lambda: next(ticks),
        cpu_clock=lambda: next(cpu_ticks),
    )

    assert sample.throughput.elapsed_ms == 125
    assert sample.cpu_time_ms == pytest.approx(10.0)
    assert sample.peak_tracemalloc_bytes >= 0
    assert sample.as_dict()["recording_duration_ns"] == "120000000000"


def test_measure_callable_with_resources_restores_tracemalloc_after_failure() -> None:
    was_tracing = tracemalloc.is_tracing()

    def workload() -> None:
        raise RuntimeError("workload failed")

    with pytest.raises(RuntimeError, match="workload failed"):
        measure_callable_with_resources(
            workload,
            recording_duration_ns=1,
            clock=lambda: 1.0,
            cpu_clock=lambda: 1.0,
        )

    assert tracemalloc.is_tracing() is was_tracing
