from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from robata.runtime.capacity import (
    BottleneckKind,
    CapacityRegressionPolicy,
    LocalSloPolicy,
    SyntheticLoadProfile,
    SyntheticOutcome,
    compare_capacity_reports,
    evaluate_local_slo,
    generate_synthetic_load,
    simulate_capacity,
)

_HOUR_NS = 3_600_000_000_000


def _profile(**changes: Any) -> SyntheticLoadProfile:
    base = SyntheticLoadProfile(
        version="synthetic-load-v1",
        unit_count=4,
        recording_duration_ns=_HOUR_NS,
        camera_stream_durations_ns=(_HOUR_NS,) * 6,
        arrival_interval_ms=100,
        arrival_batch_size=2,
        service_time_pattern_ms=(100, 200),
        deadline_budget_ms=500,
        failed_ordinals=(2,),
        skipped_ordinals=(3,),
    )
    return replace(base, **changes)


def test_generator_is_deterministic_and_preserves_burst_shape() -> None:
    profile = _profile()

    first = generate_synthetic_load(profile)
    second = generate_synthetic_load(profile)

    assert first == second
    assert tuple(item.arrival_at_ms for item in first) == (0, 0, 100, 100)
    assert tuple(item.service_time_ms for item in first) == (100, 200, 100, 200)
    assert tuple(item.planned_outcome for item in first) == (
        SyntheticOutcome.SUCCEEDED,
        SyntheticOutcome.SUCCEEDED,
        SyntheticOutcome.FAILED,
        SyntheticOutcome.SKIPPED,
    )
    assert len({item.work_id for item in first}) == 4


def test_report_reconciles_outcomes_and_both_hour_denominators() -> None:
    report = simulate_capacity(_profile(), worker_count=2)

    assert report.total_count == 4
    assert report.succeeded_count == 2
    assert report.failed_count == 1
    assert report.skipped_count == 1
    assert report.pending_count == 0
    assert report.offered_recording_hours == pytest.approx(4.0)
    assert report.offered_camera_video_hours == pytest.approx(24.0)
    assert report.completed_recording_hours == pytest.approx(2.0)
    assert report.completed_camera_video_hours == pytest.approx(12.0)
    assert report.recording_hours_per_wall_hour == pytest.approx(36_000.0)
    assert report.camera_video_hours_per_wall_hour == pytest.approx(216_000.0)
    assert report.utilization == pytest.approx(1.0)
    assert report.backlog_peak == 2
    assert report.backlog_end == 0
    assert report.wall_time is not None
    assert report.wall_time.p95_ms == 200
    assert report.terminal_failure_rate == pytest.approx(1 / 3)
    assert report.skipped_rate == pytest.approx(0.25)
    assert report.bottlenecks == (BottleneckKind.RELIABILITY,)
    assert report.evidence_class == "SYNTHETIC_LOCAL"
    assert report.measurement_status == "NOT_MEASURED"
    assert report.production_eligible is False


def test_camera_video_hours_use_measured_stream_durations() -> None:
    profile = _profile(
        unit_count=1,
        camera_stream_durations_ns=(_HOUR_NS, _HOUR_NS // 2),
        failed_ordinals=(),
        skipped_ordinals=(),
    )

    report = simulate_capacity(profile, worker_count=1)

    assert report.offered_recording_hours == pytest.approx(1.0)
    assert report.offered_camera_video_hours == pytest.approx(1.5)
    assert report.completed_camera_video_hours == pytest.approx(1.5)


def test_saturation_exposes_pending_backlog_and_deadline_misses() -> None:
    profile = _profile(
        unit_count=10,
        arrival_interval_ms=0,
        arrival_batch_size=10,
        service_time_pattern_ms=(100,),
        deadline_budget_ms=150,
        observation_window_ms=250,
        failed_ordinals=(),
        skipped_ordinals=(),
    )

    report = simulate_capacity(profile, worker_count=1)

    assert report.succeeded_count == 2
    assert report.pending_count == 8
    assert report.deadline_miss_count == 9
    assert report.backlog_peak == 10
    assert report.backlog_end == 8
    assert report.utilization == pytest.approx(1.0)
    assert BottleneckKind.SERVICE_CAPACITY in report.bottlenecks
    assert BottleneckKind.DEADLINE in report.bottlenecks


def test_local_slo_evaluation_never_promotes_synthetic_evidence() -> None:
    healthy = simulate_capacity(_profile(), worker_count=2)
    passing_policy = LocalSloPolicy(
        version="local-slo-v1",
        maximum_terminal_failure_rate=0.34,
        maximum_skipped_rate=0.25,
        maximum_deadline_miss_rate=0.0,
        maximum_p95_wall_time_ms=250,
    )

    passing = evaluate_local_slo(healthy, passing_policy)

    assert passing.within_local_thresholds is True
    assert passing.violations == ()
    assert passing.measurement_status == "NOT_MEASURED"
    assert passing.production_eligible is False

    saturated = simulate_capacity(
        _profile(
            unit_count=10,
            arrival_interval_ms=0,
            arrival_batch_size=10,
            service_time_pattern_ms=(100,),
            deadline_budget_ms=150,
            observation_window_ms=250,
            failed_ordinals=(),
            skipped_ordinals=(),
        ),
        worker_count=1,
    )
    failing = evaluate_local_slo(
        saturated,
        LocalSloPolicy(
            version="local-slo-v2",
            maximum_terminal_failure_rate=0.0,
            maximum_skipped_rate=0.0,
            maximum_deadline_miss_rate=0.0,
            maximum_p95_wall_time_ms=150,
        ),
    )

    assert failing.within_local_thresholds is False
    assert failing.violations == (
        "DEADLINE_MISS_RATE",
        "P95_WALL_TIME",
        "BACKLOG_NOT_DRAINED",
    )


def test_local_slo_does_not_hide_skipped_work_in_failure_denominator() -> None:
    report = simulate_capacity(
        _profile(
            unit_count=10,
            arrival_interval_ms=0,
            arrival_batch_size=10,
            service_time_pattern_ms=(100,),
            failed_ordinals=(),
            skipped_ordinals=tuple(range(1, 10)),
        ),
        worker_count=1,
    )
    policy = LocalSloPolicy(
        version="local-slo-skip-v1",
        maximum_terminal_failure_rate=1.0,
        maximum_skipped_rate=0.1,
        maximum_deadline_miss_rate=1.0,
        maximum_p95_wall_time_ms=1_000,
    )

    evaluation = evaluate_local_slo(report, policy)

    assert report.terminal_failure_rate == 0.0
    assert report.skipped_rate == pytest.approx(0.9)
    assert evaluation.violations == ("SKIPPED_RATE",)
    assert evaluation.measurement_status == "NOT_MEASURED"


def test_like_for_like_regression_detects_throughput_and_latency() -> None:
    profile = _profile(
        unit_count=6,
        arrival_interval_ms=0,
        arrival_batch_size=6,
        service_time_pattern_ms=(100,),
        deadline_budget_ms=5_000,
        failed_ordinals=(),
        skipped_ordinals=(),
    )
    baseline = simulate_capacity(profile, worker_count=2)
    candidate = simulate_capacity(profile, worker_count=1)
    policy = CapacityRegressionPolicy(
        version="local-capacity-regression-v1",
        minimum_throughput_ratio=0.9,
        maximum_p95_wall_time_ratio=1.1,
        maximum_failure_rate_increase=0.0,
        maximum_deadline_miss_rate_increase=0.0,
        maximum_backlog_end_increase=0,
    )

    result = compare_capacity_reports(baseline, candidate, policy)

    assert result.within_local_thresholds is False
    assert result.regressions == ("THROUGHPUT", "P95_WALL_TIME")
    assert result.throughput_ratio == pytest.approx(0.5)
    assert result.p95_wall_time_ratio == pytest.approx(2.0)
    assert result.measurement_status == "NOT_MEASURED"
    assert result.production_eligible is False


def test_regression_policy_detects_new_deadline_and_backlog_pressure() -> None:
    profile = _profile(
        unit_count=6,
        arrival_interval_ms=0,
        arrival_batch_size=6,
        service_time_pattern_ms=(100,),
        deadline_budget_ms=150,
        observation_window_ms=300,
        failed_ordinals=(),
        skipped_ordinals=(),
    )
    baseline = simulate_capacity(profile, worker_count=2)
    candidate = simulate_capacity(profile, worker_count=1)
    policy = CapacityRegressionPolicy(
        version="local-capacity-regression-v2",
        minimum_throughput_ratio=0.4,
        maximum_p95_wall_time_ratio=2.0,
        maximum_failure_rate_increase=1.0,
        maximum_deadline_miss_rate_increase=0.1,
        maximum_backlog_end_increase=0,
    )

    result = compare_capacity_reports(baseline, candidate, policy)

    assert result.regressions == ("DEADLINE_MISS_RATE", "BACKLOG_END")
    assert result.deadline_miss_rate_increase == pytest.approx(1 / 6)
    assert result.backlog_end_increase == 3
    assert result.measurement_status == "NOT_MEASURED"
    assert result.production_eligible is False


def test_regression_rejects_different_workload_identity() -> None:
    baseline = simulate_capacity(_profile(), worker_count=2)
    candidate = simulate_capacity(
        _profile(version="synthetic-load-v2"),
        worker_count=2,
    )
    policy = CapacityRegressionPolicy(
        version="local-capacity-regression-v1",
        minimum_throughput_ratio=0.9,
        maximum_p95_wall_time_ratio=1.1,
        maximum_failure_rate_increase=0.0,
        maximum_deadline_miss_rate_increase=0.0,
        maximum_backlog_end_increase=0,
    )

    with pytest.raises(ValueError, match="same workload profile"):
        compare_capacity_reports(baseline, candidate, policy)


def test_explicit_zero_work_path_does_not_report_a_capacity_bottleneck() -> None:
    report = simulate_capacity(
        _profile(
            unit_count=2,
            arrival_interval_ms=0,
            arrival_batch_size=2,
            failed_ordinals=(),
            skipped_ordinals=(0, 1),
        ),
        worker_count=1,
    )

    assert report.skipped_count == 2
    assert report.succeeded_count == 0
    assert report.nominal_service_capacity_units_per_hour == 0.0
    assert report.bottlenecks == (BottleneckKind.NONE,)


def test_cutoff_does_not_claim_future_scheduled_work_has_started() -> None:
    report = simulate_capacity(
        _profile(
            unit_count=3,
            arrival_interval_ms=0,
            arrival_batch_size=3,
            service_time_pattern_ms=(100,),
            deadline_budget_ms=1_000,
            observation_window_ms=50,
            failed_ordinals=(),
            skipped_ordinals=(),
        ),
        worker_count=1,
    )

    first, second, third = report.observations
    assert first.started_at_ms == 0
    assert first.worker_ordinal == 0
    assert first.queue_wait_ms == 0
    for observation in (second, third):
        assert observation.started_at_ms is None
        assert observation.worker_ordinal is None
        assert observation.queue_wait_ms == 50
        assert observation.outcome is SyntheticOutcome.PENDING


def test_latency_percentiles_use_nearest_rank_over_terminal_work() -> None:
    report = simulate_capacity(
        _profile(
            unit_count=4,
            arrival_interval_ms=0,
            arrival_batch_size=4,
            service_time_pattern_ms=(100,),
            deadline_budget_ms=1_000,
            failed_ordinals=(),
            skipped_ordinals=(),
        ),
        worker_count=1,
    )

    assert report.wall_time is not None
    assert report.wall_time.count == 4
    assert report.wall_time.p50_ms == 200
    assert report.wall_time.p95_ms == 400
    assert report.wall_time.p99_ms == 400
    assert report.queue_wait is not None
    assert report.queue_wait.p50_ms == 100
    assert report.queue_wait.p95_ms == 300


def test_profile_identity_canonicalizes_outcome_ordinal_sets() -> None:
    left = _profile(failed_ordinals=(1, 2), skipped_ordinals=())
    right = _profile(failed_ordinals=(2, 1), skipped_ordinals=())

    assert left.profile_digest == right.profile_digest
    assert generate_synthetic_load(left) == generate_synthetic_load(right)


def test_cutoff_cannot_exclude_declared_future_arrivals() -> None:
    profile = _profile(
        unit_count=2,
        arrival_interval_ms=100,
        arrival_batch_size=1,
        observation_window_ms=50,
        failed_ordinals=(),
        skipped_ordinals=(),
    )

    with pytest.raises(ValueError, match="complete arrival schedule"):
        simulate_capacity(profile, worker_count=1)


def test_profile_rejects_ambiguous_outcome_ordinals() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        _profile(failed_ordinals=(1,), skipped_ordinals=(1,))
    with pytest.raises(ValueError, match="duplicates"):
        _profile(failed_ordinals=(1, 1), skipped_ordinals=())
