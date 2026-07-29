from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from robata.runtime.capacity import (
    BackpressureControllerKind,
    BackpressureObservation,
    BackpressureRecordingBacklog,
    BackpressureScenarioKind,
    BackpressureScenarioReport,
    BackpressureStabilityReport,
    BottleneckKind,
    CapacityEvidenceClass,
    CapacityRegressionPolicy,
    LocalSloPolicy,
    MeasuredCapacityComparisonKind,
    MeasuredCapacityInput,
    MeasuredCapacityStatus,
    ProviderMode,
    SyntheticLoadProfile,
    SyntheticOutcome,
    WorkerScalingReport,
    build_backpressure_stability_report,
    build_measured_capacity_report,
    build_worker_scaling_report,
    compare_backpressure_stability_reports,
    compare_capacity_reports,
    compare_fixed_and_adaptive_backpressure,
    compare_measured_capacity_reports,
    evaluate_local_slo,
    generate_synthetic_load,
    required_worker_count_for_rtf,
    run_worker_scaling,
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


def _measured_input(**changes: Any) -> MeasuredCapacityInput:
    base = MeasuredCapacityInput(
        workload_fingerprint="a" * 64,
        evidence_class=CapacityEvidenceClass.LOCAL_CONFORMANCE,
        provider_mode=ProviderMode.LOCAL_OFFLINE_FIXTURE,
        execution_mode="FRESH",
        recording_count=1,
        recording_worker_count=1,
        camera_count=6,
        recording_duration_ns=_HOUR_NS,
        wall_time_ns=_HOUR_NS // 2,
        windows=4,
        unique_images=10,
        coarse_unique_images=8,
        dense_unique_images=4,
        provider_images=25,
        logical_calls=5,
        call_parts=8,
        call_splits=3,
        http_requests=0,
        retries=2,
        batches=2,
        batch_requests=5,
        input_tokens=1_000,
        output_tokens=250,
        output_token_responses=5,
        dense_logical_calls=2,
        dense_provider_images=5,
    )
    return replace(base, **changes)


def test_measured_capacity_keeps_recording_camera_and_provider_units_distinct() -> None:
    report = build_measured_capacity_report(_measured_input())

    assert report.measurement_status is MeasuredCapacityStatus.AVAILABLE
    assert report.unavailable_reasons == ()
    assert report.recording_hours == pytest.approx(1.0)
    assert report.camera_hours == pytest.approx(6.0)
    assert report.wall_hours == pytest.approx(0.5)
    assert report.recording_hours_per_wall_hour == pytest.approx(2.0)
    assert report.camera_hours_per_wall_hour == pytest.approx(12.0)
    assert report.provider_images_per_unique_image == pytest.approx(2.5)
    assert report.unique_images_per_camera_hour == pytest.approx(10 / 6)
    assert report.effective_fps_per_camera == pytest.approx(10 / (6 * 3_600))
    assert report.windows_per_recording_hour == pytest.approx(4.0)
    assert report.logical_calls_per_window == pytest.approx(1.25)
    assert report.windows_per_batch == pytest.approx(2.0)
    assert report.logical_calls_per_provider_image == pytest.approx(0.2)
    assert report.call_splits_per_logical_call == pytest.approx(0.6)
    assert report.call_parts_per_logical_call == pytest.approx(1.6)
    assert report.requests_per_batch == pytest.approx(2.5)
    assert report.dense_logical_call_fraction == pytest.approx(0.4)
    assert report.dense_upgrade_fraction == pytest.approx(0.4)
    assert report.dense_provider_image_fraction == pytest.approx(0.2)
    assert report.call_splits_per_wall_hour == pytest.approx(6.0)
    assert report.retries_per_wall_hour == pytest.approx(4.0)
    assert report.batches_per_wall_hour == pytest.approx(4.0)
    assert report.batch_requests_per_wall_hour == pytest.approx(10.0)
    assert report.input_tokens_per_wall_hour == pytest.approx(2_000.0)
    assert report.output_tokens_per_wall_hour == pytest.approx(500.0)
    assert report.output_token_responses_per_wall_hour == pytest.approx(10.0)
    assert report.dense_logical_calls_per_wall_hour == pytest.approx(4.0)
    assert report.dense_provider_images_per_wall_hour == pytest.approx(10.0)
    assert report.production_eligible is False


def test_measured_capacity_allows_one_response_per_split_call_part() -> None:
    report = build_measured_capacity_report(
        _measured_input(
            logical_calls=1,
            call_parts=2,
            call_splits=1,
            dense_logical_calls=1,
            output_token_responses=2,
        )
    )

    assert report.logical_calls == 1
    assert report.call_parts == 2
    assert report.output_token_responses == 2


def test_measured_capacity_refuses_rates_without_workload_duration_or_provider_mode() -> None:
    report = build_measured_capacity_report(
        _measured_input(
            provider_mode=ProviderMode.UNKNOWN,
            recording_duration_ns=None,
        )
    )

    assert report.measurement_status is MeasuredCapacityStatus.NOT_AVAILABLE
    assert report.unavailable_reasons == (
        "MISSING_PROVIDER_MODE",
        "MISSING_WORKLOAD_DURATION",
    )
    assert report.recording_hours_per_wall_hour is None
    assert report.camera_hours_per_wall_hour is None
    assert report.logical_calls_per_wall_hour is None
    assert report.windows_per_wall_hour is None
    assert report.call_splits_per_wall_hour is None
    assert report.effective_fps_per_camera is None
    assert report.retries_per_wall_hour is None
    assert report.input_tokens_per_wall_hour is None


def test_measured_capacity_comparison_labels_fresh_replay_and_worker_scaling() -> None:
    baseline = build_measured_capacity_report(_measured_input())
    replay = build_measured_capacity_report(
        _measured_input(execution_mode="REPLAY", wall_time_ns=_HOUR_NS // 4)
    )
    scaling = build_measured_capacity_report(
        _measured_input(recording_worker_count=2, wall_time_ns=_HOUR_NS // 4)
    )

    fresh_replay = compare_measured_capacity_reports(baseline, replay)
    workers = compare_measured_capacity_reports(baseline, scaling)

    assert fresh_replay.comparable is True
    assert fresh_replay.comparison_kind is MeasuredCapacityComparisonKind.FRESH_VS_REPLAY
    assert fresh_replay.recording_hours_per_wall_hour_ratio == pytest.approx(2.0)
    assert fresh_replay.call_splits_per_wall_hour_ratio == pytest.approx(2.0)
    assert fresh_replay.retries_per_wall_hour_ratio == pytest.approx(2.0)
    assert fresh_replay.dense_provider_images_per_wall_hour_ratio == pytest.approx(2.0)
    assert workers.comparable is True
    assert workers.comparison_kind is MeasuredCapacityComparisonKind.RECORDING_WORKER_SCALING
    assert workers.camera_hours_per_wall_hour_ratio == pytest.approx(2.0)


def test_fresh_replay_comparison_allows_expected_provider_mode_change() -> None:
    fresh = build_measured_capacity_report(_measured_input())
    replay = build_measured_capacity_report(
        _measured_input(
            execution_mode="REPLAY",
            provider_mode=ProviderMode.NO_PROVIDER_CALLS,
            windows=0,
            unique_images=0,
            coarse_unique_images=0,
            dense_unique_images=0,
            provider_images=0,
            logical_calls=0,
            call_parts=0,
            call_splits=0,
            http_requests=0,
            retries=0,
            batches=0,
            batch_requests=0,
            input_tokens=0,
            output_tokens=0,
            output_token_responses=0,
            dense_logical_calls=0,
            dense_provider_images=0,
        )
    )

    comparison = compare_measured_capacity_reports(fresh, replay)

    assert comparison.comparable is True
    assert comparison.comparison_kind is MeasuredCapacityComparisonKind.FRESH_VS_REPLAY
    assert comparison.baseline_provider_mode is ProviderMode.LOCAL_OFFLINE_FIXTURE
    assert comparison.candidate_provider_mode is ProviderMode.NO_PROVIDER_CALLS
    assert comparison.provider_images_per_wall_hour_ratio == pytest.approx(0.0)


def test_measured_capacity_comparison_rejects_changed_provider_or_workload() -> None:
    baseline = build_measured_capacity_report(_measured_input())
    candidate = build_measured_capacity_report(
        _measured_input(
            workload_fingerprint="b" * 64,
            provider_mode=ProviderMode.NETWORK_PROVIDER,
        )
    )

    comparison = compare_measured_capacity_reports(baseline, candidate)

    assert comparison.comparable is False
    assert comparison.non_comparable_reasons == (
        "PROVIDER_MODE_CHANGED",
        "WORKLOAD_FINGERPRINT_CHANGED",
    )
    assert comparison.recording_hours_per_wall_hour_ratio is None


def test_worker_scaling_reports_speedup_backlog_drain_and_local_projection() -> None:
    report = run_worker_scaling(
        _profile(
            unit_count=8,
            arrival_interval_ms=0,
            arrival_batch_size=8,
            service_time_pattern_ms=(100,),
            failed_ordinals=(),
            skipped_ordinals=(),
        ),
        worker_counts=(1, 2, 4, 8),
        target_recording_rtf=25.0,
        queue_capacity=8,
    )

    assert isinstance(report, WorkerScalingReport)
    assert report.worker_counts == (1, 2, 4, 8)
    assert report.four_worker_speedup == pytest.approx(4.0)
    assert report.four_worker_meets_2_5x is True
    assert report.backlog_drains_after_burst is True
    assert report.queues_bounded is True
    assert report.capacity_projection is not None
    assert report.capacity_projection.required_cpu_worker_count == 1
    assert report.capacity_projection.required_nvme_worker_count == 1
    assert report.evidence_class == "SYNTHETIC_LOCAL"
    assert report.measurement_status == "NOT_MEASURED"
    assert report.production_eligible is False
    assert report.as_dict()["worker_counts"] == [1, 2, 4, 8]


def test_worker_scaling_rejects_unordered_matrix_and_reports_unbounded_queue() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        run_worker_scaling(_profile(), worker_counts=(1, 4, 2))

    report = build_worker_scaling_report(_profile(unit_count=10), queue_capacity=1)
    assert report.queues_bounded is False
    assert all(point.queue_bounded is False for point in report.points)


def test_required_worker_count_for_rtf_uses_ceiling_and_rejects_zero_rate() -> None:
    assert required_worker_count_for_rtf(10.0, target_recording_rtf=25.0) == 3
    assert required_worker_count_for_rtf(25.0, target_recording_rtf=25.0) == 1
    with pytest.raises(ValueError, match="finite and positive"):
        required_worker_count_for_rtf(0.0)


def _backpressure_scenario(
    kind: BackpressureScenarioKind,
    *,
    adaptive: bool,
) -> BackpressureScenarioReport:
    peak = 5 if adaptive else 8
    end = 0 if kind is BackpressureScenarioKind.DRAIN else (1 if adaptive else 2)
    arrival_rate = 10.0 if kind is BackpressureScenarioKind.OVERLOAD else 6.0
    service_rate = 8.0 if kind is BackpressureScenarioKind.OVERLOAD else 6.0
    quota = 0 if kind is BackpressureScenarioKind.PROVIDER_QUOTA else 8

    def recording_backlogs(depth: int) -> tuple[BackpressureRecordingBacklog, ...]:
        if kind is not BackpressureScenarioKind.FAIRNESS:
            return ()
        first_depth = max(0, depth - 2)
        return (
            BackpressureRecordingBacklog(
                recording_key="recording-a",
                backlog_depth=first_depth,
            ),
            BackpressureRecordingBacklog(
                recording_key="recording-b",
                backlog_depth=depth - first_depth,
            ),
        )

    observations = [
        BackpressureObservation(
            observed_at_ms=0,
            backlog_depth=peak,
            controller_limit=8,
            arrival_rate_per_second=arrival_rate,
            service_rate_per_second=service_rate,
            backlog_slope_per_second=float(end - peak),
            oldest_backlog_age_ms=50,
            provider_quota=quota,
            worker_utilization=0.9,
            recording_backlogs=recording_backlogs(peak),
        ),
        BackpressureObservation(
            observed_at_ms=1_000,
            backlog_depth=end,
            controller_limit=6 if adaptive else 8,
            arrival_rate_per_second=arrival_rate,
            service_rate_per_second=service_rate,
            backlog_slope_per_second=float(end - peak),
            oldest_backlog_age_ms=0 if end == 0 else 100,
            provider_quota=4 if kind is BackpressureScenarioKind.PROVIDER_QUOTA else quota,
            worker_utilization=0.8,
            recording_backlogs=recording_backlogs(end),
        ),
    ]
    if kind is BackpressureScenarioKind.OSCILLATION and adaptive:
        observations.append(
            BackpressureObservation(
                observed_at_ms=2_000,
                backlog_depth=end,
                controller_limit=7,
                arrival_rate_per_second=arrival_rate,
                service_rate_per_second=service_rate,
                backlog_slope_per_second=0.0,
                oldest_backlog_age_ms=100,
                provider_quota=quota,
                worker_utilization=0.8,
            )
        )
    return BackpressureScenarioReport(kind, tuple(observations))


def _backpressure_report(
    controller_kind: BackpressureControllerKind,
    *,
    workload_fingerprint: str = "local-backpressure-workload-v1",
) -> BackpressureStabilityReport:
    scenarios = tuple(
        _backpressure_scenario(kind, adaptive=controller_kind is BackpressureControllerKind.AIMD)
        for kind in reversed(tuple(BackpressureScenarioKind))
    )
    return build_backpressure_stability_report(
        version="local-backpressure-stability-v1",
        workload_fingerprint=workload_fingerprint,
        controller_kind=controller_kind,
        controller_version=f"{controller_kind.value.lower()}-controller-v1",
        controller_policy_digest=f"{controller_kind.value.lower()}-policy-digest",
        scenarios=scenarios,
    )


def test_backpressure_stability_keeps_unknown_signals_and_signed_drain_explicit() -> None:
    drain = BackpressureScenarioReport(
        BackpressureScenarioKind.DRAIN,
        (
            BackpressureObservation(
                observed_at_ms=0,
                backlog_depth=4,
                controller_limit=8,
                backlog_slope_per_second=-4.0,
            ),
            BackpressureObservation(
                observed_at_ms=1_000,
                backlog_depth=0,
                controller_limit=8,
                backlog_slope_per_second=-4.0,
            ),
        ),
    )
    report = build_backpressure_stability_report(
        version="local-backpressure-stability-v1",
        workload_fingerprint="local-backpressure-workload-v1",
        controller_kind=BackpressureControllerKind.FIXED,
        controller_version="fixed-controller-v1",
        controller_policy_digest="fixed-policy-digest",
        scenarios=(drain,),
    )

    assert drain.mean_arrival_rate_per_second is None
    assert drain.mean_service_rate_per_second is None
    assert drain.minimum_provider_quota is None
    assert drain.mean_worker_utilization is None
    assert drain.net_backlog_slope_per_second == pytest.approx(-4.0)
    assert drain.drain_completed is True
    assert drain.missing_signal_kinds == (
        "ARRIVAL_RATE",
        "SERVICE_RATE",
        "PROVIDER_QUOTA",
        "WORKER_UTILIZATION",
    )
    assert report.complete_scenario_matrix is False
    assert report.measurement_status == "NOT_MEASURED"
    assert report.qualification_status == "NOT_PRODUCTION_QUALIFIED"
    assert report.production_eligible is False

    with pytest.raises(ValueError, match="reconcile"):
        BackpressureObservation(
            observed_at_ms=0,
            backlog_depth=2,
            controller_limit=4,
            recording_backlogs=(
                BackpressureRecordingBacklog("recording-a", 1),
                BackpressureRecordingBacklog("recording-b", 2),
            ),
        )


def test_backpressure_stability_compares_complete_fixed_and_adaptive_matrix() -> None:
    fixed = _backpressure_report(BackpressureControllerKind.FIXED)
    adaptive = _backpressure_report(BackpressureControllerKind.AIMD)

    comparison = compare_fixed_and_adaptive_backpressure(fixed, adaptive)

    assert fixed.complete_scenario_matrix is True
    assert adaptive.complete_scenario_matrix is True
    assert comparison.comparable is True
    assert comparison.non_comparable_reasons == ()
    assert comparison.adaptive_matches_or_beats_fixed is True
    assert compare_backpressure_stability_reports(fixed, adaptive) == comparison
    assert (
        fixed.report_digest == _backpressure_report(BackpressureControllerKind.FIXED).report_digest
    )
    assert fixed.scenario(BackpressureScenarioKind.OVERLOAD).overload_observed is True
    assert fixed.scenario(BackpressureScenarioKind.PROVIDER_QUOTA).minimum_provider_quota == 0
    assert fixed.scenario(BackpressureScenarioKind.DRAIN).drain_completed is True
    assert (
        adaptive.scenario(
            BackpressureScenarioKind.OSCILLATION
        ).controller_limit_direction_reversal_count
        == 1
    )
    assert fixed.scenario(BackpressureScenarioKind.FAIRNESS).recording_backlog_spread_peak == 4
    assert adaptive.scenario(BackpressureScenarioKind.FAIRNESS).recording_backlog_spread_peak == 1
    assert [item.scenario for item in comparison.scenario_comparisons] == sorted(
        BackpressureScenarioKind,
        key=lambda item: item.value,
    )
    assert comparison.measurement_status == "NOT_MEASURED"
    assert comparison.qualification_status == "NOT_PRODUCTION_QUALIFIED"
    assert comparison.production_eligible is False


def test_backpressure_comparison_refuses_incomplete_or_changed_workload() -> None:
    fixed = build_backpressure_stability_report(
        version="local-backpressure-stability-v1",
        workload_fingerprint="fixed-workload",
        controller_kind=BackpressureControllerKind.FIXED,
        controller_version="fixed-controller-v1",
        controller_policy_digest="fixed-policy-digest",
        scenarios=(_backpressure_scenario(BackpressureScenarioKind.STEADY, adaptive=False),),
    )
    adaptive = build_backpressure_stability_report(
        version="local-backpressure-stability-v1",
        workload_fingerprint="adaptive-workload",
        controller_kind=BackpressureControllerKind.AIMD,
        controller_version="aimd-controller-v1",
        controller_policy_digest="aimd-policy-digest",
        scenarios=(_backpressure_scenario(BackpressureScenarioKind.STEADY, adaptive=True),),
    )

    comparison = compare_fixed_and_adaptive_backpressure(fixed, adaptive)

    assert comparison.comparable is False
    assert comparison.scenario_comparisons == ()
    assert comparison.adaptive_matches_or_beats_fixed is None
    assert comparison.non_comparable_reasons == (
        "BASELINE_SCENARIO_MATRIX_INCOMPLETE",
        "CANDIDATE_SCENARIO_MATRIX_INCOMPLETE",
        "WORKLOAD_FINGERPRINT_CHANGED",
    )
