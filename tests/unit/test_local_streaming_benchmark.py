from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from robata.runtime.benchmark import (
    RecordingWorkerBatchFacts,
    RecordingWorkerQueueObservation,
    build_measured_recording_worker_scaling_report,
    run_measured_recording_worker_matrix,
)
from robata.runtime.local_streaming_benchmark import (
    WP6_MINIMUM_SMOKE_DURATION_MS,
    BenchmarkCacheState,
    BenchmarkProtocolPins,
    BenchmarkRunMode,
    BenchmarkRunStatePins,
    BurstShapePins,
    CandidateSourcePins,
    HostRuntimePins,
    LocalGateStatus,
    LocalStreamingBenchmarkManifest,
    LongStreamKind,
    MockFailureDistribution,
    MockLatencyDistribution,
    MockLatencyPoint,
    MockProviderPins,
    MockRetryPolicyPins,
    OfferedLoadScenario,
    OfferedLoadUnit,
    StreamingPolicyPins,
    StreamingViolation,
    StructuralBenchmarkObservation,
    StructuralViolation,
    evaluate_local_streaming_virtual_estimate,
)

SOURCE_DURATION_NS = 40_890_455_000
SOURCE_BYTES = 130_303_923


def _sha(character: str) -> str:
    return character * 64


def _run_states() -> tuple[BenchmarkRunStatePins, ...]:
    return (
        BenchmarkRunStatePins(mode=BenchmarkRunMode.COLD, cache_state=BenchmarkCacheState.EMPTY),
        BenchmarkRunStatePins(
            mode=BenchmarkRunMode.FRESH,
            cache_state=BenchmarkCacheState.DISABLED,
        ),
        BenchmarkRunStatePins(
            mode=BenchmarkRunMode.REPLAY,
            cache_state=BenchmarkCacheState.RESTORED,
        ),
    )


def _policies() -> StreamingPolicyPins:
    return StreamingPolicyPins(
        chunk_duration_ms=1_000,
        window_duration_ms=2_000,
        window_hop_ms=1_000,
        allowed_lateness_ms=300,
        ring_capacity_ms=5_000,
        chunk_policy_sha256=_sha("1"),
        window_policy_sha256=_sha("2"),
        lateness_policy_sha256=_sha("3"),
        ring_policy_sha256=_sha("4"),
        sampling_policy_sha256=_sha("5"),
        trigger_policy_sha256=_sha("6"),
        candidate_policy_sha256=_sha("7"),
        boundary_policy_sha256=_sha("8"),
        fan_out_policy_sha256=_sha("9"),
    )


def _provider(
    *,
    seed: int = 29,
    latency_ms: int = 100,
    failure_probability_ppm: int = 0,
    maximum_attempts: int = 2,
    timeout_ms: int = 500,
    batch_size: int = 2,
    concurrency: int = 2,
) -> MockProviderPins:
    return MockProviderPins(
        latency=MockLatencyDistribution(
            points=(
                MockLatencyPoint(latency_ms=latency_ms, weight=3),
                MockLatencyPoint(latency_ms=latency_ms * 2, weight=1),
            )
        ),
        failure=MockFailureDistribution(failure_probability_ppm=failure_probability_ppm),
        retry=MockRetryPolicyPins(
            maximum_attempts=maximum_attempts,
            backoff_ms=tuple(10 for _ in range(maximum_attempts - 1)),
        ),
        seed=seed,
        request_timeout_ms=timeout_ms,
        request_limit_per_second=40,
        max_batch_size=batch_size,
        max_concurrency_per_group=concurrency,
    )


def _protocol(*, latency_target_ms: int = 500) -> BenchmarkProtocolPins:
    return BenchmarkProtocolPins(
        warmup_count=1,
        repetition_count=2,
        burst_shape=BurstShapePins(
            bucket_duration_ms=1_000,
            relative_load_pattern=(4, 0, 0, 0),
        ),
        observation_cutoff_ms=20_000,
        smoke_duration_ms=WP6_MINIMUM_SMOKE_DURATION_MS,
        short_source_first=True,
        long_stream_kind=LongStreamKind.REPEATED_SOURCE,
        incremental_latency_target_ms=latency_target_ms,
    )


def _offered_loads(
    *,
    recording_groups: int = 2,
    camera_video_groups: int = 1,
) -> tuple[OfferedLoadScenario, ...]:
    return (
        OfferedLoadScenario(
            unit=OfferedLoadUnit.RECORDING_HOURS_PER_DAY,
            offered_hours_per_day=500,
            provisioned_six_camera_groups=recording_groups,
        ),
        OfferedLoadScenario(
            unit=OfferedLoadUnit.AGGREGATE_CAMERA_VIDEO_HOURS_PER_DAY,
            offered_hours_per_day=500,
            provisioned_six_camera_groups=camera_video_groups,
        ),
    )


def _manifest(
    *,
    run_states: tuple[BenchmarkRunStatePins, ...] | None = None,
    provider: MockProviderPins | None = None,
    protocol: BenchmarkProtocolPins | None = None,
    offered_loads: tuple[OfferedLoadScenario, ...] | None = None,
) -> LocalStreamingBenchmarkManifest:
    return LocalStreamingBenchmarkManifest.create(
        candidate=CandidateSourcePins(
            candidate_commit="a" * 40,
            source_sha256=_sha("b"),
            source_byte_count=SOURCE_BYTES,
            source_duration_ns=SOURCE_DURATION_NS,
            lockfile_sha256=_sha("c"),
        ),
        host=HostRuntimePins(
            cpu_model="Test CPU",
            logical_cpu_count=16,
            gpu_model="NONE",
            memory_bytes=32 * 1024**3,
            driver_version="NONE",
            operating_system="Test OS 1",
            power_mode="PERFORMANCE",
            runtime="CPython 3.12.10",
        ),
        run_states=run_states if run_states is not None else _run_states(),
        artifact_retention_profile_sha256=_sha("d"),
        policies=_policies(),
        mock_provider=provider if provider is not None else _provider(),
        protocol=protocol if protocol is not None else _protocol(),
        offered_loads=offered_loads if offered_loads is not None else _offered_loads(),
    )


def _passing_structural_observation() -> StructuralBenchmarkObservation:
    return StructuralBenchmarkObservation(
        fresh_wall_time_ns=100_000_000_000,
        required_state_bytes=200_000_000,
        duplicate_decode_export_wall_time_ns=30_000_000_000,
        per_row_audit_wall_time_ns=20_000_000_000,
    )


def test_manifest_content_addresses_complete_local_protocol() -> None:
    first = _manifest()
    second = _manifest()

    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.candidate.source_duration_ns == SOURCE_DURATION_NS
    assert first.host.power_mode == "PERFORMANCE"
    assert tuple(state.mode for state in first.run_states) == tuple(BenchmarkRunMode)
    assert first.policies.allowed_lateness_ms == 300
    assert first.mock_provider.seed == 29
    assert first.protocol.short_source_first is True
    assert first.protocol.smoke_duration_ms == 30 * 60 * 1_000
    assert tuple(item.unit for item in first.offered_loads) == tuple(OfferedLoadUnit)
    assert first.evidence_class == "LOCAL_CONFORMANCE"
    assert first.measurement_status == "NOT_MEASURED"
    assert first.production_eligible is False
    assert first.qualification_status == "NOT_PRODUCTION_QUALIFIED"
    assert LocalStreamingBenchmarkManifest.model_validate_json(first.model_dump_json()) == first

    different_seed = _manifest(provider=_provider(seed=30))
    assert different_seed.manifest_sha256 != first.manifest_sha256


def test_manifest_rejects_tampering_and_incomplete_required_scenarios() -> None:
    manifest = _manifest()
    payload = manifest.model_dump(mode="json")
    payload["manifest_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="manifest_sha256"):
        LocalStreamingBenchmarkManifest.model_validate_json(json.dumps(payload))

    with pytest.raises(ValidationError, match="COLD, FRESH, and REPLAY"):
        _manifest(run_states=_run_states()[:-1])
    with pytest.raises(ValidationError, match="recording-hour and aggregate"):
        _manifest(offered_loads=_offered_loads()[:1])
    with pytest.raises(ValidationError):
        BenchmarkProtocolPins(
            **{
                **_protocol().model_dump(mode="python"),
                "smoke_duration_ms": WP6_MINIMUM_SMOKE_DURATION_MS - 1,
            }
        )


def test_virtual_estimate_is_deterministic_and_non_authoritative() -> None:
    manifest = _manifest()
    observation = _passing_structural_observation()

    first = evaluate_local_streaming_virtual_estimate(manifest, observation)
    second = evaluate_local_streaming_virtual_estimate(manifest, observation)

    assert first == second
    assert first.report_sha256 == second.report_sha256
    assert first.manifest_sha256 == manifest.manifest_sha256
    assert first.simulated_gate_status is LocalGateStatus.PASS
    assert first.structural_gate.status is LocalGateStatus.PASS
    assert first.streaming_gate.status is LocalGateStatus.PASS
    assert first.streaming_gate.sustained_service_capacity_recording_seconds_per_wall_second >= 1.86
    assert first.streaming_gate.incremental_latency.p95_ms <= 500
    assert all(
        result.status is LocalGateStatus.PASS and result.required_backlog_growing is False
        for result in first.streaming_gate.offered_loads
    )
    assert first.provider_traffic == "MOCKED"
    assert first.execution_mode == "VIRTUAL_SIMULATION_ONLY"
    assert first.authority_status == "NON_AUTHORITATIVE"
    assert first.wp6_smoke_gate_eligible is False
    assert first.authoritative_report_model == "local-streaming-smoke-report-v1"
    assert first.evidence_class == "VIRTUAL_MODEL_DIAGNOSTIC"
    assert first.measurement_status == "NOT_MEASURED"
    assert first.production_eligible is False
    assert first.qualification_status == "NOT_PRODUCTION_QUALIFIED"


def test_structural_and_streaming_gates_are_evaluated_separately() -> None:
    structurally_bad = StructuralBenchmarkObservation(
        fresh_wall_time_ns=SOURCE_DURATION_NS * 6,
        required_state_bytes=SOURCE_BYTES * 3,
        duplicate_decode_export_wall_time_ns=SOURCE_DURATION_NS * 4,
        per_row_audit_wall_time_ns=SOURCE_DURATION_NS * 3,
    )

    report = evaluate_local_streaming_virtual_estimate(_manifest(), structurally_bad)

    assert report.simulated_gate_status is LocalGateStatus.FAIL
    assert report.structural_gate.status is LocalGateStatus.FAIL
    assert report.structural_gate.violations == (
        StructuralViolation.FRESH_SOURCE_TIME,
        StructuralViolation.REQUIRED_STATE_BYTES,
        StructuralViolation.DUPLICATE_DECODE_EXPORT_DOMINANT,
        StructuralViolation.PER_ROW_AUDIT_DOMINANT,
    )
    assert report.streaming_gate.status is LocalGateStatus.PASS
    assert report.streaming_gate.violations == ()


def test_retry_exhaustion_fails_streaming_capacity_and_virtual_smoke() -> None:
    provider = _provider(
        latency_ms=100,
        failure_probability_ppm=1_000_000,
        maximum_attempts=3,
        batch_size=1,
        concurrency=1,
    )
    manifest = _manifest(
        provider=provider,
        protocol=_protocol(latency_target_ms=250),
        offered_loads=_offered_loads(recording_groups=100, camera_video_groups=100),
    )

    report = evaluate_local_streaming_virtual_estimate(
        manifest,
        _passing_structural_observation(),
    )

    assert report.structural_gate.status is LocalGateStatus.PASS
    assert report.streaming_gate.status is LocalGateStatus.FAIL
    assert report.streaming_gate.violations == (
        StreamingViolation.SERVICE_CAPACITY,
        StreamingViolation.INCREMENTAL_LATENCY,
        StreamingViolation.RECORDING_HOURS_BACKLOG_GROWTH,
        StreamingViolation.CAMERA_VIDEO_HOURS_BACKLOG_GROWTH,
    )
    assert report.streaming_gate.terminal_failure_rate == 1.0
    assert report.streaming_gate.sustained_service_capacity_recording_seconds_per_wall_second == 0
    for repetition in report.streaming_gate.capacity_repetitions:
        assert repetition.failed_request_count == repetition.probe_request_count
        assert repetition.provider_attempt_count == repetition.probe_request_count * 3
    assert all(
        result.required_six_camera_groups is None
        and result.virtual_end_required_backlog_recording_seconds > 0
        for result in report.streaming_gate.offered_loads
    )


def test_mock_latency_distribution_requires_canonical_order() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        MockLatencyDistribution(
            points=(
                MockLatencyPoint(latency_ms=200, weight=1),
                MockLatencyPoint(latency_ms=100, weight=1),
            )
        )
    with pytest.raises(ValidationError, match="unique"):
        MockLatencyDistribution(
            points=(
                MockLatencyPoint(latency_ms=100, weight=1),
                MockLatencyPoint(latency_ms=100, weight=2),
            )
        )



def _worker_facts(
    *,
    recording_count: int = 4,
    named_shared_resource_limit: str | None = None,
    cancelled_recording_count: int = 0,
) -> RecordingWorkerBatchFacts:
    return RecordingWorkerBatchFacts(
        successful_recording_count=recording_count - cancelled_recording_count,
        failed_recording_count=0,
        cancelled_recording_count=cancelled_recording_count,
        replay_verified_recording_count=cancelled_recording_count,
        distinct_state_root_count=recording_count,
        state_affinity_violation_count=0,
        queues=(
            RecordingWorkerQueueObservation(
                name="ingress",
                capacity=4,
                high_watermark=4,
                end_depth=0,
                backpressure_event_count=1,
            ),
            RecordingWorkerQueueObservation(
                name="provider",
                capacity=8,
                high_watermark=6,
                end_depth=0,
            ),
            RecordingWorkerQueueObservation(
                name="publish",
                capacity=4,
                high_watermark=3,
                end_depth=0,
            ),
        ),
        named_shared_resource_limit=named_shared_resource_limit,
    )


def test_measured_recording_worker_matrix_reports_queue_drain_and_local_sizing() -> None:
    hour_ns = 3_600_000_000_000
    ticks = iter((0, 4 * hour_ns, 5 * hour_ns, 7 * hour_ns, 8 * hour_ns, 9 * hour_ns))

    report = run_measured_recording_worker_matrix(
        lambda _worker_count: lambda: _worker_facts(),
        workload_id="f" * 64,
        recording_count=4,
        recording_duration_ns=hour_ns,
        worker_counts=(1, 2, 4),
        target_recording_rtf=25.0,
        clock_ns=lambda: next(ticks),
    )

    assert report.worker_counts == (1, 2, 4)
    assert report.four_worker_speedup == pytest.approx(4.0)
    assert report.four_worker_meets_2_5x is True
    assert report.four_worker_outcome_explained is True
    assert report.queues_bounded is True
    assert report.backlog_drains_after_burst is True
    assert report.cancellation_restart_replayable is True
    assert report.capacity_projection is not None
    assert report.capacity_projection.per_worker_recording_rtf == pytest.approx(1.0)
    assert report.capacity_projection.required_cpu_worker_count == 25
    assert report.capacity_projection.required_nvme_worker_count == 25
    assert report.evidence_class == "LOCAL_CONFORMANCE"
    assert report.measurement_status == "MEASURED"
    assert report.production_eligible is False
    assert report.as_dict()["runs"][2]["recording_rtf"] == pytest.approx(4.0)


def test_measured_worker_report_requires_named_limit_when_four_workers_do_not_scale() -> None:
    hour_ns = 3_600_000_000_000
    ticks = iter((0, 4 * hour_ns, 5 * hour_ns, 7 * hour_ns, 8 * hour_ns, 10 * hour_ns))

    measured = run_measured_recording_worker_matrix(
        lambda worker_count: lambda: _worker_facts(
            named_shared_resource_limit=(
                None if worker_count < 4 else "offline fixture provider concurrency=2"
            ),
        ),
        workload_id="e" * 64,
        recording_count=4,
        recording_duration_ns=hour_ns,
        worker_counts=(1, 2, 4),
        clock_ns=lambda: next(ticks),
    )
    report = build_measured_recording_worker_scaling_report(measured.runs)

    assert report.four_worker_speedup == pytest.approx(2.0)
    assert report.four_worker_meets_2_5x is False
    assert (
        report.four_worker_named_shared_resource_limit
        == "offline fixture provider concurrency=2"
    )
    assert report.four_worker_outcome_explained is True



def test_verified_cancellation_replay_remains_state_affine_and_sustainable() -> None:
    hour_ns = 3_600_000_000_000
    ticks = iter((0, 4 * hour_ns, 5 * hour_ns, 7 * hour_ns, 8 * hour_ns, 9 * hour_ns))

    report = run_measured_recording_worker_matrix(
        lambda _worker_count: lambda: _worker_facts(cancelled_recording_count=1),
        workload_id="c" * 64,
        recording_count=4,
        recording_duration_ns=hour_ns,
        worker_counts=(1, 2, 4),
        clock_ns=lambda: next(ticks),
    )

    first_run = report.runs[0]
    assert first_run.terminal_or_replay_completed_recording_count == 4
    assert first_run.complete_without_state_leakage is True
    assert first_run.sustainable is True
    assert report.cancellation_restart_replayable is True
    assert report.capacity_projection is not None
    assert report.as_dict()["runs"][0]["replay_verified_cancelled_recording_count"] == 1


def test_queue_burst_observation_reports_overflow_and_missing_drain() -> None:
    facts = RecordingWorkerBatchFacts(
        successful_recording_count=4,
        failed_recording_count=0,
        cancelled_recording_count=0,
        replay_verified_recording_count=0,
        distinct_state_root_count=4,
        state_affinity_violation_count=0,
        queues=(
            RecordingWorkerQueueObservation(
                name="ingress",
                capacity=2,
                high_watermark=3,
                end_depth=1,
                backpressure_event_count=1,
            ),
            RecordingWorkerQueueObservation(
                name="provider",
                capacity=2,
                high_watermark=1,
                end_depth=0,
            ),
            RecordingWorkerQueueObservation(
                name="publish",
                capacity=2,
                high_watermark=1,
                end_depth=0,
            ),
        ),
    )

    assert facts.queues_bounded is False
    assert facts.queue_burst_observed is True
    assert facts.ingress_backpressure_observed is True
    assert facts.burst_backpressure_drained is False
    assert facts.queues[0].as_dict()["bounded"] is False


def test_n_worker_saturation_reports_named_resource_limit() -> None:
    hour_ns = 3_600_000_000_000
    ticks = iter(
        (
            0,
            4 * hour_ns,
            5 * hour_ns,
            7 * hour_ns,
            8 * hour_ns,
            9 * hour_ns,
            10 * hour_ns,
            11 * hour_ns,
        )
    )
    resource_limit = "offline fixture provider concurrency=4"

    report = run_measured_recording_worker_matrix(
        lambda worker_count: lambda: _worker_facts(
            named_shared_resource_limit=(
                resource_limit if worker_count == 8 else None
            ),
        ),
        workload_id="d" * 64,
        recording_count=4,
        recording_duration_ns=hour_ns,
        worker_counts=(1, 2, 4, 8),
        clock_ns=lambda: next(ticks),
    )

    assert report.scale_out_worker_counts == (8,)
    assert report.saturation_worker_count == 8
    assert report.saturation_named_shared_resource_limit == resource_limit
    assert report.named_shared_resource_limits == ((8, resource_limit),)
    assert report.saturation_outcome_explained is True
    assert report.unexplained_non_scaling_worker_counts == ()
    assert report.scale_out_outcome_explained is True
    assert report.n_worker_outcome_explained is True
    assert report.as_dict()["saturation_worker_count"] == 8
