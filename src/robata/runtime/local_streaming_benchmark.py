"""Content-addressed inputs for a deterministic local streaming estimate.

The evaluator in this module advances no wall clock and makes no provider call.  It models the
pinned mock latency, failure, retry, concurrency, request-rate, and batching inputs so capacity
arithmetic can be exercised repeatably.  Its report is a non-authoritative virtual estimate; the
actual WP6 smoke authority is local-streaming-smoke-report-v1.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256

LOCAL_STREAMING_BENCHMARK_MANIFEST_VERSION: Final = "local-streaming-benchmark-manifest-v1"
LOCAL_STREAMING_VIRTUAL_ESTIMATE_REPORT_VERSION: Final = (
    "local-streaming-virtual-estimate-report-v1"
)
WP6_MINIMUM_SMOKE_DURATION_MS: Final = 30 * 60 * 1_000
WP6_MAXIMUM_FRESH_SOURCE_TIME_RATIO: Final = 5.0
WP6_MAXIMUM_REQUIRED_STATE_BYTES_RATIO: Final = 2.0
WP6_DOMINANT_WALL_TIME_SHARE: Final = 0.5
WP6_MINIMUM_SERVICE_CAPACITY: Final = 1.86
WP6_PLANNING_HEADROOM: Final = 1.3
WP6_PLANNING_UTILIZATION: Final = 0.70

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
GitCommit = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Seed = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
PartsPerMillion = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0.0, allow_inf_nan=False)]


class BenchmarkRunMode(StrEnum):
    """Required local run modes whose cache state is fixed before execution."""

    COLD = "COLD"
    FRESH = "FRESH"
    REPLAY = "REPLAY"


class BenchmarkCacheState(StrEnum):
    """Explicit cache state for one benchmark run mode."""

    DISABLED = "DISABLED"
    EMPTY = "EMPTY"
    WARM = "WARM"
    RESTORED = "RESTORED"


class LongStreamKind(StrEnum):
    """How the at-least-30-minute local stream is supplied."""

    REPEATED_SOURCE = "REPEATED_SOURCE"
    GENERATED = "GENERATED"


class OfferedLoadUnit(StrEnum):
    """The two intentionally distinct 500-hour planning interpretations."""

    RECORDING_HOURS_PER_DAY = "RECORDING_HOURS_PER_DAY"
    AGGREGATE_CAMERA_VIDEO_HOURS_PER_DAY = "AGGREGATE_CAMERA_VIDEO_HOURS_PER_DAY"


class LocalGateStatus(StrEnum):
    """An unvalidated local engineering gate result."""

    PASS = "PASS"
    FAIL = "FAIL"


class StructuralViolation(StrEnum):
    FRESH_SOURCE_TIME = "FRESH_SOURCE_TIME"
    REQUIRED_STATE_BYTES = "REQUIRED_STATE_BYTES"
    DUPLICATE_DECODE_EXPORT_DOMINANT = "DUPLICATE_DECODE_EXPORT_DOMINANT"
    PER_ROW_AUDIT_DOMINANT = "PER_ROW_AUDIT_DOMINANT"


class StreamingViolation(StrEnum):
    SERVICE_CAPACITY = "SERVICE_CAPACITY"
    INCREMENTAL_LATENCY = "INCREMENTAL_LATENCY"
    RECORDING_HOURS_BACKLOG_GROWTH = "RECORDING_HOURS_BACKLOG_GROWTH"
    CAMERA_VIDEO_HOURS_BACKLOG_GROWTH = "CAMERA_VIDEO_HOURS_BACKLOG_GROWTH"


class CandidateSourcePins(StrictModel):
    """Exact candidate, source, and dependency-lock identities."""

    candidate_commit: GitCommit
    source_sha256: Sha256Digest
    source_byte_count: PositiveInt
    source_duration_ns: Nanoseconds
    lockfile_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_source_duration(self) -> Self:
        if self.source_duration_ns <= 0:
            raise ValueError("source_duration_ns must be positive")
        return self


class HostRuntimePins(StrictModel):
    """Portable host facts, including explicit NONE values for absent accelerators."""

    cpu_model: NonEmptyString
    logical_cpu_count: PositiveInt
    gpu_model: NonEmptyString
    memory_bytes: PositiveInt
    driver_version: NonEmptyString
    operating_system: NonEmptyString
    power_mode: NonEmptyString
    runtime: NonEmptyString


class BenchmarkRunStatePins(StrictModel):
    """One required cold, fresh, or replay phase and its cache state."""

    mode: BenchmarkRunMode
    cache_state: BenchmarkCacheState


class StreamingPolicyPins(StrictModel):
    """Exact streaming values and policy artifact digests used by the candidate."""

    chunk_duration_ms: PositiveInt
    window_duration_ms: PositiveInt
    window_hop_ms: PositiveInt
    allowed_lateness_ms: NonNegativeInt
    ring_capacity_ms: PositiveInt
    chunk_policy_sha256: Sha256Digest
    window_policy_sha256: Sha256Digest
    lateness_policy_sha256: Sha256Digest
    ring_policy_sha256: Sha256Digest
    sampling_policy_sha256: Sha256Digest
    trigger_policy_sha256: Sha256Digest
    candidate_policy_sha256: Sha256Digest
    boundary_policy_sha256: Sha256Digest
    fan_out_policy_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_stream_shape(self) -> Self:
        if self.window_hop_ms > self.window_duration_ms:
            raise ValueError("window_hop_ms cannot exceed window_duration_ms")
        if self.chunk_duration_ms > self.ring_capacity_ms:
            raise ValueError("chunk_duration_ms cannot exceed ring_capacity_ms")
        if self.ring_capacity_ms < self.window_duration_ms + self.allowed_lateness_ms:
            raise ValueError("ring_capacity_ms must retain one window plus allowed lateness")
        return self


class MockLatencyPoint(StrictModel):
    """One point in an integer-weighted discrete mock latency distribution."""

    latency_ms: PositiveInt
    weight: PositiveInt


class MockLatencyDistribution(StrictModel):
    """Canonical discrete latency distribution sampled with digest-derived draws."""

    points: tuple[MockLatencyPoint, ...]

    @model_validator(mode="after")
    def validate_points(self) -> Self:
        if not self.points:
            raise ValueError("latency distribution points must be nonempty")
        latencies = tuple(point.latency_ms for point in self.points)
        if latencies != tuple(sorted(latencies)):
            raise ValueError("latency distribution points must be ordered by latency_ms")
        if len(set(latencies)) != len(latencies):
            raise ValueError("latency distribution latency_ms values must be unique")
        return self


class MockFailureDistribution(StrictModel):
    """Independent per-attempt failure probability in integer parts per million."""

    failure_probability_ppm: PartsPerMillion


class MockRetryPolicyPins(StrictModel):
    """Pinned retry count and delay after each retryable mock failure."""

    maximum_attempts: PositiveInt
    backoff_ms: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def validate_backoffs(self) -> Self:
        expected = self.maximum_attempts - 1
        if len(self.backoff_ms) != expected:
            raise ValueError("backoff_ms must contain one delay per possible retry")
        return self


class MockProviderPins(StrictModel):
    """Complete deterministic provider mock and local resource limits."""

    latency: MockLatencyDistribution
    failure: MockFailureDistribution
    retry: MockRetryPolicyPins
    seed: Seed
    request_timeout_ms: PositiveInt
    request_limit_per_second: PositiveInt
    max_batch_size: PositiveInt
    max_concurrency_per_group: PositiveInt


class BurstShapePins(StrictModel):
    """Relative offered-load weights applied over equal-duration buckets."""

    bucket_duration_ms: PositiveInt
    relative_load_pattern: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def validate_pattern(self) -> Self:
        if not self.relative_load_pattern:
            raise ValueError("relative_load_pattern must be nonempty")
        if len(self.relative_load_pattern) > 1_024:
            raise ValueError("relative_load_pattern cannot exceed 1024 buckets")
        if not any(self.relative_load_pattern):
            raise ValueError("relative_load_pattern must contain offered load")
        return self


class BenchmarkProtocolPins(StrictModel):
    """Warm-up, repetitions, cutoff, burst, latency target, and smoke protocol."""

    warmup_count: NonNegativeInt
    repetition_count: PositiveInt
    burst_shape: BurstShapePins
    observation_cutoff_ms: PositiveInt
    smoke_duration_ms: Annotated[int, Field(strict=True, ge=WP6_MINIMUM_SMOKE_DURATION_MS)]
    short_source_first: Literal[True]
    long_stream_kind: LongStreamKind
    incremental_latency_target_ms: PositiveInt

    @model_validator(mode="after")
    def validate_cutoff(self) -> Self:
        if self.observation_cutoff_ms > self.smoke_duration_ms:
            raise ValueError("observation_cutoff_ms cannot exceed smoke_duration_ms")
        return self


class OfferedLoadScenario(StrictModel):
    """One 500-hour/day interpretation and its pinned local group allocation."""

    unit: OfferedLoadUnit
    offered_hours_per_day: Literal[500]
    provisioned_six_camera_groups: PositiveInt

    @property
    def recording_seconds_per_wall_second(self) -> float:
        if self.unit is OfferedLoadUnit.RECORDING_HOURS_PER_DAY:
            return self.offered_hours_per_day / 24
        return self.offered_hours_per_day / (6 * 24)

    @property
    def camera_video_seconds_per_wall_second(self) -> float:
        if self.unit is OfferedLoadUnit.AGGREGATE_CAMERA_VIDEO_HOURS_PER_DAY:
            return self.offered_hours_per_day / 24
        return self.offered_hours_per_day * 6 / 24


def local_streaming_benchmark_manifest_projection(
    manifest: LocalStreamingBenchmarkManifest,
) -> dict[str, object]:
    """Return the complete canonical preimage of a WP6 local manifest."""

    return manifest.model_dump(mode="json", exclude={"manifest_sha256"})


class LocalStreamingBenchmarkManifest(StrictModel):
    """Content-addressed pre-execution input closure for WP6 local qualification."""

    schema_version: Literal["1.0"]
    model_version: Literal["local-streaming-benchmark-manifest-v1"]
    manifest_sha256: Sha256Digest
    candidate: CandidateSourcePins
    host: HostRuntimePins
    run_states: tuple[BenchmarkRunStatePins, ...]
    artifact_retention_profile_sha256: Sha256Digest
    policies: StreamingPolicyPins
    mock_provider: MockProviderPins
    protocol: BenchmarkProtocolPins
    offered_loads: tuple[OfferedLoadScenario, ...]
    provider_traffic: Literal["MOCKED"]
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    measurement_status: Literal["NOT_MEASURED"]
    production_eligible: Literal[False]
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"]

    @classmethod
    def create(
        cls,
        *,
        candidate: CandidateSourcePins,
        host: HostRuntimePins,
        run_states: tuple[BenchmarkRunStatePins, ...],
        artifact_retention_profile_sha256: Sha256Digest,
        policies: StreamingPolicyPins,
        mock_provider: MockProviderPins,
        protocol: BenchmarkProtocolPins,
        offered_loads: tuple[OfferedLoadScenario, ...],
    ) -> Self:
        draft = cls.model_construct(
            schema_version="1.0",
            model_version=LOCAL_STREAMING_BENCHMARK_MANIFEST_VERSION,
            manifest_sha256="0" * 64,
            candidate=candidate,
            host=host,
            run_states=run_states,
            artifact_retention_profile_sha256=artifact_retention_profile_sha256,
            policies=policies,
            mock_provider=mock_provider,
            protocol=protocol,
            offered_loads=offered_loads,
            provider_traffic="MOCKED",
            evidence_class="LOCAL_CONFORMANCE",
            measurement_status="NOT_MEASURED",
            production_eligible=False,
            qualification_status="NOT_PRODUCTION_QUALIFIED",
        )
        digest = semantic_sha256(local_streaming_benchmark_manifest_projection(draft))
        return cls.model_validate(
            {**draft.model_dump(mode="python"), "manifest_sha256": digest},
            strict=True,
        )

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        run_modes = tuple(state.mode for state in self.run_states)
        if run_modes != tuple(BenchmarkRunMode):
            raise ValueError(
                "run_states must contain COLD, FRESH, and REPLAY exactly once in order"
            )
        load_units = tuple(scenario.unit for scenario in self.offered_loads)
        if load_units != tuple(OfferedLoadUnit):
            raise ValueError(
                "offered_loads must contain the recording-hour and aggregate camera-video-hour "
                "scenarios exactly once in order"
            )
        expected = semantic_sha256(local_streaming_benchmark_manifest_projection(self))
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 does not match canonical manifest bytes")
        return self


class StructuralBenchmarkObservation(StrictModel):
    """Measured local structural values evaluated separately from mock streaming behavior."""

    fresh_wall_time_ns: Nanoseconds
    required_state_bytes: NonNegativeInt
    duplicate_decode_export_wall_time_ns: Nanoseconds
    per_row_audit_wall_time_ns: Nanoseconds

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.fresh_wall_time_ns <= 0:
            raise ValueError("fresh_wall_time_ns must be positive")
        for name, value in (
            ("duplicate_decode_export_wall_time_ns", self.duplicate_decode_export_wall_time_ns),
            ("per_row_audit_wall_time_ns", self.per_row_audit_wall_time_ns),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if value > self.fresh_wall_time_ns:
                raise ValueError(f"{name} cannot exceed fresh_wall_time_ns")
        return self


class StructuralGateResult(StrictModel):
    """WP6 structural checks with explicit measured ratios and violation codes."""

    status: LocalGateStatus
    fresh_source_time_ratio: NonNegativeFiniteFloat
    required_state_bytes_ratio: NonNegativeFiniteFloat
    duplicate_decode_export_wall_time_share: NonNegativeFiniteFloat
    per_row_audit_wall_time_share: NonNegativeFiniteFloat
    violations: tuple[StructuralViolation, ...]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        expected = LocalGateStatus.PASS if not self.violations else LocalGateStatus.FAIL
        if self.status is not expected:
            raise ValueError("structural gate status must agree with violations")
        return self


class MockLatencySummary(StrictModel):
    """Nearest-rank terminal mock latency summary."""

    count: PositiveInt
    mean_ms: PositiveFiniteFloat
    p50_ms: PositiveInt
    p95_ms: PositiveInt
    p99_ms: PositiveInt


class MockCapacityRepetition(StrictModel):
    """One deterministic, virtually timed saturated mock repetition."""

    ordinal: NonNegativeInt
    probe_request_count: PositiveInt
    provider_attempt_count: PositiveInt
    succeeded_request_count: NonNegativeInt
    failed_request_count: NonNegativeInt
    virtual_elapsed_ms: PositiveFiniteFloat
    service_capacity_recording_seconds_per_wall_second: NonNegativeFiniteFloat
    terminal_failure_rate: NonNegativeFiniteFloat

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.succeeded_request_count + self.failed_request_count != self.probe_request_count:
            raise ValueError(
                "mock repetition terminal counts must reconcile to probe_request_count"
            )
        if self.provider_attempt_count < self.probe_request_count:
            raise ValueError("provider_attempt_count cannot be less than probe_request_count")
        if self.terminal_failure_rate > 1:
            raise ValueError("terminal_failure_rate cannot exceed 1")
        return self


class OfferedLoadEstimate(StrictModel):
    """Virtual 30-minute stability arithmetic for one offered-load interpretation."""

    unit: OfferedLoadUnit
    offered_hours_per_day: Literal[500]
    provisioned_six_camera_groups: PositiveInt
    offered_recording_seconds_per_wall_second: PositiveFiniteFloat
    offered_camera_video_seconds_per_wall_second: PositiveFiniteFloat
    planned_recording_capacity_per_group: NonNegativeFiniteFloat
    planned_total_recording_capacity: NonNegativeFiniteFloat
    required_six_camera_groups: PositiveInt | None
    virtual_smoke_duration_ms: PositiveInt
    virtual_peak_required_backlog_recording_seconds: NonNegativeFiniteFloat
    virtual_end_required_backlog_recording_seconds: NonNegativeFiniteFloat
    required_backlog_growing: bool
    status: LocalGateStatus

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        expected = LocalGateStatus.FAIL if self.required_backlog_growing else LocalGateStatus.PASS
        if self.status is not expected:
            raise ValueError("offered-load status must agree with required_backlog_growing")
        return self


class StreamingGateResult(StrictModel):
    """Mock-only WP6 streaming capacity, latency, and backlog checks."""

    status: LocalGateStatus
    capacity_repetitions: tuple[MockCapacityRepetition, ...]
    sustained_service_capacity_recording_seconds_per_wall_second: NonNegativeFiniteFloat
    terminal_failure_rate: NonNegativeFiniteFloat
    incremental_latency: MockLatencySummary
    offered_loads: tuple[OfferedLoadEstimate, ...]
    violations: tuple[StreamingViolation, ...]

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if not self.capacity_repetitions:
            raise ValueError("capacity_repetitions must be nonempty")
        expected = LocalGateStatus.PASS if not self.violations else LocalGateStatus.FAIL
        if self.status is not expected:
            raise ValueError("streaming gate status must agree with violations")
        units = tuple(item.unit for item in self.offered_loads)
        if units != tuple(OfferedLoadUnit):
            raise ValueError("streaming offered-load results must preserve both manifest scenarios")
        return self


def local_streaming_virtual_estimate_report_projection(
    report: LocalStreamingVirtualEstimateReport,
) -> dict[str, object]:
    """Return the canonical report preimage, excluding its self digest."""

    return report.model_dump(mode="json", exclude={"report_sha256"})


class LocalStreamingVirtualEstimateReport(StrictModel):
    """Non-authoritative simulator output that cannot satisfy the actual WP6 smoke gate."""

    schema_version: Literal["1.0"]
    model_version: Literal["local-streaming-virtual-estimate-report-v1"]
    report_sha256: Sha256Digest
    manifest_sha256: Sha256Digest
    structural_gate: StructuralGateResult
    streaming_gate: StreamingGateResult
    simulated_gate_status: LocalGateStatus
    provider_traffic: Literal["MOCKED"]
    execution_mode: Literal["VIRTUAL_SIMULATION_ONLY"]
    authority_status: Literal["NON_AUTHORITATIVE"]
    wp6_smoke_gate_eligible: Literal[False]
    authoritative_report_model: Literal["local-streaming-smoke-report-v1"]
    evidence_class: Literal["VIRTUAL_MODEL_DIAGNOSTIC"]
    measurement_status: Literal["NOT_MEASURED"]
    production_eligible: Literal[False]
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"]

    @classmethod
    def create(
        cls,
        *,
        manifest_sha256: Sha256Digest,
        structural_gate: StructuralGateResult,
        streaming_gate: StreamingGateResult,
    ) -> Self:
        status = (
            LocalGateStatus.PASS
            if structural_gate.status is LocalGateStatus.PASS
            and streaming_gate.status is LocalGateStatus.PASS
            else LocalGateStatus.FAIL
        )
        draft = cls.model_construct(
            schema_version="1.0",
            model_version=LOCAL_STREAMING_VIRTUAL_ESTIMATE_REPORT_VERSION,
            report_sha256="0" * 64,
            manifest_sha256=manifest_sha256,
            structural_gate=structural_gate,
            streaming_gate=streaming_gate,
            simulated_gate_status=status,
            provider_traffic="MOCKED",
            execution_mode="VIRTUAL_SIMULATION_ONLY",
            authority_status="NON_AUTHORITATIVE",
            wp6_smoke_gate_eligible=False,
            authoritative_report_model="local-streaming-smoke-report-v1",
            evidence_class="VIRTUAL_MODEL_DIAGNOSTIC",
            measurement_status="NOT_MEASURED",
            production_eligible=False,
            qualification_status="NOT_PRODUCTION_QUALIFIED",
        )
        digest = semantic_sha256(local_streaming_virtual_estimate_report_projection(draft))
        return cls.model_validate(
            {**draft.model_dump(mode="python"), "report_sha256": digest},
            strict=True,
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        expected_status = (
            LocalGateStatus.PASS
            if self.structural_gate.status is LocalGateStatus.PASS
            and self.streaming_gate.status is LocalGateStatus.PASS
            else LocalGateStatus.FAIL
        )
        if self.simulated_gate_status is not expected_status:
            raise ValueError("simulated_gate_status must agree with both simulated gates")
        expected_digest = semantic_sha256(local_streaming_virtual_estimate_report_projection(self))
        if self.report_sha256 != expected_digest:
            raise ValueError("report_sha256 does not match canonical report bytes")
        return self


def evaluate_local_streaming_virtual_estimate(
    manifest: LocalStreamingBenchmarkManifest,
    structural_observation: StructuralBenchmarkObservation,
) -> LocalStreamingVirtualEstimateReport:
    """Evaluate structural observations and a deterministic virtual provider workload."""

    if not isinstance(manifest, LocalStreamingBenchmarkManifest):
        raise TypeError("manifest must be a LocalStreamingBenchmarkManifest")
    if not isinstance(structural_observation, StructuralBenchmarkObservation):
        raise TypeError("structural_observation must be a StructuralBenchmarkObservation")

    structural_gate = _evaluate_structural_gate(manifest, structural_observation)
    repetitions, terminal_latencies = _simulate_mock_capacity(manifest)
    sustained_capacity = min(
        item.service_capacity_recording_seconds_per_wall_second for item in repetitions
    )
    attempted = sum(item.probe_request_count for item in repetitions)
    failed = sum(item.failed_request_count for item in repetitions)
    latency_summary = _latency_summary(terminal_latencies)
    load_results = tuple(
        _evaluate_offered_load(manifest, scenario, sustained_capacity)
        for scenario in manifest.offered_loads
    )

    streaming_violations: list[StreamingViolation] = []
    if sustained_capacity < WP6_MINIMUM_SERVICE_CAPACITY:
        streaming_violations.append(StreamingViolation.SERVICE_CAPACITY)
    if latency_summary.p95_ms > manifest.protocol.incremental_latency_target_ms:
        streaming_violations.append(StreamingViolation.INCREMENTAL_LATENCY)
    for result in load_results:
        if not result.required_backlog_growing:
            continue
        if result.unit is OfferedLoadUnit.RECORDING_HOURS_PER_DAY:
            streaming_violations.append(StreamingViolation.RECORDING_HOURS_BACKLOG_GROWTH)
        else:
            streaming_violations.append(StreamingViolation.CAMERA_VIDEO_HOURS_BACKLOG_GROWTH)

    streaming_gate = StreamingGateResult(
        status=(LocalGateStatus.FAIL if streaming_violations else LocalGateStatus.PASS),
        capacity_repetitions=repetitions,
        sustained_service_capacity_recording_seconds_per_wall_second=sustained_capacity,
        terminal_failure_rate=failed / attempted,
        incremental_latency=latency_summary,
        offered_loads=load_results,
        violations=tuple(streaming_violations),
    )
    return LocalStreamingVirtualEstimateReport.create(
        manifest_sha256=manifest.manifest_sha256,
        structural_gate=structural_gate,
        streaming_gate=streaming_gate,
    )


def _evaluate_structural_gate(
    manifest: LocalStreamingBenchmarkManifest,
    observation: StructuralBenchmarkObservation,
) -> StructuralGateResult:
    fresh_ratio = observation.fresh_wall_time_ns / manifest.candidate.source_duration_ns
    state_ratio = observation.required_state_bytes / manifest.candidate.source_byte_count
    duplicate_share = (
        observation.duplicate_decode_export_wall_time_ns / observation.fresh_wall_time_ns
    )
    audit_share = observation.per_row_audit_wall_time_ns / observation.fresh_wall_time_ns
    violations: list[StructuralViolation] = []
    if fresh_ratio > WP6_MAXIMUM_FRESH_SOURCE_TIME_RATIO:
        violations.append(StructuralViolation.FRESH_SOURCE_TIME)
    if state_ratio > WP6_MAXIMUM_REQUIRED_STATE_BYTES_RATIO:
        violations.append(StructuralViolation.REQUIRED_STATE_BYTES)
    if duplicate_share >= WP6_DOMINANT_WALL_TIME_SHARE:
        violations.append(StructuralViolation.DUPLICATE_DECODE_EXPORT_DOMINANT)
    if audit_share >= WP6_DOMINANT_WALL_TIME_SHARE:
        violations.append(StructuralViolation.PER_ROW_AUDIT_DOMINANT)
    return StructuralGateResult(
        status=LocalGateStatus.FAIL if violations else LocalGateStatus.PASS,
        fresh_source_time_ratio=fresh_ratio,
        required_state_bytes_ratio=state_ratio,
        duplicate_decode_export_wall_time_share=duplicate_share,
        per_row_audit_wall_time_share=audit_share,
        violations=tuple(violations),
    )


def _simulate_mock_capacity(
    manifest: LocalStreamingBenchmarkManifest,
) -> tuple[tuple[MockCapacityRepetition, ...], tuple[int, ...]]:
    provider = manifest.mock_provider
    probe_request_count = math.ceil(
        manifest.protocol.observation_cutoff_ms / manifest.policies.window_hop_ms
    )
    repetitions: list[MockCapacityRepetition] = []
    all_latencies: list[int] = []
    for ordinal in range(manifest.protocol.repetition_count):
        draw_ordinal = manifest.protocol.warmup_count + ordinal
        lane_available_ms = [0 for _ in range(provider.max_concurrency_per_group)]
        heapq.heapify(lane_available_ms)
        provider_attempt_count = 0
        succeeded = 0
        failed = 0
        for request_ordinal in range(probe_request_count):
            terminal_latency_ms = 0
            request_succeeded = False
            for attempt in range(provider.retry.maximum_attempts):
                sampled_latency_ms = _draw_latency_ms(
                    manifest,
                    draw_ordinal,
                    request_ordinal,
                    attempt,
                )
                timed_out = sampled_latency_ms > provider.request_timeout_ms
                terminal_latency_ms += min(sampled_latency_ms, provider.request_timeout_ms)
                provider_attempt_count += 1
                injected_failure = _draw_failure(
                    manifest,
                    draw_ordinal,
                    request_ordinal,
                    attempt,
                )
                if not timed_out and not injected_failure:
                    request_succeeded = True
                    break
                if attempt + 1 < provider.retry.maximum_attempts:
                    terminal_latency_ms += provider.retry.backoff_ms[attempt]

            available_ms = heapq.heappop(lane_available_ms)
            heapq.heappush(lane_available_ms, available_ms + terminal_latency_ms)
            all_latencies.append(terminal_latency_ms)
            if request_succeeded:
                succeeded += 1
            else:
                failed += 1

        concurrency_elapsed_ms = max(lane_available_ms)
        request_limit_elapsed_ms = (
            provider_attempt_count * 1_000 / provider.request_limit_per_second
        )
        virtual_elapsed_ms = max(float(concurrency_elapsed_ms), request_limit_elapsed_ms)
        recording_work_ms = succeeded * provider.max_batch_size * manifest.policies.window_hop_ms
        repetitions.append(
            MockCapacityRepetition(
                ordinal=ordinal,
                probe_request_count=probe_request_count,
                provider_attempt_count=provider_attempt_count,
                succeeded_request_count=succeeded,
                failed_request_count=failed,
                virtual_elapsed_ms=virtual_elapsed_ms,
                service_capacity_recording_seconds_per_wall_second=(
                    recording_work_ms / virtual_elapsed_ms
                ),
                terminal_failure_rate=failed / probe_request_count,
            )
        )
    return tuple(repetitions), tuple(all_latencies)


def _draw_latency_ms(
    manifest: LocalStreamingBenchmarkManifest,
    repetition: int,
    request: int,
    attempt: int,
) -> int:
    distribution = manifest.mock_provider.latency
    total_weight = sum(point.weight for point in distribution.points)
    draw = _digest_draw(manifest, repetition, request, attempt, "latency") % total_weight
    cumulative = 0
    for point in distribution.points:
        cumulative += point.weight
        if draw < cumulative:
            return point.latency_ms
    raise AssertionError("weighted latency selection did not terminate")


def _draw_failure(
    manifest: LocalStreamingBenchmarkManifest,
    repetition: int,
    request: int,
    attempt: int,
) -> bool:
    threshold = manifest.mock_provider.failure.failure_probability_ppm
    if threshold == 0:
        return False
    if threshold == 1_000_000:
        return True
    draw = _digest_draw(manifest, repetition, request, attempt, "failure") % 1_000_000
    return draw < threshold


def _digest_draw(
    manifest: LocalStreamingBenchmarkManifest,
    repetition: int,
    request: int,
    attempt: int,
    channel: str,
) -> int:
    preimage = (
        f"{manifest.manifest_sha256}:{manifest.mock_provider.seed}:"
        f"{repetition}:{request}:{attempt}:{channel}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")


def _latency_summary(values: tuple[int, ...]) -> MockLatencySummary:
    ordered = tuple(sorted(values))
    return MockLatencySummary(
        count=len(ordered),
        mean_ms=sum(ordered) / len(ordered),
        p50_ms=_nearest_rank(ordered, 0.50),
        p95_ms=_nearest_rank(ordered, 0.95),
        p99_ms=_nearest_rank(ordered, 0.99),
    )


def _nearest_rank(ordered_values: tuple[int, ...], quantile: float) -> int:
    index = max(0, math.ceil(quantile * len(ordered_values)) - 1)
    return ordered_values[index]


def _evaluate_offered_load(
    manifest: LocalStreamingBenchmarkManifest,
    scenario: OfferedLoadScenario,
    service_capacity: float,
) -> OfferedLoadEstimate:
    planned_capacity_per_group = service_capacity * WP6_PLANNING_UTILIZATION / WP6_PLANNING_HEADROOM
    planned_total_capacity = planned_capacity_per_group * scenario.provisioned_six_camera_groups
    offered_rate = scenario.recording_seconds_per_wall_second
    required_groups = (
        None
        if planned_capacity_per_group == 0
        else math.ceil(offered_rate / planned_capacity_per_group)
    )
    backlog_growing = offered_rate > planned_total_capacity + 1e-12
    duration_seconds = manifest.protocol.smoke_duration_ms / 1_000
    growth = max(0.0, offered_rate - planned_total_capacity) * duration_seconds
    burst_peak = _one_cycle_burst_peak(
        manifest.protocol.burst_shape,
        offered_rate=offered_rate,
        service_capacity=planned_total_capacity,
    )
    return OfferedLoadEstimate(
        unit=scenario.unit,
        offered_hours_per_day=scenario.offered_hours_per_day,
        provisioned_six_camera_groups=scenario.provisioned_six_camera_groups,
        offered_recording_seconds_per_wall_second=offered_rate,
        offered_camera_video_seconds_per_wall_second=(
            scenario.camera_video_seconds_per_wall_second
        ),
        planned_recording_capacity_per_group=planned_capacity_per_group,
        planned_total_recording_capacity=planned_total_capacity,
        required_six_camera_groups=required_groups,
        virtual_smoke_duration_ms=manifest.protocol.smoke_duration_ms,
        virtual_peak_required_backlog_recording_seconds=burst_peak + growth,
        virtual_end_required_backlog_recording_seconds=growth,
        required_backlog_growing=backlog_growing,
        status=LocalGateStatus.FAIL if backlog_growing else LocalGateStatus.PASS,
    )


def _one_cycle_burst_peak(
    burst: BurstShapePins,
    *,
    offered_rate: float,
    service_capacity: float,
) -> float:
    mean_weight = sum(burst.relative_load_pattern) / len(burst.relative_load_pattern)
    bucket_seconds = burst.bucket_duration_ms / 1_000
    backlog = 0.0
    peak = 0.0
    for weight in burst.relative_load_pattern:
        bucket_rate = offered_rate * weight / mean_weight
        backlog = max(0.0, backlog + (bucket_rate - service_capacity) * bucket_seconds)
        peak = max(peak, backlog)
    return peak


__all__ = [
    "LOCAL_STREAMING_BENCHMARK_MANIFEST_VERSION",
    "LOCAL_STREAMING_VIRTUAL_ESTIMATE_REPORT_VERSION",
    "WP6_DOMINANT_WALL_TIME_SHARE",
    "WP6_MAXIMUM_FRESH_SOURCE_TIME_RATIO",
    "WP6_MAXIMUM_REQUIRED_STATE_BYTES_RATIO",
    "WP6_MINIMUM_SERVICE_CAPACITY",
    "WP6_MINIMUM_SMOKE_DURATION_MS",
    "WP6_PLANNING_HEADROOM",
    "WP6_PLANNING_UTILIZATION",
    "BenchmarkCacheState",
    "BenchmarkProtocolPins",
    "BenchmarkRunMode",
    "BenchmarkRunStatePins",
    "BurstShapePins",
    "CandidateSourcePins",
    "HostRuntimePins",
    "LocalGateStatus",
    "LocalStreamingBenchmarkManifest",
    "LocalStreamingVirtualEstimateReport",
    "LongStreamKind",
    "MockCapacityRepetition",
    "MockFailureDistribution",
    "MockLatencyDistribution",
    "MockLatencyPoint",
    "MockLatencySummary",
    "MockProviderPins",
    "MockRetryPolicyPins",
    "OfferedLoadEstimate",
    "OfferedLoadScenario",
    "OfferedLoadUnit",
    "StreamingGateResult",
    "StreamingPolicyPins",
    "StreamingViolation",
    "StructuralBenchmarkObservation",
    "StructuralGateResult",
    "StructuralViolation",
    "evaluate_local_streaming_virtual_estimate",
    "local_streaming_benchmark_manifest_projection",
    "local_streaming_virtual_estimate_report_projection",
]
