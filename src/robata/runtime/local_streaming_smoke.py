"""Executable local mock smoke for the durable streaming composition.

Unlike :mod:`robata.runtime.local_streaming_benchmark`, this module advances the
real SQLite-backed stream scheduler and records wall-clock observations. It uses
the deterministic local mock inference path; no external-provider measurement or
production-qualification claim is made.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Final, Literal, Self
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import Field, StringConstraints, model_validator

from robata.adapters.sqlite_stream_delivery import SQLiteStreamDeliveryAuthority
from robata.adapters.sqlite_work_scheduler import SQLiteWorkScheduler
from robata.application.canonical.bounded_media import (
    BoundedWindowPlan,
    CameraStreamFacts,
    CameraWindowPlan,
    PlannerFinish,
    WindowClosureReason,
    WindowMember,
)
from robata.application.canonical.local_stream_finalization import (
    LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID,
    LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION,
    FinalRecordingFacts,
    LocalConformanceStreamFinalizer,
    LocalStreamFinalizationSchemaRefs,
)
from robata.application.canonical.stream_recording_reduction import (
    LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
    LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
)
from robata.application.canonical.stream_scheduler import (
    DurableStreamWindowScheduler,
    EosSealInputs,
    StreamDrainWorkSnapshot,
    StreamSchedulerSchemaRefs,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import NanosecondInterval, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.local_stream_causal import (
    LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID,
    LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION,
)
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.contracts.stream_common import (
    AuthorityBinding,
    CameraAbsenceReason,
    ChannelBinding,
    StreamPolicyBinding,
    StreamPurpose,
    StreamStage,
)
from robata.contracts.stream_finalization import (
    RECORDING_FINALIZATION_SCHEMA_ID,
    RECORDING_FINALIZATION_SCHEMA_VERSION,
    WINDOW_TERMINAL_CLOSURE_SCHEMA_ID,
    WINDOW_TERMINAL_CLOSURE_SCHEMA_VERSION,
    WINDOW_TERMINAL_MEMBER_SCHEMA_ID,
    WINDOW_TERMINAL_MEMBER_SCHEMA_VERSION,
)
from robata.contracts.stream_inference import (
    STREAM_ACCEPTED_CALL_SCHEMA_ID,
    STREAM_ACCEPTED_CALL_SCHEMA_VERSION,
    STREAM_INFERENCE_INTENT_SCHEMA_ID,
    STREAM_INFERENCE_INTENT_SCHEMA_VERSION,
    STREAM_INFERENCE_TERMINAL_SCHEMA_ID,
    STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION,
    STREAM_WINDOW_RESULT_SCHEMA_ID,
    STREAM_WINDOW_RESULT_SCHEMA_VERSION,
)
from robata.contracts.stream_planning import (
    EXPECTED_WINDOW_DECLARATION_SCHEMA_ID,
    EXPECTED_WINDOW_DECLARATION_SCHEMA_VERSION,
    EXPECTED_WINDOW_PLAN_SCHEMA_ID,
    EXPECTED_WINDOW_PLAN_SCHEMA_VERSION,
    EXPECTED_WINDOW_SEAL_SCHEMA_ID,
    EXPECTED_WINDOW_SEAL_SCHEMA_VERSION,
    STREAM_WORK_PLAN_SCHEMA_ID,
    STREAM_WORK_PLAN_SCHEMA_VERSION,
    create_expected_window_plan,
)
from robata.contracts.stream_source import (
    PRE_EOS_CAPTURE_SCHEMA_ID,
    PRE_EOS_CAPTURE_SCHEMA_VERSION,
    create_pre_eos_capture_subject,
)
from robata.contracts.stream_window import (
    INCREMENTAL_WINDOW_SCHEMA_ID,
    INCREMENTAL_WINDOW_SCHEMA_VERSION,
    STREAM_INFERENCE_ATTEMPT_SCHEMA_ID,
    STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION,
    STREAM_INFERENCE_SCHEMA_ID,
    STREAM_INFERENCE_SCHEMA_VERSION,
)
from robata.queue.outbox import OutboxRetryPolicy
from robata.runtime.local_streaming_benchmark import (
    WP6_MINIMUM_SERVICE_CAPACITY,
    WP6_MINIMUM_SMOKE_DURATION_MS,
    WP6_PLANNING_HEADROOM,
    WP6_PLANNING_UTILIZATION,
    BenchmarkCacheState,
    BenchmarkProtocolPins,
    BenchmarkRunMode,
    BenchmarkRunStatePins,
    BurstShapePins,
    CandidateSourcePins,
    HostRuntimePins,
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
)

LOCAL_STREAMING_SMOKE_MANIFEST_VERSION: Final = "local-streaming-smoke-manifest-v1"
LOCAL_STREAMING_SMOKE_REPORT_VERSION: Final = "local-streaming-smoke-report-v1"
DEFAULT_SOURCE_DURATION_MS: Final = WP6_MINIMUM_SMOKE_DURATION_MS
WP6_INCREMENTAL_P95_TARGET_MS: Final = 5_000
WP6_INCREMENTAL_P99_TARGET_MS: Final = 15_000

_MOCK_PROVIDER_STAGE_ORDER: Final = (
    StreamStage.QA_COARSE,
    StreamStage.QA_DENSE,
    StreamStage.EVENT_PROPOSAL,
    StreamStage.ACTION_DENSE,
    StreamStage.BOUNDARY_REFINEMENT,
)

_SMOKE_NAMESPACE: Final = uuid5(NAMESPACE_URL, "robata:local-streaming-smoke-v1")
_SCHEMA_VERSION: Final = "1.0"

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PartsPerMillion = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
FiniteNonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
GitCommit = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]


class _WindowsMemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class LocalStreamingSmokeConfig(StrictModel):
    """Pinned event-time generation and bounded mock-execution controls."""

    source_duration_ms: PositiveInt = DEFAULT_SOURCE_DURATION_MS
    chunk_duration_ms: PositiveInt = 1_000
    window_duration_ms: PositiveInt = 2_000
    window_hop_ms: PositiveInt = 1_000
    allowed_lateness_ms: NonNegativeInt = 0
    ring_capacity_ms: PositiveInt = 2_000
    window_batch_size: PositiveInt = 8
    drain_batch_size: PositiveInt = 64
    mock_fixed_latency_ms: PositiveInt = 5
    mock_failure_probability_ppm: PartsPerMillion = 10_000
    mock_failure_seed: NonNegativeInt = 29
    mock_retry_limit: NonNegativeInt = 2
    mock_request_timeout_ms: PositiveInt = 30_000
    mock_request_limit_per_second: PositiveInt = 1_000
    mock_max_batch_size: PositiveInt = 16
    incremental_latency_p95_target_ms: PositiveInt = WP6_INCREMENTAL_P95_TARGET_MS
    incremental_latency_p99_target_ms: PositiveInt = WP6_INCREMENTAL_P99_TARGET_MS
    sqlite_synchronous: Literal["FULL", "NORMAL"] = "NORMAL"

    @model_validator(mode="after")
    def validate_window_shape(self) -> Self:
        if self.window_hop_ms > self.window_duration_ms:
            raise ValueError("window_hop_ms cannot exceed window_duration_ms")
        if self.chunk_duration_ms > self.ring_capacity_ms:
            raise ValueError("chunk_duration_ms cannot exceed ring_capacity_ms")
        if self.ring_capacity_ms < self.window_duration_ms + self.allowed_lateness_ms:
            raise ValueError("ring_capacity_ms must retain one window plus allowed lateness")
        return self


def local_streaming_smoke_manifest_projection(
    manifest: LocalStreamingSmokeManifest,
) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"manifest_sha256"})


class LocalStreamingSmokeManifest(StrictModel):
    """Content-addressed input closure written before real local execution."""

    schema_version: Literal["1.0"]
    model_version: Literal["local-streaming-smoke-manifest-v1"]
    manifest_sha256: Sha256Digest
    candidate_commit: GitCommit
    candidate_worktree_state: Literal["CLEAN", "DIRTY"]
    candidate_worktree_status_sha256: Sha256Digest
    lockfile_sha256: Sha256Digest
    generated_source_sha256: Sha256Digest
    generated_source_descriptor_byte_count: PositiveInt
    host: HostRuntimePins
    config: LocalStreamingSmokeConfig
    benchmark_manifest: LocalStreamingBenchmarkManifest
    source_kind: Literal["GENERATED_EVENT_TIME"]
    run_state: Literal["FRESH"]
    execution_mode: Literal["ACTUAL_LOCAL_MOCK"]
    provider_traffic: Literal["MOCKED"]
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    measurement_status: Literal["NOT_MEASURED"]
    production_eligible: Literal[False]
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"]

    @classmethod
    def create(
        cls,
        *,
        candidate_commit: GitCommit,
        candidate_worktree_state: Literal["CLEAN", "DIRTY"],
        candidate_worktree_status_sha256: Sha256Digest,
        lockfile_sha256: Sha256Digest,
        host: HostRuntimePins,
        config: LocalStreamingSmokeConfig,
    ) -> Self:
        source_descriptor = canonical_json_bytes(
            {
                "source_kind": "GENERATED_EVENT_TIME",
                "source_duration_ms": config.source_duration_ms,
                "chunk_duration_ms": config.chunk_duration_ms,
                "window_duration_ms": config.window_duration_ms,
                "window_hop_ms": config.window_hop_ms,
                "allowed_lateness_ms": config.allowed_lateness_ms,
                "ring_capacity_ms": config.ring_capacity_ms,
                "camera_ids": [camera.value for camera in CAMERA_IDS],
            }
        )
        generated_source_sha256 = exact_bytes_sha256(source_descriptor)
        benchmark_manifest = _create_benchmark_manifest(
            candidate_commit=candidate_commit,
            lockfile_sha256=lockfile_sha256,
            generated_source_sha256=generated_source_sha256,
            generated_source_descriptor_byte_count=len(source_descriptor),
            host=host,
            config=config,
        )
        draft = cls.model_construct(
            schema_version=_SCHEMA_VERSION,
            model_version=LOCAL_STREAMING_SMOKE_MANIFEST_VERSION,
            manifest_sha256="0" * 64,
            candidate_commit=candidate_commit,
            candidate_worktree_state=candidate_worktree_state,
            candidate_worktree_status_sha256=candidate_worktree_status_sha256,
            lockfile_sha256=lockfile_sha256,
            generated_source_sha256=generated_source_sha256,
            generated_source_descriptor_byte_count=len(source_descriptor),
            host=host,
            config=config,
            benchmark_manifest=benchmark_manifest,
            source_kind="GENERATED_EVENT_TIME",
            run_state="FRESH",
            execution_mode="ACTUAL_LOCAL_MOCK",
            provider_traffic="MOCKED",
            evidence_class="LOCAL_CONFORMANCE",
            measurement_status="NOT_MEASURED",
            production_eligible=False,
            qualification_status="NOT_PRODUCTION_QUALIFIED",
        )
        digest = semantic_sha256(local_streaming_smoke_manifest_projection(draft))
        return cls.model_validate(
            {**draft.model_dump(mode="python"), "manifest_sha256": digest},
            strict=True,
        )

    @model_validator(mode="after")
    def validate_manifest_digest(self) -> Self:
        if self.benchmark_manifest.candidate.candidate_commit != self.candidate_commit:
            raise ValueError("benchmark candidate commit must match the actual smoke candidate")
        if self.benchmark_manifest.candidate.source_sha256 != self.generated_source_sha256:
            raise ValueError("benchmark source digest must match the generated smoke source")
        if self.benchmark_manifest.host != self.host:
            raise ValueError("benchmark host pins must match the actual smoke host")
        expected = semantic_sha256(local_streaming_smoke_manifest_projection(self))
        if self.manifest_sha256 != expected:
            raise ValueError("manifest_sha256 does not match the canonical input projection")
        return self


def _create_benchmark_manifest(
    *,
    candidate_commit: str,
    lockfile_sha256: str,
    generated_source_sha256: str,
    generated_source_descriptor_byte_count: int,
    host: HostRuntimePins,
    config: LocalStreamingSmokeConfig,
) -> LocalStreamingBenchmarkManifest:
    """Bind the complete WP6 protocol to the actual smoke before execution."""

    def policy_digest(policy: str, values: dict[str, object]) -> str:
        return semantic_sha256(
            {
                "policy_family": "local-streaming-smoke-v1",
                "policy": policy,
                "values": values,
            }
        )

    policies = StreamingPolicyPins(
        chunk_duration_ms=config.chunk_duration_ms,
        window_duration_ms=config.window_duration_ms,
        window_hop_ms=config.window_hop_ms,
        allowed_lateness_ms=config.allowed_lateness_ms,
        ring_capacity_ms=config.ring_capacity_ms,
        chunk_policy_sha256=policy_digest(
            "chunk",
            {"duration_ms": config.chunk_duration_ms},
        ),
        window_policy_sha256=policy_digest(
            "window",
            {
                "duration_ms": config.window_duration_ms,
                "hop_ms": config.window_hop_ms,
                "closure": "EVENT_TIME_WATERMARK",
            },
        ),
        lateness_policy_sha256=policy_digest(
            "lateness",
            {"allowed_lateness_ms": config.allowed_lateness_ms},
        ),
        ring_policy_sha256=policy_digest(
            "ring",
            {"capacity_ms": config.ring_capacity_ms},
        ),
        sampling_policy_sha256=policy_digest(
            "sampling",
            {"camera_count": len(CAMERA_IDS), "fixture": "ABSENCE_ONLY"},
        ),
        trigger_policy_sha256=policy_digest(
            "trigger",
            {"window_batch_size": config.window_batch_size},
        ),
        candidate_policy_sha256=policy_digest(
            "candidate",
            {"eligibility": "EVERY_DECLARED_WINDOW"},
        ),
        boundary_policy_sha256=policy_digest(
            "boundary",
            {"window": "WATERMARK", "recording": "EOS"},
        ),
        fan_out_policy_sha256=policy_digest(
            "fan-out",
            {
                "window_dag": [
                    "WINDOW",
                    "QA_COARSE",
                    "QA_DENSE",
                    "EVENT_PROPOSAL",
                    "WINDOW_REDUCTION",
                ],
                "maximum_ready_items_per_drain": config.drain_batch_size,
            },
        ),
    )
    return LocalStreamingBenchmarkManifest.create(
        candidate=CandidateSourcePins(
            candidate_commit=candidate_commit,
            source_sha256=generated_source_sha256,
            source_byte_count=generated_source_descriptor_byte_count,
            source_duration_ns=config.source_duration_ms * 1_000_000,
            lockfile_sha256=lockfile_sha256,
        ),
        host=host,
        run_states=(
            BenchmarkRunStatePins(
                mode=BenchmarkRunMode.COLD,
                cache_state=BenchmarkCacheState.EMPTY,
            ),
            BenchmarkRunStatePins(
                mode=BenchmarkRunMode.FRESH,
                cache_state=BenchmarkCacheState.DISABLED,
            ),
            BenchmarkRunStatePins(
                mode=BenchmarkRunMode.REPLAY,
                cache_state=BenchmarkCacheState.RESTORED,
            ),
        ),
        artifact_retention_profile_sha256=semantic_sha256(
            {
                "profile": "local-streaming-smoke-retention-v1",
                "content_addressed_manifest": "RETAIN",
                "content_addressed_report": "RETAIN",
                "sqlite_authority": "RETAIN",
                "work_artifacts": "RETAIN",
                "generated_source_materialization": "DESCRIPTOR_ONLY",
            }
        ),
        policies=policies,
        mock_provider=MockProviderPins(
            latency=MockLatencyDistribution(
                points=(
                    MockLatencyPoint(
                        latency_ms=config.mock_fixed_latency_ms,
                        weight=1,
                    ),
                )
            ),
            failure=MockFailureDistribution(
                failure_probability_ppm=config.mock_failure_probability_ppm,
            ),
            retry=MockRetryPolicyPins(
                maximum_attempts=config.mock_retry_limit + 1,
                backoff_ms=tuple(0 for _ in range(config.mock_retry_limit)),
            ),
            seed=config.mock_failure_seed,
            request_timeout_ms=config.mock_request_timeout_ms,
            request_limit_per_second=config.mock_request_limit_per_second,
            max_batch_size=config.mock_max_batch_size,
            max_concurrency_per_group=1,
        ),
        protocol=BenchmarkProtocolPins(
            warmup_count=0,
            repetition_count=1,
            burst_shape=BurstShapePins(
                bucket_duration_ms=config.window_batch_size * config.window_hop_ms,
                relative_load_pattern=(1,),
            ),
            observation_cutoff_ms=min(config.source_duration_ms, 60_000),
            smoke_duration_ms=WP6_MINIMUM_SMOKE_DURATION_MS,
            short_source_first=True,
            long_stream_kind=LongStreamKind.GENERATED,
            incremental_latency_target_ms=config.incremental_latency_p95_target_ms,
        ),
        offered_loads=(
            OfferedLoadScenario(
                unit=OfferedLoadUnit.RECORDING_HOURS_PER_DAY,
                offered_hours_per_day=500,
                provisioned_six_camera_groups=24,
            ),
            OfferedLoadScenario(
                unit=OfferedLoadUnit.AGGREGATE_CAMERA_VIDEO_HOURS_PER_DAY,
                offered_hours_per_day=500,
                provisioned_six_camera_groups=4,
            ),
        ),
    )


class LocalStreamingSmokeMetrics(StrictModel):
    """Wall-clock and durable-state observations from one actual local run."""

    actual_wall_time_ns: PositiveInt
    source_duration_ns: PositiveInt
    declared_window_count: PositiveInt
    terminal_window_count: PositiveInt
    total_work_item_count: PositiveInt
    bounded_drain_call_count: NonNegativeInt
    drain_opportunity_count: NonNegativeInt
    mock_provider_batch_request_count: NonNegativeInt
    mock_provider_attempt_count: NonNegativeInt
    mock_provider_timeout_count: NonNegativeInt
    mock_provider_rate_limit_wait_count: NonNegativeInt
    mock_provider_max_batch_size_observed: NonNegativeInt
    injected_retryable_failure_count: NonNegativeInt
    bounded_executed_work_count: NonNegativeInt
    eos_execute_work_count: PositiveInt
    eos_recovery_used: bool
    eligible_to_terminal_sample_count: PositiveInt
    eligible_to_terminal_p50_ns: NonNegativeInt
    eligible_to_terminal_p95_ns: NonNegativeInt
    eligible_to_terminal_p99_ns: NonNegativeInt
    eligible_to_terminal_max_ns: NonNegativeInt
    active_backlog_high_water: PositiveInt
    active_backlog_before_eos: NonNegativeInt
    active_backlog_end: NonNegativeInt
    active_backlog_after_drain_samples: tuple[NonNegativeInt, ...]
    recording_seconds_per_wall_second: FiniteNonNegativeFloat
    windows_per_wall_second: FiniteNonNegativeFloat

    @model_validator(mode="after")
    def validate_terminal_coverage(self) -> Self:
        if self.terminal_window_count != self.declared_window_count:
            raise ValueError("every declared window must have a terminal member")
        if self.eligible_to_terminal_sample_count != self.terminal_window_count:
            raise ValueError("every terminal window must have one latency sample")
        if self.active_backlog_end != 0:
            raise ValueError("completed local smoke must end without active backlog")
        if not self.active_backlog_after_drain_samples:
            raise ValueError("actual smoke must record backlog after every bounded drain")
        if self.drain_opportunity_count != self.mock_provider_attempt_count:
            raise ValueError("drain opportunities must count provider request attempts")
        if self.mock_provider_batch_request_count > self.mock_provider_attempt_count:
            raise ValueError("each provider batch request must have at least one attempt")
        return self


class ActualLocalMockOfferedLoadProjection(StrictModel):
    """Scale-out arithmetic from observed local mock capacity, never production evidence."""

    unit: OfferedLoadUnit
    offered_hours_per_day: Literal[500]
    provisioned_six_camera_groups: PositiveInt
    offered_recording_seconds_per_wall_second: FiniteNonNegativeFloat
    observed_local_mock_capacity_per_group: FiniteNonNegativeFloat
    planning_utilization: FiniteNonNegativeFloat
    planning_headroom: FiniteNonNegativeFloat
    planned_recording_capacity_per_group: FiniteNonNegativeFloat
    required_six_camera_groups: PositiveInt
    provisioned_capacity_sufficient: bool
    projection_authority: Literal["LOCAL_MOCK_SCALE_OUT_ARITHMETIC_ONLY"]
    production_eligible: Literal[False]


def local_streaming_smoke_report_projection(
    report: LocalStreamingSmokeReport,
) -> dict[str, object]:
    return report.model_dump(mode="json", exclude={"report_sha256"})


class LocalStreamingSmokeReport(StrictModel):
    """Content-addressed result that cannot be mistaken for production evidence."""

    schema_version: Literal["1.0"]
    model_version: Literal["local-streaming-smoke-report-v1"]
    report_sha256: Sha256Digest
    manifest_sha256: Sha256Digest
    execution_run_id: str
    execution_mode: Literal["ACTUAL_LOCAL_MOCK"]
    eligibility_boundary: Literal["EXPECTED_DECLARATION_DURABLE"]
    benchmark_manifest_sha256: Sha256Digest
    wp6_minimum_duration_met: bool
    local_capacity_floor_met: bool
    incremental_latency_p95_target_ms: PositiveInt
    incremental_latency_p99_target_ms: PositiveInt
    incremental_latency_p95_target_met: bool
    incremental_latency_p99_target_met: bool
    no_growing_backlog_met: bool
    wp6_actual_smoke_gate_met: bool
    offered_load_projections: tuple[ActualLocalMockOfferedLoadProjection, ...]
    recording_finalization_semantic_sha256: Sha256Digest
    recording_result_semantic_sha256: Sha256Digest
    recording_output_decision: Literal["ADMITTED", "NO_EVENTS", "ABSTAINED"]
    metrics: LocalStreamingSmokeMetrics
    provider_traffic: Literal["MOCKED"]
    evidence_class: Literal["LOCAL_CONFORMANCE"]
    measurement_status: Literal["NOT_MEASURED"]
    production_eligible: Literal[False]
    qualification_status: Literal["NOT_PRODUCTION_QUALIFIED"]
    authority_status: Literal["AUTHORITATIVE_LOCAL_MOCK_SMOKE"]

    @classmethod
    def create(
        cls,
        *,
        manifest: LocalStreamingSmokeManifest,
        execution_run_id: str,
        metrics: LocalStreamingSmokeMetrics,
        recording_finalization_semantic_sha256: Sha256Digest,
        recording_result_semantic_sha256: Sha256Digest,
        recording_output_decision: Literal["ADMITTED", "NO_EVENTS", "ABSTAINED"],
    ) -> Self:
        UUID(execution_run_id)
        if (
            metrics.mock_provider_max_batch_size_observed
            > manifest.benchmark_manifest.mock_provider.max_batch_size
        ):
            raise ValueError("observed mock provider batch exceeded the manifest limit")
        minimum_duration_met = manifest.config.source_duration_ms >= WP6_MINIMUM_SMOKE_DURATION_MS
        capacity_floor_met = (
            metrics.recording_seconds_per_wall_second >= WP6_MINIMUM_SERVICE_CAPACITY
        )
        p95_target_met = (
            metrics.eligible_to_terminal_p95_ns
            <= manifest.config.incremental_latency_p95_target_ms * 1_000_000
        )
        p99_target_met = (
            metrics.eligible_to_terminal_p99_ns
            <= manifest.config.incremental_latency_p99_target_ms * 1_000_000
        )
        backlog_gate_met = metrics.active_backlog_end == 0 and _backlog_did_not_grow(
            metrics.active_backlog_after_drain_samples
        )
        draft = cls.model_construct(
            schema_version=_SCHEMA_VERSION,
            model_version=LOCAL_STREAMING_SMOKE_REPORT_VERSION,
            report_sha256="0" * 64,
            manifest_sha256=manifest.manifest_sha256,
            execution_run_id=execution_run_id,
            execution_mode="ACTUAL_LOCAL_MOCK",
            eligibility_boundary="EXPECTED_DECLARATION_DURABLE",
            benchmark_manifest_sha256=manifest.benchmark_manifest.manifest_sha256,
            wp6_minimum_duration_met=minimum_duration_met,
            local_capacity_floor_met=capacity_floor_met,
            incremental_latency_p95_target_ms=(manifest.config.incremental_latency_p95_target_ms),
            incremental_latency_p99_target_ms=(manifest.config.incremental_latency_p99_target_ms),
            incremental_latency_p95_target_met=p95_target_met,
            incremental_latency_p99_target_met=p99_target_met,
            no_growing_backlog_met=backlog_gate_met,
            wp6_actual_smoke_gate_met=(
                minimum_duration_met
                and capacity_floor_met
                and p95_target_met
                and p99_target_met
                and backlog_gate_met
            ),
            offered_load_projections=_actual_offered_load_projections(
                manifest,
                metrics.recording_seconds_per_wall_second,
            ),
            recording_finalization_semantic_sha256=(recording_finalization_semantic_sha256),
            recording_result_semantic_sha256=recording_result_semantic_sha256,
            recording_output_decision=recording_output_decision,
            metrics=metrics,
            provider_traffic="MOCKED",
            evidence_class="LOCAL_CONFORMANCE",
            measurement_status="NOT_MEASURED",
            production_eligible=False,
            qualification_status="NOT_PRODUCTION_QUALIFIED",
            authority_status="AUTHORITATIVE_LOCAL_MOCK_SMOKE",
        )
        digest = semantic_sha256(local_streaming_smoke_report_projection(draft))
        return cls.model_validate(
            {**draft.model_dump(mode="python"), "report_sha256": digest},
            strict=True,
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        UUID(self.execution_run_id)
        units = tuple(item.unit for item in self.offered_load_projections)
        if units != tuple(OfferedLoadUnit):
            raise ValueError("actual offered-load projections must preserve both pinned scenarios")
        expected = semantic_sha256(local_streaming_smoke_report_projection(self))
        if self.report_sha256 != expected:
            raise ValueError("report_sha256 does not match the canonical observation projection")
        return self


def _actual_offered_load_projections(
    manifest: LocalStreamingSmokeManifest,
    observed_capacity: float,
) -> tuple[ActualLocalMockOfferedLoadProjection, ...]:
    planned_capacity = observed_capacity * WP6_PLANNING_UTILIZATION / WP6_PLANNING_HEADROOM
    return tuple(
        ActualLocalMockOfferedLoadProjection(
            unit=scenario.unit,
            offered_hours_per_day=scenario.offered_hours_per_day,
            provisioned_six_camera_groups=scenario.provisioned_six_camera_groups,
            offered_recording_seconds_per_wall_second=(scenario.recording_seconds_per_wall_second),
            observed_local_mock_capacity_per_group=observed_capacity,
            planning_utilization=WP6_PLANNING_UTILIZATION,
            planning_headroom=WP6_PLANNING_HEADROOM,
            planned_recording_capacity_per_group=planned_capacity,
            required_six_camera_groups=math.ceil(
                scenario.recording_seconds_per_wall_second / planned_capacity
            ),
            provisioned_capacity_sufficient=(
                planned_capacity * scenario.provisioned_six_camera_groups
                >= scenario.recording_seconds_per_wall_second
            ),
            projection_authority="LOCAL_MOCK_SCALE_OUT_ARITHMETIC_ONLY",
            production_eligible=False,
        )
        for scenario in manifest.benchmark_manifest.offered_loads
    )


@dataclass(frozen=True, slots=True)
class LocalStreamingSmokeArtifacts:
    manifest: LocalStreamingSmokeManifest
    report: LocalStreamingSmokeReport
    manifest_path: Path
    report_path: Path
    database_path: Path


def create_manifest_from_repository(
    *,
    repository_root: str | Path,
    config: LocalStreamingSmokeConfig,
    host: HostRuntimePins | None = None,
) -> LocalStreamingSmokeManifest:
    """Pin the current repository, lockfile, host, and generated source."""

    root = Path(repository_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    lockfile = root / "uv.lock"
    if not lockfile.is_file():
        raise FileNotFoundError(f"dependency lockfile is missing: {lockfile}")
    return LocalStreamingSmokeManifest.create(
        candidate_commit=commit,
        candidate_worktree_state="CLEAN" if not status else "DIRTY",
        candidate_worktree_status_sha256=exact_bytes_sha256(status),
        lockfile_sha256=exact_bytes_sha256(lockfile.read_bytes()),
        host=collect_local_host_runtime_pins() if host is None else host,
        config=config,
    )


def collect_local_host_runtime_pins() -> HostRuntimePins:
    """Collect portable host pins without an optional system-information package."""

    cpu_model = platform.processor().strip() or platform.machine().strip() or "UNKNOWN_CPU"
    logical_cpu_count = os.cpu_count() or 1
    return HostRuntimePins(
        cpu_model=cpu_model,
        logical_cpu_count=logical_cpu_count,
        gpu_model="NOT_PROBED",
        memory_bytes=_physical_memory_bytes(),
        driver_version="NOT_PROBED",
        operating_system=platform.platform(),
        power_mode="NOT_PROBED",
        runtime=f"{platform.python_implementation()} {platform.python_version()}",
    )


def run_local_streaming_smoke(
    *,
    manifest: LocalStreamingSmokeManifest,
    output_root: str | Path,
    execution_run_id: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
) -> LocalStreamingSmokeArtifacts:
    """Run generated event-time windows through the actual durable local graph."""

    run_id = str(uuid4()) if execution_run_id is None else str(UUID(execution_run_id))
    root = Path(output_root).resolve()
    manifest_path = _write_content_addressed(
        root / "evidence" / "manifests",
        manifest.manifest_sha256,
        canonical_json_bytes(manifest),
    )
    state_root = root / "runs" / run_id
    state_root.mkdir(parents=True, exist_ok=False)
    database_path = state_root / "stream.sqlite3"
    artifact_root = state_root / "artifacts"

    started_ns = perf_counter_ns()
    scheduler, finalizer, capture_scope_digest = _build_composition(
        manifest=manifest,
        database_path=database_path,
        artifact_root=artifact_root,
        stream_run_id=run_id,
    )

    eligible_at_ns: dict[int, int] = {}
    latencies_ns: list[int] = []
    next_terminal_ordinal = 0
    backlog_after_drain_samples: list[int] = []
    high_water = 0
    drain_calls = 0
    drain_opportunities = 0
    provider_batch_requests = 0
    provider_attempts = 0
    provider_timeouts = 0
    provider_rate_limit_waits = 0
    provider_max_batch_size_observed = 0
    injected_failures = 0
    bounded_executed = 0
    failure_draw_ordinal = 0
    last_provider_attempt_duration_ms: float | None = None

    def observe_backlog() -> int:
        nonlocal high_water
        active = scheduler.backlog().active_backlog
        high_water = max(high_water, active)
        return active

    def observe_terminals() -> None:
        nonlocal next_terminal_ordinal
        observed_at = perf_counter_ns()
        while next_terminal_ordinal in eligible_at_ns:
            member = scheduler.terminal_member_at(next_terminal_ordinal)
            if member is None:
                break
            latencies_ns.append(max(0, observed_at - eligible_at_ns[next_terminal_ordinal]))
            next_terminal_ordinal += 1

    def execute_mock_provider_batches(
        scope: tuple[StreamDrainWorkSnapshot, ...],
    ) -> bool:
        nonlocal drain_opportunities
        nonlocal failure_draw_ordinal
        nonlocal injected_failures
        nonlocal last_provider_attempt_duration_ms
        nonlocal provider_attempts
        nonlocal provider_batch_requests
        nonlocal provider_max_batch_size_observed
        nonlocal provider_rate_limit_waits
        nonlocal provider_timeouts

        provider = manifest.benchmark_manifest.mock_provider
        latency_ms = provider.latency.points[0].latency_ms
        minimum_request_interval_ms = 1_000 / provider.request_limit_per_second
        for batch in _mock_provider_batches(
            scope,
            max_batch_size=provider.max_batch_size,
        ):
            provider_batch_requests += 1
            provider_max_batch_size_observed = max(
                provider_max_batch_size_observed,
                len(batch),
            )
            batch_succeeded = False
            for attempt_index in range(provider.retry.maximum_attempts):
                if last_provider_attempt_duration_ms is not None:
                    wait_ms = max(
                        0.0,
                        minimum_request_interval_ms - last_provider_attempt_duration_ms,
                    )
                    if wait_ms:
                        provider_rate_limit_waits += 1
                        sleep(wait_ms / 1_000)

                provider_attempts += 1
                drain_opportunities += 1
                attempt_duration_ms = min(latency_ms, provider.request_timeout_ms)
                sleep(attempt_duration_ms / 1_000)
                last_provider_attempt_duration_ms = attempt_duration_ms
                failure_draw = _failure_draw(
                    seed=provider.seed,
                    ordinal=failure_draw_ordinal,
                )
                failure_draw_ordinal += 1

                if latency_ms > provider.request_timeout_ms:
                    provider_timeouts += 1
                elif failure_draw < provider.failure.failure_probability_ppm:
                    injected_failures += 1
                else:
                    batch_succeeded = True
                    break

                if attempt_index < provider.retry.maximum_attempts - 1:
                    backoff_ms = provider.retry.backoff_ms[attempt_index]
                    if backoff_ms:
                        sleep(backoff_ms / 1_000)
            if not batch_succeeded:
                return False
        return True

    def bounded_drain() -> None:
        nonlocal bounded_executed
        nonlocal drain_calls
        scope = scheduler.bounded_drain_scope(manifest.config.drain_batch_size)
        if not execute_mock_provider_batches(scope):
            backlog_after_drain_samples.append(observe_backlog())
            return
        bounded_executed += finalizer.drain_ready(
            max_items=manifest.config.drain_batch_size,
            scope=scope,
        )
        drain_calls += 1
        observe_terminals()
        backlog_after_drain_samples.append(observe_backlog())

    window_count = _window_count(manifest.config)
    for batch_start in range(0, window_count, manifest.config.window_batch_size):
        batch_end = min(batch_start + manifest.config.window_batch_size, window_count)
        for ordinal in range(batch_start, batch_end):
            scheduler.append_window(
                _window(
                    manifest,
                    ordinal,
                    capture_scope_digest=capture_scope_digest,
                )
            )
            eligible_at_ns[ordinal] = perf_counter_ns()
        observe_backlog()
        bounded_drain()

    catchup_cycles = max(
        1,
        math.ceil(max(1, observe_backlog()) / manifest.config.drain_batch_size),
    )
    for _cycle in range(catchup_cycles):
        if observe_backlog() == 0:
            break
        bounded_drain()
    backlog_before_eos = observe_backlog()

    _close_eos(scheduler, manifest)
    eos_recovery_used = backlog_before_eos > 0
    if eos_recovery_used:
        execute_mock_provider_batches(scheduler.bounded_drain_scope(max(1, backlog_before_eos)))
    outcome = finalizer.execute()
    observe_terminals()
    backlog_end = observe_backlog()
    finished_ns = perf_counter_ns()

    wall_ns = max(1, finished_ns - started_ns)
    source_duration_ns = manifest.config.source_duration_ms * 1_000_000
    declared = len(scheduler.declarations())
    terminals = scheduler.terminal_member_count()
    metrics = LocalStreamingSmokeMetrics(
        actual_wall_time_ns=wall_ns,
        source_duration_ns=source_duration_ns,
        declared_window_count=declared,
        terminal_window_count=terminals,
        total_work_item_count=len(scheduler.work_plans()),
        bounded_drain_call_count=drain_calls,
        drain_opportunity_count=drain_opportunities,
        mock_provider_batch_request_count=provider_batch_requests,
        mock_provider_attempt_count=provider_attempts,
        mock_provider_timeout_count=provider_timeouts,
        mock_provider_rate_limit_wait_count=provider_rate_limit_waits,
        mock_provider_max_batch_size_observed=provider_max_batch_size_observed,
        injected_retryable_failure_count=injected_failures,
        bounded_executed_work_count=bounded_executed,
        eos_execute_work_count=outcome.newly_executed_work_count,
        eos_recovery_used=eos_recovery_used,
        eligible_to_terminal_sample_count=len(latencies_ns),
        eligible_to_terminal_p50_ns=_percentile(latencies_ns, 0.50),
        eligible_to_terminal_p95_ns=_percentile(latencies_ns, 0.95),
        eligible_to_terminal_p99_ns=_percentile(latencies_ns, 0.99),
        eligible_to_terminal_max_ns=max(latencies_ns),
        active_backlog_high_water=high_water,
        active_backlog_before_eos=backlog_before_eos,
        active_backlog_end=backlog_end,
        active_backlog_after_drain_samples=tuple(backlog_after_drain_samples),
        recording_seconds_per_wall_second=(source_duration_ns / wall_ns),
        windows_per_wall_second=(declared * 1_000_000_000 / wall_ns),
    )
    report = LocalStreamingSmokeReport.create(
        manifest=manifest,
        execution_run_id=run_id,
        metrics=metrics,
        recording_finalization_semantic_sha256=(
            outcome.recording_finalization.finalization_semantic_sha256
        ),
        recording_result_semantic_sha256=(
            outcome.recording_result.recording_result_semantic_sha256
        ),
        recording_output_decision=outcome.recording_result.output_decision,
    )
    report_path = _write_content_addressed(
        root / "evidence" / "reports",
        report.report_sha256,
        canonical_json_bytes(report),
    )
    return LocalStreamingSmokeArtifacts(
        manifest=manifest,
        report=report,
        manifest_path=manifest_path,
        report_path=report_path,
        database_path=database_path,
    )


def _build_composition(
    *,
    manifest: LocalStreamingSmokeManifest,
    database_path: Path,
    artifact_root: Path,
    stream_run_id: str,
) -> tuple[DurableStreamWindowScheduler, LocalConformanceStreamFinalizer, str]:
    registry = SchemaRegistry()
    authority = AuthorityBinding(
        authority_id="local-streaming-smoke-authority",
        authority_epoch=1,
        policy_version="local-streaming-smoke-authority-v1",
        initial_binding_semantic_sha256=_digest(manifest, "authority"),
    )
    capture = create_pre_eos_capture_subject(
        schema_ref=_schema_ref(registry, PRE_EOS_CAPTURE_SCHEMA_ID, PRE_EOS_CAPTURE_SCHEMA_VERSION),
        capture_authority_id="local-streaming-smoke-capture",
        capture_authority_epoch=1,
        capture_assignment_policy_version="local-streaming-smoke-capture-v1",
        acquisition_id=f"generated:{manifest.generated_source_sha256}",
        acquisition_epoch=1,
        channel_bindings=tuple(
            ChannelBinding(
                camera_id=camera_id,
                source_channel_id=f"generated-{camera_id.value}",
                source_channel_epoch=1,
                channel_binding_semantic_sha256=_digest(manifest, "channel", camera_id.value),
            )
            for camera_id in CAMERA_IDS
        ),
        mapping_authority=authority,
        clock_authority=authority,
    )
    policy = StreamPolicyBinding(
        version="local-streaming-smoke-policy-v1",
        semantic_sha256=_digest(manifest, "stream-policy"),
    )
    expected = create_expected_window_plan(
        schema_ref=_schema_ref(
            registry, EXPECTED_WINDOW_PLAN_SCHEMA_ID, EXPECTED_WINDOW_PLAN_SCHEMA_VERSION
        ),
        capture_scope_digest=capture.capture_scope_digest,
        segmentation_policy=policy,
        window_policy=policy,
        watermark_policy=policy,
        lateness_policy=policy,
        idle_source_policy=policy,
        planner_version="local-streaming-smoke-planner-v1",
    )
    execution = SQLiteWorkScheduler(
        database_path,
        synchronous=manifest.config.sqlite_synchronous,
    )
    scheduler = DurableStreamWindowScheduler(
        database_path=database_path,
        execution_scheduler=execution,
        expected_plan=expected,
        source_subject=capture.reference(),
        stream_run_id=stream_run_id,
        schema_refs=StreamSchedulerSchemaRefs(
            incremental_window=_schema_ref(
                registry, INCREMENTAL_WINDOW_SCHEMA_ID, INCREMENTAL_WINDOW_SCHEMA_VERSION
            ),
            expected_declaration=_schema_ref(
                registry,
                EXPECTED_WINDOW_DECLARATION_SCHEMA_ID,
                EXPECTED_WINDOW_DECLARATION_SCHEMA_VERSION,
            ),
            expected_plan_seal=_schema_ref(
                registry,
                EXPECTED_WINDOW_SEAL_SCHEMA_ID,
                EXPECTED_WINDOW_SEAL_SCHEMA_VERSION,
            ),
            stream_work_plan=_schema_ref(
                registry, STREAM_WORK_PLAN_SCHEMA_ID, STREAM_WORK_PLAN_SCHEMA_VERSION
            ),
            terminal_member=_schema_ref(
                registry,
                WINDOW_TERMINAL_MEMBER_SCHEMA_ID,
                WINDOW_TERMINAL_MEMBER_SCHEMA_VERSION,
            ),
            terminal_closure=_schema_ref(
                registry,
                WINDOW_TERMINAL_CLOSURE_SCHEMA_ID,
                WINDOW_TERMINAL_CLOSURE_SCHEMA_VERSION,
            ),
        ),
        dag_config_semantic_sha256=_digest(manifest, "stream-dag"),
    )
    finalizer = LocalConformanceStreamFinalizer(
        scheduler=scheduler,
        delivery_authority=SQLiteStreamDeliveryAuthority(
            execution,
            retry_policy=OutboxRetryPolicy(
                version="local-streaming-smoke-outbox-v1",
                max_attempts=3,
                base_delay_seconds=1,
                max_delay_seconds=4,
            ),
        ),
        artifact_root=artifact_root,
        schema_refs=LocalStreamFinalizationSchemaRefs(
            local_work_receipt=_schema_ref(
                registry,
                LOCAL_STREAM_WORK_RECEIPT_SCHEMA_ID,
                LOCAL_STREAM_WORK_RECEIPT_SCHEMA_VERSION,
            ),
            stream_window_result=_schema_ref(
                registry, STREAM_WINDOW_RESULT_SCHEMA_ID, STREAM_WINDOW_RESULT_SCHEMA_VERSION
            ),
            recording_finalization=_schema_ref(
                registry,
                RECORDING_FINALIZATION_SCHEMA_ID,
                RECORDING_FINALIZATION_SCHEMA_VERSION,
            ),
            stream_recording_result=_schema_ref(
                registry,
                LOCAL_STREAM_RECORDING_RESULT_SCHEMA_ID,
                LOCAL_STREAM_RECORDING_RESULT_V4_SCHEMA_VERSION,
            ),
            window_inference_plan=_schema_ref(
                registry,
                LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_ID,
                LOCAL_STREAM_WINDOW_INFERENCE_PLAN_SCHEMA_VERSION,
            ),
            window_semantic_evidence_v2=_schema_ref(
                registry,
                LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_ID,
                LOCAL_STREAM_WINDOW_SEMANTIC_EVIDENCE_V2_SCHEMA_VERSION,
            ),
            stream_inference_identity=_schema_ref(
                registry,
                STREAM_INFERENCE_SCHEMA_ID,
                STREAM_INFERENCE_SCHEMA_VERSION,
            ),
            stream_inference_attempt=_schema_ref(
                registry,
                STREAM_INFERENCE_ATTEMPT_SCHEMA_ID,
                STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION,
            ),
            stream_inference_intent=_schema_ref(
                registry,
                STREAM_INFERENCE_INTENT_SCHEMA_ID,
                STREAM_INFERENCE_INTENT_SCHEMA_VERSION,
            ),
            stream_accepted_call=_schema_ref(
                registry,
                STREAM_ACCEPTED_CALL_SCHEMA_ID,
                STREAM_ACCEPTED_CALL_SCHEMA_VERSION,
            ),
            stream_inference_terminal=_schema_ref(
                registry,
                STREAM_INFERENCE_TERMINAL_SCHEMA_ID,
                STREAM_INFERENCE_TERMINAL_SCHEMA_VERSION,
            ),
        ),
        final_recording=FinalRecordingFacts(
            final_source_subject_type="GENERATED_EVENT_TIME_STREAM",
            final_source_subject_id=str(uuid5(_SMOKE_NAMESPACE, manifest.generated_source_sha256)),
            final_source_exact_sha256=manifest.generated_source_sha256,
            final_recording_identity=_digest(manifest, "recording-identity"),
            final_duration_ns=manifest.config.source_duration_ms * 1_000_000,
        ),
        window_purpose=StreamPurpose.EVENT_PROPOSAL,
        mock_executor_policy_version="actual-local-mock-stream-executor-v1",
        recover_graph_before_execute=False,
    )
    return scheduler, finalizer, capture.capture_scope_digest


def _window(
    manifest: LocalStreamingSmokeManifest,
    ordinal: int,
    *,
    capture_scope_digest: str,
) -> BoundedWindowPlan:
    start_ms = ordinal * manifest.config.window_hop_ms
    end_ms = min(
        start_ms + manifest.config.window_duration_ms,
        manifest.config.source_duration_ms,
    )
    interval = NanosecondInterval(
        start_ns=start_ms * 1_000_000,
        end_ns=end_ms * 1_000_000,
    )
    return BoundedWindowPlan(
        ordinal=ordinal,
        requested_interval=interval,
        effective_interval=interval,
        camera_plans=tuple(
            CameraWindowPlan(
                camera_id=camera_id,
                members=(
                    WindowMember(
                        camera_id=camera_id,
                        interval=interval,
                        absence_reason=CameraAbsenceReason.ABSENT,
                        absence_evidence_sha256=_digest(
                            manifest,
                            "absence",
                            camera_id.value,
                            str(ordinal),
                        ),
                    ),
                ),
            )
            for camera_id in CAMERA_IDS
        ),
        quality_targets=(),
        quality_gaps=(),
        watermark_ns=end_ms * 1_000_000,
        closure_reason=WindowClosureReason.WATERMARK,
        capture_scope_digest=capture_scope_digest,
        mapping_semantic_sha256=_digest(manifest, "mapping"),
        clock_or_alignment_semantic_sha256=_digest(manifest, "clock-alignment"),
        window_policy_version="local-streaming-smoke-window-v1",
        quality_policy_version="local-streaming-smoke-quality-v1",
        purpose=StreamPurpose.EVENT_PROPOSAL,
    )


def _close_eos(
    scheduler: DurableStreamWindowScheduler,
    manifest: LocalStreamingSmokeManifest,
) -> None:
    scheduler.seal(
        PlannerFinish(
            closed_segments=(),
            quality_targets=(),
            windows=(),
            facts=tuple(
                CameraStreamFacts(
                    camera_id=camera_id,
                    packet_count=0,
                    payload_bytes=0,
                    first_timestamp_ns=None,
                    last_timestamp_ns=None,
                    first_sequence=None,
                    last_sequence=None,
                    sequence_gap_count=0,
                )
                for camera_id in CAMERA_IDS
            ),
        )
    )
    scheduler.finalize_eos(
        EosSealInputs(
            eos_source_receipt_semantic_sha256=_digest(manifest, "eos-source-receipt"),
            final_source_timeline_semantic_sha256=_digest(manifest, "final-timeline"),
            final_duration_ns=manifest.config.source_duration_ms * 1_000_000,
            ordered_six_channel_health_closure_sha256=_digest(manifest, "channel-health-closure"),
            mapping_closure_semantic_sha256=_digest(manifest, "mapping-closure"),
            clock_or_alignment_closure_semantic_sha256=_digest(manifest, "clock-alignment-closure"),
        )
    )
    scheduler.mark_export_barrier_complete(
        export_manifest_semantic_sha256=_digest(manifest, "export-manifest"),
        completed_member_count=len(CAMERA_IDS),
    )


def _schema_ref(registry: SchemaRegistry, schema_id: str, version: str) -> SchemaRef:
    return registry.resolve_version(schema_id, version).ref


def _window_count(config: LocalStreamingSmokeConfig) -> int:
    return math.ceil(config.source_duration_ms / config.window_hop_ms)


def _mock_provider_batches(
    scope: tuple[StreamDrainWorkSnapshot, ...],
    *,
    max_batch_size: int,
) -> tuple[tuple[StreamDrainWorkSnapshot, ...], ...]:
    batches: list[tuple[StreamDrainWorkSnapshot, ...]] = []
    for stage in _MOCK_PROVIDER_STAGE_ORDER:
        stage_items = tuple(
            item for item in scope if not item.is_terminal and item.plan.stage is stage
        )
        batches.extend(
            stage_items[start : start + max_batch_size]
            for start in range(0, len(stage_items), max_batch_size)
        )
    return tuple(batches)


def _failure_draw(*, seed: int, ordinal: int) -> int:
    raw = hashlib.sha256(f"{seed}:{ordinal}".encode("ascii")).digest()
    return int.from_bytes(raw[:8], "big") % 1_000_000


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _backlog_did_not_grow(samples: tuple[int, ...]) -> bool:
    return all(current <= previous for previous, current in pairwise(samples))


def _digest(manifest: LocalStreamingSmokeManifest, *parts: str) -> str:
    return semantic_sha256(
        {
            "manifest_sha256": manifest.manifest_sha256,
            "parts": list(parts),
        }
    )


def _write_content_addressed(root: Path, digest: str, payload: bytes) -> Path:
    path = root / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if _content_addressed_bytes_match(path, payload):
        return path

    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary)
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            try:
                if not _content_addressed_bytes_match(path, payload):
                    raise RuntimeError(
                        "content-addressed smoke artifact disappeared during publication"
                    )
            except RuntimeError as conflict:
                raise conflict from error
        if not _content_addressed_bytes_match(path, payload):
            raise RuntimeError("content-addressed smoke artifact disappeared after publication")
        _fsync_directory(path.parent)
        return path
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _content_addressed_bytes_match(path: Path, payload: bytes) -> bool:
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError("content-addressed smoke artifact cannot be verified") from error
    if existing != payload:
        raise RuntimeError("content-addressed smoke artifact conflicts with existing bytes")
    return True


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git_output(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _physical_memory_bytes() -> int:
    proc_meminfo = Path("/proc/meminfo")
    if proc_meminfo.is_file():
        for line in proc_meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1_024
    if os.name == "nt":
        status = _WindowsMemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel32.GlobalMemoryStatusEx
        query.argtypes = [ctypes.POINTER(_WindowsMemoryStatus)]
        query.restype = ctypes.c_int
        if query(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    raise RuntimeError("cannot determine physical memory without an optional dependency")


__all__ = [
    "DEFAULT_SOURCE_DURATION_MS",
    "LOCAL_STREAMING_SMOKE_MANIFEST_VERSION",
    "LOCAL_STREAMING_SMOKE_REPORT_VERSION",
    "WP6_INCREMENTAL_P95_TARGET_MS",
    "WP6_INCREMENTAL_P99_TARGET_MS",
    "ActualLocalMockOfferedLoadProjection",
    "LocalStreamingSmokeArtifacts",
    "LocalStreamingSmokeConfig",
    "LocalStreamingSmokeManifest",
    "LocalStreamingSmokeMetrics",
    "LocalStreamingSmokeReport",
    "collect_local_host_runtime_pins",
    "create_manifest_from_repository",
    "local_streaming_smoke_manifest_projection",
    "local_streaming_smoke_report_projection",
    "run_local_streaming_smoke",
]
