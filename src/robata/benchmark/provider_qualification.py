"""Two-H100 provider saturation qualification artifacts.

These artifacts bind real-provider response observations, capacity denominators, and
external GPU telemetry without turning a local fixture or report into a production
eligibility claim.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from enum import StrEnum
from threading import RLock
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.inference.adapter import (
    ProviderQualificationRequestContract,
    ProviderQualificationSession,
)
from robata.inference.runpod import (
    RunPodDeploymentConfiguration,
    RunPodEndpointConfig,
    RunPodRetryPolicy,
)
from robata.runtime.capacity import (
    CapacityEvidenceClass,
    MeasuredCapacityInput,
    MeasuredCapacityReport,
    MeasuredCapacityStatus,
    ProviderMode,
    build_measured_capacity_report,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class TwoH100Topology(StrEnum):
    TWO_SINGLE_CARD_REPLICAS = "TWO_SINGLE_CARD_REPLICAS"
    TWO_CARD_TENSOR_PARALLEL = "TWO_CARD_TENSOR_PARALLEL"


def _nearest_rank(values: tuple[int, ...], quantile: float) -> int:
    if not values:
        raise ValueError("latency samples must be nonempty")
    ordered = tuple(sorted(values))
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _parse_rfc3339_timestamp(value: str, *, field_name: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an RFC3339 timezone")
    return parsed


class ProviderLatencyPercentiles(StrictModel):
    """Nearest-rank latency summary returned by one measured population."""

    count: PositiveInt
    p50_ms: NonNegativeInt
    p95_ms: NonNegativeInt
    p99_ms: NonNegativeInt

    @classmethod
    def from_samples(cls, samples_ms: tuple[int, ...]) -> Self:
        """Calculate P50/P95/P99 with the explicit nearest-rank convention."""

        if not isinstance(samples_ms, tuple) or not samples_ms:
            raise ValueError("latency samples must be a nonempty tuple")
        if any(type(value) is not int or value < 0 for value in samples_ms):
            raise ValueError("latency samples must be nonnegative integers")
        return cls(
            count=len(samples_ms),
            p50_ms=_nearest_rank(samples_ms, 0.50),
            p95_ms=_nearest_rank(samples_ms, 0.95),
            p99_ms=_nearest_rank(samples_ms, 0.99),
        )

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if not self.p50_ms <= self.p95_ms <= self.p99_ms:
            raise ValueError("latency percentiles must be ordered p50 <= p95 <= p99")
        return self


class TwoH100ProviderConfiguration(StrictModel):
    """Pinned, non-secret configuration for one real-provider matrix."""

    workload_manifest_digest: Sha256Digest
    provider: NonEmptyString
    model_identifier: NonEmptyString
    model_version: NonEmptyString
    request_contracts: tuple[ProviderQualificationRequestContract, ...] = Field(min_length=1)
    inference_engine: NonEmptyString
    precision_or_quantization: NonEmptyString
    topology: TwoH100Topology
    gpu_count: Literal[2] = 2
    max_images_per_request: PositiveInt
    max_input_tokens: PositiveInt
    max_output_tokens: PositiveInt
    native_batch_max_size: PositiveInt
    max_concurrent_requests: PositiveInt
    endpoint_configuration: RunPodEndpointConfig
    retry_policy: RunPodRetryPolicy
    supported_topologies: tuple[TwoH100Topology, ...] = ()
    configuration_digest: Sha256Digest

    @classmethod
    def create(cls, **values: object) -> Self:
        if "configuration_digest" in values:
            raise ValueError("configuration_digest is derived")
        draft = cls.model_construct(**{**values, "configuration_digest": "0" * 64})
        digest = exact_bytes_sha256(
            canonical_json_bytes(draft.model_dump(mode="json", exclude={"configuration_digest"}))
        )
        return cls.model_validate(
            {**draft.model_dump(mode="python"), "configuration_digest": digest}
        )

    def validate_runpod_configuration(
        self,
        *,
        endpoint_config: object,
        capabilities: object,
        retry_policy: object,
    ) -> None:
        """Reject a report configuration that does not match the active adapter."""

        from robata.inference.models import ModelCapabilities

        if not isinstance(endpoint_config, RunPodEndpointConfig):
            raise TypeError("endpoint_config must be a RunPodEndpointConfig")
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be a ModelCapabilities")
        if not isinstance(retry_policy, RunPodRetryPolicy):
            raise TypeError("retry_policy must be a RunPodRetryPolicy")
        if endpoint_config != self.endpoint_configuration:
            raise ValueError(
                "RunPod endpoint configuration does not match qualification configuration"
            )
        if retry_policy != self.retry_policy:
            raise ValueError("RunPod retry policy does not match qualification configuration")
        if endpoint_config.provider != self.provider:
            raise ValueError("RunPod endpoint provider does not match qualification configuration")
        deployment = endpoint_config.deployment_configuration
        if deployment is None:
            raise ValueError("RunPod endpoint lacks a pinned deployment configuration")
        if not isinstance(deployment, RunPodDeploymentConfiguration):
            raise TypeError("RunPod deployment configuration has an invalid type")
        if (
            deployment.model_identifier != self.model_identifier
            or deployment.model_version != self.model_version
            or deployment.inference_engine != self.inference_engine
            or deployment.precision_or_quantization != self.precision_or_quantization
            or deployment.topology != self.topology.value
            or deployment.max_output_tokens != self.max_output_tokens
            or deployment.supported_topologies
            != tuple(topology.value for topology in self.supported_topologies)
        ):
            raise ValueError("RunPod deployment does not match qualification configuration")
        if (
            capabilities.provider != self.provider
            or capabilities.model_name != self.model_identifier
            or capabilities.model_version != self.model_version
        ):
            raise ValueError("RunPod capabilities do not match qualification model pin")
        if endpoint_config.max_concurrent_requests != self.max_concurrent_requests:
            raise ValueError("RunPod concurrency does not match qualification configuration")
        if endpoint_config.native_batch_max_size != self.native_batch_max_size:
            raise ValueError("RunPod batch limit does not match qualification configuration")
        if capabilities.max_images_per_request != self.max_images_per_request:
            raise ValueError("RunPod image limit does not match qualification configuration")
        if capabilities.max_input_tokens != self.max_input_tokens:
            raise ValueError("RunPod input-token limit does not match qualification configuration")

    @model_validator(mode="after")
    def validate_digest(self) -> Self:
        if len(set(self.supported_topologies)) != len(self.supported_topologies):
            raise ValueError("supported_topologies must be unique")
        if self.supported_topologies and self.topology not in self.supported_topologies:
            raise ValueError("supported_topologies must include the deployed topology")
        request_contract_identities = tuple(
            (
                contract.task,
                contract.prompt_artifact_id,
                contract.prompt_version,
                contract.prompt_sha256,
                contract.output_schema_sha256,
                contract.max_input_tokens,
                contract.timeout_ms,
                contract.generation_config_sha256,
            )
            for contract in self.request_contracts
        )
        if len(set(request_contract_identities)) != len(request_contract_identities):
            raise ValueError("qualification request contracts must be unique")
        if any(
            contract.max_input_tokens > self.max_input_tokens
            for contract in self.request_contracts
        ):
            raise ValueError("qualification prompt context exceeds the configured input limit")
        if any(
            contract.timeout_ms > self.endpoint_configuration.request_timeout_cap_ms
            for contract in self.request_contracts
        ):
            raise ValueError("qualification request timeout exceeds the endpoint timeout cap")
        endpoint = self.endpoint_configuration
        deployment = endpoint.deployment_configuration
        if (
            endpoint.provider != self.provider
            or endpoint.native_batch_max_size != self.native_batch_max_size
            or endpoint.max_concurrent_requests != self.max_concurrent_requests
            or deployment is None
            or deployment.model_identifier != self.model_identifier
            or deployment.model_version != self.model_version
            or deployment.inference_engine != self.inference_engine
            or deployment.precision_or_quantization != self.precision_or_quantization
            or deployment.topology != self.topology.value
            or deployment.max_output_tokens != self.max_output_tokens
            or deployment.supported_topologies
            != tuple(topology.value for topology in self.supported_topologies)
        ):
            raise ValueError("embedded RunPod endpoint does not match qualification configuration")
        expected = exact_bytes_sha256(
            canonical_json_bytes(self.model_dump(mode="json", exclude={"configuration_digest"}))
        )
        if self.configuration_digest != expected:
            raise ValueError("configuration_digest does not match configuration")
        return self

class ProviderGpuMeasurement(StrictModel):
    """Externally exported H100 telemetry tied to one qualification session."""

    qualification_session_id: OpaqueUuid
    hardware_inventory_artifact_uri: NonEmptyString
    hardware_inventory_sha256: Sha256Digest
    telemetry_artifact_uri: NonEmptyString
    telemetry_artifact_sha256: Sha256Digest
    gpu_sku: NonEmptyString
    driver_version: NonEmptyString
    runtime_version: NonEmptyString
    metric_source: NonEmptyString
    measurement_started_at: Rfc3339Timestamp
    measurement_completed_at: Rfc3339Timestamp
    gpu_count: Literal[2] = 2
    aggregate_gpu_seconds: NonNegativeFloat
    gpu_utilization_fraction: UnitInterval
    gpu_memory_bytes: PositiveInt
    kv_cache_utilization_fraction: UnitInterval
    oom_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_measurement_window_and_hardware(self) -> Self:
        started = _parse_rfc3339_timestamp(
            self.measurement_started_at,
            field_name="measurement_started_at",
        )
        completed = _parse_rfc3339_timestamp(
            self.measurement_completed_at,
            field_name="measurement_completed_at",
        )
        if completed <= started:
            raise ValueError("GPU measurement window must be nonempty")
        if "H100" not in self.gpu_sku.upper():
            raise ValueError("two-H100 qualification requires an H100 GPU SKU")
        maximum_gpu_seconds = (completed - started).total_seconds() * self.gpu_count
        if self.aggregate_gpu_seconds > maximum_gpu_seconds + 1e-9:
            raise ValueError("aggregate GPU seconds exceed the measured two-GPU window")
        return self


class ProviderTimingSample(StrictModel):
    """One terminal request-bound RunPod observation in a qualification session."""

    request_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    input_plan_part_ordinal: NonNegativeInt
    provider_image_count: NonNegativeInt
    input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None
    provider_queue_ms: NonNegativeInt | None
    provider_execution_ms: NonNegativeInt | None
    time_to_first_token_ms: NonNegativeInt | None
    end_to_end_ms: NonNegativeInt
    input_tokens_known: bool
    output_tokens_known: bool
    accepted: bool

    @model_validator(mode="after")
    def validate_token_observation(self) -> Self:
        if self.input_tokens_known != (self.input_tokens is not None):
            raise ValueError("input token value must match input_tokens_known")
        if self.output_tokens_known != (self.output_tokens is not None):
            raise ValueError("output token value must match output_tokens_known")
        return self


def _final_provider_samples(
    samples: tuple[ProviderTimingSample, ...],
) -> tuple[ProviderTimingSample, ...]:
    """Collapse canonical retry attempts to one final terminal sample per call part."""

    if len({sample.request_id for sample in samples}) != len(samples):
        raise ValueError("provider timing samples duplicate a request observation")
    attempts_by_part: dict[tuple[str, int], list[ProviderTimingSample]] = {}
    for sample in samples:
        attempts_by_part.setdefault(
            (sample.logical_invocation_id, sample.input_plan_part_ordinal),
            [],
        ).append(sample)
    finals: list[ProviderTimingSample] = []
    for attempts in attempts_by_part.values():
        if any(sample.accepted for sample in attempts[:-1]):
            raise ValueError(
                "provider timing samples contain a nonfinal or duplicate accepted logical call part"
            )
        finals.append(attempts[-1])
    return tuple(finals)


class ProviderAdapterTerminalWorkload(StrictModel):
    """Actual provider attempts, separate from canonical planned-work counters."""

    provider_images: NonNegativeInt
    logical_calls: NonNegativeInt
    call_parts: NonNegativeInt
    provider_attempt_count: NonNegativeInt
    input_tokens: NonNegativeInt
    input_token_responses: NonNegativeInt
    output_tokens: NonNegativeInt
    output_token_responses: NonNegativeInt
    http_requests: NonNegativeInt

    @model_validator(mode="after")
    def validate_workload_totals(self) -> Self:
        if self.logical_calls > self.call_parts:
            raise ValueError("logical calls cannot exceed observed call parts")
        if self.call_parts > self.provider_attempt_count:
            raise ValueError("observed call parts cannot exceed provider attempts")
        if self.input_token_responses > self.provider_attempt_count:
            raise ValueError("known input-token responses cannot exceed provider attempts")
        if self.output_token_responses > self.provider_attempt_count:
            raise ValueError("known output-token responses cannot exceed provider attempts")
        return self


class ProviderRuntimeTelemetry(StrictModel):
    """Final logical outcomes, actual attempts, timing, and GPU telemetry."""

    accepted_response_count: NonNegativeInt
    rejected_response_count: NonNegativeInt
    input_known_response_count: NonNegativeInt
    usage_known_response_count: NonNegativeInt
    input_known_attempt_count: NonNegativeInt
    usage_known_attempt_count: NonNegativeInt
    canonical_retry_attempt_count: NonNegativeInt
    adapter_terminal_workload: ProviderAdapterTerminalWorkload
    provider_queue: ProviderLatencyPercentiles | None
    provider_execution: ProviderLatencyPercentiles | None
    time_to_first_token: ProviderLatencyPercentiles | None
    end_to_end: ProviderLatencyPercentiles
    gpu: ProviderGpuMeasurement
    adapter_transport_retry_count: NonNegativeInt

    @classmethod
    def from_samples(
        cls,
        *,
        samples: tuple[ProviderTimingSample, ...],
        http_requests: int,
        gpu: ProviderGpuMeasurement,
        adapter_transport_retry_count: int,
    ) -> Self:
        """Build final-call latency plus exact provider-attempt work observations."""

        if not isinstance(samples, tuple) or not samples:
            raise ValueError("provider timing samples must be a nonempty tuple")
        if any(not isinstance(sample, ProviderTimingSample) for sample in samples):
            raise TypeError("provider timing samples must contain ProviderTimingSample values")
        if not isinstance(gpu, ProviderGpuMeasurement):
            raise TypeError("gpu must be a ProviderGpuMeasurement")
        final_samples = _final_provider_samples(samples)
        logical_call_ids = {sample.logical_invocation_id for sample in final_samples}
        accepted_samples = tuple(sample for sample in final_samples if sample.accepted)
        provider_queue_ms = tuple(
            sample.provider_queue_ms
            for sample in accepted_samples
            if sample.provider_queue_ms is not None
        )
        provider_execution_ms = tuple(
            sample.provider_execution_ms
            for sample in accepted_samples
            if sample.provider_execution_ms is not None
        )
        time_to_first_token_ms = tuple(
            sample.time_to_first_token_ms
            for sample in accepted_samples
            if sample.time_to_first_token_ms is not None
        )
        input_known_response_count = sum(
            sample.input_tokens_known for sample in final_samples
        )
        usage_known_response_count = sum(
            sample.output_tokens_known for sample in final_samples
        )
        input_known_attempt_count = sum(sample.input_tokens_known for sample in samples)
        usage_known_attempt_count = sum(sample.output_tokens_known for sample in samples)
        return cls(
            accepted_response_count=len(accepted_samples),
            rejected_response_count=len(final_samples) - len(accepted_samples),
            input_known_response_count=input_known_response_count,
            usage_known_response_count=usage_known_response_count,
            input_known_attempt_count=input_known_attempt_count,
            usage_known_attempt_count=usage_known_attempt_count,
            canonical_retry_attempt_count=len(samples) - len(final_samples),
            adapter_terminal_workload=ProviderAdapterTerminalWorkload(
                provider_images=sum(sample.provider_image_count for sample in samples),
                logical_calls=len(logical_call_ids),
                call_parts=len(final_samples),
                provider_attempt_count=len(samples),
                input_tokens=sum(sample.input_tokens or 0 for sample in samples),
                input_token_responses=input_known_attempt_count,
                output_tokens=sum(sample.output_tokens or 0 for sample in samples),
                output_token_responses=usage_known_attempt_count,
                http_requests=http_requests,
            ),
            provider_queue=(
                ProviderLatencyPercentiles.from_samples(provider_queue_ms)
                if provider_queue_ms
                else None
            ),
            provider_execution=(
                ProviderLatencyPercentiles.from_samples(provider_execution_ms)
                if provider_execution_ms
                else None
            ),
            time_to_first_token=(
                ProviderLatencyPercentiles.from_samples(time_to_first_token_ms)
                if time_to_first_token_ms
                else None
            ),
            end_to_end=ProviderLatencyPercentiles.from_samples(
                tuple(sample.end_to_end_ms for sample in final_samples)
            ),
            gpu=gpu,
            adapter_transport_retry_count=adapter_transport_retry_count,
        )

    @property
    def terminal_response_count(self) -> int:
        return self.accepted_response_count + self.rejected_response_count

    @model_validator(mode="after")
    def validate_terminal_populations(self) -> Self:
        if self.terminal_response_count <= 0:
            raise ValueError("provider telemetry requires a terminal response observation")
        attempts = self.adapter_terminal_workload.provider_attempt_count
        if self.terminal_response_count > attempts:
            raise ValueError("terminal provider responses cannot exceed provider attempts")
        if self.canonical_retry_attempt_count != attempts - self.terminal_response_count:
            raise ValueError("canonical retry attempts do not match provider attempt population")
        if self.input_known_response_count > self.terminal_response_count:
            raise ValueError("known input-token responses exceed terminal provider responses")
        if self.usage_known_response_count > self.terminal_response_count:
            raise ValueError("known output-token responses exceed terminal provider responses")
        if self.input_known_attempt_count > attempts:
            raise ValueError("known input-token attempts exceed provider attempts")
        if self.usage_known_attempt_count > attempts:
            raise ValueError("known output-token attempts exceed provider attempts")
        if (
            self.adapter_terminal_workload.input_token_responses
            != self.input_known_attempt_count
        ):
            raise ValueError("workload input-token attempts must match provider observations")
        if (
            self.adapter_terminal_workload.output_token_responses
            != self.usage_known_attempt_count
        ):
            raise ValueError("workload output-token attempts must match provider observations")
        if self.adapter_terminal_workload.call_parts != self.terminal_response_count:
            raise ValueError("workload call parts must match terminal provider responses")
        for name, latency in (
            ("provider_queue", self.provider_queue),
            ("provider_execution", self.provider_execution),
            ("time_to_first_token", self.time_to_first_token),
        ):
            if self.accepted_response_count == 0:
                if latency is not None:
                    raise ValueError(f"{name} requires an accepted provider response")
            elif latency is None or latency.count != self.accepted_response_count:
                raise ValueError(f"{name} samples must match accepted provider responses")
        if self.end_to_end.count != self.terminal_response_count:
            raise ValueError("end-to-end samples must match terminal provider responses")
        return self

class ProviderSaturationPoint(StrictModel):
    """One sealed session at one offered-concurrency point of the adaptive workload."""

    configuration_digest: Sha256Digest
    qualification_session: ProviderQualificationSession
    run_namespace: NonEmptyString
    offered_concurrency: PositiveInt
    capacity: MeasuredCapacityReport
    telemetry: ProviderRuntimeTelemetry

    @model_validator(mode="after")
    def validate_point(self) -> Self:
        if self.configuration_digest != self.qualification_session.configuration_digest:
            raise ValueError("saturation point configuration does not match qualification session")
        if self.run_namespace != self.qualification_session.run_namespace:
            raise ValueError("saturation point namespace does not match qualification session")
        if self.capacity.measurement_status is not MeasuredCapacityStatus.AVAILABLE:
            raise ValueError("saturation points require denominator-safe capacity")
        if self.capacity.execution_mode != "FRESH":
            raise ValueError("saturation points require fresh provider execution")
        if self.capacity.provider_mode is not ProviderMode.NETWORK_PROVIDER:
            raise ValueError("saturation points require NETWORK_PROVIDER measurements")
        if (
            self.capacity.workload_fingerprint
            != self.qualification_session.workload_manifest_digest
        ):
            raise ValueError(
                "saturation point capacity does not match qualification session workload"
            )
        if self.capacity.recording_hours is None or self.capacity.recording_hours <= 0:
            raise ValueError("saturation points require recording-hour denominators")
        required_counts = (
            self.capacity.provider_images,
            self.capacity.logical_calls,
            self.capacity.http_requests,
            self.capacity.input_tokens,
            self.capacity.output_tokens,
            self.capacity.output_token_responses,
        )
        if any(value is None for value in required_counts):
            raise ValueError("saturation points require image, call, request, and token counts")
        if (
            self.capacity.provider_images <= 0
            or self.capacity.logical_calls <= 0
            or self.capacity.http_requests <= 0
            or self.capacity.input_tokens <= 0
        ):
            raise ValueError("saturation points require observed provider work")
        observed = self.telemetry.adapter_terminal_workload
        if (
            self.capacity.output_tokens != observed.output_tokens
            or self.capacity.output_token_responses != observed.output_token_responses
        ):
            raise ValueError("capacity output-token totals do not match adapter observations")
        if (
            self.telemetry.usage_known_attempt_count
            != self.capacity.output_token_responses
        ):
            raise ValueError(
                "known output-token attempts must match the capacity observation"
            )
        if self.telemetry.gpu.qualification_session_id != self.qualification_session.session_id:
            raise ValueError("GPU telemetry does not match the qualification session")
        if self.telemetry.gpu.aggregate_gpu_seconds <= 0:
            raise ValueError("saturation points require observed GPU time")
        return self

    @property
    def aggregate_gpu_minutes_per_recording_hour(self) -> float:
        assert self.capacity.recording_hours is not None
        return self.telemetry.gpu.aggregate_gpu_seconds / 60 / self.capacity.recording_hours

    def _observed_rate_per_second(self, amount: int) -> float:
        return amount * 1_000_000_000 / self.capacity.wall_time_ns

    @property
    def adapter_provider_images_per_second(self) -> float:
        return self._observed_rate_per_second(
            self.telemetry.adapter_terminal_workload.provider_images
        )

    @property
    def adapter_logical_calls_per_second(self) -> float:
        return self._observed_rate_per_second(
            self.telemetry.adapter_terminal_workload.logical_calls
        )

    @property
    def adapter_http_requests_per_second(self) -> float:
        return self._observed_rate_per_second(
            self.telemetry.adapter_terminal_workload.http_requests
        )

    @property
    def adapter_input_tokens_per_second(self) -> float | None:
        workload = self.telemetry.adapter_terminal_workload
        if workload.input_token_responses != workload.provider_attempt_count:
            return None
        return self._observed_rate_per_second(workload.input_tokens)

    @property
    def adapter_output_tokens_per_second(self) -> float | None:
        workload = self.telemetry.adapter_terminal_workload
        if workload.output_token_responses != workload.provider_attempt_count:
            return None
        return self._observed_rate_per_second(workload.output_tokens)

    @property
    def safe_envelope(self) -> bool:
        """Derive safety from final logical outcomes and GPU OOM telemetry."""

        return (
            self.telemetry.terminal_response_count > 0
            and self.telemetry.rejected_response_count == 0
            and self.telemetry.input_known_response_count
            == self.telemetry.terminal_response_count
            and self.telemetry.usage_known_response_count
            == self.telemetry.terminal_response_count
            and self.telemetry.gpu.oom_count == 0
        )

class ProviderQualificationCollector:
    """One-shot collector for one scoped provider saturation-point execution."""

    def __init__(
        self,
        *,
        qualification_session: ProviderQualificationSession,
        run_namespace: str,
        offered_concurrency: int,
    ) -> None:
        if not isinstance(qualification_session, ProviderQualificationSession):
            raise TypeError("qualification_session must be a ProviderQualificationSession")
        if type(run_namespace) is not str or not run_namespace:
            raise ValueError("run_namespace must be a nonempty string")
        if run_namespace != qualification_session.run_namespace:
            raise ValueError("run_namespace must match the qualification session")
        if type(offered_concurrency) is not int or offered_concurrency <= 0:
            raise ValueError("offered_concurrency must be a positive integer")
        self._qualification_session = qualification_session
        self._run_namespace = run_namespace
        self._offered_concurrency = offered_concurrency
        self._lock = RLock()
        self._timing_samples: list[ProviderTimingSample] = []
        self._request_ids: set[str] = set()
        self._retry_count = 0
        self._http_request_count = 0
        self._sealed = False

    @property
    def qualification_session(self) -> ProviderQualificationSession:
        return self._qualification_session

    @property
    def run_namespace(self) -> str:
        return self._run_namespace

    @property
    def offered_concurrency(self) -> int:
        return self._offered_concurrency

    def record_provider_timing(
        self,
        *,
        qualification_session: ProviderQualificationSession,
        request_id: str,
        logical_invocation_id: str,
        input_plan_part_ordinal: int,
        provider_image_count: int,
        input_tokens: int | None,
        output_tokens: int | None,
        provider_queue_ms: int | None,
        provider_execution_ms: int | None,
        time_to_first_token_ms: int | None,
        end_to_end_ms: int,
        input_tokens_known: bool,
        output_tokens_known: bool,
        accepted: bool,
    ) -> None:
        self._require_matching_session(qualification_session)
        sample = ProviderTimingSample(
            request_id=request_id,
            logical_invocation_id=logical_invocation_id,
            input_plan_part_ordinal=input_plan_part_ordinal,
            provider_image_count=provider_image_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_queue_ms=provider_queue_ms,
            provider_execution_ms=provider_execution_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            end_to_end_ms=end_to_end_ms,
            input_tokens_known=input_tokens_known,
            output_tokens_known=output_tokens_known,
            accepted=accepted,
        )
        with self._lock:
            if self._sealed:
                raise ValueError("qualification collector is sealed")
            if sample.request_id in self._request_ids:
                raise ValueError("qualification collector received a duplicate request observation")
            self._request_ids.add(sample.request_id)
            self._timing_samples.append(sample)

    def record_provider_http_requests(
        self,
        *,
        qualification_session: ProviderQualificationSession,
        count: int,
    ) -> None:
        self._require_matching_session(qualification_session)
        if type(count) is not int or count <= 0:
            raise ValueError("provider HTTP request count must be a positive integer")
        with self._lock:
            if self._sealed:
                raise ValueError("qualification collector is sealed")
            self._http_request_count += count

    def record_provider_retries(
        self,
        *,
        qualification_session: ProviderQualificationSession,
        count: int,
    ) -> None:
        self._require_matching_session(qualification_session)
        if type(count) is not int or count < 0:
            raise ValueError("provider retry count must be a nonnegative integer")
        with self._lock:
            if self._sealed:
                raise ValueError("qualification collector is sealed")
            self._retry_count += count

    @property
    def timing_samples(self) -> tuple[ProviderTimingSample, ...]:
        with self._lock:
            return tuple(self._timing_samples)

    @property
    def timing_sample_count(self) -> int:
        with self._lock:
            return len(self._timing_samples)

    @property
    def adapter_transport_retry_count(self) -> int:
        with self._lock:
            return self._retry_count

    @property
    def http_request_count(self) -> int:
        with self._lock:
            return self._http_request_count

    def build_point(
        self,
        *,
        capacity: MeasuredCapacityReport,
        gpu: ProviderGpuMeasurement,
    ) -> ProviderSaturationPoint:
        """Seal one session after it matches exact provider and capacity observations."""

        if not isinstance(capacity, MeasuredCapacityReport):
            raise TypeError("capacity must be a MeasuredCapacityReport")
        if not isinstance(gpu, ProviderGpuMeasurement):
            raise TypeError("gpu must be a ProviderGpuMeasurement")
        session = self._qualification_session
        if capacity.workload_fingerprint != session.workload_manifest_digest:
            raise ValueError("capacity workload does not match qualification session")
        if gpu.qualification_session_id != session.session_id:
            raise ValueError("GPU telemetry does not match qualification session")
        with self._lock:
            if self._sealed:
                raise ValueError("qualification collector is already sealed")
            samples = tuple(self._timing_samples)
            retry_count = self._retry_count
            http_request_count = self._http_request_count
            expected_known_responses = capacity.output_token_responses
            if expected_known_responses is None:
                raise ValueError("capacity must record output-token responses for qualification")
            if not samples:
                raise ValueError("qualification requires at least one terminal observation")
            telemetry = ProviderRuntimeTelemetry.from_samples(
                samples=samples,
                http_requests=http_request_count,
                gpu=gpu,
                adapter_transport_retry_count=retry_count,
            )
            final_samples = _final_provider_samples(samples)
            accepted_samples = tuple(sample for sample in final_samples if sample.accepted)
            if any(
                value is None
                for sample in accepted_samples
                for value in (
                    sample.provider_queue_ms,
                    sample.provider_execution_ms,
                    sample.time_to_first_token_ms,
                )
            ):
                raise ValueError(
                    "accepted timing observations require queue, execution, and TTFT"
                )
            if telemetry.usage_known_attempt_count != expected_known_responses:
                raise ValueError(
                    "known output-token attempts must match capacity output-token responses"
                )
            point = ProviderSaturationPoint(
                configuration_digest=session.configuration_digest,
                qualification_session=session,
                run_namespace=self._run_namespace,
                offered_concurrency=self._offered_concurrency,
                capacity=capacity,
                telemetry=telemetry,
            )
            self._sealed = True
            return point

    def _require_matching_session(
        self,
        qualification_session: ProviderQualificationSession,
    ) -> None:
        if not isinstance(qualification_session, ProviderQualificationSession):
            raise TypeError("qualification_session must be a ProviderQualificationSession")
        if qualification_session != self._qualification_session:
            raise ValueError("qualification observation belongs to a different session")

class ProviderQualificationRunContext(StrictModel):
    """Explicit session and fresh namespace handed to one P6 workload callback."""

    qualification_session: ProviderQualificationSession
    run_namespace: NonEmptyString

    @model_validator(mode="after")
    def validate_namespace(self) -> Self:
        if self.run_namespace != self.qualification_session.run_namespace:
            raise ValueError("qualification run namespace must match its session")
        return self

    def bind_request_metadata(
        self,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Attach the immutable session/namespace marker without caller overrides."""

        from robata.inference.runpod import (
            RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY,
            RUNPOD_QUALIFICATION_SESSION_METADATA_KEY,
        )

        values = dict(metadata or {})
        expected = {
            RUNPOD_QUALIFICATION_SESSION_METADATA_KEY: self.qualification_session.session_id,
            RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY: self.run_namespace,
        }
        for key, value in expected.items():
            if key in values and values[key] != value:
                raise ValueError("qualification request metadata has a conflicting scope")
            values[key] = value
        return values


async def run_provider_saturation_point(
    *,
    configuration: TwoH100ProviderConfiguration,
    context: ProviderQualificationRunContext,
    offered_concurrency: int,
    workload_manifest_bytes: bytes,
    adapter_factory: Callable[
        [ProviderQualificationCollector, ProviderQualificationRunContext],
        object,
    ],
    workload: Callable[[object, ProviderQualificationRunContext], Awaitable[MeasuredCapacityInput]],
    gpu: ProviderGpuMeasurement,
) -> ProviderSaturationPoint:
    """Run one fresh P6 workload case and seal its measured provider point.

    ``run_namespace`` must map to a fresh P6 invocation/ledger namespace. Run one
    invocation for each immutable P6 sampling-policy workload manifest. The caller
    supplies the actual P6 workload callback; this helper binds its immutable manifest,
    request contracts, adapter session, collector, capacity denominator, and GPU
    telemetry into a single reportable point.
    """

    from robata.inference.runpod import RunPodVisionAdapter

    if not isinstance(configuration, TwoH100ProviderConfiguration):
        raise TypeError("configuration must be a TwoH100ProviderConfiguration")
    if not isinstance(context, ProviderQualificationRunContext):
        raise TypeError("context must be a ProviderQualificationRunContext")
    if type(offered_concurrency) is not int or offered_concurrency <= 0:
        raise ValueError("offered_concurrency must be a positive integer")
    if not isinstance(workload_manifest_bytes, bytes) or not workload_manifest_bytes:
        raise ValueError("workload_manifest_bytes must be nonempty bytes")
    if not callable(adapter_factory) or not callable(workload):
        raise TypeError("adapter_factory and workload must be callable")
    session = context.qualification_session
    if session.configuration_digest != configuration.configuration_digest:
        raise ValueError("qualification session does not match the provider configuration")
    if session.request_contracts != configuration.request_contracts:
        raise ValueError("qualification session does not match the pinned request contracts")
    if exact_bytes_sha256(workload_manifest_bytes) != session.workload_manifest_digest:
        raise ValueError("workload manifest bytes do not match the qualification session")
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace=context.run_namespace,
        offered_concurrency=offered_concurrency,
    )
    adapter = adapter_factory(collector, context)
    if not isinstance(adapter, RunPodVisionAdapter):
        raise TypeError("adapter_factory must return a RunPodVisionAdapter")
    if adapter.qualification_session != session or adapter.qualification_observer is not collector:
        raise ValueError("RunPod adapter is not bound to the qualification session collector")
    configuration.validate_runpod_configuration(
        endpoint_config=adapter.config,
        capabilities=adapter.capabilities_snapshot,
        retry_policy=adapter.retry_policy,
    )
    capacity_input = await workload(adapter, context)
    if not isinstance(capacity_input, MeasuredCapacityInput):
        raise TypeError("workload must return a MeasuredCapacityInput")
    if adapter.qualification_observation_error is not None:
        raise ValueError(
            "RunPod qualification observation failed: "
            f"{adapter.qualification_observation_error}"
        )
    observed_request_ids = set(adapter.qualification_observed_request_ids)
    collector_request_ids = {sample.request_id for sample in collector.timing_samples}
    if not observed_request_ids or collector_request_ids != observed_request_ids:
        raise ValueError(
            "qualification collector observations must come from the scoped RunPod adapter"
        )
    if collector.http_request_count != adapter.qualification_observed_http_request_count:
        raise ValueError(
            "qualification collector HTTP observations do not match the scoped RunPod adapter"
        )
    if (
        collector.adapter_transport_retry_count
        != adapter.qualification_observed_transport_retry_count
    ):
        raise ValueError(
            "qualification collector retry observations do not match the scoped RunPod adapter"
        )
    capacity = build_measured_capacity_report(capacity_input)
    return collector.build_point(capacity=capacity, gpu=gpu)

class TwoH100ProviderQualificationReport(StrictModel):
    """Measured saturation curve for one pinned model/topology configuration."""

    report_version: Literal["two-h100-provider-qualification-v1"] = (
        "two-h100-provider-qualification-v1"
    )
    configuration: TwoH100ProviderConfiguration
    endpoint_config: object
    capabilities: object
    retry_policy: object
    points: tuple[ProviderSaturationPoint, ...] = Field(min_length=2)
    evidence_class: CapacityEvidenceClass
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        self.configuration.validate_runpod_configuration(
            endpoint_config=self.endpoint_config,
            capabilities=self.capabilities,
            retry_policy=self.retry_policy,
        )
        offered = tuple(point.offered_concurrency for point in self.points)
        if offered != tuple(sorted(offered)) or len(set(offered)) != len(offered):
            raise ValueError("saturation points must have unique increasing concurrency")
        if any(
            point.configuration_digest != self.configuration.configuration_digest
            for point in self.points
        ):
            raise ValueError("saturation point configuration does not match report")
        if any(
            point.capacity.workload_fingerprint != self.configuration.workload_manifest_digest
            for point in self.points
        ):
            raise ValueError("saturation point workload does not match configuration")
        if any(
            point.qualification_session.request_contracts
            != self.configuration.request_contracts
            for point in self.points
        ):
            raise ValueError("saturation point prompt contracts do not match configuration")
        session_ids = tuple(point.qualification_session.session_id for point in self.points)
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("saturation points must use distinct sealed qualification sessions")
        run_namespaces = tuple(point.run_namespace for point in self.points)
        if len(set(run_namespaces)) != len(run_namespaces):
            raise ValueError("saturation points must use distinct fresh run namespaces")
        if any(point.capacity.evidence_class is not self.evidence_class for point in self.points):
            raise ValueError("saturation point evidence class does not match report")
        if any(
            point.offered_concurrency > self.configuration.max_concurrent_requests
            for point in self.points
        ):
            raise ValueError("saturation point exceeds configured provider concurrency")
        if any(
            point.telemetry.gpu.gpu_count != self.configuration.gpu_count
            for point in self.points
        ):
            raise ValueError("GPU measurement count does not match qualification configuration")
        hardware_identity = (
            self.points[0].telemetry.gpu.hardware_inventory_artifact_uri,
            self.points[0].telemetry.gpu.hardware_inventory_sha256,
            self.points[0].telemetry.gpu.gpu_sku,
            self.points[0].telemetry.gpu.driver_version,
            self.points[0].telemetry.gpu.runtime_version,
            self.points[0].telemetry.gpu.metric_source,
        )
        if any(
            (
                point.telemetry.gpu.hardware_inventory_artifact_uri,
                point.telemetry.gpu.hardware_inventory_sha256,
                point.telemetry.gpu.gpu_sku,
                point.telemetry.gpu.driver_version,
                point.telemetry.gpu.runtime_version,
                point.telemetry.gpu.metric_source,
            )
            != hardware_identity
            for point in self.points
        ):
            raise ValueError("saturation points must use the same GPU inventory and metric source")
        if any(point.safe_envelope and point.telemetry.gpu.oom_count for point in self.points):
            raise ValueError("a safe saturation point cannot report an OOM")
        if not any(point.safe_envelope for point in self.points):
            raise ValueError("saturation report requires at least one safe envelope point")
        unsafe_boundary_seen = False
        for point in self.points:
            if point.safe_envelope:
                if unsafe_boundary_seen:
                    raise ValueError(
                        "safe saturation points must precede the first unsafe boundary"
                    )
            else:
                unsafe_boundary_seen = True
        return self

    @classmethod
    def from_runpod_adapter(
        cls,
        *,
        configuration: TwoH100ProviderConfiguration,
        adapter: object,
        points: tuple[ProviderSaturationPoint, ...],
        evidence_class: CapacityEvidenceClass,
    ) -> Self:
        """Create a report from the exact configuration used by an adapter."""

        from robata.inference.runpod import RunPodVisionAdapter

        if not isinstance(adapter, RunPodVisionAdapter):
            raise TypeError("adapter must be a RunPodVisionAdapter")
        return cls(
            configuration=configuration,
            endpoint_config=adapter.config,
            capabilities=adapter.capabilities_snapshot,
            retry_policy=adapter.retry_policy,
            points=points,
            evidence_class=evidence_class,
        )

    @property
    def safe_point(self) -> ProviderSaturationPoint:
        return max(
            (point for point in self.points if point.safe_envelope),
            key=lambda point: point.offered_concurrency,
        )

    def render_markdown(self) -> str:
        gpu = self.safe_point.telemetry.gpu
        retry = self.configuration.retry_policy
        lines = [
            "# Two-H100 Provider Saturation Report",
            "",
            f"- Evidence class: {self.evidence_class.value}",
            "- Provider/model: "
            f"{self.configuration.provider} / {self.configuration.model_identifier}",
            f"- Topology: {self.configuration.topology.value}",
            "- Pinned prompt/context contracts: "
            + ", ".join(
                f"{contract.task.value}:{contract.prompt_version}/"
                f"{contract.prompt_sha256} (max input {contract.max_input_tokens}, "
                f"timeout {contract.timeout_ms} ms, generation "
                f"{contract.generation_config_sha256})"
                for contract in self.configuration.request_contracts
            ),
            f"- Pinned output limit: {self.configuration.max_output_tokens} tokens",
            "- Endpoint/adapter: "
            f"{self.configuration.endpoint_configuration.endpoint_url}; "
            f"{self.configuration.endpoint_configuration.adapter_version}",
            "- Retry policy: "
            f"{retry.version}; max attempts {retry.max_attempts}; "
            f"delay {retry.base_delay_ms}/{retry.max_delay_ms} ms",
            f"- GPU telemetry: {gpu.gpu_count} x {gpu.gpu_sku}; driver {gpu.driver_version}; "
            f"runtime {gpu.runtime_version}",
            f"- GPU inventory: {gpu.hardware_inventory_artifact_uri} "
            f"(sha256:{gpu.hardware_inventory_sha256})",
            f"- Metric source: {gpu.metric_source}; telemetry: {gpu.telemetry_artifact_uri} "
            f"(sha256:{gpu.telemetry_artifact_sha256})",
            "- Qualification sessions: "
            + ", ".join(point.qualification_session.session_id for point in self.points),
            "- Fresh P6 namespaces: "
            + ", ".join(point.run_namespace for point in self.points),
            "- Work-count reconciliation: adapter-observed provider output tokens and "
            "known attempt counts match the P6 capacity observation; input/output token "
            "rates are shown only when every provider attempt reported that usage.",
            "- Safe envelope rule: every final logical call is accepted with known input "
            "and output tokens, and the GPU telemetry reports zero OOMs.",
            "",
            "| Concurrency | Safe | Final/accepted/rejected | Attempts | Adapter images/s | "
            "Adapter calls/s | Adapter HTTP/s | Adapter input tokens/s | "
            "Adapter output tokens/s | GPU min/recording-h | "
            "Accepted queue P50/P95/P99 ms | Accepted execution P50/P95/P99 ms | "
            "Accepted TTFT P50/P95/P99 ms | Final E2E P50/P95/P99 ms | "
            "GPU util | GPU memory GiB | KV cache | OOM | Canonical retries | "
            "Adapter transport retries |",
            "| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for point in self.points:
            workload = point.telemetry.adapter_terminal_workload
            lines.append(
                (
                    "| {concurrency} | {safe} | {terminal}/{accepted}/{rejected} | "
                    "{attempts} | {images:.2f} | {calls:.2f} | {http_requests:.2f} | "
                    "{input_tokens} | {output_tokens} | {gpu_minutes:.2f} | "
                    "{queue} | {execution} | {ttft} | {e2e} | {util:.1%} | "
                    "{memory:.2f} | {kv_cache:.1%} | {oom} | {canonical_retries} | "
                    "{transport_retries} |"
                ).format(
                    concurrency=point.offered_concurrency,
                    safe="yes" if point.safe_envelope else "no",
                    terminal=point.telemetry.terminal_response_count,
                    accepted=point.telemetry.accepted_response_count,
                    rejected=point.telemetry.rejected_response_count,
                    attempts=workload.provider_attempt_count,
                    images=point.adapter_provider_images_per_second,
                    calls=point.adapter_logical_calls_per_second,
                    http_requests=point.adapter_http_requests_per_second,
                    input_tokens=_format_rate(point.adapter_input_tokens_per_second),
                    output_tokens=_format_rate(point.adapter_output_tokens_per_second),
                    gpu_minutes=point.aggregate_gpu_minutes_per_recording_hour,
                    queue=_format_latency(point.telemetry.provider_queue),
                    execution=_format_latency(point.telemetry.provider_execution),
                    ttft=_format_latency(point.telemetry.time_to_first_token),
                    e2e=_format_latency(point.telemetry.end_to_end),
                    util=point.telemetry.gpu.gpu_utilization_fraction,
                    memory=point.telemetry.gpu.gpu_memory_bytes / 1_073_741_824,
                    kv_cache=point.telemetry.gpu.kv_cache_utilization_fraction,
                    oom=point.telemetry.gpu.oom_count,
                    canonical_retries=point.telemetry.canonical_retry_attempt_count,
                    transport_retries=point.telemetry.adapter_transport_retry_count,
                )
            )
        return "\n".join(lines)

class TwoH100TopologyComparison(StrictModel):
    """Like-for-like comparison of two explicitly supported two-H100 topologies."""

    single_card_replicas: TwoH100ProviderQualificationReport
    tensor_parallel: TwoH100ProviderQualificationReport
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        replicas = self.single_card_replicas
        tensor_parallel = self.tensor_parallel
        if replicas.configuration.topology is not TwoH100Topology.TWO_SINGLE_CARD_REPLICAS:
            raise ValueError("single_card_replicas report has the wrong topology")
        if tensor_parallel.configuration.topology is not TwoH100Topology.TWO_CARD_TENSOR_PARALLEL:
            raise ValueError("tensor_parallel report has the wrong topology")
        required_topologies = {
            TwoH100Topology.TWO_SINGLE_CARD_REPLICAS,
            TwoH100Topology.TWO_CARD_TENSOR_PARALLEL,
        }
        replicas_support_both = required_topologies.issubset(
            replicas.configuration.supported_topologies
        )
        tensor_parallel_support_both = required_topologies.issubset(
            tensor_parallel.configuration.supported_topologies
        )
        if not replicas_support_both or not tensor_parallel_support_both:
            raise ValueError("topology comparison requires model and engine support for both modes")
        if replicas.evidence_class is not tensor_parallel.evidence_class:
            raise ValueError("topology reports must use the same evidence class")
        comparable_fields = (
            "workload_manifest_digest",
            "provider",
            "model_identifier",
            "model_version",
            "request_contracts",
            "inference_engine",
            "precision_or_quantization",
            "max_images_per_request",
            "max_input_tokens",
            "max_output_tokens",
            "native_batch_max_size",
            "max_concurrent_requests",
            "retry_policy",
            "supported_topologies",
        )
        if any(
            getattr(replicas.configuration, field) != getattr(tensor_parallel.configuration, field)
            for field in comparable_fields
        ):
            raise ValueError("topology reports must use the same workload and model configuration")
        if (
            _qualification_transport_projection(replicas.configuration.endpoint_configuration)
            != _qualification_transport_projection(
                tensor_parallel.configuration.endpoint_configuration
            )
        ):
            raise ValueError("topology reports must use the same transport configuration")
        replica_gpu = replicas.safe_point.telemetry.gpu
        tensor_gpu = tensor_parallel.safe_point.telemetry.gpu
        if (
            replica_gpu.hardware_inventory_sha256,
            replica_gpu.gpu_sku,
            replica_gpu.driver_version,
            replica_gpu.runtime_version,
            replica_gpu.metric_source,
        ) != (
            tensor_gpu.hardware_inventory_sha256,
            tensor_gpu.gpu_sku,
            tensor_gpu.driver_version,
            tensor_gpu.runtime_version,
            tensor_gpu.metric_source,
        ):
            raise ValueError("topology reports must use the same GPU inventory and metric source")
        return self

    def render_markdown(self) -> str:
        replicas = self.single_card_replicas.safe_point
        tensor_parallel = self.tensor_parallel.safe_point
        replicas_images = replicas.capacity.provider_images_per_wall_hour
        tensor_images = tensor_parallel.capacity.provider_images_per_wall_hour
        assert replicas_images is not None
        assert tensor_images is not None
        return "\n".join(
            (
                "# Two-H100 Topology Comparison",
                "",
                f"- Evidence class: {self.single_card_replicas.evidence_class.value}",
                "",
                "| Topology | Safe concurrency | Provider images/h | GPU min/recording-h |",
                "| --- | ---: | ---: | ---: |",
                "| Two single-card replicas | "
                f"{replicas.offered_concurrency} | {replicas_images:.2f} | "
                f"{replicas.aggregate_gpu_minutes_per_recording_hour:.2f} |",
                "| Two-card tensor parallel | "
                f"{tensor_parallel.offered_concurrency} | {tensor_images:.2f} | "
                f"{tensor_parallel.aggregate_gpu_minutes_per_recording_hour:.2f} |",
            )
        )


def _qualification_transport_projection(
    endpoint: RunPodEndpointConfig,
) -> tuple[object, ...]:
    """Compare transport behavior while allowing topology-specific endpoint URLs."""

    return (
        endpoint.provider,
        endpoint.adapter_version,
        endpoint.request_contract_version,
        endpoint.response_contract_version,
        endpoint.native_batch_enabled,
        endpoint.batch_request_contract_version,
        endpoint.batch_response_contract_version,
        endpoint.native_batch_max_size,
        endpoint.max_concurrent_requests,
        endpoint.request_timeout_cap_ms,
        endpoint.max_response_bytes,
    )


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.2f}"

def _format_latency(latency: ProviderLatencyPercentiles | None) -> str:
    if latency is None:
        return "n/a"
    return f"{latency.p50_ms}/{latency.p95_ms}/{latency.p99_ms}"


def compare_two_h100_topologies(
    *,
    single_card_replicas: TwoH100ProviderQualificationReport,
    tensor_parallel: TwoH100ProviderQualificationReport,
) -> TwoH100TopologyComparison:
    """Build a non-promotional comparison after each topology has been measured."""

    return TwoH100TopologyComparison(
        single_card_replicas=single_card_replicas,
        tensor_parallel=tensor_parallel,
    )


__all__ = [
    "ProviderAdapterTerminalWorkload",
    "ProviderGpuMeasurement",
    "ProviderLatencyPercentiles",
    "ProviderQualificationCollector",
    "ProviderQualificationRunContext",
    "ProviderRuntimeTelemetry",
    "ProviderSaturationPoint",
    "ProviderTimingSample",
    "TwoH100ProviderConfiguration",
    "TwoH100ProviderQualificationReport",
    "TwoH100Topology",
    "TwoH100TopologyComparison",
    "compare_two_h100_topologies",
    "run_provider_saturation_point",
]