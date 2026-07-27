"""Bound local quality/capacity qualification artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import isclose, isfinite
from typing import Annotated, Any, Final, Literal, Self

from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from robata.application.canonical.qualification_evidence import (
    CanonicalRecoveryEvidenceClass,
    CanonicalRecoveryQualificationEvidence,
    CanonicalRecoveryScenario,
)
from robata.benchmark.evidence import BenchmarkEvidenceContext
from robata.benchmark.pareto import LocalSamplingDenseParetoReport
from robata.benchmark.promotion import (
    GateCategory,
    GateResult,
    PromotionDecision,
    PromotionGate,
    PromotionGateRegistry,
)
from robata.benchmark.provider_qualification import (
    ProviderSaturationPoint,
    TwoH100ProviderConfiguration,
    TwoH100ProviderQualificationReport,
)
from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.qa import ProductQAIssue
from robata.inference.models import ModelCapabilities
from robata.inference.runpod import RunPodEndpointConfig, RunPodRetryPolicy
from robata.runtime.capacity import (
    CapacityEvidenceClass,
    MeasuredCapacityReport,
    MeasuredCapacityStatus,
    ProviderMode,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class LocalQualificationContext(StrictModel):
    """Fixed environment and workload scope for one local qualification run."""

    workload_manifest_digest: Sha256Digest
    recording_count: PositiveInt
    recording_duration_ns: Nanoseconds
    camera_count: Literal[6]
    model_identifier: NonEmptyString
    provider_mode: ProviderMode
    provider_concurrency: PositiveInt
    hardware_identifier: NonEmptyString
    run_duration_ns: Nanoseconds


class LocalRecoveryScenario(StrictModel):
    """One exercised local failure/recovery path retained with the qualification."""

    scenario_id: Literal[
        "RESTART_REPLAY",
        "PROVIDER_RETRY",
        "PROVIDER_TIMEOUT",
        "OUTBOX_RECONCILIATION",
    ]
    terminal_reconciled: bool
    outbox_reconciled: bool

    @model_validator(mode="after")
    def validate_reconciliation(self) -> Self:
        if not self.terminal_reconciled or not self.outbox_reconciled:
            raise ValueError("local recovery scenarios must reconcile terminal and outbox state")
        return self


class LocalQualityCapacityQualificationPackage(StrictModel):
    """One reproducible local P8 artifact, explicitly not production evidence."""

    package_version: Literal["local-quality-capacity-qualification-v1"] = (
        "local-quality-capacity-qualification-v1"
    )
    context: LocalQualificationContext
    pareto: LocalSamplingDenseParetoReport
    capacity: MeasuredCapacityReport
    recovery_scenarios: tuple[LocalRecoveryScenario, ...] = Field(min_length=4)
    evidence_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    representative_data_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    real_hardware_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_package(self) -> Self:
        if self.pareto.fixture_manifest_digest != self.context.workload_manifest_digest:
            raise ValueError("Pareto fixture manifest does not match qualification workload")
        if self.pareto.model_identifier != self.context.model_identifier:
            raise ValueError("Pareto model identifier does not match qualification context")
        if self.pareto.provider_mode != self.context.provider_mode.value:
            raise ValueError("Pareto provider mode does not match qualification context")
        if self.capacity.evidence_class is not CapacityEvidenceClass.LOCAL_CONFORMANCE:
            raise ValueError("local qualification capacity must use LOCAL_CONFORMANCE evidence")
        if self.capacity.workload_fingerprint != self.context.workload_manifest_digest:
            raise ValueError("capacity workload does not match qualification context")
        if self.capacity.provider_mode is not self.context.provider_mode:
            raise ValueError("capacity provider mode does not match qualification context")
        if self.capacity.recording_count != self.context.recording_count:
            raise ValueError("capacity recording count does not match qualification context")
        if self.capacity.camera_count != self.context.camera_count:
            raise ValueError("capacity camera count does not match qualification context")
        if self.capacity.recording_duration_ns != self.context.recording_duration_ns:
            raise ValueError("capacity recording duration does not match qualification context")
        if self.capacity.wall_time_ns != self.context.run_duration_ns:
            raise ValueError("capacity wall time does not match qualification run duration")
        if self.capacity.measurement_status is not MeasuredCapacityStatus.AVAILABLE:
            raise ValueError("local qualification requires denominator-safe capacity facts")
        scenario_ids = tuple(scenario.scenario_id for scenario in self.recovery_scenarios)
        required = ("OUTBOX_RECONCILIATION", "PROVIDER_RETRY", "PROVIDER_TIMEOUT", "RESTART_REPLAY")
        if tuple(sorted(scenario_ids)) != required:
            raise ValueError("qualification package requires each recovery scenario exactly once")
        return self

    def as_dict(self) -> dict[str, object]:
        """Return one JSON-ready local qualification artifact."""

        return self.model_dump(mode="json")

    def render_markdown(self) -> str:
        """Render quality, capacity, and recovery facts as one local-only report."""

        capacity = self.capacity
        recording_rtf = capacity.recording_hours_per_wall_hour
        camera_rtf = capacity.camera_hours_per_wall_hour
        pareto_markdown = self.pareto.render_markdown().rstrip()
        pareto_body = pareto_markdown.split("\n", 2)[-1]
        lines = [
            "# Local quality-capacity qualification package",
            "",
            f"- Workload manifest: {self.context.workload_manifest_digest}",
            f"- Recording count: {self.context.recording_count}",
            f"- Recording duration (ns): {self.context.recording_duration_ns}",
            f"- Camera count: {self.context.camera_count}",
            f"- Model: {self.context.model_identifier}",
            f"- Provider mode: {self.context.provider_mode.value}",
            f"- Provider concurrency: {self.context.provider_concurrency}",
            f"- Hardware: {self.context.hardware_identifier}",
            f"- Run duration (ns): {self.context.run_duration_ns}",
            "- Evidence class: LOCAL_CONFORMANCE",
            "- Measurement status: NOT_MEASURED",
            "- Representative data: NOT_MEASURED",
            "- Real hardware: NOT_MEASURED",
            "- Production eligible: NO",
            "",
            "## Capacity",
            (
                "- Recording RTF: NOT_AVAILABLE"
                if recording_rtf is None
                else f"- Recording RTF: {recording_rtf:.6f}"
            ),
            (
                "- Camera RTF: NOT_AVAILABLE"
                if camera_rtf is None
                else f"- Camera RTF: {camera_rtf:.6f}"
            ),
            (
                "- Provider images: NOT_AVAILABLE"
                if capacity.provider_images is None
                else f"- Provider images: {capacity.provider_images}"
            ),
            (
                "- Logical calls: NOT_AVAILABLE"
                if capacity.logical_calls is None
                else f"- Logical calls: {capacity.logical_calls}"
            ),
            "",
            "## Pareto frontier",
            pareto_body,
            "",
            "## Recovery evidence",
            "| Scenario | Terminal reconciled | Outbox reconciled |",
            "| --- | --- | --- |",
        ]
        lines.extend(
            f"| {scenario.scenario_id} | YES | YES |"
            for scenario in self.recovery_scenarios
        )
        return "\n".join(lines) + "\n"


def build_local_quality_capacity_qualification_package(
    *,
    context: LocalQualificationContext,
    pareto: LocalSamplingDenseParetoReport,
    capacity: MeasuredCapacityReport,
    recovery_scenarios: Iterable[LocalRecoveryScenario],
) -> LocalQualityCapacityQualificationPackage:
    """Build one deterministic local quality/capacity qualification artifact.

    This helper keeps the P8 assembly boundary explicit while preserving the
    package's local-only evidence class and denominator-safe capacity checks.
    """

    checked_context = TypeAdapter(LocalQualificationContext).validate_python(
        context,
        strict=True,
    )
    checked_pareto = TypeAdapter(LocalSamplingDenseParetoReport).validate_python(
        pareto,
        strict=True,
    )
    checked_capacity = TypeAdapter(MeasuredCapacityReport).validate_python(
        capacity,
        strict=True,
    )
    checked_scenarios = tuple(
        TypeAdapter(LocalRecoveryScenario).validate_python(scenario, strict=True)
        for scenario in recovery_scenarios
    )
    ordered_scenarios = tuple(
        sorted(checked_scenarios, key=lambda scenario: scenario.scenario_id)
    )
    return LocalQualityCapacityQualificationPackage(
        context=checked_context,
        pareto=checked_pareto,
        capacity=checked_capacity,
        recovery_scenarios=ordered_scenarios,
    )


_PRODUCTION_QUALIFICATION_REPORT_VERSION: Final[
    Literal["representative-production-qualification-v1"]
] = "representative-production-qualification-v1"
_HOUR_NS: Final[int] = 3_600_000_000_000
_DAY_NS: Final[int] = 24 * _HOUR_NS
_THREE_DAYS_NS: Final[int] = 3 * _DAY_NS
_MINIMUM_SERVICE_RECORDING_RTF: Final[float] = 25.0
_NOMINAL_500_HOURS_PER_DAY_RTF: Final[float] = 500.0 / 24.0
_PREFERRED_SATURATION_RECORDING_RTF: Final[float] = 29.762
_MAXIMUM_NOMINAL_GPU_UTILIZATION: Final[float] = 0.70
_MAXIMUM_GPU_MINUTES_PER_RECORDING_HOUR: Final[float] = 4.03

NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFiniteFloat = Annotated[
    float,
    Field(strict=True, ge=0, allow_inf_nan=False),
]
PositiveFiniteFloat = Annotated[
    float,
    Field(strict=True, gt=0, allow_inf_nan=False),
]
ExternalQualificationGateId = Literal["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
ExternalQualificationGateStatus = Literal[
    "NOT_MEASURED",
    "MEASURED_PENDING_INDEPENDENT_REVIEW",
    "PENDING_INDEPENDENT_REVIEW",
]


class ProductionQualificationScope(StrictModel):
    """Immutable configuration and input identities for one P10 qualification."""

    scope_version: Literal["production-qualification-scope-v1"] = (
        "production-qualification-scope-v1"
    )
    scope_sha256: Sha256Digest
    qualification_id: OpaqueUuid
    qualification_run_namespace: NonEmptyString
    workload_manifest_digest: Sha256Digest
    benchmark_manifest_digest: Sha256Digest
    governed_corpus_digest: Sha256Digest
    ground_truth_manifest_digest: Sha256Digest
    grouped_split_manifest_digest: Sha256Digest
    provider_configuration_digest: Sha256Digest
    code_revision_sha256: Sha256Digest
    schema_catalog_sha256: Sha256Digest
    sampler_policy_sha256: Sha256Digest
    media_manifest_sha256: Sha256Digest
    calibration_profile_sha256: Sha256Digest
    preprocess_policy_sha256: Sha256Digest
    arrival_distribution_sha256: Sha256Digest
    provider_adapter_version: NonEmptyString
    storage_adapter_version: NonEmptyString
    storage_configuration_sha256: Sha256Digest
    broker_adapter_version: NonEmptyString
    broker_configuration_sha256: Sha256Digest
    host_resource_inventory_uri: NonEmptyString
    host_resource_inventory_sha256: Sha256Digest

    @classmethod
    def create(cls, **values: object) -> Self:
        """Build a content-addressed scope over every frozen qualification input."""

        if "scope_sha256" in values:
            raise ValueError("scope_sha256 is derived")
        scope_values: dict[str, Any] = {**values, "scope_sha256": "0" * 64}
        draft = cls.model_construct(**scope_values)
        scope_sha256 = semantic_sha256(draft.model_dump(mode="json", exclude={"scope_sha256"}))
        return cls.model_validate({**draft.model_dump(mode="python"), "scope_sha256": scope_sha256})

    @model_validator(mode="after")
    def validate_scope_digest(self) -> Self:
        self._assert_scope_digest()
        return self

    def _assert_scope_digest(self) -> None:
        expected_sha256 = semantic_sha256(self.model_dump(mode="json", exclude={"scope_sha256"}))
        if self.scope_sha256 != expected_sha256:
            raise ValueError("scope_sha256 does not match the frozen qualification scope")


class RepresentativeQueueObservation(StrictModel):
    """One bounded queue observation from the representative arrival profile."""

    name: Literal["ingress", "provider", "publish"]
    capacity: PositiveInt
    high_watermark: NonNegativeInt
    end_depth: NonNegativeInt

    @model_validator(mode="after")
    def validate_queue_depths(self) -> Self:
        if self.high_watermark > self.capacity:
            raise ValueError("queue high-watermark cannot exceed queue capacity")
        if self.end_depth > self.capacity:
            raise ValueError("queue end depth cannot exceed queue capacity")
        return self


class RepresentativeServiceCapacityEvidence(StrictModel):
    """Measured 24-hour assembled-service capacity and queue evidence for P10."""

    evidence_version: Literal["representative-service-capacity-evidence-v2"] = (
        "representative-service-capacity-evidence-v2"
    )
    run_namespace: NonEmptyString
    qualification_scope_sha256: Sha256Digest
    arrival_distribution_sha256: Sha256Digest
    arrival_observation_uri: NonEmptyString
    arrival_observation_sha256: Sha256Digest
    arrival_recording_count: PositiveInt
    arrival_recording_hours: PositiveFiniteFloat
    arrival_camera_hours: PositiveFiniteFloat
    provider_configuration_digest: Sha256Digest
    code_revision_sha256: Sha256Digest
    schema_catalog_sha256: Sha256Digest
    sampler_policy_sha256: Sha256Digest
    media_manifest_sha256: Sha256Digest
    calibration_profile_sha256: Sha256Digest
    preprocess_policy_sha256: Sha256Digest
    storage_adapter_version: NonEmptyString
    storage_configuration_sha256: Sha256Digest
    broker_adapter_version: NonEmptyString
    broker_configuration_sha256: Sha256Digest
    capacity: MeasuredCapacityReport
    run_duration_ns: Nanoseconds
    queues: tuple[RepresentativeQueueObservation, ...] = Field(min_length=3)
    arrival_peak_observed: Literal[True] = True
    backlog_drained_after_peak: Literal[True] = True
    backlog_start_count: NonNegativeInt
    backlog_end_count: NonNegativeInt
    outbox_delivery_p95_ns: Nanoseconds
    outbox_delivery_observation_uri: NonEmptyString
    outbox_delivery_observation_sha256: Sha256Digest
    outbox_delivery_observation_count: PositiveInt
    review_delivery_p95_ns: Nanoseconds
    review_delivery_observation_uri: NonEmptyString
    review_delivery_observation_sha256: Sha256Digest
    review_delivery_observation_count: PositiveInt
    evidence_class: Literal["PRODUCTION_QUALIFICATION"] = "PRODUCTION_QUALIFICATION"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_service_capacity(self) -> Self:
        self._assert_service_capacity()
        return self

    def _assert_service_capacity(self) -> None:
        if self.run_duration_ns < 24 * _HOUR_NS:
            raise ValueError("representative service capacity requires at least a 24-hour soak")
        minimum_arrival_recording_hours = (
            _NOMINAL_500_HOURS_PER_DAY_RTF * self.run_duration_ns / _HOUR_NS
        )
        if self.arrival_recording_hours < minimum_arrival_recording_hours:
            raise ValueError(
                "service capacity arrival observation requires 500 recording-hours per 24 hours"
            )
        if not isclose(
            self.arrival_camera_hours,
            self.arrival_recording_hours * 6,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("arrival camera-hours must match the six-camera recording workload")
        if self.outbox_delivery_p95_ns < 0 or self.review_delivery_p95_ns < 0:
            raise ValueError("delivery latency observations must be nonnegative")
        if (
            self.outbox_delivery_observation_count <= 0
            or self.review_delivery_observation_count <= 0
        ):
            raise ValueError("delivery latency observations require positive sample counts")
        if self.capacity.measurement_status is not MeasuredCapacityStatus.AVAILABLE:
            raise ValueError("service capacity requires denominator-safe capacity facts")
        if self.capacity.evidence_class is not CapacityEvidenceClass.PRODUCTION_QUALIFICATION:
            raise ValueError("service capacity requires PRODUCTION_QUALIFICATION evidence")
        if self.capacity.provider_mode is not ProviderMode.NETWORK_PROVIDER:
            raise ValueError("service capacity requires a network provider measurement")
        if self.capacity.execution_mode != "FRESH":
            raise ValueError("service capacity requires fresh assembled-pipeline execution")
        if self.capacity.camera_count != 6:
            raise ValueError("service capacity requires the six-camera workload")
        if self.capacity.wall_time_ns != self.run_duration_ns:
            raise ValueError("service capacity duration must match the capacity wall time")
        recording_rtf = self.capacity.recording_hours_per_wall_hour
        if recording_rtf is None or recording_rtf < _MINIMUM_SERVICE_RECORDING_RTF:
            raise ValueError("service capacity must sustain at least 25 recording-RTF")
        names = tuple(queue.name for queue in self.queues)
        if names != ("ingress", "provider", "publish"):
            raise ValueError("service capacity queues must be ingress, provider, and publish")
        if any(queue.end_depth != 0 for queue in self.queues):
            raise ValueError("service capacity queues must drain by the end of the soak")
        if not any(queue.high_watermark for queue in self.queues):
            raise ValueError("service capacity requires an observed queue peak")
        if self.backlog_start_count == 0:
            raise ValueError("service capacity requires a nonzero backlog before draining")
        if self.backlog_end_count != 0:
            raise ValueError("service capacity must end without durable backlog")
        if self.backlog_end_count > self.backlog_start_count:
            raise ValueError("service capacity backlog may not grow across the soak")


class RepresentativeMediaProfile(StrictModel):
    """One codec/resolution/FPS/GOP stratum in the representative media matrix."""

    codec: NonEmptyString
    width: PositiveInt
    height: PositiveInt
    frames_per_second: PositiveFiniteFloat
    gop_frames: PositiveInt
    camera_count: Literal[6] = 6


class RepresentativeQualityCoverageEvidence(StrictModel):
    """Coverage facts that keep proxy and semantic P10 claims distinct."""

    evidence_version: Literal["representative-quality-coverage-evidence-v1"] = (
        "representative-quality-coverage-evidence-v1"
    )
    run_namespace: NonEmptyString
    qualification_scope_sha256: Sha256Digest
    qa_class_ids: tuple[ProductQAIssue, ...] = Field(min_length=21, max_length=21)
    media_profiles: tuple[RepresentativeMediaProfile, ...] = Field(min_length=1)
    preprocess_views: tuple[Literal["RAW", "FISHEYE", "PERSPECTIVE", "EGOCENTRIC"], ...] = (
        "RAW",
        "FISHEYE",
        "PERSPECTIVE",
        "EGOCENTRIC",
    )
    warning_mark_observed: Literal[True] = True
    fail_observed: Literal[True] = True
    abstained_observed: Literal[True] = True
    incomplete_observed: Literal[True] = True
    missing_source_observed: Literal[True] = True
    decode_gap_observed: Literal[True] = True
    event_boundary_observed: Literal[True] = True
    temporal_adjudication_observed: Literal[True] = True
    observation_uri: NonEmptyString
    observation_sha256: Sha256Digest
    evidence_class: Literal["PRODUCTION_QUALIFICATION"] = "PRODUCTION_QUALIFICATION"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        return self._assert_coverage()

    def _assert_coverage(self) -> Self:
        if self.qa_class_ids != tuple(ProductQAIssue):
            raise ValueError(
                "quality coverage must retain all 21 product QA classes in vocabulary order"
            )
        required_views = {"RAW", "FISHEYE", "PERSPECTIVE", "EGOCENTRIC"}
        if set(self.preprocess_views) != required_views:
            raise ValueError("quality coverage must include raw and all declared geometry views")
        if len(self.preprocess_views) != len(set(self.preprocess_views)):
            raise ValueError("quality coverage preprocess views must be unique")
        return self


class RepresentativeDeadlineEvidence(StrictModel):
    """Measured QA T+1 and annotation T+3 completion populations."""

    evidence_version: Literal["representative-deadline-evidence-v1"] = (
        "representative-deadline-evidence-v1"
    )
    run_namespace: NonEmptyString
    qualification_scope_sha256: Sha256Digest
    qa_observation_count: PositiveInt
    qa_deadline_miss_count: NonNegativeInt
    qa_completion_p50_ns: Nanoseconds
    qa_completion_p95_ns: Nanoseconds
    qa_completion_p99_ns: Nanoseconds
    qa_deadline_ns: Nanoseconds = _DAY_NS
    annotation_observation_count: PositiveInt
    annotation_deadline_miss_count: NonNegativeInt
    annotation_completion_p50_ns: Nanoseconds
    annotation_completion_p95_ns: Nanoseconds
    annotation_completion_p99_ns: Nanoseconds
    annotation_deadline_ns: Nanoseconds = _THREE_DAYS_NS
    observation_uri: NonEmptyString
    observation_sha256: Sha256Digest
    evidence_class: Literal["PRODUCTION_QUALIFICATION"] = "PRODUCTION_QUALIFICATION"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_deadlines(self) -> Self:
        return self._assert_deadlines()

    def _assert_deadlines(self) -> Self:
        for prefix, count, misses, p50, p95, p99, deadline, expected_deadline in (
            (
                "QA",
                self.qa_observation_count,
                self.qa_deadline_miss_count,
                self.qa_completion_p50_ns,
                self.qa_completion_p95_ns,
                self.qa_completion_p99_ns,
                self.qa_deadline_ns,
                _DAY_NS,
            ),
            (
                "annotation",
                self.annotation_observation_count,
                self.annotation_deadline_miss_count,
                self.annotation_completion_p50_ns,
                self.annotation_completion_p95_ns,
                self.annotation_completion_p99_ns,
                self.annotation_deadline_ns,
                _THREE_DAYS_NS,
            ),
        ):
            if deadline != expected_deadline:
                raise ValueError(f"{prefix} deadline does not match the P10 target")
            if misses > count:
                raise ValueError(f"{prefix} deadline misses exceed the observed population")
            if misses or p50 <= 0 or not p50 <= p95 <= p99 <= deadline:
                raise ValueError(f"{prefix} deadline evidence does not meet its target")
        return self


class RepresentativeCostResourceEvidence(StrictModel):
    """Measured cost and resource axes retained with one P10 qualification run."""

    evidence_version: Literal["representative-cost-resource-evidence-v1"] = (
        "representative-cost-resource-evidence-v1"
    )
    run_namespace: NonEmptyString
    qualification_scope_sha256: Sha256Digest
    recording_hours: PositiveFiniteFloat
    camera_hours: PositiveFiniteFloat
    currency: Literal["USD"] = "USD"
    provider_cost_usd: NonNegativeFiniteFloat
    gpu_cost_usd: NonNegativeFiniteFloat
    object_storage_cost_usd: NonNegativeFiniteFloat
    object_egress_cost_usd: NonNegativeFiniteFloat
    database_cost_usd: NonNegativeFiniteFloat
    queue_cost_usd: NonNegativeFiniteFloat
    total_cost_usd: NonNegativeFiniteFloat
    cpu_seconds: NonNegativeFiniteFloat
    gpu_seconds: NonNegativeFiniteFloat
    nvme_read_bytes: NonNegativeInt
    nvme_write_bytes: NonNegativeInt
    object_storage_bytes: NonNegativeInt
    object_egress_bytes: NonNegativeInt
    database_operation_count: NonNegativeInt
    queue_operation_count: NonNegativeInt
    observation_uri: NonEmptyString
    observation_sha256: Sha256Digest
    evidence_class: Literal["PRODUCTION_QUALIFICATION"] = "PRODUCTION_QUALIFICATION"
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_total_cost(self) -> Self:
        return self._assert_total_cost()

    def _assert_total_cost(self) -> Self:
        if not isclose(
            self.camera_hours,
            self.recording_hours * 6,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("cost evidence camera-hours must match the six-camera workload")
        expected_total = sum(
            (
                self.provider_cost_usd,
                self.gpu_cost_usd,
                self.object_storage_cost_usd,
                self.object_egress_cost_usd,
                self.database_cost_usd,
                self.queue_cost_usd,
            )
        )
        if not isclose(self.total_cost_usd, expected_total, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("cost evidence total does not match its measured components")
        return self


class RepresentativeOperationsEvidence(StrictModel):
    """Runbook, security, and retention artifacts attached to a P10 report."""

    evidence_version: Literal["representative-operations-evidence-v1"] = (
        "representative-operations-evidence-v1"
    )
    run_namespace: NonEmptyString
    qualification_scope_sha256: Sha256Digest
    runbook_uri: NonEmptyString
    runbook_sha256: Sha256Digest
    security_retention_uri: NonEmptyString
    security_retention_sha256: Sha256Digest
    incident_response_uri: NonEmptyString
    incident_response_sha256: Sha256Digest
    evidence_class: Literal["PRODUCTION_QUALIFICATION"] = "PRODUCTION_QUALIFICATION"
    production_eligible: Literal[False] = False


class ExternalQualificationGateEvidence(StrictModel):
    """One unresolved external qualification gate retained without self-promotion."""

    gate_id: ExternalQualificationGateId
    status: ExternalQualificationGateStatus
    unresolved_reason: NonEmptyString
    supporting_artifact_uri: NonEmptyString | None = None
    supporting_artifact_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def validate_gate(self) -> Self:
        has_uri = self.supporting_artifact_uri is not None
        has_digest = self.supporting_artifact_sha256 is not None
        if has_uri != has_digest:
            raise ValueError("external gate artifact URI and digest must be supplied together")
        if self.status == "MEASURED_PENDING_INDEPENDENT_REVIEW" and not has_uri:
            raise ValueError("measured external gates require a supporting artifact")
        if self.status == "NOT_MEASURED" and has_uri:
            raise ValueError("unmeasured external gates cannot claim a supporting artifact")
        if self.gate_id == "E6" and self.status != "PENDING_INDEPENDENT_REVIEW":
            raise ValueError("E6 must remain pending independent review")
        return self


class GovernedQualityQualificationEvidence(StrictModel):
    """Frozen-label quality decision bound to its registry and evidence context."""

    context: BenchmarkEvidenceContext
    qualification_scope_sha256: Sha256Digest
    code_revision_sha256: Sha256Digest
    schema_catalog_sha256: Sha256Digest
    sampler_policy_sha256: Sha256Digest
    media_manifest_sha256: Sha256Digest
    calibration_profile_sha256: Sha256Digest
    preprocess_policy_sha256: Sha256Digest
    registry: PromotionGateRegistry
    decision: PromotionDecision
    qa_gate_ids: tuple[OpaqueUuid, ...] = Field(min_length=1)
    event_gate_ids: tuple[OpaqueUuid, ...] = Field(min_length=1)
    calibration_gate_ids: tuple[OpaqueUuid, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_governed_quality(self) -> Self:
        self._assert_governed_quality()
        return self

    def _assert_governed_quality(self) -> None:
        if self.registry.benchmark_id != self.context.benchmark_id:
            raise ValueError("quality registry benchmark does not match the governed context")
        if self.registry.evidence_context_digest != self.context.context_digest:
            raise ValueError("quality registry context does not match the governed context")
        if self.decision.timestamp.tzinfo is None or self.decision.timestamp.utcoffset() is None:
            raise ValueError("quality decision timestamp must be timezone-aware")
        if self.decision.timestamp < self.registry.frozen_at:
            raise ValueError("quality decision predates its frozen gate registry")
        if not self.decision.approved:
            raise ValueError("quality qualification requires an approved promotion decision")
        if self.decision.rejected_gates or self.decision.validation_errors:
            raise ValueError("approved quality decision cannot contain rejected or invalid gates")
        gate_by_id = {gate.gate_id: gate for gate in self.registry.gates}
        if len(gate_by_id) != len(self.registry.gates):
            raise ValueError("quality registry gate identities must be unique")
        if {gate.category for gate in self.registry.gates} != set(GateCategory):
            raise ValueError("quality registry must retain every required gate category")
        decision_results = self.decision.approved_gates + self.decision.rejected_gates
        result_gate_ids = tuple(result.evidence.get("gate_id") for result in decision_results)
        if any(not isinstance(gate_id, str) for gate_id in result_gate_ids):
            raise ValueError("quality decision results must identify their registered gate")
        if len(set(result_gate_ids)) != len(result_gate_ids):
            raise ValueError("quality decision may not report a gate more than once")
        if set(result_gate_ids) != set(gate_by_id):
            raise ValueError("quality decision results must cover exactly the frozen registry")
        if any(not result.passed for result in decision_results):
            raise ValueError("quality decision results must all pass")
        for result in decision_results:
            gate_id = result.evidence["gate_id"]
            assert isinstance(gate_id, str)
            gate = gate_by_id[gate_id]
            if result.category is not gate.category or result.threshold != gate.threshold:
                raise ValueError("quality decision result does not match its frozen gate")
            self._validate_gate_result_binding(result, gate)
            if (
                result.evidence.get("evidence_context_digest") != self.context.context_digest
                or result.evidence.get("evidence_context_identity") != self.context.context_identity
            ):
                raise ValueError("quality decision result does not match governed evidence")
            self._validate_gate_result_threshold(result, gate)
        if any(gate.data_split != self.context.data_split for gate in self.registry.gates):
            raise ValueError("quality gates must use the governed frozen-test split")
        selected_ids = self.qa_gate_ids + self.event_gate_ids + self.calibration_gate_ids
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("QA, event, and calibration gate IDs must be disjoint")
        if any(gate_id not in gate_by_id for gate_id in selected_ids):
            raise ValueError("quality qualification selected an unknown promotion gate")
        if any(gate_by_id[gate_id].category is not GateCategory.QA for gate_id in self.qa_gate_ids):
            raise ValueError("QA gate IDs must identify QA promotion gates")
        event_categories = {gate_by_id[gate_id].category for gate_id in self.event_gate_ids}
        if (
            not {
                GateCategory.EVENT_PROPOSAL,
                GateCategory.ACTION_BOUNDARY,
            }
            <= event_categories
        ):
            raise ValueError("event gate IDs must cover event proposal and action boundary")
        for gate_id in self.calibration_gate_ids:
            gate = gate_by_id[gate_id]
            if (
                gate.category is not GateCategory.QA
                or "calibrat" not in gate.metric_definition.casefold()
            ):
                raise ValueError("calibration gate IDs must identify calibration QA gates")

    @staticmethod
    def _validate_gate_result_binding(result: GateResult, gate: PromotionGate) -> None:
        required_strata = result.evidence.get("required_strata")
        if (
            result.evidence.get("metric_definition") != gate.metric_definition
            or result.evidence.get("comparison") != gate.comparison
            or result.evidence.get("margin") != gate.margin
            or result.evidence.get("denominator") != gate.denominator
            or result.evidence.get("data_split") != gate.data_split
            or result.evidence.get("measurement_status") != "MEASURED"
            or not isinstance(required_strata, (list, tuple))
            or tuple(required_strata) != gate.required_strata
        ):
            raise ValueError("quality decision result does not match its frozen gate")

    @staticmethod
    def _validate_gate_result_threshold(result: GateResult, gate: PromotionGate) -> None:
        actual_value = result.actual_value
        if actual_value is None or not isfinite(actual_value):
            raise ValueError("quality decision result must report a finite actual value")
        raw_stratum_values = result.evidence.get("stratum_values")
        if not isinstance(raw_stratum_values, dict) or set(raw_stratum_values) != set(
            gate.required_strata
        ):
            raise ValueError("quality decision result must report every required stratum")
        values = [actual_value]
        for value in raw_stratum_values.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError("quality decision stratum values must be finite numbers")
            values.append(float(value))
        if gate.comparison == "GTE":
            passed = all(value >= gate.threshold - gate.margin for value in values)
        elif gate.comparison == "LTE":
            passed = all(value <= gate.threshold + gate.margin for value in values)
        else:
            passed = all(abs(value - gate.threshold) <= gate.margin for value in values)
        if not passed:
            raise ValueError("quality decision result does not satisfy the frozen threshold")


def representative_production_qualification_report_projection(
    report: RepresentativeProductionQualificationReport,
) -> dict[str, object]:
    """Return the semantic preimage for a P10 report's content-addressed digest."""

    return report.model_dump(mode="json", exclude={"report_sha256"})


def _normalize_measured_capacity_input(value: object) -> object:
    '''Restore enum-bearing capacity dataclasses from a serialized mapping.'''
    if isinstance(value, MeasuredCapacityReport):
        return value

    if not isinstance(value, Mapping):
        return value
    values: dict[str, Any] = dict(value)
    for field_name, enum_type in (
        ("measurement_status", MeasuredCapacityStatus),
        ("evidence_class", CapacityEvidenceClass),
        ("provider_mode", ProviderMode),
    ):
        raw_value = values.get(field_name)
        if isinstance(raw_value, str):
            values[field_name] = enum_type(raw_value)
    unavailable_reasons = values.get("unavailable_reasons")
    if isinstance(unavailable_reasons, (list, tuple)):
        values["unavailable_reasons"] = tuple(unavailable_reasons)
    return MeasuredCapacityReport(**values)


def _normalize_service_capacity_input(value: object) -> RepresentativeServiceCapacityEvidence:
    """Restore nested capacity records without allowing scalar coercion."""

    if isinstance(value, RepresentativeServiceCapacityEvidence):
        return value
    if not isinstance(value, Mapping):
        return TypeAdapter(RepresentativeServiceCapacityEvidence).validate_python(
            value,
            strict=True,
        )
    values: dict[str, Any] = dict(value)
    values["capacity"] = _normalize_measured_capacity_input(values.get("capacity"))
    raw_queues = values.get("queues")
    if isinstance(raw_queues, (tuple, list)):
        values["queues"] = tuple(
            TypeAdapter(RepresentativeQueueObservation).validate_python(
                queue,
                strict=True,
            )
            for queue in raw_queues
        )
    return TypeAdapter(RepresentativeServiceCapacityEvidence).validate_python(
        values,
        strict=True,
    )

def _reject_lossy_scalar_coercion(
    raw: object,
    normalized: object,
    *,
    path: str = "provider saturation",
) -> None:
    """Reject raw mappings whose scalar values need permissive coercion."""

    if isinstance(normalized, StrictModel):
        normalized = normalized.model_dump(mode="json")
    if isinstance(raw, Mapping) and isinstance(normalized, Mapping):
        for key, raw_value in raw.items():
            if key in normalized:
                _reject_lossy_scalar_coercion(
                    raw_value,
                    normalized[key],
                    path=f"{path}.{key}",
                )
        return
    if isinstance(raw, (tuple, list)) and isinstance(normalized, (tuple, list)):
        for index, raw_value in enumerate(raw):
            if index < len(normalized):
                _reject_lossy_scalar_coercion(
                    raw_value,
                    normalized[index],
                    path=f"{path}[{index}]",
                )
        return
    if type(raw) is bool and type(normalized) in (int, float):
        raise ValueError(f"{path} cannot coerce a boolean into a numeric value")
    if isinstance(raw, str) and type(normalized) in (bool, int, float):
        raise ValueError(f"{path} cannot coerce a string into a scalar value")
    if isinstance(raw, float) and type(normalized) is int:
        raise ValueError(f"{path} cannot coerce a floating point value into an integer")
    if type(raw) is int and type(normalized) is bool:
        raise ValueError(f"{path} cannot coerce an integer into a boolean value")


def _normalize_provider_saturation_input(
    value: object,
) -> TwoH100ProviderQualificationReport:
    """Normalize a provider report mapping before running its invariant checks."""

    if not isinstance(value, Mapping):
        report = TypeAdapter(TwoH100ProviderQualificationReport).validate_python(
            value,
            strict=False,
        )
        _reject_lossy_scalar_coercion(value, report)
        return report
    values: dict[str, Any] = dict(value)
    raw_configuration = values.get('configuration')
    if isinstance(raw_configuration, Mapping):
        values['configuration'] = TwoH100ProviderConfiguration.model_validate(
            raw_configuration,
            strict=False,
        )
    for field_name, model_type in (
        ("endpoint_config", RunPodEndpointConfig),
        ("capabilities", ModelCapabilities),
        ("retry_policy", RunPodRetryPolicy),
    ):
        raw_value = values.get(field_name)
        if isinstance(raw_value, Mapping):
            values[field_name] = model_type.model_validate(raw_value, strict=False)
    raw_points = values.get("points")
    if isinstance(raw_points, (tuple, list)):
        normalized_points: list[object] = []
        for raw_point in raw_points:
            if isinstance(raw_point, Mapping):
                point_values: dict[str, Any] = dict(raw_point)
                point_values["capacity"] = _normalize_measured_capacity_input(
                    point_values.get("capacity")
                )
                normalized_points.append(
                    ProviderSaturationPoint.model_validate(point_values, strict=False)
                )
            else:
                normalized_points.append(
                    TypeAdapter(ProviderSaturationPoint).validate_python(
                        raw_point,
                        strict=False,
                    )
                )
        values["points"] = tuple(normalized_points)
    report = TypeAdapter(TwoH100ProviderQualificationReport).validate_python(
        values,
        strict=False,
    )
    _reject_lossy_scalar_coercion(value, report)
    return report


class RepresentativeProductionQualificationReport(StrictModel):
    """Non-promotional aggregate evidence for P10 representative qualification."""

    report_version: Literal["representative-production-qualification-v1"] = (
        _PRODUCTION_QUALIFICATION_REPORT_VERSION
    )
    report_sha256: Sha256Digest
    scope: ProductionQualificationScope
    service_capacity: RepresentativeServiceCapacityEvidence
    provider_saturation: TwoH100ProviderQualificationReport
    preferred_operating_point_run_namespace: NonEmptyString
    saturation_supporting_point_run_namespace: NonEmptyString
    quality: GovernedQualityQualificationEvidence
    recovery_evidence: tuple[CanonicalRecoveryQualificationEvidence, ...] = Field(min_length=10)
    quality_coverage: RepresentativeQualityCoverageEvidence
    deadlines: RepresentativeDeadlineEvidence
    cost_resources: RepresentativeCostResourceEvidence
    operations: RepresentativeOperationsEvidence
    external_gates: tuple[ExternalQualificationGateEvidence, ...] = Field(min_length=7)
    technical_requirements_satisfied: bool
    evidence_class: Literal["PRODUCTION_QUALIFICATION"] = "PRODUCTION_QUALIFICATION"
    production_eligible: Literal[False] = False
    qualification_status: Literal["PENDING_GOVERNED_QUALIFICATION_GATEWAY"] = (
        "PENDING_GOVERNED_QUALIFICATION_GATEWAY"
    )

    @classmethod
    def create(
        cls,
        *,
        scope: ProductionQualificationScope,
        service_capacity: RepresentativeServiceCapacityEvidence,
        provider_saturation: TwoH100ProviderQualificationReport,
        preferred_operating_point_run_namespace: str,
        saturation_supporting_point_run_namespace: str,
        quality: GovernedQualityQualificationEvidence,
        recovery_evidence: tuple[CanonicalRecoveryQualificationEvidence, ...],
        quality_coverage: RepresentativeQualityCoverageEvidence,
        deadlines: RepresentativeDeadlineEvidence,
        cost_resources: RepresentativeCostResourceEvidence,
        operations: RepresentativeOperationsEvidence,
        external_gates: tuple[ExternalQualificationGateEvidence, ...],
    ) -> Self:
        """Build and validate a content-addressed P10 evidence aggregate."""

        checked_scope = TypeAdapter(ProductionQualificationScope).validate_python(
            scope,
            strict=True,
        )
        checked_service_capacity = _normalize_service_capacity_input(service_capacity)
        checked_provider_saturation = _normalize_provider_saturation_input(provider_saturation)
        checked_quality = TypeAdapter(GovernedQualityQualificationEvidence).validate_python(
            quality,
            strict=True,
        )
        checked_recovery_evidence = TypeAdapter(
            tuple[CanonicalRecoveryQualificationEvidence, ...]
        ).validate_python(recovery_evidence, strict=True)
        checked_quality_coverage = TypeAdapter(
            RepresentativeQualityCoverageEvidence
        ).validate_python(quality_coverage, strict=True)
        checked_deadlines = TypeAdapter(RepresentativeDeadlineEvidence).validate_python(
            deadlines,
            strict=True,
        )
        checked_cost_resources = TypeAdapter(RepresentativeCostResourceEvidence).validate_python(
            cost_resources,
            strict=True,
        )
        checked_operations = TypeAdapter(RepresentativeOperationsEvidence).validate_python(
            operations,
            strict=True,
        )
        checked_external_gates = TypeAdapter(
            tuple[ExternalQualificationGateEvidence, ...]
        ).validate_python(external_gates, strict=True)
        checked_preferred_namespace = TypeAdapter(NonEmptyString).validate_python(
            preferred_operating_point_run_namespace,
            strict=True,
        )
        technical_requirements_satisfied = all(
            gate.status == "MEASURED_PENDING_INDEPENDENT_REVIEW"
            for gate in checked_external_gates
            if gate.gate_id != "E6"
        )
        checked_saturation_namespace = TypeAdapter(NonEmptyString).validate_python(
            saturation_supporting_point_run_namespace,
            strict=True,
        )

        draft = cls.model_construct(
            report_version=_PRODUCTION_QUALIFICATION_REPORT_VERSION,
            report_sha256="0" * 64,
            scope=checked_scope,
            service_capacity=checked_service_capacity,
            provider_saturation=checked_provider_saturation,
            preferred_operating_point_run_namespace=checked_preferred_namespace,
            saturation_supporting_point_run_namespace=checked_saturation_namespace,
            quality=checked_quality,
            recovery_evidence=checked_recovery_evidence,
            quality_coverage=checked_quality_coverage,
            deadlines=checked_deadlines,
            cost_resources=checked_cost_resources,
            operations=checked_operations,
            external_gates=checked_external_gates,
            technical_requirements_satisfied=technical_requirements_satisfied,
            evidence_class="PRODUCTION_QUALIFICATION",
            production_eligible=False,
            qualification_status="PENDING_GOVERNED_QUALIFICATION_GATEWAY",
        )
        report_sha256 = semantic_sha256(
            representative_production_qualification_report_projection(draft)
        )
        return cls(
            report_version=_PRODUCTION_QUALIFICATION_REPORT_VERSION,
            report_sha256=report_sha256,
            scope=checked_scope,
            service_capacity=checked_service_capacity,
            provider_saturation=checked_provider_saturation,
            preferred_operating_point_run_namespace=checked_preferred_namespace,
            saturation_supporting_point_run_namespace=checked_saturation_namespace,
            quality=checked_quality,
            recovery_evidence=checked_recovery_evidence,
            quality_coverage=checked_quality_coverage,
            deadlines=checked_deadlines,
            cost_resources=checked_cost_resources,
            operations=checked_operations,
            external_gates=checked_external_gates,
            technical_requirements_satisfied=technical_requirements_satisfied,
            evidence_class="PRODUCTION_QUALIFICATION",
            production_eligible=False,
            qualification_status="PENDING_GOVERNED_QUALIFICATION_GATEWAY",
        )

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        self.scope._assert_scope_digest()
        self.service_capacity._assert_service_capacity()
        self.provider_saturation.validate_report()  # type: ignore[operator]
        self.quality._assert_governed_quality()
        self.quality_coverage._assert_coverage()
        self.deadlines._assert_deadlines()
        self.cost_resources._assert_total_cost()
        self._validate_operating_evidence()
        self._validate_configuration_scope()
        self._validate_provider_operating_points()
        self._validate_recovery_evidence()
        expected_sha256 = semantic_sha256(
            representative_production_qualification_report_projection(self)
        )
        if self.report_sha256 != expected_sha256:
            raise ValueError("report_sha256 does not match the qualification report projection")
        return self

    def _validate_configuration_scope(self) -> None:
        configuration = self.provider_saturation.configuration
        if self.scope.provider_configuration_digest != configuration.configuration_digest:
            raise ValueError("provider configuration does not match qualification scope")
        if self.service_capacity.qualification_scope_sha256 != self.scope.scope_sha256:
            raise ValueError("service capacity scope does not match qualification scope")
        for field_name in (
            "code_revision_sha256",
            "schema_catalog_sha256",
            "sampler_policy_sha256",
            "media_manifest_sha256",
            "calibration_profile_sha256",
            "preprocess_policy_sha256",
        ):
            if getattr(self.service_capacity, field_name) != getattr(self.scope, field_name):
                raise ValueError(
                    f"service capacity {field_name} does not match qualification scope"
                )
        if (
            self.service_capacity.arrival_distribution_sha256
            != self.scope.arrival_distribution_sha256
        ):
            raise ValueError("service capacity arrival distribution does not match scope")
        if self.service_capacity.run_namespace != self.scope.qualification_run_namespace:
            raise ValueError("service capacity run namespace does not match qualification scope")
        if self.scope.workload_manifest_digest != configuration.workload_manifest_digest:
            raise ValueError("provider saturation workload does not match qualification scope")
        if (
            self.service_capacity.capacity.workload_fingerprint
            != self.scope.workload_manifest_digest
        ):
            raise ValueError("service capacity workload does not match qualification scope")
        if (
            self.service_capacity.provider_configuration_digest
            != configuration.configuration_digest
        ):
            raise ValueError("service capacity configuration does not match provider saturation")
        if (
            self.scope.provider_adapter_version
            != configuration.endpoint_configuration.adapter_version
        ):
            raise ValueError("provider adapter version does not match provider saturation")
        if (
            self.service_capacity.storage_adapter_version != self.scope.storage_adapter_version
            or self.service_capacity.storage_configuration_sha256
            != self.scope.storage_configuration_sha256
        ):
            raise ValueError("storage configuration does not match qualification scope")
        if (
            self.service_capacity.broker_adapter_version != self.scope.broker_adapter_version
            or self.service_capacity.broker_configuration_sha256
            != self.scope.broker_configuration_sha256
        ):
            raise ValueError("broker configuration does not match qualification scope")
        if self.quality.qualification_scope_sha256 != self.scope.scope_sha256:
            raise ValueError("quality scope does not match qualification scope")
        for field_name in (
            "code_revision_sha256",
            "schema_catalog_sha256",
            "sampler_policy_sha256",
            "media_manifest_sha256",
            "calibration_profile_sha256",
            "preprocess_policy_sha256",
        ):
            if getattr(self.quality, field_name) != getattr(self.scope, field_name):
                raise ValueError(
                    f"quality {field_name} does not match qualification scope"
                )
        if self.scope.benchmark_manifest_digest != self.quality.context.benchmark_manifest_digest:
            raise ValueError("quality context benchmark does not match qualification scope")
        if self.scope.governed_corpus_digest != self.quality.context.governed_corpus_digest:
            raise ValueError("quality governed corpus does not match qualification scope")
        if (
            self.scope.ground_truth_manifest_digest
            != self.quality.context.ground_truth_manifest_digest
        ):
            raise ValueError("quality label manifest does not match qualification scope")
        if (
            self.scope.grouped_split_manifest_digest
            != self.quality.context.grouped_split_manifest_digest
        ):
            raise ValueError("quality split manifest does not match qualification scope")
        if (
            self.provider_saturation.evidence_class
            is not CapacityEvidenceClass.PRODUCTION_QUALIFICATION
        ):
            raise ValueError("provider saturation requires PRODUCTION_QUALIFICATION evidence")
        for point in self.provider_saturation.points:
            gpu = point.telemetry.gpu
            if (
                gpu.hardware_inventory_artifact_uri != self.scope.host_resource_inventory_uri
                or gpu.hardware_inventory_sha256 != self.scope.host_resource_inventory_sha256
            ):
                raise ValueError("provider GPU inventory does not match qualification scope")

    def _validate_operating_evidence(self) -> None:
        """Bind P10 quality, delivery, cost, and operations facts to the frozen run."""

        for label, evidence in (
            ("quality coverage", self.quality_coverage),
            ("deadline", self.deadlines),
            ("cost/resource", self.cost_resources),
            ("operations", self.operations),
        ):
            if evidence.qualification_scope_sha256 != self.scope.scope_sha256:
                raise ValueError(f"{label} scope does not match qualification scope")
            if evidence.run_namespace != self.scope.qualification_run_namespace:
                raise ValueError(f"{label} run namespace does not match qualification scope")
        if not isclose(
            self.cost_resources.recording_hours,
            self.service_capacity.arrival_recording_hours,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "cost/resource recording-hours do not match the service arrival observation"
            )
        if not isclose(
            self.cost_resources.camera_hours,
            self.service_capacity.arrival_camera_hours,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "cost/resource camera-hours do not match the service arrival observation"
            )
        if self.deadlines.qa_observation_count < self.service_capacity.arrival_recording_count:
            raise ValueError("deadline QA observations do not cover all arrived recordings")
        if (
            self.deadlines.annotation_observation_count
            < self.service_capacity.arrival_recording_count
        ):
            raise ValueError("deadline annotation observations do not cover all arrived recordings")
        expected_gate_ids = (
            "E0",
            "E1",
            "E2",
            "E3",
            "E4",
            "E5",
            "E6",
        )
        gate_ids = tuple(gate.gate_id for gate in self.external_gates)
        if gate_ids != expected_gate_ids:
            raise ValueError("external gates must retain E0 through E6 in order")
        if self.external_gates[-1].status != "PENDING_INDEPENDENT_REVIEW":
            raise ValueError("E6 must remain pending independent review")
        expected_technical_requirements_satisfied = all(
            gate.status == "MEASURED_PENDING_INDEPENDENT_REVIEW"
            for gate in self.external_gates[:-1]
        )
        if self.technical_requirements_satisfied != expected_technical_requirements_satisfied:
            raise ValueError(
                "technical requirements state does not match the external gate evidence"
            )


    def _validate_provider_operating_points(self) -> None:
        if (
            self.preferred_operating_point_run_namespace
            == self.saturation_supporting_point_run_namespace
        ):
            raise ValueError("preferred and saturation points must be distinct executions")
        points_by_namespace = {
            point.run_namespace: point for point in self.provider_saturation.points
        }
        try:
            preferred = points_by_namespace[self.preferred_operating_point_run_namespace]
            saturation = points_by_namespace[self.saturation_supporting_point_run_namespace]
        except KeyError as error:
            raise ValueError(
                "named provider operating point is absent from the saturation report"
            ) from error
        self._validate_preferred_operating_point(preferred)
        self._validate_saturation_supporting_point(saturation)

    @staticmethod
    def _recording_rtf(point: ProviderSaturationPoint) -> float:
        recording_rtf = point.capacity.recording_hours_per_wall_hour
        if recording_rtf is None:
            raise ValueError("provider operating point lacks recording-RTF capacity")
        return recording_rtf

    @classmethod
    def _validate_preferred_operating_point(cls, point: ProviderSaturationPoint) -> None:
        if not point.safe_envelope:
            raise ValueError("preferred operating point must be inside the safe provider envelope")
        if cls._recording_rtf(point) < _NOMINAL_500_HOURS_PER_DAY_RTF:
            raise ValueError("preferred operating point cannot sustain nominal 500 h/day")
        if point.telemetry.gpu.gpu_utilization_fraction > _MAXIMUM_NOMINAL_GPU_UTILIZATION:
            raise ValueError("preferred operating point exceeds 70 percent average GPU utilization")
        if point.aggregate_gpu_minutes_per_recording_hour > _MAXIMUM_GPU_MINUTES_PER_RECORDING_HOUR:
            raise ValueError("preferred operating point exceeds the 4.03 GPU-minute budget")

    @classmethod
    def _validate_saturation_supporting_point(cls, point: ProviderSaturationPoint) -> None:
        if not point.safe_envelope:
            raise ValueError(
                "saturation supporting point must be inside the safe provider envelope"
            )
        if cls._recording_rtf(point) < _PREFERRED_SATURATION_RECORDING_RTF:
            raise ValueError("saturation point must sustain at least 29.762 recording-RTF")

    def _validate_recovery_evidence(self) -> None:
        scenarios = tuple(item.scenario for item in self.recovery_evidence)
        if scenarios != tuple(CanonicalRecoveryScenario):
            raise ValueError("qualification requires each canonical recovery scenario exactly once")
        if len({item.scenario_evidence_sha256 for item in self.recovery_evidence}) != len(
            self.recovery_evidence
        ):
            raise ValueError("canonical recovery scenarios require distinct evidence artifacts")
        if len({item.fresh.run_id for item in self.recovery_evidence}) != len(
            self.recovery_evidence
        ):
            raise ValueError("canonical recovery scenarios require distinct executions")
        if not any(item.outbox_delivery_count for item in self.recovery_evidence):
            raise ValueError("qualification requires observed outbox delivery")
        if not any(item.review_task_count for item in self.recovery_evidence):
            raise ValueError("qualification requires observed review routing")
        for item in self.recovery_evidence:
            item._assert_recovery_closure()
            if item.qualification_scope_sha256 != self.scope.scope_sha256:
                raise ValueError("canonical recovery scope does not match qualification scope")
            if item.run_namespace != self.scope.qualification_run_namespace:
                raise ValueError(
                    "canonical recovery run namespace does not match qualification scope"
                )
            if item.workload_fingerprint != self.scope.workload_manifest_digest:
                raise ValueError("canonical recovery workload does not match qualification scope")
            if item.evidence_class is not CanonicalRecoveryEvidenceClass.PRODUCTION_QUALIFICATION:
                raise ValueError("canonical recovery requires production qualification provenance")
            if not item.recovery_completed:
                raise ValueError("canonical recovery scenario did not complete")
            if (
                item.duplicate_terminal_count
                or item.duplicate_outbox_delivery_count
                or item.duplicate_review_task_count
            ):
                raise ValueError("canonical recovery scenario observed duplicate durable delivery")
        soak = self.recovery_evidence[-1]
        if (
            soak.scenario is not CanonicalRecoveryScenario.SOAK
            or soak.run_duration_ns < 24 * _HOUR_NS
        ):
            raise ValueError("canonical soak evidence must cover at least 24 hours")


    def as_dict(self) -> dict[str, object]:
        """Return the complete JSON-ready, non-promotional P10 report."""

        return self.model_dump(mode="json")

    def render_markdown(self) -> str:
        """Render one frozen P10 operating-gate report without self-promotion."""

        capacity = self.service_capacity.capacity
        recording_rtf = capacity.recording_hours_per_wall_hour
        camera_rtf = capacity.camera_hours_per_wall_hour
        lines = [
            "# Representative production qualification report",
            "",
            f"- Report SHA-256: {self.report_sha256}",
            f"- Qualification scope SHA-256: {self.scope.scope_sha256}",
            f"- Run namespace: {self.scope.qualification_run_namespace}",
            f"- Evidence class: {self.evidence_class}",
            f"- Qualification status: {self.qualification_status}",
            "- Production eligible: NO",
            "",
            "## Quality coverage",
            f"- QA classes: {len(self.quality_coverage.qa_class_ids)}",
            f"- Media matrix entries: {len(self.quality_coverage.media_profiles)}",
            f"- Geometry views: {", ".join(self.quality_coverage.preprocess_views)}",
            "",
            "## Capacity and deadlines",
            (
                "- Recording RTF: NOT_AVAILABLE"
                if recording_rtf is None
                else f"- Recording RTF: {recording_rtf:.6f}"
            ),
            (
                "- Camera RTF: NOT_AVAILABLE"
                if camera_rtf is None
                else f"- Camera RTF: {camera_rtf:.6f}"
            ),
            f"- QA P99 (ns): {self.deadlines.qa_completion_p99_ns}",
            f"- Annotation P99 (ns): {self.deadlines.annotation_completion_p99_ns}",
            "",
            "## Cost and resources",
            (
                f"- Total cost ({self.cost_resources.currency}): "
                f"{self.cost_resources.total_cost_usd:.6f}"
            ),
            f"- CPU seconds: {self.cost_resources.cpu_seconds:.6f}",
            f"- GPU seconds: {self.cost_resources.gpu_seconds:.6f}",
            f"- NVMe read bytes: {self.cost_resources.nvme_read_bytes}",
            f"- NVMe write bytes: {self.cost_resources.nvme_write_bytes}",
            "",
            "## External gates",
        ]
        lines.extend(
            f"- {gate.gate_id}: {gate.status} ({gate.unresolved_reason})"
            for gate in self.external_gates
        )
        return "\n".join(lines) + "\n"


__all__ = [
    "ExternalQualificationGateEvidence",
    "GovernedQualityQualificationEvidence",
    "LocalQualificationContext",
    "LocalQualityCapacityQualificationPackage",
    "LocalRecoveryScenario",
    "ProductionQualificationScope",
    "RepresentativeCostResourceEvidence",
    "RepresentativeDeadlineEvidence",
    "RepresentativeMediaProfile",
    "RepresentativeOperationsEvidence",
    "RepresentativeProductionQualificationReport",
    "RepresentativeQualityCoverageEvidence",
    "RepresentativeQueueObservation",
    "RepresentativeServiceCapacityEvidence",
    "build_local_quality_capacity_qualification_package",
    "representative_production_qualification_report_projection",
]
