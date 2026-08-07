"""Lightweight, workload-bound qualification for a two-model experiment.

This module reports evidence and comparability only. It does not choose a
model, authorize a route, or replace the provider saturation collector.
Published wire schemas remain unchanged while the first RunPod comparison is
an internal engineering artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, model_validator

from robata.benchmark.evidence import BenchmarkEvidenceContext
from robata.benchmark.metrics import (
    BoundaryMetrics,
    CalibrationMetrics,
    EventMetrics,
    QAMetrics,
)
from robata.benchmark.provider_qualification import TwoH100ProviderQualificationReport
from robata.contracts.common import Sha256Digest, StrictModel
from robata.inference.experiment_execution import (
    ExperimentComparisonStatus,
    ExperimentPairComparison,
)
from robata.inference.models import InferenceStatus
from robata.inference.routing import (
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
    ModelDeployment,
    ModelRouteRole,
    endpoint_config_digest,
)
from robata.runtime.capacity import (
    MeasuredCapacityComparison,
    compare_measured_capacity_reports,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
MetricValue = QAMetrics | EventMetrics | BoundaryMetrics | CalibrationMetrics


def _revalidate_strict_model(value: StrictModel) -> None:
    """Reconstruct a frozen evidence tree to defeat shallow ``model_copy`` updates."""

    for nested in value.__dict__.values():
        _revalidate_nested_value(nested)
    type(value)(**value.__dict__)


def _revalidate_nested_value(value: object) -> None:
    if isinstance(value, StrictModel):
        _revalidate_strict_model(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            _revalidate_nested_value(nested)
    elif isinstance(value, (tuple, list, frozenset, set)):
        for nested in value:
            _revalidate_nested_value(nested)


def _endpoint_identity(endpoint_url: object) -> tuple[str, str, int, str, str]:
    """Normalize an HTTP endpoint enough to reject same-endpoint placement claims."""

    parsed = urlsplit(str(endpoint_url))
    if not parsed.scheme or parsed.hostname is None:
        raise ValueError("endpoint URL must have a scheme and host")
    scheme = parsed.scheme.lower()
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("endpoint URL has an invalid port") from error
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else -1
    return (
        scheme,
        parsed.hostname.lower().rstrip("."),
        port,
        parsed.path or "/",
        parsed.query,
    )


class FairLoadQualificationStatus(StrEnum):
    """How far a two-model report may be interpreted without promotion."""

    FAIR_LOAD_COMPARABLE = "FAIR_LOAD_COMPARABLE"
    CONTENTION_ONLY = "CONTENTION_ONLY"
    OBSERVATION_ONLY = "OBSERVATION_ONLY"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ExternalArtifactReference(StrictModel):
    """A content-addressed external artifact kept outside this local report."""

    uri: NonEmptyString
    sha256: Sha256Digest


class ExecutionPlacementEvidence(StrictModel):
    """Non-secret placement identity and its content-addressed evidence."""

    worker_pool_identity: NonEmptyString
    artifact: ExternalArtifactReference

    @model_validator(mode="after")
    def validate_placement(self) -> Self:
        _revalidate_strict_model(self.artifact)
        return self


class DeploymentRunManifest(StrictModel):
    """Content-addressed manifest binding one deployment's fair-load inputs."""

    artifact: ExternalArtifactReference
    endpoint_config_digest: Sha256Digest
    handler_image: ExternalArtifactReference
    source_package_digests: tuple[Sha256Digest, ...] = Field(min_length=1)
    workload_manifest_sha256: Sha256Digest
    arrival_schedule_sha256: Sha256Digest
    placement: ExecutionPlacementEvidence

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _revalidate_strict_model(self.artifact)
        _revalidate_strict_model(self.handler_image)
        _revalidate_strict_model(self.placement)
        if self.source_package_digests != tuple(sorted(set(self.source_package_digests))):
            raise ValueError("source_package_digests must be unique and ordered")
        return self


class ModelCostEvidence(StrictModel):
    """Optional per-deployment cost evidence without estimating an unknown bill."""

    measurement_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED"
    currency: NonEmptyString | None = None
    cost_per_recording_hour: NonNegativeFloat | None = None
    billing_artifact: ExternalArtifactReference | None = None

    @model_validator(mode="after")
    def validate_cost_measurement(self) -> Self:
        if self.billing_artifact is not None:
            _revalidate_strict_model(self.billing_artifact)
        values = (self.currency, self.cost_per_recording_hour, self.billing_artifact)
        if self.measurement_status == "MEASURED":
            if any(value is None for value in values):
                raise ValueError("MEASURED cost evidence requires currency, rate, and bill")
        elif any(value is not None for value in values):
            raise ValueError("NOT_MEASURED cost evidence cannot retain measured values")
        return self


class ModelQualityEvidence(StrictModel):
    """Quality metrics tied to frozen labels and completed P17 observations."""

    measurement_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED"
    evidence_context: BenchmarkEvidenceContext | None = None
    quality_artifact: ExternalArtifactReference | None = None
    source_package_digests: tuple[Sha256Digest, ...] = ()
    paired_observations: tuple[ExperimentPairComparison, ...] = ()
    human_adjudication: ExternalArtifactReference | None = None
    qa_metrics: QAMetrics | None = None
    event_metrics: EventMetrics | None = None
    boundary_metrics: BoundaryMetrics | None = None
    calibration_metrics: CalibrationMetrics | None = None

    @property
    def metrics(self) -> tuple[MetricValue, ...]:
        return tuple(
            metric
            for metric in (
                self.qa_metrics,
                self.event_metrics,
                self.boundary_metrics,
                self.calibration_metrics,
            )
            if metric is not None
        )

    @property
    def metric_coverage(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, metric in (
                ("QA", self.qa_metrics),
                ("EVENT", self.event_metrics),
                ("BOUNDARY", self.boundary_metrics),
                ("CALIBRATION", self.calibration_metrics),
            )
            if metric is not None
        )

    @property
    def metric_sample_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (name, metric.sample_count)
            for name, metric in (
                ("QA", self.qa_metrics),
                ("EVENT", self.event_metrics),
                ("BOUNDARY", self.boundary_metrics),
                ("CALIBRATION", self.calibration_metrics),
            )
            if metric is not None
        )

    @model_validator(mode="after")
    def validate_quality_measurement(self) -> Self:
        for value in (self.evidence_context, self.quality_artifact, self.human_adjudication):
            if value is not None:
                _revalidate_strict_model(value)
        for observation in self.paired_observations:
            _revalidate_strict_model(observation)
        if self.source_package_digests != tuple(sorted(set(self.source_package_digests))):
            raise ValueError("quality source_package_digests must be unique and ordered")
        if self.measurement_status == "NOT_MEASURED":
            values = (
                self.evidence_context,
                self.quality_artifact,
                self.source_package_digests,
                self.paired_observations,
            )
            if any(value for value in values):
                raise ValueError("NOT_MEASURED quality cannot retain measured provenance")
            if any(metric.measurement_status == "MEASURED" for metric in self.metrics):
                raise ValueError("NOT_MEASURED quality cannot retain measured metrics")
            return self

        context = self.evidence_context
        if (
            context is None
            or self.quality_artifact is None
            or not self.source_package_digests
            or not self.paired_observations
            or not self.metrics
        ):
            raise ValueError(
                "MEASURED quality requires labels, artifact, source packages, "
                "observations, and metrics"
            )
        expected_identity = context.context_identity
        expected_digest = context.context_digest
        policy_identity: str | None = None
        policy_digest: str | None = None
        policy_version: str | None = None
        for metric in self.metrics:
            _revalidate_strict_model(metric)
            if metric.measurement_status != "MEASURED":
                raise ValueError("MEASURED quality requires measured metric values")
            if (
                metric.evidence_context_identity != expected_identity
                or metric.evidence_context_digest != expected_digest
            ):
                raise ValueError("quality metric does not match its evidence context")
            current_policy = (
                metric.metric_policy_identity,
                metric.metric_policy_digest,
                metric.metric_policy_version,
            )
            if policy_identity is None:
                policy_identity, policy_digest, policy_version = current_policy
            elif current_policy != (policy_identity, policy_digest, policy_version):
                raise ValueError("quality metrics must use one metric policy")
        comparison_ids = tuple(item.comparison_id for item in self.paired_observations)
        if len(set(comparison_ids)) != len(comparison_ids):
            raise ValueError("quality observations must not repeat a P17 comparison")
        input_identities = tuple(item.input_identity_sha256 for item in self.paired_observations)
        if len(set(input_identities)) != len(input_identities):
            raise ValueError("quality observations must not repeat an experiment input")
        return self


class SaturationOutcomeSummary(StrictModel):
    """Optional schema/failure classification for one P6 saturation point."""

    offered_concurrency: NonNegativeInt
    measurement_status: Literal["NOT_MEASURED", "MEASURED"] = "NOT_MEASURED"
    terminal_artifact: ExternalArtifactReference | None = None
    schema_valid_terminal_count: NonNegativeInt | None = None
    schema_invalid_terminal_count: NonNegativeInt | None = None
    timeout_terminal_count: NonNegativeInt | None = None
    provider_error_terminal_count: NonNegativeInt | None = None

    @property
    def accepted_terminal_count(self) -> int | None:
        if self.schema_valid_terminal_count is None or self.schema_invalid_terminal_count is None:
            return None
        return self.schema_valid_terminal_count + self.schema_invalid_terminal_count

    @property
    def rejected_terminal_count(self) -> int | None:
        if self.timeout_terminal_count is None or self.provider_error_terminal_count is None:
            return None
        return self.timeout_terminal_count + self.provider_error_terminal_count

    @property
    def terminal_response_count(self) -> int | None:
        accepted = self.accepted_terminal_count
        rejected = self.rejected_terminal_count
        if accepted is None or rejected is None:
            return None
        return accepted + rejected

    @property
    def schema_validity_rate(self) -> float | None:
        accepted = self.accepted_terminal_count
        if accepted is None or accepted == 0 or self.schema_valid_terminal_count is None:
            return None
        return self.schema_valid_terminal_count / accepted

    @property
    def timeout_rate(self) -> float | None:
        terminal = self.terminal_response_count
        if terminal is None or terminal == 0 or self.timeout_terminal_count is None:
            return None
        return self.timeout_terminal_count / terminal

    @model_validator(mode="after")
    def validate_outcome_measurement(self) -> Self:
        if self.terminal_artifact is not None:
            _revalidate_strict_model(self.terminal_artifact)
        values = (
            self.terminal_artifact,
            self.schema_valid_terminal_count,
            self.schema_invalid_terminal_count,
            self.timeout_terminal_count,
            self.provider_error_terminal_count,
        )
        if self.measurement_status == "MEASURED":
            if any(value is None for value in values):
                raise ValueError("MEASURED outcomes require an artifact and every terminal count")
        elif any(value is not None for value in values):
            raise ValueError("NOT_MEASURED outcomes cannot retain classified counts")
        return self


class FairLoadDeploymentEvidence(StrictModel):
    """All independently attributable evidence for one model deployment."""

    deployment: ModelDeployment
    saturation: TwoH100ProviderQualificationReport
    run_manifest: DeploymentRunManifest
    outcomes: tuple[SaturationOutcomeSummary, ...] = Field(min_length=2)
    quality: ModelQualityEvidence = Field(default_factory=ModelQualityEvidence)
    cost: ModelCostEvidence = Field(default_factory=ModelCostEvidence)

    @model_validator(mode="after")
    def validate_deployment_evidence(self) -> Self:
        for value in (
            self.deployment,
            self.saturation,
            self.run_manifest,
            self.quality,
            self.cost,
        ):
            _revalidate_strict_model(value)
        for outcome in self.outcomes:
            _revalidate_strict_model(outcome)

        configuration = self.saturation.configuration
        endpoint = self.saturation.endpoint_config
        capabilities = self.saturation.capabilities
        if (
            self.deployment.provider != configuration.provider
            or self.deployment.model_name != configuration.model_identifier
            or self.deployment.model_version != configuration.model_version
            or self.deployment.adapter_version != endpoint.adapter_version
            or self.deployment.max_concurrent_requests != configuration.max_concurrent_requests
        ):
            raise ValueError("deployment does not match its saturation configuration")
        if (
            self.deployment.capability_snapshot_id != capabilities.snapshot_id
            or self.deployment.capability_snapshot_digest != capabilities.snapshot_digest
        ):
            raise ValueError("deployment does not match its capability snapshot")
        if self.deployment.endpoint_config_digest != endpoint_config_digest(endpoint):
            raise ValueError("deployment does not match its endpoint configuration")
        if self.run_manifest.endpoint_config_digest != self.deployment.endpoint_config_digest:
            raise ValueError("run manifest does not match its deployment endpoint")
        if self.run_manifest.workload_manifest_sha256 != configuration.workload_manifest_digest:
            raise ValueError("run manifest does not match its saturation workload")

        offered = tuple(point.offered_concurrency for point in self.saturation.points)
        outcome_offered = tuple(item.offered_concurrency for item in self.outcomes)
        if outcome_offered != offered:
            raise ValueError("outcome summaries must match the saturation concurrency ladder")
        for outcome, point in zip(self.outcomes, self.saturation.points, strict=True):
            if outcome.measurement_status == "NOT_MEASURED":
                continue
            telemetry = point.telemetry
            if outcome.accepted_terminal_count != telemetry.accepted_response_count:
                raise ValueError("outcome accepted count does not match provider telemetry")
            if outcome.rejected_terminal_count != telemetry.rejected_response_count:
                raise ValueError("outcome rejected count does not match provider telemetry")
        return self

    @property
    def concurrency_ladder(self) -> tuple[int, ...]:
        return tuple(point.offered_concurrency for point in self.saturation.points)

    @property
    def hardware_profile(self) -> tuple[str | int, ...]:
        gpu = self.saturation.safe_point.telemetry.gpu
        return (
            gpu.gpu_sku,
            gpu.gpu_count,
            gpu.driver_version,
            gpu.runtime_version,
            gpu.metric_source,
        )

    @property
    def runtime_configuration(self) -> tuple[str, ...]:
        configuration = self.saturation.configuration
        return (
            configuration.topology.value,
            configuration.inference_engine,
            configuration.precision_or_quantization,
        )


class FairLoadModelComparisonReport(StrictModel):
    """Two-model evidence report with no winner or production authority."""

    report_version: Literal["fair-load-model-comparison-v1"] = "fair-load-model-comparison-v1"
    contract: ExperimentContract
    control: FairLoadDeploymentEvidence
    candidate: FairLoadDeploymentEvidence
    declared_external_limits: tuple[NonEmptyString, ...] = ()
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _revalidate_strict_model(self.contract)
        _revalidate_strict_model(self.control)
        _revalidate_strict_model(self.candidate)
        if self.control.deployment != self.contract.control:
            raise ValueError("control evidence deployment does not match the experiment contract")
        if self.candidate.deployment != self.contract.candidate:
            raise ValueError("candidate evidence deployment does not match the experiment contract")
        for evidence in (self.control, self.candidate):
            manifest = evidence.run_manifest
            if manifest.arrival_schedule_sha256 != self.contract.arrival_schedule_sha256:
                raise ValueError(
                    "deployment arrival schedule does not match the experiment contract"
                )
            if manifest.workload_manifest_sha256 != self.contract.workload_manifest_sha256:
                raise ValueError("deployment workload does not match the experiment contract")
        if (
            self.control.run_manifest.source_package_digests
            != self.candidate.run_manifest.source_package_digests
        ):
            raise ValueError("model comparison requires the same source package digests")
        if self.control.concurrency_ladder != self.candidate.concurrency_ladder:
            raise ValueError("model comparison requires the same concurrency ladder")
        if self.control.saturation.evidence_class is not self.candidate.saturation.evidence_class:
            raise ValueError("model comparison requires one capacity evidence class")
        if self.control.hardware_profile != self.candidate.hardware_profile:
            raise ValueError("model comparison requires matching GPU hardware profiles")
        if self.contract.isolation_profile is ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE:
            if (
                self.control.deployment.endpoint_config_digest
                == self.candidate.deployment.endpoint_config_digest
            ):
                raise ValueError("independent model comparison requires separate endpoints")
            if _endpoint_identity(
                self.control.saturation.endpoint_config.endpoint_url
            ) == _endpoint_identity(self.candidate.saturation.endpoint_config.endpoint_url):
                raise ValueError("independent model comparison requires separate endpoint URLs")
            if (
                self.control.run_manifest.placement.worker_pool_identity
                == self.candidate.run_manifest.placement.worker_pool_identity
            ):
                raise ValueError("independent model comparison requires separate worker pools")
        if (
            self.contract.input_representation
            is ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING
            and self.control.saturation.configuration.request_contracts
            != self.candidate.saturation.configuration.request_contracts
        ):
            raise ValueError("identical rendering requires matching prompt and output contracts")
        self._validate_quality_pair()
        if self.declared_external_limits != tuple(sorted(set(self.declared_external_limits))):
            raise ValueError("declared_external_limits must be unique and ordered")
        return self

    def _validate_quality_pair(self) -> None:
        control = self.control.quality
        candidate = self.candidate.quality
        if control.measurement_status != candidate.measurement_status:
            raise ValueError("model quality evidence must have matching measurement status")
        if control.measurement_status == "NOT_MEASURED":
            return
        self._validate_quality_against_experiment(
            evidence=self.control,
            quality=control,
            role=ModelRouteRole.CONTROL,
        )
        self._validate_quality_against_experiment(
            evidence=self.candidate,
            quality=candidate,
            role=ModelRouteRole.CANDIDATE,
        )
        if control.evidence_context != candidate.evidence_context:
            raise ValueError("model quality evidence must use one frozen label context")
        if control.metric_coverage != candidate.metric_coverage:
            raise ValueError("model quality evidence must cover the same metrics")
        if control.metric_sample_counts != candidate.metric_sample_counts:
            raise ValueError("model quality evidence must use matching sample coverage")
        if control.paired_observations != candidate.paired_observations:
            raise ValueError("model quality evidence must use the same P17 observations")
        for control_metric, candidate_metric in zip(
            control.metrics,
            candidate.metrics,
            strict=True,
        ):
            if (
                control_metric.metric_policy_identity,
                control_metric.metric_policy_digest,
                control_metric.metric_policy_version,
            ) != (
                candidate_metric.metric_policy_identity,
                candidate_metric.metric_policy_digest,
                candidate_metric.metric_policy_version,
            ):
                raise ValueError("model quality evidence must use matching metric policies")

    def _validate_quality_against_experiment(
        self,
        *,
        evidence: FairLoadDeploymentEvidence,
        quality: ModelQualityEvidence,
        role: ModelRouteRole,
    ) -> None:
        if quality.source_package_digests != evidence.run_manifest.source_package_digests:
            raise ValueError("quality evidence does not match its source package manifest")
        for observation in quality.paired_observations:
            if not observation.comparable or observation.status not in {
                ExperimentComparisonStatus.AGREEMENT,
                ExperimentComparisonStatus.DIFFERENCE,
            }:
                raise ValueError("quality observation must be a comparable P17 result")
            if (
                observation.experiment_id != self.contract.experiment_id
                or observation.experiment_contract_digest != self.contract.contract_digest
                or observation.workload_manifest_sha256 != self.contract.workload_manifest_sha256
                or observation.comparison_config_sha256 != self.contract.comparison_config_sha256
                or observation.input_representation != self.contract.input_representation
            ):
                raise ValueError("quality observation does not match the experiment contract")
            if observation.control_inference_id == observation.candidate_inference_id:
                raise ValueError("quality observation cannot reuse one inference terminal")
            side = observation.control if role is ModelRouteRole.CONTROL else observation.candidate
            side_inference_id = (
                observation.control_inference_id
                if role is ModelRouteRole.CONTROL
                else observation.candidate_inference_id
            )
            if (
                side is None
                or side.role is not role
                or side.deployment_id != evidence.deployment.deployment_id
                or side.inference_id is None
                or side_inference_id != side.inference_id
            ):
                raise ValueError("quality observation does not match its model deployment")
            if side.status is not InferenceStatus.SUCCEEDED or side.output_valid is not True:
                raise ValueError("quality observation requires a successful valid model output")

    @property
    def runtime_configuration_matches(self) -> bool:
        return self.control.runtime_configuration == self.candidate.runtime_configuration

    @property
    def capacity_comparison_blockers(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.contract.isolation_profile is ExperimentIsolationProfile.COLOCATED_SHARED_HARDWARE:
            reasons.append("COLOCATED_SHARED_HARDWARE")
        if (
            self.contract.input_representation
            is ExperimentInputRepresentation.MODEL_SPECIFIC_RENDERING
        ):
            reasons.append("MODEL_SPECIFIC_RENDERING")
        return tuple(sorted(reasons))

    @property
    def capacity_comparisons(self) -> tuple[MeasuredCapacityComparison, ...]:
        return tuple(
            compare_measured_capacity_reports(
                control_point.capacity,
                candidate_point.capacity,
                additional_non_comparable_reasons=self.capacity_comparison_blockers,
            )
            for control_point, candidate_point in zip(
                self.control.saturation.points,
                self.candidate.saturation.points,
                strict=True,
            )
        )

    @property
    def capacity_comparison_eligible(self) -> bool:
        return all(item.comparable for item in self.capacity_comparisons)

    @property
    def quality_metrics_comparison_eligible(self) -> bool:
        return (
            self.capacity_comparison_eligible
            and self.control.quality.measurement_status == "MEASURED"
            and self.candidate.quality.measurement_status == "MEASURED"
        )

    @property
    def qualification_status(self) -> FairLoadQualificationStatus:
        if self.contract.isolation_profile is ExperimentIsolationProfile.COLOCATED_SHARED_HARDWARE:
            return FairLoadQualificationStatus.CONTENTION_ONLY
        if (
            self.contract.input_representation
            is ExperimentInputRepresentation.MODEL_SPECIFIC_RENDERING
        ):
            return FairLoadQualificationStatus.OBSERVATION_ONLY
        if self.capacity_comparison_eligible:
            return FairLoadQualificationStatus.FAIR_LOAD_COMPARABLE
        return FairLoadQualificationStatus.NOT_COMPARABLE

    @property
    def unresolved_external_limits(self) -> tuple[str, ...]:
        limits = set(self.declared_external_limits)
        if not self.runtime_configuration_matches:
            limits.add("RUNTIME_CONFIGURATION_DIFFERS")
        for role, evidence in (
            ("CONTROL", self.control),
            ("CANDIDATE", self.candidate),
        ):
            if evidence.quality.measurement_status == "NOT_MEASURED":
                limits.add(f"{role}_QUALITY_EVIDENCE_NOT_MEASURED")
            if evidence.quality.human_adjudication is None:
                limits.add(f"{role}_HUMAN_ADJUDICATION_NOT_RECORDED")
            if evidence.cost.measurement_status == "NOT_MEASURED":
                limits.add(f"{role}_COST_EVIDENCE_NOT_MEASURED")
            if any(item.measurement_status == "NOT_MEASURED" for item in evidence.outcomes):
                limits.add(f"{role}_OUTCOME_CLASSIFICATION_NOT_MEASURED")
        return tuple(sorted(limits))


__all__ = [
    "DeploymentRunManifest",
    "ExecutionPlacementEvidence",
    "ExternalArtifactReference",
    "FairLoadDeploymentEvidence",
    "FairLoadModelComparisonReport",
    "FairLoadQualificationStatus",
    "ModelCostEvidence",
    "ModelQualityEvidence",
    "SaturationOutcomeSummary",
]
