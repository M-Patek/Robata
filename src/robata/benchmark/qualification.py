"""Bound local quality/capacity qualification artifacts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.benchmark.pareto import LocalSamplingDenseParetoReport
from robata.contracts.common import Nanoseconds, Sha256Digest, StrictModel
from robata.runtime.capacity import CapacityEvidenceClass, MeasuredCapacityReport, MeasuredCapacityStatus, ProviderMode

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

    scenario_id: Literal["RESTART_REPLAY", "PROVIDER_RETRY", "PROVIDER_TIMEOUT", "OUTBOX_RECONCILIATION"]
    terminal_reconciled: bool
    outbox_reconciled: bool

    @model_validator(mode="after")
    def validate_reconciliation(self) -> Self:
        if not self.terminal_reconciled or not self.outbox_reconciled:
            raise ValueError("local recovery scenarios must reconcile terminal and outbox state")
        return self


class LocalQualityCapacityQualificationPackage(StrictModel):
    """One reproducible local P8 artifact, explicitly not production evidence."""

    package_version: Literal["local-quality-capacity-qualification-v1"] = "local-quality-capacity-qualification-v1"
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


__all__ = ["LocalQualificationContext", "LocalQualityCapacityQualificationPackage", "LocalRecoveryScenario"]
