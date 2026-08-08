"""Non-canonical, process-local trace sidecars for production qualification.

The trace models in this module are deliberately outside the published schema
catalog and never participate in canonical identity, selection, or evidence
bytes.  They preserve observed runtime facts and, just as importantly, make
unobserved boundaries explicit instead of inventing zero-valued measurements.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp

if TYPE_CHECKING:
    from robata.benchmark.external_paired_qualification import (
        ExternalEndpointBinding,
        ExternalPairedQualificationReport,
        ExternalProviderObservation,
        TransportFactory,
    )

from robata.runtime.observability import (
    RuntimeProfileRecorder,
    RuntimeProfileSnapshot,
    RuntimeSpanSnapshot,
    runtime_span,
)

E2E_TRACE_VERSION: Literal["robata-e2e-trace-v1"] = "robata-e2e-trace-v1"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=256)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class E2ETraceMeasurementStatus(StrEnum):
    """Whether a value was observed at the boundary represented by a trace."""

    MEASURED = "MEASURED"
    NOT_MEASURED = "NOT_MEASURED"


class E2ETraceStage(StrEnum):
    """Stable high-level boundaries in a Robata production path."""

    ORCHESTRATION = "ORCHESTRATION"
    SOURCE = "SOURCE"
    SCHEDULING = "SCHEDULING"
    INFERENCE = "INFERENCE"
    EVIDENCE = "EVIDENCE"
    REDUCTION = "REDUCTION"
    PUBLICATION = "PUBLICATION"


class E2ETraceFragmentRole(StrEnum):
    """Process-local fragment purpose; monotonic offsets never cross fragments."""

    LAUNCHER = "LAUNCHER"
    CONTROL = "CONTROL"
    CANDIDATE = "CANDIDATE"


class E2ETraceCoverage(StrEnum):
    """Coverage state, not an indication of run or model quality."""

    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"


class E2ETraceStageMeasurement(StrictModel):
    """Union wall time for one stage within one process-local fragment."""

    stage: E2ETraceStage
    measurement_status: E2ETraceMeasurementStatus
    observed_span_count: NonNegativeInt
    wall_time_union_ns: NonNegativeInt | None = None
    inclusive_span_time_ns: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_measurement_shape(self) -> Self:
        values = (self.wall_time_union_ns, self.inclusive_span_time_ns)
        if self.measurement_status is E2ETraceMeasurementStatus.MEASURED:
            if self.observed_span_count < 1 or any(value is None for value in values):
                raise ValueError("MEASURED stage requires spans and both wall-time values")
        elif self.observed_span_count != 0 or any(value is not None for value in values):
            raise ValueError("NOT_MEASURED stage cannot retain a count or timing value")
        return self


class E2ETraceRuntimeFragment(StrictModel):
    """One frozen local recorder snapshot and its safely reconciled stage view."""

    role: E2ETraceFragmentRole
    clock_domain: Literal["PROCESS_LOCAL_MONOTONIC"] = "PROCESS_LOCAL_MONOTONIC"
    runtime_profile: RuntimeProfileSnapshot
    stages: tuple[E2ETraceStageMeasurement, ...]
    unclassified_span_count: NonNegativeInt

    @model_validator(mode="after")
    def validate_stage_coverage(self) -> Self:
        expected = tuple(E2ETraceStage)
        observed = tuple(item.stage for item in self.stages)
        if observed != expected:
            raise ValueError("runtime fragment stages must cover every stage in stable order")
        span_count = sum(item.observed_span_count for item in self.stages)
        if span_count + self.unclassified_span_count != len(self.runtime_profile.spans):
            raise ValueError("runtime fragment stage counts do not cover runtime spans")
        return self


def summarize_e2e_trace_stages(
    profile: RuntimeProfileSnapshot,
) -> tuple[tuple[E2ETraceStageMeasurement, ...], int]:
    """Build stage summaries without summing nested or concurrent wall time twice."""

    if not isinstance(profile, RuntimeProfileSnapshot):
        raise TypeError("profile must be RuntimeProfileSnapshot")
    grouped: dict[E2ETraceStage, list[RuntimeSpanSnapshot]] = {stage: [] for stage in E2ETraceStage}
    unclassified = 0
    for span in profile.spans:
        stage = _stage_for_span(span)
        if stage is None:
            unclassified += 1
        else:
            grouped[stage].append(span)

    summaries: list[E2ETraceStageMeasurement] = []
    for stage in E2ETraceStage:
        spans = grouped[stage]
        if not spans:
            summaries.append(
                E2ETraceStageMeasurement(
                    stage=stage,
                    measurement_status=E2ETraceMeasurementStatus.NOT_MEASURED,
                    observed_span_count=0,
                )
            )
            continue
        summaries.append(
            E2ETraceStageMeasurement(
                stage=stage,
                measurement_status=E2ETraceMeasurementStatus.MEASURED,
                observed_span_count=len(spans),
                wall_time_union_ns=_interval_union_ns(
                    (span.started_offset_ns, span.ended_offset_ns) for span in spans
                ),
                inclusive_span_time_ns=sum(span.elapsed_ns for span in spans),
            )
        )
    return tuple(summaries), unclassified


def build_e2e_trace_runtime_fragment(
    *,
    role: E2ETraceFragmentRole,
    runtime_profile: RuntimeProfileSnapshot,
) -> E2ETraceRuntimeFragment:
    """Freeze a stage view of a recorder snapshot without modifying that snapshot."""

    if not isinstance(role, E2ETraceFragmentRole):
        raise TypeError("role must be E2ETraceFragmentRole")
    stages, unclassified = summarize_e2e_trace_stages(runtime_profile)
    return E2ETraceRuntimeFragment(
        role=role,
        runtime_profile=runtime_profile,
        stages=stages,
        unclassified_span_count=unclassified,
    )


class E2ETraceArtifactReference(StrictModel):
    """Non-secret content reference to an external telemetry artifact."""

    artifact_kind: NonEmptyString
    exact_bytes_sha256: Sha256Digest


class E2ETraceHandlerTelemetry(StrictModel):
    """Optional handler-side telemetry; no GPU values are inferred from a client trace."""

    measurement_status: E2ETraceMeasurementStatus = E2ETraceMeasurementStatus.NOT_MEASURED
    artifact: E2ETraceArtifactReference | None = None

    @model_validator(mode="after")
    def validate_handler_telemetry(self) -> Self:
        if self.measurement_status is E2ETraceMeasurementStatus.MEASURED:
            if self.artifact is None:
                raise ValueError("MEASURED handler telemetry requires an artifact reference")
        elif self.artifact is not None:
            raise ValueError("NOT_MEASURED handler telemetry cannot retain an artifact")
        return self


class E2ETraceFunnelStep(StrictModel):
    """A count whose absence is represented as a measurement status, never a zero."""

    name: Literal[
        "DISPATCHED",
        "TERMINAL",
        "PROVIDER_SUCCEEDED",
        "SCHEMA_VALID",
        "PAIR_COMPARABLE",
        "GROUND_TRUTH_QUALITY",
    ]
    measurement_status: E2ETraceMeasurementStatus
    count: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.measurement_status is E2ETraceMeasurementStatus.MEASURED:
            if self.count is None:
                raise ValueError("MEASURED funnel steps require a count")
        elif self.count is not None:
            raise ValueError("NOT_MEASURED funnel steps cannot retain a count")
        return self


class E2ETraceCostInput(StrictModel):
    """Provider-reported input only; this is not a reconciled cloud bill."""

    role: Literal["CONTROL", "CANDIDATE"]
    measurement_status: E2ETraceMeasurementStatus
    input_frames: NonNegativeInt | None = None
    input_images: NonNegativeInt | None = None
    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    provider_reported_cost: (
        Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)] | None
    ) = None
    currency: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_cost_input(self) -> Self:
        if self.measurement_status is E2ETraceMeasurementStatus.MEASURED:
            if self.input_frames is None or self.input_images is None:
                raise ValueError("MEASURED cost input requires observed frame and image counts")
        elif any(
            value is not None
            for value in (
                self.input_frames,
                self.input_images,
                self.input_tokens,
                self.output_tokens,
                self.provider_reported_cost,
                self.currency,
            )
        ):
            raise ValueError("NOT_MEASURED cost input cannot retain provider usage")
        return self


class E2ETraceBilledCost(StrictModel):
    """Actual cloud bill availability, kept separate from a provider response field."""

    measurement_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    detail: NonEmptyString = "no billing artifact was bound to this trace"


class E2ETraceEndpointCorrelation(StrictModel):
    """Request-level links safe to retain in a noncanonical observation sidecar."""

    role: Literal["CONTROL", "CANDIDATE"]
    deployment_id: NonEmptyString
    endpoint_config_sha256: Sha256Digest
    handler_image_sha256: Sha256Digest
    capability_snapshot_sha256: Sha256Digest
    inference_id: OpaqueUuid | None = None
    logical_invocation_id: OpaqueUuid | None = None
    request_id: OpaqueUuid | None = None
    provider_request_id: NonEmptyString | None = None
    terminal_status: NonEmptyString | None = None
    retry_count: NonNegativeInt | None = None
    latency_ms: NonNegativeInt | None = None
    output_valid: bool | None = None
    raw_response_sha256s: tuple[Sha256Digest, ...] = ()
    transport_request_count: NonNegativeInt | None = None
    handler_telemetry: E2ETraceHandlerTelemetry = Field(default_factory=E2ETraceHandlerTelemetry)

    @model_validator(mode="after")
    def validate_endpoint_correlation(self) -> Self:
        terminal_values = (
            self.inference_id,
            self.logical_invocation_id,
            self.request_id,
            self.terminal_status,
            self.retry_count,
            self.latency_ms,
            self.output_valid,
        )
        if any(value is not None for value in terminal_values) and any(
            value is None for value in terminal_values
        ):
            raise ValueError("terminal correlation fields must be populated together")
        if self.raw_response_sha256s != tuple(sorted(set(self.raw_response_sha256s))):
            raise ValueError("raw response digests must be unique and ordered")
        return self


class E2ETraceCorrelation(StrictModel):
    """Observed links to a frozen paired workload, never a canonical identity claim."""

    external_report_file_sha256: Sha256Digest
    comparison_id: OpaqueUuid
    experiment_id: NonEmptyString
    experiment_contract_sha256: Sha256Digest
    source_workload_manifest_sha256: Sha256Digest
    input_identity_sha256: Sha256Digest
    control: E2ETraceEndpointCorrelation
    candidate: E2ETraceEndpointCorrelation

    @model_validator(mode="after")
    def validate_side_order(self) -> Self:
        if self.control.role != "CONTROL" or self.candidate.role != "CANDIDATE":
            raise ValueError("trace correlations must retain control and candidate roles")
        return self


class ExternalPairedE2ETraceBundle(StrictModel):
    """Observation-only trace sidecar for one bounded external paired invocation."""

    format_version: Literal["robata-e2e-trace-v1"] = E2E_TRACE_VERSION
    evidence_class: Literal["EXTERNAL_PROVIDER_OBSERVATION"] = "EXTERNAL_PROVIDER_OBSERVATION"
    execution_class: Literal["EXTERNAL_PAIRED_OBSERVATION"] = "EXTERNAL_PAIRED_OBSERVATION"
    selection_eligible: Literal[False] = False
    production_eligible: Literal[False] = False
    trace_id: OpaqueUuid
    observed_at: Rfc3339Timestamp
    coverage: Literal["PARTIAL"] = "PARTIAL"
    correlation: E2ETraceCorrelation
    launcher: E2ETraceRuntimeFragment
    control: E2ETraceRuntimeFragment
    candidate: E2ETraceRuntimeFragment
    quality_funnel: tuple[E2ETraceFunnelStep, ...]
    provider_cost_inputs: tuple[E2ETraceCostInput, ...]
    billed_cost: E2ETraceBilledCost = Field(default_factory=E2ETraceBilledCost)
    limitations: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_observation_only_bundle(self) -> Self:
        if self.launcher.role is not E2ETraceFragmentRole.LAUNCHER:
            raise ValueError("launcher fragment must have LAUNCHER role")
        if self.control.role is not E2ETraceFragmentRole.CONTROL:
            raise ValueError("control fragment must have CONTROL role")
        if self.candidate.role is not E2ETraceFragmentRole.CANDIDATE:
            raise ValueError("candidate fragment must have CANDIDATE role")
        expected_funnel = (
            "DISPATCHED",
            "TERMINAL",
            "PROVIDER_SUCCEEDED",
            "SCHEMA_VALID",
            "PAIR_COMPARABLE",
            "GROUND_TRUTH_QUALITY",
        )
        if tuple(item.name for item in self.quality_funnel) != expected_funnel:
            raise ValueError("quality funnel steps must use the stable trace order")
        if tuple(item.role for item in self.provider_cost_inputs) != ("CONTROL", "CANDIDATE"):
            raise ValueError("provider cost inputs must retain control and candidate order")
        if not self.limitations:
            raise ValueError("partial trace coverage requires declared limitations")
        return self


def write_external_paired_e2e_trace(
    bundle: ExternalPairedE2ETraceBundle,
    output_path: Path,
) -> None:
    """Atomically persist a sidecar without making it canonical evidence."""

    if not isinstance(bundle, ExternalPairedE2ETraceBundle):
        raise TypeError("bundle must be ExternalPairedE2ETraceBundle")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(bundle.model_dump(mode="json")) + b"\n"
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except FileExistsError as error:
        raise RuntimeError(
            f"temporary trace output path is already occupied: {temporary}"
        ) from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class ExternalPairedE2ETraceExecution:
    """The original P20 report and its separately serialized trace sidecar."""

    report: ExternalPairedQualificationReport
    trace: ExternalPairedE2ETraceBundle


async def run_external_paired_qualification_with_trace(
    *,
    environment: Mapping[str, str],
    control_capabilities_path: Path,
    candidate_capabilities_path: Path,
    workload_path: Path,
    trace_id: str | None = None,
    observed_at: str | None = None,
    max_attempts_per_endpoint: int = 1,
    evidence_directory: Path | None = None,
    transport_factory: TransportFactory | None = None,
) -> ExternalPairedE2ETraceExecution:
    """Run a bounded paired observation with three independent local recorders.

    The launcher profile is intentionally separate from the two endpoint profiles.
    Both endpoint adapters identify as ``runpod``, so sharing a recorder would make
    their request counters ambiguous.
    """

    from robata.benchmark.external_paired_qualification import run_external_paired_qualification

    launcher_recorder = RuntimeProfileRecorder()
    control_recorder = RuntimeProfileRecorder()
    candidate_recorder = RuntimeProfileRecorder()
    with runtime_span(
        launcher_recorder,
        "qualification.external_paired",
        {"mode": "paired"},
    ):
        report = await run_external_paired_qualification(
            environment=environment,
            control_capabilities_path=control_capabilities_path,
            candidate_capabilities_path=candidate_capabilities_path,
            workload_path=workload_path,
            max_attempts_per_endpoint=max_attempts_per_endpoint,
            evidence_directory=evidence_directory,
            transport_factory=transport_factory,
            control_runtime_observer=control_recorder,
            candidate_runtime_observer=candidate_recorder,
        )
    trace = build_external_paired_e2e_trace(
        report=report,
        trace_id=str(uuid4()) if trace_id is None else trace_id,
        observed_at=_utc_now() if observed_at is None else observed_at,
        launcher_profile=launcher_recorder.snapshot(),
        control_profile=control_recorder.snapshot(),
        candidate_profile=candidate_recorder.snapshot(),
    )
    return ExternalPairedE2ETraceExecution(report=report, trace=trace)


def build_external_paired_e2e_trace(
    *,
    report: ExternalPairedQualificationReport,
    trace_id: str,
    observed_at: str,
    launcher_profile: RuntimeProfileSnapshot,
    control_profile: RuntimeProfileSnapshot,
    candidate_profile: RuntimeProfileSnapshot,
) -> ExternalPairedE2ETraceBundle:
    """Bind frozen local observations to an unchanged external paired report."""

    from robata.benchmark.external_paired_qualification import (
        ExternalPairedQualificationReport as ExternalPairedReport,
    )

    if not isinstance(report, ExternalPairedReport):
        raise TypeError("report must be ExternalPairedQualificationReport")
    if not isinstance(trace_id, str):
        raise TypeError("trace_id must be a string")
    if not isinstance(observed_at, str):
        raise TypeError("observed_at must be a string")
    report_file_bytes = canonical_json_bytes(report.model_dump(mode="json")) + b"\n"
    terminals = (
        report.control_observation.terminal,
        report.candidate_observation.terminal,
    )
    terminal_count = sum(item is not None for item in terminals)
    provider_succeeded_count = sum(
        item is not None and item.status.value == "SUCCEEDED" for item in terminals
    )
    schema_valid_count = sum(item is not None and item.output_valid for item in terminals)
    return ExternalPairedE2ETraceBundle(
        trace_id=trace_id,
        observed_at=observed_at,
        correlation=E2ETraceCorrelation(
            external_report_file_sha256=exact_bytes_sha256(report_file_bytes),
            comparison_id=report.comparison.comparison_id,
            experiment_id=report.comparison.experiment_id,
            experiment_contract_sha256=report.comparison.experiment_contract_digest,
            source_workload_manifest_sha256=report.source_workload_manifest_sha256,
            input_identity_sha256=report.comparison.input_identity_sha256,
            control=_endpoint_correlation(
                role="CONTROL",
                binding=report.control,
                observation=report.control_observation,
            ),
            candidate=_endpoint_correlation(
                role="CANDIDATE",
                binding=report.candidate,
                observation=report.candidate_observation,
            ),
        ),
        launcher=build_e2e_trace_runtime_fragment(
            role=E2ETraceFragmentRole.LAUNCHER,
            runtime_profile=launcher_profile,
        ),
        control=build_e2e_trace_runtime_fragment(
            role=E2ETraceFragmentRole.CONTROL,
            runtime_profile=control_profile,
        ),
        candidate=build_e2e_trace_runtime_fragment(
            role=E2ETraceFragmentRole.CANDIDATE,
            runtime_profile=candidate_profile,
        ),
        quality_funnel=(
            E2ETraceFunnelStep(
                name="DISPATCHED",
                measurement_status=E2ETraceMeasurementStatus.MEASURED,
                count=len(report.experiment_decision.dispatches),
            ),
            E2ETraceFunnelStep(
                name="TERMINAL",
                measurement_status=E2ETraceMeasurementStatus.MEASURED,
                count=terminal_count,
            ),
            E2ETraceFunnelStep(
                name="PROVIDER_SUCCEEDED",
                measurement_status=E2ETraceMeasurementStatus.MEASURED,
                count=provider_succeeded_count,
            ),
            E2ETraceFunnelStep(
                name="SCHEMA_VALID",
                measurement_status=E2ETraceMeasurementStatus.MEASURED,
                count=schema_valid_count,
            ),
            E2ETraceFunnelStep(
                name="PAIR_COMPARABLE",
                measurement_status=E2ETraceMeasurementStatus.MEASURED,
                count=1 if report.comparison.comparable else 0,
            ),
            E2ETraceFunnelStep(
                name="GROUND_TRUTH_QUALITY",
                measurement_status=E2ETraceMeasurementStatus.NOT_MEASURED,
            ),
        ),
        provider_cost_inputs=(
            _cost_input(role="CONTROL", observation=report.control_observation),
            _cost_input(role="CANDIDATE", observation=report.candidate_observation),
        ),
        limitations=(
            "source ingestion and source R2 staging are not observed by the bounded launcher",
            "the paired launcher uses isolated local evidence rather than canonical "
            "PostgreSQL/R2 evidence",
            "canonical scheduling, reduction, and publication are not invoked by this P20 path",
            "handler GPU telemetry and reconciled cloud billing are not bound to this trace",
        ),
    )


def _endpoint_correlation(
    *,
    role: Literal["CONTROL", "CANDIDATE"],
    binding: ExternalEndpointBinding,
    observation: ExternalProviderObservation,
) -> E2ETraceEndpointCorrelation:
    terminal = observation.terminal
    raw_response_sha256s = tuple(
        sorted({item.exact_bytes_sha256 for item in observation.raw_evidence})
    )
    if terminal is None:
        return E2ETraceEndpointCorrelation(
            role=role,
            deployment_id=binding.deployment.deployment_id,
            endpoint_config_sha256=binding.endpoint_config_digest,
            handler_image_sha256=binding.handler_image_sha256,
            capability_snapshot_sha256=binding.capability_snapshot_digest,
            raw_response_sha256s=raw_response_sha256s,
            transport_request_count=observation.transport_request_count,
        )
    return E2ETraceEndpointCorrelation(
        role=role,
        deployment_id=binding.deployment.deployment_id,
        endpoint_config_sha256=binding.endpoint_config_digest,
        handler_image_sha256=binding.handler_image_sha256,
        capability_snapshot_sha256=binding.capability_snapshot_digest,
        inference_id=terminal.inference_id,
        logical_invocation_id=terminal.logical_invocation_id,
        request_id=terminal.request_id,
        provider_request_id=terminal.provider_request_id,
        terminal_status=terminal.status.value,
        retry_count=terminal.retry_count,
        latency_ms=terminal.latency_ms,
        output_valid=terminal.output_valid,
        raw_response_sha256s=raw_response_sha256s,
        transport_request_count=observation.transport_request_count,
    )


def _cost_input(
    *,
    role: Literal["CONTROL", "CANDIDATE"],
    observation: ExternalProviderObservation,
) -> E2ETraceCostInput:
    terminal = observation.terminal
    if terminal is None:
        return E2ETraceCostInput(
            role=role,
            measurement_status=E2ETraceMeasurementStatus.NOT_MEASURED,
        )
    usage = terminal.usage
    return E2ETraceCostInput(
        role=role,
        measurement_status=E2ETraceMeasurementStatus.MEASURED,
        input_frames=usage.input_frames,
        input_images=usage.input_images,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        provider_reported_cost=usage.cost,
        currency=usage.currency,
    )


def _stage_for_span(span: RuntimeSpanSnapshot) -> E2ETraceStage | None:
    """Classify stable names and fixed PostgreSQL operation families."""

    name = span.name
    if name == "postgres.authority.transaction":
        operation_family = {attribute.name: attribute.value for attribute in span.attributes}.get(
            "operation_family"
        )
        if isinstance(operation_family, str):
            return {
                "ORCHESTRATION": E2ETraceStage.ORCHESTRATION,
                "SOURCE": E2ETraceStage.SOURCE,
                "SCHEDULING": E2ETraceStage.SCHEDULING,
                "EVIDENCE": E2ETraceStage.EVIDENCE,
                "PUBLICATION": E2ETraceStage.PUBLICATION,
                "OTHER": E2ETraceStage.ORCHESTRATION,
            }.get(operation_family)

    # Local canonical composition uses a few stable names that predate the
    # generic production prefixes below. Keep them classified so a control
    # trace does not lose orchestration/scheduler evidence merely because the
    # local adapter names its SQLite operation family explicitly.
    if name in {"completion.storage.open", "completion.recovery.lookup"}:
        return E2ETraceStage.ORCHESTRATION
    if name.startswith(("qualification.", "runtime.", "canonical.")):
        return E2ETraceStage.ORCHESTRATION
    if name == "perception.media_scan":
        return E2ETraceStage.SOURCE
    if name in {"perception.observe", "perception.refine"}:
        return E2ETraceStage.INFERENCE
    if name == "perception.project":
        return E2ETraceStage.EVIDENCE
    if name in {
        "perception.temporal_reconcile",
        "perception.fusion",
        "perception.finalize",
    }:
        return E2ETraceStage.REDUCTION
    if name.startswith(("source.", "mcap.", "capture.", "media.", "frame.", "nvdec.")):
        return E2ETraceStage.SOURCE
    if name.startswith(
        (
            "scheduler.",
            "stream.",
            "queue.",
            "work.",
            "barrier.",
            "sqlite.work_scheduler.",
            "sqlite.barrier.",
        )
    ):
        return E2ETraceStage.SCHEDULING
    if name.startswith("inference."):
        return E2ETraceStage.INFERENCE
    if name.startswith(
        ("r2.", "postgres.", "sqlite.", "artifact.", "evidence.", "quality.evidence.")
    ):
        return E2ETraceStage.EVIDENCE
    if name.startswith(("reduction.", "fusion.")):
        return E2ETraceStage.REDUCTION
    if name.startswith(("completion.", "publication.", "outbox.", "review.")):
        return E2ETraceStage.PUBLICATION
    return None


def _interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


__all__ = [
    "E2E_TRACE_VERSION",
    "E2ETraceArtifactReference",
    "E2ETraceBilledCost",
    "E2ETraceCostInput",
    "E2ETraceCoverage",
    "E2ETraceEndpointCorrelation",
    "E2ETraceFragmentRole",
    "E2ETraceFunnelStep",
    "E2ETraceHandlerTelemetry",
    "E2ETraceMeasurementStatus",
    "E2ETraceRuntimeFragment",
    "E2ETraceStage",
    "E2ETraceStageMeasurement",
    "ExternalPairedE2ETraceBundle",
    "ExternalPairedE2ETraceExecution",
    "build_e2e_trace_runtime_fragment",
    "build_external_paired_e2e_trace",
    "run_external_paired_qualification_with_trace",
    "summarize_e2e_trace_stages",
    "write_external_paired_e2e_trace",
]
