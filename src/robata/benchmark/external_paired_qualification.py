"""Bounded external RunPod paired-observation launcher for P20.

This module deliberately constructs only an experiment-plane ``PAIRED`` route.
It is not a production composition root: both terminals are shadows, the two
adapters and ledgers are separate, and every emitted report remains
non-promotional.  The launcher is intentionally one paired invocation at a
time so an operator can collect a first real endpoint observation before
starting a saturation or representative-data qualification run.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.adapters.sqlite_inference_evidence import SQLiteInferenceEvidenceLedger
from robata.contracts.common import Nanoseconds, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.inference.adapter import PackageInput
from robata.inference.enrichment import PROVIDER_CLAIM_SCHEMA_ID
from robata.inference.experiment_execution import (
    ExperimentDeploymentBinding,
    ExperimentDeploymentRegistry,
    ExperimentExecutionCoordinator,
    ExperimentInvocation,
    ExperimentPairComparison,
    ExperimentTargetInput,
)
from robata.inference.input_plan import InferenceInputPlan
from robata.inference.models import ModelCapabilities, ModelInference, VisionTask
from robata.inference.offline_fixture import (
    InMemoryRawProviderBytesStore,
    RawProviderBytesStore,
    StrictProviderClaimParser,
)
from robata.inference.orchestrator import (
    InferenceLedger,
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)
from robata.inference.routing import (
    DispatchDisposition,
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
    ExperimentRoute,
    ModelDeployment,
    ModelRouteDecision,
    RouteMode,
    endpoint_config_digest,
)
from robata.inference.runpod import (
    RunPodApiKey,
    RunPodDeploymentConfiguration,
    RunPodEndpointConfig,
    RunPodRetryPolicy,
    RunPodTransport,
    RunPodVisionAdapter,
    StdlibRunPodTransport,
)
from robata.runtime.observability import RuntimeObserver

EXTERNAL_PAIRED_QUALIFICATION_WORKLOAD_VERSION = "robata-external-paired-workload-v1"
EXTERNAL_PAIRED_QUALIFICATION_REPORT_VERSION = "robata-external-paired-report-v1"
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA256_DIGEST = re.compile(r"^[0-9a-f]{64}$")

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1, le=5)]

type RunPodTopology = Literal["TWO_SINGLE_CARD_REPLICAS", "TWO_CARD_TENSOR_PARALLEL"]


class ExternalPairedQualificationError(ValueError):
    """Raised before a real paired endpoint observation can safely start."""


class ExternalPairedRole(StrEnum):
    """Stable two-endpoint roles used by the external qualification launcher."""

    CONTROL = "CONTROL"
    CANDIDATE = "CANDIDATE"


class ExternalEvidenceMode(StrEnum):
    """Where the two isolated experiment ledgers retain raw endpoint evidence."""

    IN_MEMORY = "IN_MEMORY"
    DURABLE_LOCAL_SQLITE = "DURABLE_LOCAL_SQLITE"


class ExternalGateStatus(StrEnum):
    """P20 gate state deliberately narrower than a qualification pass/fail result."""

    NOT_MEASURED = "NOT_MEASURED"
    OBSERVED_PROVIDER_OUTCOME = "OBSERVED_PROVIDER_OUTCOME"


class ExternalPairedWorkloadTarget(StrictModel):
    """One fully formed target-specific plan in a paired workload manifest."""

    deployment_id: NonEmptyString
    policy: InferencePolicy
    input_plan: InferenceInputPlan
    input_plan_part_ordinal: NonNegativeInt


class ExternalPairedWorkloadManifest(StrictModel):
    """Exact manifest for one bounded P20 paired endpoint observation.

    ``source_workload_manifest_sha256`` identifies the frozen source workload.
    The exact bytes digest of this launcher manifest is separately emitted in
    the report, avoiding a self-referential digest field.
    """

    format_version: Literal["robata-external-paired-workload-v1"] = (
        "robata-external-paired-workload-v1"
    )
    experiment_id: NonEmptyString
    contract_version: SchemaVersion
    route_id: NonEmptyString
    route_policy_version: SchemaVersion
    source_workload_manifest_sha256: Sha256Digest
    arrival_schedule_sha256: Sha256Digest
    comparison_config: dict[str, object]
    input_representation: ExperimentInputRepresentation
    isolation_profile: ExperimentIsolationProfile
    input_identity_sha256: Sha256Digest
    task: VisionTask
    package_set_id: OpaqueUuid | None = None
    mcap_id: OpaqueUuid
    camera_mapping_run_id: OpaqueUuid
    alignment_id: OpaqueUuid
    start_ns: Nanoseconds
    end_ns: Nanoseconds
    package_inputs: tuple[PackageInput, ...]
    input_config: dict[str, object] = Field(default_factory=dict)
    sampling_config: dict[str, object] = Field(default_factory=dict)
    metadata: dict[NonEmptyString, NonEmptyString] = Field(default_factory=dict)
    attempt: PositiveInt = 1
    retry_count: NonNegativeInt = 0
    control: ExternalPairedWorkloadTarget
    candidate: ExternalPairedWorkloadTarget

    @model_validator(mode="after")
    def validate_bounded_paired_request(self) -> ExternalPairedWorkloadManifest:
        if self.start_ns >= self.end_ns:
            raise ValueError("paired workload interval must be nonempty")
        if self.retry_count >= self.attempt:
            raise ValueError("paired workload retry_count must be lower than attempt")
        if self.control.deployment_id == self.candidate.deployment_id:
            raise ValueError("paired workload deployment identifiers must differ")
        for role, target in (
            (ExternalPairedRole.CONTROL, self.control),
            (ExternalPairedRole.CANDIDATE, self.candidate),
        ):
            if target.policy.task is not self.task:
                raise ValueError(f"{role.value.lower()} policy task must match workload task")
            if target.input_plan.subject.task is not self.task:
                raise ValueError(f"{role.value.lower()} input plan task must match workload task")
            if target.input_plan_part_ordinal >= len(target.input_plan.call_plan.parts):
                raise ValueError(
                    f"{role.value.lower()} input_plan_part_ordinal is outside the call plan"
                )
        return self


class ExternalEndpointBinding(StrictModel):
    """Non-secret endpoint/capability facts bound into a P20 observation report."""

    role: ExternalPairedRole
    deployment: ModelDeployment
    endpoint_config: RunPodEndpointConfig
    endpoint_config_digest: Sha256Digest
    pinned_configuration_sha256: Sha256Digest
    handler_image: NonEmptyString
    handler_image_sha256: Sha256Digest
    capability_file_sha256: Sha256Digest
    capability_snapshot_digest: Sha256Digest


class ExternalRawEvidenceRecord(StrictModel):
    """Identity-only summary of exact raw response bytes retained by one side."""

    artifact_id: OpaqueUuid
    request_id: OpaqueUuid
    provider_request_id: NonEmptyString
    exact_bytes_sha256: Sha256Digest


class ExternalProviderObservation(StrictModel):
    """One endpoint's terminal and retained raw evidence after a paired launch."""

    role: ExternalPairedRole
    terminal: ModelInference | None = None
    raw_evidence: tuple[ExternalRawEvidenceRecord, ...]
    transport_request_count: NonNegativeInt | None = None


class ExternalGateObservation(StrictModel):
    """Explicitly retain what a single endpoint observation does not establish."""

    gate: Literal["M0", "M1", "M2", "M3", "M4", "M5"]
    status: ExternalGateStatus
    detail: NonEmptyString


class ExternalPairedQualificationReport(StrictModel):
    """Local, non-promotional artifact emitted by the bounded P20 launcher."""

    format_version: Literal["robata-external-paired-report-v1"] = "robata-external-paired-report-v1"
    evidence_class: Literal["EXTERNAL_PROVIDER_OBSERVATION"] = "EXTERNAL_PROVIDER_OBSERVATION"
    execution_class: Literal["EXTERNAL_PAIRED_OBSERVATION"] = "EXTERNAL_PAIRED_OBSERVATION"
    route_mode: Literal["PAIRED"] = "PAIRED"
    selection_eligible: Literal[False] = False
    production_eligible: Literal[False] = False
    evidence_mode: ExternalEvidenceMode
    max_attempts_per_endpoint: PositiveInt
    workload_file_sha256: Sha256Digest
    source_workload_manifest_sha256: Sha256Digest
    control: ExternalEndpointBinding
    candidate: ExternalEndpointBinding
    experiment_contract: ExperimentContract
    experiment_decision: ModelRouteDecision
    comparison: ExperimentPairComparison
    control_observation: ExternalProviderObservation
    candidate_observation: ExternalProviderObservation
    external_gates: tuple[ExternalGateObservation, ...]

    @model_validator(mode="after")
    def validate_non_promotional_observation(self) -> ExternalPairedQualificationReport:
        if self.experiment_decision.mode is not RouteMode.PAIRED:
            raise ValueError("external paired report must retain a PAIRED route decision")
        if any(
            dispatch.disposition is not DispatchDisposition.OBSERVATION
            for dispatch in self.experiment_decision.dispatches
        ):
            raise ValueError("external paired report cannot contain authoritative dispatches")
        for observation in (self.control_observation, self.candidate_observation):
            if observation.terminal is not None and not observation.terminal.shadow:
                raise ValueError("external paired report cannot contain a non-shadow terminal")
        expected_gates = ("M0", "M1", "M2", "M3", "M4", "M5")
        if tuple(item.gate for item in self.external_gates) != expected_gates:
            raise ValueError("external paired report must state every P20 gate in order")
        if self.external_gates[3].status is not ExternalGateStatus.OBSERVED_PROVIDER_OUTCOME:
            raise ValueError("M3 must retain the observed bounded provider outcome")
        if any(
            item.status is not ExternalGateStatus.NOT_MEASURED
            for index, item in enumerate(self.external_gates)
            if index != 3
        ):
            raise ValueError("only the bounded M3 provider outcome may be observed here")
        return self


@dataclass(frozen=True, slots=True)
class _RoleConfiguration:
    role: ExternalPairedRole
    config: RunPodEndpointConfig
    credential: RunPodApiKey
    handler_image: str
    handler_image_sha256: str
    capability_snapshot_sha256: str


@dataclass(slots=True)
class _EvidenceScope:
    ledger: InferenceLedger
    raw_store: RawProviderBytesStore
    close: Callable[[], None]


type TransportFactory = Callable[[ExternalPairedRole], RunPodTransport]


def load_external_qualification_environment(
    base_values: Mapping[str, str], environment_file: Path | None
) -> dict[str, str]:
    """Load one explicitly named ``KEY=VALUE`` file without mutating process state."""

    values = dict(base_values)
    if environment_file is None:
        return values
    try:
        raw = environment_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ExternalPairedQualificationError(
            f"cannot read environment file: {environment_file}"
        ) from error
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or _ENVIRONMENT_KEY.fullmatch(key) is None:
            raise ExternalPairedQualificationError(
                f"environment file has an invalid KEY=VALUE assignment at line {line_number}"
            )
        values[key] = _parse_environment_value(raw_value.strip(), line_number)
    return values


def redact_external_qualification_detail(detail: str, environment: Mapping[str, str]) -> str:
    """Remove API keys from CLI errors without serializing any credential elsewhere."""

    redacted = detail
    for key, value in environment.items():
        if key.endswith("_API_KEY") and isinstance(value, str) and value:
            redacted = redacted.replace(value, "REDACTED")
    return redacted


async def run_external_paired_qualification(
    *,
    environment: Mapping[str, str],
    control_capabilities_path: Path,
    candidate_capabilities_path: Path,
    workload_path: Path,
    max_attempts_per_endpoint: int = 1,
    evidence_directory: Path | None = None,
    transport_factory: TransportFactory | None = None,
    control_runtime_observer: RuntimeObserver | None = None,
    candidate_runtime_observer: RuntimeObserver | None = None,
) -> ExternalPairedQualificationReport:
    """Execute exactly one paired, shadow-only P20 endpoint observation.

    The default transport is the real stdlib RunPod transport. Tests must pass
    an explicit recorded transport factory; no implicit fixture or fallback
    exists in this entry point.
    """

    max_attempts = _bounded_max_attempts(max_attempts_per_endpoint)
    control_config = _load_role_configuration(environment, ExternalPairedRole.CONTROL)
    candidate_config = _load_role_configuration(environment, ExternalPairedRole.CANDIDATE)
    _validate_distinct_endpoint_configurations(control_config, candidate_config)

    control_capabilities, control_capability_file_sha256 = _load_capabilities(
        control_capabilities_path,
        role=ExternalPairedRole.CONTROL,
        expected_snapshot_digest=control_config.capability_snapshot_sha256,
    )
    candidate_capabilities, candidate_capability_file_sha256 = _load_capabilities(
        candidate_capabilities_path,
        role=ExternalPairedRole.CANDIDATE,
        expected_snapshot_digest=candidate_config.capability_snapshot_sha256,
    )
    workload, workload_file_sha256 = _load_workload(workload_path)
    _validate_workload_shape(workload)

    schema_registry = SchemaRegistry()
    control_deployment = _deployment_from(
        role_configuration=control_config,
        capabilities=control_capabilities,
        deployment_id=workload.control.deployment_id,
    )
    candidate_deployment = _deployment_from(
        role_configuration=candidate_config,
        capabilities=candidate_capabilities,
        deployment_id=workload.candidate.deployment_id,
    )
    _validate_target_against_deployment(
        target=workload.control,
        deployment=control_deployment,
        capabilities=control_capabilities,
        role=ExternalPairedRole.CONTROL,
    )
    _validate_target_against_deployment(
        target=workload.candidate,
        deployment=candidate_deployment,
        capabilities=candidate_capabilities,
        role=ExternalPairedRole.CANDIDATE,
    )

    contract = ExperimentContract(
        experiment_id=workload.experiment_id,
        contract_version=workload.contract_version,
        workload_manifest_sha256=workload.source_workload_manifest_sha256,
        arrival_schedule_sha256=workload.arrival_schedule_sha256,
        comparison_config_sha256=semantic_sha256(workload.comparison_config),
        input_representation=workload.input_representation,
        isolation_profile=workload.isolation_profile,
        control=control_deployment,
        candidate=candidate_deployment,
    )
    route = ExperimentRoute(
        route_id=workload.route_id,
        policy_version=workload.route_policy_version,
        mode=RouteMode.PAIRED,
        sample_ratio=1.0,
        contract=contract,
    )
    decision = route.decide(input_identity_sha256=workload.input_identity_sha256)
    if len(decision.dispatches) != 2:
        raise ExternalPairedQualificationError(
            "paired qualification route did not select two sides"
        )

    evidence_mode = (
        ExternalEvidenceMode.DURABLE_LOCAL_SQLITE
        if evidence_directory is not None
        else ExternalEvidenceMode.IN_MEMORY
    )
    control_evidence, candidate_evidence = _create_evidence_scopes(
        mode=evidence_mode,
        directory=evidence_directory,
        schema_registry=schema_registry,
    )
    try:
        control_transport = _transport_for(
            transport_factory=transport_factory,
            role=ExternalPairedRole.CONTROL,
        )
        candidate_transport = _transport_for(
            transport_factory=transport_factory,
            role=ExternalPairedRole.CANDIDATE,
        )
        retry_policy = RunPodRetryPolicy(
            version="p20-external-paired-retry-v1",
            max_attempts=max_attempts,
            base_delay_ms=0,
            max_delay_ms=0,
        )
        control_adapter = _adapter_from(
            role_configuration=control_config,
            capabilities=control_capabilities,
            retry_policy=retry_policy,
            raw_store=control_evidence.raw_store,
            schema_registry=schema_registry,
            runtime_observer=control_runtime_observer,
            transport=control_transport,
        )
        candidate_adapter = _adapter_from(
            role_configuration=candidate_config,
            capabilities=candidate_capabilities,
            retry_policy=retry_policy,
            raw_store=candidate_evidence.raw_store,
            schema_registry=schema_registry,
            runtime_observer=candidate_runtime_observer,
            transport=candidate_transport,
        )
        control_orchestrator = _orchestrator_from(
            adapter=control_adapter,
            target=workload.control,
            schema_registry=schema_registry,
            ledger=control_evidence.ledger,
        )
        candidate_orchestrator = _orchestrator_from(
            adapter=candidate_adapter,
            target=workload.candidate,
            schema_registry=schema_registry,
            ledger=candidate_evidence.ledger,
        )
        registry = ExperimentDeploymentRegistry(
            bindings={
                control_deployment.deployment_id: ExperimentDeploymentBinding(
                    deployment=control_deployment,
                    orchestrator=control_orchestrator,
                ),
                candidate_deployment.deployment_id: ExperimentDeploymentBinding(
                    deployment=candidate_deployment,
                    orchestrator=candidate_orchestrator,
                ),
            }
        )
        invocation = ExperimentInvocation(
            source_workload_manifest_sha256=workload.source_workload_manifest_sha256,
            input_identity_sha256=workload.input_identity_sha256,
            task=workload.task,
            package_set_id=workload.package_set_id,
            mcap_id=workload.mcap_id,
            camera_mapping_run_id=workload.camera_mapping_run_id,
            alignment_id=workload.alignment_id,
            start_ns=workload.start_ns,
            end_ns=workload.end_ns,
            package_inputs=workload.package_inputs,
            control=ExperimentTargetInput(
                input_plan=workload.control.input_plan,
                input_plan_part_ordinal=workload.control.input_plan_part_ordinal,
            ),
            candidate=ExperimentTargetInput(
                input_plan=workload.candidate.input_plan,
                input_plan_part_ordinal=workload.candidate.input_plan_part_ordinal,
            ),
            comparison_config=workload.comparison_config,
            input_config=workload.input_config,
            sampling_config=workload.sampling_config,
            metadata=workload.metadata,
            attempt=workload.attempt,
            retry_count=workload.retry_count,
        )
        result = await ExperimentExecutionCoordinator(registry=registry).execute(
            route=route,
            decision=decision,
            invocation=invocation,
        )
        _assert_observation_only(
            result.comparison,
            control_terminal=result.control_terminal,
            candidate_terminal=result.candidate_terminal,
            control_ledger=control_evidence.ledger,
            candidate_ledger=candidate_evidence.ledger,
            control_policy=workload.control.policy,
            candidate_policy=workload.candidate.policy,
        )
        return ExternalPairedQualificationReport(
            evidence_mode=evidence_mode,
            max_attempts_per_endpoint=max_attempts,
            workload_file_sha256=workload_file_sha256,
            source_workload_manifest_sha256=workload.source_workload_manifest_sha256,
            control=_endpoint_binding(
                role_configuration=control_config,
                capabilities=control_capabilities,
                capability_file_sha256=control_capability_file_sha256,
                deployment=control_deployment,
            ),
            candidate=_endpoint_binding(
                role_configuration=candidate_config,
                capabilities=candidate_capabilities,
                capability_file_sha256=candidate_capability_file_sha256,
                deployment=candidate_deployment,
            ),
            experiment_contract=contract,
            experiment_decision=decision,
            comparison=result.comparison,
            control_observation=_provider_observation(
                role=ExternalPairedRole.CONTROL,
                terminal=result.control_terminal,
                raw_store=control_evidence.raw_store,
                transport=control_transport,
            ),
            candidate_observation=_provider_observation(
                role=ExternalPairedRole.CANDIDATE,
                terminal=result.candidate_terminal,
                raw_store=candidate_evidence.raw_store,
                transport=candidate_transport,
            ),
            external_gates=_gate_observations(result.comparison),
        )
    finally:
        control_evidence.close()
        candidate_evidence.close()


def write_external_paired_qualification_report(
    report: ExternalPairedQualificationReport, output_path: Path
) -> None:
    """Atomically persist the non-secret report without overwriting inputs in-place."""

    if not isinstance(report, ExternalPairedQualificationReport):
        raise TypeError("report must be ExternalPairedQualificationReport")
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report.model_dump(mode="json")) + b"\n"
    temporary = parent / f".{output_path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except FileExistsError as error:
        raise ExternalPairedQualificationError(
            f"temporary output path is already occupied: {temporary}"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_environment_value(value: str, line_number: int) -> str:
    if not value or value[0] not in {"'", '"'}:
        return value
    if len(value) < 2 or value[-1] != value[0]:
        raise ExternalPairedQualificationError(
            f"environment file has an unclosed quoted value at line {line_number}"
        )
    return value[1:-1]


def _load_role_configuration(
    environment: Mapping[str, str], role: ExternalPairedRole
) -> _RoleConfiguration:
    prefix = f"RUNPOD_{role.value}_"
    credential = RunPodApiKey(
        _required_environment_value(
            environment,
            f"{prefix}API_KEY",
            fallback="RUNPOD_API_KEY",
        )
    )
    handler_image = _required_environment_value(environment, f"{prefix}HANDLER_IMAGE")
    handler_image_sha256 = _environment_sha256(environment, f"{prefix}HANDLER_IMAGE_SHA256")
    capability_snapshot_sha256 = _environment_sha256(
        environment, f"{prefix}CAPABILITY_SNAPSHOT_SHA256"
    )
    deployment_configuration = RunPodDeploymentConfiguration(
        model_identifier=_required_environment_value(environment, f"{prefix}MODEL_IDENTIFIER"),
        model_version=_required_environment_value(environment, f"{prefix}MODEL_VERSION"),
        inference_engine=_required_environment_value(environment, f"{prefix}INFERENCE_ENGINE"),
        precision_or_quantization=_required_environment_value(
            environment, f"{prefix}PRECISION_OR_QUANTIZATION"
        ),
        topology=_required_topology(environment, f"{prefix}TOPOLOGY"),
        max_output_tokens=_positive_environment_integer(environment, f"{prefix}MAX_OUTPUT_TOKENS"),
        supported_topologies=_supported_topologies(environment, prefix=prefix),
    )
    config = RunPodEndpointConfig(
        provider="runpod",
        deployment_configuration=deployment_configuration,
        endpoint_url=_required_environment_value(environment, f"{prefix}ENDPOINT_URL"),
        adapter_version=_required_environment_value(environment, f"{prefix}ADAPTER_VERSION"),
        native_batch_enabled=_environment_boolean(
            environment, f"{prefix}NATIVE_BATCH_ENABLED", default=False
        ),
        native_batch_max_size=_positive_environment_integer(
            environment, f"{prefix}NATIVE_BATCH_MAX_SIZE", default=1
        ),
        max_concurrent_requests=_positive_environment_integer(
            environment, f"{prefix}MAX_CONCURRENT_REQUESTS"
        ),
        request_timeout_cap_ms=_positive_environment_integer(
            environment, f"{prefix}REQUEST_TIMEOUT_CAP_MS"
        ),
        max_response_bytes=_positive_environment_integer(
            environment, f"{prefix}MAX_RESPONSE_BYTES"
        ),
    )
    if config.native_batch_enabled:
        raise ExternalPairedQualificationError(
            "bounded external paired qualification requires native batch dispatch to be disabled"
        )
    return _RoleConfiguration(
        role=role,
        config=config,
        credential=credential,
        handler_image=handler_image,
        handler_image_sha256=handler_image_sha256,
        capability_snapshot_sha256=capability_snapshot_sha256,
    )


def _required_environment_value(
    environment: Mapping[str, str], name: str, *, fallback: str | None = None
) -> str:
    value = environment.get(name)
    if value is None and fallback is not None:
        value = environment.get(fallback)
    if not isinstance(value, str) or not value:
        suffix = f" or {fallback}" if fallback is not None else ""
        raise ExternalPairedQualificationError(f"{name}{suffix} must be configured")
    return value


def _environment_sha256(environment: Mapping[str, str], name: str) -> str:
    value = _required_environment_value(environment, name)
    if _SHA256_DIGEST.fullmatch(value) is None:
        raise ExternalPairedQualificationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _required_topology(environment: Mapping[str, str], name: str) -> RunPodTopology:
    value = _required_environment_value(environment, name)
    if value == "TWO_SINGLE_CARD_REPLICAS":
        return cast(RunPodTopology, value)
    if value == "TWO_CARD_TENSOR_PARALLEL":
        return cast(RunPodTopology, value)
    raise ExternalPairedQualificationError(
        f"{name} must be TWO_SINGLE_CARD_REPLICAS or TWO_CARD_TENSOR_PARALLEL"
    )


def _positive_environment_integer(
    environment: Mapping[str, str], name: str, *, default: int | None = None
) -> int:
    value = environment.get(name)
    if value is None:
        if default is None:
            raise ExternalPairedQualificationError(f"{name} must be configured")
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ExternalPairedQualificationError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ExternalPairedQualificationError(f"{name} must be positive")
    return parsed


def _environment_boolean(environment: Mapping[str, str], name: str, *, default: bool) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    if value == "true":
        return True
    if value == "false":
        return False
    raise ExternalPairedQualificationError(f"{name} must be true or false")


def _supported_topologies(
    environment: Mapping[str, str], *, prefix: str
) -> tuple[RunPodTopology, ...]:
    value = environment.get(f"{prefix}SUPPORTED_TOPOLOGIES")
    if value is None:
        return ()
    if not isinstance(value, str) or not value:
        raise ExternalPairedQualificationError(
            f"{prefix}SUPPORTED_TOPOLOGIES must be non-empty when configured"
        )
    raw = value.split(",")
    parsed = tuple(item.strip() for item in raw)
    if not all(parsed) or any(item != item.strip() for item in raw):
        raise ExternalPairedQualificationError(
            f"{prefix}SUPPORTED_TOPOLOGIES must be comma-separated without whitespace"
        )
    topologies: list[RunPodTopology] = []
    for topology in parsed:
        if topology == "TWO_SINGLE_CARD_REPLICAS" or topology == "TWO_CARD_TENSOR_PARALLEL":
            topologies.append(cast(RunPodTopology, topology))
        else:
            raise ExternalPairedQualificationError(
                f"{prefix}SUPPORTED_TOPOLOGIES contains an unsupported topology"
            )
    return tuple(topologies)


def _validate_distinct_endpoint_configurations(
    control: _RoleConfiguration, candidate: _RoleConfiguration
) -> None:
    if control.config.endpoint_url == candidate.config.endpoint_url:
        raise ExternalPairedQualificationError(
            "control and candidate endpoint URLs must differ for a paired observation"
        )


def _load_capabilities(
    path: Path,
    *,
    role: ExternalPairedRole,
    expected_snapshot_digest: str,
) -> tuple[ModelCapabilities, str]:
    raw, _document = _load_exact_json_object(path, label=f"{role.value.lower()} capabilities")
    try:
        capabilities = ModelCapabilities.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise ExternalPairedQualificationError(
            f"{role.value.lower()} capabilities file is invalid"
        ) from error
    if capabilities.snapshot_digest != expected_snapshot_digest:
        raise ExternalPairedQualificationError(
            f"{role.value.lower()} capabilities snapshot digest does not match its environment pin"
        )
    return capabilities, exact_bytes_sha256(raw)


def _load_workload(path: Path) -> tuple[ExternalPairedWorkloadManifest, str]:
    raw, _document = _load_exact_json_object(path, label="paired workload manifest")
    try:
        workload = ExternalPairedWorkloadManifest.model_validate_json(raw, strict=True)
    except ValidationError as error:
        raise ExternalPairedQualificationError("paired workload manifest is invalid") from error
    return workload, exact_bytes_sha256(raw)


def _load_exact_json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, object]]:
    if not isinstance(path, Path):
        raise TypeError(f"{label} path must be pathlib.Path")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ExternalPairedQualificationError(f"cannot read {label}: {path}") from error
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ExternalPairedQualificationError(f"{label} must not contain a UTF-8 BOM")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        document = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ExternalPairedQualificationError(f"{label} is not strict JSON") from error
    if not isinstance(document, dict):
        raise ExternalPairedQualificationError(f"{label} root must be a JSON object")
    return raw, document


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _validate_workload_shape(workload: ExternalPairedWorkloadManifest) -> None:
    if workload.control.input_plan_part_ordinal >= len(workload.control.input_plan.call_plan.parts):
        raise ExternalPairedQualificationError("control request is outside its call plan")
    if workload.candidate.input_plan_part_ordinal >= len(
        workload.candidate.input_plan.call_plan.parts
    ):
        raise ExternalPairedQualificationError("candidate request is outside its call plan")


def _deployment_from(
    *,
    role_configuration: _RoleConfiguration,
    capabilities: ModelCapabilities,
    deployment_id: str,
) -> ModelDeployment:
    deployment_config = role_configuration.config.deployment_configuration
    if deployment_config is None:
        raise ExternalPairedQualificationError(
            "RunPod endpoint configuration requires a deployment pin"
        )
    if (
        capabilities.provider != role_configuration.config.provider
        or capabilities.model_name != deployment_config.model_identifier
        or capabilities.model_version != deployment_config.model_version
    ):
        raise ExternalPairedQualificationError(
            f"{role_configuration.role.value.lower()} capabilities do not match "
            "endpoint deployment pin"
        )
    return ModelDeployment(
        deployment_id=deployment_id,
        provider=capabilities.provider,
        model_name=capabilities.model_name,
        model_version=capabilities.model_version,
        adapter_version=role_configuration.config.adapter_version,
        capability_snapshot_id=capabilities.snapshot_id,
        capability_snapshot_digest=capabilities.snapshot_digest,
        endpoint_config_digest=endpoint_config_digest(role_configuration.config),
        max_concurrent_requests=role_configuration.config.max_concurrent_requests,
    )


def _validate_target_against_deployment(
    *,
    target: ExternalPairedWorkloadTarget,
    deployment: ModelDeployment,
    capabilities: ModelCapabilities,
    role: ExternalPairedRole,
) -> None:
    policy = target.policy
    expected = (
        deployment.provider,
        deployment.model_name,
        deployment.model_version,
        deployment.adapter_version,
    )
    if (
        policy.provider,
        policy.model_name,
        policy.model_version,
        policy.adapter_version,
    ) != expected:
        raise ExternalPairedQualificationError(
            f"{role.value.lower()} inference policy does not match its endpoint deployment"
        )
    if policy.output_schema.schema_id != PROVIDER_CLAIM_SCHEMA_ID:
        raise ExternalPairedQualificationError(
            f"{role.value.lower()} policy must use the registered provider-claim output schema"
        )
    plan = target.input_plan
    if (
        plan.target.provider,
        plan.target.model_name,
        plan.target.model_version,
        plan.target.adapter_version,
        plan.target.capability_snapshot_id,
        plan.target.capability_snapshot_sha256,
    ) != (
        deployment.provider,
        deployment.model_name,
        deployment.model_version,
        deployment.adapter_version,
        deployment.capability_snapshot_id,
        deployment.capability_snapshot_digest,
    ):
        raise ExternalPairedQualificationError(
            f"{role.value.lower()} input plan does not match its endpoint deployment"
        )
    if capabilities.snapshot_digest != deployment.capability_snapshot_digest:
        raise ExternalPairedQualificationError(
            f"{role.value.lower()} capability identity changed during launcher assembly"
        )


def _create_evidence_scopes(
    *,
    mode: ExternalEvidenceMode,
    directory: Path | None,
    schema_registry: SchemaRegistry,
) -> tuple[_EvidenceScope, _EvidenceScope]:
    if mode is ExternalEvidenceMode.IN_MEMORY:
        control_raw = InMemoryRawProviderBytesStore()
        candidate_raw = InMemoryRawProviderBytesStore()
        return (
            _EvidenceScope(
                ledger=InMemoryInferenceLedger(),
                raw_store=control_raw,
                close=lambda: None,
            ),
            _EvidenceScope(
                ledger=InMemoryInferenceLedger(),
                raw_store=candidate_raw,
                close=lambda: None,
            ),
        )
    if directory is None:
        raise ExternalPairedQualificationError(
            "durable evidence mode requires an evidence directory"
        )
    control_ledger: SQLiteInferenceEvidenceLedger | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        control_ledger = SQLiteInferenceEvidenceLedger(
            directory / "control-inference-evidence.sqlite",
            schema_registry,
            raw_bytes_cas_root=directory / "control-raw-provider-bytes",
        )
        candidate_ledger = SQLiteInferenceEvidenceLedger(
            directory / "candidate-inference-evidence.sqlite",
            schema_registry,
            raw_bytes_cas_root=directory / "candidate-raw-provider-bytes",
        )
    except Exception as error:
        if control_ledger is not None:
            control_ledger.close()
        raise ExternalPairedQualificationError(
            "cannot create durable local evidence stores"
        ) from error
    assert control_ledger is not None
    return (
        _EvidenceScope(ledger=control_ledger, raw_store=control_ledger, close=control_ledger.close),
        _EvidenceScope(
            ledger=candidate_ledger,
            raw_store=candidate_ledger,
            close=candidate_ledger.close,
        ),
    )


def _transport_for(
    *, transport_factory: TransportFactory | None, role: ExternalPairedRole
) -> RunPodTransport:
    transport = StdlibRunPodTransport() if transport_factory is None else transport_factory(role)
    if not callable(getattr(transport, "post", None)):
        raise TypeError("transport_factory must return a RunPodTransport")
    return transport


def _adapter_from(
    *,
    role_configuration: _RoleConfiguration,
    capabilities: ModelCapabilities,
    retry_policy: RunPodRetryPolicy,
    raw_store: RawProviderBytesStore,
    schema_registry: SchemaRegistry,
    runtime_observer: RuntimeObserver | None = None,
    transport: RunPodTransport,
) -> RunPodVisionAdapter:
    return RunPodVisionAdapter(
        config=role_configuration.config,
        credential=role_configuration.credential,
        capabilities=capabilities,
        retry_policy=retry_policy,
        raw_store=raw_store,
        parser=StrictProviderClaimParser(
            schema_registry,
            parser_version="p20-external-paired-parser-v1",
        ),
        runtime_observer=runtime_observer,
        transport=transport,
    )


def _orchestrator_from(
    *,
    adapter: RunPodVisionAdapter,
    target: ExternalPairedWorkloadTarget,
    schema_registry: SchemaRegistry,
    ledger: InferenceLedger,
) -> InferenceOrchestrator:
    schema_reference = SchemaRef.model_validate(
        target.policy.output_schema.model_dump(mode="python")
    )
    _ = schema_registry.resolve_exact(schema_reference)
    schema_artifacts = {
        entry.ref.artifact_id: schema_registry.resolve_exact(entry.ref).document_bytes
        for entry in schema_registry.catalog.schemas
    }
    return InferenceOrchestrator(
        adapters={target.policy.provider: adapter},
        task_policies={target.policy.task: target.policy},
        schema_artifacts=schema_artifacts,
        ledger=ledger,
        max_batch_size=1,
    )


def _assert_observation_only(
    comparison: ExperimentPairComparison,
    *,
    control_terminal: ModelInference | None,
    candidate_terminal: ModelInference | None,
    control_ledger: InferenceLedger,
    candidate_ledger: InferenceLedger,
    control_policy: InferencePolicy,
    candidate_policy: InferencePolicy,
) -> None:
    for role, terminal, ledger, policy in (
        (
            ExternalPairedRole.CONTROL,
            control_terminal,
            control_ledger,
            control_policy,
        ),
        (
            ExternalPairedRole.CANDIDATE,
            candidate_terminal,
            candidate_ledger,
            candidate_policy,
        ),
    ):
        if terminal is None:
            continue
        if not terminal.shadow:
            raise ExternalPairedQualificationError(
                f"{role.value.lower()} external observation produced a non-shadow terminal"
            )
        if ledger.get_selection(terminal.logical_invocation_id, policy.selection_policy_version):
            raise ExternalPairedQualificationError(
                f"{role.value.lower()} external observation produced an authoritative selection"
            )
    if comparison.experiment_id == "":
        raise ExternalPairedQualificationError(
            "paired external observation has no experiment identity"
        )


def _endpoint_binding(
    *,
    role_configuration: _RoleConfiguration,
    capabilities: ModelCapabilities,
    capability_file_sha256: str,
    deployment: ModelDeployment,
) -> ExternalEndpointBinding:
    pinned_configuration = {
        "endpoint_config": role_configuration.config.model_dump(mode="json"),
        "handler_image": role_configuration.handler_image,
        "handler_image_sha256": role_configuration.handler_image_sha256,
        "capability_snapshot_sha256": role_configuration.capability_snapshot_sha256,
    }
    return ExternalEndpointBinding(
        role=role_configuration.role,
        deployment=deployment,
        endpoint_config=role_configuration.config,
        endpoint_config_digest=endpoint_config_digest(role_configuration.config),
        pinned_configuration_sha256=exact_bytes_sha256(canonical_json_bytes(pinned_configuration)),
        handler_image=role_configuration.handler_image,
        handler_image_sha256=role_configuration.handler_image_sha256,
        capability_file_sha256=capability_file_sha256,
        capability_snapshot_digest=capabilities.snapshot_digest,
    )


def _provider_observation(
    *,
    role: ExternalPairedRole,
    terminal: ModelInference | None,
    raw_store: RawProviderBytesStore,
    transport: RunPodTransport,
) -> ExternalProviderObservation:
    raw_evidence = tuple(
        ExternalRawEvidenceRecord(
            artifact_id=record.artifact_id,
            request_id=record.request_id,
            provider_request_id=record.provider_request_id,
            exact_bytes_sha256=record.exact_bytes_sha256,
        )
        for record in raw_store.list_records()
    )
    request_count = _transport_request_count(transport)
    return ExternalProviderObservation(
        role=role,
        terminal=terminal,
        raw_evidence=raw_evidence,
        transport_request_count=request_count,
    )


def _transport_request_count(transport: RunPodTransport) -> int | None:
    for attribute in ("network_call_count", "request_count"):
        value = getattr(transport, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _gate_observations(comparison: ExperimentPairComparison) -> tuple[ExternalGateObservation, ...]:
    return (
        ExternalGateObservation(
            gate="M0",
            status=ExternalGateStatus.NOT_MEASURED,
            detail="configuration was bound locally; endpoint reproducibility is not qualified",
        ),
        ExternalGateObservation(
            gate="M1",
            status=ExternalGateStatus.NOT_MEASURED,
            detail="one manifest was bound; source/R2 and representation scope are not qualified",
        ),
        ExternalGateObservation(
            gate="M2",
            status=ExternalGateStatus.NOT_MEASURED,
            detail="no frozen labels, critical-error review, or adjudication were executed",
        ),
        ExternalGateObservation(
            gate="M3",
            status=ExternalGateStatus.OBSERVED_PROVIDER_OUTCOME,
            detail=(
                "one bounded paired endpoint observation was retained with comparison status "
                f"{comparison.status.value}; this is not a saturation or cost qualification"
            ),
        ),
        ExternalGateObservation(
            gate="M4",
            status=ExternalGateStatus.NOT_MEASURED,
            detail="no reliability, restart, queue-budget, or canary qualification was executed",
        ),
        ExternalGateObservation(
            gate="M5",
            status=ExternalGateStatus.NOT_MEASURED,
            detail="no independent release review or go/no-go decision was recorded",
        ),
    )


def _bounded_max_attempts(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ExternalPairedQualificationError(
            "max_attempts_per_endpoint must be an integer from 1 to 5"
        )
    return value


__all__ = [
    "EXTERNAL_PAIRED_QUALIFICATION_REPORT_VERSION",
    "EXTERNAL_PAIRED_QUALIFICATION_WORKLOAD_VERSION",
    "ExternalEndpointBinding",
    "ExternalEvidenceMode",
    "ExternalGateObservation",
    "ExternalGateStatus",
    "ExternalPairedQualificationError",
    "ExternalPairedQualificationReport",
    "ExternalPairedRole",
    "ExternalPairedWorkloadManifest",
    "ExternalPairedWorkloadTarget",
    "ExternalProviderObservation",
    "ExternalRawEvidenceRecord",
    "TransportFactory",
    "load_external_qualification_environment",
    "redact_external_qualification_detail",
    "run_external_paired_qualification",
    "write_external_paired_qualification_report",
]
