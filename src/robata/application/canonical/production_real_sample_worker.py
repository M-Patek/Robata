"""Fail-closed production real-sample orchestration boundary.

This module is the deliberately small bridge between the reviewed production
composition root and a source-specific real-sample worker.  It does not create
new canonical wire contracts and it never pretends that a local/offline runner
is a PostgreSQL production execution.  The worker verifies a pinned source,
loads the reviewed bootstrap, constructs the production runtime, and delegates
the actual scheduler -> inference -> evidence -> completion -> publication
sequence to an explicit driver.  Until that driver is supplied, the worker
writes a complete non-canonical audit bundle and fails closed.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol
from uuid import uuid4

from pydantic import Field, StringConstraints, model_validator

from robata.adapters.pgvector_runtime import (
    PgVectorRuntimeConfig,
    create_verified_pgvector_projection_store_from_environment,
)
from robata.adapters.postgres_completion_evidence import PostgresInferenceEvidenceLedger
from robata.adapters.r2_object_store import create_r2_object_store_from_environment
from robata.application.canonical.mcap_source import (
    CanonicalMcapSourceBundle,
    authorize_mcap_mapping,
    load_canonical_mcap_source,
)
from robata.application.canonical.production_bootstrap import (
    ProductionRuntimeBootstrapConfiguration,
    load_production_runtime_bootstrap_configuration,
)
from robata.application.canonical.production_composition import (
    CanonicalPostgresRuntimeConfig,
    ProductionCompositionContract,
)
from robata.application.canonical.production_runtime import (
    CanonicalPostgresRuntimeCredentials,
    ProductionCanonicalRuntime,
    ProductionTenantContext,
    build_production_canonical_runtime,
)
from robata.application.canonical.production_traffic import (
    ProductionTrafficBridge,
    ProductionTrafficError,
    ProductionTrafficReadiness,
    ProductionTrafficRoutePlan,
)
from robata.application.canonical.r2_mcap_staging import (
    R2McapSourceStageReceipt,
    load_r2_mcap_source_manifest,
    stage_r2_mcap_source,
)
from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import exact_bytes_sha256
from robata.contracts.schema_registry import default_schema_registry
from robata.inference.offline_fixture import StrictProviderClaimParser
from robata.inference.runpod import RunPodApiKey, RunPodVisionAdapter
from robata.runtime.e2e_participation import (
    E2EParticipationBoundary,
    E2EParticipationCoverage,
    E2EParticipationDeclaration,
    E2EParticipationState,
    build_e2e_participation_manifest,
    write_e2e_participation_manifest,
)
from robata.runtime.e2e_trace import (
    E2ETraceFragmentRole,
    build_e2e_trace_runtime_fragment,
)
from robata.runtime.observability import RuntimeObserver, RuntimeProfileRecorder, runtime_span

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

WORKER_REPORT_VERSION: Literal["robata-production-real-sample-worker-v1"] = (
    "robata-production-real-sample-worker-v1"
)


class ProductionRealSampleWorkerErrorCode(StrEnum):
    """Stable fail-closed reasons emitted by this launcher boundary."""

    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    BOOTSTRAP_UNAVAILABLE = "BOOTSTRAP_UNAVAILABLE"
    SOURCE_UNPINNED = "SOURCE_UNPINNED"
    SOURCE_STAGING_FAILED = "SOURCE_STAGING_FAILED"
    SOURCE_PREPARATION_FAILED = "SOURCE_PREPARATION_FAILED"
    RUNTIME_CONSTRUCTION_FAILED = "RUNTIME_CONSTRUCTION_FAILED"
    CAPABILITY_PREFLIGHT_FAILED = "CAPABILITY_PREFLIGHT_FAILED"
    CANONICAL_EXECUTION_BRIDGE_UNAVAILABLE = "CANONICAL_EXECUTION_BRIDGE_UNAVAILABLE"
    CANONICAL_EXECUTION_FAILED = "CANONICAL_EXECUTION_FAILED"
    TRAFFIC_NOT_READY = "TRAFFIC_NOT_READY"
    TRAFFIC_ROUTE_PLAN_FAILED = "TRAFFIC_ROUTE_PLAN_FAILED"
    ARTIFACT_PERSISTENCE_FAILED = "ARTIFACT_PERSISTENCE_FAILED"
    E2E_COVERAGE_INCOMPLETE = "E2E_COVERAGE_INCOMPLETE"


class ProductionRealSampleWorkerError(RuntimeError):
    """A bounded production sample cannot safely continue."""

    def __init__(self, code: ProductionRealSampleWorkerErrorCode, detail: str) -> None:
        self.code = code
        super().__init__(detail)


class _TrafficBindingFailure(ProductionRealSampleWorkerError):
    """Fail-closed traffic admission while retaining its non-canonical evidence."""

    def __init__(
        self,
        code: ProductionRealSampleWorkerErrorCode,
        detail: str,
        *,
        readiness: ProductionTrafficReadiness | None = None,
        readiness_ref: WorkerArtifactReference | None = None,
    ) -> None:
        super().__init__(code, detail)
        self.readiness = readiness
        self.readiness_ref = readiness_ref


class ParticipationStatus(StrEnum):
    """Per-run participation; absence is not silently treated as zero cost."""

    PARTICIPATING = "PARTICIPATING"
    BYPASSED = "BYPASSED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    NOT_MEASURED = "NOT_MEASURED"
    FAILED = "FAILED"


class ProductionStage(StrEnum):
    """Stable boundaries in the production real-sample path."""

    BOOTSTRAP = "BOOTSTRAP"
    SOURCE = "SOURCE"
    R2 = "R2"
    SCHEDULING = "SCHEDULING"
    INFERENCE = "INFERENCE"
    EVIDENCE = "EVIDENCE"
    REDUCTION = "REDUCTION"
    COMPLETION = "COMPLETION"
    OUTBOX = "OUTBOX"
    PUBLICATION = "PUBLICATION"


class WorkerStageParticipation(StrictModel):
    """Observed or explicitly bypassed participation for one boundary."""

    stage: ProductionStage
    status: ParticipationStatus
    reason: NonEmptyString


class WorkerArtifactReference(StrictModel):
    """Reference to one non-canonical audit artifact."""

    kind: NonEmptyString
    path: NonEmptyString
    sha256: Sha256Digest
    byte_count: PositiveInt


class WorkerSourceObservation(StrictModel):
    """Pinned source facts retained in the operator report."""

    kind: Literal["LOCAL_PINNED", "R2_STAGED"]
    path: NonEmptyString
    sha256: Sha256Digest
    byte_count: PositiveInt
    media_type: NonEmptyString
    mapping_profile_id: NonEmptyString
    mapping_approval_status: NonEmptyString
    staged_reused_existing_file: bool = False


class WorkerExecutionObservation(StrictModel):
    """Safe projection returned by the explicit production execution driver."""

    execution_driver: NonEmptyString
    canonical_run_id: NonEmptyString | None = None
    output_refs: tuple[NonEmptyString, ...] = ()
    participation: tuple[WorkerStageParticipation, ...]
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_execution_participation(self) -> WorkerExecutionObservation:
        expected = (
            ProductionStage.SCHEDULING,
            ProductionStage.INFERENCE,
            ProductionStage.EVIDENCE,
            ProductionStage.REDUCTION,
            ProductionStage.COMPLETION,
            ProductionStage.OUTBOX,
            ProductionStage.PUBLICATION,
        )
        if tuple(item.stage for item in self.participation) != expected:
            raise ValueError(
                "execution participation must cover scheduling through publication in stable order"
            )
        return self


class ProductionRealSampleWorkerReport(StrictModel):
    """Non-canonical, deterministic report for one bounded production attempt."""

    report_version: Literal["robata-production-real-sample-worker-v1"] = WORKER_REPORT_VERSION
    run_id: NonEmptyString
    observed_at: NonEmptyString
    execution_class: Literal["PRODUCTION_REAL_SAMPLE"] = "PRODUCTION_REAL_SAMPLE"
    status: Literal["SUCCEEDED", "FAILED"]
    production_eligible: Literal[False] = False
    canonical_authority: Literal[False] = False
    bootstrap_path: NonEmptyString
    source: WorkerSourceObservation | None = None
    participation: tuple[WorkerStageParticipation, ...]
    participation_coverage: E2EParticipationCoverage
    participation_manifest: WorkerArtifactReference
    traffic_readiness: WorkerArtifactReference | None = None
    traffic_route_plan: WorkerArtifactReference | None = None
    trace: WorkerArtifactReference | None = None
    report_artifact: WorkerArtifactReference | None = None
    execution: WorkerExecutionObservation | None = None
    failure_code: NonEmptyString | None = None
    failure_detail: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_status_shape(self) -> ProductionRealSampleWorkerReport:
        if self.status == "SUCCEEDED":
            if self.failure_code is not None or self.failure_detail is not None:
                raise ValueError("successful report cannot contain failure fields")
            if self.execution is None:
                raise ValueError("successful report requires execution observation")
        elif self.failure_code is None or self.failure_detail is None:
            raise ValueError("failed report requires failure fields")
        return self


class ProductionRealSampleWorkerConfig(StrictModel):
    """Launch inputs; all source bytes must be explicitly pinned."""

    bootstrap_config_path: NonEmptyString
    mapping_config_path: NonEmptyString
    output_directory: NonEmptyString
    state_directory: NonEmptyString
    source_path: NonEmptyString | None = None
    source_manifest_path: NonEmptyString | None = None
    source_sha256: Sha256Digest | None = None
    source_byte_count: PositiveInt | None = None
    source_media_type: NonEmptyString = "application/x-mcap"
    allow_unapproved_profile: bool = False
    max_duration_ns: PositiveInt | None = None
    run_id: NonEmptyString | None = None
    e2e_participation: tuple[E2EParticipationDeclaration, ...] = ()

    @model_validator(mode="after")
    def validate_source_inputs(self) -> ProductionRealSampleWorkerConfig:
        if (self.source_path is None) == (self.source_manifest_path is None):
            raise ValueError("exactly one of source_path or source_manifest_path is required")
        if self.source_path is not None:
            if self.source_sha256 is None or self.source_byte_count is None:
                raise ValueError("local source requires source_sha256 and source_byte_count")
        elif self.source_sha256 is not None or self.source_byte_count is not None:
            raise ValueError("source digest/size belong only to a local pinned source")
        if self.e2e_participation and tuple(
            item.boundary for item in self.e2e_participation
        ) != tuple(E2EParticipationBoundary):
            raise ValueError("e2e_participation must cover every E2E boundary in stable order")
        return self


@dataclass(frozen=True, slots=True)
class ProductionSampleContext:
    """Inputs handed to an explicit canonical execution bridge."""

    run_id: str
    runtime: ProductionCanonicalRuntime
    source_path: Path
    source_bundle: CanonicalMcapSourceBundle
    bootstrap: ProductionRuntimeBootstrapConfiguration
    observer: RuntimeObserver
    traffic_bridge: ProductionTrafficBridge | None = None
    traffic_readiness: ProductionTrafficReadiness | None = None
    traffic_route_plan: ProductionTrafficRoutePlan | None = None
    traffic_readiness_ref: WorkerArtifactReference | None = None
    traffic_route_plan_ref: WorkerArtifactReference | None = None


class ProductionCanonicalExecutionDriver(Protocol):
    """Port for the source-specific scheduler/inference/evidence bridge.

    The production adapters intentionally expose lower-level repositories.  A
    driver must bind one admitted source bundle to those repositories and return
    only non-canonical observations.  The default worker has no implementation
    and therefore refuses to invent a completion or publication.
    """

    def preflight(self, *, runtime: ProductionCanonicalRuntime) -> None:
        """Verify that all required production stage methods are available."""

    def execute(self, *, context: ProductionSampleContext) -> WorkerExecutionObservation:
        """Execute one bounded source-specific production run."""


class UnavailableProductionCanonicalExecutionDriver:
    """Default driver which documents the missing bridge and fails closed."""

    def preflight(self, *, runtime: ProductionCanonicalRuntime) -> None:
        del runtime
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_BRIDGE_UNAVAILABLE,
            "no source-specific production bridge is registered; mapping the MCAP source bundle "
            "to PostgresWorkScheduler, RunPod inference, evidence ledger, reduction/barrier, "
            "primary completion, outbox, and publication is required before real production "
            "data may be marked successful",
        )

    def execute(self, *, context: ProductionSampleContext) -> WorkerExecutionObservation:
        del context
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_BRIDGE_UNAVAILABLE,
            "canonical execution bridge is unavailable",
        )


RuntimeBuilder = Callable[
    [ProductionRuntimeBootstrapConfiguration, Mapping[str, str], RuntimeObserver],
    ProductionCanonicalRuntime,
]


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be configured")
    return value


def _default_runtime_builder(
    bootstrap: ProductionRuntimeBootstrapConfiguration,
    environment: Mapping[str, str],
    observer: RuntimeObserver,
) -> ProductionCanonicalRuntime:
    """Construct the reviewed production graph; no local fallback is allowed."""

    registry = default_schema_registry()
    r2_object_store = create_r2_object_store_from_environment(
        environment,
        runtime_observer=observer,
    )
    contract = ProductionCompositionContract(
        canonical_postgres=CanonicalPostgresRuntimeConfig.from_environment(environment),
        r2=r2_object_store.config,
        pgvector=PgVectorRuntimeConfig.from_environment(environment),
        primary_runpod=bootstrap.primary_binding,
    )
    pgvector_projection = create_verified_pgvector_projection_store_from_environment(environment)

    def primary_adapter_factory(raw_store: PostgresInferenceEvidenceLedger) -> RunPodVisionAdapter:
        return RunPodVisionAdapter(
            config=bootstrap.primary_binding.endpoint,
            credential=RunPodApiKey(_required(environment, "RUNPOD_API_KEY")),
            capabilities=bootstrap.primary_capabilities,
            retry_policy=bootstrap.primary_retry_policy,
            raw_store=raw_store,
            parser=StrictProviderClaimParser(
                registry,
                parser_version=bootstrap.primary_parser_version,
            ),
            runtime_observer=observer,
        )

    return build_production_canonical_runtime(
        contract=contract,
        credentials=CanonicalPostgresRuntimeCredentials.from_environment(environment),
        tenant=ProductionTenantContext(tenant_id=_required(environment, "ROBATA_TENANT_ID")),
        capture_authority=bootstrap.capture_authority,
        r2_object_store=r2_object_store,
        pgvector_projection=pgvector_projection,
        primary_adapter_factory=primary_adapter_factory,
        primary_route=bootstrap.primary_route,
        release_verifier=bootstrap.release_verifier(),
        outbox_retry_policy=bootstrap.outbox_retry_policy(),
        schema_registry=registry,
        runtime_observer=observer,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _observational_json_bytes(value: object) -> bytes:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (serialized + "\n").encode("utf-8")


def _artifact_reference(kind: str, path: Path, payload: bytes) -> WorkerArtifactReference:
    return WorkerArtifactReference(
        kind=kind,
        path=str(path),
        sha256=exact_bytes_sha256(payload),
        byte_count=len(payload),
    )


def _write_traffic_sidecars(
    *,
    bridge: ProductionTrafficBridge,
    source_sha256: Sha256Digest,
    readiness_path: Path,
    route_plan_path: Path,
) -> tuple[
    ProductionTrafficReadiness,
    ProductionTrafficRoutePlan,
    WorkerArtifactReference,
    WorkerArtifactReference,
]:
    """Admit one source identity and persist only non-canonical traffic facts.

    ``ProductionTrafficBridge`` owns release, canary, and shadow validation.
    The worker deliberately calls ``require_ready`` before asking for a route
    plan; a readiness failure therefore cannot be mistaken for a partially
    executed production attempt.  Both sidecars are content-addressed by the
    returned references and neither is a canonical result.
    """

    try:
        readiness = bridge.require_ready()
    except ProductionTrafficError as traffic_error:
        # Persist the observed blockers before propagating a fail-closed
        # launcher error.  This is useful operator evidence and does not grant
        # authority to a not-ready route.
        readiness = bridge.readiness
        payload = _observational_json_bytes(readiness.model_dump(mode="json"))
        try:
            _atomic_write(readiness_path, payload)
        except Exception as error:
            raise ProductionRealSampleWorkerError(
                ProductionRealSampleWorkerErrorCode.ARTIFACT_PERSISTENCE_FAILED,
                f"traffic readiness sidecar persistence failed: {type(error).__name__}",
            ) from error
        readiness_ref = _artifact_reference("TRAFFIC_READINESS", readiness_path, payload)
        raise _TrafficBindingFailure(
            ProductionRealSampleWorkerErrorCode.TRAFFIC_NOT_READY,
            "production traffic bridge is not ready: " + ", ".join(readiness.blockers),
            readiness=readiness,
            readiness_ref=readiness_ref,
        ) from traffic_error

    readiness_payload = _observational_json_bytes(readiness.model_dump(mode="json"))
    try:
        _atomic_write(readiness_path, readiness_payload)
    except Exception as error:
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.ARTIFACT_PERSISTENCE_FAILED,
            f"traffic readiness sidecar persistence failed: {type(error).__name__}",
        ) from error

    readiness_ref = _artifact_reference("TRAFFIC_READINESS", readiness_path, readiness_payload)
    try:
        route_plan = bridge.route_plan(input_identity_sha256=source_sha256)
    except ProductionTrafficError as error:
        raise _TrafficBindingFailure(
            ProductionRealSampleWorkerErrorCode.TRAFFIC_ROUTE_PLAN_FAILED,
            f"production traffic route plan failed: {error}",
            readiness=readiness,
            readiness_ref=readiness_ref,
        ) from error
    except Exception as error:
        raise _TrafficBindingFailure(
            ProductionRealSampleWorkerErrorCode.TRAFFIC_ROUTE_PLAN_FAILED,
            f"production traffic route plan failed: {type(error).__name__}",
            readiness=readiness,
            readiness_ref=readiness_ref,
        ) from error
    route_plan_payload = _observational_json_bytes(route_plan.model_dump(mode="json"))
    try:
        _atomic_write(route_plan_path, route_plan_payload)
    except Exception as error:
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.ARTIFACT_PERSISTENCE_FAILED,
            f"traffic route-plan sidecar persistence failed: {type(error).__name__}",
        ) from error

    return (
        readiness,
        route_plan,
        _artifact_reference("TRAFFIC_READINESS", readiness_path, readiness_payload),
        _artifact_reference("TRAFFIC_ROUTE_PLAN", route_plan_path, route_plan_payload),
    )


def _verify_runtime_boundary_shape(runtime: object) -> None:
    """Reject a runtime graph that cannot expose every required production edge."""

    required_attributes = (
        "work_scheduler",
        "capture_authority",
        "primary_adapter",
        "inference_evidence",
        "barrier_storage",
        "primary_completion",
        "outbox_delivery",
        "read_model",
        "r2_object_store",
    )
    missing = [name for name in required_attributes if getattr(runtime, name, None) is None]
    if missing:
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.CAPABILITY_PREFLIGHT_FAILED,
            "production runtime is missing required boundaries: " + ", ".join(missing),
        )


def _initial_participation() -> list[WorkerStageParticipation]:
    return [
        WorkerStageParticipation(
            stage=stage,
            status=ParticipationStatus.NOT_MEASURED,
            reason="stage has not executed",
        )
        for stage in ProductionStage
    ]


def _replace_participation(
    values: list[WorkerStageParticipation],
    stage: ProductionStage,
    status: ParticipationStatus,
    reason: str,
) -> None:
    for index, item in enumerate(values):
        if item.stage is stage:
            values[index] = WorkerStageParticipation(stage=stage, status=status, reason=reason)
            return
    raise AssertionError(f"unknown production stage: {stage}")


_COMPONENT_BOUNDARY_GROUPS: dict[E2EParticipationBoundary, tuple[ProductionStage, ...]] = {
    E2EParticipationBoundary.SCHEDULING: (ProductionStage.SCHEDULING,),
    E2EParticipationBoundary.INFERENCE: (ProductionStage.INFERENCE,),
    E2EParticipationBoundary.EVIDENCE: (ProductionStage.EVIDENCE,),
    E2EParticipationBoundary.REDUCTION: (ProductionStage.REDUCTION,),
    E2EParticipationBoundary.PUBLICATION: (
        ProductionStage.COMPLETION,
        ProductionStage.OUTBOX,
        ProductionStage.PUBLICATION,
    ),
}


def _reconcile_component_participation(
    *,
    execution: WorkerExecutionObservation | None,
    declarations: tuple[E2EParticipationDeclaration, ...],
) -> tuple[str, ...]:
    """Ensure coarse E2E declarations cannot hide omitted production edges.

    The trace has one coarse ``PUBLICATION`` boundary while the execution
    driver reports completion, outbox, and publication separately.  A coarse
    span therefore only qualifies when all of its component edges agree with
    the declared state.  This is an audit invariant, not a canonical result.
    """

    if execution is None:
        return ()
    declaration_by_boundary = {item.boundary: item for item in declarations}
    component_by_stage = {item.stage: item for item in execution.participation}
    issues: list[str] = []
    for boundary, stages in _COMPONENT_BOUNDARY_GROUPS.items():
        declaration = declaration_by_boundary[boundary]
        components = tuple(component_by_stage[stage] for stage in stages)
        statuses = tuple(item.status for item in components)
        if any(status is ParticipationStatus.FAILED for status in statuses):
            issues.append(f"{boundary.value}: component execution reported FAILED")
            continue
        if declaration.state is E2EParticipationState.PARTICIPATING:
            if any(status is not ParticipationStatus.PARTICIPATING for status in statuses):
                rendered = ", ".join(
                    f"{stage.value}={component_by_stage[stage].status.value}" for stage in stages
                )
                issues.append(
                    f"{boundary.value}: declared PARTICIPATING but components are {rendered}"
                )
        elif declaration.state is E2EParticipationState.BYPASSED:
            if any(status is ParticipationStatus.PARTICIPATING for status in statuses):
                rendered = ", ".join(
                    f"{stage.value}={component_by_stage[stage].status.value}" for stage in stages
                )
                issues.append(f"{boundary.value}: declared BYPASSED but components are {rendered}")
            elif any(status is not ParticipationStatus.BYPASSED for status in statuses):
                rendered = ", ".join(
                    f"{stage.value}={component_by_stage[stage].status.value}" for stage in stages
                )
                issues.append(f"{boundary.value}: BYPASSED declaration is not proven by {rendered}")
        elif declaration.state is E2EParticipationState.NOT_CONFIGURED:
            if any(status is ParticipationStatus.PARTICIPATING for status in statuses):
                rendered = ", ".join(
                    f"{stage.value}={component_by_stage[stage].status.value}" for stage in stages
                )
                issues.append(
                    f"{boundary.value}: declared NOT_CONFIGURED but components are {rendered}"
                )
            elif any(status is not ParticipationStatus.NOT_CONFIGURED for status in statuses):
                rendered = ", ".join(
                    f"{stage.value}={component_by_stage[stage].status.value}" for stage in stages
                )
                issues.append(
                    f"{boundary.value}: NOT_CONFIGURED declaration is not proven by {rendered}"
                )
        else:
            issues.append(f"{boundary.value}: declaration is FAILED")
    return tuple(issues)


def _default_e2e_participation() -> tuple[E2EParticipationDeclaration, ...]:
    return tuple(
        E2EParticipationDeclaration(
            boundary=boundary,
            state=E2EParticipationState.PARTICIPATING,
            required=True,
        )
        for boundary in E2EParticipationBoundary
    )


def _replace_e2e_declaration(
    values: list[E2EParticipationDeclaration],
    boundary: E2EParticipationBoundary,
    state: E2EParticipationState,
    reason: str,
) -> None:
    for index, item in enumerate(values):
        if item.boundary is boundary:
            values[index] = E2EParticipationDeclaration(
                boundary=boundary,
                state=state,
                required=item.required,
                reason=reason,
            )
            return
    raise AssertionError(f"unknown E2E participation boundary: {boundary}")


def _verify_local_source(
    config: ProductionRealSampleWorkerConfig,
) -> tuple[Path, WorkerSourceObservation, None]:
    assert config.source_path is not None
    source = Path(config.source_path)
    if source.is_symlink() or not source.is_file():
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.SOURCE_UNPINNED,
            "local source must be an existing regular file",
        )
    try:
        stat = source.stat()
        payload = source.read_bytes()
    except OSError as error:
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.SOURCE_PREPARATION_FAILED,
            f"cannot read local source: {error}",
        ) from error
    assert config.source_sha256 is not None
    assert config.source_byte_count is not None
    actual_sha = exact_bytes_sha256(payload)
    if len(payload) != config.source_byte_count or actual_sha != config.source_sha256:
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.SOURCE_UNPINNED,
            "local source bytes do not match the pinned digest and byte count",
        )
    if stat.st_size != config.source_byte_count:
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.SOURCE_UNPINNED,
            "local source stat size differs from the pinned byte count",
        )
    return (
        source,
        WorkerSourceObservation(
            kind="LOCAL_PINNED",
            path=str(source),
            sha256=actual_sha,
            byte_count=len(payload),
            media_type=config.source_media_type,
            mapping_profile_id="UNLOADED",
            mapping_approval_status="UNLOADED",
        ),
        None,
    )


def _stage_r2_source(
    config: ProductionRealSampleWorkerConfig,
    runtime: ProductionCanonicalRuntime,
) -> tuple[Path, WorkerSourceObservation, R2McapSourceStageReceipt]:
    assert config.source_manifest_path is not None
    try:
        manifest = load_r2_mcap_source_manifest(config.source_manifest_path)
        destination = (
            Path(config.state_directory) / "staged-source" / f"{manifest.expected_sha256}.mcap"
        )
        receipt = stage_r2_mcap_source(
            manifest=manifest,
            object_store=runtime.r2_object_store,
            destination=destination,
        )
    except Exception as error:
        raise ProductionRealSampleWorkerError(
            ProductionRealSampleWorkerErrorCode.SOURCE_STAGING_FAILED,
            f"pinned R2 MCAP source could not be staged: {type(error).__name__}",
        ) from error
    return (
        receipt.destination,
        WorkerSourceObservation(
            kind="R2_STAGED",
            path=str(receipt.destination),
            sha256=receipt.content_sha256,
            byte_count=receipt.byte_count,
            media_type=receipt.manifest.expected_media_type,
            mapping_profile_id="UNLOADED",
            mapping_approval_status="UNLOADED",
            staged_reused_existing_file=receipt.reused_existing_file,
        ),
        receipt,
    )


@dataclass(frozen=True, slots=True)
class ProductionRealSampleWorkerResult:
    """Report plus paths written by one worker attempt."""

    report: ProductionRealSampleWorkerReport
    report_path: Path
    trace_path: Path
    participation_path: Path
    component_participation_path: Path
    traffic_readiness_path: Path | None = None
    traffic_route_plan_path: Path | None = None
    traffic_readiness_ref: WorkerArtifactReference | None = None
    traffic_route_plan_ref: WorkerArtifactReference | None = None


class ProductionRealSampleWorker:
    """Run one bounded, source-specific production attempt."""

    def __init__(
        self,
        *,
        config: ProductionRealSampleWorkerConfig,
        environment: Mapping[str, str] | None = None,
        runtime_builder: RuntimeBuilder | None = None,
        execution_driver: ProductionCanonicalExecutionDriver | None = None,
        traffic_bridge: ProductionTrafficBridge | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, ProductionRealSampleWorkerConfig)
            else ProductionRealSampleWorkerConfig.model_validate(config, strict=True)
        )
        self.environment = dict(os.environ if environment is None else environment)
        self.runtime_builder = (
            _default_runtime_builder if runtime_builder is None else runtime_builder
        )
        self.execution_driver = (
            UnavailableProductionCanonicalExecutionDriver()
            if execution_driver is None
            else execution_driver
        )
        if traffic_bridge is not None and not isinstance(traffic_bridge, ProductionTrafficBridge):
            raise TypeError("traffic_bridge must be a ProductionTrafficBridge")
        self.traffic_bridge = traffic_bridge

    def run(self) -> ProductionRealSampleWorkerResult:
        run_id = self.config.run_id or str(uuid4())
        output_root = Path(self.config.output_directory) / run_id
        report_path = output_root / "report.json"
        trace_path = output_root / "trace.json"
        participation_path = output_root / "participation.json"
        component_participation_path = output_root / "component-participation.json"
        traffic_readiness_path = output_root / "traffic-readiness.json"
        traffic_route_plan_path = output_root / "traffic-route-plan.json"
        recorder = RuntimeProfileRecorder()
        participation = _initial_participation()
        e2e_declarations = list(self.config.e2e_participation or _default_e2e_participation())
        source_observation: WorkerSourceObservation | None = None
        execution: WorkerExecutionObservation | None = None
        traffic_readiness: ProductionTrafficReadiness | None = None
        traffic_route_plan: ProductionTrafficRoutePlan | None = None
        traffic_readiness_ref: WorkerArtifactReference | None = None
        traffic_route_plan_ref: WorkerArtifactReference | None = None
        component_reconciliation_issues: tuple[str, ...] = ()
        failure_code: str | None = None
        failure_detail: str | None = None
        bootstrap_path = str(Path(self.config.bootstrap_config_path))

        try:
            with runtime_span(recorder, "runtime.production_real_sample"):
                try:
                    bootstrap = load_production_runtime_bootstrap_configuration(
                        Path(self.config.bootstrap_config_path)
                    )
                except Exception as error:
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.BOOTSTRAP_UNAVAILABLE,
                        "reviewed production bootstrap could not be loaded: "
                        f"{type(error).__name__}",
                    ) from error
                _replace_participation(
                    participation,
                    ProductionStage.BOOTSTRAP,
                    ParticipationStatus.PARTICIPATING,
                    "reviewed production bootstrap loaded",
                )
                try:
                    with runtime_span(recorder, "runtime.production_runtime_construct"):
                        runtime = self.runtime_builder(bootstrap, self.environment, recorder)
                except ProductionRealSampleWorkerError:
                    raise
                except Exception as error:
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.RUNTIME_CONSTRUCTION_FAILED,
                        f"production runtime construction failed: {type(error).__name__}",
                    ) from error
                _verify_runtime_boundary_shape(runtime)
                if self.config.source_path is not None:
                    try:
                        source_path, source_observation, _ = _verify_local_source(self.config)
                    except ProductionRealSampleWorkerError:
                        _replace_participation(
                            participation,
                            ProductionStage.SOURCE,
                            ParticipationStatus.FAILED,
                            "local source pin verification failed",
                        )
                        raise
                    _replace_participation(
                        participation,
                        ProductionStage.R2,
                        ParticipationStatus.BYPASSED,
                        "local pinned source selected; R2 staging is explicitly bypassed",
                    )
                else:
                    try:
                        source_path, source_observation, _ = _stage_r2_source(
                            self.config,
                            runtime,
                        )
                    except ProductionRealSampleWorkerError:
                        _replace_participation(
                            participation,
                            ProductionStage.R2,
                            ParticipationStatus.FAILED,
                            "pinned R2 source staging failed",
                        )
                        raise
                    _replace_participation(
                        participation,
                        ProductionStage.R2,
                        ParticipationStatus.PARTICIPATING,
                        "pinned R2 source was staged and verified",
                    )
                if source_observation is None:
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.SOURCE_PREPARATION_FAILED,
                        "source observation was not created before traffic admission",
                    )
                # The source digest is pinned before mapping/materialization.
                # Admit traffic at that boundary so a not-ready route fails
                # closed even when source preparation would otherwise proceed.
                if self.traffic_bridge is not None:
                    (
                        traffic_readiness,
                        traffic_route_plan,
                        traffic_readiness_ref,
                        traffic_route_plan_ref,
                    ) = _write_traffic_sidecars(
                        bridge=self.traffic_bridge,
                        source_sha256=source_observation.sha256,
                        readiness_path=traffic_readiness_path,
                        route_plan_path=traffic_route_plan_path,
                    )
                try:
                    with runtime_span(recorder, "source.mapping.read_validate"):
                        authorization = authorize_mcap_mapping(
                            Path(self.config.mapping_config_path),
                            allow_unapproved_profile=self.config.allow_unapproved_profile,
                        )
                except Exception as error:
                    _replace_participation(
                        participation,
                        ProductionStage.SOURCE,
                        ParticipationStatus.FAILED,
                        "MCAP mapping authorization failed",
                    )
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.SOURCE_PREPARATION_FAILED,
                        f"MCAP mapping authorization failed: {type(error).__name__}",
                    ) from error
                source_observation = source_observation.model_copy(
                    update={
                        "mapping_profile_id": authorization.profile.profile_id,
                        "mapping_approval_status": authorization.profile.approval_status,
                    }
                )
                try:
                    with runtime_span(recorder, "source.mcap.prepare"):
                        source_bundle = load_canonical_mcap_source(
                            source_path,
                            authorization=authorization,
                            state_dir=Path(self.config.state_directory),
                            expected_source_sha256=source_observation.sha256,
                            max_duration_ns=self.config.max_duration_ns,
                            schema_registry=default_schema_registry(),
                            runtime_observer=recorder,
                        )
                except Exception as error:
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.SOURCE_PREPARATION_FAILED,
                        f"canonical MCAP source preparation failed: {type(error).__name__}",
                    ) from error
                _replace_participation(
                    participation,
                    ProductionStage.SOURCE,
                    ParticipationStatus.PARTICIPATING,
                    "pinned MCAP inspected, admitted, indexed, and materialized",
                )
                try:
                    with runtime_span(recorder, "runtime.production_capability_preflight"):
                        self.execution_driver.preflight(runtime=runtime)
                except ProductionRealSampleWorkerError as error:
                    unavailable = error.code is (
                        ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_BRIDGE_UNAVAILABLE
                    )
                    for stage in (
                        ProductionStage.SCHEDULING,
                        ProductionStage.INFERENCE,
                        ProductionStage.EVIDENCE,
                        ProductionStage.REDUCTION,
                        ProductionStage.COMPLETION,
                        ProductionStage.OUTBOX,
                        ProductionStage.PUBLICATION,
                    ):
                        _replace_participation(
                            participation,
                            stage,
                            (
                                ParticipationStatus.NOT_CONFIGURED
                                if unavailable
                                else ParticipationStatus.FAILED
                            ),
                            str(error),
                        )
                    raise
                except Exception as error:
                    _replace_participation(
                        participation,
                        ProductionStage.SCHEDULING,
                        ParticipationStatus.FAILED,
                        "production execution capability preflight failed",
                    )
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.CAPABILITY_PREFLIGHT_FAILED,
                        f"production execution capability preflight failed: {type(error).__name__}",
                    ) from error
                try:
                    execution = self.execution_driver.execute(
                        context=ProductionSampleContext(
                            run_id=run_id,
                            runtime=runtime,
                            source_path=source_path,
                            source_bundle=source_bundle,
                            bootstrap=bootstrap,
                            observer=recorder,
                            traffic_bridge=self.traffic_bridge,
                            traffic_readiness=traffic_readiness,
                            traffic_route_plan=traffic_route_plan,
                            traffic_readiness_ref=traffic_readiness_ref,
                            traffic_route_plan_ref=traffic_route_plan_ref,
                        )
                    )
                except ProductionRealSampleWorkerError:
                    raise
                except Exception as error:
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_FAILED,
                        f"production canonical execution failed: {type(error).__name__}",
                    ) from error
                if not isinstance(execution, WorkerExecutionObservation):
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_FAILED,
                        "execution driver returned an invalid observation",
                    )
                for item in execution.participation:
                    _replace_participation(
                        participation,
                        item.stage,
                        item.status,
                        item.reason,
                    )
                failed_components = tuple(
                    item.stage.value
                    for item in execution.participation
                    if item.status is ParticipationStatus.FAILED
                )
                if failed_components:
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_FAILED,
                        "execution driver reported failed production boundaries: "
                        + ", ".join(failed_components),
                    )
                if execution.canonical_run_id is None or not execution.output_refs:
                    raise ProductionRealSampleWorkerError(
                        ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_FAILED,
                        "execution driver did not provide canonical_run_id and output_refs "
                        "receipts required for a successful attempt",
                    )
        except ProductionRealSampleWorkerError as error:
            if isinstance(error, _TrafficBindingFailure):
                traffic_readiness = error.readiness
                traffic_readiness_ref = error.readiness_ref
            failure_code = error.code.value
            failure_detail = str(error)
        except Exception as error:
            failure_code = ProductionRealSampleWorkerErrorCode.CANONICAL_EXECUTION_FAILED.value
            failure_detail = f"{type(error).__name__}: {error}"

        profile = recorder.snapshot()
        trace = build_e2e_trace_runtime_fragment(
            role=E2ETraceFragmentRole.LAUNCHER,
            runtime_profile=profile,
        )
        trace_payload = _observational_json_bytes(trace.model_dump(mode="json"))
        _atomic_write(trace_path, trace_payload)
        trace_ref = _artifact_reference("E2E_TRACE", trace_path, trace_payload)

        if failure_code is not None:
            _replace_e2e_declaration(
                e2e_declarations,
                E2EParticipationBoundary.ORCHESTRATION,
                E2EParticipationState.FAILED,
                failure_detail or failure_code,
            )
        observed_at = _utc_now()
        e2e_manifest = build_e2e_participation_manifest(
            runtime_fragment=trace,
            declarations=tuple(e2e_declarations),
            observed_at=observed_at,
        )
        participation_digest = write_e2e_participation_manifest(
            e2e_manifest,
            participation_path,
        )
        participation_ref = WorkerArtifactReference(
            kind="E2E_PARTICIPATION",
            path=str(participation_path),
            sha256=participation_digest,
            byte_count=participation_path.stat().st_size,
        )
        component_reconciliation_issues = _reconcile_component_participation(
            execution=execution,
            declarations=tuple(e2e_declarations),
        )
        if failure_code is None and component_reconciliation_issues:
            failure_code = ProductionRealSampleWorkerErrorCode.E2E_COVERAGE_INCOMPLETE.value
            failure_detail = (
                "production component participation does not reconcile with coarse E2E "
                "boundaries: " + "; ".join(component_reconciliation_issues)
            )
        if failure_code is None and e2e_manifest.coverage is not E2EParticipationCoverage.COMPLETE:
            failure_code = ProductionRealSampleWorkerErrorCode.E2E_COVERAGE_INCOMPLETE.value
            issue_codes = ", ".join(item.code.value for item in e2e_manifest.issues)
            failure_detail = (
                "production E2E participation coverage is "
                f"{e2e_manifest.coverage.value}: {issue_codes or 'unknown coverage issue'}"
            )
        status: Literal["SUCCEEDED", "FAILED"] = "SUCCEEDED" if failure_code is None else "FAILED"
        if status == "SUCCEEDED":
            assert execution is not None
        report = ProductionRealSampleWorkerReport(
            run_id=run_id,
            observed_at=observed_at,
            status=status,
            bootstrap_path=bootstrap_path,
            source=source_observation,
            participation=tuple(participation),
            participation_coverage=e2e_manifest.coverage,
            participation_manifest=participation_ref,
            traffic_readiness=traffic_readiness_ref,
            traffic_route_plan=traffic_route_plan_ref,
            trace=trace_ref,
            execution=execution,
            failure_code=failure_code,
            failure_detail=failure_detail,
        )
        report_payload = _observational_json_bytes(report.model_dump(mode="json"))
        _atomic_write(report_path, report_payload)
        report_ref = _artifact_reference("WORKER_REPORT", report_path, report_payload)
        component_payload = _observational_json_bytes(
            {
                "format_version": WORKER_REPORT_VERSION,
                "run_id": run_id,
                "status": status,
                "stages": [item.model_dump(mode="json") for item in participation],
                "report_sha256": report_ref.sha256,
                "trace_sha256": trace_ref.sha256,
                "participation_sha256": participation_ref.sha256,
                "traffic_readiness_sha256": (
                    traffic_readiness_ref.sha256 if traffic_readiness_ref is not None else None
                ),
                "traffic_route_plan_sha256": (
                    traffic_route_plan_ref.sha256 if traffic_route_plan_ref is not None else None
                ),
                "component_reconciliation_issues": list(component_reconciliation_issues),
            }
        )
        _atomic_write(component_participation_path, component_payload)
        return ProductionRealSampleWorkerResult(
            report=report,
            report_path=report_path,
            trace_path=trace_path,
            participation_path=participation_path,
            component_participation_path=component_participation_path,
            traffic_readiness_path=(traffic_readiness_path if traffic_readiness_ref else None),
            traffic_route_plan_path=(traffic_route_plan_path if traffic_route_plan_ref else None),
            traffic_readiness_ref=traffic_readiness_ref,
            traffic_route_plan_ref=traffic_route_plan_ref,
        )


def run_production_real_sample(
    config: ProductionRealSampleWorkerConfig,
    *,
    environment: Mapping[str, str] | None = None,
    runtime_builder: RuntimeBuilder | None = None,
    execution_driver: ProductionCanonicalExecutionDriver | None = None,
    traffic_bridge: ProductionTrafficBridge | None = None,
) -> ProductionRealSampleWorkerResult:
    """Convenience wrapper used by tests and the launcher script."""

    return ProductionRealSampleWorker(
        config=config,
        environment=environment,
        runtime_builder=runtime_builder,
        execution_driver=execution_driver,
        traffic_bridge=traffic_bridge,
    ).run()


__all__ = [
    "ParticipationStatus",
    "ProductionCanonicalExecutionDriver",
    "ProductionRealSampleWorker",
    "ProductionRealSampleWorkerConfig",
    "ProductionRealSampleWorkerError",
    "ProductionRealSampleWorkerErrorCode",
    "ProductionRealSampleWorkerReport",
    "ProductionRealSampleWorkerResult",
    "ProductionSampleContext",
    "ProductionStage",
    "UnavailableProductionCanonicalExecutionDriver",
    "WorkerArtifactReference",
    "WorkerExecutionObservation",
    "WorkerSourceObservation",
    "WorkerStageParticipation",
    "run_production_real_sample",
]
