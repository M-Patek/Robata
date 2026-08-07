"""Deterministic local rehearsal for the first Mage/Qwen qualification run.

The rehearsal executes the real P16 router and P17 experiment coordinator with
three isolated deterministic adapters: one production Qwen call plus paired
Qwen and Mage observations.  Its comparison sidecar is the same P17 artifact
that P18 quality evidence later binds to.  It deliberately does *not* invent
provider saturation, governed labels, cost, or release evidence, so the report
is permanently non-promotional and lists M0-M5 as external work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.common import StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.inference.adapter import (
    JsonSchemaRef,
    NormalizedOutputEnvelope,
    PackageInput,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.experiment_execution import (
    ExperimentComparisonStatus,
    ExperimentDeploymentBinding,
    ExperimentDeploymentRegistry,
    ExperimentExecutionCoordinator,
    ExperimentInvocation,
    ExperimentPairComparison,
    ExperimentTargetInput,
)
from robata.inference.input_plan import (
    INFERENCE_INPUT_PLANNER_VERSION,
    ApplicableProviderLimits,
    CallPartSpec,
    CatalogCamera,
    CatalogFrame,
    CatalogPackage,
    FrameTransform,
    InferenceInputPlan,
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    RequestCatalog,
    TransformOperation,
)
from robata.inference.models import (
    ConcurrencyClass,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    ModelInference,
    VisionTask,
)
from robata.inference.orchestrator import (
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
    ModelRouter,
    ProductionRoute,
    ProductionRouteAuthorization,
    RouteMode,
    RoutePlane,
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]

LOCAL_MODEL_COMPARISON_DRY_RUN_VERSION = "local-model-comparison-dry-run-v1"
LOCAL_MODEL_COMPARISON_EXTERNAL_GATES = (
    "M0_DEPLOYMENT_FREEZE",
    "M1_SOURCE_AND_REPRESENTATION",
    "M2_PAIRED_QUALITY",
    "M3_PROVIDER_SATURATION",
    "M4_RELIABILITY_AND_CANARY",
    "M5_INDEPENDENT_RELEASE",
)

_NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
_NOW_TEXT = "2026-07-30T12:00:00Z"
_TASK = VisionTask.ACTION_EVIDENCE
_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["label"],
    "properties": {"label": {"type": "string"}},
}
_SCHEMA_REF = JsonSchemaRef(
    schema_id="local-action-output",
    version="1.0",
    artifact_id="local-action-output-schema",
    sha256=semantic_sha256(_SCHEMA),
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


class LocalP18ComparisonReadiness(StrictModel):
    """A non-promotional record of what the local P17 sidecar can prove.

    ``FairLoadModelComparisonReport`` intentionally requires measured P6 and
    governed quality evidence.  This record preserves its exact P17 input but
    prevents a deterministic adapter run from being presented as that report.
    """

    readiness_version: Literal["local-p18-comparison-readiness-v1"] = (
        "local-p18-comparison-readiness-v1"
    )
    comparison_id: NonEmptyString
    comparison_status: ExperimentComparisonStatus
    comparison_comparable: bool
    fair_load_evidence_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    quality_evidence_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    cost_evidence_status: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    fair_load_report_emitted: Literal[False] = False
    candidate_authority: Literal[False] = False
    unresolved_external_gates: tuple[NonEmptyString, ...] = LOCAL_MODEL_COMPARISON_EXTERNAL_GATES
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_external_limits(self) -> Self:
        if self.unresolved_external_gates != LOCAL_MODEL_COMPARISON_EXTERNAL_GATES:
            raise ValueError("local readiness must retain every unresolved P20 external gate")
        return self


class LocalModelComparisonDryRunReport(StrictModel):
    """Machine-readable output for a deterministic no-network local rehearsal."""

    report_version: Literal["local-model-comparison-dry-run-v1"] = (
        "local-model-comparison-dry-run-v1"
    )
    execution_class: Literal["LOCAL_CONFORMANCE"] = "LOCAL_CONFORMANCE"
    production_decision: ModelRouteDecision
    experiment_decision: ModelRouteDecision
    experiment_contract: ExperimentContract
    production_terminal: ModelInference
    control_terminal: ModelInference
    candidate_terminal: ModelInference
    comparison: ExperimentPairComparison
    comparison_id: NonEmptyString
    comparison_status: ExperimentComparisonStatus
    comparison_comparable: bool
    comparison_field_delta_count: NonNegativeInt
    deterministic_adapter_call_count: NonNegativeInt
    network_call_count: Literal[0] = 0
    p18_readiness: LocalP18ComparisonReadiness
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def validate_rehearsal_boundaries(self) -> Self:
        if (
            self.production_decision.plane is not RoutePlane.PRODUCTION
            or self.production_decision.mode is not RouteMode.PRIMARY
            or len(self.production_decision.dispatches) != 1
            or self.production_decision.dispatches[0].disposition
            is not DispatchDisposition.AUTHORITATIVE
        ):
            raise ValueError("local dry-run requires exactly one production authority")
        if (
            self.experiment_decision.plane is not RoutePlane.EXPERIMENT
            or self.experiment_decision.mode is not RouteMode.PAIRED
            or self.experiment_decision.experiment_id != self.experiment_contract.experiment_id
            or any(
                dispatch.disposition is not DispatchDisposition.OBSERVATION
                for dispatch in self.experiment_decision.dispatches
            )
        ):
            raise ValueError("local dry-run experiment must remain observation-only")
        if self.production_terminal.shadow:
            raise ValueError("local production terminal cannot be a shadow")
        if not self.control_terminal.shadow or not self.candidate_terminal.shadow:
            raise ValueError("local experiment terminals must remain shadows")
        if (
            self.comparison.comparison_id != self.comparison_id
            or self.comparison.status is not self.comparison_status
            or self.comparison.comparable is not self.comparison_comparable
            or len(self.comparison.field_deltas) != self.comparison_field_delta_count
        ):
            raise ValueError("local dry-run comparison summary must retain the P17 sidecar")
        if self.comparison_id != self.p18_readiness.comparison_id:
            raise ValueError("P18 readiness must bind the emitted P17 comparison")
        if (
            self.comparison_status is not self.p18_readiness.comparison_status
            or self.comparison_comparable is not self.p18_readiness.comparison_comparable
        ):
            raise ValueError("P18 readiness must retain the comparison outcome")
        if self.deterministic_adapter_call_count != 3:
            raise ValueError("local dry-run must execute one production and two experiment calls")
        return self


@dataclass(slots=True)
class _DeterministicVisionAdapter:
    """A provider-shaped adapter that cannot make an external request."""

    name: str
    capabilities_snapshot: ModelCapabilities
    payload: dict[str, object]
    infer_calls: int = 0
    network_calls: int = 0
    provider: str = "runpod"

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        if (model_name, model_version) != (
            self.capabilities_snapshot.model_name,
            self.capabilities_snapshot.model_version,
        ):
            raise ValueError("deterministic adapter received an unexpected model identity")
        return self.capabilities_snapshot

    async def infer(self, request: VisionInferenceRequest) -> VisionInferenceSuccess:
        self.infer_calls += 1
        return VisionInferenceSuccess(
            status=InferenceStatus.SUCCEEDED,
            provider_request_id=f"local-{self.name}-success",
            provider=request.provider,
            model_name=request.model_name,
            model_version=request.model_version,
            normalized_output=NormalizedOutputEnvelope(
                task=request.task,
                output_schema=request.output_schema,
                package_input_set_sha256=request.package_input_set_sha256,
                input_plan_semantic_sha256=request.input_plan_semantic_sha256,
                input_plan_part_ordinal=request.input_plan_part_ordinal,
                input_plan_part_semantic_sha256=request.input_plan_part_semantic_sha256,
                payload=self.payload,
            ),
            raw_output_artifact_id=f"local-{self.name}-raw-response",
            schema_valid=True,
            reported_confidence=0.8,
            usage=VisionUsage(
                input_frames=6,
                input_images=6,
                input_tokens=60,
                output_tokens=12,
            ),
            latency_ms=1,
        )


def _capabilities(*, name: str, snapshot: int) -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(snapshot),
        snapshot_digest=_digest(snapshot),
        provider="runpod",
        model_name=name,
        model_version="local-4b-v1",
        supported_tasks=(_TASK,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=6,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=1_024,
        max_input_tokens=128,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="local-conformance-v1",
        observed_at=_NOW_TEXT,
    )


def _policy(capabilities: ModelCapabilities) -> InferencePolicy:
    return InferencePolicy(
        policy_version="local-model-comparison-policy-v1",
        task=_TASK,
        provider=capabilities.provider,
        model_name=capabilities.model_name,
        model_version=capabilities.model_version,
        adapter_version="deterministic-local-v1",
        prompt_version="local-prompt-v1",
        prompt_artifact_id="local-action-evidence-prompt",
        prompt_sha256=_digest(700),
        output_schema=_SCHEMA_REF,
        generation_config={"temperature": 0.0},
        timeout_ms=500,
        selection_policy_version="local-selection-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="local-conformance-v1",
    )


def _deployment(
    *, deployment_id: str, capabilities: ModelCapabilities, endpoint_offset: int
) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=deployment_id,
        provider=capabilities.provider,
        model_name=capabilities.model_name,
        model_version=capabilities.model_version,
        adapter_version="deterministic-local-v1",
        capability_snapshot_id=capabilities.snapshot_id,
        capability_snapshot_digest=capabilities.snapshot_digest,
        endpoint_config_digest=_digest(800 + endpoint_offset),
        max_concurrent_requests=1,
    )


def _source() -> tuple[RequestCatalog, tuple[RenderedProviderItem, ...], tuple[PackageInput, ...]]:
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    cameras = tuple(
        CatalogCamera(
            camera_id=camera_id,
            ordinal=ordinal,
            frames=(
                CatalogFrame(
                    frame_id=_uuid(100 + ordinal),
                    ordinal=0,
                    aligned_timestamp_ns=1_000_000 + ordinal,
                    source_timestamp_ns=2_000_000 + ordinal,
                    source_artifact_uri=f"object://local-dry-run/source/{ordinal}.png",
                    source_artifact_sha256=_digest(200 + ordinal),
                    source_artifact_bytes=128,
                    media_type="image/png",
                    encoding="png",
                    width=640,
                    height=480,
                ),
            ),
        )
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )
    package = CatalogPackage(
        package_id=_uuid(50),
        ordinal=0,
        semantic_content_sha256=_digest(300),
        manifest_bytes_sha256=_digest(301),
        cameras=cameras,
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(60),
        task=_TASK,
        packages=(package,),
        created_at=_NOW_TEXT,
    )
    items = tuple(
        RenderedProviderItem(
            provider_item_ordinal=ordinal,
            package_id=package.package_id,
            package_ordinal=package.ordinal,
            camera_id=camera.camera_id,
            camera_ordinal=camera.ordinal,
            frame_id=frame.frame_id,
            frame_ordinal=frame.ordinal,
            aligned_timestamp_ns=frame.aligned_timestamp_ns,
            source_timestamp_ns=frame.source_timestamp_ns,
            source_artifact_sha256=frame.source_artifact_sha256,
            artifact=RenderedArtifact(
                artifact_id=_uuid(400 + ordinal),
                uri=f"object://local-dry-run/rendered/{ordinal}.png",
                sha256=frame.source_artifact_sha256,
                byte_count=frame.source_artifact_bytes,
                media_type=frame.media_type,
                encoding=frame.encoding,
                width=frame.width,
                height=frame.height,
            ),
            transform=FrameTransform.create(
                operation=TransformOperation.NONE,
                policy_version="local-render-v1",
            ),
        )
        for ordinal, (camera, frame) in enumerate(
            (camera, frame) for camera in package.cameras for frame in camera.frames
        )
    )
    package_inputs = (
        PackageInput(
            package_id=package.package_id,
            package_semantic_content_sha256=package.semantic_content_sha256,
            package_manifest_sha256=package.manifest_bytes_sha256,
            role="primary",
            ordinal=0,
        ),
    )
    return catalog, items, package_inputs


def _plan(
    *,
    catalog: RequestCatalog,
    source_items: tuple[RenderedProviderItem, ...],
    capabilities: ModelCapabilities,
    plan_id: int,
) -> InferenceInputPlan:
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    return planner.build(
        input_plan_id=_uuid(plan_id),
        created_at=_NOW_TEXT,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            model_version=capabilities.model_version,
            adapter_version="deterministic-local-v1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=capabilities.snapshot_id,
            capability_snapshot_sha256=capabilities.snapshot_digest,
        ),
        rendered_items=source_items,
        prompt_output=PromptOutputContract(
            prompt_version="local-prompt-v1",
            prompt_sha256=_digest(700),
            rendered_message_sha256=_digest(701),
            provider_response_schema_sha256=_SCHEMA_REF.sha256,
            enriched_domain_schema_sha256=_digest(702),
            protocol_mode="json-schema",
            tool_mode="none",
        ),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=capabilities.max_images_per_request,
            max_pixels_per_image=capabilities.max_pixels_per_image,
            max_payload_bytes_per_request=capabilities.max_payload_bytes,
            max_input_tokens_per_request=capabilities.max_input_tokens,
        ),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=6,
                measured_input_tokens=60,
            ),
        ),
        idempotency_policy_version="local-idempotency-v1",
        reduction_policy="ordered-concat",
        reduction_policy_version="local-reduction-v1",
    )


def _orchestrator(
    *, adapter: _DeterministicVisionAdapter, capabilities: ModelCapabilities
) -> tuple[InferenceOrchestrator, InMemoryInferenceLedger]:
    ledger = InMemoryInferenceLedger()
    return (
        InferenceOrchestrator(
            adapters={"runpod": adapter},
            task_policies={_TASK: _policy(capabilities)},
            schema_documents={_SCHEMA_REF.artifact_id: _SCHEMA},
            ledger=ledger,
            clock=lambda: _NOW,
        ),
        ledger,
    )


async def run_local_model_comparison_dry_run() -> LocalModelComparisonDryRunReport:
    """Run the no-network P16-P18 rehearsal for Qwen3-VL-4B and Mage-VL-4B."""

    catalog, source_items, package_inputs = _source()
    qwen_capabilities = _capabilities(name="Qwen3-VL-4B", snapshot=500)
    mage_capabilities = _capabilities(name="Mage-VL-4B", snapshot=501)
    qwen_deployment = _deployment(
        deployment_id="local-qwen3-vl-4b-control",
        capabilities=qwen_capabilities,
        endpoint_offset=1,
    )
    mage_deployment = _deployment(
        deployment_id="local-mage-vl-4b-candidate",
        capabilities=mage_capabilities,
        endpoint_offset=2,
    )
    comparison_config = {"numeric_tolerance": 0.0, "default_severity": "MATERIAL"}
    contract = ExperimentContract(
        experiment_id="local-mage-vs-qwen3-vl-4b",
        contract_version="1.0",
        workload_manifest_sha256=_digest(600),
        arrival_schedule_sha256=_digest(601),
        comparison_config_sha256=semantic_sha256(comparison_config),
        input_representation=ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING,
        isolation_profile=ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE,
        control=qwen_deployment,
        candidate=mage_deployment,
    )
    production = ProductionRoute(
        route_id="local-qwen3-vl-4b-production",
        policy_version="1.0",
        deployment=qwen_deployment,
        authorization=ProductionRouteAuthorization(
            qualification_report_ref="local://not-measured/p20-m0-m4",
            qualification_report_sha256=_digest(610),
            release_decision_ref="local://not-measured/p20-m5",
            release_decision_sha256=_digest(611),
        ),
    )
    experiment = ExperimentRoute(
        route_id="local-mage-vs-qwen3-vl-4b-paired",
        policy_version="1.0",
        mode=RouteMode.PAIRED,
        sample_ratio=1.0,
        contract=contract,
    )
    router = ModelRouter(
        production=production,
        experiments={contract.experiment_id: experiment},
    )
    input_identity = _digest(602)
    production_decision = router.route_production(input_identity_sha256=input_identity)
    experiment_decision = router.route_experiment(
        experiment_id=contract.experiment_id,
        input_identity_sha256=input_identity,
    )

    production_adapter = _DeterministicVisionAdapter(
        name="production-qwen",
        capabilities_snapshot=qwen_capabilities,
        payload={"label": "grasp"},
    )
    experiment_control_adapter = _DeterministicVisionAdapter(
        name="experiment-qwen",
        capabilities_snapshot=qwen_capabilities,
        payload={"label": "grasp"},
    )
    experiment_candidate_adapter = _DeterministicVisionAdapter(
        name="experiment-mage",
        capabilities_snapshot=mage_capabilities,
        payload={"label": "reach"},
    )
    production_orchestrator, production_ledger = _orchestrator(
        adapter=production_adapter,
        capabilities=qwen_capabilities,
    )
    control_orchestrator, control_ledger = _orchestrator(
        adapter=experiment_control_adapter,
        capabilities=qwen_capabilities,
    )
    candidate_orchestrator, candidate_ledger = _orchestrator(
        adapter=experiment_candidate_adapter,
        capabilities=mage_capabilities,
    )
    qwen_plan = _plan(
        catalog=catalog,
        source_items=source_items,
        capabilities=qwen_capabilities,
        plan_id=800,
    )
    mage_plan = _plan(
        catalog=catalog,
        source_items=source_items,
        capabilities=mage_capabilities,
        plan_id=801,
    )

    production_terminal = await production_orchestrator.orchestrate(
        task=_TASK,
        package_set_id=_uuid(70),
        mcap_id=_uuid(71),
        camera_mapping_run_id=_uuid(72),
        alignment_id=_uuid(73),
        start_ns=100,
        end_ns=200,
        package_inputs=package_inputs,
        input_plan=qwen_plan,
        input_config={"fixture": "local-production"},
        sampling_config={"policy": "local-production"},
        metadata={"fixture": "local-model-comparison-dry-run"},
    )
    registry = ExperimentDeploymentRegistry(
        bindings={
            qwen_deployment.deployment_id: ExperimentDeploymentBinding(
                deployment=qwen_deployment,
                orchestrator=control_orchestrator,
            ),
            mage_deployment.deployment_id: ExperimentDeploymentBinding(
                deployment=mage_deployment,
                orchestrator=candidate_orchestrator,
            ),
        }
    )
    invocation = ExperimentInvocation(
        source_workload_manifest_sha256=contract.workload_manifest_sha256,
        input_identity_sha256=input_identity,
        task=_TASK,
        package_set_id=_uuid(70),
        mcap_id=_uuid(71),
        camera_mapping_run_id=_uuid(72),
        alignment_id=_uuid(73),
        start_ns=100,
        end_ns=200,
        package_inputs=package_inputs,
        control=ExperimentTargetInput(input_plan=qwen_plan),
        candidate=ExperimentTargetInput(input_plan=mage_plan),
        comparison_config=comparison_config,
        input_config={"fixture": "local-experiment"},
        sampling_config={"policy": "local-paired"},
        metadata={"fixture": "local-model-comparison-dry-run"},
    )
    result = await ExperimentExecutionCoordinator(registry=registry).execute(
        route=experiment,
        decision=experiment_decision,
        invocation=invocation,
    )
    if result.control_terminal is None or result.candidate_terminal is None:
        raise RuntimeError("local paired rehearsal did not retain both terminal observations")
    if len(production_ledger.list_selections()) != 1:
        raise RuntimeError("local production rehearsal did not retain one authoritative selection")
    if control_ledger.list_selections() or candidate_ledger.list_selections():
        raise RuntimeError("local experiment rehearsal produced an authoritative selection")

    adapter_call_count = sum(
        adapter.infer_calls
        for adapter in (
            production_adapter,
            experiment_control_adapter,
            experiment_candidate_adapter,
        )
    )
    observed_network_call_count = sum(
        adapter.network_calls
        for adapter in (
            production_adapter,
            experiment_control_adapter,
            experiment_candidate_adapter,
        )
    )
    if observed_network_call_count != 0:
        raise RuntimeError("local deterministic adapters must not make network calls")
    network_call_count: Literal[0] = 0
    readiness = LocalP18ComparisonReadiness(
        comparison_id=result.comparison.comparison_id,
        comparison_status=result.comparison.status,
        comparison_comparable=result.comparison.comparable,
    )
    return LocalModelComparisonDryRunReport(
        production_decision=production_decision,
        experiment_decision=experiment_decision,
        experiment_contract=contract,
        production_terminal=production_terminal,
        control_terminal=result.control_terminal,
        candidate_terminal=result.candidate_terminal,
        comparison=result.comparison,
        comparison_id=result.comparison.comparison_id,
        comparison_status=result.comparison.status,
        comparison_comparable=result.comparison.comparable,
        comparison_field_delta_count=len(result.comparison.field_deltas),
        deterministic_adapter_call_count=adapter_call_count,
        network_call_count=network_call_count,
        p18_readiness=readiness,
    )


__all__ = [
    "LOCAL_MODEL_COMPARISON_DRY_RUN_VERSION",
    "LOCAL_MODEL_COMPARISON_EXTERNAL_GATES",
    "LocalModelComparisonDryRunReport",
    "LocalP18ComparisonReadiness",
    "run_local_model_comparison_dry_run",
]
