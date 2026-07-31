from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import semantic_sha256
from robata.inference.adapter import (
    JsonSchemaRef,
    NormalizedOutputEnvelope,
    PackageInput,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.experiment_execution import (
    ExperimentComparisonStatus,
    ExperimentDeploymentBinding,
    ExperimentDeploymentRegistry,
    ExperimentExecutionCoordinator,
    ExperimentExecutionError,
    ExperimentInvocation,
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
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
    TransformParameter,
)
from robata.inference.models import (
    ConcurrencyClass,
    InferenceFailure,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    Retryability,
    VisionTask,
)
from robata.inference.orchestrator import (
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)
from robata.inference.routing import (
    ExperimentContract,
    ExperimentInputRepresentation,
    ExperimentIsolationProfile,
    ExperimentRoute,
    ModelDeployment,
    RouteMode,
)

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
NOW_TEXT = "2026-07-30T12:00:00Z"
TASK = VisionTask.ACTION_EVIDENCE
SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["label"],
    "properties": {"label": {"type": "string"}},
}
SCHEMA_REF = JsonSchemaRef(
    schema_id="action-output",
    version="1.0",
    artifact_id="schema-action-output",
    sha256=semantic_sha256(SCHEMA),
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


@dataclass
class _StartBarrier:
    names: set[str]
    all_started: asyncio.Event

    @classmethod
    def create(cls) -> _StartBarrier:
        return cls(names=set(), all_started=asyncio.Event())

    async def wait(self, name: str) -> None:
        self.names.add(name)
        if len(self.names) == 2:
            self.all_started.set()
        await asyncio.wait_for(self.all_started.wait(), timeout=1)


class _FakeAdapter:
    provider = "runpod"

    def __init__(
        self,
        *,
        name: str,
        capabilities: ModelCapabilities,
        payload: dict[str, object],
        mode: str = "success",
        barrier: _StartBarrier | None = None,
    ) -> None:
        self.name = name
        self._capabilities = capabilities
        self._payload = payload
        self._mode = mode
        self._barrier = barrier
        self.infer_calls = 0

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        assert (model_name, model_version) == (
            self._capabilities.model_name,
            self._capabilities.model_version,
        )
        return self._capabilities

    async def infer(
        self, request: VisionInferenceRequest
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        self.infer_calls += 1
        if self._barrier is not None:
            await self._barrier.wait(self.name)
        if self._mode == "cancelled":
            raise asyncio.CancelledError("fixture candidate cancellation")
        if self._mode in {"failure", "invalid"}:
            return VisionInferenceFailure(
                status=(
                    InferenceStatus.INVALID_OUTPUT
                    if self._mode == "invalid"
                    else InferenceStatus.TIMEOUT
                ),
                provider_request_id=f"{self.name}-timeout",
                provider=request.provider,
                model_name=request.model_name,
                model_version=request.model_version,
                raw_output_artifact_id=f"{self.name}-raw-timeout",
                schema_valid=False,
                usage=VisionUsage(input_frames=6, input_images=6),
                latency_ms=8,
                failure=InferenceFailure(
                    code="PROVIDER_TIMEOUT",
                    detail="offline fixture timeout",
                    retryability=Retryability.RETRYABLE,
                ),
            )
        return VisionInferenceSuccess(
            status=InferenceStatus.SUCCEEDED,
            provider_request_id=f"{self.name}-success",
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
                payload=self._payload,
            ),
            raw_output_artifact_id=f"{self.name}-raw-success",
            schema_valid=True,
            reported_confidence=0.8,
            usage=VisionUsage(
                input_frames=6,
                input_images=6,
                input_tokens=60,
                output_tokens=12,
                cost=0.01,
                currency="USD",
            ),
            latency_ms=5,
        )


def _capabilities(
    *, name: str, snapshot: int, snapshot_digest: int | None = None
) -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(snapshot),
        snapshot_digest=_digest(snapshot if snapshot_digest is None else snapshot_digest),
        provider="runpod",
        model_name=name,
        model_version="1.0",
        supported_tasks=(TASK,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=6,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=1024,
        max_input_tokens=128,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="1.0",
        observed_at=NOW_TEXT,
    )


def _policy(capabilities: ModelCapabilities) -> InferencePolicy:
    return InferencePolicy(
        policy_version="paired-policy-1",
        task=TASK,
        provider=capabilities.provider,
        model_name=capabilities.model_name,
        model_version=capabilities.model_version,
        adapter_version="adapter-1",
        prompt_version="prompt-1",
        prompt_artifact_id="prompt-action-evidence",
        prompt_sha256=_digest(700),
        output_schema=SCHEMA_REF,
        generation_config={"temperature": 0.0},
        timeout_ms=500,
        selection_policy_version="selection-1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="1.0",
    )


def _deployment(
    *, deployment_id: str, capabilities: ModelCapabilities, offset: int
) -> ModelDeployment:
    return ModelDeployment(
        deployment_id=deployment_id,
        provider=capabilities.provider,
        model_name=capabilities.model_name,
        model_version=capabilities.model_version,
        adapter_version="adapter-1",
        capability_snapshot_id=capabilities.snapshot_id,
        capability_snapshot_digest=capabilities.snapshot_digest,
        endpoint_config_digest=_digest(800 + offset),
        max_concurrent_requests=2,
    )


def _source() -> tuple[object, tuple[RenderedProviderItem, ...], tuple[PackageInput, ...]]:
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
                    source_artifact_uri=f"object://source/{ordinal}",
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
        task=TASK,
        packages=(package,),
        created_at=NOW_TEXT,
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
                uri=f"object://rendered/{ordinal}",
                sha256=frame.source_artifact_sha256,
                byte_count=frame.source_artifact_bytes,
                media_type=frame.media_type,
                encoding=frame.encoding,
                width=frame.width,
                height=frame.height,
            ),
            transform=FrameTransform.create(
                operation=TransformOperation.NONE,
                policy_version="render-1",
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
    catalog: object,
    source_items: tuple[RenderedProviderItem, ...],
    capabilities: ModelCapabilities,
    plan_id: int,
    transcoded: bool = False,
):
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    rendered_items = source_items
    if transcoded:
        rendered_items = tuple(
            item.model_copy(
                update={
                    "artifact": item.artifact.model_copy(
                        update={
                            "sha256": _digest(900 + item.provider_item_ordinal),
                            "byte_count": item.artifact.byte_count + 1,
                        }
                    ),
                    "transform": FrameTransform.create(
                        operation=TransformOperation.TRANSCODE,
                        policy_version="render-1",
                        parameters=(TransformParameter(name="quality", value=80),),
                    ),
                }
            )
            for item in source_items
        )
    return planner.build(
        input_plan_id=_uuid(plan_id),
        created_at=NOW_TEXT,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider=capabilities.provider,
            model_name=capabilities.model_name,
            model_version=capabilities.model_version,
            adapter_version="adapter-1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=capabilities.snapshot_id,
            capability_snapshot_sha256=capabilities.snapshot_digest,
        ),
        rendered_items=rendered_items,
        prompt_output=PromptOutputContract(
            prompt_version="prompt-1",
            prompt_sha256=_digest(700),
            rendered_message_sha256=_digest(701),
            provider_response_schema_sha256=SCHEMA_REF.sha256,
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
        idempotency_policy_version="idempotency-1",
        reduction_policy="ordered-concat",
        reduction_policy_version="reduce-1",
    )


def _orchestrator(
    *,
    adapter: _FakeAdapter,
    capabilities: ModelCapabilities,
) -> tuple[InferenceOrchestrator, InMemoryInferenceLedger]:
    ledger = InMemoryInferenceLedger()
    orchestrator = InferenceOrchestrator(
        adapters={"runpod": adapter},
        task_policies={TASK: _policy(capabilities)},
        schema_documents={SCHEMA_REF.artifact_id: SCHEMA},
        ledger=ledger,
        clock=lambda: NOW,
    )
    return orchestrator, ledger


@dataclass
class _Fixture:
    contract: ExperimentContract
    route: ExperimentRoute
    decision: object
    invocation: ExperimentInvocation
    coordinator: ExperimentExecutionCoordinator
    control_adapter: _FakeAdapter
    candidate_adapter: _FakeAdapter
    control_ledger: InMemoryInferenceLedger
    candidate_ledger: InMemoryInferenceLedger


def _fixture(
    *,
    candidate_mode: str = "success",
    representation: ExperimentInputRepresentation = (
        ExperimentInputRepresentation.IDENTICAL_FRAME_RENDERING
    ),
    candidate_transcoded: bool = False,
    barrier: _StartBarrier | None = None,
    shared_adapter: bool = False,
    same_logical_model_identity: bool = False,
    mode: RouteMode = RouteMode.PAIRED,
) -> _Fixture:
    catalog, source_items, package_inputs = _source()
    control_capabilities = _capabilities(name="qwen3-vl-4b", snapshot=500)
    candidate_capabilities = _capabilities(
        name=("qwen3-vl-4b" if same_logical_model_identity else "mage-vl-4b"),
        snapshot=501,
        snapshot_digest=(500 if same_logical_model_identity else None),
    )
    control_deployment = _deployment(
        deployment_id="qwen-control",
        capabilities=control_capabilities,
        offset=1,
    )
    candidate_deployment = _deployment(
        deployment_id="mage-candidate",
        capabilities=candidate_capabilities,
        offset=2,
    )
    comparison_config = {"numeric_tolerance": 0.0, "default_severity": "MATERIAL"}
    contract = ExperimentContract(
        experiment_id="qwen-mage-first-pass",
        contract_version="1.0",
        workload_manifest_sha256=_digest(600),
        arrival_schedule_sha256=_digest(601),
        comparison_config_sha256=semantic_sha256(comparison_config),
        input_representation=representation,
        isolation_profile=ExperimentIsolationProfile.INDEPENDENT_EQUAL_HARDWARE,
        control=control_deployment,
        candidate=candidate_deployment,
    )
    route = ExperimentRoute(
        route_id="qwen-mage-paired-route",
        policy_version="1.0",
        mode=mode,
        sample_ratio=1.0,
        contract=contract,
    )
    input_identity = _digest(602)
    decision = route.decide(input_identity_sha256=input_identity)
    control_adapter = _FakeAdapter(
        name="control",
        capabilities=control_capabilities,
        payload={"label": "grasp"},
        barrier=barrier,
    )
    candidate_adapter = _FakeAdapter(
        name="candidate",
        capabilities=candidate_capabilities,
        payload={"label": "reach"},
        mode=candidate_mode,
        barrier=barrier,
    )
    control_orchestrator, control_ledger = _orchestrator(
        adapter=control_adapter,
        capabilities=control_capabilities,
    )
    candidate_orchestrator, candidate_ledger = _orchestrator(
        adapter=(control_adapter if shared_adapter else candidate_adapter),
        capabilities=candidate_capabilities,
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
        source_workload_manifest_sha256=contract.workload_manifest_sha256,
        input_identity_sha256=input_identity,
        task=TASK,
        package_set_id=_uuid(70),
        mcap_id=_uuid(71),
        camera_mapping_run_id=_uuid(72),
        alignment_id=_uuid(73),
        start_ns=100,
        end_ns=200,
        package_inputs=package_inputs,
        control=ExperimentTargetInput(
            input_plan=_plan(
                catalog=catalog,
                source_items=source_items,
                capabilities=control_capabilities,
                plan_id=800,
            )
        ),
        candidate=ExperimentTargetInput(
            input_plan=_plan(
                catalog=catalog,
                source_items=source_items,
                capabilities=candidate_capabilities,
                plan_id=801,
                transcoded=candidate_transcoded,
            )
        ),
        comparison_config=comparison_config,
        input_config={"fixture": "paired"},
        sampling_config={"policy": "paired-replay"},
        metadata={"fixture": "experiment-execution"},
    )
    return _Fixture(
        contract=contract,
        route=route,
        decision=decision,
        invocation=invocation,
        coordinator=ExperimentExecutionCoordinator(registry=registry),
        control_adapter=control_adapter,
        candidate_adapter=candidate_adapter,
        control_ledger=control_ledger,
        candidate_ledger=candidate_ledger,
    )


def _run(coroutine):
    return asyncio.run(coroutine)


def test_paired_execution_is_concurrent_observation_only_and_contract_bound() -> None:
    barrier = _StartBarrier.create()
    fixture = _fixture(barrier=barrier)

    result = _run(
        fixture.coordinator.execute(
            route=fixture.route,
            decision=fixture.decision,
            invocation=fixture.invocation,
        )
    )

    assert barrier.names == {"control", "candidate"}
    assert result.comparison.status is ExperimentComparisonStatus.DIFFERENCE
    assert result.comparison.comparable is True
    assert [delta.path for delta in result.comparison.field_deltas] == ["label"]
    assert result.control_terminal is not None
    assert result.candidate_terminal is not None
    assert result.control_terminal.shadow is True
    assert result.candidate_terminal.shadow is True
    assert (
        result.control_terminal.logical_invocation_id
        != result.candidate_terminal.logical_invocation_id
    )
    assert result.control_terminal.input_config["logical_dependency_sha256"] == (
        fixture.contract.contract_digest
    )
    assert result.candidate_terminal.input_config["logical_dependency_sha256"] == (
        fixture.contract.contract_digest
    )
    assert fixture.control_ledger.list_selections() == ()
    assert fixture.candidate_ledger.list_selections() == ()
    assert result.comparison.control_inference_id == result.control_terminal.inference_id
    assert result.comparison.candidate_inference_id == result.candidate_terminal.inference_id


def test_contract_and_retry_identity_are_separate_and_replays_are_idempotent() -> None:
    fixture = _fixture()
    first = _run(
        fixture.coordinator.execute(
            route=fixture.route,
            decision=fixture.decision,
            invocation=fixture.invocation,
        )
    )
    replay = _run(
        fixture.coordinator.execute(
            route=fixture.route,
            decision=fixture.decision,
            invocation=fixture.invocation,
        )
    )
    assert first.comparison == replay.comparison
    assert fixture.control_adapter.infer_calls == 1
    assert fixture.candidate_adapter.infer_calls == 1

    retried_invocation = replace(fixture.invocation, attempt=2, retry_count=1)
    retried = _run(
        fixture.coordinator.execute(
            route=fixture.route,
            decision=fixture.decision,
            invocation=retried_invocation,
        )
    )
    assert retried.control_terminal is not None
    assert retried.candidate_terminal is not None
    assert retried.control_terminal.inference_id != first.control_terminal.inference_id
    assert retried.candidate_terminal.inference_id != first.candidate_terminal.inference_id

    updated_contract = fixture.contract.model_copy(
        update={"experiment_id": "qwen-mage-second-pass"}
    )
    updated_route = fixture.route.model_copy(
        update={"route_id": "qwen-mage-second-route", "contract": updated_contract}
    )
    updated_decision = updated_route.decide(
        input_identity_sha256=fixture.invocation.input_identity_sha256
    )
    updated = _run(
        fixture.coordinator.execute(
            route=updated_route,
            decision=updated_decision,
            invocation=fixture.invocation,
        )
    )
    assert updated.control_terminal is not None
    assert updated.candidate_terminal is not None
    assert (
        updated.control_terminal.logical_invocation_id
        != first.control_terminal.logical_invocation_id
    )
    assert (
        updated.candidate_terminal.logical_invocation_id
        != first.candidate_terminal.logical_invocation_id
    )
    assert updated.control_terminal.input_config["logical_dependency_sha256"] == (
        updated_contract.contract_digest
    )


@pytest.mark.parametrize("candidate_mode", ("failure", "cancelled", "invalid"))
def test_partial_failure_or_cancellation_retains_the_sibling_outcome(candidate_mode: str) -> None:
    fixture = _fixture(candidate_mode=candidate_mode)

    result = _run(
        fixture.coordinator.execute(
            route=fixture.route,
            decision=fixture.decision,
            invocation=fixture.invocation,
        )
    )

    assert result.comparison.status is ExperimentComparisonStatus.CANDIDATE_FAILURE
    assert result.control_terminal is not None
    assert result.control_terminal.status is InferenceStatus.SUCCEEDED
    if candidate_mode == "cancelled":
        assert result.candidate_terminal is None
    else:
        assert result.candidate_terminal is not None
    assert fixture.control_adapter.infer_calls == 1
    assert fixture.candidate_adapter.infer_calls == 1
    assert fixture.control_ledger.list_selections() == ()
    assert fixture.candidate_ledger.list_selections() == ()
    if candidate_mode == "cancelled":
        assert fixture.candidate_ledger.list_terminals()[0].status is InferenceStatus.CANCELLED
        assert result.comparison.candidate is not None
        assert result.comparison.candidate.execution_error_type == "CancelledError"
        replay = _run(
            fixture.coordinator.execute(
                route=fixture.route,
                decision=fixture.decision,
                invocation=fixture.invocation,
            )
        )
        assert replay.comparison.comparison_id != result.comparison.comparison_id
        assert replay.candidate_terminal is not None
        assert replay.candidate_terminal.status is InferenceStatus.CANCELLED
    else:
        assert result.candidate_terminal is not None
        expected_status = (
            InferenceStatus.INVALID_OUTPUT
            if candidate_mode == "invalid"
            else InferenceStatus.TIMEOUT
        )
        assert result.candidate_terminal.status is expected_status


def test_model_specific_rendering_retains_outcomes_but_suppresses_quality_delta() -> None:
    fixture = _fixture(
        representation=ExperimentInputRepresentation.MODEL_SPECIFIC_RENDERING,
        candidate_transcoded=True,
    )

    result = _run(
        fixture.coordinator.execute(
            route=fixture.route,
            decision=fixture.decision,
            invocation=fixture.invocation,
        )
    )

    assert result.control_terminal is not None
    assert result.candidate_terminal is not None
    assert result.comparison.status is ExperimentComparisonStatus.NOT_COMPARABLE
    assert result.comparison.comparable is False
    assert result.comparison.field_deltas == ()


def test_shadow_route_runs_only_the_candidate_as_an_observation() -> None:
    fixture = _fixture(mode=RouteMode.SHADOW)
    invocation = replace(fixture.invocation, control=None)

    result = _run(
        fixture.coordinator.execute(
            route=fixture.route,
            decision=fixture.decision,
            invocation=invocation,
        )
    )

    assert result.comparison.status is ExperimentComparisonStatus.SINGLE_OBSERVATION
    assert result.comparison.comparable is False
    assert result.control_terminal is None
    assert result.candidate_terminal is not None
    assert fixture.control_adapter.infer_calls == 0
    assert fixture.candidate_adapter.infer_calls == 1
    assert fixture.candidate_ledger.list_selections() == ()


def test_identical_frame_experiment_rejects_unequal_rendering_before_dispatch() -> None:
    fixture = _fixture(candidate_transcoded=True)

    with pytest.raises(ExperimentExecutionError, match="identical-frame"):
        _run(
            fixture.coordinator.execute(
                route=fixture.route,
                decision=fixture.decision,
                invocation=fixture.invocation,
            )
        )

    assert fixture.control_adapter.infer_calls == 0
    assert fixture.candidate_adapter.infer_calls == 0


@pytest.mark.parametrize(
    ("shared_adapter", "same_logical_model_identity", "message"),
    (
        (True, False, "separate adapters"),
        (False, True, "identical inference identities"),
    ),
)
def test_paired_registry_rejects_nonisolated_or_replay_prone_configuration(
    shared_adapter: bool,
    same_logical_model_identity: bool,
    message: str,
) -> None:
    fixture = _fixture(
        shared_adapter=shared_adapter,
        same_logical_model_identity=same_logical_model_identity,
    )

    with pytest.raises(ExperimentExecutionError, match=message):
        _run(
            fixture.coordinator.execute(
                route=fixture.route,
                decision=fixture.decision,
                invocation=fixture.invocation,
            )
        )

    assert fixture.control_adapter.infer_calls == 0
    assert fixture.candidate_adapter.infer_calls == 0
