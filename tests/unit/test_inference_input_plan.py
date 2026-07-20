from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import semantic_sha256
from robata.inference.adapter import (
    JsonSchemaRef,
    NormalizedOutputEnvelope,
    PackageInput,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.call_barrier import (
    InferenceCallBarrierConflictError,
    InferenceCallBarrierCoordinator,
    InferenceCallBarrierError,
    InferenceCallBarrierFailedError,
    InferenceCallBarrierOpenError,
    InferenceCallPartCompletion,
    InMemoryInferenceCallBarrierStorage,
)
from robata.inference.input_plan import (
    ApplicableProviderLimits,
    CallPartSpec,
    CatalogCamera,
    CatalogFrame,
    CatalogPackage,
    FrameTransform,
    InferenceInputPlan,
    InferenceInputPlanner,
    InputPlanLimitError,
    InputPlanTarget,
    LimitDecisionStatus,
    LimitMetric,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
)
from robata.inference.models import (
    ConcurrencyClass,
    InferenceFailure,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    ModelInference,
    Retryability,
    VisionTask,
)
from robata.inference.orchestrator import (
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)
from robata.queue.barrier import BarrierCoordinator, InMemoryBarrierStorage
from robata.queue.stage import StageStatus

NOW = "2026-07-19T12:00:00Z"
TASK = VisionTask.ACTION_EVIDENCE


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _package(*, row_offset: int = 0) -> CatalogPackage:
    cameras = tuple(
        CatalogCamera(
            camera_id=camera_id,
            ordinal=ordinal,
            frames=(
                CatalogFrame(
                    frame_id=_uuid(1000 + row_offset + ordinal),
                    ordinal=0,
                    aligned_timestamp_ns=1_000_000_000 + ordinal,
                    source_timestamp_ns=1_700_000_000_000_000_000 + ordinal,
                    source_artifact_uri=f"object://source/{row_offset}/{ordinal}",
                    source_artifact_sha256=_digest(200 + ordinal),
                    source_artifact_bytes=100,
                    media_type="image/png",
                    encoding="png",
                    width=640,
                    height=480,
                ),
            ),
        )
        for ordinal, camera_id in enumerate(CAMERA_IDS)
    )
    return CatalogPackage(
        package_id=_uuid(500 + row_offset),
        ordinal=0,
        semantic_content_sha256=_digest(300),
        manifest_bytes_sha256=_digest(301),
        cameras=cameras,
    )


def _target(*, row_offset: int = 0) -> InputPlanTarget:
    return InputPlanTarget(
        provider="local-fake",
        model_name="vision-model",
        model_version="1.0",
        adapter_version="adapter-1",
        planner_version="planner-1",
        capability_snapshot_id=_uuid(700 + row_offset),
        capability_snapshot_sha256=_digest(701),
    )


def _prompt() -> PromptOutputContract:
    return PromptOutputContract(
        prompt_version="prompt-1",
        prompt_sha256=_digest(800),
        rendered_message_sha256=_digest(801),
        provider_response_schema_sha256=_digest(802),
        enriched_domain_schema_sha256=_digest(803),
        protocol_mode="json-schema",
        tool_mode="none",
    )


def _limits(*, max_images: int | None = 4) -> ApplicableProviderLimits:
    return ApplicableProviderLimits(
        max_images_per_request=max_images,
        max_pixels_per_image=640 * 480,
        max_payload_bytes_per_request=500,
        max_input_tokens_per_request=100,
    )


def _fixture(
    *,
    row_offset: int = 0,
    max_images: int | None = 4,
    prompt_output: PromptOutputContract | None = None,
):
    planner = InferenceInputPlanner("planner-1")
    package = _package(row_offset=row_offset)
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(900 + row_offset),
        task=TASK,
        packages=(package,),
        created_at=NOW,
    )
    items = tuple(
        RenderedProviderItem(
            provider_item_ordinal=ordinal,
            package_id=package.package_id,
            package_ordinal=0,
            camera_id=camera.camera_id,
            camera_ordinal=camera.ordinal,
            frame_id=frame.frame_id,
            frame_ordinal=frame.ordinal,
            aligned_timestamp_ns=frame.aligned_timestamp_ns,
            source_timestamp_ns=frame.source_timestamp_ns,
            source_artifact_sha256=frame.source_artifact_sha256,
            artifact=RenderedArtifact(
                artifact_id=_uuid(2000 + row_offset + ordinal),
                uri=f"object://rendered/{row_offset}/{ordinal}",
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
    plan = planner.build(
        input_plan_id=_uuid(3000 + row_offset),
        created_at=f"2026-07-19T12:00:{row_offset:02d}Z",
        request_catalog=catalog,
        target=_target(row_offset=row_offset),
        rendered_items=items,
        prompt_output=prompt_output or _prompt(),
        applicable_limits=_limits(max_images=max_images),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=4,
                measured_input_tokens=60,
            ),
            CallPartSpec(
                start_item_ordinal=3,
                end_item_ordinal_exclusive=6,
                measured_input_tokens=70,
            ),
        ),
        idempotency_policy_version="idempotency-1",
        reduction_policy="ordered-concat",
        reduction_policy_version="reduce-1",
    )
    return planner, catalog, items, plan


def test_builds_lossless_ordered_plan_with_limits_and_barrier() -> None:
    _, catalog, items, plan = _fixture()

    assert catalog.semantic_sha256 == plan.subject.request_catalog_sha256
    assert len(plan.rendered_items) == 6
    assert [part.overlap_before_items for part in plan.call_plan.parts] == [0, 1]
    assert [part.overlap_after_items for part in plan.call_plan.parts] == [1, 0]
    assert plan.measured_limits.max_images_per_request == 4
    assert all(decision.status is LimitDecisionStatus.PASS for decision in plan.limit_decisions)
    assert plan.call_plan.barrier_logical_key.startswith("inference-input-barrier:")
    assert len({part.idempotency_key for part in plan.call_plan.parts}) == 2
    assert tuple(item.provider_item_ordinal for item in items) == tuple(range(6))


def test_semantic_identity_excludes_row_ids_locators_and_clock_fields() -> None:
    first = _fixture(row_offset=0)[3]
    second = _fixture(row_offset=50)[3]

    assert first.input_plan_id != second.input_plan_id
    assert first.request_catalog.request_catalog_id != second.request_catalog.request_catalog_id
    assert first.created_at != second.created_at
    assert first.target.capability_snapshot_id != second.target.capability_snapshot_id
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.rendering_sha256 == second.rendering_sha256
    assert first.call_plan.call_plan_sha256 == second.call_plan.call_plan_sha256
    assert first.call_plan.barrier_semantic_sha256 == second.call_plan.barrier_semantic_sha256
    assert [part.idempotency_key for part in first.call_plan.parts] == [
        part.idempotency_key for part in second.call_plan.parts
    ]


def test_dropped_or_reordered_frame_is_rejected() -> None:
    planner, catalog, items, _ = _fixture()
    with pytest.raises(ValueError, match="every catalog frame"):
        planner.build(
            input_plan_id=_uuid(4000),
            created_at=NOW,
            request_catalog=catalog,
            target=_target(),
            rendered_items=items[:-1],
            prompt_output=_prompt(),
            applicable_limits=_limits(),
            call_parts=(
                CallPartSpec(
                    start_item_ordinal=0, end_item_ordinal_exclusive=5, measured_input_tokens=1
                ),
            ),
            idempotency_policy_version="idempotency-1",
            reduction_policy="ordered-concat",
            reduction_policy_version="reduce-1",
        )

    reordered = tuple(reversed(items))
    with pytest.raises(ValueError, match=r"ordinals|catalog order"):
        planner.build(
            input_plan_id=_uuid(4001),
            created_at=NOW,
            request_catalog=catalog,
            target=_target(),
            rendered_items=reordered,
            prompt_output=_prompt(),
            applicable_limits=_limits(),
            call_parts=(
                CallPartSpec(
                    start_item_ordinal=0, end_item_ordinal_exclusive=6, measured_input_tokens=1
                ),
            ),
            idempotency_policy_version="idempotency-1",
            reduction_policy="ordered-concat",
            reduction_policy_version="reduce-1",
        )


def test_plan_digest_tampering_and_catalog_digest_tampering_fail_closed() -> None:
    plan = _fixture()[3]
    payload = plan.model_dump(mode="json")
    payload["semantic_sha256"] = _digest(9999)
    with pytest.raises(ValidationError, match="semantic_sha256"):
        InferenceInputPlan.model_validate_json(json.dumps(payload))

    payload = plan.model_dump(mode="json")
    payload["request_catalog"]["semantic_sha256"] = _digest(9998)
    with pytest.raises(ValidationError, match="semantic_sha256"):
        InferenceInputPlan.model_validate_json(json.dumps(payload))


def test_implicit_provider_split_is_not_created_and_failed_limit_is_explicit() -> None:
    planner, catalog, items, _ = _fixture()
    with pytest.raises(InputPlanLimitError) as raised:
        planner.build(
            input_plan_id=_uuid(5000),
            created_at=NOW,
            request_catalog=catalog,
            target=_target(),
            rendered_items=items,
            prompt_output=_prompt(),
            applicable_limits=_limits(max_images=3),
            call_parts=(
                CallPartSpec(
                    start_item_ordinal=0,
                    end_item_ordinal_exclusive=6,
                    measured_input_tokens=70,
                ),
            ),
            idempotency_policy_version="idempotency-1",
            reduction_policy="ordered-concat",
            reduction_policy_version="reduce-1",
        )
    decisions = {decision.metric: decision for decision in raised.value.decisions}
    assert decisions[LimitMetric.IMAGES_PER_REQUEST].status is LimitDecisionStatus.FAIL

    with pytest.raises(ValueError, match="explicit"):
        planner.build(
            input_plan_id=_uuid(5001),
            created_at=NOW,
            request_catalog=catalog,
            target=_target(),
            rendered_items=items,
            prompt_output=_prompt(),
            applicable_limits=_limits(),
            call_parts=(),
            idempotency_policy_version="idempotency-1",
            reduction_policy="ordered-concat",
            reduction_policy_version="reduce-1",
        )


def test_transform_and_camera_contracts_reject_inconsistent_inputs() -> None:
    with pytest.raises(ValidationError):
        FrameTransform(
            operation=TransformOperation.NONE,
            policy_version="render-1",
            parameters=(),
            semantic_sha256=_digest(42),
        )

    package = _package()
    with pytest.raises(ValidationError, match="six cameras"):
        CatalogPackage(
            package_id=package.package_id,
            ordinal=0,
            semantic_content_sha256=package.semantic_content_sha256,
            manifest_bytes_sha256=package.manifest_bytes_sha256,
            cameras=package.cameras[:-1],
        )


def test_unbounded_limit_still_records_explicit_pass() -> None:
    plan = _fixture(max_images=None)[3]
    image_decision = next(
        decision
        for decision in plan.limit_decisions
        if decision.metric is LimitMetric.IMAGES_PER_REQUEST
    )
    assert image_decision.applicable_limit is None
    assert image_decision.status is LimitDecisionStatus.PASS


class _InputPlanAdapter:
    provider = "local-fake"

    def __init__(self, capabilities: ModelCapabilities) -> None:
        self._capabilities = capabilities
        self.requests: list[VisionInferenceRequest] = []

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        assert (model_name, model_version) == ("vision-model", "1.0")
        return self._capabilities

    async def infer(self, request: VisionInferenceRequest) -> VisionInferenceSuccess:
        self.requests.append(request)
        return VisionInferenceSuccess(
            status=InferenceStatus.SUCCEEDED,
            provider_request_id=f"provider-request-{len(self.requests)}",
            provider=request.provider,
            model_name=request.model_name,
            model_version=request.model_version,
            normalized_output=NormalizedOutputEnvelope(
                task=request.task,
                output_schema=request.output_schema,
                package_input_set_sha256=request.package_input_set_sha256,
                input_plan_semantic_sha256=request.input_plan_semantic_sha256,
                input_plan_part_ordinal=request.input_plan_part_ordinal,
                input_plan_part_semantic_sha256=(request.input_plan_part_semantic_sha256),
                payload={
                    "label": (
                        f"grasp-{request.input_plan_part_ordinal}"
                        if request.input_plan_part_ordinal is not None
                        else "grasp"
                    )
                },
            ),
            raw_output_artifact_id=f"raw-output-{len(self.requests)}",
            schema_valid=True,
            reported_confidence=None,
            usage=VisionUsage(
                input_frames=6,
                input_images=6,
                input_tokens=130,
                output_tokens=5,
                cost=0.0,
                currency="USD",
            ),
            latency_ms=1,
        )


def _execution_fixture() -> tuple[
    InferenceInputPlan,
    _InputPlanAdapter,
    InferenceOrchestrator,
    dict[str, object],
    ModelInference,
    ModelInference,
]:
    output_schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["label"],
        "properties": {"label": {"type": "string"}},
    }
    output_schema_ref = JsonSchemaRef(
        schema_id="action-evidence",
        version="1.0",
        artifact_id="schema-action-evidence",
        sha256=semantic_sha256(output_schema),
    )
    enriched_schema: dict[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["claims", "authority"],
        "properties": {
            "claims": {"type": "array"},
            "authority": {"type": "object"},
        },
    }
    enriched_schema_ref = JsonSchemaRef(
        schema_id="orchestrator-enriched-output",
        version="1.0",
        artifact_id="schema-orchestrator-enriched-output",
        sha256=semantic_sha256(enriched_schema),
    )
    prompt = PromptOutputContract(
        prompt_version="prompt-1",
        prompt_sha256=_digest(800),
        rendered_message_sha256=_digest(801),
        provider_response_schema_sha256=output_schema_ref.sha256,
        enriched_domain_schema_sha256=enriched_schema_ref.sha256,
        protocol_mode="json-schema",
        tool_mode="none",
    )
    planner, catalog, items, first_plan = _fixture(prompt_output=prompt)
    second_plan = planner.build(
        input_plan_id=_uuid(4000),
        created_at=NOW,
        request_catalog=catalog,
        target=_target(),
        rendered_items=items,
        prompt_output=prompt,
        applicable_limits=_limits(),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=3,
                measured_input_tokens=60,
            ),
            CallPartSpec(
                start_item_ordinal=3,
                end_item_ordinal_exclusive=6,
                measured_input_tokens=70,
            ),
        ),
        idempotency_policy_version="idempotency-1",
        reduction_policy="ordered-concat",
        reduction_policy_version="reduce-1",
    )
    capabilities = ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(700),
        snapshot_digest=_digest(701),
        provider="local-fake",
        model_name="vision-model",
        model_version="1.0",
        supported_tasks=(TASK,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=4,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=500,
        max_input_tokens=100,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="1.0",
        observed_at=NOW,
    )
    policy = InferencePolicy(
        policy_version="model-policy-1",
        task=TASK,
        provider="local-fake",
        model_name="vision-model",
        model_version="1.0",
        adapter_version="adapter-1",
        prompt_version="prompt-1",
        prompt_artifact_id="prompt-action-evidence",
        prompt_sha256=_digest(800),
        output_schema=output_schema_ref,
        enriched_output_schema=enriched_schema_ref,
        generation_config={"temperature": 0},
        timeout_ms=1000,
        selection_policy_version="selection-1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="1.0",
    )
    adapter = _InputPlanAdapter(capabilities)
    orchestrator = InferenceOrchestrator(
        adapters={"local-fake": adapter},
        task_policies={TASK: policy},
        schema_documents={
            output_schema_ref.artifact_id: output_schema,
            enriched_schema_ref.artifact_id: enriched_schema,
        },
        ledger=InMemoryInferenceLedger(),
        clock=lambda: datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
    )
    package = first_plan.subject.packages[0]
    request_args = {
        "task": TASK,
        "package_set_id": _uuid(6000),
        "mcap_id": _uuid(6001),
        "camera_mapping_run_id": _uuid(6002),
        "alignment_id": _uuid(6003),
        "start_ns": 1,
        "end_ns": 2,
        "package_inputs": (
            PackageInput(
                package_id=package.package_id,
                package_semantic_content_sha256=package.semantic_content_sha256,
                package_manifest_sha256=package.manifest_bytes_sha256,
                role="primary",
                ordinal=package.ordinal,
            ),
        ),
    }

    first = asyncio.run(
        orchestrator.orchestrate(
            **request_args,
            input_plan=first_plan,
            input_plan_part_ordinal=0,
        )
    )
    retry = asyncio.run(
        orchestrator.orchestrate(
            **request_args,
            input_plan=first_plan,
            input_plan_part_ordinal=0,
            attempt=2,
            retry_count=1,
        )
    )
    changed = asyncio.run(
        orchestrator.orchestrate(
            **request_args,
            input_plan=second_plan,
            input_plan_part_ordinal=0,
        )
    )
    second_part = asyncio.run(
        orchestrator.orchestrate(
            **request_args,
            input_plan=first_plan,
            input_plan_part_ordinal=1,
        )
    )

    assert first.logical_invocation_id == retry.logical_invocation_id
    assert first.inference_id != retry.inference_id
    assert changed.logical_invocation_id != first.logical_invocation_id
    assert first.input_plan_id == first_plan.input_plan_id
    assert first.input_plan_semantic_sha256 == first_plan.semantic_sha256
    assert first.input_plan_part_ordinal == 0
    assert first.input_plan_part_semantic_sha256 == (
        first_plan.call_plan.parts[0].part_semantic_sha256
    )
    assert second_part.logical_invocation_id != first.logical_invocation_id
    assert adapter.requests[0].input_plan == first_plan
    assert adapter.requests[0].output_schema == output_schema_ref
    assert adapter.requests[0].rendered_input_digest == (
        first_plan.call_plan.parts[0].item_manifest_sha256
    )
    assert adapter.requests[0].provider_idempotency_key == (
        first_plan.call_plan.parts[0].idempotency_key
    )
    assert adapter.requests[-1].rendered_input_digest == (
        first_plan.call_plan.parts[1].item_manifest_sha256
    )
    assert adapter.requests[0].metadata["input_plan_call_plan_sha256"] == (
        first_plan.call_plan.call_plan_sha256
    )
    return first_plan, adapter, orchestrator, request_args, first, second_part


def test_input_plan_is_bound_into_orchestrator_identity_and_request() -> None:
    _execution_fixture()


class _OrderedLabelReducer:
    def __init__(self) -> None:
        self.calls = 0

    def reduce(
        self,
        *,
        input_plan: InferenceInputPlan,
        ordered_completions: tuple[InferenceCallPartCompletion, ...],
    ) -> dict[str, object]:
        assert len(ordered_completions) == len(input_plan.call_plan.parts)
        self.calls += 1
        labels = [
            str(completion.normalized_output["label"])
            for completion in ordered_completions
            if completion.normalized_output is not None
        ]
        return {"label": "|".join(labels)}


def _call_barrier(
    reducer: _OrderedLabelReducer | None = None,
) -> InferenceCallBarrierCoordinator:
    reducers = {("ordered-concat", "reduce-1"): reducer} if reducer is not None else {}
    return InferenceCallBarrierCoordinator(
        barriers=BarrierCoordinator(InMemoryBarrierStorage()),
        storage=InMemoryInferenceCallBarrierStorage(),
        reducers=reducers,
    )


def test_call_parts_join_exact_barrier_and_reduce_once_in_plan_order() -> None:
    plan, adapter, orchestrator, _, part_zero, part_one = _execution_fixture()
    reducer = _OrderedLabelReducer()
    coordinator = _call_barrier(reducer)
    selection_zero = orchestrator.selected_attempt(
        logical_invocation_id=part_zero.logical_invocation_id,
        policy_version="selection-1",
    )
    selection_one = orchestrator.selected_attempt(
        logical_invocation_id=part_one.logical_invocation_id,
        policy_version="selection-1",
    )
    assert selection_zero is not None
    assert selection_one is not None

    definition = coordinator.declare(plan, created_at=NOW)
    assert coordinator.declare(plan, created_at="2026-07-19T12:00:01Z") == definition
    assert definition.expected_part_idempotency_keys == tuple(
        part.idempotency_key for part in plan.call_plan.parts
    )

    with pytest.raises(InferenceCallBarrierError, match="attempt selection"):
        coordinator.submit_part_terminal(plan, part_one)
    completion_one = coordinator.submit_part_terminal(
        plan,
        part_one,
        selection=selection_one,
    )
    assert completion_one.part_ordinal == 1
    assert not coordinator.get_aggregate_status(plan).is_complete
    with pytest.raises(InferenceCallBarrierOpenError, match="every declared"):
        coordinator.reduce(plan, reduced_at=NOW)

    completion_zero = coordinator.submit_part_terminal(
        plan,
        part_zero,
        selection=selection_zero,
    )
    assert (
        coordinator.submit_part_terminal(
            plan,
            part_zero,
            selection=selection_zero,
        )
        == completion_zero
    )
    aggregate = coordinator.get_aggregate_status(plan)
    assert aggregate.is_complete
    assert aggregate.overall_status is StageStatus.SUCCEEDED

    reduction = coordinator.reduce(plan, reduced_at=NOW)
    replay = coordinator.reduce(plan, reduced_at="2026-07-19T12:00:02Z")
    assert replay == reduction
    assert reducer.calls == 1
    assert reduction.ordered_completion_ids == (
        completion_zero.completion_id,
        completion_one.completion_id,
    )
    assert reduction.normalized_output == {"label": "grasp-0|grasp-1"}
    assert adapter.requests[0].provider_idempotency_key == (
        adapter.requests[1].provider_idempotency_key
    )


def test_call_barrier_rejects_unconfirmed_or_conflicting_failure_and_reduction() -> None:
    plan, _, orchestrator, _, part_zero, part_one = _execution_fixture()
    coordinator = _call_barrier(_OrderedLabelReducer())
    coordinator.declare(plan, created_at=NOW)
    selection_zero = orchestrator.selected_attempt(
        logical_invocation_id=part_zero.logical_invocation_id,
        policy_version="selection-1",
    )
    selection_one = orchestrator.selected_attempt(
        logical_invocation_id=part_one.logical_invocation_id,
        policy_version="selection-1",
    )
    assert selection_zero is not None
    assert selection_one is not None
    coordinator.submit_part_terminal(
        plan,
        part_zero,
        selection=selection_zero,
    )

    failed = part_one.model_copy(
        update={
            "status": InferenceStatus.FAILED,
            "normalized_output": None,
            "output_valid": False,
            "failure": InferenceFailure(
                code="FINAL_PROVIDER_FAILURE",
                detail="retry policy exhausted",
                retryability=Retryability.PERMANENT,
            ),
        }
    )
    with pytest.raises(InferenceCallBarrierError, match="final-failure"):
        coordinator.submit_part_terminal(plan, failed)
    failed_completion = coordinator.submit_part_terminal(
        plan,
        failed,
        failure_is_final=True,
    )
    assert (
        coordinator.submit_part_terminal(
            plan,
            failed,
            failure_is_final=True,
        )
        == failed_completion
    )
    aggregate = coordinator.get_aggregate_status(plan)
    assert aggregate.is_complete
    assert aggregate.overall_status is StageStatus.INCOMPLETE
    with pytest.raises(InferenceCallBarrierFailedError, match="failed required"):
        coordinator.reduce(plan, reduced_at=NOW)
    with pytest.raises(InferenceCallBarrierConflictError, match="different final"):
        coordinator.submit_part_terminal(
            plan,
            part_one,
            selection=selection_one,
        )


def test_call_barrier_rejects_terminal_with_wrong_part_idempotency_binding() -> None:
    plan, _, _, _, part_zero, _ = _execution_fixture()
    coordinator = _call_barrier(_OrderedLabelReducer())
    coordinator.declare(plan, created_at=NOW)
    wrong = part_zero.model_copy(update={"provider_idempotency_key": "wrong-key"})

    with pytest.raises(InferenceCallBarrierError, match="declared call part"):
        coordinator.submit_part_terminal(plan, wrong)
