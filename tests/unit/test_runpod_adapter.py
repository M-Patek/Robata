from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from uuid import UUID

import pytest

from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import (
    JsonSchemaRef,
    PackageInput,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
)
from robata.inference.enrichment import PROVIDER_CLAIM_SCHEMA_ID
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
    TransformOperation,
)
from robata.inference.models import (
    ConcurrencyClass,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    Retryability,
    VisionTask,
)
from robata.inference.offline_fixture import (
    InMemoryRawProviderBytesStore,
    StrictProviderClaimParser,
)
from robata.inference.runpod import (
    RUNPOD_RESPONSE_CONTRACT_VERSION,
    RunPodApiKey,
    RunPodEndpointConfig,
    RunPodHttpRequest,
    RunPodHttpResponse,
    RunPodRetryPolicy,
    RunPodTransportError,
    RunPodVisionAdapter,
)

NOW = "2026-07-21T12:00:00Z"
API_KEY = "runpod-test-secret-000000000000"
RAW_CLAIM = '{"claims":[],"abstained":true}'


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _provider_schema(registry: SchemaRegistry) -> JsonSchemaRef:
    ref = registry.resolve_version(PROVIDER_CLAIM_SCHEMA_ID, "1.0.0").ref
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _plan(schema: JsonSchemaRef) -> InferenceInputPlan:
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    cameras = tuple(
        CatalogCamera(
            camera_id=camera_id,
            ordinal=ordinal,
            frames=(
                CatalogFrame(
                    frame_id=_uuid(100 + ordinal),
                    ordinal=0,
                    aligned_timestamp_ns=1_000_000_000 + ordinal,
                    source_timestamp_ns=1_700_000_000_000_000_000 + ordinal,
                    source_artifact_uri=f"object://source/{ordinal}",
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
    package = CatalogPackage(
        package_id=_uuid(300),
        ordinal=0,
        semantic_content_sha256=_digest(301),
        manifest_bytes_sha256=_digest(302),
        cameras=cameras,
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(303),
        task=VisionTask.ACTION_EVIDENCE,
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
                policy_version="render-v1",
            ),
        )
        for ordinal, (camera, frame) in enumerate(
            (camera, frame) for camera in package.cameras for frame in camera.frames
        )
    )
    return planner.build(
        input_plan_id=_uuid(500),
        created_at=NOW,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider="runpod",
            model_name="runpod-qwen-vision",
            model_version="1.0",
            adapter_version="runpod-adapter-v1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=_uuid(501),
            capability_snapshot_sha256=_digest(502),
        ),
        rendered_items=items,
        prompt_output=PromptOutputContract(
            prompt_version="prompt-v1",
            prompt_sha256=_digest(503),
            rendered_message_sha256=_digest(504),
            provider_response_schema_sha256=schema.sha256,
            enriched_domain_schema_sha256=_digest(505),
            protocol_mode="json-schema",
            tool_mode="none",
        ),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=6,
            max_pixels_per_image=640 * 480,
            max_payload_bytes_per_request=10_000,
            max_input_tokens_per_request=1_000,
        ),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=6,
                measured_input_tokens=120,
            ),
        ),
        idempotency_policy_version="idempotency-v1",
        reduction_policy="single-part",
        reduction_policy_version="reduction-v1",
    )


def _request(schema: JsonSchemaRef) -> VisionInferenceRequest:
    plan = _plan(schema)
    package = plan.subject.packages[0]
    part = plan.call_plan.parts[0]
    return VisionInferenceRequest(
        schema_version="1.0",
        logical_invocation_id=_uuid(600),
        request_id=_uuid(601),
        idempotency_key="logical-runpod-request",
        provider="runpod",
        model_name="runpod-qwen-vision",
        model_version="1.0",
        package_set_id=_uuid(602),
        package_inputs=(
            PackageInput(
                package_id=package.package_id,
                package_semantic_content_sha256=package.semantic_content_sha256,
                package_manifest_sha256=package.manifest_bytes_sha256,
                role="primary",
                ordinal=package.ordinal,
            ),
        ),
        package_input_set_sha256=_digest(603),
        task=VisionTask.ACTION_EVIDENCE,
        prompt_version=plan.prompt_output.prompt_version,
        prompt_artifact_id=_uuid(604),
        prompt_sha256=plan.prompt_output.prompt_sha256,
        rendered_input_digest=part.item_manifest_sha256,
        input_plan_id=plan.input_plan_id,
        input_plan_semantic_sha256=plan.semantic_sha256,
        input_plan_part_ordinal=part.ordinal,
        input_plan_part_count=part.part_count,
        input_plan_part_semantic_sha256=part.part_semantic_sha256,
        input_plan=plan,
        output_schema=schema,
        capability_snapshot_id=plan.target.capability_snapshot_id,
        capability_snapshot_digest=plan.target.capability_snapshot_sha256,
        model_policy_version="model-policy-v1",
        generation_config={"temperature": 0.0},
        provider_idempotency_key=part.idempotency_key,
        timeout_ms=1_000,
        metadata={"fixture": "runpod-local-transport"},
    )


def _capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(501),
        snapshot_digest=_digest(502),
        provider="runpod",
        model_name="runpod-qwen-vision",
        model_version="1.0",
        supported_tasks=(VisionTask.ACTION_EVIDENCE,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=6,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=10_000,
        max_input_tokens=1_000,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="runpod-local-policy-v1",
        observed_at=NOW,
    )


def _config(*, max_response_bytes: int = 100_000) -> RunPodEndpointConfig:
    return RunPodEndpointConfig(
        provider="runpod",
        endpoint_url="https://api.runpod.test/v2/test-endpoint/runsync",
        adapter_version="runpod-adapter-v1",
        request_timeout_cap_ms=2_000,
        max_response_bytes=max_response_bytes,
    )


def _retry_policy(*, max_attempts: int = 3) -> RunPodRetryPolicy:
    return RunPodRetryPolicy(
        version="runpod-retry-v1",
        max_attempts=max_attempts,
        base_delay_ms=10,
        max_delay_ms=20,
    )


def _binding(request: VisionInferenceRequest) -> dict[str, object]:
    return {
        "request_id": request.request_id,
        "logical_invocation_id": request.logical_invocation_id,
        "provider_idempotency_key": request.provider_idempotency_key,
        "provider": request.provider,
        "model_name": request.model_name,
        "model_version": request.model_version,
        "task": request.task.value,
        "package_input_set_sha256": request.package_input_set_sha256,
        "rendered_input_digest": request.rendered_input_digest,
        "prompt_sha256": request.prompt_sha256,
        "output_schema_sha256": request.output_schema.sha256,
        "capability_snapshot_digest": request.capability_snapshot_digest,
        "model_policy_version": request.model_policy_version,
        "input_plan_semantic_sha256": request.input_plan_semantic_sha256,
        "input_plan_part_semantic_sha256": request.input_plan_part_semantic_sha256,
    }


def _completed_response(
    request: VisionInferenceRequest,
    *,
    raw_output_json: str = RAW_CLAIM,
    binding: dict[str, object] | None = None,
) -> RunPodHttpResponse:
    return RunPodHttpResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "id": "runpod-job-1",
                "status": "COMPLETED",
                "output": {
                    "contract_version": RUNPOD_RESPONSE_CONTRACT_VERSION,
                    "binding": binding or _binding(request),
                    "raw_output_json": raw_output_json,
                    "usage": {
                        "input_tokens": 123,
                        "output_tokens": 7,
                        "cost_usd": 0.125,
                    },
                },
                "delayTime": 2,
                "executionTime": 5,
                "workerId": "local-worker",
            }
        ),
    )


type ScriptedStep = RunPodHttpResponse | BaseException


class _ScriptedTransport:
    """Transport double with no socket or HTTP implementation."""

    network_call_count = 0

    def __init__(self, *steps: ScriptedStep) -> None:
        self._steps = list(steps)
        self.requests: list[RunPodHttpRequest] = []

    async def post(
        self,
        request: RunPodHttpRequest,
        _credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        self.requests.append(request)
        if not self._steps:
            raise AssertionError("unexpected transport call")
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class _SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _adapter(
    *,
    registry: SchemaRegistry,
    transport: _ScriptedTransport,
    retry_policy: RunPodRetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[RunPodVisionAdapter, InMemoryRawProviderBytesStore]:
    raw_store = InMemoryRawProviderBytesStore()
    resolved_sleep = asyncio.sleep if sleep is None else sleep

    return (
        RunPodVisionAdapter(
            config=_config(),
            credential=RunPodApiKey(API_KEY),
            capabilities=_capabilities(),
            retry_policy=retry_policy or _retry_policy(),
            raw_store=raw_store,
            parser=StrictProviderClaimParser(registry, parser_version="runpod-parser-v1"),
            transport=transport,
            sleep=resolved_sleep,
        ),
        raw_store,
    )


def test_success_uses_only_injected_transport_and_preserves_exact_raw_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    transport = _ScriptedTransport(_completed_response(request))
    adapter, raw_store = _adapter(registry=registry, transport=transport)

    def _forbid_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a real network call was attempted")

    monkeypatch.setattr("robata.inference.runpod.urllib_request.urlopen", _forbid_network)
    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceSuccess)
    assert result.status is InferenceStatus.SUCCEEDED
    assert result.provider_request_id == "runpod-job-1"
    assert result.normalized_output.payload == {"claims": [], "abstained": True}
    assert result.usage.input_frames == 6
    assert result.usage.input_images == 6
    assert result.usage.input_tokens == 123
    assert result.usage.output_tokens == 7
    assert result.usage.cost == 0.125
    assert raw_store.get(result.raw_output_artifact_id).data == RAW_CLAIM.encode()
    assert len(transport.requests) == 1
    assert transport.network_call_count == 0
    sent = transport.requests[0]
    document = json.loads(sent.body)
    assert len(document["input"]["rendered_items"]) == 6
    assert document["input"]["binding"] == _binding(request)
    assert API_KEY not in sent.body.decode()
    assert API_KEY not in repr(sent)
    assert API_KEY not in repr(RunPodApiKey(API_KEY))
    assert API_KEY not in adapter.config.model_dump_json()
    assert asyncio.run(adapter.capabilities("runpod-qwen-vision", "1.0")) == _capabilities()


def test_timeout_retries_with_deterministic_delay_then_succeeds() -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    transport = _ScriptedTransport(
        RunPodTransportError("local timeout"),
        _completed_response(request),
    )
    sleep = _SleepRecorder()
    adapter, _raw_store = _adapter(
        registry=registry,
        transport=transport,
        sleep=sleep,
    )

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceSuccess)
    assert len(transport.requests) == 2
    assert transport.requests[0].body == transport.requests[1].body
    assert transport.requests[0].idempotency_key == transport.requests[1].idempotency_key
    assert sleep.delays == [0.01]


def test_timeout_exhaustion_is_bounded_and_retryable() -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    transport = _ScriptedTransport(TimeoutError(), TimeoutError(), TimeoutError())
    sleep = _SleepRecorder()
    adapter, raw_store = _adapter(
        registry=registry,
        transport=transport,
        sleep=sleep,
    )

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.TIMEOUT
    assert result.failure.code == "RUNPOD_TRANSPORT_RETRY_EXHAUSTED"
    assert result.failure.retryability is Retryability.RETRYABLE
    assert len(transport.requests) == 3
    assert sleep.delays == [0.01, 0.02]
    assert raw_store.list_records() == ()


def test_terminal_http_error_is_permanent_and_does_not_expose_response_or_secret() -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    transport = _ScriptedTransport(
        RunPodHttpResponse(
            status_code=400,
            body=canonical_json_bytes({"error": API_KEY}),
        )
    )
    adapter, raw_store = _adapter(registry=registry, transport=transport)

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.FAILED
    assert result.failure.code == "RUNPOD_HTTP_REJECTED"
    assert result.failure.retryability is Retryability.PERMANENT
    assert API_KEY not in result.model_dump_json()
    assert len(transport.requests) == 1
    assert raw_store.list_records() == ()


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (
            b'{"id":"job-1","id":"job-2","status":"FAILED"}',
            "RUNPOD_RESPONSE_DUPLICATE_JSON_KEY",
        ),
        (
            canonical_json_bytes({"id": "job-1", "status": "IN_PROGRESS", "unexpected": True}),
            "RUNPOD_RESPONSE_INVALID_CONTRACT",
        ),
    ],
)
def test_protocol_errors_fail_closed_without_raw_claim_artifact(
    body: bytes,
    expected_code: str,
) -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    transport = _ScriptedTransport(RunPodHttpResponse(status_code=200, body=body))
    adapter, raw_store = _adapter(registry=registry, transport=transport)

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code == expected_code
    assert result.failure.retryability is Retryability.PERMANENT
    assert len(transport.requests) == 1
    assert raw_store.list_records() == ()


def test_mismatched_response_binding_is_rejected_before_raw_bytes_are_admitted() -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    mismatched = _binding(request)
    mismatched["request_id"] = _uuid(999)
    transport = _ScriptedTransport(_completed_response(request, binding=mismatched))
    adapter, raw_store = _adapter(registry=registry, transport=transport)

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code == "RUNPOD_RESPONSE_BINDING_MISMATCH"
    assert raw_store.list_records() == ()


def test_invalid_provider_claim_keeps_exact_raw_bytes_and_never_reports_success() -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    duplicate_claim = '{"claims":[],"abstained":true,"abstained":false}'
    transport = _ScriptedTransport(_completed_response(request, raw_output_json=duplicate_claim))
    adapter, raw_store = _adapter(registry=registry, transport=transport)

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code == "DUPLICATE_JSON_KEY"
    assert result.raw_output_artifact_id is not None
    assert raw_store.get(result.raw_output_artifact_id).data == duplicate_claim.encode()


def test_credential_material_in_request_metadata_is_rejected_before_transport() -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry)).model_copy(
        update={"metadata": {"forbidden": API_KEY}}
    )
    transport = _ScriptedTransport()
    adapter, raw_store = _adapter(registry=registry, transport=transport)

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceFailure)
    assert result.failure.code == "RUNPOD_CREDENTIAL_IN_REQUEST"
    assert transport.requests == []
    assert raw_store.list_records() == ()
    assert API_KEY not in result.model_dump_json()
