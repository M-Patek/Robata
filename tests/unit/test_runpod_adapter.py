from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from robata.adapters.sqlite_inference_evidence import SQLiteInferenceEvidenceLedger
from robata.benchmark.provider_qualification import ProviderQualificationCollector
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import (
    JsonSchemaRef,
    PackageInput,
    ProviderQualificationObserver,
    ProviderQualificationRequestContract,
    ProviderQualificationSession,
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
from robata.inference.orchestrator import (
    InferenceOrchestrator,
    InferencePolicy,
    InMemoryInferenceLedger,
)
from robata.inference.runpod import (
    RUNPOD_BATCH_REQUEST_CONTRACT_VERSION,
    RUNPOD_BATCH_RESPONSE_CONTRACT_VERSION,
    RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY,
    RUNPOD_QUALIFICATION_SESSION_METADATA_KEY,
    RUNPOD_REQUEST_CONTRACT_VERSION,
    RUNPOD_RESPONSE_CONTRACT_VERSION,
    RecordedRunPodExchange,
    RecordedRunPodTransport,
    RunPodApiKey,
    RunPodDeploymentConfiguration,
    RunPodEndpointConfig,
    RunPodHttpRequest,
    RunPodHttpResponse,
    RunPodRetryPolicy,
    RunPodTransport,
    RunPodTransportError,
    RunPodVisionAdapter,
)
from robata.runtime.observability import RuntimeProfileRecorder

NOW = "2026-07-21T12:00:00Z"
API_KEY = "runpod-test-secret-000000000000"
RAW_CLAIM = '{"claims":[],"abstained":true}'
QUALIFICATION_RUN_NAMESPACE = "runpod-adapter-test"

def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _digest(value: int) -> str:
    return f"{value:064x}"


def _qualification_session(
    request: VisionInferenceRequest,
    value: int = 9_000,
    *,
    run_namespace: str = QUALIFICATION_RUN_NAMESPACE,
) -> ProviderQualificationSession:
    assert request.input_plan is not None
    return ProviderQualificationSession(
        session_id=_uuid(value),
        run_namespace=run_namespace,
        configuration_digest=_digest(value + 1),
        workload_manifest_digest=_digest(value + 2),
        request_contracts=(
            ProviderQualificationRequestContract(
                task=request.task,
                prompt_artifact_id=request.prompt_artifact_id,
                prompt_version=request.prompt_version,
                prompt_sha256=request.prompt_sha256,
                output_schema_sha256=request.output_schema.sha256,
                max_input_tokens=(
                    request.input_plan.applicable_limits.max_input_tokens_per_request
                ),
                timeout_ms=request.timeout_ms,
                generation_config_sha256=exact_bytes_sha256(
                    canonical_json_bytes(request.generation_config)
                ),
            ),
        ),
    )


def _qualified_request(
    request: VisionInferenceRequest,
    session: ProviderQualificationSession,
) -> VisionInferenceRequest:
    return request.model_copy(
        update={
            "metadata": {
                **request.metadata,
                RUNPOD_QUALIFICATION_SESSION_METADATA_KEY: session.session_id,
                RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY: session.run_namespace,
            }
        }
    )

def _provider_schema(registry: SchemaRegistry) -> JsonSchemaRef:
    ref = registry.resolve_version(PROVIDER_CLAIM_SCHEMA_ID, "1.0.0").ref
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _plan(
    schema: JsonSchemaRef,
    *,
    call_parts: tuple[CallPartSpec, ...] | None = None,
) -> InferenceInputPlan:
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
        call_parts=call_parts
        or (
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


def _request(
    schema: JsonSchemaRef,
    *,
    plan: InferenceInputPlan | None = None,
) -> VisionInferenceRequest:
    plan = plan or _plan(schema)
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


def _config(
    *,
    max_response_bytes: int = 100_000,
    native_batch_enabled: bool = False,
    native_batch_max_size: int = 1,
    max_concurrent_requests: int = 1,
    deployment_configuration: RunPodDeploymentConfiguration | None = None,
) -> RunPodEndpointConfig:
    return RunPodEndpointConfig(
        provider="runpod",
        deployment_configuration=deployment_configuration,
        endpoint_url="https://api.runpod.test/v2/test-endpoint/runsync",
        adapter_version="runpod-adapter-v1",
        native_batch_enabled=native_batch_enabled,
        native_batch_max_size=native_batch_max_size,
        max_concurrent_requests=max_concurrent_requests,
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


def _requests(schema: JsonSchemaRef, count: int) -> tuple[VisionInferenceRequest, ...]:
    base = _request(schema)
    return tuple(
        base.model_copy(
            update={
                "logical_invocation_id": _uuid(700 + ordinal),
                "request_id": _uuid(800 + ordinal),
                "idempotency_key": f"logical-runpod-request-{ordinal}",
                "package_set_id": _uuid(900 + ordinal),
                "provider_idempotency_key": f"provider-runpod-request-{ordinal}",
            }
        )
        for ordinal in range(count)
    )


def _completed_batch_response(
    requests: tuple[VisionInferenceRequest, ...],
    *,
    statuses: dict[str, str] | None = None,
    reverse_items: bool = False,
) -> RunPodHttpResponse:
    resolved_statuses = statuses or {}
    response_items: list[dict[str, object]] = []
    for request in requests:
        status = resolved_statuses.get(request.request_id, "COMPLETED")
        response: dict[str, object] = {
            "id": f"runpod-job-{request.request_id[-8:]}",
            "status": status,
        }
        if status == "COMPLETED":
            response["output"] = {
                "contract_version": RUNPOD_RESPONSE_CONTRACT_VERSION,
                "binding": _binding(request),
                "raw_output_json": RAW_CLAIM,
                "usage": {
                    "input_tokens": 123,
                    "output_tokens": 7,
                    "cost_usd": 0.125,
                },
            }
        elif status == "FAILED":
            response["error"] = "fixture permanent failure"
        response_items.append({"request_id": request.request_id, "response": response})
    if reverse_items:
        response_items.reverse()
    return RunPodHttpResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "id": "runpod-batch-job-1",
                "status": "COMPLETED",
                "output": {
                    "contract_version": RUNPOD_BATCH_RESPONSE_CONTRACT_VERSION,
                    "items": response_items,
                },
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


class _DelayedNativeBatchTransport:
    """Wait for two batch calls so the test proves bounded concurrent dispatch."""

    def __init__(
        self,
        *,
        requests_by_id: dict[str, VisionInferenceRequest],
        expected_calls: int,
    ) -> None:
        self._requests_by_id = requests_by_id
        self._expected_calls = expected_calls
        self._release = asyncio.Event()
        self.requests: list[RunPodHttpRequest] = []
        self.active_calls = 0
        self.max_active_calls = 0

    async def post(
        self,
        request: RunPodHttpRequest,
        _credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        self.requests.append(request)
        document = json.loads(request.body)
        selected = tuple(
            self._requests_by_id[item["request_id"]] for item in document["input"]["requests"]
        )
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        if len(self.requests) == self._expected_calls:
            self._release.set()
        await self._release.wait()
        self.active_calls -= 1
        return _completed_batch_response(selected, reverse_items=True)


class _RequestRoutingTransport:
    """Route either supported wire shape to a deterministic local response."""

    network_call_count = 0

    def __init__(self, requests_by_id: dict[str, VisionInferenceRequest]) -> None:
        self._requests_by_id = requests_by_id
        self.requests: list[RunPodHttpRequest] = []

    async def post(
        self,
        request: RunPodHttpRequest,
        _credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        self.requests.append(request)
        return self._response_for(request)

    def _response_for(self, request: RunPodHttpRequest) -> RunPodHttpResponse:
        document = json.loads(request.body)
        payload = document["input"]
        batch_items = payload.get("requests")
        if isinstance(batch_items, list):
            selected = tuple(self._requests_by_id[item["request_id"]] for item in batch_items)
            return _completed_batch_response(selected)
        request_id = payload["binding"]["request_id"]
        return _completed_response(self._requests_by_id[request_id])


class _HeldRequestRoutingTransport(_RequestRoutingTransport):
    """Hold in-flight local posts to assert the adapter-wide capacity bound."""

    def __init__(self, requests_by_id: dict[str, VisionInferenceRequest]) -> None:
        super().__init__(requests_by_id)
        self._entered = asyncio.Event()
        self._release = asyncio.Event()
        self.active_calls = 0
        self.max_active_calls = 0

    async def wait_until_entered(self) -> None:
        await self._entered.wait()

    def release(self) -> None:
        self._release.set()

    async def post(
        self,
        request: RunPodHttpRequest,
        _credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        self.requests.append(request)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self._entered.set()
        try:
            await self._release.wait()
        finally:
            self.active_calls -= 1
        return self._response_for(request)


class _SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _adapter(
    *,
    registry: SchemaRegistry,
    transport: RunPodTransport,
    config: RunPodEndpointConfig | None = None,
    retry_policy: RunPodRetryPolicy | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    runtime_observer: RuntimeProfileRecorder | None = None,
    qualification_observer: ProviderQualificationObserver | None = None,
    qualification_session: ProviderQualificationSession | None = None,
) -> tuple[RunPodVisionAdapter, InMemoryRawProviderBytesStore]:
    raw_store = InMemoryRawProviderBytesStore()
    resolved_sleep = asyncio.sleep if sleep is None else sleep

    return (
        RunPodVisionAdapter(
            config=config or _config(),
            credential=RunPodApiKey(API_KEY),
            capabilities=_capabilities(),
            retry_policy=retry_policy or _retry_policy(),
            raw_store=raw_store,
            parser=StrictProviderClaimParser(registry, parser_version="runpod-parser-v1"),
            transport=transport,
            sleep=resolved_sleep,
            runtime_observer=runtime_observer,
            qualification_observer=qualification_observer,
            qualification_session=qualification_session,
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


def test_infer_batch_without_native_opt_in_uses_exact_single_request_fallback() -> None:
    registry = SchemaRegistry()
    first, second = _requests(_provider_schema(registry), 2)
    transport = _ScriptedTransport(
        _completed_response(first),
        _completed_response(second),
    )
    adapter, _raw_store = _adapter(registry=registry, transport=transport)

    results = asyncio.run(adapter.infer_batch((first, second)))

    assert all(isinstance(result, VisionInferenceSuccess) for result in results)
    assert len(transport.requests) == 2
    for request, expected in zip(transport.requests, (first, second), strict=True):
        document = json.loads(request.body)
        assert document["input"]["contract_version"] == RUNPOD_REQUEST_CONTRACT_VERSION
        assert document["input"]["binding"] == _binding(expected)
        assert "requests" not in document["input"]


def test_native_batch_splits_concurrently_and_returns_request_order() -> None:
    async def scenario() -> None:
        registry = SchemaRegistry()
        requests = _requests(_provider_schema(registry), 4)
        transport = _DelayedNativeBatchTransport(
            requests_by_id={request.request_id: request for request in requests},
            expected_calls=2,
        )
        recorder = RuntimeProfileRecorder()
        adapter, raw_store = _adapter(
            registry=registry,
            transport=transport,
            config=_config(
                native_batch_enabled=True,
                native_batch_max_size=2,
                max_concurrent_requests=2,
            ),
            runtime_observer=recorder,
        )

        results = await asyncio.wait_for(adapter.infer_batch(requests), timeout=1)

        assert all(isinstance(result, VisionInferenceSuccess) for result in results)
        assert transport.max_active_calls == 2
        assert len(transport.requests) == 2
        for request in transport.requests:
            document = json.loads(request.body)
            assert document["input"]["contract_version"] == RUNPOD_BATCH_REQUEST_CONTRACT_VERSION
            assert len(document["input"]["requests"]) == 2
            assert all(
                item["request"]["contract_version"] == RUNPOD_REQUEST_CONTRACT_VERSION
                for item in document["input"]["requests"]
            )
        assert tuple(result.provider_request_id for result in results) == tuple(
            f"runpod-job-{request.request_id[-8:]}" for request in requests
        )
        assert len(raw_store.list_records()) == 4
        profile = recorder.snapshot()
        assert (
            sum(
                counter.value
                for counter in profile.counters
                if counter.name == "inference.runpod.native_batch_dispatches"
            )
            == 2
        )
        assert (
            sum(
                counter.value
                for counter in profile.counters
                if counter.name == "inference.runpod.requests"
            )
            == 4
        )

    asyncio.run(scenario())


def test_native_batch_retries_only_timed_out_item_after_partial_completion() -> None:
    registry = SchemaRegistry()
    first, second = _requests(_provider_schema(registry), 2)
    session = _qualification_session(first, run_namespace="native-partial-retry")
    first, second = tuple(
        _qualified_request(request, session) for request in (first, second)
    )
    transport = _ScriptedTransport(
        _completed_batch_response(
            (first, second),
            statuses={first.request_id: "TIMED_OUT"},
        ),
        _completed_response(first),
    )
    sleep = _SleepRecorder()
    recorder = RuntimeProfileRecorder()
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace="native-partial-retry",
        offered_concurrency=2,
    )
    adapter, raw_store = _adapter(
        registry=registry,
        transport=transport,
        config=_config(native_batch_enabled=True, native_batch_max_size=2),
        sleep=sleep,
        runtime_observer=recorder,
        qualification_observer=collector,
        qualification_session=session,
    )

    results = asyncio.run(adapter.infer_batch((first, second)))

    assert all(isinstance(result, VisionInferenceSuccess) for result in results)
    assert sleep.delays == [0.01]
    assert len(transport.requests) == 2
    first_request = json.loads(transport.requests[0].body)
    retry_request = json.loads(transport.requests[1].body)
    assert tuple(item["request_id"] for item in first_request["input"]["requests"]) == (
        first.request_id,
        second.request_id,
    )
    assert retry_request["input"]["contract_version"] == RUNPOD_REQUEST_CONTRACT_VERSION
    assert retry_request["input"]["binding"] == _binding(first)
    assert "requests" not in retry_request["input"]
    assert transport.requests[0].idempotency_key != transport.requests[1].idempotency_key
    assert len(raw_store.list_records()) == 2
    assert collector.timing_sample_count == 2
    assert {sample.request_id for sample in collector.timing_samples} == {
        first.request_id,
        second.request_id,
    }
    assert all(sample.accepted for sample in collector.timing_samples)
    assert collector.http_request_count == 2
    assert collector.adapter_transport_retry_count == 1
    assert adapter.qualification_observation_error is None
    profile = recorder.snapshot()
    assert (
        sum(
            counter.value
            for counter in profile.counters
            if counter.name == "inference.runpod.requests"
            and {attribute.name: attribute.value for attribute in counter.attributes}
            == {"mode": "native_batch", "provider": "runpod"}
        )
        == 1
    )
    assert (
        sum(
            counter.value
            for counter in profile.counters
            if counter.name == "inference.runpod.requests"
            and {attribute.name: attribute.value for attribute in counter.attributes}
            == {"mode": "single_fallback", "provider": "runpod"}
        )
        == 1
    )


def test_native_batch_keeps_a_completed_sibling_when_one_item_permanently_fails() -> None:
    registry = SchemaRegistry()
    first, second = _requests(_provider_schema(registry), 2)
    transport = _ScriptedTransport(
        _completed_batch_response(
            (first, second),
            statuses={first.request_id: "FAILED"},
        )
    )
    adapter, raw_store = _adapter(
        registry=registry,
        transport=transport,
        config=_config(native_batch_enabled=True, native_batch_max_size=2),
    )

    first_result, second_result = asyncio.run(adapter.infer_batch((first, second)))

    assert isinstance(first_result, VisionInferenceFailure)
    assert first_result.status is InferenceStatus.FAILED
    assert first_result.failure.code == "RUNPOD_JOB_FAILED"
    assert isinstance(second_result, VisionInferenceSuccess)
    assert len(transport.requests) == 1
    assert len(raw_store.list_records()) == 1


def test_adapter_lifetime_dispatch_gate_bounds_mixed_native_and_direct_inference() -> None:
    async def scenario() -> None:
        registry = SchemaRegistry()
        batch_first, batch_second, direct_request = _requests(_provider_schema(registry), 3)
        transport = _HeldRequestRoutingTransport(
            {request.request_id: request for request in (batch_first, batch_second, direct_request)}
        )
        adapter, _raw_store = _adapter(
            registry=registry,
            transport=transport,
            config=_config(
                native_batch_enabled=True,
                native_batch_max_size=2,
                max_concurrent_requests=1,
            ),
        )

        batch_task = asyncio.create_task(adapter.infer_batch((batch_first, batch_second)))
        direct_task = asyncio.create_task(adapter.infer(direct_request))
        await asyncio.wait_for(transport.wait_until_entered(), timeout=1)
        # Give every competing task a turn to reach the shared adapter gate before
        # releasing the first HTTP attempt. Without an adapter-lifetime gate the
        # direct request and the native chunk can both be in ``post`` here.
        await asyncio.sleep(0)
        transport.release()

        batch_results, direct_result = await asyncio.wait_for(
            asyncio.gather(batch_task, direct_task),
            timeout=1,
        )

        assert all(isinstance(result, VisionInferenceSuccess) for result in batch_results)
        assert isinstance(direct_result, VisionInferenceSuccess)
        assert len(transport.requests) == 2
        assert transport.max_active_calls == 1

    asyncio.run(scenario())


def test_shared_dispatch_gate_times_out_while_a_native_batch_is_saturated() -> None:
    async def scenario() -> None:
        registry = SchemaRegistry()
        first, second, blocked = _requests(_provider_schema(registry), 3)
        blocked = blocked.model_copy(update={"timeout_ms": 20})
        transport = _HeldRequestRoutingTransport(
            {request.request_id: request for request in (first, second, blocked)}
        )
        adapter, _raw_store = _adapter(
            registry=registry,
            transport=transport,
            config=_config(
                native_batch_enabled=True,
                native_batch_max_size=2,
                max_concurrent_requests=1,
            ),
            retry_policy=_retry_policy(max_attempts=1),
        )

        batch_task = asyncio.create_task(adapter.infer_batch((first, second)))
        await asyncio.wait_for(transport.wait_until_entered(), timeout=1)
        blocked_result = await asyncio.wait_for(adapter.infer(blocked), timeout=1)

        assert isinstance(blocked_result, VisionInferenceFailure)
        assert blocked_result.status is InferenceStatus.TIMEOUT
        assert blocked_result.failure.code == "RUNPOD_TRANSPORT_RETRY_EXHAUSTED"
        # The blocked direct request timed out acquiring the shared gate and was
        # never handed to the local transport.
        assert len(transport.requests) == 1
        assert transport.max_active_calls == 1

        transport.release()
        batch_results = await asyncio.wait_for(batch_task, timeout=1)
        assert all(isinstance(result, VisionInferenceSuccess) for result in batch_results)

    asyncio.run(scenario())


def test_native_batch_metrics_use_actual_per_item_dispatch_mode() -> None:
    registry = SchemaRegistry()
    first, second, singleton, rejected = _requests(_provider_schema(registry), 4)
    rejected = rejected.model_copy(update={"model_name": "unqualified-runpod-model"})
    transport = _RequestRoutingTransport(
        {request.request_id: request for request in (first, second, singleton)}
    )
    recorder = RuntimeProfileRecorder()
    adapter, _raw_store = _adapter(
        registry=registry,
        transport=transport,
        config=_config(
            native_batch_enabled=True,
            native_batch_max_size=2,
            max_concurrent_requests=1,
        ),
        runtime_observer=recorder,
    )

    results = asyncio.run(adapter.infer_batch((first, second, singleton, rejected)))

    assert all(isinstance(result, VisionInferenceSuccess) for result in results[:3])
    assert isinstance(results[3], VisionInferenceFailure)
    assert results[3].failure.code == "RUNPOD_REQUEST_REJECTED"
    profile = recorder.snapshot()

    def counter_value(name: str, **attributes: str) -> int:
        return sum(
            counter.value
            for counter in profile.counters
            if counter.name == name
            and {attribute.name: attribute.value for attribute in counter.attributes} == attributes
        )

    assert (
        counter_value(
            "inference.runpod.requests",
            mode="native_batch",
            provider="runpod",
        )
        == 2
    )
    assert (
        counter_value(
            "inference.runpod.requests",
            mode="single_fallback",
            provider="runpod",
        )
        == 1
    )
    assert (
        counter_value(
            "inference.runpod.requests",
            mode="prevalidation",
            provider="runpod",
        )
        == 1
    )
    assert (
        counter_value(
            "inference.runpod.request_outcomes",
            mode="native_batch",
            provider="runpod",
            status=InferenceStatus.SUCCEEDED.value,
        )
        == 2
    )
    assert (
        counter_value(
            "inference.runpod.request_outcomes",
            mode="single_fallback",
            provider="runpod",
            status=InferenceStatus.SUCCEEDED.value,
        )
        == 1
    )
    assert (
        counter_value(
            "inference.runpod.request_outcomes",
            mode="prevalidation",
            provider="runpod",
            status=InferenceStatus.FAILED.value,
        )
        == 1
    )


def test_recorded_response_replays_through_runpod_parser_and_raw_evidence_ledger() -> None:
    registry = SchemaRegistry()
    request = _request(_provider_schema(registry))
    probe, _probe_store = _adapter(
        registry=registry,
        transport=_ScriptedTransport(_completed_response(request)),
    )
    request_body = canonical_json_bytes(probe._request_document(request))
    transport = RecordedRunPodTransport(
        (
            RecordedRunPodExchange(
                request_body_sha256=exact_bytes_sha256(request_body),
                response=_completed_response(request),
            ),
        )
    )
    adapter, raw_store = _adapter(registry=registry, transport=transport)

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceSuccess)
    assert result.normalized_output.payload == {"claims": [], "abstained": True}
    assert raw_store.get(result.raw_output_artifact_id).data == RAW_CLAIM.encode()
    assert transport.request_count == 1
    assert transport.requests[0].body == request_body


def test_adapter_records_provider_reported_queue_execution_and_ttft() -> None:
    registry = SchemaRegistry()
    raw_request = _request(_provider_schema(registry))
    session = _qualification_session(raw_request)
    request = _qualified_request(raw_request, session)
    response_document = json.loads(_completed_response(request).body)
    response_document["output"]["usage"]["timeToFirstToken"] = 3
    recorder = RuntimeProfileRecorder()
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace="runpod-adapter-test",
        offered_concurrency=1,
    )
    adapter, _raw_store = _adapter(
        registry=registry,
        transport=_ScriptedTransport(
            RunPodHttpResponse(status_code=200, body=canonical_json_bytes(response_document))
        ),
        runtime_observer=recorder,
        qualification_observer=collector,
        qualification_session=session,
    )

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceSuccess)
    counters = {counter.name: counter.value for counter in recorder.snapshot().counters}
    assert counters["inference.runpod.provider_queue_ms"] == 2
    assert counters["inference.runpod.provider_execution_ms"] == 5
    assert counters["inference.runpod.time_to_first_token_ms"] == 3
    assert collector.timing_sample_count == 1
    sample = collector.timing_samples[0]
    assert sample.request_id == request.request_id
    assert sample.logical_invocation_id == request.logical_invocation_id
    assert sample.input_plan_part_ordinal == 0
    assert sample.provider_image_count == 6
    assert sample.input_tokens == 123
    assert sample.input_tokens_known is True
    assert sample.output_tokens == 7
    assert sample.provider_queue_ms == 2
    assert sample.provider_execution_ms == 5
    assert sample.time_to_first_token_ms == 3
    assert sample.end_to_end_ms == result.latency_ms
    assert sample.accepted is True


def test_recorded_response_sequence_replays_timeout_retry_without_duplicate_raw_evidence() -> None:
    registry = SchemaRegistry()
    raw_request = _request(_provider_schema(registry))
    session = _qualification_session(raw_request)
    request = _qualified_request(raw_request, session)
    probe, _probe_store = _adapter(
        registry=registry,
        transport=_ScriptedTransport(_completed_response(request)),
    )
    request_body = canonical_json_bytes(probe._request_document(request))
    transport = RecordedRunPodTransport(
        (
            RecordedRunPodExchange(
                request_body_sha256=exact_bytes_sha256(request_body),
                response=RunPodTransportError("recorded timeout"),
            ),
            RecordedRunPodExchange(
                request_body_sha256=exact_bytes_sha256(request_body),
                response=_completed_response(request),
            ),
        )
    )
    sleep = _SleepRecorder()
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace="runpod-adapter-test",
        offered_concurrency=1,
    )
    adapter, raw_store = _adapter(
        registry=registry,
        transport=transport,
        sleep=sleep,
        qualification_observer=collector,
        qualification_session=session,
    )

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceSuccess)
    assert transport.request_count == 2
    assert sleep.delays == [0.01]
    assert len(raw_store.list_records()) == 1
    assert collector.adapter_transport_retry_count == 1
    assert collector.timing_sample_count == 1


def test_qualification_deployment_requires_explicit_output_limit_and_session_metadata() -> None:
    registry = SchemaRegistry()
    raw_request = _request(_provider_schema(registry)).model_copy(
        update={"generation_config": {"max_output_tokens": 64, "temperature": 0.0}}
    )
    session = _qualification_session(raw_request)
    deployment = RunPodDeploymentConfiguration(
        model_identifier="runpod-qwen-vision",
        model_version="1.0",
        inference_engine="vllm",
        precision_or_quantization="bf16",
        topology="TWO_SINGLE_CARD_REPLICAS",
        max_output_tokens=64,
        supported_topologies=("TWO_SINGLE_CARD_REPLICAS",),
    )
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace="runpod-adapter-test",
        offered_concurrency=1,
    )
    transport = _ScriptedTransport()
    adapter, _raw_store = _adapter(
        registry=registry,
        transport=transport,
        config=_config(deployment_configuration=deployment),
        qualification_observer=collector,
        qualification_session=session,
    )
    request = _qualified_request(raw_request, session)

    document = adapter._request_document(request)

    assert document["input"]["generation_config"] == {
        "max_output_tokens": 64,
        "temperature": 0.0,
    }
    for generation_config in (
        {"temperature": 0.0},
        {"max_output_tokens": 63, "temperature": 0.0},
    ):
        rejected = request.model_copy(update={"generation_config": generation_config})
        result = asyncio.run(adapter.infer(rejected))
        assert isinstance(result, VisionInferenceFailure)
        assert result.failure.code == "RUNPOD_REQUEST_REJECTED"
    assert transport.requests == []
    assert collector.timing_sample_count == 0


def test_qualification_observes_one_terminal_transport_failure() -> None:
    registry = SchemaRegistry()
    raw_request = _request(_provider_schema(registry))
    session = _qualification_session(raw_request)
    request = _qualified_request(raw_request, session)
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace="runpod-adapter-test",
        offered_concurrency=1,
    )
    adapter, _raw_store = _adapter(
        registry=registry,
        transport=_ScriptedTransport(RunPodTransportError("recorded timeout")),
        retry_policy=_retry_policy(max_attempts=1),
        qualification_observer=collector,
        qualification_session=session,
    )

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceFailure)
    assert result.failure.code == "RUNPOD_TRANSPORT_RETRY_EXHAUSTED"
    assert collector.timing_sample_count == 1
    sample = collector.timing_samples[0]
    assert sample.request_id == request.request_id
    assert sample.accepted is False
    assert sample.output_tokens_known is False


def test_qualification_rejects_missing_namespace_and_unpinned_request_settings() -> None:
    registry = SchemaRegistry()
    raw_request = _request(_provider_schema(registry))
    session = _qualification_session(raw_request)
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace=QUALIFICATION_RUN_NAMESPACE,
        offered_concurrency=1,
    )
    transport = _ScriptedTransport()
    adapter, _raw_store = _adapter(
        registry=registry,
        transport=transport,
        qualification_observer=collector,
        qualification_session=session,
    )
    request = _qualified_request(raw_request, session)
    rejected_requests = (
        request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY: "wrong-namespace",
                }
            }
        ),
        request.model_copy(update={"timeout_ms": request.timeout_ms + 1}),
        request.model_copy(update={"generation_config": {"temperature": 0.1}}),
    )

    for rejected in rejected_requests:
        result = asyncio.run(adapter.infer(rejected))
        assert isinstance(result, VisionInferenceFailure)
        assert result.failure.code == "RUNPOD_REQUEST_REJECTED"

    assert transport.requests == []
    assert collector.timing_sample_count == 0


def test_runpod_rejects_an_unparted_split_input_plan_before_dispatch() -> None:
    registry = SchemaRegistry()
    schema = _provider_schema(registry)
    split_plan = _plan(
        schema,
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=3,
                measured_input_tokens=60,
            ),
            CallPartSpec(
                start_item_ordinal=3,
                end_item_ordinal_exclusive=6,
                measured_input_tokens=60,
            ),
        ),
    )
    part_request = _request(schema, plan=split_plan)
    unparted = VisionInferenceRequest.model_validate(
        {
            **part_request.model_dump(mode="python"),
            "rendered_input_digest": split_plan.rendering_sha256,
            "input_plan_part_ordinal": None,
            "input_plan_part_count": None,
            "input_plan_part_semantic_sha256": None,
        }
    )
    transport = _ScriptedTransport()
    adapter, _raw_store = _adapter(registry=registry, transport=transport)

    result = asyncio.run(adapter.infer(unparted))

    assert isinstance(result, VisionInferenceFailure)
    assert result.failure.code == "RUNPOD_REQUEST_REJECTED"
    assert "explicit call part" in result.failure.detail
    assert transport.requests == []

def test_qualification_observer_error_is_exposed_to_the_saturation_runner() -> None:
    class FailingObserver:
        def record_provider_timing(self, **_kwargs: object) -> None:
            raise RuntimeError("collector unavailable")

        def record_provider_http_requests(self, **_kwargs: object) -> None:
            return None

        def record_provider_retries(self, **_kwargs: object) -> None:
            return None

    registry = SchemaRegistry()
    raw_request = _request(_provider_schema(registry))
    session = _qualification_session(raw_request)
    request = _qualified_request(raw_request, session)
    adapter, _raw_store = _adapter(
        registry=registry,
        transport=_ScriptedTransport(_completed_response(request)),
        qualification_observer=FailingObserver(),
        qualification_session=session,
    )

    result = asyncio.run(adapter.infer(request))

    assert isinstance(result, VisionInferenceSuccess)
    assert adapter.qualification_observation_error == "terminal:RuntimeError"


def test_recorded_response_replays_through_orchestrator_and_sqlite_evidence(
    tmp_path: Path,
) -> None:
    """A request-bound recorded response traverses the production evidence path."""

    registry = SchemaRegistry()
    schema = _provider_schema(registry)
    request = _request(schema)
    assert request.input_plan is not None
    plan = request.input_plan
    policy = InferencePolicy(
        policy_version="model-policy-v1",
        task=request.task,
        provider="runpod",
        model_name="runpod-qwen-vision",
        model_version="1.0",
        adapter_version="runpod-adapter-v1",
        prompt_version=plan.prompt_output.prompt_version,
        prompt_artifact_id=request.prompt_artifact_id,
        prompt_sha256=plan.prompt_output.prompt_sha256,
        output_schema=schema,
        generation_config={"temperature": 0.0},
        timeout_ms=1_000,
        selection_policy_version="selection-v1",
        required_input_mode=InputMode.MULTI_IMAGE,
        required_media_types=("image/png",),
        required_data_handling_policy_version="runpod-local-policy-v1",
    )
    registered_claim = registry.resolve_version(PROVIDER_CLAIM_SCHEMA_ID, "1.0.0")
    registered_common = registry.resolve_version("https://schemas.robata.dev/common", "1.0.0")
    schema_artifacts = {
        registered_claim.ref.artifact_id: registered_claim.document_bytes,
        registered_common.ref.artifact_id: registered_common.document_bytes,
    }
    call_args: dict[str, object] = {
        "task": request.task,
        "package_set_id": request.package_set_id,
        "mcap_id": _uuid(950),
        "camera_mapping_run_id": _uuid(951),
        "alignment_id": _uuid(952),
        "start_ns": 1,
        "end_ns": 2,
        "package_inputs": request.package_inputs,
        "input_plan": plan,
        "input_plan_part_ordinal": 0,
    }
    def clock() -> datetime:
        return datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


    # Capture the exact request body emitted by the normal adapter, then turn the
    # captured response shape into a request-bound replay fixture.
    probe_transport = RecordedRunPodTransport(
        (
            RecordedRunPodExchange(
                request_body_sha256=_digest(9_999),
                response=RunPodTransportError("recorded probe miss"),
            ),
        )
    )
    probe_adapter, _probe_raw_store = _adapter(
        registry=registry,
        transport=probe_transport,
        retry_policy=_retry_policy(max_attempts=1),
    )
    probe_orchestrator = InferenceOrchestrator(
        adapters={"runpod": probe_adapter},
        task_policies={request.task: policy},
        schema_artifacts=schema_artifacts,
        ledger=InMemoryInferenceLedger(),
        clock=clock,
    )
    asyncio.run(probe_orchestrator.orchestrate(**call_args))
    assert probe_transport.request_count == 1
    probe_body = probe_transport.requests[0].body
    binding = json.loads(probe_body)["input"]["binding"]
    recorded_response = RunPodHttpResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "id": "recorded-runpod-job-1",
                "status": "COMPLETED",
                "output": {
                    "contract_version": RUNPOD_RESPONSE_CONTRACT_VERSION,
                    "binding": binding,
                    "raw_output_json": RAW_CLAIM,
                    "usage": {
                        "input_tokens": 123,
                        "output_tokens": 7,
                        "cost_usd": 0.125,
                    },
                },
                "delayTime": 2,
                "executionTime": 5,
                "workerId": "recorded-worker",
            }
        ),
    )
    replay_transport = RecordedRunPodTransport(
        (
            RecordedRunPodExchange(
                request_body_sha256=exact_bytes_sha256(probe_body),
                response=recorded_response,
            ),
        )
    )

    ledger = SQLiteInferenceEvidenceLedger(tmp_path / "recorded-runpod.sqlite", registry)
    try:
        replay_adapter = RunPodVisionAdapter(
            config=_config(),
            credential=RunPodApiKey(API_KEY),
            capabilities=_capabilities(),
            retry_policy=_retry_policy(max_attempts=1),
            raw_store=ledger,
            parser=StrictProviderClaimParser(registry, parser_version="runpod-parser-v1"),
            transport=replay_transport,
        )
        replay_orchestrator = InferenceOrchestrator(
            adapters={"runpod": replay_adapter},
            task_policies={request.task: policy},
            schema_artifacts=schema_artifacts,
            ledger=ledger,
            clock=clock,
        )

        terminal = asyncio.run(replay_orchestrator.orchestrate(**call_args))

        assert terminal.status is InferenceStatus.SUCCEEDED
        assert terminal.output_valid
        assert terminal.normalized_output == {"claims": [], "abstained": True}
        assert replay_transport.request_count == 1
        assert terminal.raw_output is not None
        raw_artifact_id = terminal.raw_output["artifact_id"]
        assert isinstance(raw_artifact_id, str)
        assert ledger.get(raw_artifact_id).data == RAW_CLAIM.encode()
        assert ledger.get_raw_artifact(raw_artifact_id) is not None
        assert ledger.get_terminal(terminal.inference_id) == terminal
        selection = ledger.get_selection(terminal.logical_invocation_id, "selection-v1")
        assert selection is not None
        assert selection.inference_id == terminal.inference_id

        redelivered = asyncio.run(replay_orchestrator.orchestrate(**call_args))

        assert redelivered == terminal
        assert replay_transport.request_count == 1
        assert len(ledger.list_records()) == 1
        ledger.verify_integrity()
    finally:
        ledger.close()