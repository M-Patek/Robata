from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from robata.benchmark.provider_qualification import (
    ProviderAdapterTerminalWorkload,
    ProviderGpuMeasurement,
    ProviderLatencyPercentiles,
    ProviderQualificationCollector,
    ProviderQualificationRunContext,
    ProviderRuntimeTelemetry,
    ProviderSaturationPoint,
    ProviderTimingSample,
    TwoH100ProviderConfiguration,
    TwoH100ProviderQualificationReport,
    TwoH100Topology,
    compare_two_h100_topologies,
    run_provider_saturation_point,
)
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import (
    JsonSchemaRef,
    PackageInput,
    ProviderQualificationRequestContract,
    ProviderQualificationSession,
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
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
)
from robata.inference.models import (
    ConcurrencyClass,
    InputMode,
    ModelCapabilities,
    VisionTask,
)
from robata.inference.offline_fixture import (
    InMemoryRawProviderBytesStore,
    StrictProviderClaimParser,
)
from robata.inference.runpod import (
    RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY,
    RUNPOD_QUALIFICATION_SESSION_METADATA_KEY,
    RUNPOD_RESPONSE_CONTRACT_VERSION,
    RecordedRunPodExchange,
    RecordedRunPodTransport,
    RunPodApiKey,
    RunPodDeploymentConfiguration,
    RunPodEndpointConfig,
    RunPodHttpRequest,
    RunPodHttpResponse,
    RunPodRetryPolicy,
    RunPodTransportError,
    RunPodVisionAdapter,
)
from robata.runtime.capacity import (
    CapacityEvidenceClass,
    MeasuredCapacityInput,
    ProviderMode,
    build_measured_capacity_report,
)

_HOUR_NS = 3_600_000_000_000
_WORKLOAD_MANIFEST_BYTES = b"phase-10-p6-adaptive-workload-v1"
_DIGEST = exact_bytes_sha256(_WORKLOAD_MANIFEST_BYTES)
_GENERATION_CONFIG = {"max_output_tokens": 512, "temperature": 0.0}
_GENERATION_CONFIG_SHA256 = exact_bytes_sha256(canonical_json_bytes(_GENERATION_CONFIG))
_SUPPORTED_TOPOLOGIES = (
    TwoH100Topology.TWO_SINGLE_CARD_REPLICAS,
    TwoH100Topology.TWO_CARD_TENSOR_PARALLEL,
)


def _uuid(value: int) -> str:
    return f"00000000-0000-0000-0000-{value:012d}"


def _capacity_input(
    *,
    workload: str = _DIGEST,
    execution_mode: str = "FRESH",
    retries: int | None = None,
    provider_images: int = 120,
    logical_calls: int = 12,
    http_requests: int = 12,
    input_tokens: int = 1_200,
    output_tokens: int = 120,
    output_token_responses: int = 12,
) -> MeasuredCapacityInput:
    return MeasuredCapacityInput(
        workload_fingerprint=workload,
        evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
        provider_mode=ProviderMode.NETWORK_PROVIDER,
        execution_mode=execution_mode,
        recording_count=1,
        recording_worker_count=1,
        camera_count=6,
        recording_duration_ns=_HOUR_NS,
        wall_time_ns=_HOUR_NS,
        provider_images=provider_images,
        logical_calls=logical_calls,
        http_requests=http_requests,
        retries=retries,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_token_responses=output_token_responses,
    )


def _capacity(
    *,
    workload: str = _DIGEST,
    execution_mode: str = "FRESH",
    retries: int | None = None,
    provider_images: int = 120,
    logical_calls: int = 12,
    http_requests: int = 12,
    input_tokens: int = 1_200,
    output_tokens: int = 120,
    output_token_responses: int = 12,
):
    return build_measured_capacity_report(
        _capacity_input(
            workload=workload,
            execution_mode=execution_mode,
            retries=retries,
            provider_images=provider_images,
            logical_calls=logical_calls,
            http_requests=http_requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_token_responses=output_token_responses,
        )
    )


def _request_contract() -> ProviderQualificationRequestContract:
    return ProviderQualificationRequestContract(
        task=VisionTask.ACTION_EVIDENCE,
        prompt_artifact_id=_uuid(900),
        prompt_version="prompt-v1",
        prompt_sha256="c" * 64,
        output_schema_sha256="f" * 64,
        max_input_tokens=8_192,
        timeout_ms=1_000,
        generation_config_sha256=_GENERATION_CONFIG_SHA256,
    )


def _configuration(
    topology: TwoH100Topology = TwoH100Topology.TWO_SINGLE_CARD_REPLICAS,
    *,
    supported_topologies: tuple[TwoH100Topology, ...] = _SUPPORTED_TOPOLOGIES,
    request_contracts: tuple[ProviderQualificationRequestContract, ...] | None = None,
    native_batch_enabled: bool = True,
) -> TwoH100ProviderConfiguration:
    return TwoH100ProviderConfiguration.create(
        workload_manifest_digest=_DIGEST,
        provider="runpod",
        model_identifier="chosen-vlm",
        model_version="1.0",
        request_contracts=request_contracts or (_request_contract(),),
        inference_engine="vllm",
        precision_or_quantization="bf16",
        topology=topology,
        max_images_per_request=12,
        max_input_tokens=8_192,
        max_output_tokens=512,
        native_batch_enabled=native_batch_enabled,
        native_batch_max_size=8,
        max_concurrent_requests=16,
        endpoint_configuration=_make_endpoint_configuration(
            topology=topology,
            supported_topologies=supported_topologies,
        ),
        retry_policy=_retry_policy(),
        supported_topologies=supported_topologies,
    )


def _session(
    configuration: TwoH100ProviderConfiguration,
    value: int,
) -> ProviderQualificationSession:
    return ProviderQualificationSession(
        session_id=_uuid(value),
        run_namespace=f"p6-test-{value}",
        configuration_digest=configuration.configuration_digest,
        workload_manifest_digest=configuration.workload_manifest_digest,
        request_contracts=configuration.request_contracts,
    )


def _make_endpoint_configuration(
    *,
    topology: TwoH100Topology,
    supported_topologies: tuple[TwoH100Topology, ...],
) -> RunPodEndpointConfig:
    return RunPodEndpointConfig(
        provider="runpod",
        endpoint_url="https://api.runpod.test/v2/qualified/runsync",
        adapter_version="runpod-adapter-v1",
        deployment_configuration=RunPodDeploymentConfiguration(
            model_identifier="chosen-vlm",
            model_version="1.0",
            inference_engine="vllm",
            precision_or_quantization="bf16",
            topology=topology.value,
            max_output_tokens=512,
            supported_topologies=tuple(
                supported_topology.value for supported_topology in supported_topologies
            ),
        ),
        native_batch_enabled=True,
        native_batch_max_size=8,
        max_concurrent_requests=16,
    )


def _retry_policy() -> RunPodRetryPolicy:
    return RunPodRetryPolicy(
        version="runpod-retry-v1",
        max_attempts=2,
        base_delay_ms=0,
        max_delay_ms=0,
    )


def _endpoint_config(
    configuration: TwoH100ProviderConfiguration,
) -> RunPodEndpointConfig:
    return configuration.endpoint_configuration


def _gpu(session: ProviderQualificationSession) -> ProviderGpuMeasurement:
    return ProviderGpuMeasurement(
        qualification_session_id=session.session_id,
        hardware_inventory_artifact_uri="object://qualification/h100-inventory.json",
        hardware_inventory_sha256="d" * 64,
        telemetry_artifact_uri=(f"object://qualification/{session.session_id}/gpu-metrics.json"),
        telemetry_artifact_sha256="e" * 64,
        gpu_sku="NVIDIA H100 SXM5 80GB",
        driver_version="555.42.06",
        runtime_version="CUDA 12.4 / vLLM 0.8.5",
        metric_source="nvidia-smi dmon + vLLM metrics",
        measurement_started_at="2026-07-25T00:00:00Z",
        measurement_completed_at="2026-07-25T01:00:00Z",
        aggregate_gpu_seconds=120.0,
        gpu_utilization_fraction=0.7,
        gpu_memory_bytes=40_000_000_000,
        kv_cache_utilization_fraction=0.5,
        oom_count=0,
    )


def _sample(
    value: int,
    *,
    accepted: bool = True,
    input_tokens_known: bool = True,
    input_tokens: int | None = None,
    output_tokens_known: bool = True,
    output_tokens: int | None = None,
    provider_queue_ms: int | None = None,
    provider_execution_ms: int | None = None,
    time_to_first_token_ms: int | None = None,
    end_to_end_ms: int | None = None,
    logical_invocation_value: int | None = None,
    input_plan_part_ordinal: int = 0,
) -> ProviderTimingSample:
    if input_tokens_known and input_tokens is None:
        input_tokens = 100
    if output_tokens_known and output_tokens is None:
        output_tokens = 10
    if accepted:
        provider_queue_ms = value if provider_queue_ms is None else provider_queue_ms
        provider_execution_ms = (
            value + 10 if provider_execution_ms is None else provider_execution_ms
        )
        time_to_first_token_ms = (
            value + 20 if time_to_first_token_ms is None else time_to_first_token_ms
        )
    return ProviderTimingSample(
        request_id=_uuid(1_000 + value),
        logical_invocation_id=_uuid(
            logical_invocation_value if logical_invocation_value is not None else 10_000 + value
        ),
        input_plan_part_ordinal=input_plan_part_ordinal,
        provider_image_count=10,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_queue_ms=provider_queue_ms,
        provider_execution_ms=provider_execution_ms,
        time_to_first_token_ms=time_to_first_token_ms,
        end_to_end_ms=value + 30 if end_to_end_ms is None else end_to_end_ms,
        input_tokens_known=input_tokens_known,
        output_tokens_known=output_tokens_known,
        accepted=accepted,
    )


def _telemetry(
    session: ProviderQualificationSession,
    *,
    rejected_response_count: int = 0,
) -> ProviderRuntimeTelemetry:
    samples = tuple(_sample(value) for value in range(12)) + tuple(
        _sample(
            100 + value,
            accepted=False,
            output_tokens_known=False,
        )
        for value in range(rejected_response_count)
    )
    return ProviderRuntimeTelemetry.from_samples(
        samples=samples,
        http_requests=len(samples),
        gpu=_gpu(session),
        adapter_transport_retry_count=1,
    )


def _point(
    configuration: TwoH100ProviderConfiguration,
    *,
    session_value: int,
    offered_concurrency: int,
    rejected_response_count: int = 0,
    capacity=None,
) -> ProviderSaturationPoint:
    session = _session(configuration, session_value)
    return ProviderSaturationPoint(
        configuration_digest=configuration.configuration_digest,
        qualification_session=session,
        run_namespace=session.run_namespace,
        offered_concurrency=offered_concurrency,
        capacity=(
            _capacity(
                provider_images=120 + 10 * rejected_response_count,
                logical_calls=12 + rejected_response_count,
                http_requests=12 + rejected_response_count,
                input_tokens=1_200 + 100 * rejected_response_count,
            )
            if capacity is None
            else capacity
        ),
        telemetry=_telemetry(session, rejected_response_count=rejected_response_count),
    )


def _runpod_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(1),
        snapshot_digest="b" * 64,
        provider="runpod",
        model_name="chosen-vlm",
        model_version="1.0",
        supported_tasks=(VisionTask.ACTION_EVIDENCE,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=12,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=1_000_000,
        max_input_tokens=8_192,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.LIMITED,
        data_handling_policy_version="real-provider-policy-v1",
        observed_at="2026-07-25T00:00:00Z",
    )


def _report(
    configuration: TwoH100ProviderConfiguration,
    *,
    session_offset: int = 10,
) -> TwoH100ProviderQualificationReport:
    return TwoH100ProviderQualificationReport(
        configuration=configuration,
        endpoint_config=_endpoint_config(configuration),
        capabilities=_runpod_capabilities(),
        retry_policy=configuration.retry_policy,
        evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
        points=(
            _point(
                configuration,
                session_value=session_offset,
                offered_concurrency=4,
            ),
            _point(
                configuration,
                session_value=session_offset + 1,
                offered_concurrency=8,
                rejected_response_count=1,
            ),
        ),
    )


def _record_sample(
    collector: ProviderQualificationCollector,
    session: ProviderQualificationSession,
    sample: ProviderTimingSample,
) -> None:
    collector.record_provider_timing(
        qualification_session=session,
        request_id=sample.request_id,
        logical_invocation_id=sample.logical_invocation_id,
        input_plan_part_ordinal=sample.input_plan_part_ordinal,
        provider_image_count=sample.provider_image_count,
        input_tokens=sample.input_tokens,
        output_tokens=sample.output_tokens,
        provider_queue_ms=sample.provider_queue_ms,
        provider_execution_ms=sample.provider_execution_ms,
        time_to_first_token_ms=sample.time_to_first_token_ms,
        end_to_end_ms=sample.end_to_end_ms,
        input_tokens_known=sample.input_tokens_known,
        output_tokens_known=sample.output_tokens_known,
        accepted=sample.accepted,
    )


def _record_accepted_samples(
    collector: ProviderQualificationCollector,
    session: ProviderQualificationSession,
    *,
    start: int = 0,
) -> tuple[ProviderTimingSample, ...]:
    samples = tuple(_sample(start + value) for value in range(12))
    for sample in samples:
        _record_sample(collector, session, sample)
    return samples


class _RetryThenSuccessTransport:
    """Inject one retryable local transport fault before a pinned response."""

    network_call_count = 0

    def __init__(self, response: RunPodHttpResponse) -> None:
        self._response = response
        self.requests: list[RunPodHttpRequest] = []

    async def post(
        self,
        request: RunPodHttpRequest,
        _credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise RunPodTransportError("fixture transient transport failure")
        return self._response


def _provider_schema(registry: SchemaRegistry) -> JsonSchemaRef:
    ref = registry.resolve_version(PROVIDER_CLAIM_SCHEMA_ID, "1.0.0").ref
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _qualification_request(schema: JsonSchemaRef) -> VisionInferenceRequest:
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    cameras = tuple(
        CatalogCamera(
            camera_id=camera_id,
            ordinal=ordinal,
            frames=(
                CatalogFrame(
                    frame_id=_uuid(2_000 + ordinal),
                    ordinal=0,
                    aligned_timestamp_ns=1_000_000_000 + ordinal,
                    source_timestamp_ns=1_700_000_000_000_000_000 + ordinal,
                    source_artifact_uri=f"object://qualification/source/{ordinal}",
                    source_artifact_sha256=f"{2_100 + ordinal:064x}",
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
        package_id=_uuid(2_200),
        ordinal=0,
        semantic_content_sha256=f"{2_201:064x}",
        manifest_bytes_sha256=f"{2_202:064x}",
        cameras=cameras,
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(2_203),
        task=VisionTask.ACTION_EVIDENCE,
        packages=(package,),
        created_at="2026-07-25T00:00:00Z",
    )
    rendered_items = tuple(
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
                artifact_id=_uuid(2_300 + ordinal),
                uri=f"object://qualification/rendered/{ordinal}",
                sha256=frame.source_artifact_sha256,
                byte_count=frame.source_artifact_bytes,
                media_type=frame.media_type,
                encoding=frame.encoding,
                width=frame.width,
                height=frame.height,
            ),
            transform=FrameTransform.create(
                operation=TransformOperation.NONE,
                policy_version="qualification-render-v1",
            ),
        )
        for ordinal, (camera, frame) in enumerate(
            (camera, frame) for camera in package.cameras for frame in camera.frames
        )
    )
    plan = planner.build(
        input_plan_id=_uuid(2_400),
        created_at="2026-07-25T00:00:00Z",
        request_catalog=catalog,
        target=InputPlanTarget(
            provider="runpod",
            model_name="chosen-vlm",
            model_version="1.0",
            adapter_version="runpod-adapter-v1",
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=_uuid(1),
            capability_snapshot_sha256="b" * 64,
        ),
        rendered_items=rendered_items,
        prompt_output=PromptOutputContract(
            prompt_version="prompt-v1",
            prompt_sha256="c" * 64,
            rendered_message_sha256="d" * 64,
            provider_response_schema_sha256=schema.sha256,
            enriched_domain_schema_sha256="e" * 64,
            protocol_mode="json-schema",
            tool_mode="none",
        ),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=12,
            max_pixels_per_image=640 * 480,
            max_payload_bytes_per_request=1_000_000,
            max_input_tokens_per_request=8_192,
        ),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=len(rendered_items),
                measured_input_tokens=120,
            ),
        ),
        idempotency_policy_version="qualification-idempotency-v1",
        reduction_policy="single-part",
        reduction_policy_version="qualification-reduction-v1",
    )
    package_input = plan.subject.packages[0]
    part = plan.call_plan.parts[0]
    return VisionInferenceRequest(
        schema_version="1.0",
        logical_invocation_id=_uuid(2_500),
        request_id=_uuid(2_501),
        idempotency_key="qualification-logical-request",
        provider="runpod",
        model_name="chosen-vlm",
        model_version="1.0",
        package_set_id=_uuid(2_502),
        package_inputs=(
            PackageInput(
                package_id=package_input.package_id,
                package_semantic_content_sha256=package_input.semantic_content_sha256,
                package_manifest_sha256=package_input.manifest_bytes_sha256,
                role="primary",
                ordinal=package_input.ordinal,
            ),
        ),
        package_input_set_sha256=f"{2_503:064x}",
        task=VisionTask.ACTION_EVIDENCE,
        prompt_version=plan.prompt_output.prompt_version,
        prompt_artifact_id=_uuid(2_504),
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
        model_policy_version="qualification-model-policy-v1",
        generation_config=dict(_GENERATION_CONFIG),
        provider_idempotency_key=part.idempotency_key,
        timeout_ms=1_000,
        metadata={"fixture": "qualification-local-transport"},
    )


def _response_binding(request: VisionInferenceRequest) -> dict[str, object]:
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


def _completed_qualification_response(request: VisionInferenceRequest) -> RunPodHttpResponse:
    return RunPodHttpResponse(
        status_code=200,
        body=canonical_json_bytes(
            {
                "id": "qualification-runpod-job-1",
                "status": "COMPLETED",
                "output": {
                    "contract_version": RUNPOD_RESPONSE_CONTRACT_VERSION,
                    "binding": _response_binding(request),
                    "raw_output_json": '{"claims":[],"abstained":true}',
                    "usage": {
                        "input_tokens": 123,
                        "output_tokens": 7,
                        "cost_usd": 0.125,
                        "timeToFirstToken": 1,
                    },
                },
                "delayTime": 2,
                "executionTime": 5,
                "workerId": "qualification-local-worker",
            }
        ),
    )


def test_recorded_runpod_transport_binds_response_to_exact_request_bytes() -> None:
    request = RunPodHttpRequest(
        url="https://api.runpod.test/runsync",
        body=b'{"input":"recorded"}',
        timeout_seconds=1.0,
        max_response_bytes=100,
        idempotency_key="recorded-request",
    )
    expected_response = RunPodHttpResponse(status_code=200, body=b"{}")
    transport = RecordedRunPodTransport(
        (
            RecordedRunPodExchange(
                request_body_sha256=exact_bytes_sha256(request.body),
                response=expected_response,
            ),
        )
    )
    credential = RunPodApiKey("runpod-test-secret-000000000000")

    assert asyncio.run(transport.post(request, credential)) == expected_response
    with pytest.raises(RunPodTransportError, match="no recorded"):
        asyncio.run(
            transport.post(
                replace(request, body=b'{"input":"different"}'),
                credential,
            )
        )
    assert transport.request_count == 2


def test_two_h100_report_binds_sessions_and_renders_measured_facts() -> None:
    report = _report(_configuration())
    rendered = report.render_markdown()

    assert report.safe_point.offered_concurrency == 4
    assert report.safe_point.aggregate_gpu_minutes_per_recording_hour == pytest.approx(2.0)
    assert "Pinned prompt/context contracts" in rendered
    assert "Production eligible: NO" in rendered
    assert "Fresh P6 namespaces" in rendered
    assert "Accepted queue P50/P95/P99 ms" in rendered
    assert "object://qualification/h100-inventory.json" in rendered
    assert _uuid(10) in rendered
    assert report.production_eligible is False


def test_saturation_point_rejects_unbound_workload_and_gpu_session() -> None:
    configuration = _configuration()
    session = _session(configuration, 20)
    with pytest.raises(ValueError, match="session workload"):
        ProviderSaturationPoint(
            configuration_digest=configuration.configuration_digest,
            qualification_session=session,
            run_namespace=session.run_namespace,
            offered_concurrency=1,
            capacity=_capacity(workload="c" * 64),
            telemetry=_telemetry(session),
        )

    wrong_session = _session(configuration, 21)
    with pytest.raises(ValueError, match="GPU telemetry"):
        ProviderSaturationPoint(
            configuration_digest=configuration.configuration_digest,
            qualification_session=session,
            run_namespace=session.run_namespace,
            offered_concurrency=1,
            capacity=_capacity(),
            telemetry=_telemetry(wrong_session),
        )


def test_two_h100_configuration_binds_to_active_runpod_limits_and_model() -> None:
    configuration = _configuration()
    endpoint = _endpoint_config(configuration)
    capabilities = _runpod_capabilities()

    configuration.validate_runpod_configuration(
        endpoint_config=endpoint,
        capabilities=capabilities,
        retry_policy=configuration.retry_policy,
    )
    with pytest.raises(ValueError, match="image limit"):
        configuration.validate_runpod_configuration(
            endpoint_config=endpoint,
            capabilities=capabilities.model_copy(update={"max_images_per_request": 11}),
            retry_policy=configuration.retry_policy,
        )
    with pytest.raises(ValueError, match="endpoint configuration"):
        configuration.validate_runpod_configuration(
            endpoint_config=endpoint.model_copy(
                update={"endpoint_url": "https://api.runpod.test/v2/other/runsync"}
            ),
            capabilities=capabilities,
            retry_policy=configuration.retry_policy,
        )
    with pytest.raises(ValueError, match="native batch"):
        configuration.validate_runpod_configuration(
            endpoint_config=endpoint.model_copy(update={"native_batch_enabled": False}),
            capabilities=capabilities,
            retry_policy=configuration.retry_policy,
        )
    with pytest.raises(ValueError, match="retry policy"):
        configuration.validate_runpod_configuration(
            endpoint_config=endpoint,
            capabilities=capabilities,
            retry_policy=RunPodRetryPolicy(
                version="runpod-retry-v2",
                max_attempts=2,
                base_delay_ms=0,
                max_delay_ms=0,
            ),
        )


def test_two_h100_configuration_rejects_native_batch_enablement_mismatch() -> None:
    with pytest.raises(ValueError, match="embedded RunPod endpoint"):
        _configuration(native_batch_enabled=False)


def test_two_h100_report_rejects_replay_or_unmeasured_provider_work() -> None:
    configuration = _configuration()
    session = _session(configuration, 30)
    with pytest.raises(ValueError, match="fresh"):
        ProviderSaturationPoint(
            configuration_digest=configuration.configuration_digest,
            qualification_session=session,
            run_namespace=session.run_namespace,
            offered_concurrency=1,
            capacity=_capacity(execution_mode="REPLAY"),
            telemetry=_telemetry(session),
        )

    no_calls = build_measured_capacity_report(
        MeasuredCapacityInput(
            workload_fingerprint=_DIGEST,
            evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
            provider_mode=ProviderMode.NETWORK_PROVIDER,
            execution_mode="FRESH",
            recording_count=1,
            recording_worker_count=1,
            camera_count=6,
            recording_duration_ns=_HOUR_NS,
            wall_time_ns=_HOUR_NS,
        )
    )
    with pytest.raises(ValueError, match="image, call, request, and token"):
        ProviderSaturationPoint(
            configuration_digest=configuration.configuration_digest,
            qualification_session=session,
            run_namespace=session.run_namespace,
            offered_concurrency=1,
            capacity=no_calls,
            telemetry=_telemetry(session),
        )


def test_saturation_point_rejects_capacity_workload_count_mismatch() -> None:
    configuration = _configuration()
    session = _session(configuration, 31)
    with pytest.raises(ValueError, match="provider_images"):
        ProviderSaturationPoint(
            configuration_digest=configuration.configuration_digest,
            qualification_session=session,
            run_namespace=session.run_namespace,
            offered_concurrency=1,
            capacity=_capacity(provider_images=1),
            telemetry=_telemetry(session),
        )


def test_report_revalidates_nested_saturation_point_after_model_copy() -> None:
    configuration = _configuration()
    report = _report(configuration)
    mutated_point = report.points[0].model_copy(
        update={"capacity": replace(report.points[0].capacity, provider_images=1)}
    )
    mutated_report = report.model_copy(update={"points": (mutated_point, report.points[1])})

    with pytest.raises(ValueError, match="provider_images"):
        mutated_report.validate_report()


def test_two_h100_topology_comparison_requires_declared_support() -> None:
    replicas = _report(_configuration(), session_offset=40)
    tensor = _report(
        _configuration(TwoH100Topology.TWO_CARD_TENSOR_PARALLEL),
        session_offset=50,
    )
    comparison = compare_two_h100_topologies(
        single_card_replicas=replicas,
        tensor_parallel=tensor,
    )
    assert "Two-card tensor parallel" in comparison.render_markdown()

    one_mode = _configuration(supported_topologies=(TwoH100Topology.TWO_SINGLE_CARD_REPLICAS,))
    with pytest.raises(ValueError, match="support for both modes"):
        compare_two_h100_topologies(
            single_card_replicas=_report(one_mode, session_offset=60),
            tensor_parallel=tensor,
        )


def test_provider_telemetry_derives_nearest_rank_percentiles_from_samples() -> None:
    configuration = _configuration()
    session = _session(configuration, 70)
    samples = tuple(
        _sample(
            value,
            input_plan_part_ordinal=index % 2,
            provider_queue_ms=value,
            provider_execution_ms=value + 2,
            time_to_first_token_ms=value + 1,
            end_to_end_ms=value + 3,
        )
        for index, value in enumerate((0, 10, 20, 30))
    )
    telemetry = ProviderRuntimeTelemetry.from_samples(
        samples=samples,
        http_requests=4,
        gpu=_gpu(session),
        adapter_transport_retry_count=1,
    )

    assert telemetry.provider_queue is not None
    assert telemetry.provider_execution is not None
    assert telemetry.end_to_end.p99_ms == 33
    assert telemetry.provider_queue.p50_ms == 10
    assert telemetry.provider_execution.p50_ms == 12
    assert telemetry.adapter_terminal_workload.logical_calls == 4
    assert telemetry.adapter_terminal_workload.input_tokens == 400

    accepted_latency = ProviderLatencyPercentiles.from_samples((1,))
    with pytest.raises(ValueError, match="match accepted"):
        ProviderRuntimeTelemetry(
            accepted_response_count=1,
            rejected_response_count=0,
            input_known_response_count=1,
            usage_known_response_count=1,
            input_known_attempt_count=1,
            usage_known_attempt_count=1,
            canonical_retry_attempt_count=0,
            adapter_terminal_workload=ProviderAdapterTerminalWorkload(
                provider_images=1,
                logical_calls=1,
                call_parts=1,
                provider_attempt_count=1,
                input_tokens=1,
                input_token_responses=1,
                output_tokens=1,
                output_token_responses=1,
                http_requests=1,
            ),
            provider_queue=accepted_latency,
            provider_execution=accepted_latency,
            time_to_first_token=ProviderLatencyPercentiles.from_samples((1, 2)),
            end_to_end=accepted_latency,
            gpu=_gpu(_session(configuration, 71)),
            adapter_transport_retry_count=0,
        )


def test_provider_telemetry_collapses_canonical_retry_to_one_final_success() -> None:
    configuration = _configuration()
    session = _session(configuration, 75)
    logical_invocation_value = 75_000
    retry_attempt = _sample(
        750,
        accepted=False,
        input_tokens_known=False,
        output_tokens_known=False,
        logical_invocation_value=logical_invocation_value,
    )
    final_success = _sample(
        751,
        logical_invocation_value=logical_invocation_value,
    )
    telemetry = ProviderRuntimeTelemetry.from_samples(
        samples=(retry_attempt, final_success),
        http_requests=2,
        gpu=_gpu(session),
        adapter_transport_retry_count=0,
    )

    assert telemetry.accepted_response_count == 1
    assert telemetry.rejected_response_count == 0
    assert telemetry.terminal_response_count == 1
    assert telemetry.canonical_retry_attempt_count == 1
    assert telemetry.input_known_response_count == 1
    assert telemetry.usage_known_response_count == 1
    assert telemetry.adapter_terminal_workload.call_parts == 1
    assert telemetry.adapter_terminal_workload.provider_attempt_count == 2

    with pytest.raises(ValueError, match="duplicate accepted"):
        ProviderRuntimeTelemetry.from_samples(
            samples=(
                final_success,
                _sample(
                    752,
                    logical_invocation_value=logical_invocation_value,
                ),
            ),
            http_requests=2,
            gpu=_gpu(session),
            adapter_transport_retry_count=0,
        )


def test_gpu_measurement_requires_h100_and_feasible_two_gpu_window() -> None:
    session = _session(_configuration(), 80)
    values = _gpu(session).model_dump(mode="python")
    with pytest.raises(ValueError, match="H100"):
        ProviderGpuMeasurement(**{**values, "gpu_sku": "NVIDIA A100"})
    with pytest.raises(ValueError, match="aggregate GPU seconds"):
        ProviderGpuMeasurement(**{**values, "aggregate_gpu_seconds": 8_000.0})


def test_collector_seals_one_session_and_keeps_transport_retries_separate() -> None:
    configuration = _configuration()
    session = _session(configuration, 90)
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace=session.run_namespace,
        offered_concurrency=4,
    )
    _record_accepted_samples(collector, session)
    collector.record_provider_http_requests(qualification_session=session, count=12)
    collector.record_provider_retries(qualification_session=session, count=2)

    point = collector.build_point(
        capacity=_capacity(retries=7),
        gpu=_gpu(session),
    )

    assert point.configuration_digest == configuration.configuration_digest
    assert point.offered_concurrency == 4
    assert point.telemetry.provider_execution is not None
    assert point.telemetry.provider_execution.p95_ms == 21
    assert point.telemetry.adapter_transport_retry_count == 2
    assert point.telemetry.adapter_terminal_workload.http_requests == 12
    assert point.capacity.retries == 7
    assert collector.timing_sample_count == 12
    with pytest.raises(ValueError, match="already sealed"):
        collector.build_point(capacity=_capacity(retries=7), gpu=_gpu(session))


def test_collector_rejects_cross_session_duplicate_and_missing_latency() -> None:
    configuration = _configuration()
    session = _session(configuration, 100)
    other_session = _session(configuration, 101)
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace=session.run_namespace,
        offered_concurrency=4,
    )
    with pytest.raises(ValueError, match="different session"):
        collector.record_provider_retries(qualification_session=other_session, count=1)

    for value in range(12):
        sample = _sample(200 + value, provider_execution_ms=210 + value)
        if value == 0:
            sample = sample.model_copy(update={"provider_execution_ms": None})
        _record_sample(collector, session, sample)
    with pytest.raises(ValueError, match="provider_execution samples must match accepted"):
        collector.build_point(capacity=_capacity(), gpu=_gpu(session))

    duplicate = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace=session.run_namespace,
        offered_concurrency=4,
    )
    sample = _sample(300)
    _record_sample(duplicate, session, sample)
    with pytest.raises(ValueError, match="duplicate request"):
        _record_sample(duplicate, session, sample)


def test_collector_marks_rejected_terminal_outcome_unsafe() -> None:
    configuration = _configuration()
    session = _session(configuration, 110)
    collector = ProviderQualificationCollector(
        qualification_session=session,
        run_namespace=session.run_namespace,
        offered_concurrency=8,
    )
    _record_accepted_samples(collector, session, start=400)
    _record_sample(
        collector,
        session,
        _sample(
            500,
            accepted=False,
            output_tokens_known=False,
        ),
    )
    collector.record_provider_http_requests(qualification_session=session, count=13)

    point = collector.build_point(
        capacity=_capacity(
            provider_images=130,
            logical_calls=13,
            http_requests=13,
            input_tokens=1_300,
        ),
        gpu=_gpu(session),
    )

    assert point.safe_envelope is False
    assert point.telemetry.terminal_response_count == 13
    assert point.telemetry.rejected_response_count == 1
    assert point.telemetry.usage_known_response_count == 12
    assert point.telemetry.adapter_terminal_workload.provider_images == 130
    assert point.telemetry.adapter_terminal_workload.input_tokens == 1_300
    assert point.telemetry.adapter_terminal_workload.call_parts == 13
    assert point.telemetry.adapter_terminal_workload.http_requests == 13


def test_report_rejects_safe_point_after_unsafe_boundary() -> None:
    configuration = _configuration()
    unsafe = _point(
        configuration,
        session_value=120,
        offered_concurrency=4,
        rejected_response_count=1,
    )
    safe = _point(
        configuration,
        session_value=121,
        offered_concurrency=8,
    )

    with pytest.raises(ValueError, match="must precede the first unsafe boundary"):
        TwoH100ProviderQualificationReport(
            configuration=configuration,
            endpoint_config=_endpoint_config(configuration),
            capabilities=_runpod_capabilities(),
            retry_policy=configuration.retry_policy,
            evidence_class=CapacityEvidenceClass.PRODUCTION_QUALIFICATION,
            points=(unsafe, safe),
        )


def test_run_provider_saturation_point_binds_actual_runpod_adapter() -> None:
    registry = SchemaRegistry()
    request = _qualification_request(_provider_schema(registry))
    assert request.input_plan is not None
    configuration = _configuration(
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
        )
    )
    session = _session(configuration, 130)
    context = ProviderQualificationRunContext(
        qualification_session=session,
        run_namespace=session.run_namespace,
    )
    qualified_request = request.model_copy(
        update={"metadata": context.bind_request_metadata(request.metadata)}
    )
    transport = _RetryThenSuccessTransport(_completed_qualification_response(qualified_request))
    captured: dict[str, ProviderQualificationCollector] = {}

    def adapter_factory(
        collector: ProviderQualificationCollector,
        received_context: ProviderQualificationRunContext,
    ) -> RunPodVisionAdapter:
        assert received_context == context
        captured["collector"] = collector
        return RunPodVisionAdapter(
            config=configuration.endpoint_configuration,
            credential=RunPodApiKey("runpod-test-secret-000000000000"),
            capabilities=_runpod_capabilities(),
            retry_policy=configuration.retry_policy,
            raw_store=InMemoryRawProviderBytesStore(),
            parser=StrictProviderClaimParser(
                registry,
                parser_version="qualification-test-parser-v1",
            ),
            transport=transport,
            qualification_observer=collector,
            qualification_session=received_context.qualification_session,
        )

    async def workload(
        adapter: object,
        received_context: ProviderQualificationRunContext,
    ) -> MeasuredCapacityInput:
        assert isinstance(adapter, RunPodVisionAdapter)
        assert adapter.qualification_session == session
        assert adapter.qualification_observer is captured["collector"]
        metadata = received_context.bind_request_metadata({"source": "p6"})
        assert metadata[RUNPOD_QUALIFICATION_SESSION_METADATA_KEY] == session.session_id
        assert metadata[RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY] == session.run_namespace
        with pytest.raises(ValueError, match="conflicting scope"):
            received_context.bind_request_metadata(
                {RUNPOD_QUALIFICATION_SESSION_METADATA_KEY: _uuid(999)}
            )

        result = await adapter.infer(qualified_request)
        assert isinstance(result, VisionInferenceSuccess)
        assert result.usage.input_tokens == 123
        assert result.usage.output_tokens == 7
        assert adapter.qualification_observed_request_ids == (qualified_request.request_id,)
        assert adapter.qualification_observed_http_request_count == 2
        assert adapter.qualification_observed_transport_retry_count == 1
        return _capacity_input(
            retries=7,
            provider_images=6,
            logical_calls=1,
            http_requests=2,
            input_tokens=123,
            output_tokens=7,
            output_token_responses=1,
        )

    point = asyncio.run(
        run_provider_saturation_point(
            configuration=configuration,
            context=context,
            offered_concurrency=4,
            workload_manifest_bytes=_WORKLOAD_MANIFEST_BYTES,
            adapter_factory=adapter_factory,
            workload=workload,
            gpu=_gpu(session),
        )
    )

    assert len(transport.requests) == 2
    assert point.run_namespace == context.run_namespace
    assert point.qualification_session == session
    assert point.capacity.retries == 7
    assert point.safe_envelope is True
    assert point.telemetry.adapter_transport_retry_count == 1
    assert point.telemetry.canonical_retry_attempt_count == 0
    assert point.telemetry.adapter_terminal_workload.http_requests == 2
    assert point.telemetry.adapter_terminal_workload.output_tokens == 7


def test_run_provider_saturation_point_rejects_excess_concurrency_before_dispatch() -> None:
    configuration = _configuration()
    session = _session(configuration, 135)
    context = ProviderQualificationRunContext(
        qualification_session=session,
        run_namespace=session.run_namespace,
    )

    async def workload(
        _adapter: object,
        _context: ProviderQualificationRunContext,
    ) -> MeasuredCapacityInput:
        pytest.fail("an over-limit saturation point must be rejected before dispatch")

    with pytest.raises(ValueError, match="configured provider concurrency"):
        asyncio.run(
            run_provider_saturation_point(
                configuration=configuration,
                context=context,
                offered_concurrency=configuration.max_concurrent_requests + 1,
                workload_manifest_bytes=_WORKLOAD_MANIFEST_BYTES,
                adapter_factory=lambda _collector, _context: pytest.fail(
                    "an over-limit saturation point must not construct an adapter"
                ),
                workload=workload,
                gpu=_gpu(session),
            )
        )


def test_run_provider_saturation_point_rejects_manual_collector_injection() -> None:
    configuration = _configuration()
    session = _session(configuration, 140)
    context = ProviderQualificationRunContext(
        qualification_session=session,
        run_namespace=session.run_namespace,
    )
    captured: dict[str, ProviderQualificationCollector] = {}

    def adapter_factory(
        collector: ProviderQualificationCollector,
        received_context: ProviderQualificationRunContext,
    ) -> RunPodVisionAdapter:
        captured["collector"] = collector
        return RunPodVisionAdapter(
            config=configuration.endpoint_configuration,
            credential=RunPodApiKey("runpod-test-secret-000000000000"),
            capabilities=_runpod_capabilities(),
            retry_policy=configuration.retry_policy,
            raw_store=InMemoryRawProviderBytesStore(),
            parser=StrictProviderClaimParser(
                SchemaRegistry(),
                parser_version="qualification-test-parser-v1",
            ),
            transport=RecordedRunPodTransport(
                (
                    RecordedRunPodExchange(
                        request_body_sha256="0" * 64,
                        response=RunPodHttpResponse(status_code=200, body=b"{}"),
                    ),
                )
            ),
            qualification_observer=collector,
            qualification_session=received_context.qualification_session,
        )

    async def workload(
        _adapter: object,
        _context: ProviderQualificationRunContext,
    ) -> MeasuredCapacityInput:
        collector = captured["collector"]
        _record_accepted_samples(collector, session, start=800)
        collector.record_provider_http_requests(qualification_session=session, count=12)
        return _capacity_input()

    with pytest.raises(ValueError, match="scoped RunPod adapter"):
        asyncio.run(
            run_provider_saturation_point(
                configuration=configuration,
                context=context,
                offered_concurrency=4,
                workload_manifest_bytes=_WORKLOAD_MANIFEST_BYTES,
                adapter_factory=adapter_factory,
                workload=workload,
                gpu=_gpu(session),
            )
        )
