from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from robata.adapters.sqlite_inference_evidence import SQLiteInferenceEvidenceLedger
from robata.contracts.cameras import CAMERA_IDS
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import SchemaRegistry
from robata.inference import local_hf_adapter as local_hf
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
    InferenceInputPlanner,
    InputPlanTarget,
    PromptOutputContract,
    RenderedArtifact,
    RenderedProviderItem,
    TransformOperation,
)
from robata.inference.local_hf_adapter import (
    LOCAL_HF_COMPACT_BARE_LABEL_POLICY_VERSION,
    LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION,
    LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION,
    LOCAL_HF_COMPACT_SINGLE_DECISION_POLICY_VERSION,
    LOCAL_HF_COMPACT_TOKEN_ORDER_POLICY_VERSION,
    LOCAL_HF_HYBRID_BATCH_MAX_SIZE,
    LOCAL_HF_HYBRID_BATCH_POLICY_VERSION,
    LOCAL_HF_LOOPBACK_ADAPTER_VERSION,
    LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
    LOCAL_HF_LOOPBACK_ENDPOINT_URL,
    LocalHfHttpRequest,
    LocalHfHttpResponse,
    LocalHfLoopbackAdapterConfig,
    LocalHfLoopbackVisionAdapter,
)
from robata.inference.local_hf_endpoint import (
    LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION,
    LOCAL_HF_BATCH_POLICY_VERSION,
    LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION,
    LOCAL_HF_ENDPOINT_RESPONSE_VERSION,
    LocalHfBatchEndpointRequest,
    LocalHfEndpointRequest,
)
from robata.inference.models import (
    ConcurrencyClass,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    VisionTask,
)
from robata.inference.offline_fixture import (
    InMemoryRawProviderBytesStore,
    StrictProviderClaimParser,
)
from robata.inference.orchestrator import InferenceIntent

NOW = "2026-08-06T12:00:00Z"
PROVIDER = "local-huggingface"
MODEL_NAME = "Qwen3-VL-4B-Instruct"
MODEL_VERSION = "local"
RAW_ABSTENTION = '{"claims":[],"abstained":true}'


@dataclass(frozen=True, slots=True)
class _Fixture:
    registry: SchemaRegistry
    request: VisionInferenceRequest
    paths: tuple[Path, ...]
    payloads: tuple[bytes, ...]


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


def _capabilities(*, task: VisionTask = VisionTask.ACTION_EVIDENCE) -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(500),
        snapshot_digest=_digest(501),
        provider=PROVIDER,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        supported_tasks=(task,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=6,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=100_000,
        max_input_tokens=1_000,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.SERIAL,
        data_handling_policy_version="local-loopback-data-v1",
        observed_at=NOW,
    )


def _fixture(
    tmp_path: Path,
    *,
    image_count: int = 6,
    task: VisionTask = VisionTask.ACTION_EVIDENCE,
    frame_counts_override: tuple[int, ...] | None = None,
) -> _Fixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = SchemaRegistry()
    schema = _provider_schema(registry)
    planner = InferenceInputPlanner(INFERENCE_INPUT_PLANNER_VERSION)
    frame_counts = (
        frame_counts_override
        if frame_counts_override is not None
        else tuple(2 if ordinal == 0 and image_count == 7 else 1 for ordinal in range(6))
    )
    camera_frames: list[CatalogCamera] = []
    rendered_items: list[RenderedProviderItem] = []
    paths: list[Path] = []
    payloads: list[bytes] = []
    provider_ordinal = 0
    for camera_ordinal, (camera_id, frame_count) in enumerate(
        zip(CAMERA_IDS, frame_counts, strict=True)
    ):
        frames: list[CatalogFrame] = []
        for frame_ordinal in range(frame_count):
            payload = f"local-png-{camera_ordinal}-{frame_ordinal}".encode("ascii")
            path = tmp_path / f"camera-{camera_ordinal}-{frame_ordinal}.png"
            path.write_bytes(payload)
            paths.append(path)
            payloads.append(payload)
            frame = CatalogFrame(
                frame_id=_uuid(100 + provider_ordinal),
                ordinal=frame_ordinal,
                aligned_timestamp_ns=1_000_000_000 + provider_ordinal,
                source_timestamp_ns=1_700_000_000_000_000_000 + provider_ordinal,
                source_artifact_uri=path.resolve().as_uri(),
                source_artifact_sha256=exact_bytes_sha256(payload),
                source_artifact_bytes=len(payload),
                media_type="image/png",
                encoding="png",
                width=640,
                height=480,
            )
            frames.append(frame)
            rendered_items.append(
                RenderedProviderItem(
                    provider_item_ordinal=provider_ordinal,
                    package_id=_uuid(300),
                    package_ordinal=0,
                    camera_id=camera_id,
                    camera_ordinal=camera_ordinal,
                    frame_id=frame.frame_id,
                    frame_ordinal=frame_ordinal,
                    aligned_timestamp_ns=frame.aligned_timestamp_ns,
                    source_timestamp_ns=frame.source_timestamp_ns,
                    source_artifact_sha256=frame.source_artifact_sha256,
                    artifact=RenderedArtifact(
                        artifact_id=_uuid(400 + provider_ordinal),
                        uri=path.resolve().as_uri(),
                        sha256=exact_bytes_sha256(payload),
                        byte_count=len(payload),
                        media_type="image/png",
                        encoding="png",
                        width=640,
                        height=480,
                    ),
                    transform=FrameTransform.create(
                        operation=TransformOperation.NONE,
                        policy_version="render-v1",
                    ),
                )
            )
            provider_ordinal += 1
        camera_frames.append(
            CatalogCamera(camera_id=camera_id, ordinal=camera_ordinal, frames=tuple(frames))
        )
    package = CatalogPackage(
        package_id=_uuid(300),
        ordinal=0,
        semantic_content_sha256=_digest(301),
        manifest_bytes_sha256=_digest(302),
        cameras=tuple(camera_frames),
    )
    catalog = planner.build_request_catalog(
        request_catalog_id=_uuid(303),
        task=task,
        packages=(package,),
        created_at=NOW,
    )
    plan = planner.build(
        input_plan_id=_uuid(304),
        created_at=NOW,
        request_catalog=catalog,
        target=InputPlanTarget(
            provider=PROVIDER,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            adapter_version=LOCAL_HF_LOOPBACK_ADAPTER_VERSION,
            planner_version=INFERENCE_INPUT_PLANNER_VERSION,
            capability_snapshot_id=_uuid(500),
            capability_snapshot_sha256=_digest(501),
        ),
        rendered_items=tuple(rendered_items),
        prompt_output=PromptOutputContract(
            prompt_version=f"local-{task.value.lower()}-prompt-v1",
            prompt_sha256=_digest(503),
            rendered_message_sha256=_digest(504),
            provider_response_schema_sha256=schema.sha256,
            enriched_domain_schema_sha256=_digest(505),
            protocol_mode="json-schema",
            tool_mode="none",
        ),
        applicable_limits=ApplicableProviderLimits(
            max_images_per_request=image_count,
            max_pixels_per_image=640 * 480,
            max_payload_bytes_per_request=100_000,
            max_input_tokens_per_request=1_000,
        ),
        call_parts=(
            CallPartSpec(
                start_item_ordinal=0,
                end_item_ordinal_exclusive=image_count,
                measured_input_tokens=120,
            ),
        ),
        idempotency_policy_version="local-idempotency-v1",
        reduction_policy="single-part",
        reduction_policy_version="local-reduction-v1",
    )
    part = plan.call_plan.parts[0]
    request = VisionInferenceRequest(
        schema_version="1.0",
        logical_invocation_id=_uuid(600),
        request_id=_uuid(601),
        idempotency_key="local-hf-logical-request",
        provider=PROVIDER,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        package_set_id=_uuid(602),
        package_inputs=(
            PackageInput(
                package_id=package.package_id,
                package_semantic_content_sha256=package.semantic_content_sha256,
                package_manifest_sha256=package.manifest_bytes_sha256,
                role="primary",
                ordinal=0,
            ),
        ),
        package_input_set_sha256=_digest(603),
        task=task,
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
        model_policy_version="local-model-policy-v1",
        generation_config={"max_new_tokens": 12, "temperature": 0},
        provider_idempotency_key=part.idempotency_key,
        timeout_ms=10_000,
        metadata={},
    )
    return _Fixture(
        registry=registry,
        request=request,
        paths=tuple(paths),
        payloads=tuple(payloads),
    )


class _CapturingTransport:
    def __init__(self, *, output_text: str = RAW_ABSTENTION) -> None:
        self.output_text = output_text
        self.requests: list[LocalHfHttpRequest] = []

    async def post(self, request: LocalHfHttpRequest) -> LocalHfHttpResponse:
        self.requests.append(request)
        endpoint_request = LocalHfEndpointRequest.model_validate_json(request.body, strict=True)
        response = {
            "contract_version": LOCAL_HF_ENDPOINT_RESPONSE_VERSION,
            "request_id": endpoint_request.request_id,
            "model_identifier": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "quantization": "bnb-nf4-double-quant",
            "precision": "bfloat16-compute",
            "input_image_count": len(endpoint_request.images),
            "rendered_image_sizes": [[640, 480] for _ in endpoint_request.images],
            "prompt_tokens": 21,
            "output_tokens": 4,
            "load_seconds": 0.0,
            "generation_seconds": 0.01,
            "gpu_name": "test-gpu",
            "gpu_total_bytes": 1,
            "gpu_free_before_bytes": 1,
            "gpu_allocated_after_load_bytes": 0,
            "gpu_peak_allocated_bytes": 1,
            "output_text": self.output_text,
        }
        return LocalHfHttpResponse(status_code=200, body=canonical_json_bytes(response))


def _intent(request: VisionInferenceRequest) -> InferenceIntent:
    return InferenceIntent(
        inference_id=_uuid(700),
        logical_invocation_id=request.logical_invocation_id,
        request_id=request.request_id,
        idempotency_key=request.idempotency_key,
        task=request.task,
        provider=request.provider,
        model_name=request.model_name,
        model_version=request.model_version,
        adapter_version=LOCAL_HF_LOOPBACK_ADAPTER_VERSION,
        mcap_id=_uuid(701),
        camera_mapping_run_id=_uuid(702),
        alignment_id=_uuid(703),
        start_ns=1_000_000_000,
        end_ns=1_000_000_001,
        input_config={},
        sampling_config={},
        input_plan_id=request.input_plan_id,
        input_plan_semantic_sha256=request.input_plan_semantic_sha256,
        input_plan_part_ordinal=request.input_plan_part_ordinal,
        input_plan_part_count=request.input_plan_part_count,
        input_plan_part_semantic_sha256=request.input_plan_part_semantic_sha256,
        attempt=1,
        retry_count=0,
        shadow=False,
        request=request,
        queued_at=NOW,
        created_at=NOW,
    )


def _adapter(
    fixture: _Fixture,
    *,
    transport: _CapturingTransport,
    evidence_ledger: InMemoryRawProviderBytesStore | SQLiteInferenceEvidenceLedger,
) -> LocalHfLoopbackVisionAdapter:
    return LocalHfLoopbackVisionAdapter(
        capabilities=_capabilities(task=fixture.request.task),
        parser=StrictProviderClaimParser(fixture.registry, parser_version="local-hf-parser-v1"),
        evidence_ledger=evidence_ledger,
        config=LocalHfLoopbackAdapterConfig(provider=PROVIDER),
        transport=transport,
    )


def test_loopback_adapter_sends_verified_files_and_persists_raw_sqlite_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    transport = _CapturingTransport()
    ledger = SQLiteInferenceEvidenceLedger(tmp_path / "evidence.sqlite3", fixture.registry)
    try:
        ledger.append_intent(_intent(fixture.request))
        result = asyncio.run(
            _adapter(fixture, transport=transport, evidence_ledger=ledger).infer(fixture.request)
        )

        assert isinstance(result, VisionInferenceSuccess)
        assert result.status is InferenceStatus.SUCCEEDED
        assert result.provider_request_id == fixture.request.request_id
        assert result.normalized_output.payload == {"claims": [], "abstained": True}
        assert result.usage.input_images == 6
        assert result.usage.input_tokens == 21
        assert result.usage.output_tokens == 4
        assert result.raw_output_artifact_id is not None
        assert ledger.get(result.raw_output_artifact_id).data == RAW_ABSTENTION.encode("utf-8")

        assert len(transport.requests) == 1
        sent = LocalHfEndpointRequest.model_validate_json(transport.requests[0].body, strict=True)
        assert len(sent.images) == 6
        assert (
            tuple(base64.b64decode(image.base64_data) for image in sent.images) == fixture.payloads
        )
        prompt = json.loads(sent.prompt)
        assert prompt["protocol"] == "robata-provider-claim-v1"
        compact_contract = prompt["compact_output_contract"]
        assert (
            compact_contract["single_value_broadcast_policy"]
            == LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION
        )
        assert (
            compact_contract["bare_label_recovery_policy"]
            == LOCAL_HF_COMPACT_BARE_LABEL_POLICY_VERSION
        )
        assert (
            compact_contract["scalar_value_policy"] == LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION
        )
        assert (
            compact_contract["single_decision_policy"]
            == LOCAL_HF_COMPACT_SINGLE_DECISION_POLICY_VERSION
        )
        assert (
            compact_contract["reference_token_order_policy"]
            == LOCAL_HF_COMPACT_TOKEN_ORDER_POLICY_VERSION
        )
        assert (
            compact_contract["normalization_contract_sha256"]
            == local_hf.local_hf_compact_prompt_normalization_contract_sha256()
        )
        assert len(prompt["evidence_catalog"]) == 6
        assert all(
            entry["correlation_token"].startswith("ref:") for entry in prompt["evidence_catalog"]
        )
        adapter = _adapter(fixture, transport=transport, evidence_ledger=ledger)
        assert adapter.endpoint_url == LOCAL_HF_LOOPBACK_ENDPOINT_URL
        assert adapter.production_eligible is False
        assert adapter.canonical_authority is False
    finally:
        ledger.close()


def test_loopback_adapter_persists_raw_output_before_strict_parse_failure(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw_store = InMemoryRawProviderBytesStore()
    duplicate_key_output = '{"claims":[],"claims":[],"abstained":true}'
    transport = _CapturingTransport(output_text=duplicate_key_output)

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code == "DUPLICATE_JSON_KEY"
    assert result.raw_output_artifact_id is not None
    assert raw_store.get(result.raw_output_artifact_id).data == duplicate_key_output.encode("utf-8")
    assert len(transport.requests) == 1


def test_loopback_adapter_fails_closed_when_selected_file_hash_changes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.paths[0].write_bytes(b"tampered")
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport()

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.FAILED
    assert result.failure.code == "LOCAL_HF_FILE_BYTE_COUNT_MISMATCH"
    assert raw_store.list_records() == ()
    assert transport.requests == []


def test_loopback_adapter_rejects_more_than_six_selected_images(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, image_count=7)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport()

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.FAILED
    assert result.failure.code == "LOCAL_HF_TOO_MANY_IMAGES"
    assert raw_store.list_records() == ()
    assert transport.requests == []


def test_loopback_adapter_broadcasts_single_compact_decision_across_part(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport(output_text='["SUPPORTING"]')

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceSuccess)
    assert len(result.normalized_output.payload["claims"]) == 6
    assert all(
        claim["observation"] == "SUPPORTING" for claim in result.normalized_output.payload["claims"]
    )


def test_loopback_adapter_recovers_one_exact_bare_allowed_label(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport(output_text="SUPPORTING")

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceSuccess)
    assert len(result.normalized_output.payload["claims"]) == 6
    assert all(
        claim["observation"] == "SUPPORTING" for claim in result.normalized_output.payload["claims"]
    )
    assert result.raw_output_artifact_id is not None
    assert raw_store.get(result.raw_output_artifact_id).data == b"SUPPORTING"


@pytest.mark.parametrize(
    "output_text",
    (
        "SUPPORTING because the image is clear",
        "SUPPORTING,",
        "```SUPPORTING```",
        "NOT_ALLOWED",
    ),
)
def test_loopback_adapter_rejects_non_exact_bare_compact_output(
    tmp_path: Path,
    output_text: str,
) -> None:
    fixture = _fixture(tmp_path)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport(output_text=output_text)

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code == "INVALID_JSON"
    assert result.raw_output_artifact_id is not None
    assert raw_store.get(result.raw_output_artifact_id).data == output_text.encode("utf-8")


def test_compact_camera_grouping_collapses_only_unanimous_image_values(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, image_count=7)
    assert fixture.request.input_plan is not None
    items = fixture.request.input_plan.rendered_items

    groups = local_hf._compact_camera_groups(items)

    assert len(groups) == 6
    assert len(groups[0]) == 2
    assert (
        local_hf._compact_camera_group_values(
            values=["SUPPORTING"] * len(items),
            items=items,
            groups=groups,
        )
        == ("SUPPORTING",) * 6
    )
    with pytest.raises(ValueError, match="disagree within one package/camera group"):
        local_hf._compact_camera_group_values(
            values=["SUPPORTING", "NO_EVENT", *("SUPPORTING" for _ in items[2:])],
            items=items,
            groups=groups,
        )


def test_loopback_adapter_compact_scalar_observation_is_scoped_and_explicit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport(output_text='{"observation":"SUPPORTING"}')

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceSuccess)
    assert len(result.normalized_output.payload["claims"]) == 6
    assert all(
        claim["observation"] == "SUPPORTING" for claim in result.normalized_output.payload["claims"]
    )
    sent = LocalHfEndpointRequest.model_validate_json(transport.requests[0].body, strict=True)
    compact_contract = json.loads(sent.prompt)["compact_output_contract"]
    assert compact_contract["scalar_value_policy"] == LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION
    assert (
        compact_contract["single_value_broadcast_policy"]
        == LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION
    )


@pytest.mark.parametrize(
    "output_text",
    (
        '{"observation":true}',
        '{"observations":{"value":"SUPPORTING"}}',
        '{"observation":"SUPPORTING","decisions":["SUPPORTING"]}',
    ),
)
def test_loopback_adapter_rejects_untyped_or_ambiguous_compact_object_recovery(
    tmp_path: Path,
    output_text: str,
) -> None:
    fixture = _fixture(tmp_path)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport(output_text=output_text)

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code in {"INVALID_JSON", "INVALID_SCHEMA"}
    assert result.raw_output_artifact_id is not None
    assert raw_store.get(result.raw_output_artifact_id).data == output_text.encode("utf-8")


@pytest.mark.parametrize(
    "task",
    (VisionTask.EVENT_PROPOSAL, VisionTask.FUSION_ADJUDICATION),
)
def test_loopback_adapter_compact_event_and_fusion_abstain_is_explicit(
    tmp_path: Path,
    task: VisionTask,
) -> None:
    fixture = _fixture(tmp_path, task=task)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport(output_text='["ABSTAIN"]')

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceSuccess)
    assert result.normalized_output.payload == {"claims": [], "abstained": True}


@pytest.mark.parametrize(
    ("task", "output_text"),
    (
        (VisionTask.EVENT_PROPOSAL, '["EVENT"]'),
        (VisionTask.EVENT_PROPOSAL, '["PROPOSED","PROPOSED"]'),
        (VisionTask.EVENT_PROPOSAL, "[]"),
        (VisionTask.FUSION_ADJUDICATION, '["REJECT"]'),
        (VisionTask.FUSION_ADJUDICATION, '["PROPOSED","CONFLICT"]'),
        (VisionTask.FUSION_ADJUDICATION, "[]"),
    ),
)
def test_loopback_adapter_compact_event_and_fusion_reject_nonexact_or_nonsingular_decisions(
    tmp_path: Path,
    task: VisionTask,
    output_text: str,
) -> None:
    fixture = _fixture(tmp_path, task=task)
    raw_store = InMemoryRawProviderBytesStore()
    transport = _CapturingTransport(output_text=output_text)

    result = asyncio.run(
        _adapter(fixture, transport=transport, evidence_ledger=raw_store).infer(fixture.request)
    )

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code == "INVALID_JSON"


def test_loopback_adapter_compact_event_and_fusion_tokens_are_deterministic(tmp_path: Path) -> None:
    event_fixture = _fixture(tmp_path / "event", task=VisionTask.EVENT_PROPOSAL)
    event_store = InMemoryRawProviderBytesStore()
    event_transport = _CapturingTransport(output_text='["PROPOSED"]')
    event_adapter = _adapter(
        event_fixture,
        transport=event_transport,
        evidence_ledger=event_store,
    )
    assert event_fixture.request.input_plan is not None
    event_items = event_fixture.request.input_plan.rendered_items
    expected_event_tokens = tuple(
        sorted(event_adapter._allowed_reference_tokens(event_fixture.request, event_items))
    )

    event_result = asyncio.run(event_adapter.infer(event_fixture.request))

    assert isinstance(event_result, VisionInferenceSuccess)
    event_claim = event_result.normalized_output.payload["claims"][0]
    assert tuple(event_claim["evidence_tokens"]) == expected_event_tokens

    fusion_fixture = _fixture(tmp_path / "fusion", task=VisionTask.FUSION_ADJUDICATION)
    fusion_store = InMemoryRawProviderBytesStore()
    fusion_transport = _CapturingTransport(output_text='["CONFLICT"]')
    fusion_adapter = _adapter(
        fusion_fixture,
        transport=fusion_transport,
        evidence_ledger=fusion_store,
    )
    assert fusion_fixture.request.input_plan is not None
    fusion_items = fusion_fixture.request.input_plan.rendered_items
    expected_fusion_tokens = tuple(
        sorted(fusion_adapter._allowed_reference_tokens(fusion_fixture.request, fusion_items))
    )

    fusion_result = asyncio.run(fusion_adapter.infer(fusion_fixture.request))

    assert isinstance(fusion_result, VisionInferenceSuccess)
    fusion_claim = fusion_result.normalized_output.payload["claims"][0]
    assert fusion_claim["observation"] == "CONFLICT"
    assert tuple(fusion_claim["evidence_tokens"]) == (expected_fusion_tokens[0],)


def test_compact_prompt_normalization_contract_digest_is_stable_and_complete() -> None:
    contract = local_hf.local_hf_compact_prompt_normalization_contract()
    digest = local_hf.local_hf_compact_prompt_normalization_contract_sha256()

    assert digest == exact_bytes_sha256(canonical_json_bytes(contract))
    assert digest == local_hf.local_hf_compact_prompt_normalization_contract_sha256()
    assert contract["protocol"] == local_hf.LOCAL_HF_COMPACT_PROMPT_PROTOCOL
    assert contract["response_instruction"] == local_hf.LOCAL_HF_COMPACT_RESPONSE_INSTRUCTION
    assert set(contract["allowed_values_by_task"]) == {task.value for task in VisionTask}
    assert contract["policies"] == {
        "bare_label_recovery": LOCAL_HF_COMPACT_BARE_LABEL_POLICY_VERSION,
        "scalar_value": LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION,
        "single_value_broadcast": LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION,
        "single_decision": LOCAL_HF_COMPACT_SINGLE_DECISION_POLICY_VERSION,
        "reference_token_order": LOCAL_HF_COMPACT_TOKEN_ORDER_POLICY_VERSION,
        "camera_group": local_hf.LOCAL_HF_COMPACT_CAMERA_GROUP_POLICY_VERSION,
        "dense_coordinate_reduction": local_hf.LOCAL_HF_DENSE_COORDINATE_REDUCTION_POLICY_VERSION,
        "endpoint_idempotency": LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION,
    }


class _HybridBatchTransport:
    def __init__(
        self,
        *,
        outputs: dict[str, str] | None = None,
        reverse_batch_members: bool = False,
        batch_status_code: int = 200,
        malformed_batch_body: bytes | None = None,
        batch_error: Exception | None = None,
    ) -> None:
        self.outputs = outputs or {}
        self.reverse_batch_members = reverse_batch_members
        self.batch_status_code = batch_status_code
        self.malformed_batch_body = malformed_batch_body
        self.batch_error = batch_error
        self.requests: list[LocalHfHttpRequest] = []

    async def post(self, request: LocalHfHttpRequest) -> LocalHfHttpResponse:
        self.requests.append(request)
        if request.url == LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL:
            if self.batch_error is not None:
                raise self.batch_error
            if self.malformed_batch_body is not None:
                return LocalHfHttpResponse(status_code=200, body=self.malformed_batch_body)
            if self.batch_status_code != 200:
                return LocalHfHttpResponse(status_code=self.batch_status_code, body=b"{}")
            batch_request = LocalHfBatchEndpointRequest.model_validate_json(
                request.body,
                strict=True,
            )
            members = [
                {
                    "idempotency_key": member.idempotency_key,
                    "request_id": member.request.request_id,
                    "disposition": "GENERATED",
                    "input_image_count": len(member.request.images),
                    "rendered_image_sizes": [[640, 480] for _image in member.request.images],
                    "prompt_tokens": 21,
                    "output_tokens": 4,
                    "output_text": self.outputs.get(
                        member.request.request_id,
                        '["SUPPORTING"]',
                    ),
                }
                for member in batch_request.members
            ]
            if self.reverse_batch_members:
                members.reverse()
            response = {
                "contract_version": LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION,
                "batch_policy_version": LOCAL_HF_BATCH_POLICY_VERSION,
                "batch_request_sha256": batch_request.batch_request_sha256,
                "model_identifier": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "quantization": "bnb-nf4-double-quant",
                "precision": "bfloat16-compute",
                "load_seconds": 0.0,
                "gpu_name": "test-gpu",
                "gpu_total_bytes": 1,
                "gpu_free_before_bytes": 1,
                "gpu_allocated_after_load_bytes": 0,
                "physical_generation_seconds": 0.01,
                "physical_gpu_peak_allocated_bytes": 1,
                "generated_member_count": len(members),
                "replay_member_count": 0,
                "members": members,
            }
            return LocalHfHttpResponse(status_code=200, body=canonical_json_bytes(response))

        endpoint_request = LocalHfEndpointRequest.model_validate_json(request.body, strict=True)
        response = {
            "contract_version": LOCAL_HF_ENDPOINT_RESPONSE_VERSION,
            "request_id": endpoint_request.request_id,
            "model_identifier": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "quantization": "bnb-nf4-double-quant",
            "precision": "bfloat16-compute",
            "input_image_count": len(endpoint_request.images),
            "rendered_image_sizes": [[640, 480] for _image in endpoint_request.images],
            "prompt_tokens": 21,
            "output_tokens": 4,
            "load_seconds": 0.0,
            "generation_seconds": 0.01,
            "gpu_name": "test-gpu",
            "gpu_total_bytes": 1,
            "gpu_free_before_bytes": 1,
            "gpu_allocated_after_load_bytes": 0,
            "gpu_peak_allocated_bytes": 1,
            "output_text": self.outputs.get(endpoint_request.request_id, '["SUPPORTING"]'),
        }
        return LocalHfHttpResponse(status_code=200, body=canonical_json_bytes(response))


def _request_variant(
    request: VisionInferenceRequest,
    ordinal: int,
    *,
    provider_idempotency_key: str | None = None,
    max_new_tokens: int = 12,
) -> VisionInferenceRequest:
    document = request.model_dump(mode="python")
    document.update(
        {
            "logical_invocation_id": _uuid(800 + ordinal * 2),
            "request_id": _uuid(801 + ordinal * 2),
            "idempotency_key": f"logical-batch-{ordinal}",
            "provider_idempotency_key": (provider_idempotency_key or f"provider-batch-{ordinal}"),
            "generation_config": {"max_new_tokens": max_new_tokens, "temperature": 0},
        }
    )
    return VisionInferenceRequest.model_validate(document, strict=True)


def _single_claim_fixture(tmp_path: Path) -> _Fixture:
    return _fixture(
        tmp_path,
        frame_counts_override=(6, 0, 0, 0, 0, 0),
    )


def test_loopback_batch_capability_is_explicit_and_versioned(tmp_path: Path) -> None:
    fixture = _single_claim_fixture(tmp_path)
    adapter = _adapter(
        fixture,
        transport=_HybridBatchTransport(),
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )

    capability = adapter.native_batch_capability

    assert capability.supported is True
    assert capability.adapter_policy_version == LOCAL_HF_HYBRID_BATCH_POLICY_VERSION
    assert capability.max_batch_size == LOCAL_HF_HYBRID_BATCH_MAX_SIZE == 4
    assert capability.max_dispatch_size == 8
    assert capability.native_admission == "EXACTLY_ONE_CLAIM_GROUP"
    assert capability.multi_claim_route == "SERIAL_V1"
    assert capability.hidden_error_fallback is False
    assert adapter.native_batch_policy_version == LOCAL_HF_HYBRID_BATCH_POLICY_VERSION
    assert adapter.native_batch_max_size == 4


def test_loopback_batch_dispatches_four_single_claim_members_once_and_preserves_order(
    tmp_path: Path,
) -> None:
    fixture = _single_claim_fixture(tmp_path)
    requests = tuple(_request_variant(fixture.request, ordinal) for ordinal in range(4))
    outputs = {
        request.request_id: f'["{label}"]'
        for request, label in zip(
            requests,
            ("SUPPORTING", "PARTIAL", "NO_EVENT", "OCCLUDED"),
            strict=True,
        )
    }
    transport = _HybridBatchTransport(outputs=outputs)
    raw_store = InMemoryRawProviderBytesStore()
    adapter = _adapter(fixture, transport=transport, evidence_ledger=raw_store)

    results = asyncio.run(adapter.infer_batch(requests))

    assert len(transport.requests) == 1
    assert transport.requests[0].url == LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL
    sent = LocalHfBatchEndpointRequest.model_validate_json(
        transport.requests[0].body,
        strict=True,
    )
    assert len(sent.members) == 4
    assert tuple(member.request.request_id for member in sent.members) == tuple(
        request.request_id for request in requests
    )
    assert all(isinstance(result, VisionInferenceSuccess) for result in results)
    assert tuple(result.provider_request_id for result in results) == tuple(
        request.request_id for request in requests
    )
    assert tuple(
        result.normalized_output.payload["claims"][0]["observation"]
        for result in results
        if isinstance(result, VisionInferenceSuccess)
    ) == ("SUPPORTING", "PARTIAL", "NO_EVENT", "OCCLUDED")
    assert len(raw_store.list_records()) == 4


def test_loopback_batch_mixed_partition_keeps_multi_claim_on_exact_serial_wire(
    tmp_path: Path,
) -> None:
    single_fixture = _single_claim_fixture(tmp_path / "single")
    multi_fixture = _fixture(tmp_path / "multi")
    first = _request_variant(single_fixture.request, 10)
    multi = _request_variant(multi_fixture.request, 11)
    last = _request_variant(single_fixture.request, 12)

    serial_transport = _CapturingTransport(output_text='["SUPPORTING"]')
    serial_result = asyncio.run(
        _adapter(
            multi_fixture,
            transport=serial_transport,
            evidence_ledger=InMemoryRawProviderBytesStore(),
        ).infer(multi)
    )
    assert isinstance(serial_result, VisionInferenceSuccess)

    transport = _HybridBatchTransport()
    adapter = _adapter(
        single_fixture,
        transport=transport,
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )
    results = asyncio.run(adapter.infer_batch((first, multi, last)))

    assert all(isinstance(result, VisionInferenceSuccess) for result in results)
    assert tuple(result.provider_request_id for result in results) == (
        first.request_id,
        multi.request_id,
        last.request_id,
    )
    assert tuple(request.url for request in transport.requests) == (
        LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
        LOCAL_HF_LOOPBACK_ENDPOINT_URL,
    )
    batch_request = LocalHfBatchEndpointRequest.model_validate_json(
        transport.requests[0].body,
        strict=True,
    )
    assert tuple(member.request.request_id for member in batch_request.members) == (
        first.request_id,
        last.request_id,
    )
    assert transport.requests[1].body == serial_transport.requests[0].body
    assert transport.requests[1].idempotency_key == multi.provider_idempotency_key


def test_loopback_batch_member_idempotency_binds_policy_provider_key_and_exact_body(
    tmp_path: Path,
) -> None:
    fixture = _single_claim_fixture(tmp_path)
    first = _request_variant(fixture.request, 20, provider_idempotency_key="shared-provider-key")
    second = _request_variant(fixture.request, 21, provider_idempotency_key="shared-provider-key")
    transport = _HybridBatchTransport()
    adapter = _adapter(
        fixture,
        transport=transport,
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )

    results = asyncio.run(adapter.infer_batch((first, second)))

    assert all(isinstance(result, VisionInferenceSuccess) for result in results)
    sent = LocalHfBatchEndpointRequest.model_validate_json(
        transport.requests[0].body,
        strict=True,
    )
    keys = tuple(member.idempotency_key for member in sent.members)
    assert len(set(keys)) == 2
    assert all(key.startswith(f"{LOCAL_HF_HYBRID_BATCH_POLICY_VERSION}:") for key in keys)
    assert "shared-provider-key" not in keys
    assert keys != (first.provider_idempotency_key, second.provider_idempotency_key)


def test_loopback_batch_splits_at_four_and_by_max_token_compatibility(tmp_path: Path) -> None:
    fixture = _single_claim_fixture(tmp_path)
    requests = tuple(
        _request_variant(
            fixture.request,
            30 + ordinal,
            max_new_tokens=12 if ordinal < 4 else 16,
        )
        for ordinal in range(8)
    )
    transport = _HybridBatchTransport()
    adapter = _adapter(
        fixture,
        transport=transport,
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )

    results = asyncio.run(adapter.infer_batch(requests))

    assert all(isinstance(result, VisionInferenceSuccess) for result in results)
    assert len(transport.requests) == 2
    batches = tuple(
        LocalHfBatchEndpointRequest.model_validate_json(request.body, strict=True)
        for request in transport.requests
    )
    assert tuple(len(batch.members) for batch in batches) == (4, 4)
    assert tuple(batch.members[0].request.max_new_tokens for batch in batches) == (12, 16)


def test_loopback_batch_persists_each_member_before_independent_parse(tmp_path: Path) -> None:
    fixture = _single_claim_fixture(tmp_path)
    first = _request_variant(fixture.request, 50)
    second = _request_variant(fixture.request, 51)
    transport = _HybridBatchTransport(
        outputs={first.request_id: '["SUPPORTING"]', second.request_id: "not-json"}
    )
    raw_store = InMemoryRawProviderBytesStore()
    adapter = _adapter(fixture, transport=transport, evidence_ledger=raw_store)

    first_result, second_result = asyncio.run(adapter.infer_batch((first, second)))

    assert isinstance(first_result, VisionInferenceSuccess)
    assert isinstance(second_result, VisionInferenceFailure)
    assert second_result.status is InferenceStatus.INVALID_OUTPUT
    assert second_result.failure.code == "INVALID_JSON"
    assert second_result.raw_output_artifact_id is not None
    assert raw_store.get(second_result.raw_output_artifact_id).data == b"not-json"
    assert len(raw_store.list_records()) == 2


def test_loopback_batch_rejects_member_reordering_for_the_entire_native_chunk(
    tmp_path: Path,
) -> None:
    fixture = _single_claim_fixture(tmp_path)
    requests = tuple(_request_variant(fixture.request, 60 + ordinal) for ordinal in range(2))
    transport = _HybridBatchTransport(reverse_batch_members=True)
    adapter = _adapter(
        fixture,
        transport=transport,
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )

    results = asyncio.run(adapter.infer_batch(requests))

    assert all(isinstance(result, VisionInferenceFailure) for result in results)
    assert {
        result.failure.code for result in results if isinstance(result, VisionInferenceFailure)
    } == {"LOCAL_HF_BATCH_RESPONSE_BINDING_MISMATCH"}
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("transport", "expected_code"),
    (
        (
            _HybridBatchTransport(batch_error=local_hf.LocalHfTransportError("offline")),
            "LOCAL_HF_BATCH_TRANSPORT_TIMEOUT",
        ),
        (
            _HybridBatchTransport(batch_status_code=503),
            "LOCAL_HF_BATCH_HTTP_REJECTED",
        ),
        (
            _HybridBatchTransport(malformed_batch_body=b"{"),
            "LOCAL_HF_BATCH_RESPONSE_INVALID_JSON",
        ),
    ),
)
def test_loopback_batch_errors_are_explicit_and_never_fall_back_to_serial(
    tmp_path: Path,
    transport: _HybridBatchTransport,
    expected_code: str,
) -> None:
    fixture = _single_claim_fixture(tmp_path)
    requests = tuple(_request_variant(fixture.request, 70 + ordinal) for ordinal in range(2))
    adapter = _adapter(
        fixture,
        transport=transport,
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )

    results = asyncio.run(adapter.infer_batch(requests))

    assert all(isinstance(result, VisionInferenceFailure) for result in results)
    assert {
        result.failure.code for result in results if isinstance(result, VisionInferenceFailure)
    } == {expected_code}
    assert len(transport.requests) == 1
    assert transport.requests[0].url == LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL


def test_loopback_batch_input_bounds_are_strict(tmp_path: Path) -> None:
    fixture = _single_claim_fixture(tmp_path)
    adapter = _adapter(
        fixture,
        transport=_HybridBatchTransport(),
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )

    with pytest.raises(TypeError, match="tuple"):
        asyncio.run(adapter.infer_batch([fixture.request]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between one and eight"):
        asyncio.run(adapter.infer_batch(()))
    with pytest.raises(ValueError, match="between one and eight"):
        asyncio.run(adapter.infer_batch(tuple(fixture.request for _ordinal in range(9))))


def test_loopback_batch_cancellation_propagates_without_serial_retry(tmp_path: Path) -> None:
    fixture = _single_claim_fixture(tmp_path)
    requests = tuple(_request_variant(fixture.request, 90 + ordinal) for ordinal in range(2))
    transport = _HybridBatchTransport(batch_error=asyncio.CancelledError())
    adapter = _adapter(
        fixture,
        transport=transport,
        evidence_ledger=InMemoryRawProviderBytesStore(),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(adapter.infer_batch(requests))
    assert len(transport.requests) == 1
    assert transport.requests[0].url == LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL
