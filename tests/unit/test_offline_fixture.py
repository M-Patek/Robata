from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from robata.contracts.schema_registry import SchemaRegistry
from robata.inference.adapter import (
    JsonSchemaRef,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
)
from robata.inference.enrichment import PROVIDER_CLAIM_SCHEMA_ID, ProviderClaimPayload
from robata.inference.models import (
    ConcurrencyClass,
    InferenceStatus,
    InputMode,
    ModelCapabilities,
    VisionTask,
)
from robata.inference.offline_fixture import (
    InMemoryRawProviderBytesStore,
    OfflineFixtureVisionAdapter,
    ProviderResponseParseCode,
    RawProviderBytesStoreError,
    StrictProviderClaimParseError,
    StrictProviderClaimParser,
)

NOW = "2026-07-19T12:00:00Z"


def _uuid(number: int) -> str:
    return str(UUID(int=number))


def _digest(number: int) -> str:
    return f"{number:064x}"


def _provider_schema(registry: SchemaRegistry) -> JsonSchemaRef:
    ref = registry.resolve_version(PROVIDER_CLAIM_SCHEMA_ID, "1.0.0").ref
    return JsonSchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        schema_version="1.0",
        snapshot_id=_uuid(1),
        snapshot_digest=_digest(1),
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        supported_tasks=(VisionTask.FUSION_ADJUDICATION,),
        input_modes=(InputMode.MULTI_IMAGE,),
        accepted_media_types=("image/png",),
        max_images_per_request=12,
        max_pixels_per_image=640 * 480,
        max_payload_bytes=1_000_000,
        max_input_tokens=4_096,
        supports_json_schema=True,
        supports_provider_idempotency=True,
        concurrency_class=ConcurrencyClass.SERIAL,
        data_handling_policy_version="1.0",
        observed_at=NOW,
    )


def _request(output_schema: JsonSchemaRef, *, request_id: str = _uuid(2)) -> VisionInferenceRequest:
    return VisionInferenceRequest(
        schema_version="1.0",
        logical_invocation_id=_uuid(3),
        request_id=request_id,
        idempotency_key="logical-fixture-request",
        provider="offline-fixture",
        model_name="fixture-vision",
        model_version="1.0",
        package_inputs=(),
        package_input_set_sha256=_digest(4),
        task=VisionTask.FUSION_ADJUDICATION,
        prompt_version="1.0",
        prompt_artifact_id=_uuid(5),
        prompt_sha256=_digest(5),
        rendered_input_digest=_digest(6),
        output_schema=output_schema,
        capability_snapshot_id=_uuid(1),
        capability_snapshot_digest=_digest(1),
        model_policy_version="1.0",
        generation_config={"temperature": 0},
        provider_idempotency_key="provider-fixture-request",
        timeout_ms=1_000,
        metadata={},
    )


class _PersistenceCheckingParser(StrictProviderClaimParser):
    def __init__(
        self,
        registry: SchemaRegistry,
        raw_store: InMemoryRawProviderBytesStore,
    ) -> None:
        super().__init__(registry, parser_version="strict-fixture-1")
        self._raw_store = raw_store
        self.saw_persisted_bytes = False

    def decode_payload(
        self,
        *,
        data: bytes,
        provider_claim_schema: JsonSchemaRef,
    ) -> ProviderClaimPayload:
        records = self._raw_store.list_records()
        assert len(records) == 1
        assert records[0].data == data
        self.saw_persisted_bytes = True
        return super().decode_payload(
            data=data,
            provider_claim_schema=provider_claim_schema,
        )


def test_valid_response_is_persisted_before_strict_parsing() -> None:
    registry = SchemaRegistry()
    schema = _provider_schema(registry)
    raw_store = InMemoryRawProviderBytesStore()
    parser = _PersistenceCheckingParser(registry, raw_store)
    response = b'{"claims":[],"abstained":true}'
    adapter = OfflineFixtureVisionAdapter(
        capabilities=_capabilities(),
        raw_store=raw_store,
        parser=parser,
        response_factory=lambda _request: response,
    )

    assert adapter.network_call_count == 0
    assert asyncio.run(adapter.capabilities("fixture-vision", "1.0")) == _capabilities()
    result = asyncio.run(adapter.infer(_request(schema)))

    assert isinstance(result, VisionInferenceSuccess)
    assert result.status is InferenceStatus.SUCCEEDED
    assert result.normalized_output.payload == {"claims": [], "abstained": True}
    assert parser.saw_persisted_bytes
    stored = raw_store.get(result.raw_output_artifact_id)
    assert stored.data == response
    assert stored.request_id == _uuid(2)
    assert adapter.capability_calls == 1
    assert adapter.infer_calls == 1
    assert adapter.network_call_count == 0


def test_duplicate_json_key_returns_invalid_output_and_retains_exact_raw_bytes() -> None:
    registry = SchemaRegistry()
    schema = _provider_schema(registry)
    raw_store = InMemoryRawProviderBytesStore()
    response = b'{"claims":[],"abstained":true,"abstained":false}'
    adapter = OfflineFixtureVisionAdapter(
        capabilities=_capabilities(),
        raw_store=raw_store,
        parser=StrictProviderClaimParser(registry, parser_version="strict-fixture-1"),
        response_factory=lambda _request: response,
    )

    result = asyncio.run(adapter.infer(_request(schema)))

    assert isinstance(result, VisionInferenceFailure)
    assert result.status is InferenceStatus.INVALID_OUTPUT
    assert result.failure.code == ProviderResponseParseCode.DUPLICATE_JSON_KEY.value
    assert result.raw_output_artifact_id is not None
    assert raw_store.get(result.raw_output_artifact_id).data == response
    assert len(raw_store.list_records()) == 1
    assert adapter.network_call_count == 0


@pytest.mark.parametrize(
    ("data", "expected_code"),
    [
        (b'\xef\xbb\xbf{"claims":[],"abstained":true}', ProviderResponseParseCode.INVALID_UTF8),
        (b'{"claims":[],"abstained":"\xff"}', ProviderResponseParseCode.INVALID_UTF8),
    ],
)
def test_parser_rejects_bom_and_invalid_utf8(
    data: bytes,
    expected_code: ProviderResponseParseCode,
) -> None:
    registry = SchemaRegistry()
    parser = StrictProviderClaimParser(registry, parser_version="strict-fixture-1")

    with pytest.raises(StrictProviderClaimParseError) as raised:
        parser.decode_payload(
            data=data,
            provider_claim_schema=_provider_schema(registry),
        )

    assert raised.value.code is expected_code


def test_raw_store_rejects_different_bytes_for_the_same_request() -> None:
    raw_store = InMemoryRawProviderBytesStore()
    request_id = _uuid(20)
    first = raw_store.append(
        request_id=request_id,
        provider_request_id="offline:first",
        data=b'{"claims":[],"abstained":true}',
    )

    with pytest.raises(RawProviderBytesStoreError, match="cannot append different"):
        raw_store.append(
            request_id=request_id,
            provider_request_id="offline:second",
            data=b'{"claims":[],"abstained":false}',
        )

    assert raw_store.list_records() == (first,)
    assert raw_store.get(first.artifact_id).data == b'{"claims":[],"abstained":true}'
