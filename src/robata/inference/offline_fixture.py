"""Strict offline inference fixture with exact raw-byte preservation.

This module is intentionally transport-free.  It gives the canonical local
path a real adapter boundary while a production model adapter remains an
explicitly deferred dependency.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import NoReturn
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import ValidationError

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.schema_registry import (
    SchemaRef,
    SchemaRegistry,
    SchemaRegistryError,
    SchemaValidationError,
)
from robata.inference.adapter import (
    JsonSchemaRef,
    NormalizedOutputEnvelope,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.enrichment import (
    ParsedProviderClaimArtifact,
    ProviderClaimPayload,
    RawProviderResponseArtifact,
)
from robata.inference.models import (
    InferenceFailure,
    InferenceStatus,
    ModelCapabilities,
    Retryability,
    VisionTask,
)


class RawProviderBytesStoreError(RuntimeError):
    """Raw response storage conflicted with immutable append-only state."""


class RawProviderBytesNotFoundError(RawProviderBytesStoreError, LookupError):
    """An inference terminal referenced an absent raw response artifact."""


@dataclass(frozen=True, slots=True)
class StoredRawProviderBytes:
    """Exact bytes stored before parsing or adapter success is reported."""

    artifact_id: str
    request_id: str
    provider_request_id: str
    exact_bytes_sha256: str
    media_type: str
    data: bytes

    def __post_init__(self) -> None:
        for label, value in (
            ("artifact_id", self.artifact_id),
            ("request_id", self.request_id),
        ):
            try:
                parsed = UUID(value)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"{label} must be a UUID") from exc
            if str(parsed) != value:
                raise ValueError(f"{label} must use canonical lowercase UUID text")
        if not self.provider_request_id or not self.media_type:
            raise ValueError("provider_request_id and media_type must be nonempty")
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("stored raw provider data must be nonempty bytes")
        if self.exact_bytes_sha256 != exact_bytes_sha256(self.data):
            raise ValueError("stored raw provider byte digest is inconsistent")

    @property
    def byte_count(self) -> int:
        return len(self.data)


class InMemoryRawProviderBytesStore:
    """Thread-safe reference store with one immutable artifact per request."""

    def __init__(self) -> None:
        self._by_artifact: dict[str, StoredRawProviderBytes] = {}
        self._artifact_by_request: dict[str, str] = {}
        self._lock = RLock()

    def append(
        self,
        *,
        request_id: str,
        provider_request_id: str,
        data: bytes,
        media_type: str = "application/json",
    ) -> StoredRawProviderBytes:
        if not isinstance(data, bytes) or not data:
            raise ValueError("raw provider response must be nonempty bytes")
        digest = exact_bytes_sha256(data)
        artifact_id = _stable_uuid("raw-provider-response", request_id, digest)
        record = StoredRawProviderBytes(
            artifact_id=artifact_id,
            request_id=request_id,
            provider_request_id=provider_request_id,
            exact_bytes_sha256=digest,
            media_type=media_type,
            data=data,
        )
        with self._lock:
            existing_artifact_id = self._artifact_by_request.get(request_id)
            if existing_artifact_id is not None and existing_artifact_id != artifact_id:
                raise RawProviderBytesStoreError(
                    "one inference request cannot append different raw response bytes"
                )
            existing = self._by_artifact.get(artifact_id)
            if existing is not None and existing != record:
                raise RawProviderBytesStoreError(
                    "raw response artifact identity has conflicting content"
                )
            self._by_artifact[artifact_id] = record
            self._artifact_by_request[request_id] = artifact_id
            return existing or record

    def get(self, artifact_id: str) -> StoredRawProviderBytes:
        with self._lock:
            try:
                return self._by_artifact[artifact_id]
            except KeyError as exc:
                raise RawProviderBytesNotFoundError(artifact_id) from exc

    def list_records(self) -> tuple[StoredRawProviderBytes, ...]:
        with self._lock:
            return tuple(sorted(self._by_artifact.values(), key=lambda item: item.artifact_id))


class ProviderResponseParseCode(StrEnum):
    INVALID_UTF8 = "INVALID_UTF8"
    INVALID_JSON = "INVALID_JSON"
    DUPLICATE_JSON_KEY = "DUPLICATE_JSON_KEY"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    INVALID_CONTRACT = "INVALID_CONTRACT"
    NORMALIZATION_MISMATCH = "NORMALIZATION_MISMATCH"


class StrictProviderClaimParseError(ValueError):
    """Exact provider bytes cannot become the typed untrusted claim artifact."""

    def __init__(self, code: ProviderResponseParseCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class _DuplicateJsonKeyError(ValueError):
    pass


class StrictProviderClaimParser:
    """Reject ambiguous JSON and validate both pinned schema and strict models."""

    def __init__(self, schema_registry: SchemaRegistry, parser_version: str) -> None:
        if not isinstance(schema_registry, SchemaRegistry):
            raise TypeError("schema_registry must be a SchemaRegistry")
        if not isinstance(parser_version, str) or not parser_version:
            raise ValueError("parser_version must be nonempty")
        self._schema_registry = schema_registry
        self._parser_version = parser_version

    @property
    def parser_version(self) -> str:
        return self._parser_version

    @property
    def schema_registry(self) -> SchemaRegistry:
        return self._schema_registry

    def decode_payload(
        self,
        *,
        data: bytes,
        provider_claim_schema: JsonSchemaRef,
    ) -> ProviderClaimPayload:
        """Decode exact bytes without duplicate keys or implicit normalization."""

        if not isinstance(data, bytes) or not data:
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.INVALID_JSON,
                "provider response must be nonempty bytes",
            )
        if data.startswith(b"\xef\xbb\xbf"):
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.INVALID_UTF8,
                "UTF-8 BOM is forbidden in provider response bytes",
            )
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.INVALID_UTF8,
                "provider response is not strict UTF-8",
            ) from exc
        try:
            document = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_nonfinite,
            )
        except _DuplicateJsonKeyError as exc:
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.DUPLICATE_JSON_KEY,
                str(exc),
            ) from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.INVALID_JSON,
                "provider response is not strict JSON",
            ) from exc
        if not isinstance(document, dict) or any(not isinstance(key, str) for key in document):
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.INVALID_JSON,
                "provider response root must be a JSON object",
            )

        try:
            self._schema_registry.validate_pinned(_registry_ref(provider_claim_schema), document)
        except (SchemaRegistryError, SchemaValidationError) as exc:
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.INVALID_SCHEMA,
                str(exc),
            ) from exc
        try:
            payload = ProviderClaimPayload.model_validate_json(
                canonical_json_bytes(document), strict=True
            )
        except ValidationError as exc:
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.INVALID_CONTRACT,
                str(exc),
            ) from exc
        if canonical_json_bytes(document) != canonical_json_bytes(payload.model_dump(mode="json")):
            raise StrictProviderClaimParseError(
                ProviderResponseParseCode.NORMALIZATION_MISMATCH,
                "provider JSON would be changed by typed normalization",
            )
        return payload

    def parse_artifact(
        self,
        *,
        stored: StoredRawProviderBytes,
        inference_id: str,
        provider: str,
        model_name: str,
        model_version: str,
        provider_claim_schema: JsonSchemaRef,
        task: VisionTask,
        artifact_id: str,
        created_at: str,
    ) -> ParsedProviderClaimArtifact:
        payload = self.decode_payload(
            data=stored.data,
            provider_claim_schema=provider_claim_schema,
        )
        raw = RawProviderResponseArtifact.from_bytes(
            data=stored.data,
            artifact_id=stored.artifact_id,
            media_type=stored.media_type,
            provider_request_id=stored.provider_request_id,
            inference_id=inference_id,
            provider=provider,
            model_name=model_name,
            model_version=model_version,
            created_at=created_at,
        )
        return ParsedProviderClaimArtifact.create(
            artifact_id=artifact_id,
            raw_response=raw,
            provider_claim_schema=provider_claim_schema,
            task=task,
            payload=payload,
            parser_version=self._parser_version,
            created_at=created_at,
        )


type OfflineFixtureResponse = bytes | VisionInferenceFailure
type OfflineFixtureResponseFactory = Callable[[VisionInferenceRequest], OfflineFixtureResponse]


class OfflineFixtureVisionAdapter:
    """Transport-free adapter driven only by an injected deterministic fixture."""

    def __init__(
        self,
        *,
        capabilities: ModelCapabilities,
        raw_store: InMemoryRawProviderBytesStore,
        parser: StrictProviderClaimParser,
        response_factory: OfflineFixtureResponseFactory,
    ) -> None:
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities")
        if not isinstance(raw_store, InMemoryRawProviderBytesStore):
            raise TypeError("raw_store must be an InMemoryRawProviderBytesStore")
        if not isinstance(parser, StrictProviderClaimParser):
            raise TypeError("parser must be a StrictProviderClaimParser")
        if not callable(response_factory):
            raise TypeError("response_factory must be callable")
        self._capabilities = capabilities
        self._raw_store = raw_store
        self._parser = parser
        self._response_factory = response_factory
        self._capability_calls = 0
        self._infer_calls = 0
        self._counter_lock = RLock()

    @property
    def provider(self) -> str:
        return self._capabilities.provider

    @property
    def raw_store(self) -> InMemoryRawProviderBytesStore:
        return self._raw_store

    @property
    def parser(self) -> StrictProviderClaimParser:
        return self._parser

    @property
    def capability_calls(self) -> int:
        with self._counter_lock:
            return self._capability_calls

    @property
    def infer_calls(self) -> int:
        with self._counter_lock:
            return self._infer_calls

    @property
    def network_call_count(self) -> int:
        return 0

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        with self._counter_lock:
            self._capability_calls += 1
        if (
            model_name != self._capabilities.model_name
            or model_version != self._capabilities.model_version
        ):
            raise ValueError("offline capability request does not match the fixture model")
        return self._capabilities

    async def infer(
        self,
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        with self._counter_lock:
            self._infer_calls += 1
        response = self._response_factory(request)
        if isinstance(response, VisionInferenceFailure):
            return response
        if not isinstance(response, bytes):
            raise TypeError("offline response factory must return bytes or VisionInferenceFailure")

        provider_request_id = f"offline:{request.request_id}"
        stored = self._raw_store.append(
            request_id=request.request_id,
            provider_request_id=provider_request_id,
            data=response,
        )
        usage = _usage(request)
        try:
            payload = self._parser.decode_payload(
                data=stored.data,
                provider_claim_schema=request.output_schema,
            )
        except StrictProviderClaimParseError as exc:
            return VisionInferenceFailure(
                status=InferenceStatus.INVALID_OUTPUT,
                provider_request_id=provider_request_id,
                provider=request.provider,
                model_name=request.model_name,
                model_version=request.model_version,
                raw_output_artifact_id=stored.artifact_id,
                schema_valid=False,
                usage=usage,
                latency_ms=0,
                failure=InferenceFailure(
                    code=exc.code.value,
                    detail=exc.detail,
                    retryability=Retryability.PERMANENT,
                ),
            )

        return VisionInferenceSuccess(
            status=InferenceStatus.SUCCEEDED,
            provider_request_id=provider_request_id,
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
                payload=payload.model_dump(mode="json"),
            ),
            raw_output_artifact_id=stored.artifact_id,
            schema_valid=True,
            usage=usage,
            latency_ms=0,
        )


def _usage(request: VisionInferenceRequest) -> VisionUsage:
    plan = request.input_plan
    if plan is None:
        return VisionUsage(input_frames=0, input_images=0)
    items = plan.rendered_items
    input_tokens: int | None = None
    if request.input_plan_part_ordinal is not None:
        part = plan.call_plan.parts[request.input_plan_part_ordinal]
        items = items[part.start_item_ordinal : part.end_item_ordinal_exclusive]
        input_tokens = part.measured_input_tokens
    return VisionUsage(
        input_frames=len(items),
        input_images=sum(item.artifact.media_type.startswith("image/") for item in items),
        input_tokens=input_tokens,
        output_tokens=0,
        cost=0.0,
        currency="USD",
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _registry_ref(ref: JsonSchemaRef) -> SchemaRef:
    return SchemaRef(
        schema_id=ref.schema_id,
        version=ref.version,
        artifact_id=ref.artifact_id,
        sha256=ref.sha256,
    )


def _stable_uuid(namespace: str, *parts: object) -> str:
    material = ":".join(str(item) for item in parts)
    return str(uuid5(NAMESPACE_URL, f"robata:{namespace}:{material}"))


__all__ = [
    "InMemoryRawProviderBytesStore",
    "OfflineFixtureResponse",
    "OfflineFixtureResponseFactory",
    "OfflineFixtureVisionAdapter",
    "ProviderResponseParseCode",
    "RawProviderBytesNotFoundError",
    "RawProviderBytesStoreError",
    "StoredRawProviderBytes",
    "StrictProviderClaimParseError",
    "StrictProviderClaimParser",
]
