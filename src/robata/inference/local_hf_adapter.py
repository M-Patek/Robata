"""Loopback-only Qwen/Hugging Face vision adapter.

This module is deliberately a local development boundary. It talks only to the
pinned 127.0.0.1:8101 endpoint, reads only file:// rendered artifacts, and
emits the provider-neutral adapter envelope. A successful result is local
conformance evidence; this adapter never claims production eligibility or
canonical authority.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, NoReturn, Protocol, Self
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import unquote, urlsplit

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.adapter import (
    NormalizedOutputEnvelope,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.enrichment import (
    ProviderClaimInterval,
    ProviderClaimKind,
    ProviderClaimPayload,
    ProviderObservation,
    ProviderReferenceCatalog,
    ProviderTaskClaim,
)
from robata.inference.input_plan import RenderedProviderItem
from robata.inference.local_hf_endpoint import (
    LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION,
    LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION,
    LOCAL_HF_BATCH_INFER_PATH,
    LOCAL_HF_BATCH_POLICY_VERSION,
    LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER,
    LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION,
    LOCAL_HF_ENDPOINT_RESPONSE_VERSION,
    LocalHfBatchEndpointMemberRequest,
    LocalHfBatchEndpointMemberResponse,
    LocalHfBatchEndpointRequest,
    LocalHfBatchEndpointResponse,
    LocalHfEncodedImage,
    LocalHfEndpointRequest,
    LocalHfEndpointResponse,
    build_local_hf_batch_request_sha256,
)
from robata.inference.models import (
    InferenceFailure,
    InferenceStatus,
    ModelCapabilities,
    Retryability,
    VisionTask,
)
from robata.inference.offline_fixture import (
    RawProviderBytesStore,
    StrictProviderClaimParseError,
    StrictProviderClaimParser,
)

LOCAL_HF_LOOPBACK_BASE_URL: Final = "http://127.0.0.1:8101"
LOCAL_HF_LOOPBACK_INFER_PATH: Final = "/v1/local-vision/infer"
LOCAL_HF_LOOPBACK_ENDPOINT_URL: Final = (
    f"{LOCAL_HF_LOOPBACK_BASE_URL}{LOCAL_HF_LOOPBACK_INFER_PATH}"
)
LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL: Final = (
    f"{LOCAL_HF_LOOPBACK_BASE_URL}{LOCAL_HF_BATCH_INFER_PATH}"
)
LOCAL_HF_LOOPBACK_ADAPTER_VERSION: Final = "local-hf-loopback-adapter-v1"
LOCAL_HF_HYBRID_BATCH_POLICY_VERSION: Literal["local-qwen-task-claim-group-hybrid-batch-v1"] = (
    "local-qwen-task-claim-group-hybrid-batch-v1"
)
LOCAL_HF_HYBRID_BATCH_MAX_SIZE: Literal[4] = 4
LOCAL_HF_HYBRID_BATCH_INPUT_MAX_SIZE: Literal[8] = 8
LOCAL_HF_LOOPBACK_TOKEN_POLICY_VERSION: Final = "provider-token-v1"
LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION: Final = "single-observation-broadcast-v1"
LOCAL_HF_COMPACT_CAMERA_GROUP_POLICY_VERSION: Final = "package-camera-unanimous-collapse-v1"
LOCAL_HF_COMPACT_BARE_LABEL_POLICY_VERSION: Final = "exact-allowed-label-v1"
LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION: Final = "single-scalar-observation-v1"
LOCAL_HF_COMPACT_SINGLE_DECISION_POLICY_VERSION: Final = "exact-one-decision-v1"
LOCAL_HF_COMPACT_TOKEN_ORDER_POLICY_VERSION: Final = "lexicographic-reference-token-v1"
LOCAL_HF_DENSE_COORDINATE_REDUCTION_POLICY_VERSION: Final = "severity-worst-then-earliest-v1"
LOCAL_HF_COMPACT_PROMPT_PROTOCOL: Final = "robata-provider-claim-v1"
LOCAL_HF_COMPACT_RESPONSE_INSTRUCTION: Final = (
    "Return ONLY one strict JSON array of strings; no markdown, prose, keys, "
    "timestamps, IDs, or evidence tokens. The adapter binds the array to the immutable "
    "evidence catalog."
)
LOCAL_HF_COMPACT_NORMALIZATION_CONTRACT_VERSION: Final = (
    "local-hf-compact-normalization-contract-v1"
)
LOCAL_HF_LOOPBACK_USER_AGENT: Final = "robata-local-qwen-loopback/1"
LOCAL_HF_LOOPBACK_MAX_IMAGES: Final = 6
LOCAL_HF_LOOPBACK_MAX_NEW_TOKENS: Final = 512
LOCAL_HF_LOOPBACK_DEFAULT_NEW_TOKENS: Final = 64
LOCAL_HF_LOOPBACK_DEFAULT_RESPONSE_BYTES: Final = 1_000_000

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
FailureStatus = Literal[
    InferenceStatus.FAILED,
    InferenceStatus.TIMEOUT,
    InferenceStatus.CANCELLED,
    InferenceStatus.INVALID_OUTPUT,
]


class LocalHfLoopbackAdapterConfig(StrictModel):
    """Pinned, explicitly non-authoritative configuration for the local adapter."""

    provider: NonEmptyString = "local-huggingface"
    adapter_version: SchemaVersion = LOCAL_HF_LOOPBACK_ADAPTER_VERSION
    endpoint_url: NonEmptyString = LOCAL_HF_LOOPBACK_BASE_URL
    token_policy_version: SchemaVersion = LOCAL_HF_LOOPBACK_TOKEN_POLICY_VERSION
    default_max_new_tokens: Annotated[
        int,
        Field(strict=True, ge=1, le=LOCAL_HF_LOOPBACK_MAX_NEW_TOKENS),
    ] = LOCAL_HF_LOOPBACK_DEFAULT_NEW_TOKENS
    request_timeout_cap_ms: PositiveInt = 300_000
    max_response_bytes: PositiveInt = LOCAL_HF_LOOPBACK_DEFAULT_RESPONSE_BYTES
    production_eligible: Literal[False] = False
    canonical_authority: Literal[False] = False

    @model_validator(mode="after")
    def validate_loopback_endpoint(self) -> Self:
        _normalize_loopback_base_url(self.endpoint_url)
        return self


class LocalHfNativeBatchCapability(StrictModel):
    """Versioned proof consumed by canonical composition before enabling Batch4."""

    supported: Literal[True] = True
    adapter_policy_version: Literal["local-qwen-task-claim-group-hybrid-batch-v1"] = (
        LOCAL_HF_HYBRID_BATCH_POLICY_VERSION
    )
    endpoint_request_version: Literal["local-hf-vision-batch-request-v1"] = (
        LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION
    )
    endpoint_response_version: Literal["local-hf-vision-batch-response-v1"] = (
        LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION
    )
    endpoint_policy_version: Literal["local-hf-native-batch-policy-v1"] = (
        LOCAL_HF_BATCH_POLICY_VERSION
    )
    max_batch_size: Literal[4] = LOCAL_HF_HYBRID_BATCH_MAX_SIZE
    max_dispatch_size: Literal[8] = LOCAL_HF_HYBRID_BATCH_INPUT_MAX_SIZE
    native_admission: Literal["EXACTLY_ONE_CLAIM_GROUP"] = "EXACTLY_ONE_CLAIM_GROUP"
    multi_claim_route: Literal["SERIAL_V1"] = "SERIAL_V1"
    hidden_error_fallback: Literal[False] = False


@dataclass(frozen=True, slots=True)
class LocalHfHttpRequest:
    """One exact HTTP dispatch handed to a local transport."""

    url: str
    body: bytes
    timeout_seconds: float
    max_response_bytes: int
    idempotency_key: str

    def __post_init__(self) -> None:
        if self.url not in {
            LOCAL_HF_LOOPBACK_ENDPOINT_URL,
            LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
        }:
            raise ValueError("local HF request URL is not a pinned loopback infer path")
        if not isinstance(self.body, bytes) or not self.body:
            raise ValueError("local HF request body must be nonempty bytes")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("local HF request timeout must be finite and positive")
        if isinstance(self.max_response_bytes, bool) or self.max_response_bytes <= 0:
            raise ValueError("local HF response byte limit must be positive")
        if not isinstance(self.idempotency_key, str) or not self.idempotency_key:
            raise ValueError("local HF request idempotency key must be nonempty")


@dataclass(frozen=True, slots=True)
class LocalHfHttpResponse:
    """Transport-neutral local HTTP response."""

    status_code: int
    body: bytes

    def __post_init__(self) -> None:
        if isinstance(self.status_code, bool) or not isinstance(self.status_code, int):
            raise ValueError("local HF response status_code must be an integer")
        if self.status_code < 100 or self.status_code > 599:
            raise ValueError("local HF response status_code is outside HTTP range")
        if not isinstance(self.body, bytes):
            raise ValueError("local HF response body must be bytes")


class LocalHfTransportError(OSError):
    """The local endpoint could not be reached or read."""


class LocalHfTransport(Protocol):
    """Async transport port used by LocalHfLoopbackVisionAdapter."""

    async def post(self, request: LocalHfHttpRequest) -> LocalHfHttpResponse: ...


@dataclass(frozen=True, slots=True)
class RecordedLocalHfExchange:
    """A deterministic request-body-bound local transport exchange."""

    request_body_sha256: Sha256Digest
    response: LocalHfHttpResponse


class RecordedLocalHfTransport:
    """In-memory transport for focused tests and local replay."""

    def __init__(self, exchanges: tuple[RecordedLocalHfExchange, ...]) -> None:
        if not isinstance(exchanges, tuple):
            raise TypeError("exchanges must be a tuple")
        self._responses: dict[str, list[LocalHfHttpResponse]] = defaultdict(list)
        for exchange in exchanges:
            if not isinstance(exchange, RecordedLocalHfExchange):
                raise TypeError("exchanges must contain RecordedLocalHfExchange values")
            self._responses[exchange.request_body_sha256].append(exchange.response)
        self._requests: list[LocalHfHttpRequest] = []

    @property
    def requests(self) -> tuple[LocalHfHttpRequest, ...]:
        return tuple(self._requests)

    @property
    def request_count(self) -> int:
        return len(self._requests)

    async def post(self, request: LocalHfHttpRequest) -> LocalHfHttpResponse:
        if not isinstance(request, LocalHfHttpRequest):
            raise TypeError("request must be LocalHfHttpRequest")
        self._requests.append(request)
        digest = exact_bytes_sha256(request.body)
        responses = self._responses.get(digest)
        if not responses:
            raise LocalHfTransportError(
                f"no recorded local HF response for request body digest {digest}"
            )
        return responses.pop(0)


class StdlibLocalHfTransport:
    """Small standard-library transport for the pinned loopback service."""

    async def post(self, request: LocalHfHttpRequest) -> LocalHfHttpResponse:
        return await asyncio.to_thread(self._post_sync, request)

    @staticmethod
    def _post_sync(request: LocalHfHttpRequest) -> LocalHfHttpResponse:
        http_request = urllib_request.Request(
            request.url,
            data=request.body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": LOCAL_HF_LOOPBACK_USER_AGENT,
                LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER: request.idempotency_key,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=request.timeout_seconds) as response:
                return LocalHfHttpResponse(
                    status_code=int(response.status),
                    body=response.read(request.max_response_bytes + 1),
                )
        except urllib_error.HTTPError as error:
            try:
                body = error.read(request.max_response_bytes + 1)
            except OSError as read_error:
                raise LocalHfTransportError("local HF error body could not be read") from read_error
            return LocalHfHttpResponse(status_code=int(error.code), body=body)
        except (TimeoutError, urllib_error.URLError, OSError) as error:
            raise LocalHfTransportError("local HF endpoint request failed") from error


class _LocalHfAdapterError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _LocalHfEnvelopeError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _PreparedLocalHfBatchMember:
    index: int
    request: VisionInferenceRequest
    started: float
    selected_items: tuple[RenderedProviderItem, ...]
    endpoint_request: LocalHfEndpointRequest
    endpoint_request_body: bytes
    claim_group_count: int
    member_idempotency_key: str


class LocalHfLoopbackVisionAdapter:
    """Local Qwen adapter with fail-closed evidence and parsing boundaries.

    The ordinary local composition supplies its SQLiteInferenceEvidenceLedger as
    evidence_ledger. The generic raw-byte port remains accepted only so focused
    tests can use the existing deterministic in-memory implementation. This
    adapter does not append parsed artifacts, selections, or production decisions.
    """

    def __init__(
        self,
        *,
        capabilities: ModelCapabilities,
        parser: StrictProviderClaimParser,
        evidence_ledger: RawProviderBytesStore | None = None,
        raw_store: RawProviderBytesStore | None = None,
        config: LocalHfLoopbackAdapterConfig | None = None,
        transport: LocalHfTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities")
        if not isinstance(parser, StrictProviderClaimParser):
            raise TypeError("parser must be StrictProviderClaimParser")
        if evidence_ledger is not None and raw_store is not None:
            raise ValueError("supply evidence_ledger or raw_store, not both")
        resolved_store = evidence_ledger if evidence_ledger is not None else raw_store
        if not isinstance(resolved_store, RawProviderBytesStore):
            raise TypeError("evidence_ledger must implement RawProviderBytesStore")
        if config is None:
            config = LocalHfLoopbackAdapterConfig(provider=capabilities.provider)
        if not isinstance(config, LocalHfLoopbackAdapterConfig):
            raise TypeError("config must be LocalHfLoopbackAdapterConfig")
        resolved_transport = transport or StdlibLocalHfTransport()
        if not callable(getattr(resolved_transport, "post", None)):
            raise TypeError("transport must implement LocalHfTransport")
        if not callable(monotonic):
            raise TypeError("monotonic must be callable")
        if config.provider != capabilities.provider:
            raise ValueError("config provider must match capabilities provider")
        self._config = config
        self._capabilities = capabilities
        self._parser = parser
        self._evidence_ledger = resolved_store
        self._transport = resolved_transport
        self._monotonic = monotonic

    @property
    def provider(self) -> str:
        return self._capabilities.provider

    @property
    def capabilities_snapshot(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def config(self) -> LocalHfLoopbackAdapterConfig:
        return self._config

    @property
    def endpoint_url(self) -> str:
        return LOCAL_HF_LOOPBACK_ENDPOINT_URL

    @property
    def batch_endpoint_url(self) -> str:
        return LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL

    @property
    def native_batch_capability(self) -> LocalHfNativeBatchCapability:
        """Return the immutable local Batch4 admission proof."""

        return LocalHfNativeBatchCapability()

    @property
    def native_batch_policy_version(
        self,
    ) -> Literal["local-qwen-task-claim-group-hybrid-batch-v1"]:
        return LOCAL_HF_HYBRID_BATCH_POLICY_VERSION

    @property
    def native_batch_max_size(self) -> Literal[4]:
        return LOCAL_HF_HYBRID_BATCH_MAX_SIZE

    @property
    def parser(self) -> StrictProviderClaimParser:
        return self._parser

    @property
    def evidence_ledger(self) -> RawProviderBytesStore:
        return self._evidence_ledger

    @property
    def raw_store(self) -> RawProviderBytesStore:
        """Compatibility alias for existing adapter port terminology."""

        return self._evidence_ledger

    @property
    def supports_normalized_output_lineage(self) -> Literal[True]:
        """Allow local compact-wire recovery from the normalized envelope only."""

        return True

    @property
    def dense_coordinate_reduction_policy_version(
        self,
    ) -> Literal["severity-worst-then-earliest-v1"]:
        """Opt into deterministic reduction for partitioned QA_DENSE call parts."""

        return LOCAL_HF_DENSE_COORDINATE_REDUCTION_POLICY_VERSION

    @property
    def production_eligible(self) -> Literal[False]:
        return False

    @property
    def canonical_authority(self) -> Literal[False]:
        return False

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        if (
            model_name != self._capabilities.model_name
            or model_version != self._capabilities.model_version
        ):
            raise ValueError("local HF capability request does not match the pinned model")
        return self._capabilities

    def _prepare_endpoint_request(
        self,
        request: VisionInferenceRequest,
    ) -> tuple[tuple[RenderedProviderItem, ...], LocalHfEndpointRequest, bytes]:
        """Build the exact existing serial request bytes after fail-closed validation."""

        selected_items = self._validate_request(request)
        image_payloads = self._read_selected_files(selected_items)
        prompt = self._canonical_claim_prompt(request, selected_items)
        endpoint_request = LocalHfEndpointRequest(
            request_id=request.request_id,
            images=[
                LocalHfEncodedImage(
                    camera_id=item.camera_id.value,
                    sha256=item.artifact.sha256,
                    base64_data=base64.b64encode(payload).decode("ascii"),
                )
                for item, payload in zip(selected_items, image_payloads, strict=True)
            ],
            prompt=prompt,
            max_new_tokens=self._max_new_tokens(request),
        )
        body = canonical_json_bytes(endpoint_request.model_dump(mode="json"))
        max_payload_bytes = self._capabilities.max_payload_bytes
        if max_payload_bytes is not None and len(body) > max_payload_bytes:
            raise _LocalHfAdapterError(
                "LOCAL_HF_REQUEST_TOO_LARGE",
                "local HF request exceeds the pinned payload byte limit",
            )
        return selected_items, endpoint_request, body

    async def infer(
        self,
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        """Execute one bounded loopback call and strictly parse its claim payload."""

        if not isinstance(request, VisionInferenceRequest):
            raise TypeError("request must be VisionInferenceRequest")
        started = self._monotonic()
        try:
            selected_items, _endpoint_request, body = self._prepare_endpoint_request(request)
        except _LocalHfAdapterError as error:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code=error.code,
                detail=error.detail,
                retryability=Retryability.PERMANENT,
            )
        except (TypeError, ValueError, OSError):
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_REQUEST_REJECTED",
                detail="local HF request could not be prepared",
                retryability=Retryability.PERMANENT,
            )

        http_request = LocalHfHttpRequest(
            url=LOCAL_HF_LOOPBACK_ENDPOINT_URL,
            body=body,
            timeout_seconds=min(request.timeout_ms, self._config.request_timeout_cap_ms) / 1_000,
            max_response_bytes=self._config.max_response_bytes,
            idempotency_key=request.provider_idempotency_key,
        )
        try:
            async with asyncio.timeout(http_request.timeout_seconds):
                response = await self._transport.post(http_request)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, LocalHfTransportError):
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.TIMEOUT,
                code="LOCAL_HF_TRANSPORT_TIMEOUT",
                detail="local HF loopback transport timed out or was unavailable",
                retryability=Retryability.RETRYABLE,
            )
        except Exception:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_TRANSPORT_EXCEPTION",
                detail="local HF loopback transport raised an unexpected exception",
                retryability=Retryability.PERMANENT,
            )

        if not isinstance(response, LocalHfHttpResponse):
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_TRANSPORT_CONTRACT_VIOLATION",
                detail="local HF transport returned an unsupported response",
                retryability=Retryability.PERMANENT,
            )
        if response.status_code != 200:
            retryability = (
                Retryability.RATE_LIMITED
                if response.status_code == 429
                else Retryability.RETRYABLE
                if response.status_code == 408 or response.status_code >= 500
                else Retryability.PERMANENT
            )
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_HTTP_REJECTED",
                detail=f"local HF endpoint returned HTTP status {response.status_code}",
                retryability=retryability,
            )
        if len(response.body) > self._config.max_response_bytes:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_RESPONSE_TOO_LARGE",
                detail="local HF response exceeds the configured byte limit",
                retryability=Retryability.PERMANENT,
            )
        try:
            endpoint_response = _decode_endpoint_response(response.body)
        except _LocalHfEnvelopeError as error:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.INVALID_OUTPUT,
                code=error.code,
                detail=error.detail,
                retryability=Retryability.PERMANENT,
            )

        binding_error = self._binding_error(
            request,
            endpoint_response,
            selected_image_count=len(selected_items),
        )
        if binding_error is not None:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_RESPONSE_BINDING_MISMATCH",
                detail=binding_error,
                retryability=Retryability.PERMANENT,
                provider_request_id=endpoint_response.request_id,
                usage=self._usage(request, endpoint_response),
            )

        # Preserve exact untrusted provider text before any strict parser runs.
        raw_bytes = endpoint_response.output_text.encode("utf-8")
        try:
            stored = self._evidence_ledger.append(
                request_id=request.request_id,
                provider_request_id=endpoint_response.request_id,
                data=raw_bytes,
                media_type="application/json",
            )
        except Exception:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_EVIDENCE_PERSIST_FAILED",
                detail="local HF raw provider bytes could not be persisted",
                retryability=Retryability.PERMANENT,
                provider_request_id=endpoint_response.request_id,
                usage=self._usage(request, endpoint_response),
            )

        try:
            try:
                payload = self._parser.decode_payload(
                    data=stored.data,
                    provider_claim_schema=request.output_schema,
                )
            except StrictProviderClaimParseError as direct_error:
                # Real local VLMs use a compact semantic wire to stay within the
                # 512-token loopback cap. The adapter expands that wire into the
                # canonical provider-claim payload without inventing source bytes.
                try:
                    payload = self._decode_compact_payload(
                        data=stored.data,
                        request=request,
                        items=selected_items,
                    )
                except (TypeError, ValueError, ValidationError):
                    raise direct_error from None
        except StrictProviderClaimParseError as error:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.INVALID_OUTPUT,
                code=error.code.value,
                detail=error.detail,
                retryability=Retryability.PERMANENT,
                provider_request_id=endpoint_response.request_id,
                raw_output_artifact_id=stored.artifact_id,
                usage=self._usage(request, endpoint_response),
            )

        allowed_tokens = self._allowed_reference_tokens(request, selected_items)
        if any(
            token not in allowed_tokens
            for claim in payload.claims
            for token in claim.evidence_tokens
        ):
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_REFERENCE_TOKEN_OUT_OF_SCOPE",
                detail="provider claim cites an evidence token outside the selected call part",
                retryability=Retryability.PERMANENT,
                provider_request_id=endpoint_response.request_id,
                raw_output_artifact_id=stored.artifact_id,
                usage=self._usage(request, endpoint_response),
            )

        return VisionInferenceSuccess(
            status=InferenceStatus.SUCCEEDED,
            provider_request_id=endpoint_response.request_id,
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
            usage=self._usage(request, endpoint_response),
            latency_ms=_elapsed_ms(started, self._monotonic()),
        )

    async def infer_batch(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        """Execute the qualified Batch4 hybrid policy without hidden fallback.

        Exactly-one-claim-group requests are partitioned by their deterministic
        compatibility key and sent through the native batch endpoint in chunks of
        at most four. Every other request deliberately uses the unchanged serial
        ``infer`` route; a native-batch error is never retried through that route.
        """

        if not isinstance(requests, tuple):
            raise TypeError("requests must be a tuple")
        if any(not isinstance(request, VisionInferenceRequest) for request in requests):
            raise TypeError("requests must contain only VisionInferenceRequest values")
        if not 1 <= len(requests) <= LOCAL_HF_HYBRID_BATCH_INPUT_MAX_SIZE:
            raise ValueError("local HF hybrid dispatch must contain between one and eight requests")

        outcomes: list[VisionInferenceSuccess | VisionInferenceFailure | None] = [None] * len(
            requests
        )
        prepared_members: list[_PreparedLocalHfBatchMember] = []
        for index, request in enumerate(requests):
            started = self._monotonic()
            try:
                selected_items, endpoint_request, endpoint_request_body = (
                    self._prepare_endpoint_request(request)
                )
                claim_group_count = _claim_group_count(endpoint_request.prompt)
                member_idempotency_key = _batch_member_idempotency_key(
                    provider_idempotency_key=request.provider_idempotency_key,
                    endpoint_request_body=endpoint_request_body,
                )
            except _LocalHfAdapterError as error:
                outcomes[index] = self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.FAILED,
                    code=error.code,
                    detail=error.detail,
                    retryability=Retryability.PERMANENT,
                )
                continue
            except (TypeError, ValueError, OSError):
                outcomes[index] = self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.FAILED,
                    code="LOCAL_HF_REQUEST_REJECTED",
                    detail="local HF request could not be prepared",
                    retryability=Retryability.PERMANENT,
                )
                continue
            prepared_members.append(
                _PreparedLocalHfBatchMember(
                    index=index,
                    request=request,
                    started=started,
                    selected_items=selected_items,
                    endpoint_request=endpoint_request,
                    endpoint_request_body=endpoint_request_body,
                    claim_group_count=claim_group_count,
                    member_idempotency_key=member_idempotency_key,
                )
            )

        native_groups: dict[tuple[object, ...], list[_PreparedLocalHfBatchMember]] = {}
        serial_members: list[_PreparedLocalHfBatchMember] = []
        for member in prepared_members:
            if member.claim_group_count != 1:
                serial_members.append(member)
                continue
            key = self._native_batch_compatibility_key(member)
            native_groups.setdefault(key, []).append(member)

        for compatible_members in native_groups.values():
            for chunk in self._native_batch_chunks(tuple(compatible_members)):
                chunk_outcomes = await self._infer_native_batch_chunk(chunk)
                if len(chunk_outcomes) != len(chunk):
                    raise AssertionError("local HF native batch returned an invalid cardinality")
                for member, outcome in zip(chunk, chunk_outcomes, strict=True):
                    outcomes[member.index] = outcome

        for member in serial_members:
            # This is the explicit multi-/zero-claim quality guard, not a response
            # error fallback. Calling infer preserves the exact serial wire identity.
            outcomes[member.index] = await self.infer(member.request)

        if any(outcome is None for outcome in outcomes):
            raise AssertionError("local HF hybrid batch did not resolve every request")
        return tuple(outcome for outcome in outcomes if outcome is not None)

    @staticmethod
    def _native_batch_compatibility_key(
        member: _PreparedLocalHfBatchMember,
    ) -> tuple[object, ...]:
        request = member.request
        schema = request.output_schema
        return (
            request.task.value,
            member.claim_group_count,
            member.endpoint_request.max_new_tokens,
            request.model_name,
            request.model_version,
            request.prompt_version,
            request.prompt_sha256,
            schema.schema_id,
            schema.version,
            schema.sha256,
        )

    @staticmethod
    def _native_batch_chunks(
        members: tuple[_PreparedLocalHfBatchMember, ...],
    ) -> tuple[tuple[_PreparedLocalHfBatchMember, ...], ...]:
        """Chunk stably while keeping endpoint member idempotency keys unique."""

        chunks: list[tuple[_PreparedLocalHfBatchMember, ...]] = []
        current: list[_PreparedLocalHfBatchMember] = []
        current_keys: set[str] = set()
        for member in members:
            if (
                len(current) == LOCAL_HF_HYBRID_BATCH_MAX_SIZE
                or member.member_idempotency_key in current_keys
            ):
                chunks.append(tuple(current))
                current = []
                current_keys = set()
            current.append(member)
            current_keys.add(member.member_idempotency_key)
        if current:
            chunks.append(tuple(current))
        return tuple(chunks)

    async def _infer_native_batch_chunk(
        self,
        members: tuple[_PreparedLocalHfBatchMember, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        if not 1 <= len(members) <= LOCAL_HF_HYBRID_BATCH_MAX_SIZE:
            raise ValueError(
                "native local HF batch chunk must contain between one and four members"
            )
        try:
            endpoint_members = tuple(
                LocalHfBatchEndpointMemberRequest(
                    idempotency_key=member.member_idempotency_key,
                    request=member.endpoint_request,
                )
                for member in members
            )
            batch_request_sha256 = build_local_hf_batch_request_sha256(
                members=endpoint_members,
                batch_policy_version=LOCAL_HF_BATCH_POLICY_VERSION,
            )
            endpoint_request = LocalHfBatchEndpointRequest(
                batch_policy_version=LOCAL_HF_BATCH_POLICY_VERSION,
                batch_request_sha256=batch_request_sha256,
                members=list(endpoint_members),
            )
            body = canonical_json_bytes(endpoint_request.model_dump(mode="json"))
            http_request = LocalHfHttpRequest(
                url=LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
                body=body,
                timeout_seconds=(
                    min(
                        min(member.request.timeout_ms for member in members),
                        self._config.request_timeout_cap_ms,
                    )
                    / 1_000
                ),
                max_response_bytes=self._config.max_response_bytes,
                idempotency_key=(f"{LOCAL_HF_HYBRID_BATCH_POLICY_VERSION}:{batch_request_sha256}"),
            )
        except (TypeError, ValueError):
            return self._batch_failures(
                members,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_BATCH_REQUEST_REJECTED",
                detail="local HF native-batch request could not be prepared",
                retryability=Retryability.PERMANENT,
            )

        try:
            async with asyncio.timeout(http_request.timeout_seconds):
                response = await self._transport.post(http_request)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, LocalHfTransportError):
            return self._batch_failures(
                members,
                status=InferenceStatus.TIMEOUT,
                code="LOCAL_HF_BATCH_TRANSPORT_TIMEOUT",
                detail="local HF native-batch transport timed out or was unavailable",
                retryability=Retryability.RETRYABLE,
            )
        except Exception:
            return self._batch_failures(
                members,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_BATCH_TRANSPORT_EXCEPTION",
                detail="local HF native-batch transport raised an unexpected exception",
                retryability=Retryability.PERMANENT,
            )

        if not isinstance(response, LocalHfHttpResponse):
            return self._batch_failures(
                members,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_BATCH_TRANSPORT_CONTRACT_VIOLATION",
                detail="local HF native-batch transport returned an unsupported response",
                retryability=Retryability.PERMANENT,
            )
        if response.status_code != 200:
            retryability = (
                Retryability.RATE_LIMITED
                if response.status_code == 429
                else Retryability.RETRYABLE
                if response.status_code == 408 or response.status_code >= 500
                else Retryability.PERMANENT
            )
            return self._batch_failures(
                members,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_BATCH_HTTP_REJECTED",
                detail=f"local HF batch endpoint returned HTTP status {response.status_code}",
                retryability=retryability,
            )
        if len(response.body) > self._config.max_response_bytes:
            return self._batch_failures(
                members,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_BATCH_RESPONSE_TOO_LARGE",
                detail="local HF native-batch response exceeds the configured byte limit",
                retryability=Retryability.PERMANENT,
            )
        try:
            endpoint_response = _decode_batch_endpoint_response(response.body)
        except _LocalHfEnvelopeError as error:
            return self._batch_failures(
                members,
                status=InferenceStatus.INVALID_OUTPUT,
                code=error.code,
                detail=error.detail,
                retryability=Retryability.PERMANENT,
            )

        binding_error = self._batch_binding_error(
            members=members,
            request=endpoint_request,
            response=endpoint_response,
        )
        if binding_error is not None:
            return self._batch_failures(
                members,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_BATCH_RESPONSE_BINDING_MISMATCH",
                detail=binding_error,
                retryability=Retryability.PERMANENT,
            )
        return tuple(
            self._batch_member_outcome(member=member, response=member_response)
            for member, member_response in zip(
                members,
                endpoint_response.members,
                strict=True,
            )
        )

    def _batch_failures(
        self,
        members: tuple[_PreparedLocalHfBatchMember, ...],
        *,
        status: FailureStatus,
        code: str,
        detail: str,
        retryability: Retryability,
    ) -> tuple[VisionInferenceFailure, ...]:
        return tuple(
            self._failure(
                member.request,
                started=member.started,
                status=status,
                code=code,
                detail=detail,
                retryability=retryability,
            )
            for member in members
        )

    def _batch_binding_error(
        self,
        *,
        members: tuple[_PreparedLocalHfBatchMember, ...],
        request: LocalHfBatchEndpointRequest,
        response: LocalHfBatchEndpointResponse,
    ) -> str | None:
        if response.contract_version != LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION:
            return "local HF batch response contract version is not pinned"
        if response.batch_policy_version != LOCAL_HF_BATCH_POLICY_VERSION:
            return "local HF batch response policy version is not pinned"
        if response.batch_request_sha256 != request.batch_request_sha256:
            return "local HF batch response identity differs from the request"
        if response.model_identifier != self._capabilities.model_name:
            return "local HF batch response model identifier differs from the request"
        if response.model_version != self._capabilities.model_version:
            return "local HF batch response model version differs from the request"
        if len(response.members) != len(members):
            return "local HF batch response member count differs from the request"
        if response.generated_member_count + response.replay_member_count != len(members):
            return "local HF batch response disposition counts differ from the request"
        for prepared, resolved in zip(members, response.members, strict=True):
            if resolved.idempotency_key != prepared.member_idempotency_key:
                return "local HF batch response member idempotency order differs from the request"
            if resolved.request_id != prepared.request.request_id:
                return "local HF batch response member request_id order differs from the request"
            expected_images = len(prepared.selected_items)
            if resolved.input_image_count != expected_images:
                return "local HF batch response member image count differs from the request"
            if len(resolved.rendered_image_sizes) != expected_images:
                return "local HF batch response member image-size count differs from the request"
        return None

    def _batch_member_outcome(
        self,
        *,
        member: _PreparedLocalHfBatchMember,
        response: LocalHfBatchEndpointMemberResponse,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        request = member.request
        usage = VisionUsage(
            input_frames=len(member.selected_items),
            input_images=len(member.selected_items),
            input_tokens=response.prompt_tokens,
            output_tokens=response.output_tokens,
            cost=0.0,
            currency="USD",
        )
        raw_bytes = response.output_text.encode("utf-8")
        try:
            stored = self._evidence_ledger.append(
                request_id=request.request_id,
                provider_request_id=response.request_id,
                data=raw_bytes,
                media_type="application/json",
            )
        except Exception:
            return self._failure(
                request,
                started=member.started,
                status=InferenceStatus.FAILED,
                code="LOCAL_HF_EVIDENCE_PERSIST_FAILED",
                detail="local HF raw provider bytes could not be persisted",
                retryability=Retryability.PERMANENT,
                provider_request_id=response.request_id,
                usage=usage,
            )

        try:
            try:
                payload = self._parser.decode_payload(
                    data=stored.data,
                    provider_claim_schema=request.output_schema,
                )
            except StrictProviderClaimParseError as direct_error:
                try:
                    payload = self._decode_compact_payload(
                        data=stored.data,
                        request=request,
                        items=member.selected_items,
                    )
                except (TypeError, ValueError, ValidationError):
                    raise direct_error from None
        except StrictProviderClaimParseError as error:
            return self._failure(
                request,
                started=member.started,
                status=InferenceStatus.INVALID_OUTPUT,
                code=error.code.value,
                detail=error.detail,
                retryability=Retryability.PERMANENT,
                provider_request_id=response.request_id,
                raw_output_artifact_id=stored.artifact_id,
                usage=usage,
            )

        allowed_tokens = self._allowed_reference_tokens(request, member.selected_items)
        if any(
            token not in allowed_tokens
            for claim in payload.claims
            for token in claim.evidence_tokens
        ):
            return self._failure(
                request,
                started=member.started,
                status=InferenceStatus.INVALID_OUTPUT,
                code="LOCAL_HF_REFERENCE_TOKEN_OUT_OF_SCOPE",
                detail="provider claim cites an evidence token outside the selected call part",
                retryability=Retryability.PERMANENT,
                provider_request_id=response.request_id,
                raw_output_artifact_id=stored.artifact_id,
                usage=usage,
            )

        return VisionInferenceSuccess(
            status=InferenceStatus.SUCCEEDED,
            provider_request_id=response.request_id,
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
            latency_ms=_elapsed_ms(member.started, self._monotonic()),
        )

    def _validate_request(
        self,
        request: VisionInferenceRequest,
    ) -> tuple[RenderedProviderItem, ...]:
        capabilities = self._capabilities
        if (
            request.provider != self.provider
            or request.model_name != capabilities.model_name
            or request.model_version != capabilities.model_version
            or request.capability_snapshot_id != capabilities.snapshot_id
            or request.capability_snapshot_digest != capabilities.snapshot_digest
        ):
            raise _LocalHfAdapterError(
                "LOCAL_HF_REQUEST_TARGET_MISMATCH",
                "local HF request target does not match pinned capabilities",
            )
        if request.task not in capabilities.supported_tasks:
            raise _LocalHfAdapterError(
                "LOCAL_HF_TASK_UNSUPPORTED",
                "local HF request task is outside pinned capabilities",
            )
        if not capabilities.supports_json_schema or not capabilities.supports_provider_idempotency:
            raise _LocalHfAdapterError(
                "LOCAL_HF_CAPABILITIES_INCOMPLETE",
                "local HF capabilities must support schema validation and idempotency",
            )
        plan = request.input_plan
        if plan is None or request.input_plan_semantic_sha256 is None:
            raise _LocalHfAdapterError(
                "LOCAL_HF_INPUT_PLAN_REQUIRED",
                "local HF dispatch requires an immutable input plan",
            )
        if plan.target.adapter_version != self._config.adapter_version:
            raise _LocalHfAdapterError(
                "LOCAL_HF_ADAPTER_VERSION_MISMATCH",
                "input plan adapter version does not match local HF configuration",
            )
        selected_items = _selected_items(request)
        if not selected_items:
            raise _LocalHfAdapterError(
                "LOCAL_HF_INPUT_ITEMS_REQUIRED",
                "local HF dispatch requires at least one rendered provider item",
            )
        if len(selected_items) > LOCAL_HF_LOOPBACK_MAX_IMAGES:
            raise _LocalHfAdapterError(
                "LOCAL_HF_TOO_MANY_IMAGES",
                "local HF loopback requests are capped at six images",
            )
        max_images = capabilities.max_images_per_request
        if max_images is not None and len(selected_items) > max_images:
            raise _LocalHfAdapterError(
                "LOCAL_HF_IMAGE_LIMIT_EXCEEDED",
                "selected images exceed the pinned capability limit",
            )
        part_ordinal = request.input_plan_part_ordinal
        if part_ordinal is None:
            if len(plan.call_plan.parts) != 1:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_CALL_PART_REQUIRED",
                    "split input plans require an explicit call part ordinal",
                )
            part = plan.call_plan.parts[0]
        else:
            if part_ordinal < 0 or part_ordinal >= len(plan.call_plan.parts):
                raise _LocalHfAdapterError(
                    "LOCAL_HF_CALL_PART_INVALID",
                    "input-plan call part ordinal is out of range",
                )
            part = plan.call_plan.parts[part_ordinal]
        max_input_tokens = capabilities.max_input_tokens
        if max_input_tokens is not None and part.measured_input_tokens > max_input_tokens:
            raise _LocalHfAdapterError(
                "LOCAL_HF_INPUT_TOKEN_LIMIT_EXCEEDED",
                "selected call part exceeds the pinned input-token limit",
            )
        for item in selected_items:
            artifact = item.artifact
            if artifact.media_type != "image/png":
                raise _LocalHfAdapterError(
                    "LOCAL_HF_MEDIA_UNSUPPORTED",
                    "local HF loopback accepts only image/png rendered artifacts",
                )
            if artifact.media_type not in capabilities.accepted_media_types:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_MEDIA_OUTSIDE_CAPABILITIES",
                    "rendered artifact media type is outside pinned capabilities",
                )
            max_pixels = capabilities.max_pixels_per_image
            if max_pixels is not None and artifact.width * artifact.height > max_pixels:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_PIXEL_LIMIT_EXCEEDED",
                    "rendered artifact exceeds the pinned per-image pixel limit",
                )
        return selected_items

    @staticmethod
    def _read_selected_files(items: tuple[RenderedProviderItem, ...]) -> tuple[bytes, ...]:
        payloads: list[bytes] = []
        for item in items:
            artifact = item.artifact
            path = _file_uri_path(artifact.uri)
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_FILE_NOT_FOUND",
                    "selected local file URI does not resolve to a readable file",
                ) from error
            if not resolved.is_file():
                raise _LocalHfAdapterError(
                    "LOCAL_HF_FILE_NOT_FOUND",
                    "selected local file URI does not identify a regular file",
                )
            try:
                payload = resolved.read_bytes()
            except OSError as error:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_FILE_READ_FAILED",
                    "selected local file URI could not be read",
                ) from error
            if not payload:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_FILE_EMPTY",
                    "selected local image file is empty",
                )
            if len(payload) != artifact.byte_count:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_FILE_BYTE_COUNT_MISMATCH",
                    "selected local file byte count differs from the rendered artifact",
                )
            if exact_bytes_sha256(payload) != artifact.sha256:
                raise _LocalHfAdapterError(
                    "LOCAL_HF_FILE_DIGEST_MISMATCH",
                    "selected local file digest differs from the rendered artifact",
                )
            payloads.append(payload)
        return tuple(payloads)

    def _canonical_claim_prompt(
        self,
        request: VisionInferenceRequest,
        items: tuple[RenderedProviderItem, ...],
    ) -> str:
        plan = request.input_plan
        if plan is None:
            raise _LocalHfAdapterError(
                "LOCAL_HF_INPUT_PLAN_REQUIRED",
                "local HF prompt construction requires an immutable input plan",
            )
        try:
            all_entries = ProviderReferenceCatalog.derive_entries(
                request_catalog_sha256=plan.request_catalog.semantic_sha256,
                rendered_items=plan.rendered_items,
                token_policy_version=self._config.token_policy_version,
            )
            selected_ordinals = {item.provider_item_ordinal for item in items}
            entries = tuple(
                entry for entry in all_entries if entry.provider_item_ordinal in selected_ordinals
            )
            allowed = _compact_allowed_values(request.task)
            claim_groups = (
                ()
                if request.task in {VisionTask.EVENT_PROPOSAL, VisionTask.FUSION_ADJUDICATION}
                else _compact_camera_groups(items)
            )
            if request.task in {VisionTask.EVENT_PROPOSAL, VisionTask.FUSION_ADJUDICATION}:
                output_shape = "JSON array with one or more strings"
            else:
                output_shape = (
                    f"JSON array with exactly {len(claim_groups)} strings in claim-group order"
                )
            document: dict[str, object] = {
                "protocol": LOCAL_HF_COMPACT_PROMPT_PROTOCOL,
                "response_instruction": LOCAL_HF_COMPACT_RESPONSE_INSTRUCTION,
                "task": request.task.value,
                "compact_output_contract": {
                    "shape": output_shape,
                    "allowed_values": allowed,
                    "selected_item_count": len(items),
                    "claim_group_count": len(claim_groups),
                    "claim_group_policy": LOCAL_HF_COMPACT_CAMERA_GROUP_POLICY_VERSION,
                    "bare_label_recovery_policy": LOCAL_HF_COMPACT_BARE_LABEL_POLICY_VERSION,
                    "scalar_value_policy": LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION,
                    "single_decision_policy": LOCAL_HF_COMPACT_SINGLE_DECISION_POLICY_VERSION,
                    "reference_token_order_policy": LOCAL_HF_COMPACT_TOKEN_ORDER_POLICY_VERSION,
                    "normalization_contract_sha256": (
                        local_hf_compact_prompt_normalization_contract_sha256()
                    ),
                    "claim_group_order": [
                        {
                            "package_ordinal": group[0].package_ordinal,
                            "camera": group[0].camera_id.value,
                            "camera_ordinal": group[0].camera_ordinal,
                            "image_count": len(group),
                        }
                        for group in claim_groups
                    ],
                    "single_value_broadcast_policy": LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION,
                    "image_order": [
                        {
                            "camera": item.camera_id.value,
                            "camera_ordinal": item.camera_ordinal,
                            "timestamp_ns": item.aligned_timestamp_ns,
                        }
                        for item in items
                    ],
                },
                "prompt_artifact": {
                    "version": request.prompt_version,
                    "artifact_id": request.prompt_artifact_id,
                    "sha256": request.prompt_sha256,
                },
                "request_catalog_sha256": plan.request_catalog.semantic_sha256,
                "token_policy_version": self._config.token_policy_version,
                # Retain the catalog in the prompt for auditability even though
                # compact output deliberately does not repeat its tokens.
                "evidence_catalog": [entry.model_dump(mode="json") for entry in entries],
                "provider_response_schema": request.output_schema.model_dump(mode="json"),
            }
            dependency_sha256 = request.metadata.get("logical_dependency_sha256")
            if dependency_sha256 is not None:
                document["logical_dependency_sha256"] = dependency_sha256
            prompt_bytes = canonical_json_bytes(document)
        except (TypeError, ValueError) as error:
            raise _LocalHfAdapterError(
                "LOCAL_HF_PROMPT_SERIALIZATION_FAILED",
                "canonical claim prompt could not be serialized",
            ) from error
        if not prompt_bytes or len(prompt_bytes) > 4_096:
            raise _LocalHfAdapterError(
                "LOCAL_HF_PROMPT_TOO_LARGE",
                "canonical claim prompt exceeds the loopback endpoint limit",
            )
        return prompt_bytes.decode("utf-8")

    def _decode_compact_payload(
        self,
        *,
        data: bytes,
        request: VisionInferenceRequest,
        items: tuple[RenderedProviderItem, ...],
    ) -> ProviderClaimPayload:
        """Expand the bounded local semantic wire into canonical claims.

        The model is not trusted to author lineage, timestamps, or opaque tokens.
        Those fields are copied from the immutable input plan here; only the
        compact observation choices are accepted from the provider response.
        """
        try:
            document = _strict_json_document(data)
        except json.JSONDecodeError:
            # Qwen occasionally drops the JSON quotes around a single enum label.
            # Recover only one exact allow-listed token; raw provider bytes remain
            # immutable in CAS and all prose, punctuation, or unknown labels fail closed.
            document = [
                _strict_bare_compact_label(
                    data,
                    task=request.task,
                )
            ]
        values: list[str]
        if isinstance(document, list):
            values = _compact_string_values(document)
        elif isinstance(document, dict):
            if document.get("abstained") is True:
                if set(document) != {"abstained"}:
                    raise ValueError("compact abstention object must contain only abstained=true")
                return ProviderClaimPayload(claims=(), abstained=True)
            values = _compact_object_values(document)
        else:
            raise ValueError("compact response root must be an array or object")

        if request.task is VisionTask.EVENT_PROPOSAL:
            decision = _compact_single_decision(values=values, task=request.task)
            if decision == "ABSTAIN":
                return ProviderClaimPayload(claims=(), abstained=True)
            tokens = tuple(sorted(self._allowed_reference_tokens(request, items)))
            claim = ProviderTaskClaim(
                claim_ordinal=0,
                kind=ProviderClaimKind.EVENT_PROPOSAL,
                package_ordinal=None,
                camera_ordinal=None,
                interval=_aggregate_interval(items),
                label="qwen-event-proposal",
                observation=ProviderObservation.PROPOSED,
                evidence_tokens=tokens,
                model_reported_score=None,
                conflict_codes=(),
            )
            return ProviderClaimPayload(claims=(claim,), abstained=False)

        if request.task is VisionTask.FUSION_ADJUDICATION:
            decision = _compact_single_decision(values=values, task=request.task)
            if decision == "ABSTAIN":
                return ProviderClaimPayload(claims=(), abstained=True)
            fusion_tokens = tuple(sorted(self._allowed_reference_tokens(request, items)))
            if not fusion_tokens:
                raise ValueError("fusion compact response has no evidence token")
            fusion_claim = ProviderTaskClaim(
                claim_ordinal=0,
                kind=ProviderClaimKind.FUSION_HYPOTHESIS,
                package_ordinal=None,
                camera_ordinal=None,
                interval=_aggregate_interval(items),
                label="qwen-fusion-hypothesis",
                observation=(
                    ProviderObservation.CONFLICT
                    if decision == "CONFLICT"
                    else ProviderObservation.PROPOSED
                ),
                evidence_tokens=(fusion_tokens[0],),
                model_reported_score=None,
                conflict_codes=(),
            )
            return ProviderClaimPayload(claims=(fusion_claim,), abstained=False)

        groups = _compact_camera_groups(items)
        group_values = _compact_camera_group_values(values=values, items=items, groups=groups)
        kind, allowed = _compact_task_kind_and_values(request.task)
        entries = ProviderReferenceCatalog.derive_entries(
            request_catalog_sha256=request.input_plan.request_catalog.semantic_sha256
            if request.input_plan is not None
            else "0" * 64,
            rendered_items=request.input_plan.rendered_items
            if request.input_plan is not None
            else (),
            token_policy_version=self._config.token_policy_version,
        )
        token_by_ordinal = {
            entry.provider_item_ordinal: entry.correlation_token for entry in entries
        }
        item_claims: list[ProviderTaskClaim] = []
        for ordinal, (group, value) in enumerate(zip(groups, group_values, strict=True)):
            if value not in allowed:
                raise ValueError(f"unsupported compact observation {value!r}")
            observation = ProviderObservation(value)
            cite = (
                ()
                if observation in {ProviderObservation.MISSING, ProviderObservation.NO_BOUNDARY}
                else tuple(token_by_ordinal[item.provider_item_ordinal] for item in group)
            )
            interval = (
                None
                if observation in {ProviderObservation.MISSING, ProviderObservation.NO_BOUNDARY}
                else _aggregate_interval(group)
            )
            item_claims.append(
                ProviderTaskClaim(
                    claim_ordinal=ordinal,
                    kind=kind,
                    package_ordinal=group[0].package_ordinal,
                    camera_ordinal=group[0].camera_ordinal,
                    interval=interval,
                    label=f"qwen-{request.task.value.lower()}",
                    observation=observation,
                    evidence_tokens=cite,
                    model_reported_score=None,
                    conflict_codes=(),
                )
            )
        return ProviderClaimPayload(claims=tuple(item_claims), abstained=False)

    def _allowed_reference_tokens(
        self,
        request: VisionInferenceRequest,
        items: tuple[RenderedProviderItem, ...],
    ) -> frozenset[str]:
        plan = request.input_plan
        if plan is None:
            return frozenset()
        entries = ProviderReferenceCatalog.derive_entries(
            request_catalog_sha256=plan.request_catalog.semantic_sha256,
            rendered_items=plan.rendered_items,
            token_policy_version=self._config.token_policy_version,
        )
        selected_ordinals = {item.provider_item_ordinal for item in items}
        return frozenset(
            entry.correlation_token
            for entry in entries
            if entry.provider_item_ordinal in selected_ordinals
        )

    def _max_new_tokens(self, request: VisionInferenceRequest) -> int:
        configured: list[int] = []
        for key in ("max_new_tokens", "max_output_tokens"):
            value = request.generation_config.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise _LocalHfAdapterError(
                    "LOCAL_HF_GENERATION_CONFIG_INVALID",
                    f"generation_config.{key} must be an integer",
                )
            configured.append(value)
        if len(configured) == 2 and configured[0] != configured[1]:
            raise _LocalHfAdapterError(
                "LOCAL_HF_GENERATION_CONFIG_INVALID",
                "max_new_tokens and max_output_tokens disagree",
            )
        value = configured[0] if configured else self._config.default_max_new_tokens
        if value < 1 or value > LOCAL_HF_LOOPBACK_MAX_NEW_TOKENS:
            raise _LocalHfAdapterError(
                "LOCAL_HF_GENERATION_CONFIG_INVALID",
                "local HF max_new_tokens must be between one and 512",
            )
        return value

    def _binding_error(
        self,
        request: VisionInferenceRequest,
        response: LocalHfEndpointResponse,
        *,
        selected_image_count: int,
    ) -> str | None:
        if response.contract_version != LOCAL_HF_ENDPOINT_RESPONSE_VERSION:
            return "local HF response contract version is not pinned"
        if response.request_id != request.request_id:
            return "local HF response request_id differs from the request"
        if response.model_identifier != request.model_name:
            return "local HF response model identifier differs from the request"
        if response.model_version != request.model_version:
            return "local HF response model version differs from the request"
        if response.input_image_count != selected_image_count:
            return "local HF response image count differs from the request"
        if len(response.rendered_image_sizes) != selected_image_count:
            return "local HF response rendered image-size count differs from the request"
        return None

    def _usage(
        self,
        request: VisionInferenceRequest,
        response: LocalHfEndpointResponse | None,
    ) -> VisionUsage:
        try:
            items = _selected_items(request)
        except (TypeError, ValueError, IndexError):
            items = ()
        plan = request.input_plan
        input_tokens: int | None = None
        if plan is not None:
            try:
                part_ordinal = request.input_plan_part_ordinal
                if part_ordinal is None and len(plan.call_plan.parts) == 1:
                    part_ordinal = 0
                if part_ordinal is not None:
                    input_tokens = plan.call_plan.parts[part_ordinal].measured_input_tokens
            except (IndexError, TypeError, ValueError):
                input_tokens = None
        return VisionUsage(
            input_frames=len(items),
            input_images=len(items),
            input_tokens=(response.prompt_tokens if response is not None else input_tokens),
            output_tokens=(response.output_tokens if response is not None else None),
            cost=0.0,
            currency="USD",
        )

    def _failure(
        self,
        request: VisionInferenceRequest,
        *,
        started: float,
        status: FailureStatus,
        code: str,
        detail: str,
        retryability: Retryability,
        provider_request_id: str | None = None,
        raw_output_artifact_id: str | None = None,
        usage: VisionUsage | None = None,
    ) -> VisionInferenceFailure:
        return VisionInferenceFailure(
            status=status,
            provider_request_id=provider_request_id,
            provider=request.provider,
            model_name=request.model_name,
            model_version=request.model_version,
            raw_output_artifact_id=raw_output_artifact_id,
            schema_valid=False,
            usage=usage or self._usage(request, None),
            latency_ms=_elapsed_ms(started, self._monotonic()),
            failure=InferenceFailure(
                code=code,
                detail=detail,
                retryability=retryability,
            ),
        )


def _strict_json_document(data: bytes) -> object:
    if not isinstance(data, bytes) or not data:
        raise ValueError("compact response must be nonempty bytes")
    text = data.decode("utf-8", errors="strict")
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite)


def _compact_string_values(values: list[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("compact observations must be nonempty strings")
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("compact observations must be nonempty strings")
        result.append(normalized)
    return result


def _compact_object_values(document: dict[str, object]) -> list[str]:
    """Apply the scoped compact-object scalar conversion policy.

    This recovery path intentionally accepts only one recognized observation
    field. It is not a generic structural coercion for arbitrary provider JSON.
    """

    fields = tuple(
        field for field in ("observations", "decisions", "observation") if field in document
    )
    if not fields:
        raise ValueError("compact response lacks observations")
    if len(fields) != 1:
        raise ValueError("compact response must contain exactly one observation field")
    raw_values = document[fields[0]]
    if isinstance(raw_values, list):
        return _compact_string_values(raw_values)
    return _compact_string_values([raw_values])


def _compact_single_decision(*, values: list[str], task: VisionTask) -> str:
    """Validate the one exact decision accepted by Event/Fusion compact wires."""

    if len(values) != 1:
        raise ValueError("compact event/fusion response must contain exactly one decision")
    decision = values[0]
    if decision not in _compact_allowed_values(task):
        raise ValueError("compact event/fusion decision is not an allowed task label")
    return decision


def _strict_bare_compact_label(data: bytes, *, task: VisionTask) -> str:
    """Recover only one exact allow-listed local-model enum token."""

    if not isinstance(data, bytes) or not data:
        raise ValueError("bare compact response must be nonempty bytes")
    text = data.decode("utf-8", errors="strict")
    stripped = text.strip(" \t\r\n")
    if not stripped or any(character.isspace() for character in stripped):
        raise ValueError("bare compact response must contain one token")
    normalized = stripped.upper()
    if normalized not in _compact_allowed_values(task):
        raise ValueError("bare compact response is not an allowed task label")
    return normalized


_COMPACT_ALLOWED_VALUES_BY_TASK: Final[dict[VisionTask, tuple[str, ...]]] = {
    VisionTask.QA_COARSE: ("GOOD", "DEGRADED", "UNUSABLE", "UNKNOWN"),
    VisionTask.QA_DENSE: ("GOOD", "DEGRADED", "UNUSABLE", "UNKNOWN"),
    VisionTask.ACTION_EVIDENCE: (
        "SUPPORTING",
        "PARTIAL",
        "NO_EVENT",
        "OCCLUDED",
        "UNUSABLE",
        "MISSING",
    ),
    VisionTask.BOUNDARY_REFINEMENT: (
        "OBSERVED",
        "NO_BOUNDARY",
        "OCCLUDED",
        "UNUSABLE",
        "MISSING",
    ),
    VisionTask.EVENT_PROPOSAL: ("PROPOSED", "ABSTAIN"),
    VisionTask.FUSION_ADJUDICATION: ("PROPOSED", "CONFLICT", "ABSTAIN"),
}


def _compact_allowed_values(task: VisionTask) -> tuple[str, ...]:
    try:
        return _COMPACT_ALLOWED_VALUES_BY_TASK[task]
    except KeyError as error:
        raise ValueError(f"compact task is unsupported: {task.value}") from error


def local_hf_compact_prompt_normalization_contract() -> dict[str, object]:
    """Return the static compact-prompt semantics used by the local adapter.

    The returned document contains no request-specific values and is deliberately
    limited to template/normalization policy. It can therefore be hashed into a
    capability, model policy, or runtime identity without weakening the strict
    provider-claim parser for unrelated adapters.
    """

    return {
        "contract_version": LOCAL_HF_COMPACT_NORMALIZATION_CONTRACT_VERSION,
        "protocol": LOCAL_HF_COMPACT_PROMPT_PROTOCOL,
        "response_instruction": LOCAL_HF_COMPACT_RESPONSE_INSTRUCTION,
        "allowed_values_by_task": {
            task.value: list(_compact_allowed_values(task)) for task in VisionTask
        },
        "policies": {
            "bare_label_recovery": LOCAL_HF_COMPACT_BARE_LABEL_POLICY_VERSION,
            "scalar_value": LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION,
            "single_value_broadcast": LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION,
            "single_decision": LOCAL_HF_COMPACT_SINGLE_DECISION_POLICY_VERSION,
            "reference_token_order": LOCAL_HF_COMPACT_TOKEN_ORDER_POLICY_VERSION,
            "camera_group": LOCAL_HF_COMPACT_CAMERA_GROUP_POLICY_VERSION,
            "dense_coordinate_reduction": LOCAL_HF_DENSE_COORDINATE_REDUCTION_POLICY_VERSION,
            "endpoint_idempotency": LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION,
        },
    }


def local_hf_compact_prompt_normalization_contract_sha256() -> Sha256Digest:
    """Return the stable SHA-256 of the static compact-prompt contract."""

    return exact_bytes_sha256(
        canonical_json_bytes(local_hf_compact_prompt_normalization_contract())
    )


def _compact_camera_groups(
    items: tuple[RenderedProviderItem, ...],
) -> tuple[tuple[RenderedProviderItem, ...], ...]:
    """Group selected images by the canonical package/camera claim identity."""

    grouped: dict[tuple[int, int], list[RenderedProviderItem]] = {}
    for item in items:
        grouped.setdefault((item.package_ordinal, item.camera_ordinal), []).append(item)
    return tuple(tuple(group) for group in grouped.values())


def _compact_camera_group_values(
    *,
    values: list[str],
    items: tuple[RenderedProviderItem, ...],
    groups: tuple[tuple[RenderedProviderItem, ...], ...],
) -> tuple[str, ...]:
    """Bind compact values without inventing a within-camera reduction."""

    expected = len(groups)
    if len(values) == 1 and expected > 1:
        return tuple(values * expected)
    if len(values) == expected:
        return tuple(values)
    if len(values) != len(items):
        raise ValueError(f"compact response group count {len(values)} != {expected}")

    values_by_group: dict[tuple[int, int], list[str]] = {}
    for item, value in zip(items, values, strict=True):
        values_by_group.setdefault((item.package_ordinal, item.camera_ordinal), []).append(value)
    collapsed: list[str] = []
    for group in groups:
        key = (group[0].package_ordinal, group[0].camera_ordinal)
        group_values = values_by_group[key]
        if len(set(group_values)) != 1:
            raise ValueError("compact image observations disagree within one package/camera group")
        collapsed.append(group_values[0])
    return tuple(collapsed)


def _compact_task_kind_and_values(task: VisionTask) -> tuple[ProviderClaimKind, set[str]]:
    if task in {VisionTask.QA_COARSE, VisionTask.QA_DENSE}:
        return ProviderClaimKind.QA_OBSERVATION, set(_compact_allowed_values(task))
    if task is VisionTask.ACTION_EVIDENCE:
        return ProviderClaimKind.ACTION_OBSERVATION, set(_compact_allowed_values(task))
    if task is VisionTask.BOUNDARY_REFINEMENT:
        return ProviderClaimKind.BOUNDARY_OBSERVATION, set(_compact_allowed_values(task))
    raise ValueError(f"compact item task is unsupported: {task.value}")


def _item_interval(item: RenderedProviderItem) -> ProviderClaimInterval:
    return ProviderClaimInterval(
        start_ns=item.aligned_timestamp_ns,
        end_ns=item.aligned_timestamp_ns + 1,
    )


def _aggregate_interval(items: tuple[RenderedProviderItem, ...]) -> ProviderClaimInterval:
    if not items:
        raise ValueError("cannot aggregate an empty item set")
    start = min(item.aligned_timestamp_ns for item in items)
    end = max(item.aligned_timestamp_ns for item in items) + 1
    return ProviderClaimInterval(start_ns=start, end_ns=end)


def _selected_items(request: VisionInferenceRequest) -> tuple[RenderedProviderItem, ...]:
    plan = request.input_plan
    if plan is None:
        return ()
    if request.input_plan_part_ordinal is None:
        return tuple(plan.rendered_items)
    part = plan.call_plan.parts[request.input_plan_part_ordinal]
    return tuple(plan.rendered_items[part.start_item_ordinal : part.end_item_ordinal_exclusive])


def _claim_group_count(prompt: str) -> int:
    """Read the admission count from the exact canonical prompt contract."""

    try:
        document: object = json.loads(prompt)
    except json.JSONDecodeError as error:
        raise _LocalHfAdapterError(
            "LOCAL_HF_BATCH_PROMPT_CONTRACT_INVALID",
            "local HF compact prompt does not expose a claim-group count",
        ) from error
    if not isinstance(document, dict):
        raise _LocalHfAdapterError(
            "LOCAL_HF_BATCH_PROMPT_CONTRACT_INVALID",
            "local HF compact prompt does not expose a claim-group count",
        )
    compact_contract = document.get("compact_output_contract")
    if not isinstance(compact_contract, dict):
        raise _LocalHfAdapterError(
            "LOCAL_HF_BATCH_PROMPT_CONTRACT_INVALID",
            "local HF compact prompt does not expose a claim-group count",
        )
    value: object = compact_contract.get("claim_group_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _LocalHfAdapterError(
            "LOCAL_HF_BATCH_PROMPT_CONTRACT_INVALID",
            "local HF compact prompt claim-group count is invalid",
        )
    return value


def _batch_member_idempotency_key(
    *,
    provider_idempotency_key: str,
    endpoint_request_body: bytes,
) -> str:
    """Namespace exact member bytes away from both serial and future policies."""

    if not isinstance(provider_idempotency_key, str) or not provider_idempotency_key:
        raise _LocalHfAdapterError(
            "LOCAL_HF_BATCH_IDEMPOTENCY_INVALID",
            "local HF provider idempotency key must be nonempty",
        )
    if not isinstance(endpoint_request_body, bytes) or not endpoint_request_body:
        raise _LocalHfAdapterError(
            "LOCAL_HF_BATCH_IDEMPOTENCY_INVALID",
            "local HF endpoint member request bytes must be nonempty",
        )
    binding_sha256 = exact_bytes_sha256(
        canonical_json_bytes(
            {
                "adapter_batch_policy_version": LOCAL_HF_HYBRID_BATCH_POLICY_VERSION,
                "endpoint_request_exact_sha256": exact_bytes_sha256(endpoint_request_body),
                "provider_idempotency_key": provider_idempotency_key,
            }
        )
    )
    return f"{LOCAL_HF_HYBRID_BATCH_POLICY_VERSION}:{binding_sha256}"


def _file_uri_path(uri: str) -> Path:
    try:
        parsed = urlsplit(uri)
    except ValueError as error:
        raise _LocalHfAdapterError(
            "LOCAL_HF_FILE_URI_INVALID",
            "selected rendered artifact URI is malformed",
        ) from error
    if parsed.scheme.lower() != "file":
        raise _LocalHfAdapterError(
            "LOCAL_HF_FILE_URI_REQUIRED",
            "selected rendered artifacts must use file:// URIs",
        )
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise _LocalHfAdapterError(
            "LOCAL_HF_FILE_URI_INVALID",
            "local file URIs must not contain credentials, query, or fragment components",
        )
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        raise _LocalHfAdapterError(
            "LOCAL_HF_FILE_URI_INVALID",
            "local file URIs must not reference a remote host",
        )
    path_text = unquote(parsed.path)
    if not path_text:
        raise _LocalHfAdapterError(
            "LOCAL_HF_FILE_URI_INVALID",
            "local file URI has an empty path",
        )
    is_windows_drive_uri = (
        os.name == "nt"
        and path_text.startswith("/")
        and len(path_text) >= 3
        and path_text[2] == ":"
    )
    if is_windows_drive_uri:
        path_text = path_text[1:]
    return Path(path_text)


def _normalize_loopback_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("endpoint_url must be a nonempty string")
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as error:
        raise ValueError("endpoint_url contains an invalid port") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port != 8101
    ):
        raise ValueError("endpoint_url must be exactly the pinned 127.0.0.1:8101 loopback base")
    return LOCAL_HF_LOOPBACK_BASE_URL


def _decode_endpoint_response(data: bytes) -> LocalHfEndpointResponse:
    if not isinstance(data, bytes) or not data:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_INVALID_JSON",
            "local HF response must be nonempty JSON bytes",
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_INVALID_UTF8",
            "local HF response must not contain a UTF-8 BOM",
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_INVALID_UTF8",
            "local HF response is not strict UTF-8",
        ) from error
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except _DuplicateJsonKeyError as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_DUPLICATE_JSON_KEY",
            "local HF response contains a duplicate JSON object key",
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_INVALID_JSON",
            "local HF response is not strict JSON",
        ) from error
    if not isinstance(document, dict):
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_INVALID_JSON",
            "local HF response root must be a JSON object",
        )
    try:
        response = LocalHfEndpointResponse.model_validate_json(
            canonical_json_bytes(document),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_INVALID_CONTRACT",
            "local HF response does not satisfy the pinned endpoint contract",
        ) from error
    try:
        normalized = canonical_json_bytes(response.model_dump(mode="json"))
    except (TypeError, ValueError) as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_INVALID_CONTRACT",
            "local HF response could not be canonically normalized",
        ) from error
    if canonical_json_bytes(document) != normalized:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_RESPONSE_NORMALIZATION_MISMATCH",
            "local HF response would change under typed normalization",
        )
    return response


def _decode_batch_endpoint_response(data: bytes) -> LocalHfBatchEndpointResponse:
    if not isinstance(data, bytes) or not data:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_INVALID_JSON",
            "local HF batch response must be nonempty JSON bytes",
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_INVALID_UTF8",
            "local HF batch response must not contain a UTF-8 BOM",
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_INVALID_UTF8",
            "local HF batch response is not strict UTF-8",
        ) from error
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except _DuplicateJsonKeyError as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_DUPLICATE_JSON_KEY",
            "local HF batch response contains a duplicate JSON object key",
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_INVALID_JSON",
            "local HF batch response is not strict JSON",
        ) from error
    if not isinstance(document, dict):
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_INVALID_JSON",
            "local HF batch response root must be a JSON object",
        )
    try:
        response = LocalHfBatchEndpointResponse.model_validate_json(
            canonical_json_bytes(document),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_INVALID_CONTRACT",
            "local HF batch response does not satisfy the pinned endpoint contract",
        ) from error
    try:
        normalized = canonical_json_bytes(response.model_dump(mode="json"))
    except (TypeError, ValueError) as error:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_INVALID_CONTRACT",
            "local HF batch response could not be canonically normalized",
        ) from error
    if canonical_json_bytes(document) != normalized:
        raise _LocalHfEnvelopeError(
            "LOCAL_HF_BATCH_RESPONSE_NORMALIZATION_MISMATCH",
            "local HF batch response would change under typed normalization",
        )
    return response


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _elapsed_ms(started: float, completed: float) -> int:
    return max(0, int((completed - started) * 1_000))


LocalHfVisionAdapter = LocalHfLoopbackVisionAdapter
QwenLoopbackVisionAdapter = LocalHfLoopbackVisionAdapter
QwenLoopbackAdapterConfig = LocalHfLoopbackAdapterConfig


__all__ = [
    "LOCAL_HF_COMPACT_BARE_LABEL_POLICY_VERSION",
    "LOCAL_HF_COMPACT_BROADCAST_POLICY_VERSION",
    "LOCAL_HF_COMPACT_CAMERA_GROUP_POLICY_VERSION",
    "LOCAL_HF_COMPACT_NORMALIZATION_CONTRACT_VERSION",
    "LOCAL_HF_COMPACT_PROMPT_PROTOCOL",
    "LOCAL_HF_COMPACT_RESPONSE_INSTRUCTION",
    "LOCAL_HF_COMPACT_SCALAR_VALUE_POLICY_VERSION",
    "LOCAL_HF_COMPACT_SINGLE_DECISION_POLICY_VERSION",
    "LOCAL_HF_COMPACT_TOKEN_ORDER_POLICY_VERSION",
    "LOCAL_HF_DENSE_COORDINATE_REDUCTION_POLICY_VERSION",
    "LOCAL_HF_HYBRID_BATCH_INPUT_MAX_SIZE",
    "LOCAL_HF_HYBRID_BATCH_MAX_SIZE",
    "LOCAL_HF_HYBRID_BATCH_POLICY_VERSION",
    "LOCAL_HF_LOOPBACK_ADAPTER_VERSION",
    "LOCAL_HF_LOOPBACK_BASE_URL",
    "LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL",
    "LOCAL_HF_LOOPBACK_DEFAULT_NEW_TOKENS",
    "LOCAL_HF_LOOPBACK_ENDPOINT_URL",
    "LOCAL_HF_LOOPBACK_INFER_PATH",
    "LOCAL_HF_LOOPBACK_MAX_IMAGES",
    "LOCAL_HF_LOOPBACK_TOKEN_POLICY_VERSION",
    "LocalHfHttpRequest",
    "LocalHfHttpResponse",
    "LocalHfLoopbackAdapterConfig",
    "LocalHfLoopbackVisionAdapter",
    "LocalHfNativeBatchCapability",
    "LocalHfTransport",
    "LocalHfTransportError",
    "LocalHfVisionAdapter",
    "QwenLoopbackAdapterConfig",
    "QwenLoopbackVisionAdapter",
    "RecordedLocalHfExchange",
    "RecordedLocalHfTransport",
    "StdlibLocalHfTransport",
    "local_hf_compact_prompt_normalization_contract",
    "local_hf_compact_prompt_normalization_contract_sha256",
]
