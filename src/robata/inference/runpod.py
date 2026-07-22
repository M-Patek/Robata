"""RunPod transport adapter for the provider-neutral vision boundary."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import RLock
from typing import Annotated, Final, Literal, NoReturn, Protocol, Self
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.contracts.common import SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.logical_nodes import OpaqueUuid
from robata.inference.adapter import (
    NormalizedOutputEnvelope,
    VisionInferenceFailure,
    VisionInferenceRequest,
    VisionInferenceSuccess,
    VisionUsage,
)
from robata.inference.input_plan import RenderedProviderItem
from robata.inference.models import (
    InferenceFailure,
    InferenceStatus,
    ModelCapabilities,
    Retryability,
)
from robata.inference.offline_fixture import (
    RawProviderBytesStore,
    StrictProviderClaimParseError,
    StrictProviderClaimParser,
)

RUNPOD_REQUEST_CONTRACT_VERSION: Final = "robata-runpod-vision-request-v1"
RUNPOD_RESPONSE_CONTRACT_VERSION: Final = "robata-runpod-vision-response-v1"

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class RunPodEndpointConfig(StrictModel):
    """Non-secret, serializable configuration for one pinned RunPod endpoint."""

    provider: NonEmptyString
    endpoint_url: NonEmptyString
    adapter_version: SchemaVersion
    request_contract_version: Literal["robata-runpod-vision-request-v1"] = (
        RUNPOD_REQUEST_CONTRACT_VERSION
    )
    response_contract_version: Literal["robata-runpod-vision-response-v1"] = (
        RUNPOD_RESPONSE_CONTRACT_VERSION
    )
    request_timeout_cap_ms: Annotated[int, Field(strict=True, ge=1, le=300_000)] = 120_000
    max_response_bytes: Annotated[int, Field(strict=True, ge=1, le=16_777_216)] = 4_194_304

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        try:
            parsed = urlsplit(self.endpoint_url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint_url is not a valid HTTPS URL") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path
        ):
            raise ValueError(
                "endpoint_url must be an absolute HTTPS URL without credentials, query, or fragment"
            )
        if port is not None and not 1 <= port <= 65_535:
            raise ValueError("endpoint_url port is out of range")
        return self


class RunPodRetryPolicy(StrictModel):
    """Bounded deterministic retry parameters; jitter is intentionally absent."""

    version: SchemaVersion
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=5)] = 3
    base_delay_ms: Annotated[int, Field(strict=True, ge=0, le=10_000)] = 250
    max_delay_ms: Annotated[int, Field(strict=True, ge=0, le=60_000)] = 2_000
    retryable_http_statuses: tuple[Annotated[int, Field(strict=True, ge=400, le=599)], ...] = (
        408,
        425,
        429,
        500,
        502,
        503,
        504,
    )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.base_delay_ms > self.max_delay_ms:
            raise ValueError("base_delay_ms cannot exceed max_delay_ms")
        if not self.retryable_http_statuses or self.retryable_http_statuses != tuple(
            sorted(set(self.retryable_http_statuses))
        ):
            raise ValueError("retryable_http_statuses must be nonempty, unique, and sorted")
        return self

    def delay_after_attempt_ms(self, attempt: int) -> int:
        """Return the deterministic delay after a failed one-based attempt."""

        if type(attempt) is not int or not 1 <= attempt < self.max_attempts:
            raise ValueError("attempt must identify a retryable completed attempt")
        return int(min(self.max_delay_ms, self.base_delay_ms * (2 ** (attempt - 1))))


class RunPodApiKey:
    """Opaque credential value whose string representations are always redacted."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if (
            type(value) is not str
            or not 16 <= len(value) <= 4_096
            or value != value.strip()
            or any(character.isspace() for character in value)
        ):
            raise ValueError("RunPod API key must be 16-4096 non-whitespace characters")
        self._value = value

    def __repr__(self) -> str:
        return "RunPodApiKey(<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"

    def _authorization_header(self) -> str:
        return f"Bearer {self._value}"

    def _occurs_in(self, data: bytes) -> bool:
        return self._value.encode("utf-8") in data


@dataclass(frozen=True, slots=True, repr=False)
class RunPodHttpRequest:
    """One exact HTTP request passed to a transport; it never contains credentials."""

    url: str
    body: bytes
    timeout_seconds: float
    max_response_bytes: int
    idempotency_key: str

    def __post_init__(self) -> None:
        if not self.url or not isinstance(self.body, bytes) or not self.body:
            raise ValueError("RunPod HTTP request URL and body must be nonempty")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (float, int))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("RunPod HTTP request timeout must be positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes <= 0:
            raise ValueError("RunPod max response bytes must be positive")
        if not self.idempotency_key:
            raise ValueError("RunPod idempotency key must be nonempty")

    def __repr__(self) -> str:
        return (
            "RunPodHttpRequest("
            f"url={self.url!r}, body=<redacted:{len(self.body)} bytes>, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_response_bytes={self.max_response_bytes!r}, "
            f"idempotency_key={self.idempotency_key!r})"
        )


@dataclass(frozen=True, slots=True)
class RunPodHttpResponse:
    """Exact HTTP status and body returned by a transport."""

    status_code: int
    body: bytes

    def __post_init__(self) -> None:
        if type(self.status_code) is not int or not 100 <= self.status_code <= 599:
            raise ValueError("HTTP response status_code must be an integer from 100 to 599")
        if not isinstance(self.body, bytes):
            raise TypeError("HTTP response body must be bytes")


class RunPodTransportError(OSError):
    """The HTTP transport could not obtain a response."""


class RunPodTransport(Protocol):
    """Injectable async transport used by the adapter and offline tests."""

    async def post(
        self,
        request: RunPodHttpRequest,
        credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        """POST one request, using the credential only as an Authorization header."""
        ...


class StdlibRunPodTransport:
    """Small stdlib HTTP transport; no connection is made until ``post`` is called."""

    def __init__(self) -> None:
        self._network_call_count = 0
        self._counter_lock = RLock()

    @property
    def network_call_count(self) -> int:
        with self._counter_lock:
            return self._network_call_count

    async def post(
        self,
        request: RunPodHttpRequest,
        credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        return await asyncio.to_thread(self._post_sync, request, credential)

    def _post_sync(
        self,
        request: RunPodHttpRequest,
        credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        headers = {
            "Accept": "application/json",
            "Authorization": credential._authorization_header(),
            "Content-Type": "application/json",
            "Idempotency-Key": request.idempotency_key,
        }
        outgoing = urllib_request.Request(
            request.url,
            data=request.body,
            headers=headers,
            method="POST",
        )
        with self._counter_lock:
            self._network_call_count += 1
        try:
            with urllib_request.urlopen(outgoing, timeout=float(request.timeout_seconds)) as result:
                status_code = int(result.status)
                body = result.read(request.max_response_bytes + 1)
        except urllib_error.HTTPError as exc:
            status_code = int(exc.code)
            body = exc.read(request.max_response_bytes + 1)
        except (TimeoutError, urllib_error.URLError, OSError):
            raise RunPodTransportError("RunPod HTTP transport failed") from None
        return RunPodHttpResponse(status_code=status_code, body=body)


class _RunPodResponseBinding(StrictModel):
    request_id: OpaqueUuid
    logical_invocation_id: OpaqueUuid
    provider_idempotency_key: NonEmptyString
    provider: NonEmptyString
    model_name: NonEmptyString
    model_version: SchemaVersion
    task: NonEmptyString
    package_input_set_sha256: Sha256Digest
    rendered_input_digest: Sha256Digest
    prompt_sha256: Sha256Digest
    output_schema_sha256: Sha256Digest
    capability_snapshot_digest: Sha256Digest
    model_policy_version: SchemaVersion
    input_plan_semantic_sha256: Sha256Digest
    input_plan_part_semantic_sha256: Sha256Digest | None


class _RunPodWorkerUsage(StrictModel):
    input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None
    cost_usd: NonNegativeFiniteFloat | None


class _RunPodWorkerOutput(StrictModel):
    contract_version: NonEmptyString
    binding: _RunPodResponseBinding
    raw_output_json: NonEmptyString
    usage: _RunPodWorkerUsage


class _RunPodResponseEnvelope(StrictModel):
    id: NonEmptyString
    status: Literal["IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"]
    output: _RunPodWorkerOutput | None = None
    error: NonEmptyString | None = None
    delay_time_ms: NonNegativeInt | None = Field(default=None, alias="delayTime")
    execution_time_ms: NonNegativeInt | None = Field(default=None, alias="executionTime")
    worker_id: NonEmptyString | None = Field(default=None, alias="workerId")

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> Self:
        if self.status == "COMPLETED":
            if self.output is None or self.error is not None:
                raise ValueError("COMPLETED response requires output and forbids error")
        elif self.output is not None:
            raise ValueError("only a COMPLETED response may carry output")
        return self


class _RunPodEnvelopeError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class _DuplicateJsonKeyError(ValueError):
    pass


type Sleep = Callable[[float], Awaitable[None]]
type FailureStatus = Literal[
    InferenceStatus.FAILED,
    InferenceStatus.TIMEOUT,
    InferenceStatus.CANCELLED,
    InferenceStatus.INVALID_OUTPUT,
]


class RunPodVisionAdapter:
    """RunPod implementation of ``VisionModelAdapter`` awaiting endpoint qualification."""

    def __init__(
        self,
        *,
        config: RunPodEndpointConfig,
        credential: RunPodApiKey,
        capabilities: ModelCapabilities,
        retry_policy: RunPodRetryPolicy,
        raw_store: RawProviderBytesStore,
        parser: StrictProviderClaimParser,
        transport: RunPodTransport | None = None,
        sleep: Sleep = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, RunPodEndpointConfig):
            raise TypeError("config must be RunPodEndpointConfig")
        if not isinstance(credential, RunPodApiKey):
            raise TypeError("credential must be RunPodApiKey")
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities")
        if not isinstance(retry_policy, RunPodRetryPolicy):
            raise TypeError("retry_policy must be RunPodRetryPolicy")
        if not isinstance(raw_store, RawProviderBytesStore):
            raise TypeError("raw_store must implement RawProviderBytesStore")
        if not isinstance(parser, StrictProviderClaimParser):
            raise TypeError("parser must be StrictProviderClaimParser")
        resolved_transport = transport or StdlibRunPodTransport()
        if not callable(getattr(resolved_transport, "post", None)):
            raise TypeError("transport must implement RunPodTransport")
        if not callable(sleep) or not callable(monotonic):
            raise TypeError("sleep and monotonic must be callable")
        if (
            capabilities.provider != config.provider
            or not capabilities.supports_json_schema
            or (retry_policy.max_attempts > 1 and not capabilities.supports_provider_idempotency)
        ):
            raise ValueError(
                "RunPod capabilities must match the provider and support schema/idempotent dispatch"
            )
        self._config = config
        self._credential = credential
        self._capabilities = capabilities
        self._retry_policy = retry_policy
        self._raw_store = raw_store
        self._parser = parser
        self._transport = resolved_transport
        self._sleep = sleep
        self._monotonic = monotonic

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def config(self) -> RunPodEndpointConfig:
        return self._config

    @property
    def retry_policy(self) -> RunPodRetryPolicy:
        return self._retry_policy

    @property
    def raw_store(self) -> RawProviderBytesStore:
        return self._raw_store

    async def capabilities(self, model_name: str, model_version: str) -> ModelCapabilities:
        if (
            model_name != self._capabilities.model_name
            or model_version != self._capabilities.model_version
        ):
            raise ValueError("RunPod capability request does not match the pinned model")
        return self._capabilities

    async def infer(
        self,
        request: VisionInferenceRequest,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        started = self._monotonic()
        request_error = self._request_error(request)
        if request_error is not None:
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code="RUNPOD_REQUEST_REJECTED",
                detail=request_error,
                retryability=Retryability.PERMANENT,
            )
        try:
            body = canonical_json_bytes(self._request_document(request))
        except (TypeError, ValueError):
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code="RUNPOD_REQUEST_SERIALIZATION_FAILED",
                detail="RunPod request contains a non-canonical JSON value",
                retryability=Retryability.PERMANENT,
            )
        if self._credential._occurs_in(body):
            return self._failure(
                request,
                started=started,
                status=InferenceStatus.FAILED,
                code="RUNPOD_CREDENTIAL_IN_REQUEST",
                detail="RunPod request body contains credential material",
                retryability=Retryability.PERMANENT,
            )
        http_request = RunPodHttpRequest(
            url=self._config.endpoint_url,
            body=body,
            timeout_seconds=min(
                request.timeout_ms,
                self._config.request_timeout_cap_ms,
            )
            / 1_000,
            max_response_bytes=self._config.max_response_bytes,
            idempotency_key=request.provider_idempotency_key,
        )

        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                async with asyncio.timeout(http_request.timeout_seconds):
                    response = await self._transport.post(http_request, self._credential)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, RunPodTransportError):
                if attempt < self._retry_policy.max_attempts:
                    await self._retry_delay(attempt)
                    continue
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.TIMEOUT,
                    code="RUNPOD_TRANSPORT_RETRY_EXHAUSTED",
                    detail="RunPod transport timed out or remained unavailable",
                    retryability=Retryability.RETRYABLE,
                )
            except Exception:
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.FAILED,
                    code="RUNPOD_TRANSPORT_EXCEPTION",
                    detail="RunPod transport raised an unexpected exception",
                    retryability=Retryability.PERMANENT,
                )

            if not isinstance(response, RunPodHttpResponse):
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_TRANSPORT_CONTRACT_VIOLATION",
                    detail="RunPod transport returned an unsupported response",
                    retryability=Retryability.PERMANENT,
                )
            if response.status_code != 200:
                if (
                    response.status_code in self._retry_policy.retryable_http_statuses
                    and attempt < self._retry_policy.max_attempts
                ):
                    await self._retry_delay(attempt)
                    continue
                retryable = response.status_code in self._retry_policy.retryable_http_statuses
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.FAILED,
                    code=("RUNPOD_HTTP_RETRY_EXHAUSTED" if retryable else "RUNPOD_HTTP_REJECTED"),
                    detail=f"RunPod endpoint returned HTTP status {response.status_code}",
                    retryability=(
                        Retryability.RATE_LIMITED
                        if response.status_code == 429
                        else Retryability.RETRYABLE
                        if retryable
                        else Retryability.PERMANENT
                    ),
                )
            if len(response.body) > self._config.max_response_bytes:
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_RESPONSE_TOO_LARGE",
                    detail="RunPod response exceeds the configured byte limit",
                    retryability=Retryability.PERMANENT,
                )
            try:
                envelope = _decode_envelope(response.body)
            except _RunPodEnvelopeError as exc:
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code=exc.code,
                    detail=exc.detail,
                    retryability=Retryability.PERMANENT,
                )

            if envelope.status == "TIMED_OUT":
                if attempt < self._retry_policy.max_attempts:
                    await self._retry_delay(attempt)
                    continue
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.TIMEOUT,
                    code="RUNPOD_JOB_TIMEOUT_RETRY_EXHAUSTED",
                    detail="RunPod job timed out",
                    retryability=Retryability.RETRYABLE,
                    provider_request_id=envelope.id,
                )
            if envelope.status in {"IN_QUEUE", "IN_PROGRESS"}:
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_NONTERMINAL_RESPONSE",
                    detail="synchronous RunPod endpoint returned a nonterminal job",
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                )
            if envelope.status in {"FAILED", "CANCELLED"}:
                return self._failure(
                    request,
                    started=started,
                    status=(
                        InferenceStatus.CANCELLED
                        if envelope.status == "CANCELLED"
                        else InferenceStatus.FAILED
                    ),
                    code=f"RUNPOD_JOB_{envelope.status}",
                    detail=f"RunPod job ended with status {envelope.status}",
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                )

            output = envelope.output
            assert output is not None
            binding_error = self._binding_error(request, output)
            if binding_error is not None:
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_RESPONSE_BINDING_MISMATCH",
                    detail=binding_error,
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                )
            raw_output = output.raw_output_json.encode("utf-8")
            stored = self._raw_store.append(
                request_id=request.request_id,
                provider_request_id=envelope.id,
                data=raw_output,
            )
            try:
                payload = self._parser.decode_payload(
                    data=stored.data,
                    provider_claim_schema=request.output_schema,
                )
            except StrictProviderClaimParseError as exc:
                return self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code=exc.code.value,
                    detail=f"provider raw output failed strict parsing: {exc.code.value}",
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                    raw_output_artifact_id=stored.artifact_id,
                    usage=_usage(request, output.usage),
                )
            return VisionInferenceSuccess(
                status=InferenceStatus.SUCCEEDED,
                provider_request_id=envelope.id,
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
                usage=_usage(request, output.usage),
                latency_ms=_elapsed_ms(started, self._monotonic()),
            )

        raise AssertionError("bounded RunPod retry loop did not return")

    async def _retry_delay(self, attempt: int) -> None:
        await self._sleep(self._retry_policy.delay_after_attempt_ms(attempt) / 1_000)

    def _request_error(self, request: VisionInferenceRequest) -> str | None:
        if (
            request.provider != self.provider
            or request.model_name != self._capabilities.model_name
            or request.model_version != self._capabilities.model_version
            or request.capability_snapshot_id != self._capabilities.snapshot_id
            or request.capability_snapshot_digest != self._capabilities.snapshot_digest
        ):
            return "RunPod request target does not match pinned capabilities"
        if request.task not in self._capabilities.supported_tasks:
            return "RunPod request task is outside pinned capabilities"
        if request.input_plan is None or request.input_plan_semantic_sha256 is None:
            return "RunPod dispatch requires an immutable input plan"
        if request.input_plan.target.adapter_version != self._config.adapter_version:
            return "RunPod input plan adapter version does not match endpoint configuration"
        selected_items = _selected_items(request)
        if not selected_items:
            return "RunPod dispatch requires at least one rendered provider item"
        if any(
            item.artifact.media_type not in self._capabilities.accepted_media_types
            for item in selected_items
        ):
            return "RunPod input plan contains media outside pinned capabilities"
        return None

    def _request_document(self, request: VisionInferenceRequest) -> dict[str, object]:
        plan = request.input_plan
        assert plan is not None
        items = _selected_items(request)
        return {
            "input": {
                "contract_version": self._config.request_contract_version,
                "binding": _expected_binding(request),
                "prompt_artifact": {
                    "artifact_id": request.prompt_artifact_id,
                    "version": request.prompt_version,
                    "sha256": request.prompt_sha256,
                    "rendered_message_sha256": plan.prompt_output.rendered_message_sha256,
                },
                "output_schema": request.output_schema.model_dump(mode="json"),
                "package_inputs": [item.model_dump(mode="json") for item in request.package_inputs],
                "rendered_items": [
                    {
                        "provider_item_ordinal": item.provider_item_ordinal,
                        "package_id": item.package_id,
                        "package_ordinal": item.package_ordinal,
                        "camera_id": item.camera_id.value,
                        "camera_ordinal": item.camera_ordinal,
                        "frame_id": item.frame_id,
                        "frame_ordinal": item.frame_ordinal,
                        "aligned_timestamp_ns": str(item.aligned_timestamp_ns),
                        "source_timestamp_ns": str(item.source_timestamp_ns),
                        "source_artifact_sha256": item.source_artifact_sha256,
                        "artifact": item.artifact.model_dump(mode="json"),
                        "transform": item.transform.model_dump(mode="json"),
                    }
                    for item in items
                ],
                "generation_config": request.generation_config,
                "metadata": request.metadata,
            }
        }

    def _binding_error(
        self,
        request: VisionInferenceRequest,
        output: _RunPodWorkerOutput,
    ) -> str | None:
        if output.contract_version != self._config.response_contract_version:
            return "RunPod worker response contract version is not pinned"
        if output.binding.model_dump(mode="json") != _expected_binding(request):
            return "RunPod worker response identity echo differs from the request"
        return None

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
            usage=usage or _usage(request, None),
            latency_ms=_elapsed_ms(started, self._monotonic()),
            failure=InferenceFailure(
                code=code,
                detail=detail,
                retryability=retryability,
            ),
        )


def _selected_items(request: VisionInferenceRequest) -> tuple[RenderedProviderItem, ...]:
    plan = request.input_plan
    if plan is None:
        return ()
    if request.input_plan_part_ordinal is None:
        return tuple(plan.rendered_items)
    part = plan.call_plan.parts[request.input_plan_part_ordinal]
    return tuple(plan.rendered_items[part.start_item_ordinal : part.end_item_ordinal_exclusive])


def _expected_binding(request: VisionInferenceRequest) -> dict[str, object]:
    assert request.input_plan_semantic_sha256 is not None
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


def _usage(
    request: VisionInferenceRequest,
    reported: _RunPodWorkerUsage | None,
) -> VisionUsage:
    items = _selected_items(request)
    measured_input_tokens: int | None = None
    if request.input_plan is not None and request.input_plan_part_ordinal is not None:
        measured_input_tokens = request.input_plan.call_plan.parts[
            request.input_plan_part_ordinal
        ].measured_input_tokens
    input_tokens = reported.input_tokens if reported is not None else measured_input_tokens
    cost = reported.cost_usd if reported is not None else None
    return VisionUsage(
        input_frames=len(items),
        input_images=sum(item.artifact.media_type.startswith("image/") for item in items),
        input_tokens=input_tokens,
        output_tokens=reported.output_tokens if reported is not None else None,
        cost=cost,
        currency="USD" if cost is not None else None,
    )


def _decode_envelope(data: bytes) -> _RunPodResponseEnvelope:
    if not data:
        raise _RunPodEnvelopeError(
            "RUNPOD_RESPONSE_INVALID_JSON",
            "RunPod response must be nonempty JSON bytes",
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise _RunPodEnvelopeError(
            "RUNPOD_RESPONSE_INVALID_UTF8",
            "RunPod response must not contain a UTF-8 BOM",
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_RESPONSE_INVALID_UTF8",
            "RunPod response is not strict UTF-8",
        ) from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except _DuplicateJsonKeyError as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_RESPONSE_DUPLICATE_JSON_KEY",
            "RunPod response contains a duplicate JSON object key",
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_RESPONSE_INVALID_JSON",
            "RunPod response is not strict JSON",
        ) from exc
    if not isinstance(document, dict):
        raise _RunPodEnvelopeError(
            "RUNPOD_RESPONSE_INVALID_JSON",
            "RunPod response root must be a JSON object",
        )
    try:
        return _RunPodResponseEnvelope.model_validate_json(
            canonical_json_bytes(document),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_RESPONSE_INVALID_CONTRACT",
            "RunPod response does not satisfy the pinned worker envelope",
        ) from exc


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


__all__ = [
    "RUNPOD_REQUEST_CONTRACT_VERSION",
    "RUNPOD_RESPONSE_CONTRACT_VERSION",
    "RunPodApiKey",
    "RunPodEndpointConfig",
    "RunPodHttpRequest",
    "RunPodHttpResponse",
    "RunPodRetryPolicy",
    "RunPodTransport",
    "RunPodTransportError",
    "RunPodVisionAdapter",
    "StdlibRunPodTransport",
]
