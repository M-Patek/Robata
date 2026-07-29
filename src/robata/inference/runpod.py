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
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.inference.adapter import (
    NormalizedOutputEnvelope,
    ProviderQualificationObserver,
    ProviderQualificationSession,
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
from robata.runtime.observability import RuntimeObserver, runtime_increment, runtime_span

RUNPOD_REQUEST_CONTRACT_VERSION: Final = "robata-runpod-vision-request-v1"
RUNPOD_RESPONSE_CONTRACT_VERSION: Final = "robata-runpod-vision-response-v1"
RUNPOD_BATCH_REQUEST_CONTRACT_VERSION: Final = "robata-runpod-vision-batch-request-v1"
RUNPOD_BATCH_RESPONSE_CONTRACT_VERSION: Final = "robata-runpod-vision-batch-response-v1"
RUNPOD_QUALIFICATION_SESSION_METADATA_KEY: Final = "robata_qualification_session_id"
RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY: Final = "robata_qualification_run_namespace"
_QUALIFICATION_PRE_DISPATCH_FAILURE_CODES: Final = frozenset(
    {
        "RUNPOD_BATCH_REQUEST_SERIALIZATION_FAILED",
        "RUNPOD_CREDENTIAL_IN_REQUEST",
        "RUNPOD_REQUEST_REJECTED",
        "RUNPOD_REQUEST_SERIALIZATION_FAILED",
    }
)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class RunPodDeploymentConfiguration(StrictModel):
    """Pinned, non-secret deployment facts required for P10 qualification."""

    model_identifier: NonEmptyString
    model_version: SchemaVersion
    inference_engine: NonEmptyString
    precision_or_quantization: NonEmptyString
    topology: Literal["TWO_SINGLE_CARD_REPLICAS", "TWO_CARD_TENSOR_PARALLEL"]
    max_output_tokens: PositiveInt
    supported_topologies: tuple[
        Literal["TWO_SINGLE_CARD_REPLICAS", "TWO_CARD_TENSOR_PARALLEL"], ...
    ] = ()

    @model_validator(mode="after")
    def validate_topology_support(self) -> Self:
        if len(set(self.supported_topologies)) != len(self.supported_topologies):
            raise ValueError("supported_topologies must be unique")
        if self.supported_topologies and self.topology not in self.supported_topologies:
            raise ValueError("supported_topologies must include the deployed topology")
        return self


class RunPodEndpointConfig(StrictModel):
    """Non-secret, serializable configuration for one pinned RunPod endpoint."""

    provider: NonEmptyString
    deployment_configuration: RunPodDeploymentConfiguration | None = None
    endpoint_url: NonEmptyString
    adapter_version: SchemaVersion
    request_contract_version: Literal["robata-runpod-vision-request-v1"] = (
        RUNPOD_REQUEST_CONTRACT_VERSION
    )
    response_contract_version: Literal["robata-runpod-vision-response-v1"] = (
        RUNPOD_RESPONSE_CONTRACT_VERSION
    )
    # Native batch dispatch is deliberately opt-in. The default leaves the
    # established one-request RunPod wire contract untouched, even when an
    # upstream caller happens to use ``infer_batch``.
    native_batch_enabled: bool = False
    batch_request_contract_version: Literal["robata-runpod-vision-batch-request-v1"] = (
        RUNPOD_BATCH_REQUEST_CONTRACT_VERSION
    )
    batch_response_contract_version: Literal["robata-runpod-vision-batch-response-v1"] = (
        RUNPOD_BATCH_RESPONSE_CONTRACT_VERSION
    )
    native_batch_max_size: Annotated[int, Field(strict=True, ge=1, le=256)] = 1
    max_concurrent_requests: Annotated[int, Field(strict=True, ge=1, le=256)] = 1
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
        if self.native_batch_enabled:
            if self.native_batch_max_size < 2:
                raise ValueError("native batch dispatch requires native_batch_max_size >= 2")
        elif self.native_batch_max_size != 1:
            raise ValueError(
                "native_batch_max_size must remain one until native_batch_enabled is true"
            )
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


class RunPodNativeBatchQualification(StrictModel):
    """Representative endpoint evidence authorizing one native-batch configuration.

    This record is deliberately non-promotional. It binds a completed P6
    qualification artifact to the exact transport, capability, and retry
    configuration that opens native batch dispatch. P15 remains responsible for
    any production-eligibility decision.
    """

    qualification_version: Literal["runpod-native-batch-qualification-v1"] = (
        "runpod-native-batch-qualification-v1"
    )
    evidence_class: Literal["PRODUCTION_QUALIFICATION"] = "PRODUCTION_QUALIFICATION"
    qualification_status: Literal["REPRESENTATIVE_ENDPOINT_PASSED"] = (
        "REPRESENTATIVE_ENDPOINT_PASSED"
    )
    production_eligible: Literal[False] = False
    qualification_report_uri: NonEmptyString
    qualification_report_sha256: Sha256Digest
    handler_contract_evidence_uri: NonEmptyString
    handler_contract_evidence_sha256: Sha256Digest
    streaming_wait_deadline_evidence_uri: NonEmptyString
    streaming_wait_deadline_evidence_sha256: Sha256Digest
    adapter_binding_sha256: Sha256Digest
    handler_declares_exact_native_batch_contract: Literal[True] = True
    streaming_wait_deadline_gate_passed: Literal[True] = True

    @classmethod
    def create(
        cls,
        *,
        config: RunPodEndpointConfig,
        capabilities: ModelCapabilities,
        retry_policy: RunPodRetryPolicy,
        qualification_report_uri: str,
        qualification_report_sha256: str,
        handler_contract_evidence_uri: str,
        handler_contract_evidence_sha256: str,
        streaming_wait_deadline_evidence_uri: str,
        streaming_wait_deadline_evidence_sha256: str,
    ) -> Self:
        """Bind passed external P6 evidence to one active adapter projection."""

        if not isinstance(config, RunPodEndpointConfig):
            raise TypeError("config must be a RunPodEndpointConfig")
        if not isinstance(capabilities, ModelCapabilities):
            raise TypeError("capabilities must be ModelCapabilities")
        if not isinstance(retry_policy, RunPodRetryPolicy):
            raise TypeError("retry_policy must be a RunPodRetryPolicy")
        if not config.native_batch_enabled:
            raise ValueError("native batch qualification requires native batch to be enabled")
        return cls(
            qualification_report_uri=qualification_report_uri,
            qualification_report_sha256=qualification_report_sha256,
            handler_contract_evidence_uri=handler_contract_evidence_uri,
            handler_contract_evidence_sha256=handler_contract_evidence_sha256,
            streaming_wait_deadline_evidence_uri=streaming_wait_deadline_evidence_uri,
            streaming_wait_deadline_evidence_sha256=(streaming_wait_deadline_evidence_sha256),
            adapter_binding_sha256=_native_batch_adapter_binding_sha256(
                config=config,
                capabilities=capabilities,
                retry_policy=retry_policy,
            ),
        )

    def validate_adapter_binding(
        self,
        *,
        config: RunPodEndpointConfig,
        capabilities: ModelCapabilities,
        retry_policy: RunPodRetryPolicy,
    ) -> None:
        """Reject evidence copied from a different endpoint or dispatch contract."""

        if not config.native_batch_enabled:
            raise ValueError("native batch qualification cannot authorize a disabled configuration")
        expected = _native_batch_adapter_binding_sha256(
            config=config,
            capabilities=capabilities,
            retry_policy=retry_policy,
        )
        if self.adapter_binding_sha256 != expected:
            raise ValueError(
                "native batch qualification does not bind the active endpoint configuration"
            )


def _native_batch_adapter_binding_sha256(
    *,
    config: RunPodEndpointConfig,
    capabilities: ModelCapabilities,
    retry_policy: RunPodRetryPolicy,
) -> Sha256Digest:
    """Return the exact P6 adapter projection protected by a qualification record."""

    return exact_bytes_sha256(
        canonical_json_bytes(
            {
                "endpoint_config": config.model_dump(mode="json"),
                "capabilities": capabilities.model_dump(mode="json"),
                "retry_policy": retry_policy.model_dump(mode="json"),
            }
        )
    )


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


@dataclass(frozen=True, slots=True)
class RecordedRunPodExchange:
    """One redacted, request-bound provider response retained for replay.

    Recorded exchanges deliberately bind the exact canonical request bytes by
    digest. This makes a captured production response useful for exercising
    the normal adapter parser without allowing it to answer a different
    invocation after a prompt, model, or request contract change.
    """

    request_body_sha256: Sha256Digest
    response: RunPodHttpResponse | RunPodTransportError | TimeoutError


class RecordedRunPodTransport:
    """Deterministic transport for replaying recorded RunPod HTTP responses.

    Credentials are accepted only to satisfy the transport protocol and are
    never inspected or retained. An unrecorded request fails closed rather
    than returning a plausible fixture response.
    """

    def __init__(self, exchanges: tuple[RecordedRunPodExchange, ...]) -> None:
        if not isinstance(exchanges, tuple) or not exchanges:
            raise ValueError("recorded RunPod transport requires at least one exchange")
        by_digest: dict[str, list[RunPodHttpResponse | RunPodTransportError | TimeoutError]] = {}
        for exchange in exchanges:
            if not isinstance(exchange, RecordedRunPodExchange):
                raise TypeError("exchanges must contain RecordedRunPodExchange values")
            if not isinstance(
                exchange.response,
                (RunPodHttpResponse, RunPodTransportError, TimeoutError),
            ):
                raise TypeError(
                    "recorded RunPod response must be HTTP, timeout, or transport error"
                )
            by_digest.setdefault(exchange.request_body_sha256, []).append(exchange.response)
        self._by_digest = {digest: tuple(outcomes) for digest, outcomes in by_digest.items()}
        self._next_outcome_index: dict[str, int] = {}
        self._requests: list[RunPodHttpRequest] = []

    @property
    def request_count(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> tuple[RunPodHttpRequest, ...]:
        return tuple(self._requests)

    async def post(
        self,
        request: RunPodHttpRequest,
        _credential: RunPodApiKey,
    ) -> RunPodHttpResponse:
        self._requests.append(request)
        digest = exact_bytes_sha256(request.body)
        outcomes = self._by_digest.get(digest)
        next_index = self._next_outcome_index.get(digest, 0)
        if outcomes is None or next_index >= len(outcomes):
            raise RunPodTransportError("no recorded RunPod response matches request bytes")
        self._next_outcome_index[digest] = next_index + 1
        outcome = outcomes[next_index]
        if isinstance(outcome, (RunPodTransportError, TimeoutError)):
            raise outcome
        return outcome


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
    time_to_first_token_ms: NonNegativeInt | None = Field(
        default=None,
        alias="timeToFirstToken",
    )


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


class _RunPodBatchResponseItem(StrictModel):
    """One request-correlated terminal response in the opt-in batch wire."""

    request_id: OpaqueUuid
    response: _RunPodResponseEnvelope


class _RunPodBatchWorkerOutput(StrictModel):
    """Worker payload for the opt-in RunPod batch response contract."""

    contract_version: NonEmptyString
    items: tuple[_RunPodBatchResponseItem, ...]


class _RunPodBatchResponseEnvelope(StrictModel):
    """Top-level terminal envelope for one opt-in native RunPod batch."""

    id: NonEmptyString
    status: Literal["IN_QUEUE", "IN_PROGRESS", "COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"]
    output: _RunPodBatchWorkerOutput | None = None
    error: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> Self:
        if self.status == "COMPLETED":
            if self.output is None or self.error is not None:
                raise ValueError("COMPLETED batch response requires output and forbids error")
        elif self.output is not None:
            raise ValueError("only a COMPLETED batch response may carry output")
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
type RequestDispatchMode = Literal["single", "single_fallback", "native_batch"]
type RequestMetricMode = Literal[
    "single",
    "single_fallback",
    "native_batch",
    "prevalidation",
]
type BatchDispatchResult = tuple[
    tuple[int, ...],
    tuple[VisionInferenceSuccess | VisionInferenceFailure, ...],
]


@dataclass(frozen=True, slots=True)
class _NativeBatchEntry:
    """Prevalidated request state retained across native batch retries."""

    index: int
    request: VisionInferenceRequest
    started: float
    request_document: dict[str, object]


@dataclass(frozen=True, slots=True)
class _NativeBatchChunkResult:
    """Final item outcomes and the wire mode that produced each outcome."""

    outcomes: tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]
    outcome_modes: tuple[RequestDispatchMode, ...]


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
        runtime_observer: RuntimeObserver | None = None,
        qualification_observer: ProviderQualificationObserver | None = None,
        qualification_session: ProviderQualificationSession | None = None,
        native_batch_qualification: RunPodNativeBatchQualification | None = None,
        native_batch_qualification_measurement: bool = False,
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
        if qualification_observer is not None and (
            not callable(getattr(qualification_observer, "record_provider_timing", None))
            or not callable(getattr(qualification_observer, "record_provider_http_requests", None))
            or not callable(getattr(qualification_observer, "record_provider_retries", None))
        ):
            raise TypeError("qualification_observer must implement ProviderQualificationObserver")
        if (qualification_observer is None) != (qualification_session is None):
            raise ValueError(
                "qualification_observer and qualification_session must be supplied together"
            )
        if qualification_session is not None and not isinstance(
            qualification_session,
            ProviderQualificationSession,
        ):
            raise TypeError("qualification_session must be a ProviderQualificationSession")
        if native_batch_qualification is not None and not isinstance(
            native_batch_qualification,
            RunPodNativeBatchQualification,
        ):
            raise TypeError("native_batch_qualification must be a RunPodNativeBatchQualification")
        if type(native_batch_qualification_measurement) is not bool:
            raise TypeError("native_batch_qualification_measurement must be a bool")
        native_batch_qualification_state: Literal[
            "DISABLED",
            "QUALIFIED_EVIDENCE",
            "QUALIFICATION_MEASUREMENT",
        ]
        if config.native_batch_enabled:
            if native_batch_qualification is not None:
                if native_batch_qualification_measurement:
                    raise ValueError(
                        "native batch cannot use passed evidence and qualification "
                        "measurement mode together"
                    )
                native_batch_qualification.validate_adapter_binding(
                    config=config,
                    capabilities=capabilities,
                    retry_policy=retry_policy,
                )
                native_batch_qualification_state = "QUALIFIED_EVIDENCE"
            elif native_batch_qualification_measurement:
                if qualification_observer is None or qualification_session is None:
                    raise ValueError(
                        "native batch qualification measurement requires a scoped "
                        "observer and session"
                    )
                native_batch_qualification_state = "QUALIFICATION_MEASUREMENT"
            else:
                raise ValueError(
                    "native batch dispatch requires representative endpoint qualification evidence"
                )
        elif native_batch_qualification is not None or native_batch_qualification_measurement:
            raise ValueError(
                "native batch qualification inputs require native_batch_enabled to be true"
            )
        else:
            native_batch_qualification_state = "DISABLED"
        deployment = config.deployment_configuration
        if deployment is not None and (
            deployment.model_identifier != capabilities.model_name
            or deployment.model_version != capabilities.model_version
        ):
            raise ValueError("RunPod deployment model pin does not match capabilities")
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
        self._runtime_observer = runtime_observer
        self._qualification_observer = qualification_observer
        self._qualification_session = qualification_session
        self._qualification_observation_lock = RLock()
        self._qualification_observation_error: str | None = None
        self._qualification_observed_request_ids: set[str] = set()
        self._qualification_observed_http_request_count = 0
        self._qualification_observed_transport_retry_count = 0
        self._native_batch_qualification_state = native_batch_qualification_state
        # This is intentionally adapter-scoped rather than invocation-scoped: a
        # direct ``infer`` and every chunk/retry started by any concurrent
        # ``infer_batch`` call share the same endpoint capacity budget.
        self._dispatch_gate = asyncio.Semaphore(config.max_concurrent_requests)

    @property
    def capabilities_snapshot(self) -> ModelCapabilities:
        """Return the immutable capability snapshot used for every dispatch."""

        return self._capabilities

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
    def qualification_observer(self) -> ProviderQualificationObserver | None:
        return self._qualification_observer

    @property
    def qualification_session(self) -> ProviderQualificationSession | None:
        return self._qualification_session

    @property
    def native_batch_qualification_state(
        self,
    ) -> Literal["DISABLED", "QUALIFIED_EVIDENCE", "QUALIFICATION_MEASUREMENT"]:
        """State used to distinguish ordinary dispatch from fresh P6 measurement."""

        return self._native_batch_qualification_state

    @property
    def qualification_observation_error(self) -> str | None:
        """Return the first fail-open observer error for a qualification runner to reject."""

        with self._qualification_observation_lock:
            return self._qualification_observation_error

    @property
    def qualification_observed_request_ids(self) -> tuple[str, ...]:
        """Return request IDs whose terminal observations reached the scoped sink."""

        with self._qualification_observation_lock:
            return tuple(sorted(self._qualification_observed_request_ids))

    @property
    def qualification_observed_http_request_count(self) -> int:
        """Return actual posts emitted while this adapter was qualification-scoped."""

        with self._qualification_observation_lock:
            return self._qualification_observed_http_request_count

    @property
    def qualification_observed_transport_retry_count(self) -> int:
        """Return actual adapter transport retry items for the qualification session."""

        with self._qualification_observation_lock:
            return self._qualification_observed_transport_retry_count

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
        """Dispatch one request using the established single-request wire."""

        return await self._infer_single_observed(request, mode="single")

    async def _infer_single_observed(
        self,
        request: VisionInferenceRequest,
        *,
        mode: Literal["single", "single_fallback"],
        initial_attempt: int = 1,
        started: float | None = None,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        """Run one legacy-wire request and label its actual dispatch mode."""

        attributes = {"mode": mode, "provider": self.provider}
        self._record_request_started(mode)
        with runtime_span(self._runtime_observer, "inference.runpod.request", attributes):
            outcome = await self._infer_single(
                request,
                retry_mode=mode,
                initial_attempt=initial_attempt,
                started=started,
            )
        self._record_request_outcome(outcome, mode=mode)
        return outcome

    def _record_request_started(self, mode: RequestMetricMode) -> None:
        runtime_increment(
            self._runtime_observer,
            "inference.runpod.requests",
            attributes={"mode": mode, "provider": self.provider},
        )

    def _record_request_outcome(
        self,
        outcome: VisionInferenceSuccess | VisionInferenceFailure,
        *,
        mode: RequestMetricMode,
    ) -> None:
        runtime_increment(
            self._runtime_observer,
            "inference.runpod.request_outcomes",
            attributes={
                "mode": mode,
                "provider": self.provider,
                "status": outcome.status.value,
            },
        )

    async def _post(self, request: RunPodHttpRequest) -> RunPodHttpResponse:
        """Run one HTTP attempt under the adapter-wide endpoint capacity gate."""

        # The deadline includes capacity wait time. A request that cannot obtain
        # endpoint capacity before its timeout must fail like any other exhausted
        # transport attempt rather than wait indefinitely behind unrelated work.
        async with asyncio.timeout(request.timeout_seconds):
            async with self._dispatch_gate:
                if (
                    self._qualification_observer is not None
                    and self._qualification_session is not None
                ):
                    with self._qualification_observation_lock:
                        self._qualification_observed_http_request_count += 1
                self._record_provider_http_requests()
                return await self._transport.post(request, self._credential)

    async def _infer_single(
        self,
        request: VisionInferenceRequest,
        *,
        retry_mode: Literal["single", "single_fallback"] = "single",
        initial_attempt: int = 1,
        started: float | None = None,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        if (
            type(initial_attempt) is not int
            or not 1 <= initial_attempt <= self._retry_policy.max_attempts
        ):
            raise ValueError("initial_attempt must be within the bounded retry policy")
        if started is None:
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

        for attempt in range(initial_attempt, self._retry_policy.max_attempts + 1):
            try:
                response = await self._post(http_request)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, RunPodTransportError):
                if attempt < self._retry_policy.max_attempts:
                    await self._retry_delay(attempt, mode=retry_mode)
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
                    await self._retry_delay(attempt, mode=retry_mode)
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
                    await self._retry_delay(attempt, mode=retry_mode)
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
            return self._outcome_from_terminal_envelope(request, envelope, started)

        raise AssertionError("bounded RunPod retry loop did not return")

    async def infer_batch(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        """Dispatch an ordered bounded batch without changing inference identity.

        A native batch wire is used only when the endpoint configuration explicitly
        enables it. Otherwise this method intentionally falls back to the proven
        one-request contract, with bounded concurrent request dispatch.
        """

        if not isinstance(requests, tuple):
            raise TypeError("requests must be a tuple")
        if any(not isinstance(request, VisionInferenceRequest) for request in requests):
            raise TypeError("requests must contain only VisionInferenceRequest values")
        if not requests:
            return ()
        mode = (
            "native_batch"
            if self._config.native_batch_enabled and len(requests) >= 2
            else "single_fallback"
        )
        runtime_increment(
            self._runtime_observer,
            "inference.runpod.batch_invocations",
            attributes={"batch_size": len(requests), "mode": mode, "provider": self.provider},
        )
        if not self._config.native_batch_enabled or len(requests) < 2:
            return await self._infer_single_request_fallback(requests)
        return await self._infer_native_batches(requests)

    async def _infer_single_request_fallback(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        """Retain exact legacy request bodies while bounding fallback concurrency."""

        runtime_increment(
            self._runtime_observer,
            "inference.runpod.batch_fallbacks",
            attributes={
                "batch_size": len(requests),
                "mode": "single_fallback",
                "provider": self.provider,
            },
        )
        tasks = tuple(
            asyncio.create_task(self._infer_single_observed(request, mode="single_fallback"))
            for request in requests
        )
        try:
            return tuple(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _infer_native_batches(
        self,
        requests: tuple[VisionInferenceRequest, ...],
    ) -> tuple[VisionInferenceSuccess | VisionInferenceFailure, ...]:
        """Validate, split, concurrently dispatch, and order native batch outcomes."""

        outcomes: list[VisionInferenceSuccess | VisionInferenceFailure | None] = [None] * len(
            requests
        )
        compatible_groups: dict[tuple[object, ...], list[_NativeBatchEntry]] = {}
        for index, request in enumerate(requests):
            started = self._monotonic()
            request_error = self._request_error(request)
            if request_error is not None:
                outcomes[index] = self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.FAILED,
                    code="RUNPOD_REQUEST_REJECTED",
                    detail=request_error,
                    retryability=Retryability.PERMANENT,
                )
                continue
            try:
                request_document = self._request_document(request)
                single_request_body = canonical_json_bytes(request_document)
            except (TypeError, ValueError):
                outcomes[index] = self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.FAILED,
                    code="RUNPOD_REQUEST_SERIALIZATION_FAILED",
                    detail="RunPod request contains a non-canonical JSON value",
                    retryability=Retryability.PERMANENT,
                )
                continue
            if self._credential._occurs_in(single_request_body):
                outcomes[index] = self._failure(
                    request,
                    started=started,
                    status=InferenceStatus.FAILED,
                    code="RUNPOD_CREDENTIAL_IN_REQUEST",
                    detail="RunPod request body contains credential material",
                    retryability=Retryability.PERMANENT,
                )
                continue
            entry = _NativeBatchEntry(
                index=index,
                request=request,
                started=started,
                request_document=request_document,
            )
            compatible_groups.setdefault(_native_batch_compatibility_key(request), []).append(entry)

        for outcome in outcomes:
            if outcome is not None:
                self._record_request_started("prevalidation")
                self._record_request_outcome(outcome, mode="prevalidation")

        async def dispatch_native_chunk(
            entries: tuple[_NativeBatchEntry, ...],
        ) -> BatchDispatchResult:
            chunk_result = await self._dispatch_native_batch_chunk(entries)
            for outcome, mode in zip(
                chunk_result.outcomes,
                chunk_result.outcome_modes,
                strict=True,
            ):
                # A lone timed-out item may complete on the legacy fallback wire;
                # that helper already emits its own per-item observation.
                if mode != "single_fallback":
                    self._record_request_started(mode)
                    self._record_request_outcome(outcome, mode=mode)
            return tuple(entry.index for entry in entries), chunk_result.outcomes

        async def dispatch_single_entry(
            entry: _NativeBatchEntry,
        ) -> BatchDispatchResult:
            # The configuration opted into batching, but this incompatible or
            # trailing singleton cannot safely use the batch-only wire.
            return (entry.index,), (
                await self._infer_single_observed(entry.request, mode="single_fallback"),
            )

        tasks: list[asyncio.Task[BatchDispatchResult]] = []
        for entries in compatible_groups.values():
            for start in range(0, len(entries), self._config.native_batch_max_size):
                chunk = tuple(entries[start : start + self._config.native_batch_max_size])
                if len(chunk) == 1:
                    tasks.append(asyncio.create_task(dispatch_single_entry(chunk[0])))
                else:
                    tasks.append(asyncio.create_task(dispatch_native_chunk(chunk)))
        try:
            completed = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for item in completed:
            indexes, chunk_outcomes = item
            if len(indexes) != len(chunk_outcomes):
                raise AssertionError("native RunPod batch result has an invalid cardinality")
            for index, outcome in zip(indexes, chunk_outcomes, strict=True):
                outcomes[index] = outcome

        if any(outcome is None for outcome in outcomes):
            raise AssertionError("native RunPod batch did not produce one outcome per request")
        return tuple(outcome for outcome in outcomes if outcome is not None)

    async def _dispatch_native_batch_chunk(
        self,
        entries: tuple[_NativeBatchEntry, ...],
    ) -> _NativeBatchChunkResult:
        """Dispatch one compatible native batch and retry only timed-out items."""

        if not 2 <= len(entries) <= self._config.native_batch_max_size:
            raise ValueError("native batch chunk size is outside the configured bound")
        outcomes: dict[int, VisionInferenceSuccess | VisionInferenceFailure] = {}
        outcome_modes: dict[int, RequestDispatchMode] = {}
        pending = entries
        for attempt in range(1, self._retry_policy.max_attempts + 1):
            try:
                body = canonical_json_bytes(self._batch_request_document(pending))
            except (TypeError, ValueError):
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.FAILED,
                    code="RUNPOD_BATCH_REQUEST_SERIALIZATION_FAILED",
                    detail="RunPod batch request contains a non-canonical JSON value",
                    retryability=Retryability.PERMANENT,
                )
                break
            if self._credential._occurs_in(body):
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.FAILED,
                    code="RUNPOD_CREDENTIAL_IN_REQUEST",
                    detail="RunPod batch request body contains credential material",
                    retryability=Retryability.PERMANENT,
                )
                break
            http_request = RunPodHttpRequest(
                url=self._config.endpoint_url,
                body=body,
                timeout_seconds=min(
                    *(entry.request.timeout_ms for entry in pending),
                    self._config.request_timeout_cap_ms,
                )
                / 1_000,
                max_response_bytes=self._config.max_response_bytes,
                idempotency_key=self._batch_idempotency_key(pending),
            )
            attributes: dict[str, int | str] = {
                "batch_size": len(pending),
                "provider": self.provider,
            }
            runtime_increment(
                self._runtime_observer,
                "inference.runpod.native_batch_dispatches",
                attributes=attributes,
            )
            try:
                with runtime_span(
                    self._runtime_observer,
                    "inference.runpod.native_batch_dispatch",
                    attributes,
                ):
                    response = await self._post(http_request)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, RunPodTransportError):
                if attempt < self._retry_policy.max_attempts:
                    await self._retry_delay(
                        attempt,
                        mode="native_batch",
                        request_count=len(pending),
                    )
                    continue
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.TIMEOUT,
                    code="RUNPOD_TRANSPORT_RETRY_EXHAUSTED",
                    detail="RunPod transport timed out or remained unavailable",
                    retryability=Retryability.RETRYABLE,
                )
                break
            except Exception:
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.FAILED,
                    code="RUNPOD_TRANSPORT_EXCEPTION",
                    detail="RunPod transport raised an unexpected exception",
                    retryability=Retryability.PERMANENT,
                )
                break

            if not isinstance(response, RunPodHttpResponse):
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_TRANSPORT_CONTRACT_VIOLATION",
                    detail="RunPod transport returned an unsupported response",
                    retryability=Retryability.PERMANENT,
                )
                break
            if response.status_code != 200:
                retryable = response.status_code in self._retry_policy.retryable_http_statuses
                if retryable and attempt < self._retry_policy.max_attempts:
                    await self._retry_delay(
                        attempt,
                        mode="native_batch",
                        request_count=len(pending),
                    )
                    continue
                self._set_batch_failures(
                    outcomes,
                    pending,
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
                break
            if len(response.body) > self._config.max_response_bytes:
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_RESPONSE_TOO_LARGE",
                    detail="RunPod response exceeds the configured byte limit",
                    retryability=Retryability.PERMANENT,
                )
                break
            try:
                envelope = _decode_batch_envelope(response.body)
            except _RunPodEnvelopeError as exc:
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code=exc.code,
                    detail=exc.detail,
                    retryability=Retryability.PERMANENT,
                )
                break

            if envelope.status == "TIMED_OUT":
                if attempt < self._retry_policy.max_attempts:
                    await self._retry_delay(
                        attempt,
                        mode="native_batch",
                        request_count=len(pending),
                    )
                    continue
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.TIMEOUT,
                    code="RUNPOD_BATCH_JOB_TIMEOUT_RETRY_EXHAUSTED",
                    detail="RunPod batch job timed out",
                    retryability=Retryability.RETRYABLE,
                    provider_request_id=envelope.id,
                )
                break
            if envelope.status in {"IN_QUEUE", "IN_PROGRESS"}:
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_BATCH_NONTERMINAL_RESPONSE",
                    detail="synchronous RunPod batch endpoint returned a nonterminal job",
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                )
                break
            if envelope.status in {"FAILED", "CANCELLED"}:
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=(
                        InferenceStatus.CANCELLED
                        if envelope.status == "CANCELLED"
                        else InferenceStatus.FAILED
                    ),
                    code=f"RUNPOD_BATCH_JOB_{envelope.status}",
                    detail=f"RunPod batch job ended with status {envelope.status}",
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                )
                break

            output = envelope.output
            assert output is not None
            if output.contract_version != self._config.batch_response_contract_version:
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_BATCH_RESPONSE_CONTRACT_MISMATCH",
                    detail="RunPod batch worker response contract version is not pinned",
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                )
                break
            item_by_request_id = {item.request_id: item.response for item in output.items}
            expected_request_ids = {entry.request.request_id for entry in pending}
            if (
                len(item_by_request_id) != len(output.items)
                or set(item_by_request_id) != expected_request_ids
            ):
                self._set_batch_failures(
                    outcomes,
                    pending,
                    status=InferenceStatus.INVALID_OUTPUT,
                    code="RUNPOD_BATCH_RESPONSE_ITEM_SET_MISMATCH",
                    detail="RunPod batch response does not contain one unique item per request",
                    retryability=Retryability.PERMANENT,
                    provider_request_id=envelope.id,
                )
                break

            retry_entries: list[_NativeBatchEntry] = []
            for entry in pending:
                item_envelope = item_by_request_id[entry.request.request_id]
                if item_envelope.status == "TIMED_OUT":
                    if attempt < self._retry_policy.max_attempts:
                        retry_entries.append(entry)
                    else:
                        outcomes[entry.index] = self._failure(
                            entry.request,
                            started=entry.started,
                            status=InferenceStatus.TIMEOUT,
                            code="RUNPOD_JOB_TIMEOUT_RETRY_EXHAUSTED",
                            detail="RunPod job timed out",
                            retryability=Retryability.RETRYABLE,
                            provider_request_id=item_envelope.id,
                        )
                    continue
                try:
                    outcomes[entry.index] = self._outcome_from_terminal_envelope(
                        entry.request,
                        item_envelope,
                        entry.started,
                    )
                except Exception:
                    outcomes[entry.index] = self._failure(
                        entry.request,
                        started=entry.started,
                        status=InferenceStatus.FAILED,
                        code="RUNPOD_BATCH_ITEM_PROCESSING_EXCEPTION",
                        detail="RunPod batch item could not be persisted or parsed",
                        retryability=Retryability.RETRYABLE,
                        provider_request_id=item_envelope.id,
                    )
            pending = tuple(retry_entries)
            if not pending:
                break
            if len(pending) == 1:
                # The native contract is only used for real batches. Once a
                # partial response leaves one item, retain its legacy request
                # shape for the retry and consume only its remaining attempts.
                entry = pending[0]
                await self._retry_delay(
                    attempt,
                    mode="single_fallback",
                    request_count=1,
                )
                outcomes[entry.index] = await self._infer_single_observed(
                    entry.request,
                    mode="single_fallback",
                    initial_attempt=attempt + 1,
                    started=entry.started,
                )
                outcome_modes[entry.index] = "single_fallback"
                break
            await self._retry_delay(
                attempt,
                mode="native_batch",
                request_count=len(pending),
            )

        if any(entry.index not in outcomes for entry in entries):
            raise AssertionError("native RunPod batch retry loop did not produce all outcomes")
        return _NativeBatchChunkResult(
            outcomes=tuple(outcomes[entry.index] for entry in entries),
            outcome_modes=tuple(
                outcome_modes.get(entry.index, "native_batch") for entry in entries
            ),
        )

    def _set_batch_failures(
        self,
        outcomes: dict[int, VisionInferenceSuccess | VisionInferenceFailure],
        entries: tuple[_NativeBatchEntry, ...],
        *,
        status: FailureStatus,
        code: str,
        detail: str,
        retryability: Retryability,
        provider_request_id: str | None = None,
    ) -> None:
        for entry in entries:
            outcomes[entry.index] = self._failure(
                entry.request,
                started=entry.started,
                status=status,
                code=code,
                detail=detail,
                retryability=retryability,
                provider_request_id=provider_request_id,
            )

    def _batch_request_document(
        self,
        entries: tuple[_NativeBatchEntry, ...],
    ) -> dict[str, object]:
        return {
            "input": {
                "contract_version": self._config.batch_request_contract_version,
                "requests": [
                    {
                        "request_id": entry.request.request_id,
                        "provider_idempotency_key": entry.request.provider_idempotency_key,
                        "request": entry.request_document["input"],
                    }
                    for entry in entries
                ],
            }
        }

    @staticmethod
    def _batch_idempotency_key(entries: tuple[_NativeBatchEntry, ...]) -> str:
        digest = exact_bytes_sha256(
            canonical_json_bytes(
                [
                    {
                        "request_id": entry.request.request_id,
                        "provider_idempotency_key": entry.request.provider_idempotency_key,
                    }
                    for entry in entries
                ]
            )
        )
        return f"robata-runpod-batch:{digest}"

    def _outcome_from_terminal_envelope(
        self,
        request: VisionInferenceRequest,
        envelope: _RunPodResponseEnvelope,
        started: float,
    ) -> VisionInferenceSuccess | VisionInferenceFailure:
        """Translate one terminal worker envelope after retry policy has run."""

        if envelope.status == "TIMED_OUT":
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
            end_to_end_ms = _elapsed_ms(started, self._monotonic())
            self._record_provider_timing(
                request,
                provider_queue_ms=envelope.delay_time_ms,
                provider_execution_ms=envelope.execution_time_ms,
                time_to_first_token_ms=output.usage.time_to_first_token_ms,
                end_to_end_ms=end_to_end_ms,
                input_tokens=output.usage.input_tokens,
                input_tokens_known=output.usage.input_tokens is not None,
                output_tokens=output.usage.output_tokens,
                output_tokens_known=output.usage.output_tokens is not None,
                accepted=False,
            )
            return self._failure(
                request,
                started=started,
                latency_ms=end_to_end_ms,
                status=InferenceStatus.INVALID_OUTPUT,
                code="RUNPOD_RESPONSE_BINDING_MISMATCH",
                detail=binding_error,
                retryability=Retryability.PERMANENT,
                provider_request_id=envelope.id,
                qualification_timing_recorded=True,
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
            end_to_end_ms = _elapsed_ms(started, self._monotonic())
            self._record_provider_timing(
                request,
                provider_queue_ms=envelope.delay_time_ms,
                provider_execution_ms=envelope.execution_time_ms,
                time_to_first_token_ms=output.usage.time_to_first_token_ms,
                end_to_end_ms=end_to_end_ms,
                input_tokens=output.usage.input_tokens,
                input_tokens_known=output.usage.input_tokens is not None,
                output_tokens=output.usage.output_tokens,
                output_tokens_known=output.usage.output_tokens is not None,
                accepted=False,
            )
            return self._failure(
                request,
                started=started,
                latency_ms=end_to_end_ms,
                status=InferenceStatus.INVALID_OUTPUT,
                code=exc.code.value,
                detail=f"provider raw output failed strict parsing: {exc.code.value}",
                retryability=Retryability.PERMANENT,
                provider_request_id=envelope.id,
                raw_output_artifact_id=stored.artifact_id,
                usage=_usage(request, output.usage),
                qualification_timing_recorded=True,
            )
        end_to_end_ms = _elapsed_ms(started, self._monotonic())
        self._record_provider_timing(
            request,
            provider_queue_ms=envelope.delay_time_ms,
            provider_execution_ms=envelope.execution_time_ms,
            time_to_first_token_ms=output.usage.time_to_first_token_ms,
            end_to_end_ms=end_to_end_ms,
            input_tokens=output.usage.input_tokens,
            input_tokens_known=output.usage.input_tokens is not None,
            output_tokens=output.usage.output_tokens,
            output_tokens_known=output.usage.output_tokens is not None,
            accepted=True,
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
            latency_ms=end_to_end_ms,
        )

    def _is_qualification_request(self, request: VisionInferenceRequest) -> bool:
        session = self._qualification_session
        return (
            self._qualification_observer is not None
            and session is not None
            and request.metadata.get(RUNPOD_QUALIFICATION_SESSION_METADATA_KEY)
            == session.session_id
            and request.metadata.get(RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY)
            == session.run_namespace
        )

    def _record_provider_timing(
        self,
        request: VisionInferenceRequest,
        *,
        provider_queue_ms: int | None,
        provider_execution_ms: int | None,
        time_to_first_token_ms: int | None,
        end_to_end_ms: int,
        input_tokens: int | None,
        input_tokens_known: bool,
        output_tokens: int | None,
        output_tokens_known: bool,
        accepted: bool,
    ) -> None:
        """Expose one terminal provider observation without changing inference semantics."""

        attributes = {"provider": self.provider}
        for name, value in (
            ("inference.runpod.provider_queue_ms", provider_queue_ms),
            ("inference.runpod.provider_execution_ms", provider_execution_ms),
            ("inference.runpod.time_to_first_token_ms", time_to_first_token_ms),
        ):
            if value is not None and value > 0:
                runtime_increment(self._runtime_observer, name, value, attributes)
        if not self._is_qualification_request(request):
            return
        observer = self._qualification_observer
        session = self._qualification_session
        assert observer is not None and session is not None
        try:
            (
                logical_invocation_id,
                input_plan_part_ordinal,
                provider_image_count,
            ) = _qualification_request_measurement(request)
            observer.record_provider_timing(
                qualification_session=session,
                request_id=request.request_id,
                logical_invocation_id=logical_invocation_id,
                input_plan_part_ordinal=input_plan_part_ordinal,
                provider_image_count=provider_image_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_queue_ms=provider_queue_ms,
                provider_execution_ms=provider_execution_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                end_to_end_ms=end_to_end_ms,
                input_tokens_known=input_tokens_known,
                output_tokens_known=output_tokens_known,
                accepted=accepted,
            )
        except Exception as exc:
            self._record_qualification_observation_error("terminal", exc)
            return
        with self._qualification_observation_lock:
            self._qualification_observed_request_ids.add(request.request_id)

    def _record_provider_http_requests(self) -> None:
        observer = self._qualification_observer
        session = self._qualification_session
        if observer is None or session is None:
            return
        try:
            observer.record_provider_http_requests(
                qualification_session=session,
                count=1,
            )
        except Exception as exc:
            self._record_qualification_observation_error("http", exc)

    def _record_qualification_observation_error(
        self,
        operation: str,
        exc: Exception,
    ) -> None:
        with self._qualification_observation_lock:
            if self._qualification_observation_error is None:
                self._qualification_observation_error = f"{operation}:{type(exc).__name__}"

    async def _retry_delay(
        self,
        attempt: int,
        *,
        mode: RequestDispatchMode = "single",
        request_count: int = 1,
    ) -> None:
        runtime_increment(
            self._runtime_observer,
            "inference.runpod.retries",
            request_count,
            {"mode": mode, "provider": self.provider},
        )
        observer = self._qualification_observer
        session = self._qualification_session
        if observer is not None and session is not None:
            with self._qualification_observation_lock:
                self._qualification_observed_transport_retry_count += request_count
            try:
                observer.record_provider_retries(
                    qualification_session=session,
                    count=request_count,
                )
            except Exception as exc:
                self._record_qualification_observation_error("retry", exc)
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
        deployment = self._config.deployment_configuration
        if deployment is not None:
            requested_output_tokens = request.generation_config.get("max_output_tokens")
            if (
                type(requested_output_tokens) is not int
                or requested_output_tokens != deployment.max_output_tokens
            ):
                return "RunPod request output-token limit does not match pinned deployment"
        session = self._qualification_session
        if session is not None and not self._is_qualification_request(request):
            return "RunPod qualification request is not bound to the active session and namespace"
        plan = request.input_plan
        if plan is None or request.input_plan_semantic_sha256 is None:
            return "RunPod dispatch requires an immutable input plan"
        if plan.target.adapter_version != self._config.adapter_version:
            return "RunPod input plan adapter version does not match endpoint configuration"
        if len(plan.call_plan.parts) > 1 and request.input_plan_part_ordinal is None:
            return "RunPod dispatch requires an explicit call part for a split input plan"
        selected_items = _selected_items(request)
        if not selected_items:
            return "RunPod dispatch requires at least one rendered provider item"
        selected_image_count = sum(
            item.artifact.media_type.startswith("image/") for item in selected_items
        )
        max_images_per_request = self._capabilities.max_images_per_request
        if max_images_per_request is None:
            return "RunPod capabilities do not declare an image limit"
        if selected_image_count > max_images_per_request:
            return "RunPod input plan exceeds pinned image limit"
        if any(
            item.artifact.media_type not in self._capabilities.accepted_media_types
            for item in selected_items
        ):
            return "RunPod input plan contains media outside pinned capabilities"
        part_ordinal = request.input_plan_part_ordinal or 0
        measured_input_tokens = plan.call_plan.parts[part_ordinal].measured_input_tokens
        max_input_tokens = self._capabilities.max_input_tokens
        if max_input_tokens is None:
            return "RunPod capabilities do not declare an input-token limit"
        if measured_input_tokens > max_input_tokens:
            return "RunPod input plan exceeds pinned input-token limit"
        if session is not None:
            matching_contracts = tuple(
                contract
                for contract in session.request_contracts
                if (
                    contract.task is request.task
                    and contract.prompt_artifact_id == request.prompt_artifact_id
                    and contract.prompt_version == request.prompt_version
                    and contract.prompt_sha256 == request.prompt_sha256
                    and contract.output_schema_sha256 == request.output_schema.sha256
                )
            )
            if len(matching_contracts) != 1:
                return "RunPod qualification request does not match a pinned prompt contract"
            if (
                plan.applicable_limits.max_input_tokens_per_request
                != matching_contracts[0].max_input_tokens
            ):
                return "RunPod qualification request context limit does not match pinned contract"
            if measured_input_tokens > matching_contracts[0].max_input_tokens:
                return "RunPod qualification request exceeds its pinned prompt context"
            if request.timeout_ms != matching_contracts[0].timeout_ms:
                return "RunPod qualification request timeout does not match the pinned contract"
            try:
                generation_config_sha256 = exact_bytes_sha256(
                    canonical_json_bytes(request.generation_config)
                )
            except (TypeError, ValueError):
                return "RunPod qualification request generation configuration is not canonical"
            if generation_config_sha256 != matching_contracts[0].generation_config_sha256:
                return (
                    "RunPod qualification request generation configuration does not match "
                    "the pinned contract"
                )
        return None

    def _request_document(self, request: VisionInferenceRequest) -> dict[str, object]:
        plan = request.input_plan
        assert plan is not None
        items = _selected_items(request)
        generation_config = dict(request.generation_config)
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
                "generation_config": generation_config,
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
        latency_ms: int | None = None,
        qualification_timing_recorded: bool = False,
    ) -> VisionInferenceFailure:
        resolved_usage = usage or _usage(request, None)
        resolved_latency_ms = (
            latency_ms if latency_ms is not None else _elapsed_ms(started, self._monotonic())
        )
        if (
            not qualification_timing_recorded
            and code not in _QUALIFICATION_PRE_DISPATCH_FAILURE_CODES
        ):
            self._record_provider_timing(
                request,
                provider_queue_ms=None,
                provider_execution_ms=None,
                time_to_first_token_ms=None,
                end_to_end_ms=resolved_latency_ms,
                input_tokens=None,
                input_tokens_known=False,
                output_tokens=resolved_usage.output_tokens,
                output_tokens_known=resolved_usage.output_tokens is not None,
                accepted=False,
            )
        return VisionInferenceFailure(
            status=status,
            provider_request_id=provider_request_id,
            provider=request.provider,
            model_name=request.model_name,
            model_version=request.model_version,
            raw_output_artifact_id=raw_output_artifact_id,
            schema_valid=False,
            usage=resolved_usage,
            latency_ms=resolved_latency_ms,
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


def _qualification_request_measurement(
    request: VisionInferenceRequest,
) -> tuple[str, int, int]:
    """Return immutable identity and image facts for one terminal request."""

    plan = request.input_plan
    if plan is None:
        raise ValueError("qualification request lacks an immutable input plan")
    part_ordinal = request.input_plan_part_ordinal or 0
    items = _selected_items(request)
    provider_image_count = sum(item.artifact.media_type.startswith("image/") for item in items)
    return (
        request.logical_invocation_id,
        part_ordinal,
        provider_image_count,
    )


def _native_batch_compatibility_key(request: VisionInferenceRequest) -> tuple[object, ...]:
    """Keep direct adapter callers from mixing task, shape, or deadline groups."""

    return (
        request.provider,
        request.model_name,
        request.model_version,
        request.task.value,
        request.model_policy_version,
        request.output_schema.sha256,
        request.timeout_ms,
        tuple(
            (
                item.artifact.media_type,
                item.artifact.encoding,
                item.artifact.width,
                item.artifact.height,
            )
            for item in _selected_items(request)
        ),
    )


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


def _decode_batch_envelope(data: bytes) -> _RunPodBatchResponseEnvelope:
    """Decode the opt-in batch envelope with the same strict JSON rules as v1."""

    if not data:
        raise _RunPodEnvelopeError(
            "RUNPOD_BATCH_RESPONSE_INVALID_JSON",
            "RunPod batch response must be nonempty JSON bytes",
        )
    if data.startswith(b"\xef\xbb\xbf"):
        raise _RunPodEnvelopeError(
            "RUNPOD_BATCH_RESPONSE_INVALID_UTF8",
            "RunPod batch response must not contain a UTF-8 BOM",
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_BATCH_RESPONSE_INVALID_UTF8",
            "RunPod batch response is not strict UTF-8",
        ) from exc
    try:
        document = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except _DuplicateJsonKeyError as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_BATCH_RESPONSE_DUPLICATE_JSON_KEY",
            "RunPod batch response contains a duplicate JSON object key",
        ) from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_BATCH_RESPONSE_INVALID_JSON",
            "RunPod batch response is not strict JSON",
        ) from exc
    if not isinstance(document, dict):
        raise _RunPodEnvelopeError(
            "RUNPOD_BATCH_RESPONSE_INVALID_JSON",
            "RunPod batch response root must be a JSON object",
        )
    try:
        return _RunPodBatchResponseEnvelope.model_validate_json(
            canonical_json_bytes(document),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise _RunPodEnvelopeError(
            "RUNPOD_BATCH_RESPONSE_INVALID_CONTRACT",
            "RunPod batch response does not satisfy the pinned worker envelope",
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
    "RUNPOD_BATCH_REQUEST_CONTRACT_VERSION",
    "RUNPOD_BATCH_RESPONSE_CONTRACT_VERSION",
    "RUNPOD_QUALIFICATION_RUN_NAMESPACE_METADATA_KEY",
    "RUNPOD_QUALIFICATION_SESSION_METADATA_KEY",
    "RUNPOD_REQUEST_CONTRACT_VERSION",
    "RUNPOD_RESPONSE_CONTRACT_VERSION",
    "RecordedRunPodExchange",
    "RecordedRunPodTransport",
    "RunPodApiKey",
    "RunPodDeploymentConfiguration",
    "RunPodEndpointConfig",
    "RunPodHttpRequest",
    "RunPodHttpResponse",
    "RunPodNativeBatchQualification",
    "RunPodRetryPolicy",
    "RunPodTransport",
    "RunPodTransportError",
    "RunPodVisionAdapter",
    "StdlibRunPodTransport",
]
