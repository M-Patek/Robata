"""Provider-neutral Mage video observation adapter.

The adapter turns one durable, causal perception context into a single native-codec
Mage request. The model emits only compact selected-camera semantics; Robata
constructs the six-camera observation, provenance, gate state, and timestamps
deterministically. This prevents business-oriented fields from multiplying the
visual-generation surface while keeping the resulting observation replayable.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Final, Literal, NoReturn, Protocol, Self

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import (
    INT64_MAX,
    INT64_MIN,
    NanosecondInterval,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.perception_stream import (
    ActorObservation,
    BoundaryAssessment,
    CameraEvidenceRelation,
    CameraObservationEvidence,
    CanonicalToken,
    CognitionGateSignal,
    MageActionObservation,
    MageObservation,
    NonEmptyString,
    ObjectObservation,
    PerceptionContextManifest,
    SemanticCameraQa,
    SemanticQaDisposition,
    SemanticQaIssue,
    UnitInterval,
    create_mage_observation,
)
from robata.inference.mage_video_endpoint import (
    MAGE_VIDEO_ENDPOINT_RESPONSE_VERSION,
    MageVideoCameraEncoding,
    MageVideoCodecPolicy,
    MageVideoDecoderRequest,
    MageVideoEndpointRequest,
    MageVideoEndpointResponse,
    MageVideoInferenceIdentity,
    MageVideoModelIdentity,
    MageVideoResultArtifactDocument,
    MageVideoResultArtifactReference,
    build_mage_video_codec_policy_identity,
    build_mage_video_context_manifest,
    build_mage_video_inference_identity,
    build_mage_video_segment_manifest,
    mage_video_input_manifest_sha256,
)
from robata.perception.pipeline import MageObservationProvider, PerceptionArtifactSink

MAGE_VIDEO_INFER_PATH: Final = "/v2/mage-video/infer"
MAGE_VIDEO_OBSERVATION_REQUEST_IDENTITY_VERSION: Final = "mage-video-observation-request-v6"
MAGE_VIDEO_OBSERVATION_SEGMENT_IDENTITY_VERSION: Final = "mage-video-observation-segment-v2"
MAGE_VIDEO_OBSERVATION_IDEMPOTENCY_NAMESPACE: Final = "mage-video-observation-idempotency-v6"
MAGE_VIDEO_UNIFIED_OBSERVATION_PROMPT_CONTRACT_VERSION: Final = (
    "mage-video-unified-observation-prompt-contract-v6"
)
MAGE_VIDEO_ACCEPTED_BINDING_VERSION: Final = "mage-video-accepted-binding-v1"
MAGE_VIDEO_ACCEPTED_BINDING_KEY_NAMESPACE: Final = "mage-video-accepted-binding-v1"

_FORBIDDEN_COMPACT_PAYLOAD_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "camera_evidence",
        "camera_id",
        "cognition_gate",
        "created_at",
        "segment_hashes",
        "segment_semantic_sha256_values",
    }
)

PositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictInt64 = Annotated[int, Field(strict=True, ge=INT64_MIN, le=INT64_MAX)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0.0, allow_inf_nan=False)]
DurablePath = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16_384)]
DurablePrompt = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=1_000_000)]

_CANONICAL_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_CANONICAL_SIGNED_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")


class MageVideoObservationAdapterError(ValueError):
    """The adapter could not establish a complete, durable observation."""


class MageVideoObservationDiagnostic(StrictModel):
    """Non-canonical, exact-artifact-bound handling of one compact action anomaly.

    Diagnostics deliberately live outside :class:`MageObservation`: they never alter a
    canonical fact or its identity.  They retain enough context to reproduce an
    adapter decision directly from the durable raw endpoint artifact.
    """

    schema_version: Literal["1.0"] = "1.0"
    code: Literal[
        "ACTION_INTERVAL_CLIPPED_TO_CONTEXT",
        "ACTION_INTERVAL_REJECTED_OUTSIDE_CONTEXT",
    ]
    context_manifest_semantic_sha256: Sha256Digest
    inference_artifact_exact_sha256: Sha256Digest
    action_ordinal: PositiveInt
    local_ref: NonEmptyString
    reported_interval: NanosecondInterval
    retained_interval: NanosecondInterval | None = None
    detail: NonEmptyString


class MageVideoObservationDiagnosticSink(Protocol):
    """Optional durable sidecar sink for non-canonical adapter diagnostics."""

    def record(self, diagnostic: MageVideoObservationDiagnostic) -> None: ...


class MageVideoDurableCameraSegment(StrictModel):
    """One selected camera's executable native-video representation."""

    camera_id: CameraId
    segment_semantic_sha256_values: tuple[Sha256Digest, ...]
    codec_stream_exact_sha256: Sha256Digest
    durable_path: DurablePath
    media_type: NonEmptyString = "video/mp4"
    content_sha256: Sha256Digest
    byte_count: PositiveInt

    @model_validator(mode="after")
    def validate_durable_segment(self) -> Self:
        if not self.segment_semantic_sha256_values:
            raise ValueError("durable camera segment requires storage segment lineage")
        if not self.media_type.lower().startswith("video/"):
            raise ValueError("durable camera segment media_type must identify video")
        if self.content_sha256 != self.codec_stream_exact_sha256:
            raise ValueError(
                "durable camera segment content_sha256 must equal codec_stream_exact_sha256"
            )
        return self


class MageVideoDurableSegmentResolver(Protocol):
    """Resolve durable native-video representations for one selected camera."""

    def resolve(
        self,
        *,
        context: PerceptionContextManifest,
        camera_id: CameraId,
    ) -> Sequence[MageVideoDurableCameraSegment]: ...


@dataclass(frozen=True, slots=True)
class MageVideoObservationTransportRequest:
    """Exact request bytes and binding sent to the v2 endpoint path."""

    endpoint_path: str
    request: MageVideoEndpointRequest
    request_body: bytes
    idempotency_key: str


class MageVideoInferenceTransport(Protocol):
    """Injectable transport for the versioned Mage video endpoint."""

    def infer(
        self,
        invocation: MageVideoObservationTransportRequest,
    ) -> MageVideoEndpointResponse: ...


class MageVideoEndpointLoopback(Protocol):
    """The endpoint method used by the no-HTTP loopback transport."""

    def infer_idempotently(
        self,
        *,
        request: MageVideoEndpointRequest,
        idempotency_key: str,
        request_body: bytes,
    ) -> MageVideoEndpointResponse: ...


class MageVideoEndpointLoopbackTransport:
    """Call the local endpoint service while preserving its v2 wire binding."""

    def __init__(self, endpoint: MageVideoEndpointLoopback) -> None:
        self._endpoint = endpoint

    def infer(self, invocation: MageVideoObservationTransportRequest) -> MageVideoEndpointResponse:
        if invocation.endpoint_path != MAGE_VIDEO_INFER_PATH:
            raise MageVideoObservationAdapterError(
                "Mage video transport path is not /v2/mage-video/infer"
            )
        expected_body = canonical_json_bytes(invocation.request.model_dump(mode="json"))
        if invocation.request_body != expected_body:
            raise MageVideoObservationAdapterError(
                "Mage video transport request bytes are not canonical endpoint bytes"
            )
        return self._endpoint.infer_idempotently(
            request=invocation.request,
            idempotency_key=invocation.idempotency_key,
            request_body=invocation.request_body,
        )


class MageVideoResultArtifactReader(Protocol):
    """Read exact durable endpoint-result bytes for parsing or replay."""

    def read(self, reference: MageVideoResultArtifactReference) -> bytes: ...


class FileMageVideoResultArtifactReader:
    """Read a result artifact reference without re-running any model work."""

    def read(self, reference: MageVideoResultArtifactReference) -> bytes:
        try:
            return Path(reference.durable_path).expanduser().resolve(strict=True).read_bytes()
        except (OSError, RuntimeError) as error:
            raise MageVideoObservationAdapterError(
                "Mage video durable result artifact could not be read"
            ) from error


class MageVideoCompactIntervalPayload(StrictModel):
    """Model-authored action bounds in either absolute ns or context-relative seconds."""

    start_ns: StrictInt64 | None = None
    end_ns: StrictInt64 | None = None
    start_offset_seconds: NonNegativeFiniteFloat | None = None
    end_offset_seconds: NonNegativeFiniteFloat | None = None

    @model_validator(mode="after")
    def validate_one_coordinate_system(self) -> Self:
        absolute_present = self.start_ns is not None or self.end_ns is not None
        relative_present = (
            self.start_offset_seconds is not None or self.end_offset_seconds is not None
        )
        if absolute_present == relative_present:
            raise ValueError(
                "interval must contain exactly one complete absolute-ns or relative-seconds pair"
            )
        if absolute_present and (self.start_ns is None or self.end_ns is None):
            raise ValueError("absolute interval requires start_ns and end_ns")
        if relative_present and (
            self.start_offset_seconds is None or self.end_offset_seconds is None
        ):
            raise ValueError(
                "relative interval requires start_offset_seconds and end_offset_seconds"
            )
        return self


class MageVideoCompactBoundaryPayload(StrictModel):
    """Optional compact boundary assessment for one emitted action."""

    start_confidence: UnitInterval
    end_confidence: UnitInterval
    started_before_context: bool = False
    continues_after_context: bool = False


class MageVideoSelectedCameraQaPayload(StrictModel):
    """Only selected-camera semantic QA may be authored by the model."""

    disposition: SemanticQaDisposition
    issues: tuple[SemanticQaIssue, ...] = ()
    confidence: UnitInterval | None = None


class MageVideoCompactActionPayload(StrictModel):
    """Compact selected-camera semantics for one potential action observation."""

    local_ref: NonEmptyString | None = None
    action: CanonicalToken
    interval: MageVideoCompactIntervalPayload
    confidence: UnitInterval | None = None
    actor: ActorObservation | None = None
    object: ObjectObservation | None = None
    visibility: UnitInterval | None = None
    boundary: MageVideoCompactBoundaryPayload | None = None


class MageVideoObservationPayload(StrictModel):
    """Compact model JSON; provenance and six-camera expansion remain deterministic."""

    observation_schema_version: SchemaVersion
    selected_camera_qa: MageVideoSelectedCameraQaPayload
    observations: tuple[MageVideoCompactActionPayload, ...] = ()


class MageVideoObservationAdapterConfig(StrictModel):
    """Versioned provider-neutral policy for one native-video observation call."""

    model_family: CanonicalToken = "mage_video"
    observation_schema_version: SchemaVersion = "mage-observation-v1"
    prompt_version: SchemaVersion = "mage-unified-observation-prompt-v6"
    decoder_id: NonEmptyString = "mage-observation-decoder-v2"
    out_of_context_action_policy: Literal["REJECT_ACTION_V1", "CLIP_INTERSECTION_V1"] = (
        "REJECT_ACTION_V1"
    )
    max_new_tokens: PositiveInt = 512
    cognition_gate_policy_version: SchemaVersion = "mage-gate-shadow-v1"
    cognition_gate_threshold: UnitInterval = 0.5


@dataclass(frozen=True, slots=True)
class MageVideoPreparedObservationRequest:
    """Persisted request/inference binding for normal calls and media-free replay."""

    request_identity_sha256: Sha256Digest
    context: PerceptionContextManifest
    selected_segment: MageVideoDurableCameraSegment
    prompt: str
    endpoint_request: MageVideoEndpointRequest
    request_body: bytes
    idempotency_key: str
    inference_identity: MageVideoInferenceIdentity

    @property
    def transport_request(self) -> MageVideoObservationTransportRequest:
        return MageVideoObservationTransportRequest(
            endpoint_path=MAGE_VIDEO_INFER_PATH,
            request=self.endpoint_request,
            request_body=self.request_body,
            idempotency_key=self.idempotency_key,
        )


class MageVideoAcceptedObservationBinding(StrictModel):
    """Durable request/result binding that enables media-free accepted-artifact replay."""

    schema_version: Literal["1.0"] = "1.0"
    binding_version: Literal["mage-video-accepted-binding-v1"] = MAGE_VIDEO_ACCEPTED_BINDING_VERSION
    binding_logical_key: NonEmptyString
    request_identity_sha256: Sha256Digest
    context: PerceptionContextManifest
    selected_segment: MageVideoDurableCameraSegment
    prompt: DurablePrompt
    endpoint_path: Literal["/v2/mage-video/infer"] = MAGE_VIDEO_INFER_PATH
    endpoint_request: MageVideoEndpointRequest
    request_body_exact_sha256: Sha256Digest
    idempotency_key: NonEmptyString
    inference_identity: MageVideoInferenceIdentity
    endpoint_response: MageVideoEndpointResponse
    result_artifact_exact_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        expected_key = f"{MAGE_VIDEO_ACCEPTED_BINDING_KEY_NAMESPACE}:{self.request_identity_sha256}"
        request_body = canonical_json_bytes(self.endpoint_request.model_dump(mode="json"))
        request_body_digest = exact_bytes_sha256(request_body)
        if self.binding_logical_key != expected_key:
            raise ValueError("accepted binding logical key is inconsistent")
        if self.request_body_exact_sha256 != request_body_digest:
            raise ValueError("accepted binding request body digest is inconsistent")
        if self.idempotency_key != (
            f"{MAGE_VIDEO_OBSERVATION_IDEMPOTENCY_NAMESPACE}:{request_body_digest}"
        ):
            raise ValueError("accepted binding idempotency key is inconsistent")
        if self.endpoint_request.decoder.prompt != self.prompt:
            raise ValueError("accepted binding prompt differs from endpoint request")
        if (
            self.endpoint_request.context_manifest.context_id != self.context.context_manifest_key
            or self.endpoint_request.context_manifest.context_payload_sha256
            != self.context.context_manifest_semantic_sha256
        ):
            raise ValueError("accepted binding context differs from endpoint request")
        expected_inference_identity = build_mage_video_inference_identity(self.endpoint_request)
        if self.inference_identity != expected_inference_identity:
            raise ValueError("accepted binding inference identity is inconsistent")
        if (
            self.endpoint_response.request_id != self.endpoint_request.request_id
            or self.endpoint_response.inference_identity != self.inference_identity
        ):
            raise ValueError("accepted binding response differs from its request")
        if (
            self.result_artifact_exact_sha256
            != self.endpoint_response.result_artifact.content_sha256
        ):
            raise ValueError("accepted binding result artifact digest is inconsistent")
        encoding = self.endpoint_request.camera_encodings[0].segment_manifest
        if (
            encoding.camera_id != self.selected_segment.camera_id.value
            or encoding.content_sha256 != self.selected_segment.content_sha256
            or encoding.byte_count != self.selected_segment.byte_count
            or encoding.durable_path != self.selected_segment.durable_path
        ):
            raise ValueError("accepted binding selected segment differs from endpoint request")
        return self


@dataclass(frozen=True, slots=True)
class _ResolvedCompactInterval:
    """Raw compact bounds plus their explicit policy outcome."""

    reported_interval: NanosecondInterval
    retained_interval: NanosecondInterval | None
    clipped_start: bool = False
    clipped_end: bool = False


@dataclass(frozen=True, slots=True)
class _ExpandedActionObservations:
    """Strict expanded actions plus non-canonical anomaly diagnostics."""

    actions: tuple[MageActionObservation, ...]
    diagnostics: tuple[MageVideoObservationDiagnostic, ...]


def build_mage_video_unified_observation_prompt(
    *,
    context: PerceptionContextManifest,
    segment: MageVideoDurableCameraSegment,
    config: MageVideoObservationAdapterConfig,
) -> str:
    """Build a small, unbiased output contract for one native-video pass."""

    if segment.camera_id not in CAMERA_IDS:
        raise MageVideoObservationAdapterError("selected durable segment has an unknown camera")
    duration_seconds = _exact_seconds_text(context.context_interval.duration_ns)
    prompt_projection = {
        "prompt_contract_version": MAGE_VIDEO_UNIFIED_OBSERVATION_PROMPT_CONTRACT_VERSION,
        "prompt_version": config.prompt_version,
        "task": (
            "Analyze the supplied native video once and return one JSON object only. "
            "Report visible physical actions; return an empty observations array when none exist."
        ),
        "video_duration_seconds": duration_seconds,
        "response_contract": {
            "root_keys": ["observations", "selected_camera_qa"],
            "observations": {
                "type": "array",
                "item_keys": ["action", "interval", "confidence", "visibility"],
                "action": "short observed physical-action phrase",
                "interval": {
                    "start_offset_seconds": (
                        f"number from 0 inclusive to {duration_seconds} exclusive"
                    ),
                    "end_offset_seconds": (
                        f"number after start and no greater than {duration_seconds}"
                    ),
                },
                "confidence": "optional number from 0 to 1",
                "visibility": "optional number from 0 to 1",
            },
            "selected_camera_qa": {
                "optional": True,
                "disposition": "USABLE|DEGRADED|UNUSABLE|UNKNOWN",
                "confidence": "optional number from 0 to 1",
                "issues": "optional array of {code, detail}",
            },
        },
        "rules": [
            "Use seconds relative to the supplied video start.",
            "Do not include keys that are not listed in the response contract.",
            (
                "Do not include camera identifiers, hashes, timestamps, schema versions, "
                "or gate state."
            ),
            "Omit an action whose full interval cannot be placed inside the supplied video.",
            "Do not wrap the JSON in markdown or prose.",
        ],
    }
    return canonical_json_bytes(prompt_projection).decode("utf-8")


def _exact_seconds_text(duration_ns: int) -> str:
    """Render a positive nanosecond duration as a non-exponential decimal string."""

    seconds = Decimal(duration_ns) / Decimal("1000000000")
    rendered = format(seconds, "f").rstrip("0").rstrip(".")
    return rendered or "0"


def _effective_observation_prompt_version(
    config: MageVideoObservationAdapterConfig,
) -> SchemaVersion:
    """Bind compact interval handling to the durable observation identity."""

    return f"{config.prompt_version}+{config.out_of_context_action_policy.lower()}"


def _build_decoder_request(
    *,
    config: MageVideoObservationAdapterConfig,
    prompt: str,
) -> MageVideoDecoderRequest:
    """Build against both pre-release endpoint decoder shapes.

    The target v2 endpoint explicitly carries prompt/schema versions; an older local
    candidate did not.  The adapter supplies the fields whenever the endpoint contract
    exposes them, keeping the policy binding explicit without weakening either model.
    """

    values: dict[str, object] = {
        "decoder_id": config.decoder_id,
        "prompt": prompt,
        "max_new_tokens": config.max_new_tokens,
    }
    if "prompt_contract_version" in MageVideoDecoderRequest.model_fields:
        values["prompt_contract_version"] = MAGE_VIDEO_UNIFIED_OBSERVATION_PROMPT_CONTRACT_VERSION
    if "observation_schema_version" in MageVideoDecoderRequest.model_fields:
        values["observation_schema_version"] = config.observation_schema_version
    return MageVideoDecoderRequest.model_validate(values, strict=True)


class MageVideoObservationAdapter(MageObservationProvider):
    """Normal-observation provider backed by one v2 native-video call per context."""

    def __init__(
        self,
        *,
        model_identity: MageVideoModelIdentity,
        codec_policy: MageVideoCodecPolicy,
        segment_resolver: MageVideoDurableSegmentResolver,
        transport: MageVideoInferenceTransport,
        artifact_reader: MageVideoResultArtifactReader,
        config: MageVideoObservationAdapterConfig | None = None,
        diagnostic_sink: MageVideoObservationDiagnosticSink | None = None,
        accepted_binding_sink: PerceptionArtifactSink | None = None,
    ) -> None:
        if not isinstance(model_identity, MageVideoModelIdentity):
            raise TypeError("model_identity must be MageVideoModelIdentity")
        if not isinstance(codec_policy, MageVideoCodecPolicy):
            raise TypeError("codec_policy must be MageVideoCodecPolicy")
        if config is not None and not isinstance(config, MageVideoObservationAdapterConfig):
            raise TypeError("config must be MageVideoObservationAdapterConfig or None")
        self._model_identity = model_identity
        self._codec_policy = codec_policy
        self._segment_resolver = segment_resolver
        self._transport = transport
        self._artifact_reader = artifact_reader
        self._config = config or MageVideoObservationAdapterConfig()
        self._diagnostic_sink = diagnostic_sink
        self._accepted_binding_sink = accepted_binding_sink
        self._last_diagnostics: tuple[MageVideoObservationDiagnostic, ...] = ()

    @property
    def last_diagnostics(self) -> tuple[MageVideoObservationDiagnostic, ...]:
        """Non-canonical warnings from the most recently parsed endpoint artifact.

        The raw endpoint artifact is authoritative and remains durable independently of
        this convenience view.  Production callers should provide ``diagnostic_sink``
        when they need a durable sidecar record.
        """

        return self._last_diagnostics

    def prepare_request(
        self,
        context: PerceptionContextManifest,
    ) -> MageVideoPreparedObservationRequest:
        """Build stable endpoint input without invoking transport, model, or replay media reads."""

        if not isinstance(context, PerceptionContextManifest):
            raise TypeError("context must be PerceptionContextManifest")
        selected_segment = self._select_executable_segment(context)
        prompt = build_mage_video_unified_observation_prompt(
            context=context,
            segment=selected_segment,
            config=self._config,
        )
        segment_identity = semantic_sha256(
            {
                "segment_identity_version": MAGE_VIDEO_OBSERVATION_SEGMENT_IDENTITY_VERSION,
                "camera_id": selected_segment.camera_id.value,
                "segment_semantic_sha256_values": list(
                    selected_segment.segment_semantic_sha256_values
                ),
                "codec_stream_exact_sha256": selected_segment.codec_stream_exact_sha256,
                "content_sha256": selected_segment.content_sha256,
                "byte_count": selected_segment.byte_count,
                "media_type": selected_segment.media_type,
            }
        )
        endpoint_segment = build_mage_video_segment_manifest(
            segment_id=f"mage-video-observation-segment-v2:{segment_identity}",
            camera_id=selected_segment.camera_id.value,
            durable_path=selected_segment.durable_path,
            media_type=selected_segment.media_type,
            content_sha256=selected_segment.content_sha256,
            byte_count=selected_segment.byte_count,
        )
        endpoint_context = build_mage_video_context_manifest(
            context_id=context.context_manifest_key,
            context_payload_sha256=context.context_manifest_semantic_sha256,
            segment_manifest_identities=[endpoint_segment.manifest_identity],
        )
        request_identity = semantic_sha256(
            {
                "request_identity_version": MAGE_VIDEO_OBSERVATION_REQUEST_IDENTITY_VERSION,
                "endpoint_path": MAGE_VIDEO_INFER_PATH,
                "context_manifest_semantic_sha256": context.context_manifest_semantic_sha256,
                "segment_manifest_identity": endpoint_segment.manifest_identity,
                "model_identity": self._model_identity.model_dump(mode="json"),
                "codec_policy": self._codec_policy.model_dump(mode="json"),
                "prompt_exact_sha256": exact_bytes_sha256(prompt.encode("utf-8")),
                "decoder_id": self._config.decoder_id,
                "max_new_tokens": self._config.max_new_tokens,
                "observation_schema_version": self._config.observation_schema_version,
                "prompt_version": _effective_observation_prompt_version(self._config),
                "out_of_context_action_policy": self._config.out_of_context_action_policy,
            }
        )
        endpoint_request = MageVideoEndpointRequest(
            request_id=f"mage-video-observation-request-v6:{request_identity}",
            model_identity=self._model_identity,
            codec_policy=self._codec_policy,
            context_manifest=endpoint_context,
            camera_encodings=[
                MageVideoCameraEncoding(
                    encoder_id=f"camera-encoder:{selected_segment.camera_id.value}",
                    segment_manifest=endpoint_segment,
                )
            ],
            decoder=_build_decoder_request(config=self._config, prompt=prompt),
        )
        request_body = canonical_json_bytes(endpoint_request.model_dump(mode="json"))
        request_body_exact_sha256 = exact_bytes_sha256(request_body)
        return MageVideoPreparedObservationRequest(
            request_identity_sha256=request_identity,
            context=context,
            selected_segment=selected_segment,
            prompt=prompt,
            endpoint_request=endpoint_request,
            request_body=request_body,
            idempotency_key=(
                f"{MAGE_VIDEO_OBSERVATION_IDEMPOTENCY_NAMESPACE}:{request_body_exact_sha256}"
            ),
            inference_identity=build_mage_video_inference_identity(endpoint_request),
        )

    def observe(self, context: PerceptionContextManifest) -> MageObservation:
        """Invoke exactly one normal observation request; no shadow signal can suppress it."""

        prepared = self.prepare_request(context)
        response = self._transport.infer(prepared.transport_request)
        if not isinstance(response, MageVideoEndpointResponse):
            raise MageVideoObservationAdapterError(
                "Mage video transport returned an unsupported endpoint response"
            )
        try:
            artifact_bytes = self._artifact_reader.read(response.result_artifact)
        except OSError as error:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact could not be read after inference"
            ) from error
        binding = self.build_accepted_binding(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )
        if self._accepted_binding_sink is not None:
            self._accepted_binding_sink.put(
                kind="accepted-inference-binding",
                logical_key=binding.binding_logical_key,
                payload=canonical_json_bytes(binding),
            )
        return self.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )

    def build_accepted_binding(
        self,
        *,
        prepared: MageVideoPreparedObservationRequest,
        response: MageVideoEndpointResponse,
        artifact_bytes: bytes,
    ) -> MageVideoAcceptedObservationBinding:
        """Verify and freeze one accepted endpoint request/result pair before parsing."""

        if not isinstance(prepared, MageVideoPreparedObservationRequest):
            raise TypeError("prepared must be MageVideoPreparedObservationRequest")
        if not isinstance(response, MageVideoEndpointResponse):
            raise TypeError("response must be MageVideoEndpointResponse")
        self._verify_endpoint_response(prepared, response)
        self._verify_result_artifact(response, artifact_bytes)
        return MageVideoAcceptedObservationBinding(
            binding_logical_key=(
                f"{MAGE_VIDEO_ACCEPTED_BINDING_KEY_NAMESPACE}:{prepared.request_identity_sha256}"
            ),
            request_identity_sha256=prepared.request_identity_sha256,
            context=prepared.context,
            selected_segment=prepared.selected_segment,
            prompt=prepared.prompt,
            endpoint_request=prepared.endpoint_request,
            request_body_exact_sha256=exact_bytes_sha256(prepared.request_body),
            idempotency_key=prepared.idempotency_key,
            inference_identity=prepared.inference_identity,
            endpoint_response=response,
            result_artifact_exact_sha256=exact_bytes_sha256(artifact_bytes),
        )

    def replay_accepted_binding(
        self,
        *,
        binding: MageVideoAcceptedObservationBinding,
        artifact_bytes: bytes,
    ) -> MageObservation:
        """Hydrate a durable accepted binding and replay without media or transport."""

        if not isinstance(binding, MageVideoAcceptedObservationBinding):
            raise TypeError("binding must be MageVideoAcceptedObservationBinding")
        if exact_bytes_sha256(artifact_bytes) != binding.result_artifact_exact_sha256:
            raise MageVideoObservationAdapterError(
                "accepted binding result artifact failed exact-byte verification"
            )
        prepared = MageVideoPreparedObservationRequest(
            request_identity_sha256=binding.request_identity_sha256,
            context=binding.context,
            selected_segment=binding.selected_segment,
            prompt=binding.prompt,
            endpoint_request=binding.endpoint_request,
            request_body=canonical_json_bytes(binding.endpoint_request.model_dump(mode="json")),
            idempotency_key=binding.idempotency_key,
            inference_identity=binding.inference_identity,
        )
        return self.replay_prepared_artifact(
            prepared=prepared,
            response=binding.endpoint_response,
            artifact_bytes=artifact_bytes,
        )

    def replay_prepared_artifact(
        self,
        *,
        prepared: MageVideoPreparedObservationRequest,
        response: MageVideoEndpointResponse,
        artifact_bytes: bytes,
    ) -> MageObservation:
        """Replay a persisted binding and exact artifact without source-media resolution."""

        if not isinstance(prepared, MageVideoPreparedObservationRequest):
            raise TypeError("prepared must be MageVideoPreparedObservationRequest")
        if not isinstance(response, MageVideoEndpointResponse):
            raise TypeError("response must be MageVideoEndpointResponse")
        return self._observation_from_response(
            context=prepared.context,
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )

    def replay_artifact(
        self,
        *,
        context: PerceptionContextManifest,
        response: MageVideoEndpointResponse,
        artifact_bytes: bytes,
    ) -> MageObservation:
        """Convenience replay for callers that can still resolve source-media bindings."""

        return self.replay_prepared_artifact(
            prepared=self.prepare_request(context),
            response=response,
            artifact_bytes=artifact_bytes,
        )

    def _select_executable_segment(
        self,
        context: PerceptionContextManifest,
    ) -> MageVideoDurableCameraSegment:
        selected_cameras = tuple(
            camera_id
            for camera_id in CAMERA_IDS
            if (
                context.cameras[camera_id].available
                and context.cameras[camera_id].selected_for_inference
            )
        )
        if len(selected_cameras) != 1:
            raise MageVideoObservationAdapterError(
                "Mage video v2 currently requires exactly one selected and observable camera"
            )
        selected_camera = selected_cameras[0]
        resolved = tuple(self._segment_resolver.resolve(context=context, camera_id=selected_camera))
        if len(resolved) != 1:
            raise MageVideoObservationAdapterError(
                "Mage video v2 requires exactly one executable selected camera segment"
            )
        segment = resolved[0]
        if not isinstance(segment, MageVideoDurableCameraSegment):
            raise MageVideoObservationAdapterError(
                "durable segment resolver returned an unsupported segment contract"
            )
        binding = context.cameras[selected_camera]
        if segment.camera_id is not selected_camera:
            raise MageVideoObservationAdapterError(
                "durable camera segment does not belong to the selected camera"
            )
        if segment.codec_stream_exact_sha256 != binding.codec_stream_exact_sha256:
            raise MageVideoObservationAdapterError(
                "durable camera segment codec hash does not match perception context"
            )
        if segment.segment_semantic_sha256_values != binding.segment_semantic_sha256_values:
            raise MageVideoObservationAdapterError(
                "durable camera segment lineage does not match perception context"
            )
        expected_lineage = tuple(item.segment_semantic_sha256 for item in context.ordered_segments)
        if segment.segment_semantic_sha256_values != expected_lineage:
            raise MageVideoObservationAdapterError(
                "durable camera segment lineage does not match ordered context segments"
            )
        return segment

    def _observation_from_response(
        self,
        *,
        context: PerceptionContextManifest,
        prepared: MageVideoPreparedObservationRequest,
        response: MageVideoEndpointResponse,
        artifact_bytes: bytes,
    ) -> MageObservation:
        self._verify_endpoint_response(prepared, response)
        artifact = self._verify_result_artifact(response, artifact_bytes)
        return self._parse_observation_payload(
            context=context,
            selected_camera=prepared.selected_segment.camera_id,
            artifact=artifact,
            inference_artifact_exact_sha256=exact_bytes_sha256(artifact_bytes),
        )

    def _verify_endpoint_response(
        self,
        prepared: MageVideoPreparedObservationRequest,
        response: MageVideoEndpointResponse,
    ) -> None:
        if response.contract_version != MAGE_VIDEO_ENDPOINT_RESPONSE_VERSION:
            raise MageVideoObservationAdapterError("Mage video endpoint response version mismatch")
        if response.request_id != prepared.endpoint_request.request_id:
            raise MageVideoObservationAdapterError("Mage video response request identity mismatch")

        expected = prepared.inference_identity
        actual = response.inference_identity
        if actual.model_identity != self._model_identity:
            raise MageVideoObservationAdapterError(
                "Mage video response model identifier, revision, checkpoint, or runtime mismatch"
            )
        if actual.codec_policy_identity != build_mage_video_codec_policy_identity(
            self._codec_policy
        ):
            raise MageVideoObservationAdapterError(
                "Mage video response codec policy identity mismatch"
            )
        if actual.input_manifest_sha256 != mage_video_input_manifest_sha256(
            prepared.endpoint_request
        ):
            raise MageVideoObservationAdapterError(
                "Mage video response context or segment manifest identity mismatch"
            )
        if actual != expected:
            raise MageVideoObservationAdapterError(
                "Mage video response inference identity mismatch"
            )
        if response.camera_encoding_count != len(prepared.endpoint_request.camera_encodings):
            raise MageVideoObservationAdapterError(
                "Mage video response camera encoding count mismatch"
            )
        if response.decoder_id != prepared.endpoint_request.decoder.decoder_id:
            raise MageVideoObservationAdapterError("Mage video response decoder identity mismatch")

    @staticmethod
    def _verify_result_artifact(
        response: MageVideoEndpointResponse,
        artifact_bytes: bytes,
    ) -> MageVideoResultArtifactDocument:
        if not isinstance(artifact_bytes, bytes) or not artifact_bytes:
            raise MageVideoObservationAdapterError("Mage video result artifact bytes are missing")
        reference = response.result_artifact
        if exact_bytes_sha256(artifact_bytes) != reference.content_sha256:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact exact-byte digest mismatch"
            )
        try:
            artifact = MageVideoResultArtifactDocument.model_validate_json(
                artifact_bytes,
                strict=True,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact does not satisfy its strict contract"
            ) from error
        if canonical_json_bytes(artifact.model_dump(mode="json")) != artifact_bytes:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact bytes are not canonical and replayable"
            )
        if artifact.artifact_identity != reference.artifact_identity:
            raise MageVideoObservationAdapterError("Mage video result artifact identity mismatch")
        if artifact.request_id != response.request_id:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact request identity mismatch"
            )
        if artifact.inference_identity != response.inference_identity:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact inference identity mismatch"
            )
        if artifact.camera_encoding_count != response.camera_encoding_count:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact camera encoding count mismatch"
            )
        if artifact.decoder_id != response.decoder_id:
            raise MageVideoObservationAdapterError(
                "Mage video result artifact decoder identity mismatch"
            )
        if (
            artifact.prompt_tokens != response.prompt_tokens
            or artifact.output_tokens != response.output_tokens
            or artifact.load_seconds != response.load_seconds
            or artifact.generation_seconds != response.generation_seconds
            or artifact.execution_device != response.execution_device
            or artifact.output_text != response.output_text
        ):
            raise MageVideoObservationAdapterError(
                "Mage video result artifact response payload mismatch"
            )
        return artifact

    def _parse_observation_payload(
        self,
        *,
        context: PerceptionContextManifest,
        selected_camera: CameraId,
        artifact: MageVideoResultArtifactDocument,
        inference_artifact_exact_sha256: Sha256Digest,
    ) -> MageObservation:
        self._last_diagnostics = ()
        raw_payload = _decode_compact_json_object(artifact.output_text)
        _reject_forbidden_compact_fields(raw_payload)
        compact_payload = _prepare_compact_payload(
            raw_payload,
            observation_schema_version=self._config.observation_schema_version,
        )
        normalized_payload = _normalise_compact_numeric_leaves(compact_payload)
        try:
            payload = MageVideoObservationPayload.model_validate_json(
                canonical_json_bytes(normalized_payload),
                strict=True,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise MageVideoObservationAdapterError(
                "Mage video output is not a strict compact observation payload"
            ) from error
        if payload.observation_schema_version != self._config.observation_schema_version:
            raise MageVideoObservationAdapterError("Mage observation schema version mismatch")

        semantic_qa = _expand_semantic_qa(
            selected_camera=selected_camera,
            selected_qa=payload.selected_camera_qa,
        )
        expanded_actions = _expand_action_observations(
            context=context,
            selected_camera=selected_camera,
            payloads=payload.observations,
            out_of_context_action_policy=self._config.out_of_context_action_policy,
            inference_artifact_exact_sha256=inference_artifact_exact_sha256,
        )
        # StreamMind is not wired into the adapter yet. A decoder sentence must not masquerade
        # as a gate result or control admission; the gate remains shadow and intentionally empty.
        cognition_gate = CognitionGateSignal(
            score=None,
            threshold=self._config.cognition_gate_threshold,
            would_admit=None,
            gate_policy_version=self._config.cognition_gate_policy_version,
        )
        observation = create_mage_observation(
            observation_schema_version=payload.observation_schema_version,
            context=context,
            model_family=self._config.model_family,
            model_revision=self._model_identity.model_revision,
            model_artifact_manifest_sha256=self._model_identity.checkpoint_manifest_sha256,
            prompt_version=_effective_observation_prompt_version(self._config),
            inference_artifact_exact_sha256=inference_artifact_exact_sha256,
            cognition_gate=cognition_gate,
            semantic_qa=semantic_qa,
            observations=expanded_actions.actions,
            created_at=artifact.created_at,
        )
        self._record_diagnostics(expanded_actions.diagnostics)
        return observation

    def _record_diagnostics(
        self,
        diagnostics: tuple[MageVideoObservationDiagnostic, ...],
    ) -> None:
        self._last_diagnostics = diagnostics
        if self._diagnostic_sink is None:
            return
        for diagnostic in diagnostics:
            try:
                self._diagnostic_sink.record(diagnostic)
            except Exception as error:
                raise MageVideoObservationAdapterError(
                    "Mage video observation diagnostic sink rejected a parse diagnostic"
                ) from error


def _expand_semantic_qa(
    *,
    selected_camera: CameraId,
    selected_qa: MageVideoSelectedCameraQaPayload,
) -> SixCameraMap[SemanticCameraQa]:
    values: dict[CameraId, SemanticCameraQa] = {}
    for camera_id in CAMERA_IDS:
        if camera_id is selected_camera:
            values[camera_id] = SemanticCameraQa(
                camera_id=camera_id,
                disposition=selected_qa.disposition,
                issues=selected_qa.issues,
                confidence=selected_qa.confidence,
            )
        else:
            values[camera_id] = SemanticCameraQa(
                camera_id=camera_id,
                disposition=SemanticQaDisposition.UNKNOWN,
                issues=(),
                confidence=None,
            )
    return SixCameraMap[SemanticCameraQa](root=values)


def _expand_action_observations(
    *,
    context: PerceptionContextManifest,
    selected_camera: CameraId,
    payloads: tuple[MageVideoCompactActionPayload, ...],
    out_of_context_action_policy: Literal["REJECT_ACTION_V1", "CLIP_INTERSECTION_V1"],
    inference_artifact_exact_sha256: Sha256Digest,
) -> _ExpandedActionObservations:
    """Expand compact actions without allowing one bad tail interval to abort a stream.

    ``REJECT_ACTION_V1`` is the default: it retains the exact raw artifact but emits no
    action if *any* reported bound lies outside the context.  ``CLIP_INTERSECTION_V1``
    is opt-in and clips only a non-empty intersection, explicitly marks the clipped
    boundary as a continuation, and records a diagnostic.  Neither policy invents
    out-of-context timestamps.
    """

    selected_binding = context.cameras[selected_camera]
    selected_lineage = selected_binding.segment_semantic_sha256_values
    actions: list[MageActionObservation] = []
    diagnostics: list[MageVideoObservationDiagnostic] = []
    local_refs: set[str] = set()
    semantic_action_digests: set[str] = set()
    for ordinal, payload in enumerate(payloads, start=1):
        local_ref = payload.local_ref or f"observation_{ordinal}"
        if local_ref in local_refs:
            raise MageVideoObservationAdapterError(
                "Mage compact observation local_ref values are duplicate after "
                "deterministic assignment"
            )
        local_refs.add(local_ref)
        resolved_interval = _resolve_interval_with_policy(
            payload.interval,
            context_interval=context.context_interval,
            out_of_context_action_policy=out_of_context_action_policy,
        )
        if resolved_interval.retained_interval is None:
            diagnostics.append(
                _interval_diagnostic(
                    code="ACTION_INTERVAL_REJECTED_OUTSIDE_CONTEXT",
                    context=context,
                    inference_artifact_exact_sha256=inference_artifact_exact_sha256,
                    action_ordinal=ordinal,
                    local_ref=local_ref,
                    reported_interval=resolved_interval.reported_interval,
                    retained_interval=None,
                    detail=(
                        "compact action interval is outside the perception context; "
                        "the action was rejected while the exact endpoint artifact remains durable"
                    ),
                )
            )
            continue
        interval = resolved_interval.retained_interval
        if resolved_interval.clipped_start or resolved_interval.clipped_end:
            diagnostics.append(
                _interval_diagnostic(
                    code="ACTION_INTERVAL_CLIPPED_TO_CONTEXT",
                    context=context,
                    inference_artifact_exact_sha256=inference_artifact_exact_sha256,
                    action_ordinal=ordinal,
                    local_ref=local_ref,
                    reported_interval=resolved_interval.reported_interval,
                    retained_interval=interval,
                    detail=(
                        "compact action interval was clipped to the perception context; "
                        "the affected boundary confidence was reset to zero"
                    ),
                )
            )
        boundary = _expand_boundary(
            payload.boundary,
            interval=interval,
            context_interval=context.context_interval,
            clipped_start=resolved_interval.clipped_start,
            clipped_end=resolved_interval.clipped_end,
        )
        action_semantic_digest = _compact_action_semantic_sha256(
            payload=payload,
            interval=interval,
            boundary=boundary,
        )
        if action_semantic_digest in semantic_action_digests:
            raise MageVideoObservationAdapterError(
                "Mage compact observations contain duplicate semantic actions"
            )
        semantic_action_digests.add(action_semantic_digest)
        evidence: dict[CameraId, CameraObservationEvidence] = {}
        for camera_id in CAMERA_IDS:
            if camera_id is selected_camera:
                evidence[camera_id] = CameraObservationEvidence(
                    camera_id=camera_id,
                    relation=CameraEvidenceRelation.SUPPORTS,
                    visibility=payload.visibility,
                    observed_interval=interval,
                    evidence_semantic_sha256_values=selected_lineage,
                )
            else:
                evidence[camera_id] = CameraObservationEvidence(
                    camera_id=camera_id,
                    relation=CameraEvidenceRelation.NOT_OBSERVABLE,
                )
        try:
            action = MageActionObservation(
                local_ref=local_ref,
                action=payload.action,
                interval=interval,
                confidence=payload.confidence,
                actor=payload.actor,
                object=payload.object,
                camera_evidence=SixCameraMap[CameraObservationEvidence](root=evidence),
                boundary=boundary,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise MageVideoObservationAdapterError(
                "Mage compact action cannot be expanded into a strict observation"
            ) from error
        actions.append(action)
    return _ExpandedActionObservations(actions=tuple(actions), diagnostics=tuple(diagnostics))


def _interval_diagnostic(
    *,
    code: Literal[
        "ACTION_INTERVAL_CLIPPED_TO_CONTEXT",
        "ACTION_INTERVAL_REJECTED_OUTSIDE_CONTEXT",
    ],
    context: PerceptionContextManifest,
    inference_artifact_exact_sha256: Sha256Digest,
    action_ordinal: int,
    local_ref: str,
    reported_interval: NanosecondInterval,
    retained_interval: NanosecondInterval | None,
    detail: str,
) -> MageVideoObservationDiagnostic:
    return MageVideoObservationDiagnostic(
        code=code,
        context_manifest_semantic_sha256=context.context_manifest_semantic_sha256,
        inference_artifact_exact_sha256=inference_artifact_exact_sha256,
        action_ordinal=action_ordinal,
        local_ref=local_ref,
        reported_interval=reported_interval,
        retained_interval=retained_interval,
        detail=detail,
    )


def _expand_boundary(
    payload: MageVideoCompactBoundaryPayload | None,
    *,
    interval: NanosecondInterval,
    context_interval: NanosecondInterval,
    clipped_start: bool,
    clipped_end: bool,
) -> BoundaryAssessment:
    """Retain only boundary claims compatible with the retained action interval.

    Boundary data is optional model metadata. A malformed or impossible continuation
    claim must not erase an otherwise valid, exact-artifact-bound action; the invalid
    endpoint is deterministically dropped instead.
    """

    start_confidence = 0.0 if payload is None else payload.start_confidence
    end_confidence = 0.0 if payload is None else payload.end_confidence
    started_before_context = False if payload is None else payload.started_before_context
    continues_after_context = False if payload is None else payload.continues_after_context
    if started_before_context and interval.start_ns != context_interval.start_ns:
        start_confidence = 0.0
        started_before_context = False
    if continues_after_context and interval.end_ns != context_interval.end_ns:
        end_confidence = 0.0
        continues_after_context = False
    if clipped_start:
        start_confidence = 0.0
        started_before_context = True
    if clipped_end:
        end_confidence = 0.0
        continues_after_context = True
    return BoundaryAssessment(
        start_confidence=start_confidence,
        end_confidence=end_confidence,
        started_before_context=started_before_context,
        continues_after_context=continues_after_context,
    )


def _compact_action_semantic_sha256(
    *,
    payload: MageVideoCompactActionPayload,
    interval: NanosecondInterval,
    boundary: BoundaryAssessment,
) -> Sha256Digest:
    return semantic_sha256(
        {
            "action": payload.action,
            "interval": interval.model_dump(mode="json"),
            "actor": None if payload.actor is None else payload.actor.model_dump(mode="json"),
            "object": None if payload.object is None else payload.object.model_dump(mode="json"),
            "boundary_continuation": {
                "started_before_context": boundary.started_before_context,
                "continues_after_context": boundary.continues_after_context,
            },
        }
    )


def _resolve_interval_with_policy(
    payload: MageVideoCompactIntervalPayload,
    *,
    context_interval: NanosecondInterval,
    out_of_context_action_policy: Literal["REJECT_ACTION_V1", "CLIP_INTERSECTION_V1"],
) -> _ResolvedCompactInterval:
    reported_interval = _resolve_reported_interval(payload, context_interval)
    within_context = (
        reported_interval.start_ns >= context_interval.start_ns
        and reported_interval.end_ns <= context_interval.end_ns
    )
    if within_context:
        return _ResolvedCompactInterval(
            reported_interval=reported_interval,
            retained_interval=reported_interval,
        )
    if out_of_context_action_policy == "REJECT_ACTION_V1":
        return _ResolvedCompactInterval(
            reported_interval=reported_interval,
            retained_interval=None,
        )
    retained_start_ns = max(reported_interval.start_ns, context_interval.start_ns)
    retained_end_ns = min(reported_interval.end_ns, context_interval.end_ns)
    if retained_start_ns >= retained_end_ns:
        return _ResolvedCompactInterval(
            reported_interval=reported_interval,
            retained_interval=None,
        )
    return _ResolvedCompactInterval(
        reported_interval=reported_interval,
        retained_interval=NanosecondInterval(
            start_ns=retained_start_ns,
            end_ns=retained_end_ns,
        ),
        clipped_start=reported_interval.start_ns < context_interval.start_ns,
        clipped_end=reported_interval.end_ns > context_interval.end_ns,
    )


def _resolve_reported_interval(
    payload: MageVideoCompactIntervalPayload,
    context_interval: NanosecondInterval,
) -> NanosecondInterval:
    if payload.start_ns is not None and payload.end_ns is not None:
        start_ns = payload.start_ns
        end_ns = payload.end_ns
        duration_ns = context_interval.duration_ns
        if (
            context_interval.start_ns > 0
            and (start_ns < context_interval.start_ns or end_ns > context_interval.end_ns)
            and 0 <= start_ns < end_ns <= duration_ns
        ):
            # Mage may label context-local nanoseconds as start_ns/end_ns.  The raw
            # artifact remains authoritative; project the unambiguous local form onto
            # the absolute recording clock instead of failing the whole stream.
            start_ns += context_interval.start_ns
            end_ns += context_interval.start_ns
    else:
        if payload.start_offset_seconds is None or payload.end_offset_seconds is None:
            raise MageVideoObservationAdapterError("compact interval is incomplete")
        start_ns = context_interval.start_ns + _seconds_to_exact_ns(payload.start_offset_seconds)
        end_ns = context_interval.start_ns + _seconds_to_exact_ns(payload.end_offset_seconds)
    try:
        return NanosecondInterval(start_ns=start_ns, end_ns=end_ns)
    except (TypeError, ValueError, ValidationError) as error:
        raise MageVideoObservationAdapterError(
            "Mage compact interval is not a valid ns interval"
        ) from error


def _seconds_to_exact_ns(value: float) -> int:
    try:
        nanoseconds = Decimal(str(value)) * Decimal("1000000000")
    except (InvalidOperation, ValueError) as error:
        raise MageVideoObservationAdapterError(
            "relative seconds are not a valid decimal"
        ) from error
    if not nanoseconds.is_finite() or nanoseconds != nanoseconds.to_integral_value():
        raise MageVideoObservationAdapterError(
            "relative seconds must resolve to an integral nanosecond bound"
        )
    converted = int(nanoseconds)
    if converted < INT64_MIN or converted > INT64_MAX:
        raise MageVideoObservationAdapterError("relative seconds exceed signed int64 nanoseconds")
    return converted


@dataclass(frozen=True, slots=True)
class _JsonNumber:
    """Preserve numeric source text until an allow-listed compact payload field consumes it."""

    text: str


def _decode_compact_json_object(raw_output: str) -> dict[str, object]:
    if not isinstance(raw_output, str) or not raw_output:
        raise MageVideoObservationAdapterError("Mage video output must be a nonempty JSON object")
    if raw_output.startswith("\ufeff"):
        raise MageVideoObservationAdapterError("Mage video output must not contain a UTF-8 BOM")
    try:
        raw_output.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise MageVideoObservationAdapterError(
            "Mage video output is not representable as strict UTF-8"
        ) from error
    try:
        value = json.loads(
            raw_output,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_int=_JsonNumber,
            parse_float=_JsonNumber,
            parse_constant=_reject_nonfinite_json_number,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise MageVideoObservationAdapterError("Mage video output is not strict JSON") from error
    if type(value) is not dict:
        raise MageVideoObservationAdapterError("Mage video output JSON root must be an object")
    return value


def _reject_forbidden_compact_fields(raw_payload: dict[str, object]) -> None:
    """Reject model-authored provenance before numeric normalization changes error order."""

    for path, key, value in _iter_json_object_keys(raw_payload):
        if key in _FORBIDDEN_COMPACT_PAYLOAD_FIELD_NAMES and value is not None:
            rendered_path = ".".join(str(item) for item in path)
            raise MageVideoObservationAdapterError(
                f"Mage compact output contains forbidden field {rendered_path}"
            )


def _iter_json_object_keys(
    value: object,
    *,
    path: tuple[str | int, ...] = (),
) -> Iterator[tuple[tuple[str | int, ...], str, object]]:
    if isinstance(value, dict):
        for key, item in value.items():
            key_path = (*path, key)
            yield key_path, key, item
            yield from _iter_json_object_keys(item, path=key_path)
    elif isinstance(value, list):
        for ordinal, item in enumerate(value):
            yield from _iter_json_object_keys(item, path=(*path, ordinal))


_CANONICAL_TOKEN_CLEANUP = re.compile(r"[^a-z0-9]+")


def _prepare_compact_payload(
    raw_payload: dict[str, object],
    *,
    observation_schema_version: SchemaVersion,
) -> dict[str, object]:
    """Reduce free-form Mage JSON to the small observation surface Robata consumes.

    The exact raw artifact is already durable.  Optional prose metadata must not make
    an otherwise useful action/interval fail, so this projection keeps only fields that
    affect downstream facts and deterministically normalizes natural-language labels.
    """

    prepared: dict[str, object] = {
        "observation_schema_version": observation_schema_version,
        "selected_camera_qa": {
            "disposition": SemanticQaDisposition.UNKNOWN.value,
            "issues": [],
            "confidence": None,
        },
        "observations": [],
    }
    selected_qa = raw_payload.get("selected_camera_qa")
    if isinstance(selected_qa, dict):
        qa: dict[str, object] = {}
        disposition = selected_qa.get("disposition")
        if isinstance(disposition, str) and disposition in {
            item.value for item in SemanticQaDisposition
        }:
            qa["disposition"] = disposition
        else:
            qa["disposition"] = SemanticQaDisposition.UNKNOWN.value
        qa["issues"] = _sanitize_qa_issues(
            selected_qa.get("issues"),
            disposition=qa["disposition"],
        )
        qa["confidence"] = selected_qa.get("confidence")
        prepared["selected_camera_qa"] = qa

    observations = raw_payload.get("observations")
    if not isinstance(observations, list):
        return prepared
    compact_actions: list[dict[str, object]] = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        action = item.get("action")
        interval = item.get("interval")
        if not isinstance(action, str) or not action.strip() or not isinstance(interval, dict):
            continue
        sanitized_interval = _sanitize_interval_payload(interval)
        if sanitized_interval is None:
            continue
        compact: dict[str, object] = {
            "action": _canonicalize_model_token(action),
            "interval": sanitized_interval,
        }
        local_ref = item.get("local_ref")
        if isinstance(local_ref, str) and local_ref.strip():
            compact["local_ref"] = local_ref.strip()
        for key in ("confidence", "visibility"):
            if key in item:
                compact[key] = item[key]
        actor = _sanitize_actor_payload(item.get("actor"))
        if actor is not None:
            compact["actor"] = actor
        object_payload = _sanitize_object_payload(item.get("object"))
        if object_payload is not None:
            compact["object"] = object_payload
        boundary = _sanitize_boundary_payload(item.get("boundary"))
        if boundary is not None:
            compact["boundary"] = boundary
        compact_actions.append(compact)
    prepared["observations"] = compact_actions
    return prepared


def _sanitize_interval_payload(value: dict[str, object]) -> dict[str, object] | None:
    relative = ("start_offset_seconds", "end_offset_seconds")
    absolute = ("start_ns", "end_ns")
    if all(key in value for key in relative):
        return {key: value[key] for key in relative}
    if all(key in value for key in absolute):
        return {key: value[key] for key in absolute}
    return None


def _sanitize_qa_issues(
    value: object,
    *,
    disposition: object,
) -> list[dict[str, object]]:
    if disposition == SemanticQaDisposition.USABLE.value or not isinstance(value, list):
        return []
    by_code: dict[str, dict[str, object]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_code = item.get("code")
        if not isinstance(raw_code, str) or not raw_code.strip():
            continue
        code = _canonicalize_model_token(raw_code)
        issue: dict[str, object] = {"code": code}
        detail = item.get("detail")
        if isinstance(detail, str) and detail.strip():
            issue["detail"] = detail.strip()
        by_code[code] = issue
    return [by_code[code] for code in sorted(by_code)]


def _canonicalize_model_token(value: str) -> str:
    token = _CANONICAL_TOKEN_CLEANUP.sub("_", value.strip().lower()).strip("_")
    if not token:
        return "observed_action"
    if token[0].isdigit():
        token = f"action_{token}"
    token = token[:128].rstrip("_")
    return token or "observed_action"


def _sanitize_actor_payload(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    actor: dict[str, object] = {}
    hand = value.get("hand")
    if isinstance(hand, str) and hand.strip():
        actor["hand"] = _canonicalize_model_token(hand)
    actor_type = value.get("actor_type")
    if not isinstance(actor_type, str) or not actor_type.strip():
        actor_type = value.get("name")
    if isinstance(actor_type, str) and actor_type.strip():
        actor["actor_type"] = _canonicalize_model_token(actor_type)
    return actor or None


def _sanitize_object_payload(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    object_type = value.get("object_type")
    if not isinstance(object_type, str) or not object_type.strip():
        object_type = value.get("name")
    if not isinstance(object_type, str) or not object_type.strip():
        return None
    identity_hint = value.get("identity_hint")
    if not isinstance(identity_hint, str) or not identity_hint.strip():
        local_ref = value.get("local_ref")
        identity_hint = local_ref if isinstance(local_ref, str) and local_ref.strip() else None
    payload: dict[str, object] = {"object_type": _canonicalize_model_token(object_type)}
    if identity_hint is not None:
        payload["identity_hint"] = identity_hint.strip()
    return payload


def _sanitize_boundary_payload(value: object) -> dict[str, object] | None:
    """Keep only a structurally valid optional boundary assessment.

    Native Mage outputs may use ``boundary`` for scene objects rather than temporal
    continuation metadata. Do not coerce arbitrary truthy values into a boundary fact:
    invalid optional data is omitted and the adapter supplies its zero-confidence
    default instead.
    """

    if not isinstance(value, dict):
        return None
    start_confidence = value.get("start_confidence")
    end_confidence = value.get("end_confidence")
    if not (
        _is_canonical_unit_interval_value(start_confidence)
        and _is_canonical_unit_interval_value(end_confidence)
    ):
        return None
    started_before_context = value.get("started_before_context", False)
    continues_after_context = value.get("continues_after_context", False)
    if type(started_before_context) is not bool or type(continues_after_context) is not bool:
        return None
    return {
        "start_confidence": start_confidence,
        "end_confidence": end_confidence,
        "started_before_context": started_before_context,
        "continues_after_context": continues_after_context,
    }


def _is_canonical_unit_interval_value(value: object) -> bool:
    try:
        parsed = _parse_canonical_float(value, field="boundary confidence")
    except MageVideoObservationAdapterError:
        return False
    return 0.0 <= parsed <= 1.0


def _normalise_compact_numeric_leaves(raw_payload: dict[str, object]) -> dict[str, object]:
    """Coerce only explicitly allowed model numeric leaves from canonical decimal text."""

    normalized = _copy_json_object(raw_payload)
    selected_qa = normalized.get("selected_camera_qa")
    if isinstance(selected_qa, dict):
        _normalise_float_leaf(selected_qa, "confidence")

    observations = normalized.get("observations")
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            _normalise_float_leaf(observation, "confidence")
            _normalise_float_leaf(observation, "visibility")
            interval = observation.get("interval")
            if isinstance(interval, dict):
                _normalise_integer_leaf(interval, "start_ns")
                _normalise_integer_leaf(interval, "end_ns")
                _normalise_float_leaf(interval, "start_offset_seconds")
                _normalise_float_leaf(interval, "end_offset_seconds")
            boundary = observation.get("boundary")
            if isinstance(boundary, dict):
                _normalise_float_leaf(boundary, "start_confidence")
                _normalise_float_leaf(boundary, "end_confidence")

    _reject_unconsumed_json_numbers(normalized, path=())
    return normalized


def _copy_json_object(value: dict[str, object]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key, item in value.items():
        copied[key] = _copy_json_value(item)
    return copied


def _copy_json_value(value: object) -> object:
    if isinstance(value, dict):
        return _copy_json_object(value)
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value


def _normalise_float_leaf(container: dict[str, object], key: str) -> None:
    if key not in container or container[key] is None:
        return
    container[key] = _parse_canonical_float(container[key], field=key)


def _normalise_integer_leaf(container: dict[str, object], key: str) -> None:
    if key not in container or container[key] is None:
        return
    container[key] = _parse_canonical_integer(container[key], field=key)


def _parse_canonical_float(value: object, *, field: str) -> float:
    text = _numeric_text(value, field=field)
    if _CANONICAL_DECIMAL.fullmatch(text) is None:
        raise MageVideoObservationAdapterError(
            f"{field} must be a canonical unsigned decimal number"
        )
    try:
        decimal_value = Decimal(text)
        parsed = float(decimal_value)
    except (InvalidOperation, OverflowError, ValueError) as error:
        raise MageVideoObservationAdapterError(
            f"{field} must be a finite decimal number"
        ) from error
    if not decimal_value.is_finite() or not math.isfinite(parsed):
        raise MageVideoObservationAdapterError(f"{field} must be a finite decimal number")
    return parsed


def _parse_canonical_integer(value: object, *, field: str) -> int:
    text = _numeric_text(value, field=field)
    if _CANONICAL_SIGNED_INTEGER.fullmatch(text) is None:
        raise MageVideoObservationAdapterError(f"{field} must be a canonical signed integer number")
    parsed = int(text)
    if parsed < INT64_MIN or parsed > INT64_MAX:
        raise MageVideoObservationAdapterError(f"{field} exceeds signed int64")
    return parsed


def _numeric_text(value: object, *, field: str) -> str:
    if isinstance(value, _JsonNumber):
        return value.text
    if isinstance(value, str):
        return value
    raise MageVideoObservationAdapterError(
        f"{field} must be a JSON number or a quoted canonical decimal string"
    )


def _reject_unconsumed_json_numbers(value: object, *, path: tuple[str | int, ...]) -> None:
    if isinstance(value, _JsonNumber):
        rendered_path = ".".join(str(item) for item in path) or "<root>"
        raise MageVideoObservationAdapterError(
            f"numeric model output is not permitted at {rendered_path}"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_unconsumed_json_numbers(item, path=(*path, key))
    elif isinstance(value, list):
        for ordinal, item in enumerate(value):
            _reject_unconsumed_json_numbers(item, path=(*path, ordinal))


def _reject_duplicate_json_keys(items: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in items:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key}")
        parsed[key] = value
    return parsed


def _reject_nonfinite_json_number(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


__all__ = [
    "MAGE_VIDEO_ACCEPTED_BINDING_KEY_NAMESPACE",
    "MAGE_VIDEO_ACCEPTED_BINDING_VERSION",
    "MAGE_VIDEO_INFER_PATH",
    "MAGE_VIDEO_OBSERVATION_IDEMPOTENCY_NAMESPACE",
    "MAGE_VIDEO_OBSERVATION_REQUEST_IDENTITY_VERSION",
    "MAGE_VIDEO_OBSERVATION_SEGMENT_IDENTITY_VERSION",
    "MAGE_VIDEO_UNIFIED_OBSERVATION_PROMPT_CONTRACT_VERSION",
    "FileMageVideoResultArtifactReader",
    "MageVideoAcceptedObservationBinding",
    "MageVideoCompactActionPayload",
    "MageVideoCompactBoundaryPayload",
    "MageVideoCompactIntervalPayload",
    "MageVideoDurableCameraSegment",
    "MageVideoDurableSegmentResolver",
    "MageVideoEndpointLoopback",
    "MageVideoEndpointLoopbackTransport",
    "MageVideoInferenceTransport",
    "MageVideoObservationAdapter",
    "MageVideoObservationAdapterConfig",
    "MageVideoObservationAdapterError",
    "MageVideoObservationDiagnostic",
    "MageVideoObservationDiagnosticSink",
    "MageVideoObservationPayload",
    "MageVideoObservationTransportRequest",
    "MageVideoPreparedObservationRequest",
    "MageVideoResultArtifactReader",
    "MageVideoSelectedCameraQaPayload",
    "build_mage_video_unified_observation_prompt",
]
