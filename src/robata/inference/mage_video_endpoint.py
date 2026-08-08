"""Versioned loopback endpoint for native Mage video/codec inference.

The v2 wire contract accepts durable video segment references and semantic
segment/context manifests.  The prior v1 endpoint names were an unreleased
pre-release draft: no v1 reader, route, or idempotency table is retained.
The endpoint intentionally does not accept image payloads, base64 video
bodies, model hidden state, attention KV, or recurrent state.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sqlite3
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol
from uuid import uuid4

from pydantic import Field, StringConstraints, ValidationError, field_validator, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.inference.mage_video_runtime import (
    MAGE_VIDEO_RUNTIME_IDENTITY_VERSION,
    MageVideoLoadProfile,
    MageVideoRuntimeError,
    MageVideoRuntimeIdentity,
)

MAGE_VIDEO_ENDPOINT_REQUEST_VERSION: Literal["mage-video-codec-request-v2"] = (
    "mage-video-codec-request-v2"
)
MAGE_VIDEO_ENDPOINT_RESPONSE_VERSION: Literal["mage-video-codec-response-v2"] = (
    "mage-video-codec-response-v2"
)
MAGE_VIDEO_MODEL_IDENTITY_VERSION: Literal["mage-video-model-identity-v2"] = (
    "mage-video-model-identity-v2"
)
MAGE_VIDEO_CODEC_POLICY_VERSION: Literal["mage-video-codec-policy-v2"] = (
    "mage-video-codec-policy-v2"
)
MAGE_VIDEO_SEGMENT_MANIFEST_VERSION: Literal["mage-video-segment-manifest-v1"] = (
    "mage-video-segment-manifest-v1"
)
MAGE_VIDEO_CONTEXT_MANIFEST_VERSION: Literal["mage-video-context-manifest-v1"] = (
    "mage-video-context-manifest-v1"
)
MAGE_VIDEO_INFERENCE_IDENTITY_VERSION: Literal["mage-video-inference-identity-v2"] = (
    "mage-video-inference-identity-v2"
)
MAGE_VIDEO_RESULT_ARTIFACT_VERSION: Literal["mage-video-result-artifact-v2"] = (
    "mage-video-result-artifact-v2"
)
MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER: Final = "Idempotency-Key"
MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_POLICY_VERSION: Final = "mage-video-idempotency-policy-v2"
_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_TABLE: Final = "mage_video_endpoint_idempotency_v2"
_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_PENDING: Final = "PENDING"
_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_COMPLETE: Final = "COMPLETE"
_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_LEASE_SECONDS: Final = 7_200.0
_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_WAIT_TIMEOUT_SECONDS: Final = 7_260.0
_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_POLL_SECONDS: Final = 0.025
_LOGGER = logging.getLogger(__name__)

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
DurablePath = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=16384)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
PreprocessDevice = Literal["cpu", "cuda"]


class MageVideoRuntimeIdentityBinding(StrictModel):
    """JSON-safe v2 binding for the resident runtime's versioned load profile."""

    identity_version: Literal["mage-video-runtime-identity-v1"] = (
        MAGE_VIDEO_RUNTIME_IDENTITY_VERSION
    )
    load_profile: Literal["native_bf16_v1", "bitsandbytes_4bit_nf4_v1"] = (
        MageVideoLoadProfile.NATIVE_BF16.value
    )

    @classmethod
    def from_runtime_identity(
        cls,
        identity: MageVideoRuntimeIdentity,
    ) -> MageVideoRuntimeIdentityBinding:
        """Convert the runtime's dataclass identity into its strict wire representation."""

        if not isinstance(identity, MageVideoRuntimeIdentity):
            raise TypeError("identity must be MageVideoRuntimeIdentity")
        if identity.identity_version != MAGE_VIDEO_RUNTIME_IDENTITY_VERSION:
            raise ValueError("identity has an unsupported Mage video runtime identity version")
        if identity.load_profile is MageVideoLoadProfile.NATIVE_BF16:
            return cls(load_profile="native_bf16_v1")
        if identity.load_profile is MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4:
            return cls(load_profile="bitsandbytes_4bit_nf4_v1")
        raise ValueError("identity has an unsupported Mage video load profile")

    def to_runtime_identity(self) -> MageVideoRuntimeIdentity:
        """Return the exact runtime identity used for resident-profile comparison."""

        return MageVideoRuntimeIdentity(
            identity_version=self.identity_version,
            load_profile=MageVideoLoadProfile(self.load_profile),
        )


class MageVideoModelIdentity(StrictModel):
    """Immutable model/checkpoint and resident-runtime binding selected by the caller."""

    identity_version: Literal["mage-video-model-identity-v2"] = MAGE_VIDEO_MODEL_IDENTITY_VERSION
    model_identifier: NonEmptyString
    model_revision: NonEmptyString
    checkpoint_manifest_sha256: Sha256Digest
    runtime_identity: MageVideoRuntimeIdentityBinding = Field(
        default_factory=MageVideoRuntimeIdentityBinding
    )

    @field_validator("runtime_identity", mode="before")
    @classmethod
    def normalise_runtime_identity_binding(
        cls,
        value: object,
    ) -> object:
        """Accept the runtime dataclass in local composition, but serialize only JSON fields."""

        if isinstance(value, MageVideoRuntimeIdentity):
            return MageVideoRuntimeIdentityBinding.from_runtime_identity(value)
        return value

    @model_validator(mode="after")
    def validate_runtime_identity(self) -> MageVideoModelIdentity:
        """Keep the model identity tied to the supported runtime identity family."""

        if self.runtime_identity.identity_version != MAGE_VIDEO_RUNTIME_IDENTITY_VERSION:
            raise ValueError(
                "runtime_identity.identity_version is not a supported Mage video runtime identity"
            )
        return self


class MageVideoNeuralCodecParameters(StrictModel):
    """Stable, provider-neutral neural codec controls for the native processor."""

    quantization_parameter: Annotated[int, Field(strict=True, ge=0, le=63)] = 42
    reset_interval: PositiveInt = 64
    intra_period: Annotated[int, Field(strict=True, ge=-1)] = -1
    max_side: NonNegativeInt = 0
    sequence_length_frames: NonNegativeInt = 0
    canvas_token_side: PositiveInt | None = None
    readiness_coverage_bins: PositiveInt = 3
    readiness_delta_ratio: PositiveFloat = 0.05
    bitcost_percentile: Annotated[int, Field(strict=True, ge=1, le=100)] = 99
    decode_backsearch_max: PositiveInt = 16


class MageVideoCodecPolicy(StrictModel):
    """Codec preprocessing policy, independently hashable from model identity."""

    policy_version: Literal["mage-video-codec-policy-v2"] = MAGE_VIDEO_CODEC_POLICY_VERSION
    codec_mode: Literal["traditional", "neural"] = "traditional"
    # Required: policy callers must bind CPU vs CUDA preprocessing explicitly.
    preprocess_device: PreprocessDevice
    target_canvas: PositiveInt = 32
    group_size: PositiveInt = 32
    images_per_group: PositiveInt = 4
    patch_size: PositiveInt = 16
    max_pixels: PositiveInt = 150_000
    min_group_frames: PositiveInt = 8
    max_group_frames: PositiveInt = 64
    timeout_seconds: PositiveInt = 7_200
    neural_parameters: MageVideoNeuralCodecParameters | None = None

    @model_validator(mode="after")
    def validate_shape_and_mode(self) -> MageVideoCodecPolicy:
        if self.target_canvas % self.images_per_group != 0:
            raise ValueError("target_canvas must be divisible by images_per_group")
        if self.group_size % self.images_per_group != 0:
            raise ValueError("group_size must be divisible by images_per_group")
        if self.max_group_frames < self.min_group_frames:
            raise ValueError("max_group_frames must be greater than or equal to min_group_frames")
        if self.codec_mode == "traditional" and self.neural_parameters is not None:
            raise ValueError("traditional codec_mode must not include neural_parameters")
        return self

    def native_codec_config(self) -> dict[str, Any]:
        """Translate neutral policy fields to Mage's native processor knobs."""

        config: dict[str, Any] = {
            "engine": "hevc" if self.codec_mode == "traditional" else "dcvc-rt",
            "preprocess_device": self.preprocess_device,
            "target_canvas": self.target_canvas,
            "group_size": self.group_size,
            "images_per_group": self.images_per_group,
            "patch": self.patch_size,
            "max_pixels": self.max_pixels,
            "min_group_frames": self.min_group_frames,
            "max_group_frames": self.max_group_frames,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.codec_mode == "neural":
            parameters = self.neural_parameters or MageVideoNeuralCodecParameters()
            neural_config: dict[str, Any] = {
                # DCVC must receive the policy-selected device; the runtime never
                # substitutes the decoder model device.
                "device": self.preprocess_device,
                "qp": parameters.quantization_parameter,
                "reset_interval": parameters.reset_interval,
                "intra_period": parameters.intra_period,
                "max_side": parameters.max_side,
                "seq_len_frames": parameters.sequence_length_frames,
                "readiness_coverage_bins": parameters.readiness_coverage_bins,
                "readiness_delta_ratio": parameters.readiness_delta_ratio,
                "bitcost_pct": parameters.bitcost_percentile,
                "decode_backsearch_max": parameters.decode_backsearch_max,
            }
            if parameters.canvas_token_side is not None:
                neural_config["canvas_token_side"] = parameters.canvas_token_side
            config["dcvc"] = neural_config
        return config


class MageVideoCodecPolicyIdentity(StrictModel):
    """Canonical identity for an immutable codec policy."""

    policy_version: Literal["mage-video-codec-policy-v2"] = MAGE_VIDEO_CODEC_POLICY_VERSION
    codec_mode: Literal["traditional", "neural"]
    preprocess_device: PreprocessDevice
    policy_sha256: Sha256Digest


class MageVideoSegmentManifest(StrictModel):
    """One durable video segment and the manifest identity that binds it."""

    manifest_version: Literal["mage-video-segment-manifest-v1"] = (
        MAGE_VIDEO_SEGMENT_MANIFEST_VERSION
    )
    segment_id: NonEmptyString
    camera_id: NonEmptyString
    durable_path: DurablePath
    media_type: NonEmptyString = "video/mp4"
    content_sha256: Sha256Digest
    byte_count: PositiveInt
    manifest_identity: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> MageVideoSegmentManifest:
        if not self.media_type.lower().startswith("video/"):
            raise ValueError("segment media_type must identify a video")
        expected = semantic_sha256(_segment_manifest_projection(self))
        if self.manifest_identity != expected:
            raise ValueError("segment manifest_identity does not match its canonical manifest")
        return self


class MageVideoContextManifest(StrictModel):
    """Durable context identity bound to ordered segment manifests."""

    manifest_version: Literal["mage-video-context-manifest-v1"] = (
        MAGE_VIDEO_CONTEXT_MANIFEST_VERSION
    )
    context_id: NonEmptyString
    context_payload_sha256: Sha256Digest
    segment_manifest_identities: Annotated[list[Sha256Digest], Field(min_length=1)]
    manifest_identity: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest_identity(self) -> MageVideoContextManifest:
        expected = semantic_sha256(_context_manifest_projection(self))
        if self.manifest_identity != expected:
            raise ValueError("context manifest_identity does not match its canonical manifest")
        return self


class MageVideoCameraEncoding(StrictModel):
    """One independently encoded camera segment for a shared decoder request."""

    encoder_id: NonEmptyString
    segment_manifest: MageVideoSegmentManifest


class MageVideoDecoderRequest(StrictModel):
    """Single decoder instruction, intentionally separate from camera encodings.

    All fields form part of the v2 inference identity: the prompt and generation
    budget cannot be silently changed while preserving an inference identity.
    """

    decoder_id: NonEmptyString
    prompt: NonEmptyString
    max_new_tokens: Annotated[int, Field(strict=True, ge=1, le=4096)]


class MageVideoEndpointRequest(StrictModel):
    """v2 request for one encoded video path and one decoder.

    ``camera_encodings`` is a list to reserve the future shape of many
    independent camera encoders feeding one decoder. The v2 maximum is one.
    """

    contract_version: Literal["mage-video-codec-request-v2"] = MAGE_VIDEO_ENDPOINT_REQUEST_VERSION
    request_id: NonEmptyString
    model_identity: MageVideoModelIdentity
    codec_policy: MageVideoCodecPolicy
    context_manifest: MageVideoContextManifest
    camera_encodings: Annotated[list[MageVideoCameraEncoding], Field(min_length=1, max_length=1)]
    decoder: MageVideoDecoderRequest

    @model_validator(mode="after")
    def validate_context_segment_binding(self) -> MageVideoEndpointRequest:
        segment_identities = tuple(
            encoding.segment_manifest.manifest_identity for encoding in self.camera_encodings
        )
        if tuple(self.context_manifest.segment_manifest_identities) != segment_identities:
            raise ValueError(
                "context_manifest.segment_manifest_identities must match camera_encodings in order"
            )
        return self


class MageVideoInferenceIdentity(StrictModel):
    """Model, policy, and manifest identities that bind one result."""

    identity_version: Literal["mage-video-inference-identity-v2"] = (
        MAGE_VIDEO_INFERENCE_IDENTITY_VERSION
    )
    model_identity: MageVideoModelIdentity
    model_identity_sha256: Sha256Digest
    codec_policy_identity: MageVideoCodecPolicyIdentity
    input_manifest_sha256: Sha256Digest
    decoder_identity_sha256: Sha256Digest
    inference_identity: Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self) -> MageVideoInferenceIdentity:
        if self.model_identity_sha256 != mage_video_model_identity_sha256(self.model_identity):
            raise ValueError("model_identity_sha256 does not match model_identity")
        expected = semantic_sha256(
            {
                "identity_version": self.identity_version,
                "model_identity_sha256": self.model_identity_sha256,
                "codec_policy_identity": self.codec_policy_identity.model_dump(mode="json"),
                "input_manifest_sha256": self.input_manifest_sha256,
                "decoder_identity_sha256": self.decoder_identity_sha256,
            }
        )
        if self.inference_identity != expected:
            raise ValueError("inference_identity does not match its canonical identity projection")
        return self


class MageVideoResultArtifactReference(StrictModel):
    """Explicit durable storage reference for the generated result artifact."""

    artifact_version: Literal["mage-video-result-artifact-v2"] = MAGE_VIDEO_RESULT_ARTIFACT_VERSION
    artifact_identity: Sha256Digest
    content_sha256: Sha256Digest
    durable_path: DurablePath


class MageVideoEndpointResponse(StrictModel):
    """v2 durable result envelope; transient model state is intentionally absent."""

    contract_version: Literal["mage-video-codec-response-v2"] = MAGE_VIDEO_ENDPOINT_RESPONSE_VERSION
    request_id: NonEmptyString
    inference_identity: MageVideoInferenceIdentity
    camera_encoding_count: PositiveInt
    decoder_id: NonEmptyString
    prompt_tokens: PositiveInt
    output_tokens: NonNegativeInt
    load_seconds: NonNegativeFloat
    generation_seconds: NonNegativeFloat
    execution_device: NonEmptyString
    preprocess_device: PreprocessDevice
    output_text: NonEmptyString
    result_artifact: MageVideoResultArtifactReference


class MageVideoResultArtifactDocument(StrictModel):
    """Canonical, independently verifiable persisted inference result."""

    artifact_version: Literal["mage-video-result-artifact-v2"] = MAGE_VIDEO_RESULT_ARTIFACT_VERSION
    artifact_identity: Sha256Digest
    request_id: NonEmptyString
    inference_identity: MageVideoInferenceIdentity
    camera_encoding_count: PositiveInt
    decoder_id: NonEmptyString
    prompt_tokens: PositiveInt
    output_tokens: NonNegativeInt
    load_seconds: NonNegativeFloat
    generation_seconds: NonNegativeFloat
    execution_device: NonEmptyString
    preprocess_device: PreprocessDevice
    output_text: NonEmptyString
    # This timestamp is authored by the endpoint after generation.  It is not
    # accepted in the request and model output text cannot define it.
    created_at: Annotated[
        str,
        StringConstraints(
            strict=True,
            min_length=20,
            max_length=40,
            pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$",
        ),
    ]

    @model_validator(mode="after")
    def validate_artifact_identity(self) -> MageVideoResultArtifactDocument:
        _validate_rfc3339_utc_timestamp(self.created_at)
        expected = semantic_sha256(_result_artifact_projection(self))
        if self.artifact_identity != expected:
            raise ValueError("artifact_identity does not match its canonical result projection")
        return self


class MageVideoHealthResponse(StrictModel):
    """Readiness status for the single-resident runtime."""

    status: Literal["READY"] = "READY"
    model_identity: MageVideoModelIdentity
    loaded: Literal[True] = True
    concurrency: Literal[1] = 1
    codec_policy_version: Literal["mage-video-codec-policy-v2"] = MAGE_VIDEO_CODEC_POLICY_VERSION
    # The endpoint accepts no environment-derived device default. Each request
    # carries its own explicit CPU/CUDA codec policy.
    preprocess_device_requirement: Literal["EXPLICIT_CPU_OR_CUDA"] = "EXPLICIT_CPU_OR_CUDA"


class MageVideoInferenceRuntime(Protocol):
    """Minimal runtime surface used by this endpoint without provider coupling."""

    @property
    def loaded(self) -> bool: ...

    @property
    def runtime_identity(self) -> MageVideoRuntimeIdentity: ...

    @property
    def load_observation(self) -> Any: ...

    def load(self) -> Any: ...

    def close(self) -> None: ...

    def generate(
        self,
        *,
        video_paths: Sequence[Path | str],
        prompt: str,
        max_new_tokens: int,
        codec_config: Mapping[str, Any],
    ) -> Any: ...


class MageVideoEndpointIdempotencyConflictError(MageVideoRuntimeError):
    """A durable idempotency key was reused with a different binding."""


@dataclass(frozen=True, slots=True)
class _MageVideoIdempotencyClaim:
    """Short-lived ownership or replay outcome from the SQLite idempotency ledger."""

    owner_token: str | None = None
    replay_response: MageVideoEndpointResponse | None = None


class MageVideoEndpointService:
    """Bind a native Mage runtime to durable v2 input and output contracts."""

    def __init__(
        self,
        *,
        runtime: MageVideoInferenceRuntime,
        model_identity: MageVideoModelIdentity,
        idempotency_state_path: Path,
        result_artifact_directory: Path,
        durable_input_roots: Sequence[Path],
    ) -> None:
        if not isinstance(model_identity, MageVideoModelIdentity):
            raise TypeError("model_identity must be MageVideoModelIdentity")
        if not isinstance(idempotency_state_path, Path):
            raise TypeError("idempotency_state_path must be pathlib.Path")
        if not isinstance(result_artifact_directory, Path):
            raise TypeError("result_artifact_directory must be pathlib.Path")
        if isinstance(durable_input_roots, (str, bytes)) or not isinstance(
            durable_input_roots, Sequence
        ):
            raise TypeError(
                "durable_input_roots must be a nonempty sequence of pathlib.Path values"
            )
        roots = tuple(Path(root).expanduser().resolve() for root in durable_input_roots)
        if not roots:
            raise ValueError("durable_input_roots must be nonempty")
        if any(not root.is_dir() for root in roots):
            raise ValueError("every durable_input_root must be an existing directory")

        self._runtime = runtime
        self._model_identity = model_identity
        self._idempotency_state_path = Path(idempotency_state_path).expanduser().resolve()
        self._result_artifact_directory = Path(result_artifact_directory).expanduser().resolve()
        self._durable_input_roots = roots
        self._initialize_storage()

    def _open_idempotency_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._idempotency_state_path,
            isolation_level=None,
            timeout=30.0,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_storage(self) -> None:
        try:
            self._idempotency_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._result_artifact_directory.mkdir(parents=True, exist_ok=True)
            with self._open_idempotency_connection() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = FULL")
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_TABLE} (
                        idempotency_key TEXT PRIMARY KEY,
                        request_body_sha256 TEXT NOT NULL,
                        binding_identity_sha256 TEXT NOT NULL,
                        state TEXT NOT NULL,
                        owner_token TEXT,
                        lease_expires_at_unix_seconds REAL,
                        response_json BLOB,
                        CHECK (
                            (state = '{_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_PENDING}'
                             AND owner_token IS NOT NULL
                             AND lease_expires_at_unix_seconds IS NOT NULL
                             AND response_json IS NULL)
                            OR
                            (state = '{_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_COMPLETE}'
                             AND owner_token IS NULL
                             AND lease_expires_at_unix_seconds IS NULL
                             AND response_json IS NOT NULL)
                        )
                    )
                    """
                )
        except (OSError, sqlite3.Error) as error:
            raise MageVideoRuntimeError(
                "Mage video endpoint durable storage could not be initialized"
            ) from error

    @staticmethod
    def _rollback_transaction(connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            connection.execute("ROLLBACK")

    def start(self) -> None:
        """Load the resident runtime during endpoint startup and bind its profile."""

        self._runtime.load()
        self._assert_resident_runtime_identity()

    def stop(self) -> None:
        """Release the runtime; generated results remain in explicit artifacts."""

        self._runtime.close()

    def health(self) -> MageVideoHealthResponse:
        self._assert_resident_runtime_identity()
        return MageVideoHealthResponse(model_identity=self._model_identity)

    def infer_idempotently(
        self,
        *,
        request: MageVideoEndpointRequest,
        idempotency_key: str,
        request_body: bytes,
    ) -> MageVideoEndpointResponse:
        """Replay exact request bytes without holding SQLite locks during generation.

        A short SQLite reservation elects one owner for a key.  The expensive
        native model call runs after the reservation transaction commits; other
        callers either replay a completed response or poll the pending record.
        This preserves exact-byte semantics without serialising unrelated keys
        behind a database transaction or a process-global service lock.
        """

        self._assert_model_identity(request)
        self._assert_resident_runtime_identity()
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise MageVideoRuntimeError(
                f"{MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER} must be nonempty"
            )
        if not isinstance(request_body, bytes) or not request_body:
            raise MageVideoRuntimeError("idempotency request body must be nonempty bytes")

        request_body_sha256 = exact_bytes_sha256(request_body)
        binding_identity_sha256 = _idempotency_binding_identity(
            request=request,
            request_body_sha256=request_body_sha256,
        )
        deadline = time.monotonic() + _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_WAIT_TIMEOUT_SECONDS

        while True:
            claim = self._claim_or_replay_idempotency(
                idempotency_key=idempotency_key,
                request_body_sha256=request_body_sha256,
                binding_identity_sha256=binding_identity_sha256,
            )
            if claim.replay_response is not None:
                return claim.replay_response
            if claim.owner_token is not None:
                try:
                    response = self.infer(request)
                except Exception:
                    self._release_idempotency_claim(
                        idempotency_key=idempotency_key,
                        owner_token=claim.owner_token,
                    )
                    raise
                try:
                    if self._publish_idempotency_response(
                        idempotency_key=idempotency_key,
                        request_body_sha256=request_body_sha256,
                        binding_identity_sha256=binding_identity_sha256,
                        owner_token=claim.owner_token,
                        response=response,
                    ):
                        return response
                except Exception:
                    self._release_idempotency_claim(
                        idempotency_key=idempotency_key,
                        owner_token=claim.owner_token,
                    )
                    raise

            if time.monotonic() >= deadline:
                raise MageVideoRuntimeError(
                    "Mage video idempotency reservation did not complete before its wait timeout"
                )
            time.sleep(_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_POLL_SECONDS)

    def _claim_or_replay_idempotency(
        self,
        *,
        idempotency_key: str,
        request_body_sha256: Sha256Digest,
        binding_identity_sha256: Sha256Digest,
    ) -> _MageVideoIdempotencyClaim:
        """Claim a key briefly, replay a finished response, or report a live owner."""

        owner_token = uuid4().hex
        now = time.time()
        lease_expires_at = now + _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_LEASE_SECONDS
        connection: sqlite3.Connection | None = None
        response_json: bytes | None = None
        try:
            connection = self._open_idempotency_connection()
            connection.execute("BEGIN IMMEDIATE")
            record = connection.execute(
                f"""
                SELECT request_body_sha256, binding_identity_sha256, state, owner_token,
                       lease_expires_at_unix_seconds, response_json
                FROM {_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_TABLE}
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if record is None:
                connection.execute(
                    f"""
                    INSERT INTO {_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_TABLE} (
                        idempotency_key,
                        request_body_sha256,
                        binding_identity_sha256,
                        state,
                        owner_token,
                        lease_expires_at_unix_seconds,
                        response_json
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        idempotency_key,
                        request_body_sha256,
                        binding_identity_sha256,
                        _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_PENDING,
                        owner_token,
                        lease_expires_at,
                    ),
                )
                connection.execute("COMMIT")
                return _MageVideoIdempotencyClaim(owner_token=owner_token)

            (
                existing_body_sha256,
                existing_binding_sha256,
                state,
                _existing_owner_token,
                existing_lease_expires_at,
                existing_response_json,
            ) = record
            if existing_body_sha256 != request_body_sha256:
                raise MageVideoEndpointIdempotencyConflictError(
                    "Idempotency-Key is already bound to different request bytes"
                )
            if existing_binding_sha256 != binding_identity_sha256:
                raise MageVideoEndpointIdempotencyConflictError(
                    "Idempotency-Key is already bound to a different model, resident runtime, "
                    "policy, decoder, or manifest identity"
                )
            if state == _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_COMPLETE:
                if not isinstance(existing_response_json, bytes) or not existing_response_json:
                    raise MageVideoRuntimeError(
                        "Mage video idempotency store complete record has no response"
                    )
                response_json = existing_response_json
                connection.execute("COMMIT")
            elif state == _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_PENDING:
                if (
                    not isinstance(existing_lease_expires_at, (float, int))
                    or float(existing_lease_expires_at) <= now
                ):
                    updated = connection.execute(
                        f"""
                        UPDATE {_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_TABLE}
                        SET owner_token = ?, lease_expires_at_unix_seconds = ?
                        WHERE idempotency_key = ?
                          AND request_body_sha256 = ?
                          AND binding_identity_sha256 = ?
                          AND state = ?
                          AND lease_expires_at_unix_seconds <= ?
                        """,
                        (
                            owner_token,
                            lease_expires_at,
                            idempotency_key,
                            request_body_sha256,
                            binding_identity_sha256,
                            _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_PENDING,
                            now,
                        ),
                    )
                    if updated.rowcount != 1:
                        connection.execute("COMMIT")
                        return _MageVideoIdempotencyClaim()
                    connection.execute("COMMIT")
                    return _MageVideoIdempotencyClaim(owner_token=owner_token)
                connection.execute("COMMIT")
                return _MageVideoIdempotencyClaim()
            else:
                raise MageVideoRuntimeError("Mage video idempotency store has an invalid state")
        except MageVideoRuntimeError:
            if connection is not None:
                self._rollback_transaction(connection)
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            if connection is not None:
                self._rollback_transaction(connection)
            raise MageVideoRuntimeError(
                "Mage video endpoint idempotency reservation operation failed"
            ) from error
        finally:
            if connection is not None:
                connection.close()

        assert response_json is not None
        response = self._decode_persisted_response(response_json)
        self._verify_persisted_artifact(response.result_artifact)
        return _MageVideoIdempotencyClaim(replay_response=response)

    def _publish_idempotency_response(
        self,
        *,
        idempotency_key: str,
        request_body_sha256: Sha256Digest,
        binding_identity_sha256: Sha256Digest,
        owner_token: str,
        response: MageVideoEndpointResponse,
    ) -> bool:
        """Atomically publish a response only while this caller still owns its lease."""

        response_json = canonical_json_bytes(response.model_dump(mode="json"))
        connection: sqlite3.Connection | None = None
        try:
            connection = self._open_idempotency_connection()
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                f"""
                UPDATE {_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_TABLE}
                SET state = ?, owner_token = NULL, lease_expires_at_unix_seconds = NULL,
                    response_json = ?
                WHERE idempotency_key = ?
                  AND request_body_sha256 = ?
                  AND binding_identity_sha256 = ?
                  AND state = ?
                  AND owner_token = ?
                """,
                (
                    _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_COMPLETE,
                    response_json,
                    idempotency_key,
                    request_body_sha256,
                    binding_identity_sha256,
                    _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_PENDING,
                    owner_token,
                ),
            )
            connection.execute("COMMIT")
            return updated.rowcount == 1
        except MageVideoRuntimeError:
            if connection is not None:
                self._rollback_transaction(connection)
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            if connection is not None:
                self._rollback_transaction(connection)
            raise MageVideoRuntimeError(
                "Mage video endpoint idempotency publication operation failed"
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def _release_idempotency_claim(self, *, idempotency_key: str, owner_token: str) -> None:
        """Release only this caller's pending reservation after a failed generation."""

        connection: sqlite3.Connection | None = None
        try:
            connection = self._open_idempotency_connection()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                f"""
                DELETE FROM {_MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_TABLE}
                WHERE idempotency_key = ?
                  AND state = ?
                  AND owner_token = ?
                """,
                (idempotency_key, _MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_PENDING, owner_token),
            )
            connection.execute("COMMIT")
        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
            if connection is not None:
                self._rollback_transaction(connection)
            raise MageVideoRuntimeError(
                "Mage video endpoint idempotency reservation could not be released"
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def infer(self, request: MageVideoEndpointRequest) -> MageVideoEndpointResponse:
        """Verify durable input, call the native runtime, and persist a result artifact."""

        self._assert_model_identity(request)
        self._assert_resident_runtime_identity()
        paths = [
            self._verify_durable_segment(encoding.segment_manifest)
            for encoding in request.camera_encodings
        ]
        inference_identity = build_mage_video_inference_identity(request)
        generated = self._runtime.generate(
            video_paths=paths,
            prompt=request.decoder.prompt,
            max_new_tokens=request.decoder.max_new_tokens,
            codec_config=request.codec_policy.native_codec_config(),
        )
        telemetry = getattr(generated, "telemetry", None)
        if telemetry is not None:
            _LOGGER.info(
                "mage_video_generation_telemetry",
                extra={
                    "mage_request_id": request.request_id,
                    "processor_lock_wait_seconds": telemetry.processor_lock_wait_seconds,
                    "processor_seconds": telemetry.processor_seconds,
                    "generation_lock_wait_seconds": telemetry.generation_lock_wait_seconds,
                    "input_materialization_seconds": telemetry.input_materialization_seconds,
                    "generate_seconds": telemetry.generate_seconds,
                    "decode_seconds": telemetry.decode_seconds,
                    "total_request_seconds": telemetry.total_request_seconds,
                    "output_tokens": generated.output_tokens,
                },
            )
        self._assert_resident_runtime_identity()
        loaded = self._runtime.load_observation
        document = _build_result_artifact_document(
            request=request,
            inference_identity=inference_identity,
            prompt_tokens=int(generated.prompt_tokens),
            output_tokens=int(generated.output_tokens),
            load_seconds=float(loaded.load_seconds),
            generation_seconds=float(generated.generation_seconds),
            execution_device=str(loaded.execution_device),
            preprocess_device=request.codec_policy.preprocess_device,
            output_text=str(generated.output_text),
            created_at=_server_authored_rfc3339_utc_timestamp(),
        )
        artifact_reference = self._persist_result_artifact(document)
        return MageVideoEndpointResponse(
            request_id=request.request_id,
            inference_identity=inference_identity,
            camera_encoding_count=len(request.camera_encodings),
            decoder_id=request.decoder.decoder_id,
            prompt_tokens=document.prompt_tokens,
            output_tokens=document.output_tokens,
            load_seconds=document.load_seconds,
            generation_seconds=document.generation_seconds,
            execution_device=document.execution_device,
            preprocess_device=document.preprocess_device,
            output_text=document.output_text,
            result_artifact=artifact_reference,
        )

    def _assert_model_identity(self, request: MageVideoEndpointRequest) -> None:
        if request.model_identity != self._model_identity:
            raise MageVideoRuntimeError(
                "request model identity does not match the configured Mage video model revision"
            )

    def _assert_resident_runtime_identity(self) -> None:
        """Fail closed unless the loaded runtime matches the configured v2 identity."""

        if not self._runtime.loaded:
            raise MageVideoRuntimeError("Mage video model is not loaded")
        expected = self._model_identity.runtime_identity.to_runtime_identity()
        runtime_identity = getattr(self._runtime, "runtime_identity", None)
        if not isinstance(runtime_identity, MageVideoRuntimeIdentity):
            raise MageVideoRuntimeError(
                "resident Mage video runtime does not expose a valid runtime identity"
            )
        if runtime_identity != expected:
            raise MageVideoRuntimeError(
                "resident Mage video runtime identity/load profile does not match configured "
                "model identity"
            )
        try:
            load_observation = self._runtime.load_observation
        except MageVideoRuntimeError:
            raise
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise MageVideoRuntimeError(
                "resident Mage video runtime does not expose a load observation"
            ) from error
        observed_identity = getattr(load_observation, "runtime_identity", None)
        if observed_identity != expected:
            raise MageVideoRuntimeError(
                "resident Mage video load observation identity/load profile does not match "
                "configured model identity"
            )

    def _verify_durable_segment(self, manifest: MageVideoSegmentManifest) -> Path:
        try:
            path = Path(manifest.durable_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise MageVideoRuntimeError(
                f"durable video segment path could not be resolved: {manifest.durable_path}"
            ) from error
        if not any(_is_within(path, root) for root in self._durable_input_roots):
            raise MageVideoRuntimeError(
                "durable video segment path is outside configured input roots"
            )
        if not path.is_file():
            raise MageVideoRuntimeError(f"durable video segment is not a file: {path}")
        try:
            byte_count = path.stat().st_size
        except OSError as error:
            raise MageVideoRuntimeError(
                f"durable video segment could not be stat'ed: {path}"
            ) from error
        if byte_count != manifest.byte_count:
            raise MageVideoRuntimeError(
                "durable video segment byte_count does not match its manifest"
            )
        content_sha256 = _file_sha256(path)
        if content_sha256 != manifest.content_sha256:
            raise MageVideoRuntimeError(
                "durable video segment content digest does not match its manifest"
            )
        return path

    def _persist_result_artifact(
        self,
        document: MageVideoResultArtifactDocument,
    ) -> MageVideoResultArtifactReference:
        data = canonical_json_bytes(document.model_dump(mode="json"))
        durable_path = self._result_artifact_directory / f"{document.artifact_identity}.json"
        try:
            if durable_path.exists():
                existing = durable_path.read_bytes()
                if existing != data:
                    raise MageVideoRuntimeError(
                        "result artifact identity already exists with different canonical content"
                    )
            else:
                temporary_path = durable_path.with_name(f".{durable_path.name}.{uuid4().hex}.tmp")
                try:
                    with temporary_path.open("xb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary_path, durable_path)
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
        except MageVideoRuntimeError:
            raise
        except OSError as error:
            raise MageVideoRuntimeError(
                "Mage video result artifact could not be persisted"
            ) from error
        return MageVideoResultArtifactReference(
            artifact_identity=document.artifact_identity,
            content_sha256=exact_bytes_sha256(data),
            durable_path=str(durable_path),
        )

    @staticmethod
    def _decode_persisted_response(data: object) -> MageVideoEndpointResponse:
        if not isinstance(data, bytes) or not data:
            raise MageVideoRuntimeError(
                "Mage video endpoint idempotency store contains an invalid response record"
            )
        try:
            response = MageVideoEndpointResponse.model_validate_json(data, strict=True)
        except (TypeError, ValueError, ValidationError) as error:
            raise MageVideoRuntimeError(
                "Mage video endpoint idempotency store contains an invalid response contract"
            ) from error
        if canonical_json_bytes(response.model_dump(mode="json")) != data:
            raise MageVideoRuntimeError(
                "Mage video endpoint idempotency store response must use canonical JSON bytes"
            )
        return response

    def _verify_persisted_artifact(self, reference: MageVideoResultArtifactReference) -> None:
        try:
            path = Path(reference.durable_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise MageVideoRuntimeError(
                "persisted Mage video result artifact is unavailable"
            ) from error
        if not _is_within(path, self._result_artifact_directory):
            raise MageVideoRuntimeError(
                "persisted Mage video result artifact path is outside its root"
            )
        try:
            data = path.read_bytes()
        except OSError as error:
            raise MageVideoRuntimeError(
                "persisted Mage video result artifact could not be read"
            ) from error
        if exact_bytes_sha256(data) != reference.content_sha256:
            raise MageVideoRuntimeError(
                "persisted Mage video result artifact content digest mismatch"
            )
        try:
            document = MageVideoResultArtifactDocument.model_validate_json(data, strict=True)
        except (TypeError, ValueError, ValidationError) as error:
            raise MageVideoRuntimeError(
                "persisted Mage video result artifact violates its contract"
            ) from error
        if canonical_json_bytes(document.model_dump(mode="json")) != data:
            raise MageVideoRuntimeError(
                "persisted Mage video result artifact must use canonical JSON"
            )
        if document.artifact_identity != reference.artifact_identity:
            raise MageVideoRuntimeError("persisted Mage video result artifact identity mismatch")


def build_mage_video_segment_manifest(
    *,
    segment_id: str,
    camera_id: str,
    durable_path: str,
    media_type: str,
    content_sha256: str,
    byte_count: int,
) -> MageVideoSegmentManifest:
    """Build a segment manifest with its canonical path-independent identity."""

    projection: dict[str, object] = {
        "manifest_version": MAGE_VIDEO_SEGMENT_MANIFEST_VERSION,
        "segment_id": segment_id,
        "camera_id": camera_id,
        "media_type": media_type,
        "content_sha256": content_sha256,
        "byte_count": byte_count,
    }
    return MageVideoSegmentManifest(
        manifest_version=MAGE_VIDEO_SEGMENT_MANIFEST_VERSION,
        segment_id=segment_id,
        camera_id=camera_id,
        durable_path=durable_path,
        media_type=media_type,
        content_sha256=content_sha256,
        byte_count=byte_count,
        manifest_identity=semantic_sha256(projection),
    )


def build_mage_video_context_manifest(
    *,
    context_id: str,
    context_payload_sha256: str,
    segment_manifest_identities: Sequence[str],
) -> MageVideoContextManifest:
    """Build a context manifest whose identity is bound to segment order."""

    identities = list(segment_manifest_identities)
    projection: dict[str, object] = {
        "manifest_version": MAGE_VIDEO_CONTEXT_MANIFEST_VERSION,
        "context_id": context_id,
        "context_payload_sha256": context_payload_sha256,
        "segment_manifest_identities": identities,
    }
    return MageVideoContextManifest(
        manifest_version=MAGE_VIDEO_CONTEXT_MANIFEST_VERSION,
        context_id=context_id,
        context_payload_sha256=context_payload_sha256,
        segment_manifest_identities=identities,
        manifest_identity=semantic_sha256(projection),
    )


def mage_video_model_identity_sha256(identity: MageVideoModelIdentity) -> Sha256Digest:
    """Return the canonical identity hash of model identifier, revision, and manifest."""

    if not isinstance(identity, MageVideoModelIdentity):
        raise TypeError("identity must be MageVideoModelIdentity")
    return semantic_sha256(identity.model_dump(mode="json"))


def build_mage_video_codec_policy_identity(
    policy: MageVideoCodecPolicy,
) -> MageVideoCodecPolicyIdentity:
    """Return the stable policy identity included in result and idempotency bindings."""

    if not isinstance(policy, MageVideoCodecPolicy):
        raise TypeError("policy must be MageVideoCodecPolicy")
    return MageVideoCodecPolicyIdentity(
        codec_mode=policy.codec_mode,
        preprocess_device=policy.preprocess_device,
        policy_sha256=semantic_sha256(policy.model_dump(mode="json")),
    )


def mage_video_input_manifest_sha256(request: MageVideoEndpointRequest) -> Sha256Digest:
    """Hash the ordered camera manifests and their enclosing context manifest."""

    if not isinstance(request, MageVideoEndpointRequest):
        raise TypeError("request must be MageVideoEndpointRequest")
    return semantic_sha256(
        {
            "context_manifest_identity": request.context_manifest.manifest_identity,
            "segment_manifest_identities": [
                encoding.segment_manifest.manifest_identity for encoding in request.camera_encodings
            ],
        }
    )


def build_mage_video_inference_identity(
    request: MageVideoEndpointRequest,
) -> MageVideoInferenceIdentity:
    """Combine model/revision, policy, and manifest identity into one binding."""

    if not isinstance(request, MageVideoEndpointRequest):
        raise TypeError("request must be MageVideoEndpointRequest")
    model_identity_sha256 = mage_video_model_identity_sha256(request.model_identity)
    policy_identity = build_mage_video_codec_policy_identity(request.codec_policy)
    input_manifest_sha256 = mage_video_input_manifest_sha256(request)
    decoder_identity_sha256 = semantic_sha256(request.decoder.model_dump(mode="json"))
    projection = {
        "identity_version": MAGE_VIDEO_INFERENCE_IDENTITY_VERSION,
        "model_identity_sha256": model_identity_sha256,
        "codec_policy_identity": policy_identity.model_dump(mode="json"),
        "input_manifest_sha256": input_manifest_sha256,
        "decoder_identity_sha256": decoder_identity_sha256,
    }
    return MageVideoInferenceIdentity(
        model_identity=request.model_identity,
        model_identity_sha256=model_identity_sha256,
        codec_policy_identity=policy_identity,
        input_manifest_sha256=input_manifest_sha256,
        decoder_identity_sha256=decoder_identity_sha256,
        inference_identity=semantic_sha256(projection),
    )


def create_mage_video_endpoint_app(service: MageVideoEndpointService) -> Any:
    """Create the optional FastAPI v2 application without a core web dependency."""

    try:
        fastapi = __import__("fastapi", fromlist=["FastAPI", "HTTPException"])
    except ImportError as error:
        raise MageVideoRuntimeError(
            "Mage video HTTP endpoint requires the robata[web] optional dependencies"
        ) from error

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        await asyncio.to_thread(service.start)
        try:
            yield
        finally:
            await asyncio.to_thread(service.stop)

    app = fastapi.FastAPI(title="Robata Mage Video Endpoint", lifespan=lifespan)

    async def health() -> MageVideoHealthResponse:
        try:
            return service.health()
        except MageVideoRuntimeError as error:
            raise fastapi.HTTPException(status_code=503, detail=str(error)) from error

    app.get("/healthz", response_model=MageVideoHealthResponse)(health)

    async def infer(
        endpoint_request: MageVideoEndpointRequest,
        raw_request: Any,
        idempotency_key: str = fastapi.Header(alias=MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER),
    ) -> MageVideoEndpointResponse:
        try:
            raw_body = await raw_request.body()
            return await asyncio.to_thread(
                service.infer_idempotently,
                request=endpoint_request,
                idempotency_key=idempotency_key,
                request_body=raw_body,
            )
        except MageVideoEndpointIdempotencyConflictError as error:
            raise fastapi.HTTPException(status_code=409, detail=str(error)) from error
        except MageVideoRuntimeError as error:
            raise fastapi.HTTPException(status_code=422, detail=str(error)) from error

    # FastAPI is optional, so bind its Request annotation only while creating
    # the application rather than importing it when this module is imported.
    infer.__annotations__["raw_request"] = fastapi.Request
    app.post("/v2/mage-video/infer", response_model=MageVideoEndpointResponse)(infer)
    return app


def _segment_manifest_projection(manifest: MageVideoSegmentManifest) -> dict[str, object]:
    return manifest.model_dump(
        mode="json",
        exclude={"durable_path", "manifest_identity"},
    )


def _context_manifest_projection(manifest: MageVideoContextManifest) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"manifest_identity"})


def _result_artifact_projection(document: MageVideoResultArtifactDocument) -> dict[str, object]:
    return document.model_dump(mode="json", exclude={"artifact_identity"})


def _build_result_artifact_document(
    *,
    request: MageVideoEndpointRequest,
    inference_identity: MageVideoInferenceIdentity,
    prompt_tokens: int,
    output_tokens: int,
    load_seconds: float,
    generation_seconds: float,
    execution_device: str,
    preprocess_device: PreprocessDevice,
    output_text: str,
    created_at: str,
) -> MageVideoResultArtifactDocument:
    projection: dict[str, object] = {
        "artifact_version": MAGE_VIDEO_RESULT_ARTIFACT_VERSION,
        "request_id": request.request_id,
        "inference_identity": inference_identity.model_dump(mode="json"),
        "camera_encoding_count": len(request.camera_encodings),
        "decoder_id": request.decoder.decoder_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "execution_device": execution_device,
        "preprocess_device": preprocess_device,
        "output_text": output_text,
        "created_at": created_at,
    }
    return MageVideoResultArtifactDocument(
        artifact_version=MAGE_VIDEO_RESULT_ARTIFACT_VERSION,
        artifact_identity=semantic_sha256(projection),
        request_id=request.request_id,
        inference_identity=inference_identity,
        camera_encoding_count=len(request.camera_encodings),
        decoder_id=request.decoder.decoder_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        load_seconds=load_seconds,
        generation_seconds=generation_seconds,
        execution_device=execution_device,
        preprocess_device=preprocess_device,
        output_text=output_text,
        created_at=created_at,
    )


def _idempotency_binding_identity(
    *,
    request: MageVideoEndpointRequest,
    request_body_sha256: Sha256Digest,
) -> Sha256Digest:
    inference_identity = build_mage_video_inference_identity(request)
    return semantic_sha256(
        {
            "idempotency_policy_version": MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_POLICY_VERSION,
            "request_body_sha256": request_body_sha256,
            "model_identity_sha256": inference_identity.model_identity_sha256,
            "codec_policy_sha256": inference_identity.codec_policy_identity.policy_sha256,
            "input_manifest_sha256": inference_identity.input_manifest_sha256,
            "decoder_identity_sha256": inference_identity.decoder_identity_sha256,
        }
    )


def _server_authored_rfc3339_utc_timestamp() -> str:
    """Return the endpoint-authored RFC 3339 UTC creation timestamp."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_rfc3339_utc_timestamp(value: str) -> None:
    """Reject malformed or non-UTC timestamps in independently read artifacts."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be a valid RFC 3339 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("created_at must be an RFC 3339 UTC timestamp")


def _file_sha256(path: Path) -> Sha256Digest:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MageVideoRuntimeError(f"could not hash durable video segment: {path}") from error
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = [
    "MAGE_VIDEO_CODEC_POLICY_VERSION",
    "MAGE_VIDEO_CONTEXT_MANIFEST_VERSION",
    "MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER",
    "MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_POLICY_VERSION",
    "MAGE_VIDEO_ENDPOINT_REQUEST_VERSION",
    "MAGE_VIDEO_ENDPOINT_RESPONSE_VERSION",
    "MAGE_VIDEO_INFERENCE_IDENTITY_VERSION",
    "MAGE_VIDEO_MODEL_IDENTITY_VERSION",
    "MAGE_VIDEO_RESULT_ARTIFACT_VERSION",
    "MAGE_VIDEO_SEGMENT_MANIFEST_VERSION",
    "MageVideoCameraEncoding",
    "MageVideoCodecPolicy",
    "MageVideoCodecPolicyIdentity",
    "MageVideoContextManifest",
    "MageVideoDecoderRequest",
    "MageVideoEndpointIdempotencyConflictError",
    "MageVideoEndpointRequest",
    "MageVideoEndpointResponse",
    "MageVideoEndpointService",
    "MageVideoHealthResponse",
    "MageVideoInferenceIdentity",
    "MageVideoInferenceRuntime",
    "MageVideoModelIdentity",
    "MageVideoNeuralCodecParameters",
    "MageVideoResultArtifactDocument",
    "MageVideoResultArtifactReference",
    "MageVideoRuntimeIdentityBinding",
    "MageVideoSegmentManifest",
    "build_mage_video_codec_policy_identity",
    "build_mage_video_context_manifest",
    "build_mage_video_inference_identity",
    "build_mage_video_segment_manifest",
    "create_mage_video_endpoint_app",
    "mage_video_input_manifest_sha256",
    "mage_video_model_identity_sha256",
]
