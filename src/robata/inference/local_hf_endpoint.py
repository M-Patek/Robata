"""Loopback-only HTTP contract for one resident local Hugging Face vision model."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import sqlite3
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import Condition, Lock
from typing import Annotated, Any, Final, Literal, Protocol

from pydantic import Field, StringConstraints, ValidationError, model_validator

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.local_hf_runtime import (
    LocalHfBatchGenerationObservation,
    LocalHfBatchGenerationRequest,
    LocalHuggingFaceRuntimeError,
)

LOCAL_HF_ENDPOINT_REQUEST_VERSION: Literal["local-hf-vision-request-v1"] = (
    "local-hf-vision-request-v1"
)
LOCAL_HF_ENDPOINT_RESPONSE_VERSION: Literal["local-hf-vision-response-v1"] = (
    "local-hf-vision-response-v1"
)
LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER: Final = "Idempotency-Key"
LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION: Final = "sqlite-exact-body-replay-v1"
LOCAL_HF_BATCH_INFER_PATH: Final = "/v1/local-vision/infer-batch"
LOCAL_HF_BATCH_MAX_SIZE: Final = 8
LOCAL_HF_BATCH_IDEMPOTENCY_POLICY_VERSION: Final = "sqlite-canonical-member-replay-v1"
LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION: Literal["local-hf-vision-batch-request-v1"] = (
    "local-hf-vision-batch-request-v1"
)
LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION: Literal["local-hf-vision-batch-response-v1"] = (
    "local-hf-vision-batch-response-v1"
)
LOCAL_HF_BATCH_POLICY_VERSION: Literal["local-hf-native-batch-policy-v1"] = (
    "local-hf-native-batch-policy-v1"
)
LOCAL_HF_CHECKPOINT_MANIFEST_VERSION: Literal["local-hf-checkpoint-manifest-v1"] = (
    "local-hf-checkpoint-manifest-v1"
)
_LOCAL_HF_ENDPOINT_IDEMPOTENCY_TABLE: Final = "local_hf_endpoint_idempotency_v1"
_LOCAL_HF_BATCH_IDEMPOTENCY_TABLE: Final = "local_hf_endpoint_batch_idempotency_v1"
_LOCAL_HF_CHECKPOINT_EXCLUDED_DIRECTORIES: Final = frozenset({".cache", ".git", "__pycache__"})
_LOCAL_HF_CHECKPOINT_EXACT_FILENAMES: Final = frozenset(
    {
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "feature_extractor_config.json",
        "generation_config.json",
        "image_processor_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "vocab.txt",
    }
)
_LOCAL_HF_CHECKPOINT_WEIGHT_SUFFIXES: Final = (
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
)
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
Base64Payload = Annotated[
    str,
    StringConstraints(strict=True, min_length=4, max_length=4_000_000),
]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class LocalHfEncodedImage(StrictModel):
    camera_id: NonEmptyString
    media_type: Literal["image/png"] = "image/png"
    sha256: Sha256Digest
    base64_data: Base64Payload


class LocalHfEndpointRequest(StrictModel):
    contract_version: Literal["local-hf-vision-request-v1"] = LOCAL_HF_ENDPOINT_REQUEST_VERSION
    request_id: NonEmptyString
    images: Annotated[list[LocalHfEncodedImage], Field(min_length=1, max_length=6)]
    prompt: NonEmptyString
    max_new_tokens: Annotated[int, Field(strict=True, ge=1, le=512)]


class LocalHfEndpointResponse(StrictModel):
    contract_version: Literal["local-hf-vision-response-v1"] = LOCAL_HF_ENDPOINT_RESPONSE_VERSION
    request_id: NonEmptyString
    model_identifier: NonEmptyString
    model_version: NonEmptyString
    quantization: Literal["bnb-nf4-double-quant"] = "bnb-nf4-double-quant"
    precision: Literal["bfloat16-compute"] = "bfloat16-compute"
    input_image_count: PositiveInt
    rendered_image_sizes: tuple[tuple[PositiveInt, PositiveInt], ...]
    prompt_tokens: PositiveInt
    output_tokens: NonNegativeInt
    load_seconds: NonNegativeFloat
    generation_seconds: NonNegativeFloat
    gpu_name: NonEmptyString
    gpu_total_bytes: PositiveInt
    gpu_free_before_bytes: NonNegativeInt
    gpu_allocated_after_load_bytes: NonNegativeInt
    gpu_peak_allocated_bytes: NonNegativeInt
    output_text: NonEmptyString


class LocalHfBatchEndpointMemberRequest(StrictModel):
    """One ordered logical request with its own durable idempotency key."""

    idempotency_key: NonEmptyString
    request: LocalHfEndpointRequest


def build_local_hf_batch_request_sha256(
    *,
    members: Sequence[LocalHfBatchEndpointMemberRequest],
    batch_policy_version: str = LOCAL_HF_BATCH_POLICY_VERSION,
) -> Sha256Digest:
    """Bind the ordered member contracts and native-batch policy into one identity."""

    if batch_policy_version != LOCAL_HF_BATCH_POLICY_VERSION:
        raise ValueError("unsupported local HF native-batch policy version")
    normalized_members = tuple(members)
    if not normalized_members:
        raise ValueError("at least one batch member is required")
    if len(normalized_members) > LOCAL_HF_BATCH_MAX_SIZE:
        raise ValueError(f"batch member count must not exceed {LOCAL_HF_BATCH_MAX_SIZE}")
    if any(
        not isinstance(member, LocalHfBatchEndpointMemberRequest) for member in normalized_members
    ):
        raise TypeError("members must contain LocalHfBatchEndpointMemberRequest values")
    return exact_bytes_sha256(
        canonical_json_bytes(
            {
                "contract_version": LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION,
                "batch_policy_version": batch_policy_version,
                "members": [member.model_dump(mode="json") for member in normalized_members],
            }
        )
    )


class LocalHfBatchEndpointRequest(StrictModel):
    """Strict ordered native-batch request with an explicit policy-bound identity."""

    contract_version: Literal["local-hf-vision-batch-request-v1"] = (
        LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION
    )
    batch_policy_version: Literal["local-hf-native-batch-policy-v1"] = LOCAL_HF_BATCH_POLICY_VERSION
    batch_request_sha256: Sha256Digest
    members: Annotated[
        list[LocalHfBatchEndpointMemberRequest],
        Field(min_length=1, max_length=LOCAL_HF_BATCH_MAX_SIZE),
    ]

    @model_validator(mode="after")
    def validate_batch_contract(self) -> LocalHfBatchEndpointRequest:
        idempotency_keys = [member.idempotency_key for member in self.members]
        if len(set(idempotency_keys)) != len(idempotency_keys):
            raise ValueError("batch member idempotency keys must be unique")
        max_new_tokens = {member.request.max_new_tokens for member in self.members}
        if len(max_new_tokens) != 1:
            raise ValueError("all batch members must use the same max_new_tokens")
        expected = build_local_hf_batch_request_sha256(
            members=self.members,
            batch_policy_version=self.batch_policy_version,
        )
        if self.batch_request_sha256 != expected:
            raise ValueError("batch_request_sha256 does not match the ordered batch contract")
        return self


class LocalHfBatchEndpointMemberResponse(StrictModel):
    """One logical result; physical batch timing intentionally lives at top level."""

    idempotency_key: NonEmptyString
    request_id: NonEmptyString
    disposition: Literal["GENERATED", "REPLAY"]
    input_image_count: PositiveInt
    rendered_image_sizes: Annotated[
        tuple[tuple[PositiveInt, PositiveInt], ...],
        Field(min_length=1, max_length=6),
    ]
    prompt_tokens: PositiveInt
    output_tokens: NonNegativeInt
    output_text: NonEmptyString

    @model_validator(mode="after")
    def validate_image_counts(self) -> LocalHfBatchEndpointMemberResponse:
        if len(self.rendered_image_sizes) != self.input_image_count:
            raise ValueError("rendered_image_sizes must match input_image_count")
        return self


class LocalHfBatchEndpointResponse(StrictModel):
    """Ordered batch results and truthful telemetry for the one physical call."""

    contract_version: Literal["local-hf-vision-batch-response-v1"] = (
        LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION
    )
    batch_policy_version: Literal["local-hf-native-batch-policy-v1"] = LOCAL_HF_BATCH_POLICY_VERSION
    batch_request_sha256: Sha256Digest
    model_identifier: NonEmptyString
    model_version: NonEmptyString
    quantization: Literal["bnb-nf4-double-quant"] = "bnb-nf4-double-quant"
    precision: Literal["bfloat16-compute"] = "bfloat16-compute"
    load_seconds: NonNegativeFloat
    gpu_name: NonEmptyString
    gpu_total_bytes: PositiveInt
    gpu_free_before_bytes: NonNegativeInt
    gpu_allocated_after_load_bytes: NonNegativeInt
    physical_generation_seconds: NonNegativeFloat
    physical_gpu_peak_allocated_bytes: NonNegativeInt
    generated_member_count: NonNegativeInt
    replay_member_count: NonNegativeInt
    members: Annotated[
        tuple[LocalHfBatchEndpointMemberResponse, ...],
        Field(min_length=1, max_length=LOCAL_HF_BATCH_MAX_SIZE),
    ]

    @model_validator(mode="after")
    def validate_member_counts(self) -> LocalHfBatchEndpointResponse:
        generated = sum(member.disposition == "GENERATED" for member in self.members)
        replayed = sum(member.disposition == "REPLAY" for member in self.members)
        if self.generated_member_count != generated:
            raise ValueError("generated_member_count does not match member dispositions")
        if self.replay_member_count != replayed:
            raise ValueError("replay_member_count does not match member dispositions")
        return self


class LocalHfCheckpointIdentity(StrictModel):
    """Deterministic local checkpoint identity exposed by the loopback endpoint."""

    manifest_version: Literal["local-hf-checkpoint-manifest-v1"] = (
        LOCAL_HF_CHECKPOINT_MANIFEST_VERSION
    )
    manifest_sha256: Sha256Digest
    included_file_count: PositiveInt
    hf_revision: NonEmptyString | None = None


class LocalHfHealthResponse(StrictModel):
    status: Literal["READY"] = "READY"
    model_identifier: NonEmptyString
    model_version: NonEmptyString
    checkpoint_identity: LocalHfCheckpointIdentity
    loaded: Literal[True] = True
    concurrency: Literal[1] = 1
    native_batch_available: bool
    native_batch_request_version: Literal["local-hf-vision-batch-request-v1"] | None = None
    native_batch_response_version: Literal["local-hf-vision-batch-response-v1"] | None = None
    native_batch_policy_version: Literal["local-hf-native-batch-policy-v1"] | None = None
    native_batch_idempotency_policy_version: Literal["sqlite-canonical-member-replay-v1"] | None = (
        None
    )
    native_batch_max_size: Literal[8] | None = None

    @model_validator(mode="after")
    def validate_native_batch_capability(self) -> LocalHfHealthResponse:
        advertised = (
            self.native_batch_request_version,
            self.native_batch_response_version,
            self.native_batch_policy_version,
            self.native_batch_idempotency_policy_version,
            self.native_batch_max_size,
        )
        if self.native_batch_available and any(value is None for value in advertised):
            raise ValueError("native batch capability fields are required when available")
        if not self.native_batch_available and any(value is not None for value in advertised):
            raise ValueError("native batch capability fields must be absent when unavailable")
        return self


class LocalVisionRuntime(Protocol):
    """Minimal runtime surface used by the loopback transport."""

    @property
    def loaded(self) -> bool: ...

    @property
    def load_observation(self) -> Any: ...

    def load(self) -> Any: ...

    def close(self) -> None: ...

    def generate(
        self,
        *,
        image_payloads: list[bytes],
        prompt: str,
        max_new_tokens: int,
    ) -> Any: ...


def _checkpoint_identity_includes(relative_path: Path) -> bool:
    name = relative_path.name.lower()
    if name in _LOCAL_HF_CHECKPOINT_EXACT_FILENAMES:
        return True
    if name.endswith(".index.json") and any(
        name.endswith(f"{suffix}.index.json") for suffix in _LOCAL_HF_CHECKPOINT_WEIGHT_SUFFIXES
    ):
        return True
    return name.endswith(_LOCAL_HF_CHECKPOINT_WEIGHT_SUFFIXES)


def _checkpoint_file_sha256(path: Path) -> Sha256Digest:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise LocalHuggingFaceRuntimeError(
            f"checkpoint identity could not read {path.name}"
        ) from error
    return digest.hexdigest()


def _hf_revision_metadata(model_directory: Path) -> str | None:
    """Read local Hugging Face revision hints without admitting cache payloads."""

    if model_directory.parent.name == "snapshots" and model_directory.name:
        return model_directory.name
    for candidate in (
        model_directory / "refs" / "main",
        model_directory.parent / "refs" / "main",
        model_directory.parent.parent / "refs" / "main",
    ):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    metadata_revisions: set[str] = set()
    metadata_root = model_directory / ".cache" / "huggingface" / "download"
    try:
        metadata_paths = sorted(metadata_root.glob("*.metadata"))
    except OSError:
        metadata_paths = []
    for metadata_path in metadata_paths:
        try:
            first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, UnicodeDecodeError, IndexError):
            continue
        if len(first_line) == 40 and all(
            character in "0123456789abcdefABCDEF" for character in first_line
        ):
            metadata_revisions.add(first_line.lower())
    if len(metadata_revisions) == 1:
        return next(iter(metadata_revisions))
    try:
        config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    for field in ("_commit_hash", "commit_hash", "revision", "model_revision"):
        config_value: object = config.get(field)
        if isinstance(config_value, str) and config_value.strip():
            return config_value.strip()
    return None


def build_local_hf_checkpoint_identity(*, model_directory: Path) -> LocalHfCheckpointIdentity:
    """Hash the relevant local checkpoint files into a stable identity manifest.

    Only model/config/tokenizer/processor artifacts are included. Cache payloads
    and incidental files do not affect the identity, while all selected weights
    are content-hashed rather than trusted by filename or self-reported label.
    """

    resolved_directory = Path(model_directory).expanduser().resolve()
    if not resolved_directory.is_dir():
        raise LocalHuggingFaceRuntimeError(
            f"checkpoint model_directory is not a directory: {resolved_directory}"
        )
    entries: list[dict[str, object]] = []
    try:
        for root, directories, filenames in os.walk(resolved_directory):
            directories[:] = sorted(
                directory
                for directory in directories
                if directory.lower() not in _LOCAL_HF_CHECKPOINT_EXCLUDED_DIRECTORIES
            )
            root_path = Path(root)
            for filename in sorted(filenames):
                path = root_path / filename
                relative_path = path.relative_to(resolved_directory)
                if not _checkpoint_identity_includes(relative_path) or not path.is_file():
                    continue
                entries.append(
                    {
                        "path": relative_path.as_posix(),
                        "byte_count": path.stat().st_size,
                        "sha256": _checkpoint_file_sha256(path),
                    }
                )
    except OSError as error:
        raise LocalHuggingFaceRuntimeError(
            "checkpoint identity could not enumerate the model directory"
        ) from error
    if not entries:
        raise LocalHuggingFaceRuntimeError(
            "checkpoint identity found no config, tokenizer, processor, or weight files"
        )
    revision = _hf_revision_metadata(resolved_directory)
    manifest = {
        "manifest_version": LOCAL_HF_CHECKPOINT_MANIFEST_VERSION,
        "hf_revision": revision,
        "files": entries,
    }
    return LocalHfCheckpointIdentity(
        manifest_sha256=exact_bytes_sha256(canonical_json_bytes(manifest)),
        included_file_count=len(entries),
        hf_revision=revision,
    )


def load_local_hf_checkpoint_identity(*, manifest_path: Path) -> LocalHfCheckpointIdentity:
    """Load one previously generated canonical checkpoint identity manifest."""

    resolved_path = Path(manifest_path).expanduser().resolve()
    try:
        data = resolved_path.read_bytes()
    except OSError as error:
        raise LocalHuggingFaceRuntimeError(
            f"checkpoint identity manifest could not be read: {resolved_path}"
        ) from error
    try:
        identity = LocalHfCheckpointIdentity.model_validate_json(data, strict=True)
    except (TypeError, ValueError, ValidationError) as error:
        raise LocalHuggingFaceRuntimeError(
            "checkpoint identity manifest does not satisfy the pinned contract"
        ) from error
    if canonical_json_bytes(identity.model_dump(mode="json")) != data:
        raise LocalHuggingFaceRuntimeError(
            "checkpoint identity manifest must use canonical JSON bytes"
        )
    return identity


def write_local_hf_checkpoint_identity(
    *,
    identity: LocalHfCheckpointIdentity,
    manifest_path: Path,
) -> None:
    """Persist a generated identity for later startup without rehashing weights."""

    if not isinstance(identity, LocalHfCheckpointIdentity):
        raise TypeError("identity must be LocalHfCheckpointIdentity")
    resolved_path = Path(manifest_path).expanduser().resolve()
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_bytes(canonical_json_bytes(identity.model_dump(mode="json")))
    except OSError as error:
        raise LocalHuggingFaceRuntimeError(
            f"checkpoint identity manifest could not be written: {resolved_path}"
        ) from error


class LocalHfEndpointIdempotencyConflictError(LocalHuggingFaceRuntimeError):
    """A durable idempotency key was reused with different request bytes."""


def _decode_local_hf_request_images(request: LocalHfEndpointRequest) -> tuple[bytes, ...]:
    payloads: list[bytes] = []
    for image in request.images:
        try:
            payload = base64.b64decode(image.base64_data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise LocalHuggingFaceRuntimeError(
                f"invalid base64 payload for {image.camera_id}"
            ) from error
        if exact_bytes_sha256(payload) != image.sha256:
            raise LocalHuggingFaceRuntimeError(f"image digest mismatch for {image.camera_id}")
        payloads.append(payload)
    return tuple(payloads)


def _local_hf_batch_member_request_bytes(
    *,
    request: LocalHfEndpointRequest,
    batch_policy_version: str,
) -> bytes:
    """Canonical member identity, deliberately independent from outer batch grouping.

    The serial v1 route remains raw-HTTP-body exact. Batch members instead bind the
    adapter-produced canonical serial request plus the native-batch policy so the same
    logical member can replay across different batch packing without ambiguity.
    """

    return canonical_json_bytes(
        {
            "batch_policy_version": batch_policy_version,
            "request": request.model_dump(mode="json"),
        }
    )


class LocalHfEndpointService:
    """Translate the HTTP contract to one serialized resident runtime."""

    def __init__(
        self,
        *,
        runtime: LocalVisionRuntime,
        model_identifier: str,
        model_version: str,
        checkpoint_identity: LocalHfCheckpointIdentity,
        idempotency_state_path: Path,
    ) -> None:
        if not isinstance(model_identifier, str) or not model_identifier:
            raise ValueError("model_identifier must be nonempty")
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("model_version must be nonempty")
        if not isinstance(checkpoint_identity, LocalHfCheckpointIdentity):
            raise TypeError("checkpoint_identity must be LocalHfCheckpointIdentity")
        if not isinstance(idempotency_state_path, Path):
            raise TypeError("idempotency_state_path must be pathlib.Path")
        self._runtime = runtime
        self._model_identifier = model_identifier
        self._model_version = model_version
        self._checkpoint_identity = checkpoint_identity
        self._idempotency_state_path = idempotency_state_path.expanduser().resolve()
        self._idempotency_lock = Lock()
        self._lifecycle_condition = Condition(Lock())
        self._active_operations = 0
        self._stopping = False
        self._native_batch_available = callable(getattr(runtime, "generate_batch", None))
        self._initialize_idempotency_store()

    def _open_idempotency_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._idempotency_state_path,
            isolation_level=None,
            timeout=30.0,
        )
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_idempotency_store(self) -> None:
        try:
            self._idempotency_state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._open_idempotency_connection() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = FULL")
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_LOCAL_HF_ENDPOINT_IDEMPOTENCY_TABLE} (
                        idempotency_key TEXT PRIMARY KEY,
                        request_body_sha256 TEXT NOT NULL,
                        response_json BLOB NOT NULL
                    )
                    """
                )
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_LOCAL_HF_BATCH_IDEMPOTENCY_TABLE} (
                        idempotency_key TEXT PRIMARY KEY,
                        batch_policy_version TEXT NOT NULL,
                        member_request_sha256 TEXT NOT NULL,
                        member_request_json BLOB NOT NULL,
                        response_json BLOB NOT NULL
                    )
                    """
                )
        except (OSError, sqlite3.Error) as error:
            raise LocalHuggingFaceRuntimeError(
                "local endpoint idempotency store could not be initialized"
            ) from error

    @staticmethod
    def _decode_persisted_response(data: object) -> LocalHfEndpointResponse:
        if not isinstance(data, bytes) or not data:
            raise LocalHuggingFaceRuntimeError(
                "local endpoint idempotency store contains an invalid response record"
            )
        try:
            return LocalHfEndpointResponse.model_validate_json(data, strict=True)
        except (TypeError, ValueError, ValidationError) as error:
            raise LocalHuggingFaceRuntimeError(
                "local endpoint idempotency store contains an invalid response contract"
            ) from error

    @staticmethod
    def _decode_persisted_batch_member_response(
        data: object,
    ) -> LocalHfBatchEndpointMemberResponse:
        if not isinstance(data, bytes) or not data:
            raise LocalHuggingFaceRuntimeError(
                "local batch idempotency store contains an invalid response record"
            )
        try:
            response = LocalHfBatchEndpointMemberResponse.model_validate_json(data, strict=True)
        except (TypeError, ValueError, ValidationError) as error:
            raise LocalHuggingFaceRuntimeError(
                "local batch idempotency store contains an invalid response contract"
            ) from error
        if response.disposition != "GENERATED":
            raise LocalHuggingFaceRuntimeError(
                "local batch idempotency store contains a non-generated response"
            )
        return response

    @staticmethod
    def _rollback_idempotency_transaction(connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            connection.execute("ROLLBACK")

    @property
    def native_batch_available(self) -> bool:
        return self._native_batch_available

    @contextmanager
    def _operation(self) -> Iterator[None]:
        """Track run-to-completion work so shutdown never closes an in-use model."""

        with self._lifecycle_condition:
            if self._stopping:
                raise LocalHuggingFaceRuntimeError("local endpoint is stopping")
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lifecycle_condition:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._lifecycle_condition.notify_all()

    def start(self) -> None:
        with self._lifecycle_condition:
            if self._active_operations:
                raise LocalHuggingFaceRuntimeError("cannot start with active endpoint operations")
            self._stopping = False
        self._runtime.load()

    def stop(self) -> None:
        with self._lifecycle_condition:
            self._stopping = True
            while self._active_operations:
                self._lifecycle_condition.wait()
        self._runtime.close()

    def health(self) -> LocalHfHealthResponse:
        if not self._runtime.loaded:
            raise LocalHuggingFaceRuntimeError("model is not loaded")
        native_batch = self._native_batch_available
        return LocalHfHealthResponse(
            model_identifier=self._model_identifier,
            model_version=self._model_version,
            checkpoint_identity=self._checkpoint_identity,
            native_batch_available=native_batch,
            native_batch_request_version=(
                LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION if native_batch else None
            ),
            native_batch_response_version=(
                LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION if native_batch else None
            ),
            native_batch_policy_version=(LOCAL_HF_BATCH_POLICY_VERSION if native_batch else None),
            native_batch_idempotency_policy_version=(
                LOCAL_HF_BATCH_IDEMPOTENCY_POLICY_VERSION if native_batch else None
            ),
            native_batch_max_size=(LOCAL_HF_BATCH_MAX_SIZE if native_batch else None),
        )

    def infer_idempotently(
        self,
        *,
        request: LocalHfEndpointRequest,
        idempotency_key: str,
        request_body: bytes,
    ) -> LocalHfEndpointResponse:
        with self._operation():
            return self._infer_idempotently(
                request=request,
                idempotency_key=idempotency_key,
                request_body=request_body,
            )

    def _infer_idempotently(
        self,
        *,
        request: LocalHfEndpointRequest,
        idempotency_key: str,
        request_body: bytes,
    ) -> LocalHfEndpointResponse:
        """Replay an exact-body-bound response, including after service restart.

        The SQLite transaction is held through generation so another endpoint
        process sharing this state path cannot issue a second call for the same
        key before the first response is durably recorded.
        """

        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise LocalHuggingFaceRuntimeError(
                f"{LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER} must be nonempty"
            )
        if not isinstance(request_body, bytes) or not request_body:
            raise LocalHuggingFaceRuntimeError("idempotency request body must be nonempty bytes")
        request_body_sha256 = exact_bytes_sha256(request_body)
        with self._idempotency_lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._open_idempotency_connection()
                connection.execute("BEGIN IMMEDIATE")
                record = connection.execute(
                    f"""
                    SELECT request_body_sha256, response_json
                    FROM {_LOCAL_HF_ENDPOINT_IDEMPOTENCY_TABLE}
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if record is not None:
                    existing_body_sha256, persisted_response = record
                    if existing_body_sha256 != request_body_sha256:
                        raise LocalHfEndpointIdempotencyConflictError(
                            "Idempotency-Key is already bound to different request bytes"
                        )
                    response = self._decode_persisted_response(persisted_response)
                    connection.execute("COMMIT")
                    return response
                response = self._infer(request)
                response_json = canonical_json_bytes(response.model_dump(mode="json"))
                connection.execute(
                    f"""
                    INSERT INTO {_LOCAL_HF_ENDPOINT_IDEMPOTENCY_TABLE} (
                        idempotency_key,
                        request_body_sha256,
                        response_json
                    ) VALUES (?, ?, ?)
                    """,
                    (idempotency_key, request_body_sha256, response_json),
                )
                connection.execute("COMMIT")
                return response
            except LocalHuggingFaceRuntimeError:
                if connection is not None:
                    self._rollback_idempotency_transaction(connection)
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                if connection is not None:
                    self._rollback_idempotency_transaction(connection)
                raise LocalHuggingFaceRuntimeError(
                    "local endpoint idempotency store operation failed"
                ) from error
            finally:
                if connection is not None:
                    connection.close()

    def infer_batch_idempotently(
        self,
        *,
        request: LocalHfBatchEndpointRequest,
    ) -> LocalHfBatchEndpointResponse:
        with self._operation():
            return self._infer_batch_idempotently(request=request)

    def _infer_batch_idempotently(
        self,
        *,
        request: LocalHfBatchEndpointRequest,
    ) -> LocalHfBatchEndpointResponse:
        """Generate only durable misses in one physical native-batch call.

        The new table is independent from the serial v1 idempotency namespace.
        One SQLite write transaction covers conflict detection, the physical
        generation, every miss response, and the top-level response validation.
        A failed physical batch therefore cannot persist a successful prefix.
        """

        if not self._native_batch_available:
            raise LocalHuggingFaceRuntimeError(
                "local vision runtime does not expose native batch generation"
            )
        if not isinstance(request, LocalHfBatchEndpointRequest):
            raise TypeError("request must be LocalHfBatchEndpointRequest")
        try:
            request = LocalHfBatchEndpointRequest.model_validate_json(
                canonical_json_bytes(request.model_dump(mode="json")),
                strict=True,
            )
        except (TypeError, ValueError, ValidationError) as error:
            raise LocalHuggingFaceRuntimeError(
                "invalid local native-batch request contract"
            ) from error

        with self._idempotency_lock:
            connection: sqlite3.Connection | None = None
            try:
                connection = self._open_idempotency_connection()
                connection.execute("BEGIN IMMEDIATE")
                ordered_responses: list[LocalHfBatchEndpointMemberResponse | None] = [None] * len(
                    request.members
                )
                miss_indices: list[int] = []
                miss_requests: list[LocalHfBatchGenerationRequest] = []
                miss_bindings: list[tuple[str, str, bytes]] = []

                for member_index, member in enumerate(request.members):
                    member_request_json = _local_hf_batch_member_request_bytes(
                        request=member.request,
                        batch_policy_version=request.batch_policy_version,
                    )
                    member_request_sha256 = exact_bytes_sha256(member_request_json)
                    record = connection.execute(
                        f"""
                        SELECT batch_policy_version, member_request_sha256,
                               member_request_json, response_json
                        FROM {_LOCAL_HF_BATCH_IDEMPOTENCY_TABLE}
                        WHERE idempotency_key = ?
                        """,
                        (member.idempotency_key,),
                    ).fetchone()
                    if record is not None:
                        (
                            existing_policy,
                            existing_request_sha256,
                            existing_request_json,
                            persisted_response,
                        ) = record
                        if (
                            existing_policy != request.batch_policy_version
                            or existing_request_sha256 != member_request_sha256
                            or existing_request_json != member_request_json
                        ):
                            raise LocalHfEndpointIdempotencyConflictError(
                                "batch member idempotency key is already bound to "
                                "different request bytes or batch policy"
                            )
                        response = self._decode_persisted_batch_member_response(persisted_response)
                        if (
                            response.idempotency_key != member.idempotency_key
                            or response.request_id != member.request.request_id
                        ):
                            raise LocalHuggingFaceRuntimeError(
                                "local batch idempotency response binding is invalid"
                            )
                        ordered_responses[member_index] = response.model_copy(
                            update={"disposition": "REPLAY"}
                        )
                        continue

                    payloads = _decode_local_hf_request_images(member.request)
                    miss_indices.append(member_index)
                    miss_requests.append(
                        LocalHfBatchGenerationRequest(
                            image_payloads=payloads,
                            prompt=member.request.prompt,
                            max_new_tokens=member.request.max_new_tokens,
                        )
                    )
                    miss_bindings.append(
                        (
                            member.idempotency_key,
                            member_request_sha256,
                            member_request_json,
                        )
                    )

                physical_generation_seconds = 0.0
                physical_gpu_peak_allocated_bytes = 0
                if miss_requests:
                    generate_batch = getattr(self._runtime, "generate_batch", None)
                    if not callable(generate_batch):
                        raise LocalHuggingFaceRuntimeError(
                            "local vision runtime does not expose native batch generation"
                        )
                    generated = generate_batch(requests=tuple(miss_requests))
                    if not isinstance(generated, LocalHfBatchGenerationObservation):
                        raise LocalHuggingFaceRuntimeError(
                            "local vision runtime returned an invalid native batch observation"
                        )
                    if len(generated.members) != len(miss_requests):
                        raise LocalHuggingFaceRuntimeError(
                            "local vision runtime returned the wrong native batch member count"
                        )
                    physical_generation_seconds = generated.physical_generation_seconds
                    physical_gpu_peak_allocated_bytes = generated.physical_gpu_peak_allocated_bytes
                    for miss_position, member_observation in enumerate(generated.members):
                        member_index = miss_indices[miss_position]
                        member = request.members[member_index]
                        response = LocalHfBatchEndpointMemberResponse(
                            idempotency_key=member.idempotency_key,
                            request_id=member.request.request_id,
                            disposition="GENERATED",
                            input_image_count=len(member.request.images),
                            rendered_image_sizes=member_observation.rendered_image_sizes,
                            prompt_tokens=member_observation.prompt_tokens,
                            output_tokens=member_observation.output_tokens,
                            output_text=member_observation.output_text,
                        )
                        ordered_responses[member_index] = response
                        (
                            idempotency_key,
                            member_request_sha256,
                            member_request_json,
                        ) = miss_bindings[miss_position]
                        connection.execute(
                            f"""
                            INSERT INTO {_LOCAL_HF_BATCH_IDEMPOTENCY_TABLE} (
                                idempotency_key,
                                batch_policy_version,
                                member_request_sha256,
                                member_request_json,
                                response_json
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                idempotency_key,
                                request.batch_policy_version,
                                member_request_sha256,
                                member_request_json,
                                canonical_json_bytes(response.model_dump(mode="json")),
                            ),
                        )

                if any(response is None for response in ordered_responses):
                    raise LocalHuggingFaceRuntimeError(
                        "local native-batch response assembly is incomplete"
                    )
                resolved_responses = tuple(
                    response for response in ordered_responses if response is not None
                )
                loaded = self._runtime.load_observation
                batch_response = LocalHfBatchEndpointResponse(
                    batch_policy_version=request.batch_policy_version,
                    batch_request_sha256=request.batch_request_sha256,
                    model_identifier=self._model_identifier,
                    model_version=self._model_version,
                    load_seconds=loaded.load_seconds,
                    gpu_name=loaded.gpu_name,
                    gpu_total_bytes=loaded.gpu_total_bytes,
                    gpu_free_before_bytes=loaded.gpu_free_before_bytes,
                    gpu_allocated_after_load_bytes=loaded.gpu_allocated_after_load_bytes,
                    physical_generation_seconds=physical_generation_seconds,
                    physical_gpu_peak_allocated_bytes=(physical_gpu_peak_allocated_bytes),
                    generated_member_count=len(miss_requests),
                    replay_member_count=len(request.members) - len(miss_requests),
                    members=resolved_responses,
                )
                connection.execute("COMMIT")
                return batch_response
            except LocalHuggingFaceRuntimeError:
                if connection is not None:
                    self._rollback_idempotency_transaction(connection)
                raise
            except Exception as error:
                if connection is not None:
                    self._rollback_idempotency_transaction(connection)
                raise LocalHuggingFaceRuntimeError(
                    "local native-batch idempotency operation failed"
                ) from error
            finally:
                if connection is not None:
                    connection.close()

    def infer(self, request: LocalHfEndpointRequest) -> LocalHfEndpointResponse:
        with self._operation():
            return self._infer(request)

    def _infer(self, request: LocalHfEndpointRequest) -> LocalHfEndpointResponse:
        payloads: list[bytes] = []
        for image in request.images:
            try:
                payload = base64.b64decode(image.base64_data, validate=True)
            except (binascii.Error, ValueError) as error:
                raise LocalHuggingFaceRuntimeError(
                    f"invalid base64 payload for {image.camera_id}"
                ) from error
            if exact_bytes_sha256(payload) != image.sha256:
                raise LocalHuggingFaceRuntimeError(f"image digest mismatch for {image.camera_id}")
            payloads.append(payload)
        generated = self._runtime.generate(
            image_payloads=payloads,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
        loaded = self._runtime.load_observation
        return LocalHfEndpointResponse(
            request_id=request.request_id,
            model_identifier=self._model_identifier,
            model_version=self._model_version,
            input_image_count=len(payloads),
            rendered_image_sizes=generated.rendered_image_sizes,
            prompt_tokens=generated.prompt_tokens,
            output_tokens=generated.output_tokens,
            load_seconds=loaded.load_seconds,
            generation_seconds=generated.generation_seconds,
            gpu_name=loaded.gpu_name,
            gpu_total_bytes=loaded.gpu_total_bytes,
            gpu_free_before_bytes=loaded.gpu_free_before_bytes,
            gpu_allocated_after_load_bytes=loaded.gpu_allocated_after_load_bytes,
            gpu_peak_allocated_bytes=generated.gpu_peak_allocated_bytes,
            output_text=generated.output_text,
        )


def create_local_hf_endpoint_app(service: LocalHfEndpointService) -> Any:
    """Create the optional FastAPI app without making FastAPI a core dependency."""

    try:
        fastapi = __import__("fastapi", fromlist=["FastAPI", "HTTPException"])
    except ImportError as error:
        raise LocalHuggingFaceRuntimeError(
            "local HTTP endpoint requires the robata[web] optional dependencies"
        ) from error

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        await asyncio.to_thread(service.start)
        try:
            yield
        finally:
            await asyncio.to_thread(service.stop)

    app = fastapi.FastAPI(title="Robata Local HF Vision Endpoint", lifespan=lifespan)

    async def health() -> LocalHfHealthResponse:
        try:
            return service.health()
        except LocalHuggingFaceRuntimeError as error:
            raise fastapi.HTTPException(status_code=503, detail=str(error)) from error

    app.get("/healthz", response_model=LocalHfHealthResponse)(health)

    async def infer(
        endpoint_request: LocalHfEndpointRequest,
        raw_request: Any,
        idempotency_key: str = fastapi.Header(alias=LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER),
    ) -> LocalHfEndpointResponse:
        try:
            raw_body = await raw_request.body()
            return await asyncio.to_thread(
                service.infer_idempotently,
                request=endpoint_request,
                idempotency_key=idempotency_key,
                request_body=raw_body,
            )
        except LocalHfEndpointIdempotencyConflictError as error:
            raise fastapi.HTTPException(status_code=409, detail=str(error)) from error
        except LocalHuggingFaceRuntimeError as error:
            raise fastapi.HTTPException(status_code=422, detail=str(error)) from error

    # ``fastapi`` is intentionally an optional dependency, so bind its request
    # type only while constructing the web application instead of importing it
    # at module import time.
    infer.__annotations__["raw_request"] = fastapi.Request
    app.post("/v1/local-vision/infer", response_model=LocalHfEndpointResponse)(infer)

    async def infer_batch(
        endpoint_request: LocalHfBatchEndpointRequest,
    ) -> LocalHfBatchEndpointResponse:
        try:
            return await asyncio.to_thread(
                service.infer_batch_idempotently,
                request=endpoint_request,
            )
        except LocalHfEndpointIdempotencyConflictError as error:
            raise fastapi.HTTPException(status_code=409, detail=str(error)) from error
        except LocalHuggingFaceRuntimeError as error:
            raise fastapi.HTTPException(status_code=422, detail=str(error)) from error

    if service.native_batch_available:
        app.post(
            LOCAL_HF_BATCH_INFER_PATH,
            response_model=LocalHfBatchEndpointResponse,
        )(infer_batch)

    return app


__all__ = [
    "LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION",
    "LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION",
    "LOCAL_HF_BATCH_IDEMPOTENCY_POLICY_VERSION",
    "LOCAL_HF_BATCH_INFER_PATH",
    "LOCAL_HF_BATCH_MAX_SIZE",
    "LOCAL_HF_BATCH_POLICY_VERSION",
    "LOCAL_HF_CHECKPOINT_MANIFEST_VERSION",
    "LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER",
    "LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION",
    "LOCAL_HF_ENDPOINT_REQUEST_VERSION",
    "LOCAL_HF_ENDPOINT_RESPONSE_VERSION",
    "LocalHfBatchEndpointMemberRequest",
    "LocalHfBatchEndpointMemberResponse",
    "LocalHfBatchEndpointRequest",
    "LocalHfBatchEndpointResponse",
    "LocalHfCheckpointIdentity",
    "LocalHfEncodedImage",
    "LocalHfEndpointIdempotencyConflictError",
    "LocalHfEndpointRequest",
    "LocalHfEndpointResponse",
    "LocalHfEndpointService",
    "LocalHfHealthResponse",
    "LocalVisionRuntime",
    "build_local_hf_batch_request_sha256",
    "build_local_hf_checkpoint_identity",
    "create_local_hf_endpoint_app",
    "load_local_hf_checkpoint_identity",
    "write_local_hf_checkpoint_identity",
]
