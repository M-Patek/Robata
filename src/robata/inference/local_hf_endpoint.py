"""Loopback-only HTTP contract for one resident local Hugging Face vision model."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Final, Literal, Protocol

from pydantic import Field, StringConstraints, ValidationError

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.local_hf_runtime import LocalHuggingFaceRuntimeError

LOCAL_HF_ENDPOINT_REQUEST_VERSION: Literal["local-hf-vision-request-v1"] = (
    "local-hf-vision-request-v1"
)
LOCAL_HF_ENDPOINT_RESPONSE_VERSION: Literal["local-hf-vision-response-v1"] = (
    "local-hf-vision-response-v1"
)
LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER: Final = "Idempotency-Key"
LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION: Final = "sqlite-exact-body-replay-v1"
LOCAL_HF_CHECKPOINT_MANIFEST_VERSION: Literal["local-hf-checkpoint-manifest-v1"] = (
    "local-hf-checkpoint-manifest-v1"
)
_LOCAL_HF_ENDPOINT_IDEMPOTENCY_TABLE: Final = "local_hf_endpoint_idempotency_v1"
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
    def _rollback_idempotency_transaction(connection: sqlite3.Connection) -> None:
        if connection.in_transaction:
            connection.execute("ROLLBACK")

    def start(self) -> None:
        self._runtime.load()

    def stop(self) -> None:
        self._runtime.close()

    def health(self) -> LocalHfHealthResponse:
        if not self._runtime.loaded:
            raise LocalHuggingFaceRuntimeError("model is not loaded")
        return LocalHfHealthResponse(
            model_identifier=self._model_identifier,
            model_version=self._model_version,
            checkpoint_identity=self._checkpoint_identity,
        )

    def infer_idempotently(
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
                response = self.infer(request)
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

    def infer(self, request: LocalHfEndpointRequest) -> LocalHfEndpointResponse:
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

    return app


__all__ = [
    "LOCAL_HF_CHECKPOINT_MANIFEST_VERSION",
    "LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER",
    "LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION",
    "LOCAL_HF_ENDPOINT_REQUEST_VERSION",
    "LOCAL_HF_ENDPOINT_RESPONSE_VERSION",
    "LocalHfCheckpointIdentity",
    "LocalHfEncodedImage",
    "LocalHfEndpointIdempotencyConflictError",
    "LocalHfEndpointRequest",
    "LocalHfEndpointResponse",
    "LocalHfEndpointService",
    "LocalHfHealthResponse",
    "LocalVisionRuntime",
    "build_local_hf_checkpoint_identity",
    "create_local_hf_endpoint_app",
    "load_local_hf_checkpoint_identity",
    "write_local_hf_checkpoint_identity",
]
