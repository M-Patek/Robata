"""Loopback-only HTTP contract for one resident local Hugging Face vision model."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, StringConstraints

from robata.contracts.common import Sha256Digest, StrictModel
from robata.contracts.hashing import exact_bytes_sha256
from robata.inference.local_hf_runtime import LocalHuggingFaceRuntimeError

LOCAL_HF_ENDPOINT_REQUEST_VERSION: Literal["local-hf-vision-request-v1"] = (
    "local-hf-vision-request-v1"
)
LOCAL_HF_ENDPOINT_RESPONSE_VERSION: Literal["local-hf-vision-response-v1"] = (
    "local-hf-vision-response-v1"
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


class LocalHfHealthResponse(StrictModel):
    status: Literal["READY"] = "READY"
    model_identifier: NonEmptyString
    model_version: NonEmptyString
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


class LocalHfEndpointService:
    """Translate the HTTP contract to one serialized resident runtime."""

    def __init__(
        self,
        *,
        runtime: LocalVisionRuntime,
        model_identifier: str,
        model_version: str,
    ) -> None:
        if not isinstance(model_identifier, str) or not model_identifier:
            raise ValueError("model_identifier must be nonempty")
        if not isinstance(model_version, str) or not model_version:
            raise ValueError("model_version must be nonempty")
        self._runtime = runtime
        self._model_identifier = model_identifier
        self._model_version = model_version

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
        )

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

    @app.get("/healthz", response_model=LocalHfHealthResponse)  # type: ignore[untyped-decorator]
    async def health() -> LocalHfHealthResponse:
        try:
            return service.health()
        except LocalHuggingFaceRuntimeError as error:
            raise fastapi.HTTPException(status_code=503, detail=str(error)) from error

    @app.post(  # type: ignore[untyped-decorator]
        "/v1/local-vision/infer", response_model=LocalHfEndpointResponse
    )
    async def infer(request: LocalHfEndpointRequest) -> LocalHfEndpointResponse:
        try:
            return await asyncio.to_thread(service.infer, request)
        except LocalHuggingFaceRuntimeError as error:
            raise fastapi.HTTPException(status_code=422, detail=str(error)) from error

    return app


__all__ = [
    "LOCAL_HF_ENDPOINT_REQUEST_VERSION",
    "LOCAL_HF_ENDPOINT_RESPONSE_VERSION",
    "LocalHfEncodedImage",
    "LocalHfEndpointRequest",
    "LocalHfEndpointResponse",
    "LocalHfEndpointService",
    "LocalHfHealthResponse",
    "LocalVisionRuntime",
    "create_local_hf_endpoint_app",
]
