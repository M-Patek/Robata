from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robata.contracts.hashing import exact_bytes_sha256
from robata.inference.local_hf_endpoint import (
    LocalHfEncodedImage,
    LocalHfEndpointRequest,
    LocalHfEndpointService,
    create_local_hf_endpoint_app,
)
from robata.inference.local_hf_runtime import (
    LocalHfGenerationObservation,
    LocalHfLoadObservation,
    LocalHuggingFaceRuntimeError,
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.loaded = False
        self.load_observation = LocalHfLoadObservation(
            load_seconds=1.25,
            gpu_name="fake-gpu",
            gpu_total_bytes=8_000_000_000,
            gpu_free_before_bytes=7_000_000_000,
            gpu_allocated_after_load_bytes=3_000_000_000,
        )
        self.last_payloads: list[bytes] | None = None

    def load(self) -> LocalHfLoadObservation:
        self.loaded = True
        return self.load_observation

    def close(self) -> None:
        self.loaded = False

    def generate(
        self,
        *,
        image_payloads: list[bytes],
        prompt: str,
        max_new_tokens: int,
    ) -> LocalHfGenerationObservation:
        assert prompt == "describe"
        assert max_new_tokens == 32
        self.last_payloads = image_payloads
        return LocalHfGenerationObservation(
            rendered_image_sizes=((320, 180),),
            prompt_tokens=24,
            output_tokens=7,
            generation_seconds=0.75,
            gpu_peak_allocated_bytes=3_500_000_000,
            output_text='{"scene_summary":"room"}',
        )


def _encoded_image(payload: bytes = b"image-bytes") -> LocalHfEncodedImage:
    return LocalHfEncodedImage(
        camera_id="cam_01",
        sha256=exact_bytes_sha256(payload),
        base64_data=base64.b64encode(payload).decode("ascii"),
    )


def test_loopback_service_preserves_request_binding_and_runtime_telemetry() -> None:
    runtime = _FakeRuntime()
    service = LocalHfEndpointService(
        runtime=runtime,
        model_identifier="Qwen3-VL-4B-Instruct",
        model_version="local-test",
    )
    service.start()

    health = service.health()
    response = service.infer(
        LocalHfEndpointRequest(
            request_id="request-1",
            images=[_encoded_image()],
            prompt="describe",
            max_new_tokens=32,
        )
    )

    assert health.status == "READY"
    assert health.concurrency == 1
    assert runtime.last_payloads == [b"image-bytes"]
    assert response.request_id == "request-1"
    assert response.model_identifier == "Qwen3-VL-4B-Instruct"
    assert response.gpu_peak_allocated_bytes == 3_500_000_000
    service.stop()
    with pytest.raises(LocalHuggingFaceRuntimeError, match="not loaded"):
        service.health()


def test_loopback_service_rejects_digest_mismatch() -> None:
    runtime = _FakeRuntime()
    runtime.load()
    service = LocalHfEndpointService(
        runtime=runtime,
        model_identifier="Qwen3-VL-4B-Instruct",
        model_version="local-test",
    )
    image = _encoded_image()
    image = image.model_copy(update={"sha256": "0" * 64})

    with pytest.raises(LocalHuggingFaceRuntimeError, match="digest mismatch"):
        service.infer(
            LocalHfEndpointRequest(
                request_id="request-2",
                images=[image],
                prompt="describe",
                max_new_tokens=32,
            )
        )


def test_loopback_contract_caps_request_at_six_images() -> None:
    images = [
        _encoded_image(f"image-{index}".encode()).model_copy(
            update={"camera_id": f"cam_{index:02d}"}
        )
        for index in range(1, 8)
    ]

    with pytest.raises(ValidationError):
        LocalHfEndpointRequest(
            request_id="request-3",
            images=images,
            prompt="describe",
            max_new_tokens=32,
        )


def test_fastapi_route_accepts_normal_json_image_array() -> None:
    runtime = _FakeRuntime()
    service = LocalHfEndpointService(
        runtime=runtime,
        model_identifier="Qwen3-VL-4B-Instruct",
        model_version="local-test",
    )
    application = create_local_hf_endpoint_app(service)
    image = _encoded_image()
    with TestClient(application) as client:
        response = client.post(
            "/v1/local-vision/infer",
            json={
                "contract_version": "local-hf-vision-request-v1",
                "request_id": "request-http",
                "images": [image.model_dump(mode="json")],
                "prompt": "describe",
                "max_new_tokens": 32,
            },
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "request-http"
