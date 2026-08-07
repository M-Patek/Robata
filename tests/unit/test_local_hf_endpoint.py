from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.local_hf_endpoint import (
    LOCAL_HF_CHECKPOINT_MANIFEST_VERSION,
    LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER,
    LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION,
    LocalHfCheckpointIdentity,
    LocalHfEncodedImage,
    LocalHfEndpointIdempotencyConflictError,
    LocalHfEndpointRequest,
    LocalHfEndpointService,
    build_local_hf_checkpoint_identity,
    create_local_hf_endpoint_app,
    load_local_hf_checkpoint_identity,
    write_local_hf_checkpoint_identity,
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
        self.generation_calls = 0

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
        self.generation_calls += 1
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


def _endpoint_request_body(
    image: LocalHfEncodedImage,
    *,
    request_id: str = "request-http",
) -> bytes:
    return canonical_json_bytes(
        {
            "contract_version": "local-hf-vision-request-v1",
            "request_id": request_id,
            "images": [image.model_dump(mode="json")],
            "prompt": "describe",
            "max_new_tokens": 32,
        }
    )


def _checkpoint_identity() -> LocalHfCheckpointIdentity:
    return LocalHfCheckpointIdentity(
        manifest_sha256="a" * 64,
        included_file_count=1,
        hf_revision="test-revision",
    )


def _service(runtime: _FakeRuntime, tmp_path: Path) -> LocalHfEndpointService:
    return LocalHfEndpointService(
        runtime=runtime,
        model_identifier="Qwen3-VL-4B-Instruct",
        model_version="local-test",
        checkpoint_identity=_checkpoint_identity(),
        idempotency_state_path=tmp_path / "endpoint-idempotency.sqlite3",
    )


def test_loopback_service_preserves_request_binding_and_runtime_telemetry(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path)
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
    assert health.checkpoint_identity == _checkpoint_identity()
    assert runtime.last_payloads == [b"image-bytes"]
    assert response.request_id == "request-1"
    assert response.model_identifier == "Qwen3-VL-4B-Instruct"
    assert response.gpu_peak_allocated_bytes == 3_500_000_000
    service.stop()
    with pytest.raises(LocalHuggingFaceRuntimeError, match="not loaded"):
        service.health()


def test_loopback_service_rejects_digest_mismatch(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    runtime.load()
    service = _service(runtime, tmp_path)
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


def test_fastapi_route_accepts_normal_json_image_array(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path)
    application = create_local_hf_endpoint_app(service)
    image = _encoded_image()
    with TestClient(application) as client:
        response = client.post(
            "/v1/local-vision/infer",
            content=_endpoint_request_body(image),
            headers={
                "Content-Type": "application/json",
                LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER: "request-http-key",
            },
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "request-http"
    assert runtime.generation_calls == 1


def test_fastapi_route_replays_same_idempotency_key_and_exact_body(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path)
    application = create_local_hf_endpoint_app(service)
    image = _encoded_image()
    body = _endpoint_request_body(image)
    headers = {
        "Content-Type": "application/json",
        LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER: "same-body-key",
    }

    with TestClient(application) as client:
        first = client.post("/v1/local-vision/infer", content=body, headers=headers)
        second = client.post("/v1/local-vision/infer", content=body, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert runtime.generation_calls == 1
    assert LOCAL_HF_ENDPOINT_IDEMPOTENCY_POLICY_VERSION == "sqlite-exact-body-replay-v1"


def test_fastapi_route_rejects_idempotency_key_reused_for_different_body(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path)
    application = create_local_hf_endpoint_app(service)
    image = _encoded_image()
    headers = {
        "Content-Type": "application/json",
        LOCAL_HF_ENDPOINT_IDEMPOTENCY_HEADER: "conflict-key",
    }

    body = _endpoint_request_body(image, request_id="first-request")
    with TestClient(application) as client:
        first = client.post(
            "/v1/local-vision/infer",
            content=body,
            headers=headers,
        )
        second = client.post(
            "/v1/local-vision/infer",
            # Trailing JSON whitespace parses to the same request model but is
            # intentionally different raw request bytes.
            content=body + b"\n",
            headers=headers,
        )

    assert first.status_code == 200
    assert second.status_code == 409
    assert "different request bytes" in second.json()["detail"]
    assert runtime.generation_calls == 1


def test_fastapi_route_requires_idempotency_key(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path)
    application = create_local_hf_endpoint_app(service)
    image = _encoded_image()

    with TestClient(application) as client:
        response = client.post(
            "/v1/local-vision/infer",
            content=_endpoint_request_body(image),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422
    assert runtime.generation_calls == 0


def test_service_replays_durable_idempotency_record_after_restart(tmp_path: Path) -> None:
    image = _encoded_image()
    request = LocalHfEndpointRequest(
        request_id="durable-request",
        images=[image],
        prompt="describe",
        max_new_tokens=32,
    )
    body = _endpoint_request_body(image, request_id="durable-request")
    first_runtime = _FakeRuntime()
    first_service = _service(first_runtime, tmp_path)
    first_service.start()

    first_response = first_service.infer_idempotently(
        request=request,
        idempotency_key="durable-key",
        request_body=body,
    )
    first_service.stop()

    restarted_runtime = _FakeRuntime()
    restarted_service = _service(restarted_runtime, tmp_path)
    restarted_service.start()
    replayed_response = restarted_service.infer_idempotently(
        request=request,
        idempotency_key="durable-key",
        request_body=body,
    )

    assert first_runtime.generation_calls == 1
    assert restarted_runtime.generation_calls == 0
    assert replayed_response == first_response

    with pytest.raises(LocalHfEndpointIdempotencyConflictError, match="different request bytes"):
        restarted_service.infer_idempotently(
            request=request,
            idempotency_key="durable-key",
            request_body=body + b"\n",
        )
    assert restarted_runtime.generation_calls == 0
    restarted_service.stop()


def test_checkpoint_identity_hashes_relevant_files_excludes_cache_and_persists_manifest(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "config.json").write_text(
        '{"_commit_hash":"revision-from-config","model_type":"qwen3_vl"}',
        encoding="utf-8",
    )
    (model_directory / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    (model_directory / "processor_config.json").write_text(
        '{"processor":"vision"}', encoding="utf-8"
    )
    (model_directory / "model-00001-of-00001.safetensors").write_bytes(b"weights-v1")
    (model_directory / "notes.txt").write_text("ignored", encoding="utf-8")
    cache_directory = model_directory / ".cache"
    cache_directory.mkdir()
    (cache_directory / "ignored.safetensors").write_bytes(b"cache-weights")

    identity = build_local_hf_checkpoint_identity(model_directory=model_directory)

    assert identity.manifest_version == LOCAL_HF_CHECKPOINT_MANIFEST_VERSION
    assert identity.hf_revision == "revision-from-config"
    assert identity.included_file_count == 4
    manifest_path = tmp_path / "checkpoint-identity.json"
    write_local_hf_checkpoint_identity(identity=identity, manifest_path=manifest_path)
    assert load_local_hf_checkpoint_identity(manifest_path=manifest_path) == identity

    (model_directory / "notes.txt").write_text("still ignored", encoding="utf-8")
    (cache_directory / "ignored.safetensors").write_bytes(b"different-cache-weights")
    assert build_local_hf_checkpoint_identity(model_directory=model_directory) == identity

    (model_directory / "model-00001-of-00001.safetensors").write_bytes(b"weights-v2")
    changed = build_local_hf_checkpoint_identity(model_directory=model_directory)
    assert changed.manifest_sha256 != identity.manifest_sha256


def test_checkpoint_identity_reads_consistent_huggingface_download_revision(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    (model_directory / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
    (model_directory / "model.safetensors").write_bytes(b"weights")
    metadata_root = model_directory / ".cache" / "huggingface" / "download"
    metadata_root.mkdir(parents=True)
    revision = "e" * 40
    (metadata_root / "config.json.metadata").write_text(
        f"{revision}\netag-a\n0\n", encoding="utf-8"
    )
    (metadata_root / "model.safetensors.metadata").write_text(
        f"{revision}\netag-b\n0\n", encoding="utf-8"
    )

    identity = build_local_hf_checkpoint_identity(model_directory=model_directory)

    assert identity.hf_revision == revision
    assert identity.included_file_count == 2
