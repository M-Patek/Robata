from __future__ import annotations

import asyncio
import base64
import importlib.util
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from robata.application.canonical.local_real_model import build_local_qwen_batch_model_binding
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.local_hf_adapter import (
    LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
    LocalHfHttpRequest,
    LocalHfTransportError,
)
from robata.inference.local_hf_endpoint import (
    LOCAL_HF_BATCH_POLICY_VERSION,
    LocalHfBatchEndpointMemberRequest,
    LocalHfBatchEndpointRequest,
    LocalHfBatchEndpointResponse,
    LocalHfCheckpointIdentity,
    LocalHfEncodedImage,
    LocalHfEndpointRequest,
    LocalHfEndpointService,
    build_local_hf_batch_request_sha256,
    create_local_hf_endpoint_app,
)
from robata.inference.local_hf_runtime import (
    LocalHfBatchGenerationObservation,
    LocalHfBatchMemberObservation,
    LocalHfLoadObservation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_local_qwen_batch_endpoint_smoke.py"


def _module() -> ModuleType:
    name = f"run_local_qwen_batch_endpoint_smoke_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRuntime:
    def __init__(self) -> None:
        self._loaded = False
        self._observation = LocalHfLoadObservation(
            load_seconds=0.25,
            gpu_name="fake-gpu",
            gpu_total_bytes=8_000,
            gpu_free_before_bytes=7_000,
            gpu_allocated_after_load_bytes=1_000,
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_observation(self) -> LocalHfLoadObservation:
        return self._observation

    def load(self) -> LocalHfLoadObservation:
        self._loaded = True
        return self._observation

    def close(self) -> None:
        self._loaded = False

    def generate(self, **kwargs: Any) -> None:
        del kwargs
        raise AssertionError("serial generation is forbidden in the Batch4 smoke")

    def generate_batch(self, **kwargs: Any) -> LocalHfBatchGenerationObservation:
        requests = kwargs["requests"]
        return LocalHfBatchGenerationObservation(
            members=tuple(
                LocalHfBatchMemberObservation(
                    rendered_image_sizes=((1, 1),),
                    prompt_tokens=11,
                    output_tokens=4,
                    output_text='{"claims":[],"abstained":true}',
                )
                for _request in requests
            ),
            physical_generation_seconds=0.5,
            physical_gpu_peak_allocated_bytes=2_000,
        )


def _batch_request() -> LocalHfBatchEndpointRequest:
    payload = b"fake-png"
    image = LocalHfEncodedImage(
        camera_id="cam_01",
        sha256=exact_bytes_sha256(payload),
        base64_data=base64.b64encode(payload).decode("ascii"),
    )
    members = tuple(
        LocalHfBatchEndpointMemberRequest(
            idempotency_key=f"member-{index}",
            request=LocalHfEndpointRequest(
                request_id=f"request-{index}",
                images=[image],
                prompt="return a compact result",
                max_new_tokens=16,
            ),
        )
        for index in range(4)
    )
    return LocalHfBatchEndpointRequest(
        batch_policy_version=LOCAL_HF_BATCH_POLICY_VERSION,
        batch_request_sha256=build_local_hf_batch_request_sha256(members=members),
        members=list(members),
    )


def test_counting_runtime_preserves_results_and_counts_calls() -> None:
    module = _module()
    delegate = _FakeRuntime()
    runtime = module.CountingLocalVisionRuntime(delegate)

    assert runtime.load() == delegate.load_observation
    observation = runtime.generate_batch(requests=(object(), object()))
    runtime.close()

    assert len(observation.members) == 2
    assert runtime.load_calls == 1
    assert runtime.generate_calls == 0
    assert runtime.generate_batch_calls == 1
    assert runtime.close_calls == 1
    assert not runtime.loaded


def test_httpx_asgi_transport_preserves_exact_http_response_bytes() -> None:
    module = _module()

    async def app(scope: dict[str, object], receive: Any, send: Any) -> None:
        assert scope["type"] == "http"
        assert scope["path"] == "/v1/local-vision/infer-batch"
        event = await receive()
        assert event["body"] == b"{}"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"ok":true}'})

    transport = module.HttpxAsgiLocalHfTransport(app)
    request = LocalHfHttpRequest(
        url=LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
        body=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=100,
        idempotency_key="outer-key",
    )
    response = asyncio.run(transport.post(request))

    assert response.status_code == 200
    assert response.body == b'{"ok":true}'
    assert len(transport.exchanges) == 1
    assert transport.exchanges[0].request == request


def test_httpx_asgi_transport_fails_closed_on_response_limit() -> None:
    module = _module()

    async def app(_scope: dict[str, object], _receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"0123456789"})

    request = LocalHfHttpRequest(
        url=LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
        body=b"{}",
        timeout_seconds=1.0,
        max_response_bytes=5,
        idempotency_key="outer-key",
    )
    with pytest.raises(LocalHfTransportError, match="exceeded byte limit"):
        asyncio.run(module.HttpxAsgiLocalHfTransport(app).post(request))


def test_fake_endpoint_first_pass_generates_and_second_pass_replays() -> None:
    module = _module()
    tmp_path = REPOSITORY_ROOT / ".tmp" / f"endpoint-smoke-test-{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True)
    runtime = module.CountingLocalVisionRuntime(_FakeRuntime())
    service = LocalHfEndpointService(
        runtime=runtime,
        model_identifier="Qwen3-VL-4B-Instruct",
        model_version="local-2026-08-06",
        checkpoint_identity=LocalHfCheckpointIdentity(
            manifest_sha256="1" * 64,
            included_file_count=1,
        ),
        idempotency_state_path=tmp_path / "endpoint.sqlite3",
    )
    transport = module.HttpxAsgiLocalHfTransport(create_local_hf_endpoint_app(service))
    batch = _batch_request()
    http_request = LocalHfHttpRequest(
        url=LOCAL_HF_LOOPBACK_BATCH_ENDPOINT_URL,
        body=canonical_json_bytes(batch.model_dump(mode="json")),
        timeout_seconds=5.0,
        max_response_bytes=1_000_000,
        idempotency_key="outer-key",
    )

    service.start()
    try:
        first = asyncio.run(transport.post(http_request))
        replay = asyncio.run(transport.post(http_request))
    finally:
        service.stop()

    first_contract = LocalHfBatchEndpointResponse.model_validate_json(first.body, strict=True)
    replay_contract = LocalHfBatchEndpointResponse.model_validate_json(replay.body, strict=True)
    assert first_contract.generated_member_count == 4
    assert first_contract.replay_member_count == 0
    assert {member.disposition for member in first_contract.members} == {"GENERATED"}
    assert replay_contract.generated_member_count == 0
    assert replay_contract.replay_member_count == 4
    assert {member.disposition for member in replay_contract.members} == {"REPLAY"}
    assert runtime.generate_calls == 0
    assert runtime.generate_batch_calls == 1
    assert module._sqlite_row_counts(
        tmp_path / "endpoint.sqlite3",
        (module._ENDPOINT_SERIAL_TABLE, module._ENDPOINT_BATCH_TABLE),
    ) == {
        module._ENDPOINT_SERIAL_TABLE: 0,
        module._ENDPOINT_BATCH_TABLE: 4,
    }


def test_route_projection_binds_canonical_adapter_and_endpoint_policies() -> None:
    module = _module()
    adapter = module._build_selection_adapter(
        checkpoint_manifest_sha256=module.DEFAULT_CHECKPOINT_MANIFEST_SHA256
    )
    binding = build_local_qwen_batch_model_binding(
        transport=module._FailClosedTransport(),
        checkpoint_manifest_sha256=module.DEFAULT_CHECKPOINT_MANIFEST_SHA256,
    )

    route = module._route_projection(binding, adapter)

    assert route["adapter_policy_version"] == "local-qwen-task-claim-group-hybrid-batch-v1"
    assert route["endpoint_policy_version"] == "local-hf-native-batch-policy-v1"
    assert route["adapter_max_batch_size"] == 4
    assert route["endpoint_max_batch_size"] == 8
    assert route["canonical_runtime_capacity_projection"]["max_batch_size"] == 4


def test_checkpoint_file_count_excludes_cache_payloads() -> None:
    module = _module()
    tmp_path = REPOSITORY_ROOT / ".tmp" / f"checkpoint-count-test-{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "model-copy.safetensors").write_bytes(b"ignored")
    (tmp_path / "README.md").write_text("ignored", encoding="utf-8")

    assert module._checkpoint_selected_file_count(tmp_path) == 3
