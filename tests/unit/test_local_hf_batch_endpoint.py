from __future__ import annotations

import base64
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.local_hf_endpoint import (
    LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION,
    LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION,
    LOCAL_HF_BATCH_IDEMPOTENCY_POLICY_VERSION,
    LOCAL_HF_BATCH_INFER_PATH,
    LOCAL_HF_BATCH_POLICY_VERSION,
    LocalHfBatchEndpointMemberRequest,
    LocalHfBatchEndpointMemberResponse,
    LocalHfBatchEndpointRequest,
    LocalHfCheckpointIdentity,
    LocalHfEncodedImage,
    LocalHfEndpointIdempotencyConflictError,
    LocalHfEndpointRequest,
    LocalHfEndpointService,
    build_local_hf_batch_request_sha256,
    create_local_hf_endpoint_app,
)
from robata.inference.local_hf_runtime import (
    LocalHfBatchGenerationObservation,
    LocalHfBatchGenerationRequest,
    LocalHfBatchMemberObservation,
    LocalHfGenerationObservation,
    LocalHfLoadObservation,
    LocalHuggingFaceRuntimeError,
)


class _BatchRuntime:
    def __init__(self) -> None:
        self.loaded = False
        self.load_observation = LocalHfLoadObservation(
            load_seconds=1.25,
            gpu_name="fake-gpu",
            gpu_total_bytes=8_000_000_000,
            gpu_free_before_bytes=7_000_000_000,
            gpu_allocated_after_load_bytes=3_000_000_000,
        )
        self.serial_calls = 0
        self.batch_calls: list[tuple[LocalHfBatchGenerationRequest, ...]] = []
        self.fail_next_batch = False
        self.malformed_next_batch = False

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
        self.serial_calls += 1
        assert image_payloads
        assert max_new_tokens == 32
        return LocalHfGenerationObservation(
            rendered_image_sizes=((320, 180),),
            prompt_tokens=24,
            output_tokens=7,
            generation_seconds=0.75,
            gpu_peak_allocated_bytes=3_500_000_000,
            output_text=f'{{"prompt":"{prompt}"}}',
        )

    def generate_batch(
        self,
        *,
        requests: Sequence[LocalHfBatchGenerationRequest],
    ) -> LocalHfBatchGenerationObservation:
        normalized = tuple(requests)
        self.batch_calls.append(normalized)
        if self.fail_next_batch:
            self.fail_next_batch = False
            raise LocalHuggingFaceRuntimeError("injected native batch failure")
        if self.malformed_next_batch:
            self.malformed_next_batch = False
            return LocalHfBatchGenerationObservation(
                members=(
                    LocalHfBatchMemberObservation(
                        rendered_image_sizes=(),
                        prompt_tokens=20,
                        output_tokens=5,
                        output_text='{"malformed":true}',
                    ),
                ),
                physical_generation_seconds=1.5,
                physical_gpu_peak_allocated_bytes=4_000_000_000,
            )
        return LocalHfBatchGenerationObservation(
            members=tuple(
                LocalHfBatchMemberObservation(
                    rendered_image_sizes=tuple((320, 180) for _ in request.image_payloads),
                    prompt_tokens=20 + index,
                    output_tokens=5 + index,
                    output_text=f'{{"prompt":"{request.prompt}"}}',
                )
                for index, request in enumerate(normalized)
            ),
            physical_generation_seconds=1.5,
            physical_gpu_peak_allocated_bytes=4_000_000_000,
        )


def _identity() -> LocalHfCheckpointIdentity:
    return LocalHfCheckpointIdentity(
        manifest_sha256="a" * 64,
        included_file_count=1,
        hf_revision="test-revision",
    )


def _service(runtime: _BatchRuntime, path: Path) -> LocalHfEndpointService:
    return LocalHfEndpointService(
        runtime=runtime,
        model_identifier="Qwen3-VL-4B-Instruct",
        model_version="local-test",
        checkpoint_identity=_identity(),
        idempotency_state_path=path,
    )


def _image(payload: bytes) -> LocalHfEncodedImage:
    return LocalHfEncodedImage(
        camera_id="cam_01",
        sha256=exact_bytes_sha256(payload),
        base64_data=base64.b64encode(payload).decode("ascii"),
    )


def _member(key: str, prompt: str) -> LocalHfBatchEndpointMemberRequest:
    return LocalHfBatchEndpointMemberRequest(
        idempotency_key=key,
        request=LocalHfEndpointRequest(
            request_id=f"request-{key}",
            images=[_image(f"image-{key}".encode())],
            prompt=prompt,
            max_new_tokens=32,
        ),
    )


def _batch(*members: LocalHfBatchEndpointMemberRequest) -> LocalHfBatchEndpointRequest:
    values = list(members)
    return LocalHfBatchEndpointRequest(
        batch_request_sha256=build_local_hf_batch_request_sha256(members=values),
        members=values,
    )


def test_serial_response_contract_bytes_are_unchanged(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    response = _service(runtime, tmp_path / "state.sqlite3").infer(
        LocalHfEndpointRequest(
            request_id="serial-request",
            images=[_image(b"serial-image")],
            prompt="serial",
            max_new_tokens=32,
        )
    )
    expected = {
        "contract_version": "local-hf-vision-response-v1",
        "request_id": "serial-request",
        "model_identifier": "Qwen3-VL-4B-Instruct",
        "model_version": "local-test",
        "quantization": "bnb-nf4-double-quant",
        "precision": "bfloat16-compute",
        "input_image_count": 1,
        "rendered_image_sizes": [[320, 180]],
        "prompt_tokens": 24,
        "output_tokens": 7,
        "load_seconds": 1.25,
        "generation_seconds": 0.75,
        "gpu_name": "fake-gpu",
        "gpu_total_bytes": 8_000_000_000,
        "gpu_free_before_bytes": 7_000_000_000,
        "gpu_allocated_after_load_bytes": 3_000_000_000,
        "gpu_peak_allocated_bytes": 3_500_000_000,
        "output_text": '{"prompt":"serial"}',
    }
    assert canonical_json_bytes(response.model_dump(mode="json")) == canonical_json_bytes(expected)
    assert runtime.serial_calls == 1
    assert runtime.batch_calls == []


def test_all_misses_are_one_physical_call_with_truthful_top_level_timing(
    tmp_path: Path,
) -> None:
    runtime = _BatchRuntime()
    request = _batch(_member("a", "first"), _member("b", "second"))
    response = _service(runtime, tmp_path / "state.sqlite3").infer_batch_idempotently(
        request=request
    )
    assert len(runtime.batch_calls) == 1
    assert [item.prompt for item in runtime.batch_calls[0]] == ["first", "second"]
    assert response.contract_version == LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION
    assert response.batch_request_sha256 == request.batch_request_sha256
    assert (response.generated_member_count, response.replay_member_count) == (2, 0)
    assert response.physical_generation_seconds == 1.5
    assert response.physical_gpu_peak_allocated_bytes == 4_000_000_000
    assert [item.request_id for item in response.members] == ["request-a", "request-b"]
    assert [item.output_text for item in response.members] == [
        '{"prompt":"first"}',
        '{"prompt":"second"}',
    ]
    for member in response.members:
        fields = member.model_dump(mode="json")
        assert member.disposition == "GENERATED"
        assert "generation_seconds" not in fields
        assert "gpu_peak_allocated_bytes" not in fields
        assert "load_seconds" not in fields


def test_all_replay_uses_no_physical_call(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    service = _service(runtime, tmp_path / "state.sqlite3")
    request = _batch(_member("a", "first"), _member("b", "second"))
    first = service.infer_batch_idempotently(request=request)
    replay = service.infer_batch_idempotently(request=request)
    assert len(runtime.batch_calls) == 1
    assert (first.generated_member_count, replay.generated_member_count) == (2, 0)
    assert replay.replay_member_count == 2
    assert replay.physical_generation_seconds == 0.0
    assert replay.physical_gpu_peak_allocated_bytes == 0
    assert all(item.disposition == "REPLAY" for item in replay.members)


def test_mixed_replay_batches_only_misses_and_restores_order(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    service = _service(runtime, tmp_path / "state.sqlite3")
    service.infer_batch_idempotently(request=_batch(_member("a", "first")))
    response = service.infer_batch_idempotently(
        request=_batch(_member("b", "second"), _member("a", "first"), _member("c", "third"))
    )
    assert len(runtime.batch_calls) == 2
    assert [item.prompt for item in runtime.batch_calls[1]] == ["second", "third"]
    assert [item.request_id for item in response.members] == [
        "request-b",
        "request-a",
        "request-c",
    ]
    assert [item.disposition for item in response.members] == [
        "GENERATED",
        "REPLAY",
        "GENERATED",
    ]


def test_key_conflicts_on_different_request_or_policy(tmp_path: Path) -> None:
    state_path = tmp_path / "state.sqlite3"
    runtime = _BatchRuntime()
    service = _service(runtime, state_path)
    service.infer_batch_idempotently(request=_batch(_member("a", "first")))
    with pytest.raises(LocalHfEndpointIdempotencyConflictError, match="different request"):
        service.infer_batch_idempotently(request=_batch(_member("a", "changed")))
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            "UPDATE local_hf_endpoint_batch_idempotency_v1 "
            "SET batch_policy_version = ? WHERE idempotency_key = ?",
            ("future-policy", "a"),
        )
    with pytest.raises(LocalHfEndpointIdempotencyConflictError, match="batch policy"):
        service.infer_batch_idempotently(request=_batch(_member("a", "first")))
    assert len(runtime.batch_calls) == 1


def test_runtime_failure_rolls_back_every_miss(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    service = _service(runtime, tmp_path / "state.sqlite3")
    request = _batch(_member("a", "first"), _member("b", "second"))
    runtime.fail_next_batch = True
    with pytest.raises(LocalHuggingFaceRuntimeError, match="injected native batch failure"):
        service.infer_batch_idempotently(request=request)
    retry = service.infer_batch_idempotently(request=request)
    assert len(runtime.batch_calls) == 2
    assert (retry.generated_member_count, retry.replay_member_count) == (2, 0)


def test_replay_survives_service_restart(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    request = _batch(_member("a", "first"), _member("b", "second"))
    first_runtime = _BatchRuntime()
    first_service = _service(first_runtime, path)
    first_service.start()
    generated = first_service.infer_batch_idempotently(request=request)
    first_service.stop()
    restarted_runtime = _BatchRuntime()
    restarted_service = _service(restarted_runtime, path)
    restarted_service.start()
    replayed = restarted_service.infer_batch_idempotently(request=request)
    assert restarted_runtime.batch_calls == []
    assert replayed.replay_member_count == 2
    assert [item.output_text for item in replayed.members] == [
        item.output_text for item in generated.members
    ]
    restarted_service.stop()


def test_health_and_fastapi_strict_batch_route(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    service = _service(runtime, tmp_path / "state.sqlite3")
    app = create_local_hf_endpoint_app(service)
    valid = _batch(_member("a", "first"), _member("b", "second"))
    oversized = [_member(f"k{i}", f"p{i}").model_dump(mode="json") for i in range(9)]
    with TestClient(app) as client:
        health = client.get("/healthz")
        success = client.post(LOCAL_HF_BATCH_INFER_PATH, json=valid.model_dump(mode="json"))
        empty = client.post(
            LOCAL_HF_BATCH_INFER_PATH,
            json={
                "contract_version": LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION,
                "batch_policy_version": LOCAL_HF_BATCH_POLICY_VERSION,
                "batch_request_sha256": "0" * 64,
                "members": [],
            },
        )
        maximum = client.post(
            LOCAL_HF_BATCH_INFER_PATH,
            json={
                "contract_version": LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION,
                "batch_policy_version": LOCAL_HF_BATCH_POLICY_VERSION,
                "batch_request_sha256": "0" * 64,
                "members": oversized,
            },
        )
        unknown = client.post(
            LOCAL_HF_BATCH_INFER_PATH,
            json={**valid.model_dump(mode="json"), "secret": "rejected"},
        )
    body = health.json()
    assert health.status_code == 200
    assert body["concurrency"] == 1
    assert body["native_batch_available"] is True
    assert body["native_batch_request_version"] == LOCAL_HF_BATCH_ENDPOINT_REQUEST_VERSION
    assert body["native_batch_response_version"] == LOCAL_HF_BATCH_ENDPOINT_RESPONSE_VERSION
    assert body["native_batch_policy_version"] == LOCAL_HF_BATCH_POLICY_VERSION
    assert (
        body["native_batch_idempotency_policy_version"] == LOCAL_HF_BATCH_IDEMPOTENCY_POLICY_VERSION
    )
    assert body["native_batch_max_size"] == 8
    assert success.status_code == 200
    assert empty.status_code == maximum.status_code == unknown.status_code == 422
    assert len(runtime.batch_calls) == 1


def test_request_rejects_wrong_identity_duplicate_keys_and_token_mismatch() -> None:
    first, second = _member("a", "first"), _member("b", "second")
    with pytest.raises(ValueError, match="batch_request_sha256"):
        LocalHfBatchEndpointRequest(batch_request_sha256="0" * 64, members=[first, second])
    duplicates = [first, _member("a", "same-key")]
    with pytest.raises(ValueError, match="idempotency keys must be unique"):
        LocalHfBatchEndpointRequest(
            batch_request_sha256=build_local_hf_batch_request_sha256(members=duplicates),
            members=duplicates,
        )
    changed = LocalHfBatchEndpointMemberRequest(
        idempotency_key="b",
        request=second.request.model_copy(update={"max_new_tokens": 64}),
    )
    mixed = [first, changed]
    with pytest.raises(ValueError, match="same max_new_tokens"):
        LocalHfBatchEndpointRequest(
            batch_request_sha256=build_local_hf_batch_request_sha256(members=mixed),
            members=mixed,
        )


def test_member_response_rejects_rendered_image_count_drift() -> None:
    with pytest.raises(ValueError, match="rendered_image_sizes"):
        LocalHfBatchEndpointMemberResponse(
            idempotency_key="member",
            request_id="request-member",
            disposition="GENERATED",
            input_image_count=2,
            rendered_image_sizes=((320, 180),),
            prompt_tokens=20,
            output_tokens=5,
            output_text='{"ok":true}',
        )


def test_malformed_runtime_member_rolls_back_before_replay(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    service = _service(runtime, tmp_path / "state.sqlite3")
    request = _batch(_member("a", "first"))
    runtime.malformed_next_batch = True

    with pytest.raises(LocalHuggingFaceRuntimeError, match="idempotency operation failed"):
        service.infer_batch_idempotently(request=request)

    retry = service.infer_batch_idempotently(request=request)
    assert len(runtime.batch_calls) == 2
    assert (retry.generated_member_count, retry.replay_member_count) == (1, 0)


def test_serial_only_runtime_does_not_advertise_or_register_batch(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    runtime.generate_batch = None  # type: ignore[method-assign,assignment]
    service = _service(runtime, tmp_path / "state.sqlite3")
    app = create_local_hf_endpoint_app(service)
    request = _batch(_member("a", "first"))
    with pytest.raises(LocalHuggingFaceRuntimeError, match="does not expose"):
        service.infer_batch_idempotently(request=request)

    with TestClient(app) as client:
        health = client.get("/healthz")
        batch = client.post(LOCAL_HF_BATCH_INFER_PATH, json=request.model_dump(mode="json"))

    body = health.json()
    assert body["native_batch_available"] is False
    assert body["native_batch_request_version"] is None
    assert body["native_batch_response_version"] is None
    assert body["native_batch_policy_version"] is None
    assert body["native_batch_idempotency_policy_version"] is None
    assert body["native_batch_max_size"] is None
    assert batch.status_code == 404


class _BlockingBatchRuntime(_BatchRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def generate_batch(
        self,
        *,
        requests: Sequence[LocalHfBatchGenerationRequest],
    ) -> LocalHfBatchGenerationObservation:
        self.started.set()
        if not self.release.wait(5.0):
            raise LocalHuggingFaceRuntimeError("blocking test timed out")
        return super().generate_batch(requests=requests)


def test_stop_drains_run_to_completion_batch_before_closing_runtime(tmp_path: Path) -> None:
    runtime = _BlockingBatchRuntime()
    service = _service(runtime, tmp_path / "state.sqlite3")
    service.start()
    errors: list[BaseException] = []
    request_done = Event()
    stop_done = Event()

    def invoke() -> None:
        try:
            service.infer_batch_idempotently(request=_batch(_member("a", "first")))
        except BaseException as error:  # pragma: no cover - assertion captures worker failures
            errors.append(error)
        finally:
            request_done.set()

    def stop() -> None:
        service.stop()
        stop_done.set()

    request_thread = Thread(target=invoke, daemon=True)
    request_thread.start()
    assert runtime.started.wait(1.0)
    stop_thread = Thread(target=stop, daemon=True)
    stop_thread.start()
    assert not stop_done.wait(0.05)
    assert runtime.loaded is True

    runtime.release.set()
    assert request_done.wait(2.0)
    assert stop_done.wait(2.0)
    request_thread.join(timeout=0.1)
    stop_thread.join(timeout=0.1)
    assert errors == []
    assert runtime.loaded is False


def test_batch_member_replay_is_canonical_and_grouping_independent(tmp_path: Path) -> None:
    runtime = _BatchRuntime()
    service = _service(runtime, tmp_path / "state.sqlite3")
    app = create_local_hf_endpoint_app(service)
    request = _batch(_member("a", "first"))
    canonical = canonical_json_bytes(request.model_dump(mode="json"))

    with TestClient(app) as client:
        generated = client.post(
            LOCAL_HF_BATCH_INFER_PATH,
            content=canonical,
            headers={"Content-Type": "application/json"},
        )
        replayed = client.post(
            LOCAL_HF_BATCH_INFER_PATH,
            content=b"\n" + canonical + b"\n",
            headers={"Content-Type": "application/json"},
        )

    assert generated.status_code == replayed.status_code == 200
    assert generated.json()["members"][0]["disposition"] == "GENERATED"
    assert replayed.json()["members"][0]["disposition"] == "REPLAY"
    assert len(runtime.batch_calls) == 1
