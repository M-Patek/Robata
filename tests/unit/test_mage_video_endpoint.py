from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.mage_video_endpoint import (
    MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER,
    MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_POLICY_VERSION,
    MageVideoCameraEncoding,
    MageVideoCodecPolicy,
    MageVideoDecoderRequest,
    MageVideoEndpointIdempotencyConflictError,
    MageVideoEndpointRequest,
    MageVideoEndpointService,
    MageVideoGenerationJsonlSink,
    MageVideoModelIdentity,
    MageVideoNeuralCodecParameters,
    MageVideoResultArtifactDocument,
    MageVideoRuntimeIdentityBinding,
    build_mage_video_context_manifest,
    build_mage_video_segment_manifest,
    create_mage_video_endpoint_app,
)
from robata.inference.mage_video_runtime import (
    MageVideoCodecCacheBinding,
    MageVideoGenerationObservation,
    MageVideoGenerationTelemetry,
    MageVideoLoadObservation,
    MageVideoLoadProfile,
    MageVideoRuntimeError,
    MageVideoRuntimeIdentity,
)


class _FakeRuntime:
    def __init__(
        self,
        *,
        runtime_identity: MageVideoRuntimeIdentity | None = None,
        load_observation_identity: MageVideoRuntimeIdentity | None = None,
        generation_telemetry: MageVideoGenerationTelemetry | None = None,
    ) -> None:
        self.loaded = False
        self.runtime_identity = runtime_identity or MageVideoRuntimeIdentity()
        self.load_calls = 0
        self.close_calls = 0
        self.generate_calls = 0
        self.last_video_paths: list[Path | str] | None = None
        self.last_prompt: str | None = None
        self.last_max_new_tokens: int | None = None
        self.last_codec_config: dict[str, Any] | None = None
        self.last_codec_cache_binding: MageVideoCodecCacheBinding | None = None
        self.load_observation = MageVideoLoadObservation(
            load_seconds=1.25,
            execution_device="cuda:0",
            runtime_identity=load_observation_identity or self.runtime_identity,
        )
        self.generation_telemetry = generation_telemetry

    def load(self) -> MageVideoLoadObservation:
        self.loaded = True
        self.load_calls += 1
        return self.load_observation

    def close(self) -> None:
        self.loaded = False
        self.close_calls += 1

    def generate(
        self,
        *,
        video_paths: Sequence[Path | str],
        prompt: str,
        max_new_tokens: int,
        codec_config: Mapping[str, Any],
        codec_cache_binding: MageVideoCodecCacheBinding | None = None,
    ) -> MageVideoGenerationObservation:
        assert self.loaded is True
        self.generate_calls += 1
        self.last_video_paths = list(video_paths)
        self.last_prompt = prompt
        self.last_max_new_tokens = max_new_tokens
        self.last_codec_config = dict(codec_config)
        self.last_codec_cache_binding = codec_cache_binding
        return MageVideoGenerationObservation(
            input_video_count=len(video_paths),
            prompt_tokens=18,
            output_tokens=6,
            generation_seconds=0.75,
            output_text='{"scene_summary":"room"}',
            telemetry=self.generation_telemetry,
        )


class _BlockingRuntime(_FakeRuntime):
    """Test double that exposes whether independent calls reach generation."""

    def __init__(self) -> None:
        super().__init__()
        self.first_generation_entered = Event()
        self.second_generation_entered = Event()
        self.release_generation = Event()
        self._entry_count = 0
        self._entry_lock = Lock()

    def generate(
        self,
        *,
        video_paths: Sequence[Path | str],
        prompt: str,
        max_new_tokens: int,
        codec_config: Mapping[str, Any],
        codec_cache_binding: MageVideoCodecCacheBinding | None = None,
    ) -> MageVideoGenerationObservation:
        with self._entry_lock:
            self._entry_count += 1
            if self._entry_count == 1:
                self.first_generation_entered.set()
            else:
                self.second_generation_entered.set()
        assert self.release_generation.wait(timeout=5.0)
        return super().generate(
            video_paths=video_paths,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            codec_config=codec_config,
            codec_cache_binding=codec_cache_binding,
        )


def _model_identity(
    *,
    revision: str = "revision-1",
    runtime_identity: MageVideoRuntimeIdentity | None = None,
) -> MageVideoModelIdentity:
    return MageVideoModelIdentity(
        model_identifier="mage-video-base",
        model_revision=revision,
        checkpoint_manifest_sha256="a" * 64,
        runtime_identity=MageVideoRuntimeIdentityBinding.from_runtime_identity(
            runtime_identity or MageVideoRuntimeIdentity()
        ),
    )


def _request(
    tmp_path: Path,
    *,
    request_id: str = "request-1",
    model_identity: MageVideoModelIdentity | None = None,
    codec_policy: MageVideoCodecPolicy | None = None,
) -> tuple[MageVideoEndpointRequest, Path]:
    durable_root = tmp_path / "durable"
    durable_root.mkdir(exist_ok=True)
    segment_path = durable_root / f"{request_id}.mp4"
    segment_payload = b"durable-video-segment"
    segment_path.write_bytes(segment_payload)
    segment = build_mage_video_segment_manifest(
        segment_id=f"segment-{request_id}",
        camera_id="cam-01",
        durable_path=str(segment_path),
        media_type="video/mp4",
        content_sha256=exact_bytes_sha256(segment_payload),
        byte_count=len(segment_payload),
    )
    context = build_mage_video_context_manifest(
        context_id=f"context-{request_id}",
        context_payload_sha256=exact_bytes_sha256(b"durable-context-manifest"),
        segment_manifest_identities=[segment.manifest_identity],
    )
    return (
        MageVideoEndpointRequest(
            request_id=request_id,
            model_identity=model_identity or _model_identity(),
            codec_policy=codec_policy or MageVideoCodecPolicy(preprocess_device="cpu"),
            context_manifest=context,
            camera_encodings=[
                MageVideoCameraEncoding(
                    encoder_id="camera-encoder-cam-01",
                    segment_manifest=segment,
                )
            ],
            decoder=MageVideoDecoderRequest(
                decoder_id="shared-decoder-01",
                prompt="Describe the segment.",
                max_new_tokens=32,
            ),
        ),
        durable_root,
    )


def _service(
    runtime: _FakeRuntime,
    tmp_path: Path,
    durable_root: Path,
    *,
    model_identity: MageVideoModelIdentity | None = None,
    generation_telemetry_sink: Any | None = None,
    codec_cache_admission: Any | None = None,
) -> MageVideoEndpointService:
    return MageVideoEndpointService(
        runtime=runtime,
        model_identity=model_identity or _model_identity(),
        idempotency_state_path=tmp_path / "endpoint-idempotency.sqlite3",
        result_artifact_directory=tmp_path / "results",
        durable_input_roots=[durable_root],
        generation_telemetry_sink=generation_telemetry_sink,
        codec_cache_admission=codec_cache_admission,
    )


def _request_body(request: MageVideoEndpointRequest) -> bytes:
    return canonical_json_bytes(request.model_dump(mode="json"))


def test_service_applies_codec_cache_admission_before_generation(tmp_path: Path) -> None:
    request, durable_root = _request(tmp_path, request_id="cache-admitted")
    runtime = _FakeRuntime()
    admitted: list[tuple[MageVideoEndpointRequest, list[Path]]] = []
    provider_cache = tmp_path / "provider-cache"
    provider_cache.mkdir()
    (provider_cache / "meta.json").write_text("{}", encoding="utf-8")
    (provider_cache / "src_patch_position.npy").write_bytes(b"positions")
    binding = MageVideoCodecCacheBinding(
        source_path=Path(request.camera_encodings[0].segment_manifest.durable_path),
        provider_cache_directory=provider_cache,
    )

    def admission(
        candidate: MageVideoEndpointRequest,
        paths: Sequence[Path],
    ) -> MageVideoCodecCacheBinding:
        assert runtime.generate_calls == 0
        admitted.append((candidate, list(paths)))
        return binding

    service = _service(
        runtime,
        tmp_path,
        durable_root,
        codec_cache_admission=admission,
    )
    service.start()
    service.infer(request)

    assert admitted == [
        (
            request,
            [Path(request.camera_encodings[0].segment_manifest.durable_path)],
        )
    ]
    assert runtime.generate_calls == 1
    assert runtime.last_codec_cache_binding == binding


def test_codec_cache_admission_rejection_prevents_generation(tmp_path: Path) -> None:
    request, durable_root = _request(tmp_path, request_id="cache-rejected")
    runtime = _FakeRuntime()

    def reject(_request: MageVideoEndpointRequest, _paths: Sequence[Path]) -> None:
        raise RuntimeError("verified Provider V2 cache admission rejected")

    service = _service(
        runtime,
        tmp_path,
        durable_root,
        codec_cache_admission=reject,
    )
    service.start()

    with pytest.raises(RuntimeError, match="cache admission rejected"):
        service.infer(request)

    assert runtime.generate_calls == 0
    assert not list((tmp_path / "results").glob("*.json"))


def test_service_uses_durable_video_manifest_and_persists_explicit_result_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_created_at = "2026-08-07T12:34:56.123456Z"
    monkeypatch.setattr(
        "robata.inference.mage_video_endpoint._server_authored_rfc3339_utc_timestamp",
        lambda: fixed_created_at,
    )
    request, durable_root = _request(tmp_path)
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path, durable_root)
    service.start()

    response = service.infer(request)

    assert runtime.last_video_paths == [
        Path(request.camera_encodings[0].segment_manifest.durable_path)
    ]
    assert runtime.last_prompt == "Describe the segment."
    assert runtime.last_max_new_tokens == 32
    assert runtime.last_codec_config is not None
    assert runtime.last_codec_config["engine"] == "hevc"
    assert runtime.last_codec_config["preprocess_device"] == "cpu"
    assert response.preprocess_device == "cpu"
    assert response.contract_version == "mage-video-codec-response-v2"
    assert response.inference_identity.identity_version == "mage-video-inference-identity-v2"
    assert (
        response.inference_identity.model_identity.identity_version
        == "mage-video-model-identity-v2"
    )
    assert response.inference_identity.model_identity.model_revision == "revision-1"
    assert (
        response.inference_identity.model_identity.runtime_identity.to_runtime_identity()
        == runtime.runtime_identity
    )
    assert response.inference_identity.codec_policy_identity.policy_sha256
    assert response.inference_identity.input_manifest_sha256
    assert response.inference_identity.decoder_identity_sha256
    assert response.camera_encoding_count == 1
    assert response.decoder_id == "shared-decoder-01"
    assert response.output_text == '{"scene_summary":"room"}'

    artifact_path = Path(response.result_artifact.durable_path)
    assert artifact_path.is_file()
    artifact = MageVideoResultArtifactDocument.model_validate_json(
        artifact_path.read_bytes(), strict=True
    )
    assert artifact.artifact_version == "mage-video-result-artifact-v2"
    assert artifact.artifact_identity == response.result_artifact.artifact_identity
    assert artifact.inference_identity == response.inference_identity
    assert artifact.created_at == fixed_created_at
    assert artifact.preprocess_device == "cpu"
    assert artifact.created_at not in artifact.output_text
    assert "created_at" not in MageVideoEndpointRequest.model_fields
    assert "hidden_state" not in MageVideoResultArtifactDocument.model_fields
    assert "past_key_values" not in MageVideoResultArtifactDocument.model_fields
    assert "recurrent_state" not in MageVideoResultArtifactDocument.model_fields


def _generation_telemetry() -> MageVideoGenerationTelemetry:
    return MageVideoGenerationTelemetry(
        request_started_monotonic_seconds=99.50,
        request_completed_monotonic_seconds=100.80,
        processor_started_monotonic_seconds=99.51,
        processor_completed_monotonic_seconds=99.71,
        input_materialization_started_monotonic_seconds=99.75,
        input_materialization_completed_monotonic_seconds=99.78,
        generation_started_monotonic_seconds=100.0,
        first_output_token_monotonic_seconds=100.25,
        generation_completed_monotonic_seconds=100.75,
        decode_started_monotonic_seconds=100.76,
        decode_completed_monotonic_seconds=100.80,
        processor_lock_wait_seconds=0.01,
        processor_seconds=0.20,
        generation_lock_wait_seconds=0.02,
        input_materialization_seconds=0.03,
        generate_seconds=0.75,
        decode_seconds=0.04,
        total_request_seconds=1.30,
        time_to_first_token_seconds=0.25,
        output_tokens_per_second=8.0,
    )


def test_jsonl_generation_telemetry_is_versioned_and_does_not_change_v2_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "robata.inference.mage_video_endpoint._server_authored_rfc3339_utc_timestamp",
        lambda: "2026-08-08T12:34:56.123456Z",
    )
    request, durable_root = _request(tmp_path)

    baseline = _service(_FakeRuntime(), tmp_path, durable_root)
    baseline.start()
    baseline_response = baseline.infer(request)
    baseline_response_bytes = canonical_json_bytes(baseline_response.model_dump(mode="json"))
    baseline_artifact_path = Path(baseline_response.result_artifact.durable_path)
    baseline_artifact_bytes = baseline_artifact_path.read_bytes()
    baseline.stop()
    baseline_artifact_path.unlink()

    telemetry_path = tmp_path / "telemetry" / "generation.jsonl"
    jsonl_sink = MageVideoGenerationJsonlSink(telemetry_path)

    def persisted_artifact_sink(event: Mapping[str, Any]) -> None:
        artifact_path = tmp_path / "results" / f"{event['result_artifact_identity']}.json"
        assert artifact_path.is_file()
        assert (
            exact_bytes_sha256(artifact_path.read_bytes())
            == (event["result_artifact_exact_sha256"])
        )
        jsonl_sink(event)

    instrumented = _service(
        _FakeRuntime(generation_telemetry=_generation_telemetry()),
        tmp_path,
        durable_root,
        generation_telemetry_sink=persisted_artifact_sink,
    )
    instrumented.start()
    instrumented_response = instrumented.infer(request)

    assert canonical_json_bytes(instrumented_response.model_dump(mode="json")) == (
        baseline_response_bytes
    )
    assert Path(instrumented_response.result_artifact.durable_path).read_bytes() == (
        baseline_artifact_bytes
    )
    assert "telemetry" not in MageVideoResultArtifactDocument.model_fields
    assert "telemetry" not in type(instrumented_response).model_fields

    lines = telemetry_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_version"] == "mage-video-generation-telemetry-event-v3"
    assert event["request_id"] == request.request_id
    assert event["context_id"] == request.context_manifest.context_id
    assert event["max_new_tokens"] == request.decoder.max_new_tokens
    assert event["output_tokens"] == 6
    assert event["output_budget_exhausted"] is False
    assert event["result_artifact_identity"] == (
        instrumented_response.result_artifact.artifact_identity
    )
    assert event["result_artifact_exact_sha256"] == (
        instrumented_response.result_artifact.content_sha256
    )
    assert event["result_artifact_created_at"] == "2026-08-08T12:34:56.123456Z"
    assert event["model_load_seconds"] == 1.25
    assert event["model_load_included_in_run_wall"] is False
    runtime_telemetry = event["runtime_telemetry"]
    assert set(runtime_telemetry) == {
        "telemetry_version",
        "request_started_monotonic_seconds",
        "request_completed_monotonic_seconds",
        "processor_started_monotonic_seconds",
        "processor_completed_monotonic_seconds",
        "input_materialization_started_monotonic_seconds",
        "input_materialization_completed_monotonic_seconds",
        "generation_started_monotonic_seconds",
        "first_output_token_monotonic_seconds",
        "generation_completed_monotonic_seconds",
        "decode_started_monotonic_seconds",
        "decode_completed_monotonic_seconds",
        "processor_lock_wait_seconds",
        "processor_seconds",
        "generation_lock_wait_seconds",
        "input_materialization_seconds",
        "generate_seconds",
        "decode_seconds",
        "total_request_seconds",
        "time_to_first_token_seconds",
        "output_tokens_per_second",
    }
    assert runtime_telemetry["telemetry_version"] == "mage-video-generation-telemetry-v3"
    assert runtime_telemetry["request_started_monotonic_seconds"] == 99.50
    assert runtime_telemetry["request_completed_monotonic_seconds"] == 100.80
    assert runtime_telemetry["time_to_first_token_seconds"] == 0.25
    assert runtime_telemetry["output_tokens_per_second"] == 8.0
    assert runtime_telemetry["generation_started_monotonic_seconds"] == 100.0
    assert runtime_telemetry["generation_completed_monotonic_seconds"] == 100.75


def test_generation_telemetry_sink_failure_does_not_fail_authoritative_result(
    tmp_path: Path,
) -> None:
    request, durable_root = _request(tmp_path)

    def failing_sink(_event: Mapping[str, Any]) -> None:
        raise OSError("telemetry disk unavailable")

    service = _service(
        _FakeRuntime(generation_telemetry=_generation_telemetry()),
        tmp_path,
        durable_root,
        generation_telemetry_sink=failing_sink,
    )
    service.start()

    response = service.infer(request)

    assert response.contract_version == "mage-video-codec-response-v2"
    assert Path(response.result_artifact.durable_path).is_file()


def test_neural_policy_translates_to_native_codec_parameters(tmp_path: Path) -> None:
    policy = MageVideoCodecPolicy(
        codec_mode="neural",
        preprocess_device="cuda",
        target_canvas=8,
        neural_parameters=MageVideoNeuralCodecParameters(
            quantization_parameter=21,
            reset_interval=12,
            max_side=720,
        ),
    )
    request, durable_root = _request(tmp_path, codec_policy=policy)
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path, durable_root)
    service.start()

    service.infer(request)

    assert runtime.last_codec_config is not None
    assert runtime.last_codec_config["engine"] == "dcvc-rt"
    assert runtime.last_codec_config["preprocess_device"] == "cuda"
    assert runtime.last_codec_config["dcvc"] == {
        "device": "cuda",
        "qp": 21,
        "reset_interval": 12,
        "intra_period": -1,
        "max_side": 720,
        "seq_len_frames": 0,
        "readiness_coverage_bins": 3,
        "readiness_delta_ratio": 0.05,
        "bitcost_pct": 99,
        "decode_backsearch_max": 16,
    }


def test_codec_policy_requires_explicit_device_and_changes_inference_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="preprocess_device"):
        MageVideoCodecPolicy()

    cpu_request, _ = _request(
        tmp_path,
        codec_policy=MageVideoCodecPolicy(preprocess_device="cpu"),
    )
    cuda_request, _ = _request(
        tmp_path,
        codec_policy=MageVideoCodecPolicy(preprocess_device="cuda"),
    )

    from robata.inference.mage_video_endpoint import build_mage_video_inference_identity

    cpu_identity = build_mage_video_inference_identity(cpu_request)
    cuda_identity = build_mage_video_inference_identity(cuda_request)
    assert cpu_identity.codec_policy_identity.policy_version == "mage-video-codec-policy-v2"
    assert cpu_identity.codec_policy_identity.preprocess_device == "cpu"
    assert cuda_identity.codec_policy_identity.preprocess_device == "cuda"
    assert cpu_identity.inference_identity != cuda_identity.inference_identity


def test_idempotency_replays_durable_artifact_after_restart_and_binds_exact_body(
    tmp_path: Path,
) -> None:
    request, durable_root = _request(tmp_path, request_id="durable-request")
    body = _request_body(request)
    first_runtime = _FakeRuntime()
    first_service = _service(first_runtime, tmp_path, durable_root)
    first_service.start()

    first_response = first_service.infer_idempotently(
        request=request,
        idempotency_key="durable-key",
        request_body=body,
    )
    first_service.stop()

    replay_runtime = _FakeRuntime()
    replay_service = _service(replay_runtime, tmp_path, durable_root)
    replay_service.start()
    replayed_response = replay_service.infer_idempotently(
        request=request,
        idempotency_key="durable-key",
        request_body=body,
    )

    assert first_runtime.generate_calls == 1
    assert replay_runtime.generate_calls == 0
    assert replayed_response == first_response
    assert Path(replayed_response.result_artifact.durable_path).is_file()
    assert MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_POLICY_VERSION == "mage-video-idempotency-policy-v2"

    with pytest.raises(MageVideoEndpointIdempotencyConflictError, match="different request bytes"):
        replay_service.infer_idempotently(
            request=request,
            idempotency_key="durable-key",
            request_body=body + b"\n",
        )
    assert replay_runtime.generate_calls == 0


def test_idempotent_replay_does_not_duplicate_generation_telemetry(
    tmp_path: Path,
) -> None:
    request, durable_root = _request(tmp_path, request_id="telemetry-replay")
    body = _request_body(request)
    telemetry_path = tmp_path / "telemetry" / "generation.jsonl"
    sink = MageVideoGenerationJsonlSink(telemetry_path)

    first_runtime = _FakeRuntime(generation_telemetry=_generation_telemetry())
    first_service = _service(
        first_runtime,
        tmp_path,
        durable_root,
        generation_telemetry_sink=sink,
    )
    first_service.start()
    first_service.infer_idempotently(
        request=request,
        idempotency_key="telemetry-replay-key",
        request_body=body,
    )
    first_service.stop()

    replay_runtime = _FakeRuntime(generation_telemetry=_generation_telemetry())
    replay_service = _service(
        replay_runtime,
        tmp_path,
        durable_root,
        generation_telemetry_sink=sink,
    )
    replay_service.start()
    replay_service.infer_idempotently(
        request=request,
        idempotency_key="telemetry-replay-key",
        request_body=body,
    )

    assert first_runtime.generate_calls == 1
    assert replay_runtime.generate_calls == 0
    assert len(telemetry_path.read_text(encoding="utf-8").splitlines()) == 1


def test_service_rejects_changed_durable_segment_before_runtime_call(tmp_path: Path) -> None:
    request, durable_root = _request(tmp_path)
    Path(request.camera_encodings[0].segment_manifest.durable_path).write_bytes(
        b"x" * len(b"durable-video-segment")
    )
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path, durable_root)
    service.start()

    with pytest.raises(MageVideoRuntimeError, match="content digest"):
        service.infer(request)
    assert runtime.generate_calls == 0


def test_contract_rejects_images_and_more_than_one_camera_encoding(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    payload = request.model_dump(mode="json")
    payload["images"] = []
    with pytest.raises(ValidationError):
        MageVideoEndpointRequest.model_validate(payload)

    multi_camera = request.model_copy(update={"camera_encodings": request.camera_encodings * 2})
    with pytest.raises(ValidationError):
        MageVideoEndpointRequest.model_validate(multi_camera.model_dump(mode="json"))


def test_same_idempotency_key_waits_without_a_second_model_generation(tmp_path: Path) -> None:
    request, durable_root = _request(tmp_path, request_id="same-key")
    runtime = _BlockingRuntime()
    service = _service(runtime, tmp_path, durable_root)
    service.start()
    request_body = _request_body(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.infer_idempotently,
            request=request,
            idempotency_key="same-key",
            request_body=request_body,
        )
        assert runtime.first_generation_entered.wait(timeout=1.0)
        second = executor.submit(
            service.infer_idempotently,
            request=request,
            idempotency_key="same-key",
            request_body=request_body,
        )
        assert not runtime.second_generation_entered.wait(timeout=0.25)
        runtime.release_generation.set()
        first_response = first.result(timeout=5.0)
        second_response = second.result(timeout=5.0)

    assert first_response == second_response
    assert runtime.generate_calls == 1


def test_unrelated_idempotency_keys_do_not_hold_sqlite_or_service_lock_during_generation(
    tmp_path: Path,
) -> None:
    first_request, durable_root = _request(tmp_path, request_id="parallel-first")
    second_request, _ = _request(tmp_path, request_id="parallel-second")
    runtime = _BlockingRuntime()
    service = _service(runtime, tmp_path, durable_root)
    service.start()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.infer_idempotently,
            request=first_request,
            idempotency_key="parallel-first",
            request_body=_request_body(first_request),
        )
        assert runtime.first_generation_entered.wait(timeout=1.0)
        second = executor.submit(
            service.infer_idempotently,
            request=second_request,
            idempotency_key="parallel-second",
            request_body=_request_body(second_request),
        )
        assert runtime.second_generation_entered.wait(timeout=1.0)
        runtime.release_generation.set()
        first_response = first.result(timeout=5.0)
        second_response = second.result(timeout=5.0)

    assert first_response.request_id == "parallel-first"
    assert second_response.request_id == "parallel-second"
    assert runtime.generate_calls == 2


def test_service_fails_closed_when_resident_runtime_profile_changes(tmp_path: Path) -> None:
    configured_runtime_identity = MageVideoRuntimeIdentity()
    request, durable_root = _request(
        tmp_path,
        model_identity=_model_identity(runtime_identity=configured_runtime_identity),
    )
    runtime = _FakeRuntime(runtime_identity=configured_runtime_identity)
    service = _service(
        runtime,
        tmp_path,
        durable_root,
        model_identity=_model_identity(runtime_identity=configured_runtime_identity),
    )
    service.start()

    runtime.runtime_identity = MageVideoRuntimeIdentity(
        load_profile=MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4
    )

    with pytest.raises(MageVideoRuntimeError, match="identity/load profile"):
        service.health()
    with pytest.raises(MageVideoRuntimeError, match="identity/load profile"):
        service.infer(request)
    assert runtime.generate_calls == 0


def test_health_reports_explicit_preprocessing_policy_requirement(tmp_path: Path) -> None:
    request, durable_root = _request(tmp_path)
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path, durable_root, model_identity=request.model_identity)
    service.start()

    health = service.health()

    assert health.codec_policy_version == "mage-video-codec-policy-v2"
    assert health.preprocess_device_requirement == "EXPLICIT_CPU_OR_CUDA"


def test_service_fails_closed_at_start_when_configured_profile_differs(tmp_path: Path) -> None:
    configured_runtime_identity = MageVideoRuntimeIdentity(
        load_profile=MageVideoLoadProfile.BITSANDBYTES_4BIT_NF4
    )
    request, durable_root = _request(
        tmp_path,
        model_identity=_model_identity(runtime_identity=configured_runtime_identity),
    )
    runtime = _FakeRuntime()
    service = _service(
        runtime,
        tmp_path,
        durable_root,
        model_identity=request.model_identity,
    )

    with pytest.raises(MageVideoRuntimeError, match="identity/load profile"):
        service.start()
    assert runtime.generate_calls == 0


def test_fastapi_v2_route_replays_native_video_request(tmp_path: Path) -> None:
    request, durable_root = _request(tmp_path, request_id="http-request")
    runtime = _FakeRuntime()
    service = _service(runtime, tmp_path, durable_root)
    application = create_mage_video_endpoint_app(service)
    body = _request_body(request)
    headers = {
        "Content-Type": "application/json",
        MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER: "http-idempotency-key",
    }

    with TestClient(application) as client:
        legacy = client.post("/v1/mage-video/infer", content=body, headers=headers)
        first = client.post("/v2/mage-video/infer", content=body, headers=headers)
        second = client.post("/v2/mage-video/infer", content=body, headers=headers)

    assert legacy.status_code == 404
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["camera_encoding_count"] == 1
    assert runtime.generate_calls == 1
