from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.inference.mage_video_adapter import MageVideoObservationTransportRequest
from robata.inference.mage_video_endpoint import (
    MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER,
    MageVideoCameraEncoding,
    MageVideoCodecPolicy,
    MageVideoDecoderRequest,
    MageVideoEndpointRequest,
    MageVideoEndpointResponse,
    MageVideoHealthResponse,
    MageVideoModelIdentity,
    MageVideoResultArtifactReference,
    build_mage_video_context_manifest,
    build_mage_video_inference_identity,
    build_mage_video_segment_manifest,
)
from robata.inference.mage_video_http_transport import (
    MageVideoHttpTransport,
    fetch_mage_video_endpoint_health,
)
from robata.inference.mage_video_runtime import MageVideoRuntimeIdentity


@dataclass
class _Response:
    payload: bytes
    closed: bool = False

    def read(self, _amount: int = -1) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True


@dataclass
class _Opener:
    response: _Response
    last_request: object | None = None
    last_timeout: float | None = None

    def __call__(self, request, *, timeout: float):  # type: ignore[no-untyped-def]
        self.last_request = request
        self.last_timeout = timeout
        return self.response


def _request(tmp_path: Path) -> MageVideoEndpointRequest:
    segment_path = tmp_path / "segment.mp4"
    segment_payload = b"native-video-segment"
    segment_path.write_bytes(segment_payload)
    segment = build_mage_video_segment_manifest(
        segment_id="segment-1",
        camera_id="cam_01",
        durable_path=str(segment_path),
        media_type="video/mp4",
        content_sha256=exact_bytes_sha256(segment_payload),
        byte_count=len(segment_payload),
    )
    context = build_mage_video_context_manifest(
        context_id="context-1",
        context_payload_sha256="a" * 64,
        segment_manifest_identities=[segment.manifest_identity],
    )
    return MageVideoEndpointRequest(
        request_id="request-1",
        model_identity=MageVideoModelIdentity(
            model_identifier="Mage-VL",
            model_revision="local",
            checkpoint_manifest_sha256="b" * 64,
            runtime_identity=MageVideoRuntimeIdentity(),
        ),
        codec_policy=MageVideoCodecPolicy(preprocess_device="cpu"),
        context_manifest=context,
        camera_encodings=[
            MageVideoCameraEncoding(
                encoder_id="camera-encoder-cam_01",
                segment_manifest=segment,
            )
        ],
        decoder=MageVideoDecoderRequest(
            decoder_id="mage-observation-decoder-v2",
            prompt="{}",
            max_new_tokens=32,
        ),
    )


def _response(request: MageVideoEndpointRequest) -> MageVideoEndpointResponse:
    return MageVideoEndpointResponse(
        request_id=request.request_id,
        inference_identity=build_mage_video_inference_identity(request),
        camera_encoding_count=1,
        decoder_id=request.decoder.decoder_id,
        prompt_tokens=1,
        output_tokens=1,
        load_seconds=0.0,
        generation_seconds=0.1,
        execution_device="cuda:0",
        output_text="{}",
        preprocess_device=request.codec_policy.preprocess_device,
        result_artifact=MageVideoResultArtifactReference(
            artifact_identity="c" * 64,
            content_sha256="d" * 64,
            durable_path="D:/results/result.json",
        ),
    )


def test_http_transport_sends_exact_adapter_bytes_and_idempotency_header(tmp_path: Path) -> None:
    request = _request(tmp_path)
    response = _response(request)
    opener = _Opener(_Response(canonical_json_bytes(response.model_dump(mode="json"))))
    transport = MageVideoHttpTransport(
        endpoint_url="http://127.0.0.1:8102/",
        timeout_seconds=12.5,
        opener=opener,
    )
    body = canonical_json_bytes(request.model_dump(mode="json"))

    received = transport.infer(
        MageVideoObservationTransportRequest(
            endpoint_path="/v2/mage-video/infer",
            request=request,
            request_body=body,
            idempotency_key="test-idempotency-key",
        )
    )

    assert received == response
    assert opener.last_timeout == 12.5
    sent = opener.last_request
    assert sent is not None
    assert sent.full_url == "http://127.0.0.1:8102/v2/mage-video/infer"
    assert sent.data == body
    assert (
        sent.get_header(MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER.capitalize())
        == "test-idempotency-key"
    )
    assert opener.response.closed is True


def test_health_fetch_exposes_endpoint_model_identity(tmp_path: Path) -> None:
    request = _request(tmp_path)
    health = MageVideoHealthResponse(model_identity=request.model_identity)
    opener = _Opener(_Response(canonical_json_bytes(health.model_dump(mode="json"))))

    received = fetch_mage_video_endpoint_health(
        endpoint_url="http://127.0.0.1:8102",
        timeout_seconds=3.0,
        opener=opener,
    )

    assert received.model_identity == request.model_identity
    sent = opener.last_request
    assert sent is not None
    assert sent.full_url == "http://127.0.0.1:8102/healthz"
    assert sent.get_method() == "GET"
