from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256, semantic_sha256
from robata.contracts.perception_stream import MageObservation, PerceptionContextManifest
from robata.inference.mage_video_adapter import (
    MAGE_VIDEO_COMPACT_DECODER_ID,
    MAGE_VIDEO_COMPACT_DEFAULT_MAX_NEW_TOKENS,
    MAGE_VIDEO_COMPACT_MAX_OBSERVATIONS,
    MAGE_VIDEO_COMPACT_OBSERVATION_PROMPT_VERSION,
    MAGE_VIDEO_COMPACT_OUTPUT_POLICY_VERSION,
    MAGE_VIDEO_INFER_PATH,
    MAGE_VIDEO_OBSERVATION_IDEMPOTENCY_NAMESPACE,
    MAGE_VIDEO_OBSERVATION_REQUEST_IDENTITY_VERSION,
    MAGE_VIDEO_UNIFIED_OBSERVATION_PROMPT_CONTRACT_VERSION,
    MageVideoAcceptedObservationBinding,
    MageVideoDurableCameraSegment,
    MageVideoObservationAdapter,
    MageVideoObservationAdapterConfig,
    MageVideoObservationAdapterError,
    MageVideoObservationDiagnostic,
    MageVideoObservationDiagnosticSink,
    MageVideoObservationTransportRequest,
    MageVideoPreparedObservationRequest,
    build_mage_video_unified_observation_prompt,
)
from robata.inference.mage_video_endpoint import (
    MAGE_VIDEO_ENDPOINT_RESPONSE_VERSION,
    MAGE_VIDEO_RESULT_ARTIFACT_VERSION,
    MageVideoCodecPolicy,
    MageVideoEndpointRequest,
    MageVideoEndpointResponse,
    MageVideoModelIdentity,
    MageVideoResultArtifactDocument,
    MageVideoResultArtifactReference,
    build_mage_video_inference_identity,
)
from robata.perception.pipeline import LocalPerceptionArtifactStore, PerceptionArtifactSink
from tests.support.perception_stream import make_context


@dataclass
class _Resolver:
    segments: tuple[MageVideoDurableCameraSegment, ...]
    calls: list[tuple[PerceptionContextManifest, CameraId]] = field(default_factory=list)

    def resolve(
        self,
        *,
        context: PerceptionContextManifest,
        camera_id: CameraId,
    ) -> tuple[MageVideoDurableCameraSegment, ...]:
        self.calls.append((context, camera_id))
        return self.segments


class _FailingResolver:
    def resolve(
        self,
        *,
        context: PerceptionContextManifest,
        camera_id: CameraId,
    ) -> tuple[MageVideoDurableCameraSegment, ...]:
        raise AssertionError("artifact replay must not resolve source media")


@dataclass
class _Transport:
    response: MageVideoEndpointResponse | None = None
    calls: list[MageVideoObservationTransportRequest] = field(default_factory=list)

    def infer(self, invocation: MageVideoObservationTransportRequest) -> MageVideoEndpointResponse:
        self.calls.append(invocation)
        assert self.response is not None
        return self.response


@dataclass
class _ArtifactReader:
    payload: bytes = b"unused"
    calls: list[MageVideoResultArtifactReference] = field(default_factory=list)

    def read(self, reference: MageVideoResultArtifactReference) -> bytes:
        self.calls.append(reference)
        return self.payload


@dataclass
class _DiagnosticSink(MageVideoObservationDiagnosticSink):
    diagnostics: list[MageVideoObservationDiagnostic] = field(default_factory=list)

    def record(self, diagnostic: MageVideoObservationDiagnostic) -> None:
        self.diagnostics.append(diagnostic)


def _model_identity(*, revision: str = "revision-1") -> MageVideoModelIdentity:
    return MageVideoModelIdentity(
        model_identifier="mage-video-base",
        model_revision=revision,
        checkpoint_manifest_sha256="a" * 64,
    )


def _single_camera_context() -> PerceptionContextManifest:
    return make_context(selected_cameras=(CameraId.CAM_01,))


def _durable_segment(
    context: PerceptionContextManifest,
    *,
    camera_id: CameraId = CameraId.CAM_01,
) -> MageVideoDurableCameraSegment:
    binding = context.cameras[camera_id]
    assert binding.codec_stream_exact_sha256 is not None
    return MageVideoDurableCameraSegment(
        camera_id=camera_id,
        segment_semantic_sha256_values=binding.segment_semantic_sha256_values,
        codec_stream_exact_sha256=binding.codec_stream_exact_sha256,
        durable_path=f"/durable/{camera_id.value}.mp4",
        content_sha256=binding.codec_stream_exact_sha256,
        byte_count=123,
    )


def _adapter(
    *,
    resolver: _Resolver | _FailingResolver,
    transport: _Transport,
    artifact_reader: _ArtifactReader,
    config: MageVideoObservationAdapterConfig | None = None,
    diagnostic_sink: MageVideoObservationDiagnosticSink | None = None,
    accepted_binding_sink: PerceptionArtifactSink | None = None,
) -> MageVideoObservationAdapter:
    return MageVideoObservationAdapter(
        model_identity=_model_identity(),
        codec_policy=MageVideoCodecPolicy(preprocess_device="cpu"),
        segment_resolver=resolver,
        transport=transport,
        artifact_reader=artifact_reader,
        config=config,
        diagnostic_sink=diagnostic_sink,
        accepted_binding_sink=accepted_binding_sink,
    )


def _valid_payload(context: PerceptionContextManifest) -> dict[str, object]:
    return {
        "observation_schema_version": "mage-observation-v1",
        "selected_camera_qa": {
            "disposition": "USABLE",
            "issues": [],
            "confidence": "0.98",
        },
        "observations": [
            {
                "action": "pick_up_cup",
                "interval": {
                    "start_offset_seconds": "1.0",
                    "end_offset_seconds": "2.0",
                },
                "confidence": "0.91",
                "actor": {"hand": "right", "actor_type": "robot_hand"},
                "object": {"object_type": "cup"},
                "visibility": "0.88",
                "boundary": {
                    "start_confidence": "0.81",
                    "end_confidence": "0.83",
                    "started_before_context": False,
                    "continues_after_context": False,
                },
            }
        ],
    }


def _compact_payload_bytes(context: PerceptionContextManifest) -> str:
    return canonical_json_bytes(_valid_payload(context)).decode("utf-8")


def _response_with_artifact(
    prepared: MageVideoPreparedObservationRequest,
    output_text: str,
    *,
    created_at: str = "2026-08-07T01:02:03Z",
) -> tuple[MageVideoEndpointResponse, bytes]:
    projection: dict[str, object] = {
        "artifact_version": MAGE_VIDEO_RESULT_ARTIFACT_VERSION,
        "request_id": prepared.endpoint_request.request_id,
        "inference_identity": prepared.inference_identity.model_dump(mode="json"),
        "camera_encoding_count": 1,
        "decoder_id": prepared.endpoint_request.decoder.decoder_id,
        "prompt_tokens": 17,
        "output_tokens": 31,
        "load_seconds": 0.2,
        "generation_seconds": 0.4,
        "execution_device": "loopback",
        "preprocess_device": "cpu",
        "output_text": output_text,
        "created_at": created_at,
    }
    document = MageVideoResultArtifactDocument(
        artifact_identity=semantic_sha256(projection),
        request_id=prepared.endpoint_request.request_id,
        inference_identity=prepared.inference_identity,
        camera_encoding_count=1,
        decoder_id=prepared.endpoint_request.decoder.decoder_id,
        prompt_tokens=17,
        output_tokens=31,
        load_seconds=0.2,
        generation_seconds=0.4,
        execution_device="loopback",
        preprocess_device="cpu",
        output_text=output_text,
        created_at=created_at,
    )
    artifact_bytes = canonical_json_bytes(document.model_dump(mode="json"))
    reference = MageVideoResultArtifactReference(
        artifact_identity=document.artifact_identity,
        content_sha256=exact_bytes_sha256(artifact_bytes),
        durable_path="/result-artifacts/loopback.json",
    )
    return (
        MageVideoEndpointResponse(
            contract_version=MAGE_VIDEO_ENDPOINT_RESPONSE_VERSION,
            request_id=prepared.endpoint_request.request_id,
            inference_identity=prepared.inference_identity,
            camera_encoding_count=1,
            decoder_id=prepared.endpoint_request.decoder.decoder_id,
            prompt_tokens=17,
            output_tokens=31,
            load_seconds=0.2,
            generation_seconds=0.4,
            execution_device="loopback",
            preprocess_device="cpu",
            output_text=output_text,
            result_artifact=reference,
        ),
        artifact_bytes,
    )


def test_observe_uses_one_compact_v2_request_and_deterministically_expands_observation() -> None:
    context = _single_camera_context()
    resolver = _Resolver((_durable_segment(context),))
    transport = _Transport()
    artifact_reader = _ArtifactReader()
    adapter = _adapter(
        resolver=resolver,
        transport=transport,
        artifact_reader=artifact_reader,
    )
    prepared = adapter.prepare_request(context)
    response, artifact_bytes = _response_with_artifact(prepared, _compact_payload_bytes(context))
    transport.response = response
    artifact_reader.payload = artifact_bytes

    observation = adapter.observe(context)

    assert len(transport.calls) == 1
    invocation = transport.calls[0]
    assert invocation.endpoint_path == "/v2/mage-video/infer"
    assert invocation.endpoint_path == MAGE_VIDEO_INFER_PATH
    assert invocation.request == prepared.endpoint_request
    assert invocation.request_body == prepared.request_body
    assert invocation.idempotency_key.startswith(f"{MAGE_VIDEO_OBSERVATION_IDEMPOTENCY_NAMESPACE}:")
    assert invocation.request.request_id.startswith(
        f"{MAGE_VIDEO_OBSERVATION_REQUEST_IDENTITY_VERSION}:"
    )
    assert len(invocation.request.camera_encodings) == 1
    prompt = invocation.request.decoder.prompt
    assert MAGE_VIDEO_UNIFIED_OBSERVATION_PROMPT_CONTRACT_VERSION in prompt
    assert "video_duration_seconds" in prompt
    assert "start_offset_seconds" in prompt
    assert '"example"' not in prompt
    assert "pick_up_cup" not in prompt
    assert "all_camera_ids" not in prompt
    assert "camera_context" not in prompt
    assert "segment_semantic_sha256_values" not in prompt
    assert '"camera_id"' not in prompt
    assert '"adapter_supplied_fields"' not in prompt
    assert invocation.request.decoder.max_new_tokens == 512

    assert observation.context == context
    assert observation.inference_artifact_exact_sha256 == exact_bytes_sha256(artifact_bytes)
    assert observation.model_revision == "revision-1"
    assert observation.created_at == "2026-08-07T01:02:03Z"
    assert observation.cognition_gate.mode == "SHADOW_ONLY"
    assert observation.cognition_gate.score is None
    assert observation.cognition_gate.would_admit is None
    assert observation.semantic_qa[CameraId.CAM_01].disposition.value == "USABLE"
    assert observation.semantic_qa[CameraId.CAM_01].confidence == 0.98
    assert observation.semantic_qa[CameraId.CAM_06].disposition.value == "UNKNOWN"
    assert observation.semantic_qa[CameraId.CAM_06].confidence is None
    assert len(observation.observations) == 1
    action = observation.observations[0]
    assert action.local_ref == "observation_1"
    assert action.interval.start_ns == 1_000_000_000
    assert action.interval.end_ns == 2_000_000_000
    selected_evidence = action.camera_evidence[CameraId.CAM_01]
    assert selected_evidence.relation.value == "SUPPORTS"
    assert selected_evidence.visibility == 0.88
    assert selected_evidence.evidence_semantic_sha256_values == (
        context.ordered_segments[-1].segment_semantic_sha256,
    )
    for camera_id in CAMERA_IDS[1:]:
        assert action.camera_evidence[camera_id].relation.value == "NOT_OBSERVABLE"


def test_realistic_free_form_mage_output_is_reduced_and_context_local_ns_are_rebased() -> None:
    context = make_context(
        start_ns=8_000_000_000,
        end_ns=16_000_000_000,
        segment_ordinal=1,
        selected_cameras=(CameraId.CAM_01,),
    )
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    prepared = adapter.prepare_request(context)
    realistic_payload = {
        "observation_schema_version": "v1",
        "camera_id": None,
        "camera_evidence": None,
        "segment_hashes": None,
        "created_at": None,
        "cognition_gate": None,
        "interval": {"start_offset_seconds": 0, "end_offset_seconds": 8},
        "observations": [
            {
                "action": "The person in a white shirt picks up the light green shirt",
                "actor": {"name": "person", "confidence": 1.0},
                "interval": {"start_ns": 1_000_000_000, "end_ns": 2_000_000_000},
                "object": {
                    "name": "light green shirt",
                    "local_ref": "light green shirt",
                    "confidence": 1.0,
                },
                "boundary": {
                    "name": "table",
                    "local_ref": "table",
                    "confidence": 1.0,
                },
                "visibility": 1.0,
            }
        ],
    }
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(realistic_payload).decode("utf-8"),
    )

    observation = adapter.replay_prepared_artifact(
        prepared=prepared,
        response=response,
        artifact_bytes=artifact_bytes,
    )

    assert observation.observation_schema_version == "mage-observation-v1"
    assert observation.semantic_qa[CameraId.CAM_01].disposition.value == "UNKNOWN"
    assert observation.semantic_qa[CameraId.CAM_01].confidence is None
    assert len(observation.observations) == 1
    action = observation.observations[0]
    assert action.action == "the_person_in_a_white_shirt_picks_up_the_light_green_shirt"
    assert action.interval.start_ns == 9_000_000_000
    assert action.interval.end_ns == 10_000_000_000
    assert action.actor is not None
    assert action.actor.actor_type == "person"
    assert action.object is not None
    assert action.object.object_type == "light_green_shirt"
    assert action.object.identity_hint == "light green shirt"
    assert action.boundary.start_confidence == 0.0
    assert action.boundary.end_confidence == 0.0
    assert action.camera_evidence[CameraId.CAM_01].visibility == 1.0


def test_compact_payload_rejects_model_authored_provenance_and_noncompact_fields() -> None:
    context = _single_camera_context()
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    prepared = adapter.prepare_request(context)

    for forbidden_key, forbidden_value in (
        ("created_at", "2026-08-07T00:00:00Z"),
        ("cognition_gate", {"score": 0.25}),
        ("camera_id", "cam_01"),
        ("segment_semantic_sha256_values", ["b" * 64]),
    ):
        payload = _valid_payload(context)
        payload[forbidden_key] = forbidden_value
        response, artifact_bytes = _response_with_artifact(
            prepared,
            canonical_json_bytes(payload).decode("utf-8"),
        )
        with pytest.raises(MageVideoObservationAdapterError):
            adapter.replay_prepared_artifact(
                prepared=prepared,
                response=response,
                artifact_bytes=artifact_bytes,
            )

    cognition_gate_precedence = _valid_payload(context)
    cognition_gate_precedence["cognition_gate"] = {"score": 0.25}
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(cognition_gate_precedence).decode("utf-8"),
    )
    with pytest.raises(MageVideoObservationAdapterError, match="forbidden field cognition_gate"):
        adapter.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )


def test_compact_payload_rejects_duplicate_keys_bad_numeric_bounds_and_duplicate_local_refs() -> (
    None
):
    context = _single_camera_context()
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    prepared = adapter.prepare_request(context)

    duplicate_key_output = (
        '{"observation_schema_version":"mage-observation-v1",'
        '"selected_camera_qa":{"disposition":"USABLE","confidence":0.9,"confidence":0.8},'
        '"observations":[]}'
    )
    response, artifact_bytes = _response_with_artifact(prepared, duplicate_key_output)
    with pytest.raises(MageVideoObservationAdapterError, match="strict JSON"):
        adapter.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )

    noncanonical_numeric = _valid_payload(context)
    selected_qa = noncanonical_numeric["selected_camera_qa"]
    assert isinstance(selected_qa, dict)
    selected_qa["confidence"] = "1e-1"
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(noncanonical_numeric).decode("utf-8"),
    )
    with pytest.raises(MageVideoObservationAdapterError, match="canonical unsigned decimal"):
        adapter.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )

    nonintegral_bound = _valid_payload(context)
    observations = nonintegral_bound["observations"]
    assert isinstance(observations, list)
    first = observations[0]
    assert isinstance(first, dict)
    interval = first["interval"]
    assert isinstance(interval, dict)
    interval["start_offset_seconds"] = "0.0000000001"
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(nonintegral_bound).decode("utf-8"),
    )
    with pytest.raises(MageVideoObservationAdapterError, match="integral nanosecond"):
        adapter.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )

    duplicate_local_refs = _valid_payload(context)
    duplicate_observations = duplicate_local_refs["observations"]
    assert isinstance(duplicate_observations, list)
    duplicate = dict(duplicate_observations[0])
    duplicate["local_ref"] = "observation_1"
    duplicate_observations.append(duplicate)
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(duplicate_local_refs).decode("utf-8"),
    )
    with pytest.raises(
        MageVideoObservationAdapterError, match="duplicate after deterministic assignment"
    ):
        adapter.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )

    duplicate_semantic_actions = _valid_payload(context)
    duplicate_observations = duplicate_semantic_actions["observations"]
    assert isinstance(duplicate_observations, list)
    first = duplicate_observations[0]
    assert isinstance(first, dict)
    duplicate = dict(first)
    duplicate["local_ref"] = "observation_2"
    duplicate["confidence"] = "0.90"
    duplicate["visibility"] = "0.70"
    duplicate_observations.append(duplicate)
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(duplicate_semantic_actions).decode("utf-8"),
    )
    with pytest.raises(MageVideoObservationAdapterError, match="duplicate semantic actions"):
        adapter.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes,
        )


def test_short_tail_rejects_outside_action_with_exact_artifact_bound_diagnostic() -> None:
    context = make_context(
        start_ns=40_000_000_000,
        end_ns=40_833_500_000,
        segment_ordinal=5,
        selected_cameras=(CameraId.CAM_01,),
    )
    sink = _DiagnosticSink()
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
        diagnostic_sink=sink,
    )
    prepared = adapter.prepare_request(context)
    payload = _valid_payload(context)
    observations = payload["observations"]
    assert isinstance(observations, list)
    first = observations[0]
    assert isinstance(first, dict)
    interval = first["interval"]
    assert isinstance(interval, dict)
    interval["start_offset_seconds"] = "1.0"
    interval["end_offset_seconds"] = "1.8"
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(payload).decode("utf-8"),
    )

    observation = adapter.replay_prepared_artifact(
        prepared=prepared,
        response=response,
        artifact_bytes=artifact_bytes,
    )

    assert observation.observations == ()
    assert len(adapter.last_diagnostics) == 1
    diagnostic = adapter.last_diagnostics[0]
    assert diagnostic.code == "ACTION_INTERVAL_REJECTED_OUTSIDE_CONTEXT"
    assert diagnostic.context_manifest_semantic_sha256 == context.context_manifest_semantic_sha256
    assert diagnostic.inference_artifact_exact_sha256 == exact_bytes_sha256(artifact_bytes)
    assert diagnostic.action_ordinal == 1
    assert diagnostic.local_ref == "observation_1"
    assert diagnostic.reported_interval.start_ns == 41_000_000_000
    assert diagnostic.reported_interval.end_ns == 41_800_000_000
    assert diagnostic.retained_interval is None
    assert sink.diagnostics == [diagnostic]


def test_clip_policy_is_identity_bound_and_clips_only_a_nonempty_intersection() -> None:
    context = make_context(
        start_ns=40_000_000_000,
        end_ns=40_833_500_000,
        segment_ordinal=5,
        selected_cameras=(CameraId.CAM_01,),
    )
    config = MageVideoObservationAdapterConfig(out_of_context_action_policy="CLIP_INTERSECTION_V1")
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
        config=config,
    )
    default_adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    prepared = adapter.prepare_request(context)
    default_prepared = default_adapter.prepare_request(context)
    assert prepared.request_identity_sha256 != default_prepared.request_identity_sha256
    # Interval handling is a deterministic adapter projection. It changes the
    # adapter request identity without forcing another expensive Mage generation.
    assert (
        prepared.endpoint_request.decoder.prompt == default_prepared.endpoint_request.decoder.prompt
    )
    assert prepared.inference_identity == default_prepared.inference_identity

    payload = _valid_payload(context)
    observations = payload["observations"]
    assert isinstance(observations, list)
    first = observations[0]
    assert isinstance(first, dict)
    interval = first["interval"]
    assert isinstance(interval, dict)
    interval["start_offset_seconds"] = "0.5"
    interval["end_offset_seconds"] = "1.8"
    response, artifact_bytes = _response_with_artifact(
        prepared,
        canonical_json_bytes(payload).decode("utf-8"),
    )

    observation = adapter.replay_prepared_artifact(
        prepared=prepared,
        response=response,
        artifact_bytes=artifact_bytes,
    )

    assert observation.prompt_version.endswith("+clip_intersection_v1")
    assert len(observation.observations) == 1
    action = observation.observations[0]
    assert action.interval.start_ns == 40_500_000_000
    assert action.interval.end_ns == 40_833_500_000
    assert action.boundary.continues_after_context is True
    assert action.boundary.end_confidence == 0.0
    assert len(adapter.last_diagnostics) == 1
    diagnostic = adapter.last_diagnostics[0]
    assert diagnostic.code == "ACTION_INTERVAL_CLIPPED_TO_CONTEXT"
    assert diagnostic.retained_interval == action.interval


def test_prepared_artifact_replay_is_media_free_exact_and_identity_bound() -> None:
    context = _single_camera_context()
    resolver = _Resolver((_durable_segment(context),))
    transport = _Transport()
    artifact_reader = _ArtifactReader()
    adapter = _adapter(
        resolver=resolver,
        transport=transport,
        artifact_reader=artifact_reader,
    )
    prepared = adapter.prepare_request(context)
    response, artifact_bytes = _response_with_artifact(prepared, _compact_payload_bytes(context))

    replay_adapter = _adapter(
        resolver=_FailingResolver(),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    replayed = replay_adapter.replay_prepared_artifact(
        prepared=prepared,
        response=response,
        artifact_bytes=artifact_bytes,
    )
    assert isinstance(replayed, MageObservation)
    assert replayed.inference_artifact_exact_sha256 == exact_bytes_sha256(artifact_bytes)

    with pytest.raises(MageVideoObservationAdapterError, match="exact-byte digest"):
        replay_adapter.replay_prepared_artifact(
            prepared=prepared,
            response=response,
            artifact_bytes=artifact_bytes + b"\n",
        )

    changed_model_request: MageVideoEndpointRequest = prepared.endpoint_request.model_copy(
        update={"model_identity": _model_identity(revision="revision-2")}
    )
    changed_prepared = replace(
        prepared,
        endpoint_request=changed_model_request,
        inference_identity=build_mage_video_inference_identity(changed_model_request),
    )
    changed_response, changed_artifact_bytes = _response_with_artifact(
        changed_prepared,
        _compact_payload_bytes(context),
    )
    with pytest.raises(
        MageVideoObservationAdapterError, match="model identifier, revision, checkpoint"
    ):
        replay_adapter.replay_prepared_artifact(
            prepared=prepared,
            response=changed_response,
            artifact_bytes=changed_artifact_bytes,
        )


def test_observe_persists_accepted_binding_for_media_free_restart(
    tmp_path: Path,
) -> None:
    context = _single_camera_context()
    resolver = _Resolver((_durable_segment(context),))
    bootstrap = _adapter(
        resolver=resolver,
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    prepared = bootstrap.prepare_request(context)
    response, artifact_bytes = _response_with_artifact(prepared, _compact_payload_bytes(context))
    store = LocalPerceptionArtifactStore(tmp_path / "perception-cas")
    transport = _Transport(response=response)
    adapter = _adapter(
        resolver=resolver,
        transport=transport,
        artifact_reader=_ArtifactReader(payload=artifact_bytes),
        accepted_binding_sink=store,
    )

    observed = adapter.observe(context)
    binding_key = f"mage-video-accepted-binding-v1:{prepared.request_identity_sha256}"
    binding_bytes = store.read(kind="accepted-inference-binding", logical_key=binding_key)
    binding = MageVideoAcceptedObservationBinding.model_validate_json(binding_bytes, strict=True)
    replay_adapter = _adapter(
        resolver=_FailingResolver(),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    replayed = replay_adapter.replay_accepted_binding(
        binding=binding,
        artifact_bytes=artifact_bytes,
    )

    assert replayed == observed
    assert transport.calls and len(transport.calls) == 1
    assert binding.request_identity_sha256 == prepared.request_identity_sha256
    assert binding.result_artifact_exact_sha256 == exact_bytes_sha256(artifact_bytes)
    with pytest.raises(MageVideoObservationAdapterError, match="exact-byte"):
        replay_adapter.replay_accepted_binding(
            binding=binding,
            artifact_bytes=artifact_bytes + b"\n",
        )


def test_v2_fails_closed_when_selected_or_executable_cardinality_is_not_one() -> None:
    multi_selected_context = make_context(
        selected_cameras=(CameraId.CAM_01, CameraId.CAM_02),
    )
    transport = _Transport()
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(multi_selected_context),)),
        transport=transport,
        artifact_reader=_ArtifactReader(),
    )

    with pytest.raises(MageVideoObservationAdapterError, match="exactly one selected"):
        adapter.observe(multi_selected_context)
    assert transport.calls == []

    single_context = _single_camera_context()
    segment = _durable_segment(single_context)
    one_camera_transport = _Transport()
    adapter_with_two_segments = _adapter(
        resolver=_Resolver((segment, segment)),
        transport=one_camera_transport,
        artifact_reader=_ArtifactReader(),
    )

    with pytest.raises(MageVideoObservationAdapterError, match="exactly one executable"):
        adapter_with_two_segments.observe(single_context)
    assert one_camera_transport.calls == []


def test_v6_defaults_and_sanitizes_native_mage_metadata() -> None:
    context = _single_camera_context()
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    prepared = adapter.prepare_request(context)
    native_output = (
        '{"observations":[{'
        '"action":"A person reaches for a light-green shirt!",'
        '"actor":{"name":"Person","confidence":1.0,"appearance":"white shirt"},'
        '"interval":{"start_ns":1000000000,"end_ns":2000000000},'
        '"object":{"name":"Light green shirt","local_ref":"shirt-1","confidence":1.0},'
        '"boundary":{'
        '"start_confidence":0.9,"end_confidence":0.8,'
        '"started_before_context":true,"continues_after_context":true'
        "},"
        '"visibility":1.0'
        "}]} "
    )
    response, artifact_bytes = _response_with_artifact(prepared, native_output)

    observation = adapter.replay_prepared_artifact(
        prepared=prepared,
        response=response,
        artifact_bytes=artifact_bytes,
    )

    # Missing model-authored schema/QA use adapter-owned defaults, not guesses.
    assert observation.observation_schema_version == "mage-observation-v1"
    assert observation.semantic_qa[CameraId.CAM_01].disposition.value == "UNKNOWN"
    assert observation.semantic_qa[CameraId.CAM_01].issues == ()
    assert observation.semantic_qa[CameraId.CAM_01].confidence is None

    assert len(observation.observations) == 1
    action = observation.observations[0]
    assert action.action == "a_person_reaches_for_a_light_green_shirt"
    assert action.actor is not None
    assert action.actor.hand is None
    assert action.actor.actor_type == "person"
    assert action.object is not None
    assert action.object.object_type == "light_green_shirt"
    assert action.object.identity_hint == "shirt-1"
    # The continuation claims do not agree with a mid-context interval, so optional
    # boundary metadata is dropped without discarding the otherwise valid action.
    assert action.boundary.start_confidence == 0.0
    assert action.boundary.end_confidence == 0.0
    assert action.boundary.started_before_context is False
    assert action.boundary.continues_after_context is False


def test_v6_projects_unambiguous_context_local_ns_to_a_later_absolute_clock() -> None:
    context = make_context(
        start_ns=40_000_000_000,
        end_ns=48_000_000_000,
        selected_cameras=(CameraId.CAM_01,),
    )
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    prepared = adapter.prepare_request(context)
    # Native v4-style output labels local timeline nanoseconds as start_ns/end_ns.
    # Because the interval is otherwise impossible on this later context but fits its
    # duration exactly, v6 projects it onto the absolute recording clock.
    native_output = (
        '{"observations":[{'
        '"action":"moves shirt",'
        '"interval":{"start_ns":2000000000,"end_ns":3000000000}'
        "}]} "
    )
    response, artifact_bytes = _response_with_artifact(prepared, native_output)

    observation = adapter.replay_prepared_artifact(
        prepared=prepared,
        response=response,
        artifact_bytes=artifact_bytes,
    )

    assert len(observation.observations) == 1
    action = observation.observations[0]
    assert action.interval.start_ns == 42_000_000_000
    assert action.interval.end_ns == 43_000_000_000


def test_v6_parses_persisted_v4b_artifact_without_rewriting_its_authority() -> None:
    artifact_path = (
        Path(r"D:\Github\Robata\.local\mage-full-v4b-20260808-results")
        / "131ca1c1284dc9db0312469ac62f63f5866a6d231548fa5df89577c9262b9845.json"
    )
    if not artifact_path.is_file():
        pytest.skip("local persisted Mage v4b artifact is not available")

    artifact_bytes = artifact_path.read_bytes()
    artifact = MageVideoResultArtifactDocument.model_validate_json(artifact_bytes, strict=True)
    output_text_before_parse = artifact.output_text
    assert canonical_json_bytes(artifact.model_dump(mode="json")) == artifact_bytes
    assert '"observation_schema_version"' not in artifact.output_text
    assert '"selected_camera_qa"' not in artifact.output_text

    context = _single_camera_context()
    adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    observation = adapter._parse_observation_payload(
        context=context,
        selected_camera=CameraId.CAM_01,
        artifact=artifact,
        inference_artifact_exact_sha256=exact_bytes_sha256(artifact_bytes),
    )

    # Parsing is a projection from the durable source. It neither rewrites the raw
    # document nor replaces the observation's exact-artifact identity with a normalized
    # payload digest.
    assert artifact_path.read_bytes() == artifact_bytes
    assert artifact.output_text == output_text_before_parse
    assert observation.inference_artifact_exact_sha256 == exact_bytes_sha256(artifact_bytes)
    assert observation.observation_schema_version == "mage-observation-v1"
    assert observation.semantic_qa[CameraId.CAM_01].disposition.value == "UNKNOWN"
    assert len(observation.observations) == 6

    first = observation.observations[0]
    assert (
        first.action
        == "a_person_in_a_white_shirt_reaches_for_a_light_green_shirt_on_a_wooden_table"
    )
    assert first.actor is not None
    assert first.actor.actor_type == "person"
    assert first.object is not None
    assert first.object.object_type == "light_green_shirt"
    assert first.object.identity_hint == "light green shirt"
    # The persisted model output uses boundary as a scene-object payload. It is not a
    # valid temporal boundary assessment, so the v6 parser supplies the safe default.
    assert first.boundary.start_confidence == 0.0
    assert first.boundary.end_confidence == 0.0
    assert first.boundary.started_before_context is False
    assert first.boundary.continues_after_context is False


def test_compact_v1_profile_is_versioned_and_shortens_the_instruction() -> None:
    context = _single_camera_context()
    segment = _durable_segment(context)
    default_config = MageVideoObservationAdapterConfig()
    compact_config = MageVideoObservationAdapterConfig.compact_v1()

    default_prompt = build_mage_video_unified_observation_prompt(
        context=context,
        segment=segment,
        config=default_config,
    )
    compact_prompt = build_mage_video_unified_observation_prompt(
        context=context,
        segment=segment,
        config=compact_config,
    )
    projection = json.loads(compact_prompt)

    assert compact_config.output_profile == "COMPACT_V1"
    assert compact_config.prompt_version == MAGE_VIDEO_COMPACT_OBSERVATION_PROMPT_VERSION
    assert compact_config.decoder_id == MAGE_VIDEO_COMPACT_DECODER_ID
    assert compact_config.max_new_tokens == MAGE_VIDEO_COMPACT_DEFAULT_MAX_NEW_TOKENS
    assert projection["output_policy"] == {
        "item_keys": ["action", "interval"],
        "max_observations": MAGE_VIDEO_COMPACT_MAX_OBSERVATIONS,
        "profile": "COMPACT_V1",
        "version": MAGE_VIDEO_COMPACT_OUTPUT_POLICY_VERSION,
    }
    assert projection["response_contract"]["observations"]["item_keys"] == [
        "action",
        "interval",
    ]
    assert projection["output_policy"]["max_observations"] == MAGE_VIDEO_COMPACT_MAX_OBSERVATIONS
    assert len(compact_prompt) < len(default_prompt)
    assert "one JSON object only" in compact_prompt


def test_compact_v1_reserved_identities_cannot_be_mixed_with_full_profile() -> None:
    with pytest.raises(ValueError, match="COMPACT_V1 requires"):
        MageVideoObservationAdapterConfig(output_profile="COMPACT_V1")
    with pytest.raises(ValueError, match="requires COMPACT_V1"):
        MageVideoObservationAdapterConfig(
            prompt_version=MAGE_VIDEO_COMPACT_OBSERVATION_PROMPT_VERSION,
        )


def test_compact_v1_request_identity_and_replay_are_separate_from_full_control() -> None:
    context = _single_camera_context()
    default_adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
    )
    compact_adapter = _adapter(
        resolver=_Resolver((_durable_segment(context),)),
        transport=_Transport(),
        artifact_reader=_ArtifactReader(),
        config=MageVideoObservationAdapterConfig.compact_v1(),
    )

    default_prepared = default_adapter.prepare_request(context)
    compact_prepared = compact_adapter.prepare_request(context)

    assert (
        default_prepared.request_identity_sha256
        == "2700708813ae4e8a886e1129f8f60b59adab6dcce4026d8ef9d62c893a76b9ae"
    )
    assert (
        default_prepared.inference_identity.inference_identity
        == "287655ae47bf83b4326b48672e8d28ceb2210a221cd8ad556e424a0b9c7be88d"
    )
    assert (
        exact_bytes_sha256(default_prepared.request_body)
        == "fce52dc9af9e4eb7227a72ea3acb9e35071549f4fcaa70fd1d778aa753026185"
    )
    assert (
        exact_bytes_sha256(default_prepared.prompt.encode("utf-8"))
        == "7ec247fc6672d9d0b60d82fbb482be715f3448852a5121312adb7164b4fdd779"
    )
    assert (
        compact_prepared.request_identity_sha256
        == "b4dcddc1862b702607b541638c0f98712e3e3f10716d42c44526459c6fdcd192"
    )
    assert (
        compact_prepared.inference_identity.inference_identity
        == "ded31de1fb34a11f875ec19d31440ff98841c2e5a215cf1ec30da04d36d1874d"
    )
    assert (
        exact_bytes_sha256(compact_prepared.request_body)
        == "dfb6b87a60cd010c243d1e27894a9a6837a261d257af05bfe0fce3c75ba6ff58"
    )
    assert (
        exact_bytes_sha256(compact_prepared.prompt.encode("utf-8"))
        == "9abb4348ffdbd60ea04896d2eb14403b580fa4fb81249f78072f7e833c93ee00"
    )
    assert compact_prepared.request_identity_sha256 != default_prepared.request_identity_sha256
    assert compact_prepared.inference_identity != default_prepared.inference_identity
    assert compact_prepared.endpoint_request.decoder.decoder_id == MAGE_VIDEO_COMPACT_DECODER_ID
    assert (
        compact_prepared.endpoint_request.decoder.max_new_tokens
        == MAGE_VIDEO_COMPACT_DEFAULT_MAX_NEW_TOKENS
    )
    assert compact_prepared.endpoint_request.decoder.prompt != (
        default_prepared.endpoint_request.decoder.prompt
    )

    compact_payload = {
        "observations": [
            {
                "action": "pick up cup",
                "interval": {
                    "start_offset_seconds": "1.0",
                    "end_offset_seconds": "2.0",
                },
            }
        ]
    }
    response, artifact_bytes = _response_with_artifact(
        compact_prepared,
        canonical_json_bytes(compact_payload).decode("utf-8"),
    )
    observation = compact_adapter.replay_prepared_artifact(
        prepared=compact_prepared,
        response=response,
        artifact_bytes=artifact_bytes,
    )
    assert observation.prompt_version == (
        f"{MAGE_VIDEO_COMPACT_OBSERVATION_PROMPT_VERSION}+reject_action_v1"
    )
    assert len(observation.observations) == 1
    assert observation.observations[0].action == "pick_up_cup"
    assert observation.observations[0].interval.start_ns == 1_000_000_000
    assert observation.observations[0].interval.end_ns == 2_000_000_000
    assert observation.observations[0].confidence is None
    assert observation.observations[0].camera_evidence[CameraId.CAM_01].visibility is None
