"""Identity-bound Mage fixed-frame control qualification helpers.

This module is deliberately local and non-production. It removes codec preparation and
stream memory while preserving the frozen five-segment cam_01 fixture and exact selected
frame bytes. The prompt is an explicit identity-bound input: callers can run the native
Mage binding prompt or the cross-model Qwen common prompt as separate controls. The
resulting evidence can isolate the visual frontend from codec/runtime effects; it cannot
establish ground-truth accuracy or production admission.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, Final

from pydantic import ValidationError

from robata.benchmark.qwen_mage_common_projection import (
    COMMON_FRAME_SELECTION_VERSION,
    COMMON_MEDIA_CAMERA,
    CommonProjectionCase,
    CommonProjectionError,
)
from robata.contracts.hashing import canonical_json_bytes, exact_bytes_sha256
from robata.contracts.perception_stream import (
    CognitionGateSignal,
    MageObservation,
    create_mage_observation,
)
from robata.inference.mage_video_adapter import (
    MageVideoObservationAdapterError,
    MageVideoObservationPayload,
    _decode_compact_json_object,
    _expand_action_observations,
    _expand_semantic_qa,
    _normalise_compact_numeric_leaves,
    _prepare_compact_payload,
)

MAGE_FIXED_FRAME_POLICY_VERSION: Final = "mage-fixed-frame-control-policy-v1"
MAGE_FIXED_FRAME_ARTIFACT_VERSION: Final = "mage-fixed-frame-control-artifact-v1"
MAGE_FIXED_FRAME_MODEL_FAMILY: Final = "mage_vl_fixed_frames"
MAGE_FIXED_FRAME_TEMPORAL_COORDINATE_POLICY: Final = (
    "dense_processor_ordinals_with_exact_prompt_offsets-v1"
)
MAGE_FIXED_FRAME_NATIVE_PROMPT_VERSION: Final = "mage-unified-observation-prompt-v6"
MAGE_FIXED_FRAME_GATE_POLICY_VERSION: Final = "mage-fixed-frame-gate-disabled-v1"


@dataclass(frozen=True, slots=True)
class MageFixedFrameProjection:
    """Strict compact projection for one fixed-frame Mage generation."""

    observation: MageObservation
    raw_output_text: str
    raw_output_exact_sha256: str
    inference_artifact_exact_sha256: str
    diagnostics: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class MageFixedFrameInputIdentity:
    """Stable logical identity for one exact fixed-frame control invocation."""

    policy_version: str
    context_manifest_semantic_sha256: str
    checkpoint_manifest_sha256: str
    model_revision: str
    load_profile: str
    prompt_version: str
    prompt_exact_sha256: str
    frame_selection_version: str
    temporal_coordinate_policy: str
    max_new_tokens: int
    frame_references: tuple[dict[str, object], ...]
    semantic_sha256: str

    def projection(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "context_manifest_semantic_sha256": self.context_manifest_semantic_sha256,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "model_revision": self.model_revision,
            "load_profile": self.load_profile,
            "prompt_version": self.prompt_version,
            "prompt_exact_sha256": self.prompt_exact_sha256,
            "frame_selection_version": self.frame_selection_version,
            "temporal_coordinate_policy": self.temporal_coordinate_policy,
            "max_new_tokens": self.max_new_tokens,
            "frame_references": list(self.frame_references),
        }


def build_fixed_frame_input_identity(
    *,
    case: CommonProjectionCase,
    checkpoint_manifest_sha256: str,
    model_revision: str,
    load_profile: str,
    max_new_tokens: int,
    prompt: str | None = None,
    prompt_version: str = MAGE_FIXED_FRAME_NATIVE_PROMPT_VERSION,
) -> MageFixedFrameInputIdentity:
    if max_new_tokens <= 0:
        raise CommonProjectionError("fixed-frame max_new_tokens must be positive")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise CommonProjectionError("fixed-frame prompt_version must be nonempty")
    exact_prompt = case.binding.endpoint_request.decoder.prompt if prompt is None else prompt
    if not isinstance(exact_prompt, str) or not exact_prompt.strip():
        raise CommonProjectionError("fixed-frame prompt must be nonempty")
    frame_references = tuple(
        {
            "ordinal": frame.ordinal,
            "aligned_timestamp_ns": str(frame.aligned_timestamp_ns),
            "sha256": frame.sha256,
            "byte_count": frame.byte_count,
            "width": frame.width,
            "height": frame.height,
        }
        for frame in case.selected_frames
    )
    prompt_exact_sha256 = exact_bytes_sha256(exact_prompt.encode("utf-8"))
    draft: dict[str, object] = {
        "policy_version": MAGE_FIXED_FRAME_POLICY_VERSION,
        "context_manifest_semantic_sha256": case.context.context_manifest_semantic_sha256,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "model_revision": model_revision,
        "load_profile": load_profile,
        "prompt_version": prompt_version,
        "prompt_exact_sha256": prompt_exact_sha256,
        "frame_selection_version": COMMON_FRAME_SELECTION_VERSION,
        "temporal_coordinate_policy": MAGE_FIXED_FRAME_TEMPORAL_COORDINATE_POLICY,
        "max_new_tokens": max_new_tokens,
        "frame_references": list(frame_references),
    }
    return MageFixedFrameInputIdentity(
        policy_version=MAGE_FIXED_FRAME_POLICY_VERSION,
        context_manifest_semantic_sha256=case.context.context_manifest_semantic_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        model_revision=model_revision,
        load_profile=load_profile,
        prompt_version=prompt_version,
        prompt_exact_sha256=prompt_exact_sha256,
        frame_selection_version=COMMON_FRAME_SELECTION_VERSION,
        temporal_coordinate_policy=MAGE_FIXED_FRAME_TEMPORAL_COORDINATE_POLICY,
        max_new_tokens=max_new_tokens,
        frame_references=frame_references,
        semantic_sha256=exact_bytes_sha256(canonical_json_bytes(draft)),
    )


def load_verified_fixed_frame_images(case: CommonProjectionCase) -> tuple[Any, ...]:
    """Decode exact selected image bytes after digest and shape verification."""

    image_module = import_module("PIL.Image")
    images: list[Any] = []
    try:
        for frame in case.selected_frames:
            payload = frame.path.read_bytes()
            if len(payload) != frame.byte_count:
                raise CommonProjectionError(f"fixed-frame byte count changed: {frame.path}")
            if exact_bytes_sha256(payload) != frame.sha256:
                raise CommonProjectionError(f"fixed-frame digest changed: {frame.path}")
            opened = image_module.open(io.BytesIO(payload))
            try:
                converted = opened.convert("RGB")
                converted.load()
            finally:
                opened.close()
            if tuple(converted.size) != (frame.width, frame.height):
                converted.close()
                raise CommonProjectionError(f"fixed-frame dimensions changed: {frame.path}")
            images.append(converted)
    except BaseException:
        close_fixed_frame_images(images)
        raise
    return tuple(images)


def close_fixed_frame_images(images: Sequence[Any]) -> None:
    for image in images:
        close = getattr(image, "close", None)
        if callable(close):
            close()


def fixed_frame_artifact_sha256(
    *,
    input_identity: MageFixedFrameInputIdentity,
    output_text: str,
) -> str:
    if not isinstance(output_text, str) or not output_text.strip():
        raise CommonProjectionError("Mage fixed-frame output must be nonempty")
    return exact_bytes_sha256(
        canonical_json_bytes(
            {
                "artifact_version": MAGE_FIXED_FRAME_ARTIFACT_VERSION,
                "model_family": MAGE_FIXED_FRAME_MODEL_FAMILY,
                "input_identity_semantic_sha256": input_identity.semantic_sha256,
                "input_identity": input_identity.projection(),
                "output_text": output_text,
            }
        )
    )


def project_mage_fixed_frame_output(
    *,
    case: CommonProjectionCase,
    input_identity: MageFixedFrameInputIdentity,
    output_text: str,
    created_at: str | None = None,
) -> MageFixedFrameProjection:
    artifact_sha256 = fixed_frame_artifact_sha256(
        input_identity=input_identity,
        output_text=output_text,
    )
    try:
        raw_payload = _decode_compact_json_object(output_text)
        compact_payload = _prepare_compact_payload(
            raw_payload,
            observation_schema_version=case.mage_observation.observation_schema_version,
        )
        normalized_payload = _normalise_compact_numeric_leaves(compact_payload)
        payload = MageVideoObservationPayload.model_validate_json(
            canonical_json_bytes(normalized_payload), strict=True
        )
        semantic_qa = _expand_semantic_qa(
            selected_camera=COMMON_MEDIA_CAMERA,
            selected_qa=payload.selected_camera_qa,
        )
        expanded = _expand_action_observations(
            context=case.context,
            selected_camera=COMMON_MEDIA_CAMERA,
            payloads=payload.observations,
            out_of_context_action_policy="REJECT_ACTION_V1",
            inference_artifact_exact_sha256=artifact_sha256,
        )
    except (MageVideoObservationAdapterError, TypeError, ValueError, ValidationError) as error:
        raise CommonProjectionError(
            f"Mage fixed-frame output for common case {case.ordinal} is not strict compact JSON"
        ) from error
    observation = create_mage_observation(
        observation_schema_version=payload.observation_schema_version,
        context=case.context,
        model_family=MAGE_FIXED_FRAME_MODEL_FAMILY,
        model_revision=input_identity.model_revision,
        model_artifact_manifest_sha256=input_identity.checkpoint_manifest_sha256,
        prompt_version=input_identity.prompt_version,
        inference_artifact_exact_sha256=artifact_sha256,
        cognition_gate=CognitionGateSignal(
            score=None,
            threshold=0.5,
            would_admit=None,
            gate_policy_version=MAGE_FIXED_FRAME_GATE_POLICY_VERSION,
        ),
        semantic_qa=semantic_qa,
        observations=expanded.actions,
        created_at=created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    return MageFixedFrameProjection(
        observation=observation,
        raw_output_text=output_text,
        raw_output_exact_sha256=exact_bytes_sha256(output_text.encode("utf-8")),
        inference_artifact_exact_sha256=artifact_sha256,
        diagnostics=tuple(item.model_dump(mode="json") for item in expanded.diagnostics),
    )


__all__ = [
    "MAGE_FIXED_FRAME_ARTIFACT_VERSION",
    "MAGE_FIXED_FRAME_MODEL_FAMILY",
    "MAGE_FIXED_FRAME_NATIVE_PROMPT_VERSION",
    "MAGE_FIXED_FRAME_POLICY_VERSION",
    "MAGE_FIXED_FRAME_TEMPORAL_COORDINATE_POLICY",
    "MageFixedFrameInputIdentity",
    "MageFixedFrameProjection",
    "build_fixed_frame_input_identity",
    "close_fixed_frame_images",
    "fixed_frame_artifact_sha256",
    "load_verified_fixed_frame_images",
    "project_mage_fixed_frame_output",
]
