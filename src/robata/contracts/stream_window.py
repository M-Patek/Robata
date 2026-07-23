"""Incremental window and logical stream-inference identities."""

from __future__ import annotations

from typing import Any, Literal, Self, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import model_validator

from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    CameraSlotClosure,
    NonEmptyString,
    PositiveInt,
    RefinementRole,
    SixCameraSlotClosure,
    StreamPurpose,
    StreamSegmentRef,
    StreamSegmentSequence,
    StreamSubjectRef,
    StreamSubjectType,
)

WINDOW_SEMANTIC_PROJECTION_VERSION = "incremental-window-semantic-v1"
WINDOW_IDENTITY_POLICY_VERSION = "incremental-window-identity-v1"
WINDOW_KEY_NAMESPACE = "incremental-window-v1"
INFERENCE_SEMANTIC_PROJECTION_VERSION = "stream-inference-semantic-v1"
INFERENCE_IDENTITY_POLICY_VERSION = "stream-inference-identity-v1"
INFERENCE_KEY_NAMESPACE = "stream-inference-v1"
INFERENCE_ATTEMPT_KEY_NAMESPACE = "stream-inference-attempt-v1"
INFERENCE_ATTEMPT_PROJECTION_VERSION = "stream-inference-attempt-semantic-v1"
STREAM_WINDOW_WIRE_VERSION: Literal["1.0"] = "1.0"
STREAM_INFERENCE_WIRE_VERSION: Literal["1.0"] = "1.0"
INCREMENTAL_WINDOW_SCHEMA_ID = "https://schemas.robata.dev/incremental-window"
INCREMENTAL_WINDOW_SCHEMA_VERSION = "1.0.0"
STREAM_INFERENCE_SCHEMA_ID = "https://schemas.robata.dev/stream-inference"
STREAM_INFERENCE_SCHEMA_VERSION = "1.0.0"
STREAM_INFERENCE_ATTEMPT_SCHEMA_ID = "https://schemas.robata.dev/stream-inference-attempt"
STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION = "1.0.0"


def _namespace(label: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"robata:stream-namespace:{label}")


INCREMENTAL_WINDOW_V1_NAMESPACE = _namespace(WINDOW_KEY_NAMESPACE)
STREAM_INFERENCE_V1_NAMESPACE = _namespace(INFERENCE_KEY_NAMESPACE)
STREAM_INFERENCE_ATTEMPT_V1_NAMESPACE = _namespace(INFERENCE_ATTEMPT_KEY_NAMESPACE)


def _interval_projection(interval: NanosecondInterval) -> dict[str, str]:
    return {"start_ns": str(interval.start_ns), "end_ns": str(interval.end_ns)}


def _closure_projection(slots: tuple[CameraSlotClosure, ...]) -> list[dict[str, object]]:
    return [slot.model_dump(mode="json") for slot in slots]


class IncrementalWindow(StrictModel):
    """Immutable non-final window subject under a pre-EOS capture scope."""

    schema_version: Literal["1.0"] = STREAM_WINDOW_WIRE_VERSION
    schema_ref: SchemaRef
    subject_type: Literal[StreamSubjectType.INCREMENTAL_WINDOW] = (
        StreamSubjectType.INCREMENTAL_WINDOW
    )
    window_id: OpaqueUuid
    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    capture_scope_digest: Sha256Digest
    purpose: StreamPurpose
    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    ordered_six_slot_segment_or_explicit_absence_closure: tuple[CameraSlotClosure, ...]
    mapping_semantic_sha256: Sha256Digest
    clock_or_alignment_semantic_sha256: Sha256Digest
    parent_subject_key_or_none: NonEmptyString | None = None
    refinement_role_or_none: RefinementRole | None = None
    refinement_generation: int = 0
    window_policy_version: SchemaVersion
    semantic_projection_version: SchemaVersion = WINDOW_SEMANTIC_PROJECTION_VERSION
    identity_policy_version: SchemaVersion = WINDOW_IDENTITY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.semantic_projection_version != WINDOW_SEMANTIC_PROJECTION_VERSION:
            raise ValueError("incremental window uses the registered semantic projection version")
        if self.identity_policy_version != WINDOW_IDENTITY_POLICY_VERSION:
            raise ValueError("incremental window uses the registered identity policy version")
        if (
            self.effective_interval.start_ns < self.requested_interval.start_ns
            or self.effective_interval.end_ns > self.requested_interval.end_ns
        ):
            raise ValueError("window effective interval must be contained by requested interval")
        if self.refinement_generation < 0:
            raise ValueError("refinement_generation must be nonnegative")
        if self.refinement_generation == 0 and (
            self.parent_subject_key_or_none is not None or self.refinement_role_or_none is not None
        ):
            raise ValueError("generation zero windows cannot have refinement lineage")
        if self.refinement_generation > 0 and (
            self.parent_subject_key_or_none is None or self.refinement_role_or_none is None
        ):
            raise ValueError("derived windows require a parent subject key and refinement role")
        SixCameraSlotClosure(slots=self.ordered_six_slot_segment_or_explicit_absence_closure)
        if any(
            isinstance(slot, (StreamSegmentRef, StreamSegmentSequence))
            and slot.capture_scope_digest != self.capture_scope_digest
            for slot in self.ordered_six_slot_segment_or_explicit_absence_closure
        ):
            raise ValueError("window segment references must bind to capture_scope_digest")
        expected = window_semantic_sha256(self)
        if self.window_semantic_sha256 != expected:
            raise ValueError("window_semantic_sha256 does not match the window projection")
        expected_key = derive_window_key(expected)
        if self.window_key != expected_key:
            raise ValueError("window_key does not match window_semantic_sha256")
        if self.window_id != derive_window_id(expected):
            raise ValueError("window_id does not match window_key")
        return self

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return self.window_semantic_sha256

    @property
    def camera_closure(self) -> tuple[CameraSlotClosure, ...]:
        """Short alias used by planners while the wire name stays explicit."""

        return self.ordered_six_slot_segment_or_explicit_absence_closure

    def reference(self) -> StreamSubjectRef:
        return StreamSubjectRef(
            subject_type=StreamSubjectType.INCREMENTAL_WINDOW,
            subject_key=self.window_key,
            subject_semantic_sha256=self.window_semantic_sha256,
            capture_scope_digest=self.capture_scope_digest,
            identity_policy_version=self.identity_policy_version,
            schema_ref=self.schema_ref,
        )


def window_semantic_projection(window: IncrementalWindow) -> dict[str, object]:
    """Identity projection for a window, excluding IDs and exact locators."""

    return {
        "semantic_projection_version": window.semantic_projection_version,
        "identity_policy_version": window.identity_policy_version,
        "capture_scope_digest": window.capture_scope_digest,
        "purpose": window.purpose.value,
        "requested_interval": _interval_projection(window.requested_interval),
        "effective_interval": _interval_projection(window.effective_interval),
        "ordered_six_slot_segment_or_explicit_absence_closure": _closure_projection(
            window.ordered_six_slot_segment_or_explicit_absence_closure
        ),
        "mapping_semantic_sha256": window.mapping_semantic_sha256,
        "clock_or_alignment_semantic_sha256": window.clock_or_alignment_semantic_sha256,
        "parent_subject_key_or_none": window.parent_subject_key_or_none,
        "refinement_role_or_none": (
            window.refinement_role_or_none.value if window.refinement_role_or_none else None
        ),
        "refinement_generation": window.refinement_generation,
        "window_policy_version": window.window_policy_version,
    }


def window_semantic_sha256(window: IncrementalWindow) -> Sha256Digest:
    return semantic_sha256(window_semantic_projection(window))


def derive_window_key(window_semantic_sha256: Sha256Digest) -> str:
    return f"{WINDOW_KEY_NAMESPACE}:{window_semantic_sha256}"


def derive_window_id(window_semantic_sha256: Sha256Digest) -> OpaqueUuid:
    return str(uuid5(INCREMENTAL_WINDOW_V1_NAMESPACE, derive_window_key(window_semantic_sha256)))


def create_incremental_window(
    *,
    schema_ref: SchemaRef,
    capture_scope_digest: Sha256Digest,
    purpose: StreamPurpose,
    requested_interval: NanosecondInterval,
    effective_interval: NanosecondInterval,
    ordered_six_slot_segment_or_explicit_absence_closure: tuple[CameraSlotClosure, ...],
    mapping_semantic_sha256: Sha256Digest,
    clock_or_alignment_semantic_sha256: Sha256Digest,
    window_policy_version: str,
    parent_subject_key_or_none: str | None = None,
    refinement_role_or_none: RefinementRole | None = None,
    refinement_generation: int = 0,
) -> IncrementalWindow:
    values = {
        "semantic_projection_version": WINDOW_SEMANTIC_PROJECTION_VERSION,
        "identity_policy_version": WINDOW_IDENTITY_POLICY_VERSION,
        "capture_scope_digest": capture_scope_digest,
        "purpose": purpose,
        "requested_interval": requested_interval,
        "effective_interval": effective_interval,
        "ordered_six_slot_segment_or_explicit_absence_closure": (
            ordered_six_slot_segment_or_explicit_absence_closure
        ),
        "mapping_semantic_sha256": mapping_semantic_sha256,
        "clock_or_alignment_semantic_sha256": clock_or_alignment_semantic_sha256,
        "parent_subject_key_or_none": parent_subject_key_or_none,
        "refinement_role_or_none": refinement_role_or_none,
        "refinement_generation": refinement_generation,
        "window_policy_version": window_policy_version,
    }
    digest = semantic_sha256(
        window_semantic_projection(
            IncrementalWindow.model_construct(
                **cast(dict[str, Any], values),
                window_semantic_sha256="0" * 64,
                window_key="x",
                window_id="00000000-0000-0000-0000-000000000000",
                schema_ref=schema_ref,
            )
        )
    )
    key = derive_window_key(digest)
    return IncrementalWindow(
        schema_ref=schema_ref,
        window_id=derive_window_id(digest),
        window_key=key,
        window_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


class StreamInferenceLogicalIdentity(StrictModel):
    """Stable logical invocation identity; attempts are separate records."""

    schema_version: Literal["1.0"] = STREAM_INFERENCE_WIRE_VERSION
    schema_ref: SchemaRef
    subject_type: Literal[StreamSubjectType.STREAM_INFERENCE] = StreamSubjectType.STREAM_INFERENCE
    stream_inference_logical_id: OpaqueUuid
    inference_key: NonEmptyString
    inference_semantic_sha256: Sha256Digest
    window_key: NonEmptyString
    window_semantic_sha256: Sha256Digest
    purpose: StreamPurpose
    input_plan_semantic_sha256: Sha256Digest
    inference_projection_version: SchemaVersion = INFERENCE_SEMANTIC_PROJECTION_VERSION
    inference_identity_policy_version: SchemaVersion = INFERENCE_IDENTITY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.inference_projection_version != INFERENCE_SEMANTIC_PROJECTION_VERSION:
            raise ValueError("stream inference uses the registered semantic projection version")
        if self.inference_identity_policy_version != INFERENCE_IDENTITY_POLICY_VERSION:
            raise ValueError("stream inference uses the registered identity policy version")
        expected = stream_inference_semantic_sha256(self)
        if self.inference_semantic_sha256 != expected:
            raise ValueError("inference_semantic_sha256 does not match the logical projection")
        if self.inference_key != derive_inference_key(expected):
            raise ValueError("inference_key does not match inference_semantic_sha256")
        if self.stream_inference_logical_id != derive_stream_inference_logical_id(expected):
            raise ValueError("stream_inference_logical_id does not match inference_key")
        return self

    @property
    def logical_invocation_id(self) -> OpaqueUuid:
        return self.stream_inference_logical_id


class StreamInferenceAttemptIdentity(StrictModel):
    """One dispatch attempt, separate from the stable logical invocation."""

    schema_version: Literal["1.0"] = STREAM_INFERENCE_WIRE_VERSION
    schema_ref: SchemaRef
    inference_attempt_id: OpaqueUuid
    inference_attempt_key: NonEmptyString
    stream_inference_logical_id: OpaqueUuid
    attempt_number: PositiveInt
    attempt_projection_version: SchemaVersion = INFERENCE_ATTEMPT_PROJECTION_VERSION

    @model_validator(mode="after")
    def validate_attempt_identity(self) -> Self:
        if self.attempt_projection_version != INFERENCE_ATTEMPT_PROJECTION_VERSION:
            raise ValueError("inference attempt uses the registered projection version")
        expected_key = derive_inference_attempt_key(
            logical_id=self.stream_inference_logical_id,
            attempt_number=self.attempt_number,
        )
        if self.inference_attempt_key != expected_key:
            raise ValueError("inference_attempt_key does not match logical ID and attempt number")
        if self.inference_attempt_id != derive_inference_attempt_id(expected_key):
            raise ValueError("inference_attempt_id does not match inference_attempt_key")
        return self

    @property
    def attempt_id(self) -> OpaqueUuid:
        return self.inference_attempt_id


def stream_inference_semantic_projection(
    identity: StreamInferenceLogicalIdentity,
) -> dict[str, object]:
    return {
        "inference_projection_version": identity.inference_projection_version,
        "inference_identity_policy_version": identity.inference_identity_policy_version,
        "window_key": identity.window_key,
        "window_semantic_sha256": identity.window_semantic_sha256,
        "purpose": identity.purpose.value,
        "input_plan_semantic_sha256": identity.input_plan_semantic_sha256,
    }


def stream_inference_semantic_sha256(identity: StreamInferenceLogicalIdentity) -> Sha256Digest:
    return semantic_sha256(stream_inference_semantic_projection(identity))


def derive_inference_key(inference_semantic_sha256: Sha256Digest) -> str:
    return f"{INFERENCE_KEY_NAMESPACE}:{inference_semantic_sha256}"


def derive_stream_inference_logical_id(inference_semantic_sha256: Sha256Digest) -> OpaqueUuid:
    return str(
        uuid5(STREAM_INFERENCE_V1_NAMESPACE, derive_inference_key(inference_semantic_sha256))
    )


def create_stream_inference_identity(
    *,
    schema_ref: SchemaRef,
    window_key: str,
    window_semantic_sha256: Sha256Digest,
    purpose: StreamPurpose,
    input_plan_semantic_sha256: Sha256Digest,
) -> StreamInferenceLogicalIdentity:
    values = {
        "inference_projection_version": INFERENCE_SEMANTIC_PROJECTION_VERSION,
        "inference_identity_policy_version": INFERENCE_IDENTITY_POLICY_VERSION,
        "window_key": window_key,
        "window_semantic_sha256": window_semantic_sha256,
        "purpose": purpose,
        "input_plan_semantic_sha256": input_plan_semantic_sha256,
    }
    digest = semantic_sha256(values)
    return StreamInferenceLogicalIdentity(
        schema_ref=schema_ref,
        stream_inference_logical_id=derive_stream_inference_logical_id(digest),
        inference_key=derive_inference_key(digest),
        inference_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


def derive_inference_attempt_key(*, logical_id: OpaqueUuid, attempt_number: int) -> str:
    if attempt_number < 1:
        raise ValueError("attempt_number must be positive")
    return f"{INFERENCE_ATTEMPT_KEY_NAMESPACE}:{logical_id}:{attempt_number}"


def derive_inference_attempt_id(inference_attempt_key: str) -> OpaqueUuid:
    return str(uuid5(STREAM_INFERENCE_ATTEMPT_V1_NAMESPACE, inference_attempt_key))


def create_stream_inference_attempt_identity(
    *,
    schema_ref: SchemaRef,
    stream_inference_logical_id: OpaqueUuid,
    attempt_number: int,
) -> StreamInferenceAttemptIdentity:
    key = derive_inference_attempt_key(
        logical_id=stream_inference_logical_id,
        attempt_number=attempt_number,
    )
    return StreamInferenceAttemptIdentity(
        schema_ref=schema_ref,
        inference_attempt_id=derive_inference_attempt_id(key),
        inference_attempt_key=key,
        stream_inference_logical_id=stream_inference_logical_id,
        attempt_number=attempt_number,
    )


__all__ = [
    "INCREMENTAL_WINDOW_SCHEMA_ID",
    "INCREMENTAL_WINDOW_SCHEMA_VERSION",
    "INCREMENTAL_WINDOW_V1_NAMESPACE",
    "INFERENCE_ATTEMPT_KEY_NAMESPACE",
    "INFERENCE_ATTEMPT_PROJECTION_VERSION",
    "INFERENCE_IDENTITY_POLICY_VERSION",
    "INFERENCE_KEY_NAMESPACE",
    "INFERENCE_SEMANTIC_PROJECTION_VERSION",
    "STREAM_INFERENCE_ATTEMPT_SCHEMA_ID",
    "STREAM_INFERENCE_ATTEMPT_SCHEMA_VERSION",
    "STREAM_INFERENCE_ATTEMPT_V1_NAMESPACE",
    "STREAM_INFERENCE_SCHEMA_ID",
    "STREAM_INFERENCE_SCHEMA_VERSION",
    "STREAM_INFERENCE_V1_NAMESPACE",
    "STREAM_INFERENCE_WIRE_VERSION",
    "STREAM_WINDOW_WIRE_VERSION",
    "WINDOW_IDENTITY_POLICY_VERSION",
    "WINDOW_KEY_NAMESPACE",
    "WINDOW_SEMANTIC_PROJECTION_VERSION",
    "IncrementalWindow",
    "StreamInferenceAttemptIdentity",
    "StreamInferenceIdentity",
    "StreamInferenceLogicalIdentity",
    "StreamWindow",
    "create_incremental_window",
    "create_stream_inference_attempt_identity",
    "create_stream_inference_identity",
    "derive_inference_attempt_id",
    "derive_inference_attempt_key",
    "derive_inference_key",
    "derive_stream_inference_logical_id",
    "derive_window_id",
    "derive_window_key",
    "stream_inference_semantic_projection",
    "stream_inference_semantic_sha256",
    "window_semantic_projection",
    "window_semantic_sha256",
]

StreamInferenceIdentity = StreamInferenceLogicalIdentity
StreamWindow = IncrementalWindow
