"""Authority-issued capture and immutable pre-EOS segment contracts."""

from __future__ import annotations

from typing import Any, Literal, Self, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import model_validator

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval, SchemaVersion, Sha256Digest, StrictModel
from robata.contracts.hashing import semantic_sha256
from robata.contracts.logical_nodes import OpaqueUuid
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    AuthorityBinding,
    ChannelBinding,
    NonEmptyString,
    PositiveInt,
    PreEosCaptureSubjectRef,
    StreamSegmentRef,
    StreamSubjectType,
)

STREAM_CAPTURE_WIRE_VERSION: Literal["1.0"] = "1.0"
STREAM_SEGMENT_WIRE_VERSION: Literal["1.0"] = "1.0"
PRE_EOS_CAPTURE_SCHEMA_ID = "https://schemas.robata.dev/pre-eos-capture-subject"
PRE_EOS_CAPTURE_SCHEMA_VERSION = "1.0.0"
STREAM_SEGMENT_SCHEMA_ID = "https://schemas.robata.dev/stream-segment"
STREAM_SEGMENT_SCHEMA_VERSION = "1.0.0"
CAPTURE_SEMANTIC_PROJECTION_VERSION = "pre-eos-capture-subject-semantic-v1"
CAPTURE_IDENTITY_POLICY_VERSION = "pre-eos-capture-identity-v1"
CAPTURE_KEY_NAMESPACE = "pre-eos-capture-v1"
SEGMENT_SEMANTIC_PROJECTION_VERSION = "stream-segment-semantic-v1"
SEGMENT_IDENTITY_POLICY_VERSION = "stream-segment-identity-v1"
SEGMENT_KEY_NAMESPACE = "stream-segment-v1"


def _namespace(label: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"robata:stream-namespace:{label}")


PRE_EOS_CAPTURE_V1_NAMESPACE = _namespace(CAPTURE_KEY_NAMESPACE)
STREAM_SEGMENT_V1_NAMESPACE = _namespace(SEGMENT_KEY_NAMESPACE)


def _binding_projection(binding: ChannelBinding) -> dict[str, object]:
    return {
        "camera_id": binding.camera_id.value,
        "source_channel_id": binding.source_channel_id,
        "source_channel_epoch": binding.source_channel_epoch,
        "channel_binding_semantic_sha256": binding.channel_binding_semantic_sha256,
    }


def channel_binding_semantic_projection(binding: ChannelBinding) -> dict[str, object]:
    """Return the digest preimage for a channel binding."""

    return {
        "semantic_projection_version": "stream-channel-binding-semantic-v1",
        **_binding_projection(binding),
    }


def _authority_projection(binding: AuthorityBinding) -> dict[str, object]:
    return {
        "authority_id": binding.authority_id,
        "authority_epoch": binding.authority_epoch,
        "policy_version": binding.policy_version,
        "initial_binding_semantic_sha256": binding.initial_binding_semantic_sha256,
    }


class PreEosCaptureSubject(StrictModel):
    """Immutable capture scope issued before the first stream segment."""

    schema_version: Literal["1.0"] = STREAM_CAPTURE_WIRE_VERSION
    schema_ref: SchemaRef
    subject_type: Literal[StreamSubjectType.PRE_EOS_CAPTURE] = StreamSubjectType.PRE_EOS_CAPTURE
    capture_scope_id: OpaqueUuid
    capture_scope_key: NonEmptyString
    capture_scope_digest: Sha256Digest
    capture_authority_id: NonEmptyString
    capture_authority_epoch: PositiveInt
    capture_assignment_policy_version: SchemaVersion
    acquisition_id: NonEmptyString
    acquisition_epoch: PositiveInt
    channel_bindings: tuple[ChannelBinding, ...]
    mapping_authority: AuthorityBinding
    clock_authority: AuthorityBinding
    semantic_projection_version: SchemaVersion = CAPTURE_SEMANTIC_PROJECTION_VERSION
    identity_policy_version: SchemaVersion = CAPTURE_IDENTITY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.semantic_projection_version != CAPTURE_SEMANTIC_PROJECTION_VERSION:
            raise ValueError("pre-EOS capture uses the registered semantic projection version")
        if self.identity_policy_version != CAPTURE_IDENTITY_POLICY_VERSION:
            raise ValueError("pre-EOS capture uses the registered identity policy version")
        if tuple(binding.camera_id for binding in self.channel_bindings) != tuple(CAMERA_IDS):
            raise ValueError("capture channel bindings must be ordered cam_01 through cam_06")
        if len({binding.source_channel_id for binding in self.channel_bindings}) != len(CAMERA_IDS):
            raise ValueError("capture source channel IDs must be distinct")
        expected_digest = capture_scope_digest_from_values(
            semantic_projection_version=self.semantic_projection_version,
            identity_policy_version=self.identity_policy_version,
            capture_authority_id=self.capture_authority_id,
            capture_authority_epoch=self.capture_authority_epoch,
            capture_assignment_policy_version=self.capture_assignment_policy_version,
            acquisition_id=self.acquisition_id,
            acquisition_epoch=self.acquisition_epoch,
            channel_bindings=self.channel_bindings,
            mapping_authority=self.mapping_authority,
            clock_authority=self.clock_authority,
        )
        if self.capture_scope_digest != expected_digest:
            raise ValueError("capture_scope_digest does not match the authority projection")
        expected_key = f"{CAPTURE_KEY_NAMESPACE}:{expected_digest}"
        if self.capture_scope_key != expected_key:
            raise ValueError("capture_scope_key does not match capture_scope_digest")
        expected_id = str(uuid5(PRE_EOS_CAPTURE_V1_NAMESPACE, expected_key))
        if self.capture_scope_id != expected_id:
            raise ValueError("capture_scope_id does not match capture_scope_key")
        return self

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return self.capture_scope_digest

    def reference(self) -> PreEosCaptureSubjectRef:
        return PreEosCaptureSubjectRef(
            capture_scope_id=self.capture_scope_id,
            capture_scope_key=self.capture_scope_key,
            capture_scope_digest=self.capture_scope_digest,
            identity_policy_version=self.identity_policy_version,
            schema_ref=self.schema_ref,
        )


def capture_scope_semantic_projection(subject: PreEosCaptureSubject) -> dict[str, object]:
    """Return the complete semantic capture projection (no wire metadata)."""

    return capture_scope_projection_from_values(
        semantic_projection_version=subject.semantic_projection_version,
        identity_policy_version=subject.identity_policy_version,
        capture_authority_id=subject.capture_authority_id,
        capture_authority_epoch=subject.capture_authority_epoch,
        capture_assignment_policy_version=subject.capture_assignment_policy_version,
        acquisition_id=subject.acquisition_id,
        acquisition_epoch=subject.acquisition_epoch,
        channel_bindings=subject.channel_bindings,
        mapping_authority=subject.mapping_authority,
        clock_authority=subject.clock_authority,
    )


def capture_scope_projection_from_values(
    *,
    semantic_projection_version: str,
    identity_policy_version: str,
    capture_authority_id: str,
    capture_authority_epoch: int,
    capture_assignment_policy_version: str,
    acquisition_id: str,
    acquisition_epoch: int,
    channel_bindings: tuple[ChannelBinding, ...],
    mapping_authority: AuthorityBinding,
    clock_authority: AuthorityBinding,
) -> dict[str, object]:
    return {
        "semantic_projection_version": semantic_projection_version,
        "identity_policy_version": identity_policy_version,
        "capture_authority_id": capture_authority_id,
        "capture_authority_epoch": capture_authority_epoch,
        "capture_assignment_policy_version": capture_assignment_policy_version,
        "acquisition_id": acquisition_id,
        "acquisition_epoch": acquisition_epoch,
        "ordered_channel_bindings": [_binding_projection(binding) for binding in channel_bindings],
        "mapping_authority": _authority_projection(mapping_authority),
        "clock_authority": _authority_projection(clock_authority),
    }


def capture_scope_digest_from_values(**values: object) -> Sha256Digest:
    return semantic_sha256(capture_scope_projection_from_values(**values))  # type: ignore[arg-type]


def derive_capture_scope_key(capture_scope_digest: Sha256Digest) -> str:
    return f"{CAPTURE_KEY_NAMESPACE}:{capture_scope_digest}"


def derive_capture_scope_id(capture_scope_digest: Sha256Digest) -> OpaqueUuid:
    return str(uuid5(PRE_EOS_CAPTURE_V1_NAMESPACE, derive_capture_scope_key(capture_scope_digest)))


def create_pre_eos_capture_subject(
    *,
    schema_ref: SchemaRef,
    capture_authority_id: str,
    capture_authority_epoch: int,
    capture_assignment_policy_version: str,
    acquisition_id: str,
    acquisition_epoch: int,
    channel_bindings: tuple[ChannelBinding, ...],
    mapping_authority: AuthorityBinding,
    clock_authority: AuthorityBinding,
) -> PreEosCaptureSubject:
    digest = capture_scope_digest_from_values(
        semantic_projection_version=CAPTURE_SEMANTIC_PROJECTION_VERSION,
        identity_policy_version=CAPTURE_IDENTITY_POLICY_VERSION,
        capture_authority_id=capture_authority_id,
        capture_authority_epoch=capture_authority_epoch,
        capture_assignment_policy_version=capture_assignment_policy_version,
        acquisition_id=acquisition_id,
        acquisition_epoch=acquisition_epoch,
        channel_bindings=channel_bindings,
        mapping_authority=mapping_authority,
        clock_authority=clock_authority,
    )
    key = derive_capture_scope_key(digest)
    return PreEosCaptureSubject(
        schema_ref=schema_ref,
        capture_scope_id=derive_capture_scope_id(digest),
        capture_scope_key=key,
        capture_scope_digest=digest,
        capture_authority_id=capture_authority_id,
        capture_authority_epoch=capture_authority_epoch,
        capture_assignment_policy_version=capture_assignment_policy_version,
        acquisition_id=acquisition_id,
        acquisition_epoch=acquisition_epoch,
        channel_bindings=channel_bindings,
        mapping_authority=mapping_authority,
        clock_authority=clock_authority,
    )


class StreamSegmentManifest(StrictModel):
    """Immutable non-final segment subject derived from one camera stream."""

    schema_version: Literal["1.0"] = STREAM_SEGMENT_WIRE_VERSION
    schema_ref: SchemaRef
    subject_type: Literal[StreamSubjectType.STREAM_SEGMENT] = StreamSubjectType.STREAM_SEGMENT
    segment_id: OpaqueUuid
    segment_key: NonEmptyString
    segment_semantic_sha256: Sha256Digest
    capture_scope_digest: Sha256Digest
    camera_id: CameraId
    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    ordered_packet_or_sequence_closure: tuple[NonEmptyString, ...]
    exact_content_sha256: Sha256Digest
    mapping_semantic_sha256: Sha256Digest
    clock_or_alignment_semantic_sha256: Sha256Digest
    segmentation_policy_version: SchemaVersion
    semantic_projection_version: SchemaVersion = SEGMENT_SEMANTIC_PROJECTION_VERSION
    identity_policy_version: SchemaVersion = SEGMENT_IDENTITY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.semantic_projection_version != SEGMENT_SEMANTIC_PROJECTION_VERSION:
            raise ValueError("stream segment uses the registered semantic projection version")
        if self.identity_policy_version != SEGMENT_IDENTITY_POLICY_VERSION:
            raise ValueError("stream segment uses the registered identity policy version")
        if (
            self.effective_interval.start_ns < self.requested_interval.start_ns
            or self.effective_interval.end_ns > self.requested_interval.end_ns
        ):
            raise ValueError("segment effective interval must be contained by requested interval")
        if not self.ordered_packet_or_sequence_closure:
            raise ValueError("segment sequence closure must not be empty")
        expected = segment_semantic_sha256(self)
        if self.segment_semantic_sha256 != expected:
            raise ValueError("segment_semantic_sha256 does not match the segment projection")
        expected_key = derive_segment_key(expected)
        if self.segment_key != expected_key:
            raise ValueError("segment_key does not match segment_semantic_sha256")
        if self.segment_id != derive_segment_id(expected):
            raise ValueError("segment_id does not match segment_key")
        return self

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return self.segment_semantic_sha256

    def reference(self) -> StreamSegmentRef:
        return StreamSegmentRef(
            camera_id=self.camera_id,
            capture_scope_digest=self.capture_scope_digest,
            segment_key=self.segment_key,
            segment_semantic_sha256=self.segment_semantic_sha256,
        )


def segment_semantic_projection(segment: StreamSegmentManifest) -> dict[str, object]:
    """Identity projection for a segment, excluding IDs and wire metadata."""

    return {
        "semantic_projection_version": segment.semantic_projection_version,
        "identity_policy_version": segment.identity_policy_version,
        "capture_scope_digest": segment.capture_scope_digest,
        "camera_id": segment.camera_id.value,
        "requested_interval": {
            "start_ns": str(segment.requested_interval.start_ns),
            "end_ns": str(segment.requested_interval.end_ns),
        },
        "effective_interval": {
            "start_ns": str(segment.effective_interval.start_ns),
            "end_ns": str(segment.effective_interval.end_ns),
        },
        "ordered_packet_or_sequence_closure": list(segment.ordered_packet_or_sequence_closure),
        "exact_content_sha256": segment.exact_content_sha256,
        "mapping_semantic_sha256": segment.mapping_semantic_sha256,
        "clock_or_alignment_semantic_sha256": segment.clock_or_alignment_semantic_sha256,
        "segmentation_policy_version": segment.segmentation_policy_version,
    }


def segment_semantic_sha256(segment: StreamSegmentManifest) -> Sha256Digest:
    return semantic_sha256(segment_semantic_projection(segment))


def derive_segment_key(segment_semantic_sha256: Sha256Digest) -> str:
    return f"{SEGMENT_KEY_NAMESPACE}:{segment_semantic_sha256}"


def derive_segment_id(segment_semantic_sha256: Sha256Digest) -> OpaqueUuid:
    return str(uuid5(STREAM_SEGMENT_V1_NAMESPACE, derive_segment_key(segment_semantic_sha256)))


def create_stream_segment_manifest(
    *,
    schema_ref: SchemaRef,
    capture_scope_digest: Sha256Digest,
    camera_id: CameraId,
    requested_interval: NanosecondInterval,
    effective_interval: NanosecondInterval,
    ordered_packet_or_sequence_closure: tuple[str, ...],
    exact_content_sha256: Sha256Digest,
    mapping_semantic_sha256: Sha256Digest,
    clock_or_alignment_semantic_sha256: Sha256Digest,
    segmentation_policy_version: str,
) -> StreamSegmentManifest:
    values = {
        "semantic_projection_version": SEGMENT_SEMANTIC_PROJECTION_VERSION,
        "identity_policy_version": SEGMENT_IDENTITY_POLICY_VERSION,
        "capture_scope_digest": capture_scope_digest,
        "camera_id": camera_id,
        "requested_interval": requested_interval,
        "effective_interval": effective_interval,
        "ordered_packet_or_sequence_closure": ordered_packet_or_sequence_closure,
        "exact_content_sha256": exact_content_sha256,
        "mapping_semantic_sha256": mapping_semantic_sha256,
        "clock_or_alignment_semantic_sha256": clock_or_alignment_semantic_sha256,
        "segmentation_policy_version": segmentation_policy_version,
    }
    digest = semantic_sha256(values)
    key = derive_segment_key(digest)
    return StreamSegmentManifest(
        schema_ref=schema_ref,
        segment_id=derive_segment_id(digest),
        segment_key=key,
        segment_semantic_sha256=digest,
        **cast(dict[str, Any], values),
    )


__all__ = [
    "CAPTURE_IDENTITY_POLICY_VERSION",
    "CAPTURE_KEY_NAMESPACE",
    "CAPTURE_SEMANTIC_PROJECTION_VERSION",
    "PRE_EOS_CAPTURE_SCHEMA_ID",
    "PRE_EOS_CAPTURE_SCHEMA_VERSION",
    "PRE_EOS_CAPTURE_V1_NAMESPACE",
    "SEGMENT_IDENTITY_POLICY_VERSION",
    "SEGMENT_KEY_NAMESPACE",
    "SEGMENT_SEMANTIC_PROJECTION_VERSION",
    "STREAM_CAPTURE_WIRE_VERSION",
    "STREAM_SEGMENT_SCHEMA_ID",
    "STREAM_SEGMENT_SCHEMA_VERSION",
    "STREAM_SEGMENT_V1_NAMESPACE",
    "STREAM_SEGMENT_WIRE_VERSION",
    "PreEosCaptureSubject",
    "StreamCaptureSubject",
    "StreamSegment",
    "StreamSegmentManifest",
    "capture_scope_digest_from_values",
    "capture_scope_projection_from_values",
    "capture_scope_semantic_projection",
    "channel_binding_semantic_projection",
    "create_pre_eos_capture_subject",
    "create_stream_segment_manifest",
    "derive_capture_scope_id",
    "derive_capture_scope_key",
    "derive_segment_id",
    "derive_segment_key",
    "segment_semantic_projection",
    "segment_semantic_sha256",
]

# Vocabulary aliases used by planner and adapter code.
StreamCaptureSubject = PreEosCaptureSubject
StreamSegment = StreamSegmentManifest
