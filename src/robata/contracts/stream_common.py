"""Provider-neutral primitives for the pre-EOS streaming contract family.

The types in this module are intentionally separate from the published MCAP
contracts.  A stream subject is authority-scoped and must never be encoded as
an ``mcap_id`` sentinel.  Identity helpers in the sibling modules hash only
the explicit semantic projections; storage locations, attempts, leases, and
authority-assigned timestamps stay outside those projections.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.artifacts import ArtifactId, MediaType
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import (
    NanosecondInterval,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.logical_nodes import OpaqueUuid, Rfc3339Timestamp
from robata.contracts.schema_registry import SchemaRef
from robata.queue.stage import DependencyCriticality

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=512)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


class StreamSubjectType(StrEnum):
    """Typed semantic subjects that can appear in the pre-EOS graph."""

    PRE_EOS_CAPTURE = "PRE_EOS_CAPTURE"
    STREAM_SEGMENT = "STREAM_SEGMENT"
    INCREMENTAL_WINDOW = "INCREMENTAL_WINDOW"
    STREAM_INFERENCE = "STREAM_INFERENCE"
    WINDOW_RESULT = "WINDOW_RESULT"
    STREAM_WORK = "STREAM_WORK"
    EXPECTED_WINDOW_PLAN = "EXPECTED_WINDOW_PLAN"
    EXPECTED_WINDOW_DECLARATION = "EXPECTED_WINDOW_DECLARATION"
    EXPECTED_WINDOW_PLAN_SEAL = "EXPECTED_WINDOW_PLAN_SEAL"
    WINDOW_TERMINAL_CLOSURE = "WINDOW_TERMINAL_CLOSURE"
    RECORDING_FINALIZATION = "RECORDING_FINALIZATION"


class StreamPurpose(StrEnum):
    """Provider-neutral purposes used in window and inference identity."""

    QA_COARSE = "QA_COARSE"
    QA_DENSE = "QA_DENSE"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_DENSE = "ACTION_DENSE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"


class StreamStage(StrEnum):
    """Execution stages for stream work, independent of provider names."""

    SEGMENT = "SEGMENT"
    WINDOW = "WINDOW"
    QA_COARSE = "QA_COARSE"
    QA_DENSE = "QA_DENSE"
    EVENT_PROPOSAL = "EVENT_PROPOSAL"
    ACTION_DENSE = "ACTION_DENSE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"
    WINDOW_REDUCTION = "WINDOW_REDUCTION"
    FINALIZATION = "FINALIZATION"


class RefinementRole(StrEnum):
    """Role of a derived boundary window."""

    ONSET = "ONSET"
    OFFSET = "OFFSET"


class CameraAbsenceReason(StrEnum):
    """Explicit source-health facts for a missing six-slot member."""

    ABSENT = "ABSENT"
    LATE = "LATE"
    BLACK = "BLACK"
    FROZEN = "FROZEN"
    DEGRADED = "DEGRADED"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"
    GAP = "GAP"
    UNKNOWN = "UNKNOWN"


class TerminalOutcome(StrEnum):
    """Terminal outcomes retained by stream closure, including failures."""

    SUCCEEDED = "SUCCEEDED"
    SKIPPED_POLICY = "SKIPPED_POLICY"
    SKIPPED_NOT_NEEDED = "SKIPPED_NOT_NEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"
    LATE_INPUT = "LATE_INPUT"
    INCOMPLETE = "INCOMPLETE"
    ABSTAINED = "ABSTAINED"
    NO_EVENTS = "NO_EVENTS"
    INVALIDATED = "INVALIDATED"


class StreamPolicyBinding(StrictModel):
    """A policy version and its immutable semantic digest."""

    version: SchemaVersion
    semantic_sha256: Sha256Digest


class ChannelBinding(StrictModel):
    """One authority-issued camera/channel binding."""

    camera_id: CameraId
    source_channel_id: NonEmptyString
    source_channel_epoch: PositiveInt
    channel_binding_semantic_sha256: Sha256Digest


class AuthorityBinding(StrictModel):
    """Authority and policy binding used by mapping or clock authorities."""

    authority_id: NonEmptyString
    authority_epoch: PositiveInt
    policy_version: SchemaVersion
    initial_binding_semantic_sha256: Sha256Digest


class StreamSubjectRef(StrictModel):
    """Exact typed reference shared by stream work and lineage records."""

    subject_type: StreamSubjectType
    subject_key: NonEmptyString
    subject_semantic_sha256: Sha256Digest
    capture_scope_digest: Sha256Digest
    identity_policy_version: SchemaVersion
    schema_ref: SchemaRef

    @model_validator(mode="after")
    def validate_subject_digest_binding(self) -> Self:
        if not self.subject_key.endswith(f":{self.subject_semantic_sha256}"):
            raise ValueError("subject_key must end with subject_semantic_sha256")
        return self


class PreEosCaptureSubjectRef(StrictModel):
    """Compact reference to an authority-issued capture subject."""

    subject_type: Literal[StreamSubjectType.PRE_EOS_CAPTURE] = StreamSubjectType.PRE_EOS_CAPTURE
    capture_scope_id: OpaqueUuid
    capture_scope_key: NonEmptyString
    capture_scope_digest: Sha256Digest
    identity_policy_version: SchemaVersion
    schema_ref: SchemaRef

    @model_validator(mode="after")
    def validate_capture_digest_binding(self) -> Self:
        prefix = "pre-eos-capture-v1:"
        if self.capture_scope_key != f"{prefix}{self.capture_scope_digest}":
            raise ValueError("capture_scope_key must bind to capture_scope_digest")
        return self


class StreamSegmentRef(StrictModel):
    """One immutable segment member in a camera slot."""

    kind: Literal["SEGMENT"] = "SEGMENT"
    camera_id: CameraId
    capture_scope_digest: Sha256Digest
    segment_key: NonEmptyString
    segment_semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_segment_digest_binding(self) -> Self:
        expected_key = f"stream-segment-v1:{self.segment_semantic_sha256}"
        if self.segment_key != expected_key:
            raise ValueError("segment_key must bind to segment_semantic_sha256")
        return self


class StreamIntervalAbsence(StrictModel):
    """One evidenced missing or unusable interval inside a camera slot.

    This is deliberately distinct from :class:`StreamCameraAbsence`, which
    closes an entire camera slot. Partial gaps must remain explicit members
    of the ordered segment sequence so downstream planners cannot silently
    collapse missing, late, black, frozen, or corrupt input.
    """

    kind: Literal["INTERVAL_ABSENCE"] = "INTERVAL_ABSENCE"
    camera_id: CameraId
    capture_scope_digest: Sha256Digest
    interval: NanosecondInterval
    reason: CameraAbsenceReason
    evidence_sha256: Sha256Digest


type StreamSequenceMember = Annotated[
    StreamSegmentRef | StreamIntervalAbsence,
    Field(discriminator="kind"),
]


class StreamSegmentSequence(StrictModel):
    """Ordered non-empty segment or evidenced-gap members for one camera slot.

    A sequence preserves the planner's temporal order without assuming a fixed
    segment duration in the wire contract. Continuity and coverage against the
    window policy are planner responsibilities; this primitive still rejects
    cross-camera, cross-capture, empty, and repeated members.
    """

    kind: Literal["SEGMENT_SEQUENCE"] = "SEGMENT_SEQUENCE"
    camera_id: CameraId
    capture_scope_digest: Sha256Digest
    ordered_members: tuple[StreamSequenceMember, ...]

    @model_validator(mode="after")
    def validate_sequence(self) -> Self:
        if not self.ordered_members:
            raise ValueError("segment sequence must contain at least one member")
        if any(
            member.camera_id is not self.camera_id
            or member.capture_scope_digest != self.capture_scope_digest
            for member in self.ordered_members
        ):
            raise ValueError("segment sequence members must bind to its camera and capture")
        identities = tuple(_sequence_member_identity(member) for member in self.ordered_members)
        if len(identities) != len(set(identities)):
            raise ValueError("segment sequence members must be unique")
        return self


def _sequence_member_identity(member: StreamSequenceMember) -> tuple[object, ...]:
    if isinstance(member, StreamSegmentRef):
        return (member.kind, member.segment_key)
    return (
        member.kind,
        member.interval.start_ns,
        member.interval.end_ns,
        member.reason.value,
        member.evidence_sha256,
    )


class StreamCameraAbsence(StrictModel):
    """Typed absence/degradation fact; omission is never used for a slot."""

    kind: Literal["ABSENCE"] = "ABSENCE"
    camera_id: CameraId
    reason: CameraAbsenceReason
    evidence_sha256: Sha256Digest | None = None


type CameraSlotClosure = Annotated[
    StreamSegmentRef | StreamSegmentSequence | StreamCameraAbsence,
    Field(discriminator="kind"),
]


class SixCameraSlotClosure(StrictModel):
    """Exactly one ordered camera slot, with one or more segments or absence."""

    slots: tuple[CameraSlotClosure, ...]

    @model_validator(mode="after")
    def validate_slots(self) -> Self:
        if len(self.slots) != len(CAMERA_IDS):
            raise ValueError("six-camera closure must contain exactly six slots")
        cameras = tuple(slot.camera_id for slot in self.slots)
        expected = tuple(CAMERA_IDS)
        if cameras != expected:
            raise ValueError("six-camera closure must be ordered cam_01 through cam_06")
        if len(set(cameras)) != len(cameras):
            raise ValueError("six-camera closure camera IDs must be unique")
        return self

    def as_tuple(self) -> tuple[CameraSlotClosure, ...]:
        return self.slots


class StreamArtifactRef(StrictModel):
    """Exact immutable evidence reference used by terminal closure."""

    artifact_id: ArtifactId
    exact_sha256: Sha256Digest
    byte_count: PositiveInt
    media_type: MediaType
    schema_ref: SchemaRef


def validate_rfc3339(value: str, field_name: str = "timestamp") -> str:
    """Validate calendar and timezone semantics beyond the shared regex."""

    from datetime import datetime

    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a valid RFC3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an RFC3339 timezone")
    return value


# Explicit aliases make the wire vocabulary discoverable without duplicating types.
SegmentReference = StreamSegmentRef
ExplicitAbsence = StreamCameraAbsence
ArtifactEvidenceRef = StreamArtifactRef
PolicyBinding = StreamPolicyBinding


__all__ = [
    "ArtifactEvidenceRef",
    "AuthorityBinding",
    "CameraAbsenceReason",
    "CameraSlotClosure",
    "ChannelBinding",
    "DependencyCriticality",
    "ExplicitAbsence",
    "NonEmptyString",
    "NonNegativeInt",
    "OpaqueUuid",
    "PolicyBinding",
    "PositiveInt",
    "PreEosCaptureSubjectRef",
    "RefinementRole",
    "Rfc3339Timestamp",
    "SegmentReference",
    "SixCameraSlotClosure",
    "StreamArtifactRef",
    "StreamCameraAbsence",
    "StreamIntervalAbsence",
    "StreamPolicyBinding",
    "StreamPurpose",
    "StreamSegmentRef",
    "StreamSegmentSequence",
    "StreamSequenceMember",
    "StreamStage",
    "StreamSubjectRef",
    "StreamSubjectType",
    "TerminalOutcome",
    "validate_rfc3339",
]
