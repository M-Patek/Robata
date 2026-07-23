"""Bounded, single-pass media planning primitives for the streaming iteration.

The module deliberately stops at deterministic planning.  A source adapter owns MCAP
decoding and durable publication; it feeds each encoded access unit to
``BoundedSinglePassMediaPlanner`` once and persists the emitted references.  No full
recording index, decoded frame, or provider result is retained here.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Final, Protocol

from robata.application.canonical.media_quality import FrameTimingEvidence
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval, Sha256Digest
from robata.contracts.hashing import semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.contracts.stream_common import (
    CameraAbsenceReason,
    StreamIntervalAbsence,
    StreamPurpose,
    StreamSegmentRef,
    StreamSegmentSequence,
)
from robata.contracts.stream_source import (
    StreamSegmentManifest,
    create_stream_segment_manifest,
)
from robata.contracts.stream_window import IncrementalWindow, create_incremental_window

SEGMENT_SEMANTIC_PROJECTION_VERSION: Final = "stream-segment-semantic-v1"
SEGMENT_IDENTITY_POLICY_VERSION: Final = "stream-segment-identity-v1"
SEGMENT_KEY_NAMESPACE: Final = "stream-segment-v1"
WINDOW_PLANNING_PROJECTION_VERSION: Final = "bounded-window-planning-v1"
ACCESS_UNIT_FRAMING_VERSION: Final = "length-prefixed-access-unit-v1"


class WindowClosureReason(StrEnum):
    """Why an incremental window became immutable to the planner."""

    WATERMARK = "WATERMARK"
    EOS = "EOS"


@dataclass(frozen=True, slots=True)
class BoundedMediaPolicy:
    """Versioned bounds used by the local single-pass planner.

    These are engineering defaults, not promoted quality or production policies.  The
    source scope and mapping/alignment digests are required so a local plan cannot be
    accidentally reused across recordings.
    """

    source_scope_digest: Sha256Digest
    mapping_semantic_sha256: Sha256Digest
    alignment_semantic_sha256: Sha256Digest
    source_origin_ns: int = 0
    segment_duration_ns: int = 1_000_000_000
    window_width_ns: int = 2_000_000_000
    window_hop_ns: int = 1_000_000_000
    allowed_lateness_ns: int = 300_000_000
    ring_duration_ns: int = 10_000_000_000
    ring_max_bytes_per_camera: int = 64 * 1024 * 1024
    quality_period_ns: int = 500_000_000
    quality_target_phase_ns: int = 0
    quality_selection_tolerance_ns: int = 300_000_000
    window_purpose: StreamPurpose = StreamPurpose.QA_COARSE
    segmentation_policy_version: str = "stream-segmentation-policy-v1"
    window_policy_version: str = "stream-window-policy-v1"
    quality_policy_version: str = "stream-quality-plan-v1"

    def __post_init__(self) -> None:
        for name in (
            "source_scope_digest",
            "mapping_semantic_sha256",
            "alignment_semantic_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        positive = (
            "segment_duration_ns",
            "window_width_ns",
            "window_hop_ns",
            "ring_duration_ns",
            "ring_max_bytes_per_camera",
            "quality_period_ns",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        nonnegative = (
            "source_origin_ns",
            "allowed_lateness_ns",
            "quality_target_phase_ns",
            "quality_selection_tolerance_ns",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.window_hop_ns > self.window_width_ns:
            raise ValueError("window_hop_ns cannot exceed window_width_ns")
        if (
            self.window_width_ns % self.segment_duration_ns
            or self.window_hop_ns % self.segment_duration_ns
        ):
            raise ValueError("window width and hop must align to logical segment boundaries")
        if (
            self.window_width_ns % self.quality_period_ns
            or self.window_hop_ns % self.quality_period_ns
        ):
            raise ValueError("window width and hop must align to quality target buckets")
        if self.quality_target_phase_ns >= self.quality_period_ns:
            raise ValueError("quality_target_phase_ns must be less than quality_period_ns")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.segmentation_policy_version,
                self.window_policy_version,
                self.quality_policy_version,
            )
        ):
            raise ValueError("policy versions must be non-empty strings")


@dataclass(frozen=True, slots=True)
class EncodedMediaPacket:
    """One encoded access unit observed during the source's ordered traversal."""

    traversal_index: int
    camera_id: CameraId
    source_order: int
    source_sequence: int
    source_timestamp_ns: int
    aligned_timestamp_ns: int
    source_locator: str
    payload: bytes
    is_keyframe: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "traversal_index",
            "source_order",
            "source_sequence",
            "source_timestamp_ns",
            "aligned_timestamp_ns",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.traversal_index < 0 or self.source_order < 0 or self.source_sequence < 0:
            raise ValueError("traversal and source ordinals must be nonnegative")
        if not isinstance(self.source_locator, str) or not self.source_locator:
            raise ValueError("source_locator must be non-empty")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("encoded payload must be non-empty bytes")
        if not isinstance(self.is_keyframe, bool):
            raise TypeError("is_keyframe must be a boolean")

    @property
    def payload_sha256(self) -> Sha256Digest:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def payload_bytes(self) -> int:
        return len(self.payload)

    def framing_bytes(self) -> bytes:
        """Return the deterministic append-only spool record for this access unit."""

        return len(self.payload).to_bytes(8, "big", signed=False) + self.payload

    def reference(self) -> PacketReference:
        return PacketReference(
            traversal_index=self.traversal_index,
            camera_id=self.camera_id,
            source_order=self.source_order,
            source_sequence=self.source_sequence,
            source_timestamp_ns=self.source_timestamp_ns,
            aligned_timestamp_ns=self.aligned_timestamp_ns,
            source_locator=self.source_locator,
            payload_sha256=self.payload_sha256,
            payload_bytes=self.payload_bytes,
            is_keyframe=self.is_keyframe,
        )


@dataclass(frozen=True, slots=True)
class PacketReference:
    """Metadata retained after the encoded bytes leave the bounded ring."""

    traversal_index: int
    camera_id: CameraId
    source_order: int
    source_sequence: int
    source_timestamp_ns: int
    aligned_timestamp_ns: int
    source_locator: str
    payload_sha256: Sha256Digest
    payload_bytes: int
    is_keyframe: bool

    def __post_init__(self) -> None:
        if (
            self.traversal_index < 0
            or self.source_order < 0
            or self.source_sequence < 0
            or self.payload_bytes <= 0
        ):
            raise ValueError("packet reference ordinals and byte count are invalid")
        if not self.source_locator:
            raise ValueError("packet reference locator must be non-empty")
        if len(self.payload_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.payload_sha256
        ):
            raise ValueError("packet reference payload digest must be lowercase SHA-256")

    @property
    def closure_token(self) -> str:
        return f"{self.source_order}:{self.source_sequence}:{self.source_timestamp_ns}"


@dataclass(frozen=True, slots=True)
class RingSnapshot:
    """Observable bounded compressed context for one camera."""

    camera_id: CameraId
    packets: tuple[EncodedMediaPacket, ...]
    total_bytes: int
    evicted_packet_count: int


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """One immutable logical segment, without retaining decoded frames."""

    camera_id: CameraId
    segment_ordinal: int
    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    packets: tuple[PacketReference, ...]
    exact_content_sha256: Sha256Digest
    payload_bytes: int
    capture_scope_digest: Sha256Digest
    mapping_semantic_sha256: Sha256Digest
    clock_or_alignment_semantic_sha256: Sha256Digest
    segmentation_policy_version: str
    semantic_projection_version: str = SEGMENT_SEMANTIC_PROJECTION_VERSION
    identity_policy_version: str = SEGMENT_IDENTITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.packets:
            raise ValueError("segment plan cannot be empty")
        if any(packet.camera_id is not self.camera_id for packet in self.packets):
            raise ValueError("segment packet references must bind to their camera")
        if tuple(packet.source_order for packet in self.packets) != tuple(
            sorted(packet.source_order for packet in self.packets)
        ):
            raise ValueError("segment packet references must remain in source order")
        if self.payload_bytes != sum(packet.payload_bytes for packet in self.packets):
            raise ValueError("segment payload byte count must reconcile to packet references")
        if self.semantic_projection_version != SEGMENT_SEMANTIC_PROJECTION_VERSION:
            raise ValueError("segment semantic projection version is fixed")
        if self.identity_policy_version != SEGMENT_IDENTITY_POLICY_VERSION:
            raise ValueError("segment identity policy version is fixed")

    @property
    def source_sequence_closure(self) -> tuple[str, ...]:
        return tuple(packet.closure_token for packet in self.packets)

    @property
    def semantic_projection(self) -> dict[str, object]:
        return {
            "semantic_projection_version": self.semantic_projection_version,
            "identity_policy_version": self.identity_policy_version,
            "capture_scope_digest": self.capture_scope_digest,
            "camera_id": self.camera_id.value,
            "requested_interval": {
                "start_ns": str(self.requested_interval.start_ns),
                "end_ns": str(self.requested_interval.end_ns),
            },
            "effective_interval": {
                "start_ns": str(self.effective_interval.start_ns),
                "end_ns": str(self.effective_interval.end_ns),
            },
            "ordered_packet_or_sequence_closure": list(self.source_sequence_closure),
            "exact_content_sha256": self.exact_content_sha256,
            "mapping_semantic_sha256": self.mapping_semantic_sha256,
            "clock_or_alignment_semantic_sha256": self.clock_or_alignment_semantic_sha256,
            "segmentation_policy_version": self.segmentation_policy_version,
        }

    @property
    def semantic_sha256(self) -> Sha256Digest:
        return semantic_sha256(self.semantic_projection)

    @property
    def segment_key(self) -> str:
        return f"{SEGMENT_KEY_NAMESPACE}:{self.semantic_sha256}"

    def reference(self) -> StreamSegmentRef:
        return StreamSegmentRef(
            camera_id=self.camera_id,
            capture_scope_digest=self.capture_scope_digest,
            segment_key=self.segment_key,
            segment_semantic_sha256=self.semantic_sha256,
        )

    def to_stream_segment_manifest(self, schema_ref: SchemaRef) -> StreamSegmentManifest:
        """Map to the WP1 Wire model and prove both identity projections agree."""

        manifest = create_stream_segment_manifest(
            schema_ref=schema_ref,
            capture_scope_digest=self.capture_scope_digest,
            camera_id=self.camera_id,
            requested_interval=self.requested_interval,
            effective_interval=self.effective_interval,
            ordered_packet_or_sequence_closure=self.source_sequence_closure,
            exact_content_sha256=self.exact_content_sha256,
            mapping_semantic_sha256=self.mapping_semantic_sha256,
            clock_or_alignment_semantic_sha256=self.clock_or_alignment_semantic_sha256,
            segmentation_policy_version=self.segmentation_policy_version,
        )
        if manifest.segment_semantic_sha256 != self.semantic_sha256:
            raise AssertionError("bounded segment and WP1 segment projections diverged")
        return manifest


@dataclass(frozen=True, slots=True)
class QualityTarget:
    """One bounded low-cost decode target selected from encoded packets."""

    camera_id: CameraId
    bucket_ordinal: int
    requested_target_ns: int
    packet: PacketReference
    delta_ns: int
    policy_version: str

    def timing_evidence(self) -> FrameTimingEvidence:
        """Adapt the target to the existing incremental quality analyzer port."""

        return FrameTimingEvidence(
            camera_id=self.camera_id,
            packet_index=self.packet.source_order,
            aligned_timestamp_ns=self.packet.aligned_timestamp_ns,
            source_timestamp_ns=self.packet.source_timestamp_ns,
            source_sequence=self.packet.source_sequence,
        )


@dataclass(frozen=True, slots=True)
class QualityGap:
    """An explicit quality target that had no packet within the registered tolerance."""

    camera_id: CameraId
    bucket_ordinal: int
    requested_target_ns: int
    interval: NanosecondInterval
    reason: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class WindowMember:
    """One segment bucket or explicit interval absence in a camera window."""

    camera_id: CameraId
    interval: NanosecondInterval
    segment: SegmentPlan | None = None
    absence_reason: CameraAbsenceReason | None = None
    absence_evidence_sha256: Sha256Digest | None = None

    def __post_init__(self) -> None:
        has_segment = self.segment is not None
        has_absence = self.absence_reason is not None or self.absence_evidence_sha256 is not None
        if has_segment == has_absence:
            raise ValueError("window member must be either a segment or an explicit absence")
        if (
            has_segment
            and self.segment is not None
            and self.segment.camera_id is not self.camera_id
        ):
            raise ValueError("window segment camera differs from member camera")
        if has_absence and self.absence_evidence_sha256 is None:
            raise ValueError("window absence must carry deterministic evidence")

    @property
    def kind(self) -> str:
        return "SEGMENT" if self.segment is not None else "ABSENCE"


@dataclass(frozen=True, slots=True)
class CameraWindowPlan:
    """Ordered per-camera closure; partial source gaps remain explicit."""

    camera_id: CameraId
    members: tuple[WindowMember, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("camera window plan requires at least one logical bucket")
        if tuple(member.camera_id for member in self.members) != (self.camera_id,) * len(
            self.members
        ):
            raise ValueError("camera window members must bind to their camera")
        intervals = tuple(member.interval for member in self.members)
        if any(left.end_ns != right.start_ns for left, right in pairwise(intervals)):
            raise ValueError("camera window members must be ordered and contiguous")


@dataclass(frozen=True, slots=True)
class BoundedWindowPlan:
    """A deterministic six-camera incremental window emitted at watermark or EOS."""

    ordinal: int
    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    camera_plans: tuple[CameraWindowPlan, ...]
    quality_targets: tuple[QualityTarget, ...]
    quality_gaps: tuple[QualityGap, ...]
    watermark_ns: int | None
    closure_reason: WindowClosureReason
    capture_scope_digest: Sha256Digest
    mapping_semantic_sha256: Sha256Digest
    clock_or_alignment_semantic_sha256: Sha256Digest
    window_policy_version: str
    quality_policy_version: str
    purpose: StreamPurpose = StreamPurpose.QA_COARSE

    def __post_init__(self) -> None:
        if tuple(plan.camera_id for plan in self.camera_plans) != CAMERA_IDS:
            raise ValueError("bounded window must contain six ordered camera plans")
        if self.effective_interval.start_ns < self.requested_interval.start_ns:
            raise ValueError("window effective interval must be inside requested interval")
        if self.effective_interval.end_ns > self.requested_interval.end_ns:
            raise ValueError("window effective interval must be inside requested interval")

    @property
    def planning_projection(self) -> dict[str, object]:
        """Return a diagnostic planning digest, never a logical window identity."""

        members: list[dict[str, object]] = []
        for plan in self.camera_plans:
            members.append(
                {
                    "camera_id": plan.camera_id.value,
                    "members": [
                        {
                            "interval": {
                                "start_ns": str(member.interval.start_ns),
                                "end_ns": str(member.interval.end_ns),
                            },
                            "kind": member.kind,
                            "segment_semantic_sha256": (
                                member.segment.semantic_sha256
                                if member.segment is not None
                                else None
                            ),
                            "absence_reason": (
                                member.absence_reason.value if member.absence_reason else None
                            ),
                            "absence_evidence_sha256": member.absence_evidence_sha256,
                        }
                        for member in plan.members
                    ],
                }
            )
        return {
            "planning_projection_version": WINDOW_PLANNING_PROJECTION_VERSION,
            "capture_scope_digest": self.capture_scope_digest,
            "purpose": self.purpose.value,
            "requested_interval": {
                "start_ns": str(self.requested_interval.start_ns),
                "end_ns": str(self.requested_interval.end_ns),
            },
            "effective_interval": {
                "start_ns": str(self.effective_interval.start_ns),
                "end_ns": str(self.effective_interval.end_ns),
            },
            "ordered_camera_members": members,
            "mapping_semantic_sha256": self.mapping_semantic_sha256,
            "clock_or_alignment_semantic_sha256": self.clock_or_alignment_semantic_sha256,
            "window_policy_version": self.window_policy_version,
        }

    @property
    def planning_sha256(self) -> Sha256Digest:
        return semantic_sha256(self.planning_projection)

    def to_incremental_window(self, schema_ref: SchemaRef) -> IncrementalWindow:
        """Map losslessly to the WP1 segment-or-interval-absence sequence."""

        slots = tuple(self._wire_camera_slot(plan) for plan in self.camera_plans)
        return create_incremental_window(
            schema_ref=schema_ref,
            capture_scope_digest=self.capture_scope_digest,
            purpose=self.purpose,
            requested_interval=self.requested_interval,
            effective_interval=self.effective_interval,
            ordered_six_slot_segment_or_explicit_absence_closure=slots,
            mapping_semantic_sha256=self.mapping_semantic_sha256,
            clock_or_alignment_semantic_sha256=self.clock_or_alignment_semantic_sha256,
            window_policy_version=self.window_policy_version,
        )

    def _wire_camera_slot(
        self,
        plan: CameraWindowPlan,
    ) -> StreamSegmentSequence:
        ordered_members: list[StreamSegmentRef | StreamIntervalAbsence] = []
        for member in plan.members:
            if member.segment is not None:
                ordered_members.append(member.segment.reference())
                continue
            if member.absence_reason is None or member.absence_evidence_sha256 is None:
                raise AssertionError("bounded absence member is incomplete")
            ordered_members.append(
                StreamIntervalAbsence(
                    camera_id=plan.camera_id,
                    capture_scope_digest=self.capture_scope_digest,
                    interval=member.interval,
                    reason=member.absence_reason,
                    evidence_sha256=member.absence_evidence_sha256,
                )
            )
        return StreamSegmentSequence(
            camera_id=plan.camera_id,
            capture_scope_digest=self.capture_scope_digest,
            ordered_members=tuple(ordered_members),
        )


@dataclass(frozen=True, slots=True)
class CameraStreamFacts:
    """Compact source facts retained instead of a full in-memory recording index."""

    camera_id: CameraId
    packet_count: int
    payload_bytes: int
    first_timestamp_ns: int | None
    last_timestamp_ns: int | None
    first_sequence: int | None
    last_sequence: int | None
    sequence_gap_count: int


@dataclass(frozen=True, slots=True)
class PlannerEmission:
    """Result of one source packet append; callers persist the packet fact once."""

    packet: PacketReference
    closed_segments: tuple[SegmentPlan, ...]
    quality_targets: tuple[QualityTarget, ...]
    windows: tuple[BoundedWindowPlan, ...]
    watermark_ns: int | None


@dataclass(frozen=True, slots=True)
class PlannerFinish:
    """End-of-stream output, including partial final windows and facts."""

    closed_segments: tuple[SegmentPlan, ...]
    quality_targets: tuple[QualityTarget, ...]
    windows: tuple[BoundedWindowPlan, ...]
    facts: tuple[CameraStreamFacts, ...]


class SinglePassPacketSink(Protocol):
    """Adapter boundary for index, exact spool, and asynchronous export fan-out.

    One source adapter call supplies the packet bytes and immutable reference.  A
    concrete sink may persist an append-only spool and enqueue export work, but it
    must not reopen or rescan the source to reconstruct the same packet.
    """

    def append_packet(
        self,
        packet: EncodedMediaPacket,
        reference: PacketReference,
        *,
        framing_version: str,
    ) -> None:
        """Durably accept one packet occurrence from the ordered traversal."""


class SinglePassPlanningSink(Protocol):
    """Persistence boundary for deterministic emissions from the pure planner."""

    def append_emission(self, emission: PlannerEmission) -> None:
        """Persist segment, quality-target, and window emissions in source order."""

    def seal(self, finish: PlannerFinish) -> None:
        """Persist EOS facts and the remaining closed segments/windows."""


@dataclass(slots=True)
class _SegmentBuilder:
    camera_id: CameraId
    ordinal: int
    requested_interval: NanosecondInterval
    packets: list[PacketReference] = field(default_factory=list)
    digest: hashlib._Hash = field(default_factory=hashlib.sha256)
    payload_bytes: int = 0

    def append(self, packet: EncodedMediaPacket) -> None:
        self.packets.append(packet.reference())
        framed = packet.framing_bytes()
        self.digest.update(framed)
        self.payload_bytes += packet.payload_bytes


@dataclass(slots=True)
class _QualityState:
    pending_bucket: int | None = None
    pending_packet: PacketReference | None = None


@dataclass(slots=True)
class _CameraState:
    ring: deque[EncodedMediaPacket] = field(default_factory=deque)
    ring_bytes: int = 0
    evicted_packet_count: int = 0
    max_ring_bytes: int = 0
    current_segment: _SegmentBuilder | None = None
    segments: dict[int, SegmentPlan] = field(default_factory=dict)
    quality: _QualityState = field(default_factory=_QualityState)
    facts_count: int = 0
    facts_bytes: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    first_sequence: int | None = None
    last_sequence: int | None = None
    sequence_gap_count: int = 0
    last_source_order: int | None = None
    last_source_sequence: int | None = None
    last_source_timestamp_ns: int | None = None
    last_aligned_timestamp_ns: int | None = None
    quality_targets: list[QualityTarget] = field(default_factory=list)


class BoundedSinglePassMediaPlanner:
    """Plan segments, quality targets, and windows from one ordered packet traversal."""

    def __init__(self, policy: BoundedMediaPolicy) -> None:
        self.policy = policy
        self._states = {camera_id: _CameraState() for camera_id in CAMERA_IDS}
        self._last_traversal_index: int | None = None
        self._max_event_time_ns: int | None = None
        self._next_window_start_ns = policy.source_origin_ns
        self._window_ordinal = 0
        self._finished = False

    def push(self, packet: EncodedMediaPacket) -> PlannerEmission:
        if self._finished:
            raise RuntimeError("planner cannot accept packets after finish")
        if (
            self._last_traversal_index is not None
            and packet.traversal_index <= self._last_traversal_index
        ):
            raise ValueError("source traversal indexes must be strictly increasing")
        if packet.aligned_timestamp_ns < self.policy.source_origin_ns:
            raise ValueError("aligned packet timestamps cannot precede source_origin_ns")
        self._last_traversal_index = packet.traversal_index
        state = self._states[packet.camera_id]
        self._validate_camera_order(state, packet)
        self._append_ring(state, packet)
        self._update_facts(state, packet)
        closed_segments = self._append_segment(state, packet)
        quality_targets = self._update_quality(state, packet)
        self._max_event_time_ns = (
            packet.aligned_timestamp_ns
            if self._max_event_time_ns is None
            else max(self._max_event_time_ns, packet.aligned_timestamp_ns)
        )
        watermark = self._watermark_ns()
        windows = self._emit_watermarked_windows(watermark)
        return PlannerEmission(
            packet=packet.reference(),
            closed_segments=closed_segments,
            quality_targets=quality_targets,
            windows=windows,
            watermark_ns=watermark,
        )

    def finish(self, final_end_ns: int) -> PlannerFinish:
        if self._finished:
            raise RuntimeError("planner has already been finished")
        if isinstance(final_end_ns, bool) or not isinstance(final_end_ns, int):
            raise TypeError("final_end_ns must be an integer")
        if final_end_ns <= self.policy.source_origin_ns:
            raise ValueError("final_end_ns must be after source_origin_ns")
        if self._max_event_time_ns is not None and self._max_event_time_ns >= final_end_ns:
            raise ValueError("final_end_ns must be after every observed packet")
        self._finished = True
        closed_segments: list[SegmentPlan] = []
        for camera_id in CAMERA_IDS:
            state = self._states[camera_id]
            if state.current_segment is not None:
                closed_segments.append(self._seal_segment(state, state.current_segment))
                state.current_segment = None
        quality_targets: list[QualityTarget] = []
        for camera_id in CAMERA_IDS:
            state = self._states[camera_id]
            finalized = self._finalize_quality(state)
            quality_targets.extend(finalized)
        windows = self._emit_eos_windows(final_end_ns)
        facts = tuple(self._facts(camera_id, self._states[camera_id]) for camera_id in CAMERA_IDS)
        return PlannerFinish(
            closed_segments=tuple(
                sorted(
                    closed_segments,
                    key=lambda item: (item.segment_ordinal, item.camera_id.value),
                )
            ),
            quality_targets=tuple(sorted(quality_targets, key=_quality_sort_key)),
            windows=windows,
            facts=facts,
        )

    def ring_snapshot(self, camera_id: CameraId) -> RingSnapshot:
        state = self._states[camera_id]
        return RingSnapshot(
            camera_id=camera_id,
            packets=tuple(state.ring),
            total_bytes=state.ring_bytes,
            evicted_packet_count=state.evicted_packet_count,
        )

    def ring_snapshots(self) -> tuple[RingSnapshot, ...]:
        return tuple(self.ring_snapshot(camera_id) for camera_id in CAMERA_IDS)

    def facts(self) -> tuple[CameraStreamFacts, ...]:
        return tuple(self._facts(camera_id, self._states[camera_id]) for camera_id in CAMERA_IDS)

    def _validate_camera_order(self, state: _CameraState, packet: EncodedMediaPacket) -> None:
        if state.last_source_order is not None and packet.source_order <= state.last_source_order:
            raise ValueError(f"{packet.camera_id.value} source_order must be strictly increasing")
        if (
            state.last_source_sequence is not None
            and packet.source_sequence <= state.last_source_sequence
        ):
            raise ValueError(
                f"{packet.camera_id.value} source_sequence must be strictly increasing"
            )
        if (
            state.last_source_timestamp_ns is not None
            and packet.source_timestamp_ns <= state.last_source_timestamp_ns
        ):
            raise ValueError(
                f"{packet.camera_id.value} source timestamps must be strictly increasing"
            )
        if (
            state.last_aligned_timestamp_ns is not None
            and packet.aligned_timestamp_ns <= state.last_aligned_timestamp_ns
        ):
            raise ValueError(
                f"{packet.camera_id.value} aligned timestamps must be strictly increasing"
            )

    def _append_ring(self, state: _CameraState, packet: EncodedMediaPacket) -> None:
        if packet.payload_bytes > self.policy.ring_max_bytes_per_camera:
            raise ValueError("one encoded packet exceeds the per-camera ring byte bound")
        state.ring.append(packet)
        state.ring_bytes += packet.payload_bytes
        state.max_ring_bytes = max(state.max_ring_bytes, state.ring_bytes)
        cutoff = packet.aligned_timestamp_ns - self.policy.ring_duration_ns
        while state.ring and state.ring[0].aligned_timestamp_ns < cutoff:
            state.ring_bytes -= state.ring.popleft().payload_bytes
            state.evicted_packet_count += 1
        while state.ring and state.ring_bytes > self.policy.ring_max_bytes_per_camera:
            state.ring_bytes -= state.ring.popleft().payload_bytes
            state.evicted_packet_count += 1
        if state.ring_bytes > self.policy.ring_max_bytes_per_camera:
            raise AssertionError("ring byte bound was not preserved")

    @staticmethod
    def _update_facts(state: _CameraState, packet: EncodedMediaPacket) -> None:
        state.facts_count += 1
        state.facts_bytes += packet.payload_bytes
        state.first_timestamp_ns = (
            packet.aligned_timestamp_ns
            if state.first_timestamp_ns is None
            else state.first_timestamp_ns
        )
        state.last_timestamp_ns = packet.aligned_timestamp_ns
        state.first_sequence = (
            packet.source_sequence if state.first_sequence is None else state.first_sequence
        )
        if state.last_sequence is not None and packet.source_sequence > state.last_sequence + 1:
            state.sequence_gap_count += packet.source_sequence - state.last_sequence - 1
        state.last_sequence = packet.source_sequence
        state.last_source_order = packet.source_order
        state.last_source_sequence = packet.source_sequence
        state.last_source_timestamp_ns = packet.source_timestamp_ns
        state.last_aligned_timestamp_ns = packet.aligned_timestamp_ns

    def _append_segment(
        self,
        state: _CameraState,
        packet: EncodedMediaPacket,
    ) -> tuple[SegmentPlan, ...]:
        ordinal = (
            packet.aligned_timestamp_ns - self.policy.source_origin_ns
        ) // self.policy.segment_duration_ns
        requested = NanosecondInterval(
            start_ns=self.policy.source_origin_ns + ordinal * self.policy.segment_duration_ns,
            end_ns=self.policy.source_origin_ns + (ordinal + 1) * self.policy.segment_duration_ns,
        )
        closed: list[SegmentPlan] = []
        current = state.current_segment
        if current is not None and ordinal < current.ordinal:
            raise ValueError("segment ordinals must be nondecreasing per camera")
        if current is None:
            current = _SegmentBuilder(packet.camera_id, ordinal, requested)
            state.current_segment = current
        elif ordinal != current.ordinal:
            closed.append(self._seal_segment(state, current))
            current = _SegmentBuilder(packet.camera_id, ordinal, requested)
            state.current_segment = current
        current.append(packet)
        return tuple(closed)

    def _seal_segment(self, state: _CameraState, builder: _SegmentBuilder) -> SegmentPlan:
        if not builder.packets:
            raise ValueError("cannot seal an empty segment")
        first = builder.packets[0].aligned_timestamp_ns
        last = builder.packets[-1].aligned_timestamp_ns
        effective = NanosecondInterval(
            start_ns=max(builder.requested_interval.start_ns, first),
            end_ns=min(builder.requested_interval.end_ns, max(first + 1, last + 1)),
        )
        segment = SegmentPlan(
            camera_id=builder.camera_id,
            segment_ordinal=builder.ordinal,
            requested_interval=builder.requested_interval,
            effective_interval=effective,
            packets=tuple(builder.packets),
            exact_content_sha256=builder.digest.hexdigest(),
            payload_bytes=builder.payload_bytes,
            capture_scope_digest=self.policy.source_scope_digest,
            mapping_semantic_sha256=self.policy.mapping_semantic_sha256,
            clock_or_alignment_semantic_sha256=self.policy.alignment_semantic_sha256,
            segmentation_policy_version=self.policy.segmentation_policy_version,
        )
        state.segments[segment.segment_ordinal] = segment
        return segment

    def _update_quality(
        self,
        state: _CameraState,
        packet: EncodedMediaPacket,
    ) -> tuple[QualityTarget, ...]:
        bucket = (
            packet.aligned_timestamp_ns - self.policy.source_origin_ns
        ) // self.policy.quality_period_ns
        pending = state.quality.pending_bucket
        if pending is None:
            state.quality.pending_bucket = bucket
            state.quality.pending_packet = packet.reference()
            return ()
        if bucket < pending:
            raise ValueError("quality buckets must be nondecreasing")
        if bucket == pending:
            candidate = state.quality.pending_packet
            assert candidate is not None
            reference = packet.reference()
            if _quality_candidate_key(reference, self.policy, bucket) < _quality_candidate_key(
                candidate, self.policy, bucket
            ):
                state.quality.pending_packet = reference
            return ()
        result = self._finalize_quality(state)
        state.quality.pending_bucket = bucket
        state.quality.pending_packet = packet.reference()
        return result

    def _finalize_quality(self, state: _CameraState) -> tuple[QualityTarget, ...]:
        bucket = state.quality.pending_bucket
        packet = state.quality.pending_packet
        state.quality.pending_bucket = None
        state.quality.pending_packet = None
        if bucket is None or packet is None:
            return ()
        target_ns = (
            self.policy.source_origin_ns
            + bucket * self.policy.quality_period_ns
            + self.policy.quality_target_phase_ns
        )
        delta = abs(packet.aligned_timestamp_ns - target_ns)
        if delta > self.policy.quality_selection_tolerance_ns:
            return ()
        target = QualityTarget(
            camera_id=packet.camera_id,
            bucket_ordinal=bucket,
            requested_target_ns=target_ns,
            packet=packet,
            delta_ns=delta,
            policy_version=self.policy.quality_policy_version,
        )
        state.quality_targets.append(target)
        return (target,)

    def _watermark_ns(self) -> int | None:
        latest = tuple(self._states[camera_id].last_timestamp_ns for camera_id in CAMERA_IDS)
        if any(value is None for value in latest):
            return None
        latest_values = tuple(value for value in latest if value is not None)
        assert len(latest_values) == len(CAMERA_IDS)
        return min(latest_values) - self.policy.allowed_lateness_ns

    def _emit_watermarked_windows(self, watermark_ns: int | None) -> tuple[BoundedWindowPlan, ...]:
        if watermark_ns is None:
            return ()
        windows: list[BoundedWindowPlan] = []
        while self._next_window_start_ns + self.policy.window_width_ns <= watermark_ns:
            windows.append(
                self._build_window(
                    self._next_window_start_ns,
                    self._next_window_start_ns + self.policy.window_width_ns,
                    watermark_ns=watermark_ns,
                    reason=WindowClosureReason.WATERMARK,
                )
            )
            self._next_window_start_ns += self.policy.window_hop_ns
            self._prune_state()
        return tuple(windows)

    def _emit_eos_windows(self, final_end_ns: int) -> tuple[BoundedWindowPlan, ...]:
        windows: list[BoundedWindowPlan] = []
        while self._next_window_start_ns < final_end_ns:
            requested_end = self._next_window_start_ns + self.policy.window_width_ns
            effective_end = min(requested_end, final_end_ns)
            windows.append(
                self._build_window(
                    self._next_window_start_ns,
                    requested_end,
                    watermark_ns=None,
                    reason=WindowClosureReason.EOS,
                    effective_end_ns=effective_end,
                )
            )
            self._next_window_start_ns += self.policy.window_hop_ns
            self._prune_state()
        return tuple(windows)

    def _build_window(
        self,
        start_ns: int,
        requested_end_ns: int,
        *,
        watermark_ns: int | None,
        reason: WindowClosureReason,
        effective_end_ns: int | None = None,
    ) -> BoundedWindowPlan:
        effective_end = requested_end_ns if effective_end_ns is None else effective_end_ns
        requested = NanosecondInterval(start_ns=start_ns, end_ns=requested_end_ns)
        effective = NanosecondInterval(start_ns=start_ns, end_ns=max(start_ns + 1, effective_end))
        camera_plans: list[CameraWindowPlan] = []
        quality_targets: list[QualityTarget] = []
        quality_gaps: list[QualityGap] = []
        bucket_start = (start_ns - self.policy.source_origin_ns) // self.policy.segment_duration_ns
        bucket_end = (
            effective.end_ns - self.policy.source_origin_ns - 1
        ) // self.policy.segment_duration_ns
        for camera_id in CAMERA_IDS:
            state = self._states[camera_id]
            members: list[WindowMember] = []
            for ordinal in range(bucket_start, bucket_end + 1):
                interval = NanosecondInterval(
                    start_ns=(
                        self.policy.source_origin_ns + ordinal * self.policy.segment_duration_ns
                    ),
                    end_ns=min(
                        effective.end_ns,
                        self.policy.source_origin_ns
                        + (ordinal + 1) * self.policy.segment_duration_ns,
                    ),
                )
                segment = state.segments.get(ordinal)
                if segment is not None:
                    members.append(
                        WindowMember(
                            camera_id=camera_id,
                            interval=interval,
                            segment=segment,
                        )
                    )
                else:
                    reason_code = (
                        CameraAbsenceReason.ABSENT
                        if state.facts_count == 0
                        else CameraAbsenceReason.GAP
                    )
                    evidence = semantic_sha256(
                        {
                            "camera_id": camera_id.value,
                            "interval": {
                                "start_ns": str(interval.start_ns),
                                "end_ns": str(interval.end_ns),
                            },
                            "reason": reason_code.value,
                            "source_scope_digest": self.policy.source_scope_digest,
                            "window_policy_version": self.policy.window_policy_version,
                        }
                    )
                    members.append(
                        WindowMember(
                            camera_id=camera_id,
                            interval=interval,
                            absence_reason=reason_code,
                            absence_evidence_sha256=evidence,
                        )
                    )
            camera_plans.append(CameraWindowPlan(camera_id=camera_id, members=tuple(members)))
            quality_bucket_start = (
                start_ns - self.policy.source_origin_ns
            ) // self.policy.quality_period_ns
            quality_bucket_end = (
                effective.end_ns - self.policy.source_origin_ns - 1
            ) // self.policy.quality_period_ns
            selected = {
                target.bucket_ordinal: target
                for target in self._quality_targets_for_state(state)
                if quality_bucket_start <= target.bucket_ordinal <= quality_bucket_end
            }
            for bucket in range(quality_bucket_start, quality_bucket_end + 1):
                target_ns = (
                    self.policy.source_origin_ns
                    + bucket * self.policy.quality_period_ns
                    + self.policy.quality_target_phase_ns
                )
                target = selected.get(bucket)
                if target is not None:
                    quality_targets.append(target)
                else:
                    quality_gaps.append(
                        QualityGap(
                            camera_id=camera_id,
                            bucket_ordinal=bucket,
                            requested_target_ns=target_ns,
                            interval=NanosecondInterval(
                                start_ns=(
                                    self.policy.source_origin_ns
                                    + bucket * self.policy.quality_period_ns
                                ),
                                end_ns=min(
                                    effective.end_ns,
                                    self.policy.source_origin_ns
                                    + (bucket + 1) * self.policy.quality_period_ns,
                                ),
                            ),
                            reason="NO_DECODABLE_TARGET_WITHIN_TOLERANCE",
                            policy_version=self.policy.quality_policy_version,
                        )
                    )
        plan = BoundedWindowPlan(
            ordinal=self._window_ordinal,
            requested_interval=requested,
            effective_interval=effective,
            camera_plans=tuple(camera_plans),
            quality_targets=tuple(sorted(quality_targets, key=_quality_sort_key)),
            quality_gaps=tuple(sorted(quality_gaps, key=_quality_gap_sort_key)),
            watermark_ns=watermark_ns,
            closure_reason=reason,
            capture_scope_digest=self.policy.source_scope_digest,
            mapping_semantic_sha256=self.policy.mapping_semantic_sha256,
            clock_or_alignment_semantic_sha256=self.policy.alignment_semantic_sha256,
            window_policy_version=self.policy.window_policy_version,
            quality_policy_version=self.policy.quality_policy_version,
            purpose=self.policy.window_purpose,
        )
        self._window_ordinal += 1
        return plan

    @staticmethod
    def _quality_targets_for_state(state: _CameraState) -> tuple[QualityTarget, ...]:
        return tuple(state.quality_targets)

    def _prune_state(self) -> None:
        cutoff = self._next_window_start_ns
        for state in self._states.values():
            for ordinal, segment in tuple(state.segments.items()):
                if segment.requested_interval.end_ns <= cutoff:
                    del state.segments[ordinal]
            targets = [
                target
                for target in self._quality_targets_for_state(state)
                if target.requested_target_ns + self.policy.quality_period_ns > cutoff
            ]
            state.quality_targets = targets

    @staticmethod
    def _facts(camera_id: CameraId, state: _CameraState) -> CameraStreamFacts:
        return CameraStreamFacts(
            camera_id=camera_id,
            packet_count=state.facts_count,
            payload_bytes=state.facts_bytes,
            first_timestamp_ns=state.first_timestamp_ns,
            last_timestamp_ns=state.last_timestamp_ns,
            first_sequence=state.first_sequence,
            last_sequence=state.last_sequence,
            sequence_gap_count=state.sequence_gap_count,
        )


def _quality_candidate_key(
    packet: PacketReference,
    policy: BoundedMediaPolicy,
    bucket: int,
) -> tuple[int, int, int]:
    target = (
        policy.source_origin_ns + bucket * policy.quality_period_ns + policy.quality_target_phase_ns
    )
    return abs(packet.aligned_timestamp_ns - target), packet.source_order, packet.traversal_index


def _quality_sort_key(target: QualityTarget) -> tuple[str, int, int, int]:
    return (
        target.camera_id.value,
        target.bucket_ordinal,
        target.requested_target_ns,
        target.packet.source_order,
    )


def _quality_gap_sort_key(gap: QualityGap) -> tuple[str, int]:
    return gap.camera_id.value, gap.bucket_ordinal


__all__ = [
    "ACCESS_UNIT_FRAMING_VERSION",
    "SEGMENT_IDENTITY_POLICY_VERSION",
    "SEGMENT_KEY_NAMESPACE",
    "SEGMENT_SEMANTIC_PROJECTION_VERSION",
    "WINDOW_PLANNING_PROJECTION_VERSION",
    "BoundedMediaPolicy",
    "BoundedSinglePassMediaPlanner",
    "BoundedWindowPlan",
    "CameraStreamFacts",
    "CameraWindowPlan",
    "EncodedMediaPacket",
    "PacketReference",
    "PlannerEmission",
    "PlannerFinish",
    "QualityGap",
    "QualityTarget",
    "RingSnapshot",
    "SegmentPlan",
    "SinglePassPacketSink",
    "SinglePassPlanningSink",
    "WindowClosureReason",
    "WindowMember",
]
