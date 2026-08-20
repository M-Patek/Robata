"""Deterministic Mage stream planning and codec stream-copy materialization.

Storage scan segments are a non-overlapping partition in absolute nanoseconds.
Reasoning horizons are separate causal look-backs that may contain several full
storage segments. Materialization is explicit ffmpeg stream copy only: this
module never decodes frames or silently transcodes media.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Final

from robata.contracts.cameras import CAMERA_IDS, CameraId, SixCameraMap
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.perception_stream import (
    CameraAbsenceReason,
    CameraContextBinding,
    PerceptionContextManifest,
    StorageSegmentReference,
    create_perception_context_manifest,
)

MAGE_STREAM_POLICY_VERSION: Final = "mage-stream-planner-v1"
MAGE_STREAM_SEGMENT_KEY_NAMESPACE: Final = "mage-stream-storage-segment-v1"
MAGE_STREAM_CONTEXT_KEY_NAMESPACE: Final = "mage-stream-reasoning-context-v1"
MAGE_STREAM_PLAN_KEY_NAMESPACE: Final = "mage-stream-plan-v1"
DEFAULT_SCAN_SEGMENT_DURATION_NS: Final = 8_000_000_000
DEFAULT_REASONING_HORIZON_DURATION_NS: Final = 8_000_000_000
DEFAULT_KEYFRAME_ALIGNMENT_TOLERANCE_NS: Final = 50_000_000
NANOSECONDS_PER_SECOND: Final = 1_000_000_000
MATERIALIZATION_END_EXCLUSION_NS: Final = 1_000
MAGE_STREAM_MATERIALIZATION_RECEIPT_VERSION: Final = "mage-stream-materialization-receipt-v1"
_MATERIALIZATION_RECEIPT_SUFFIX: Final = ".receipt.json"
_SHA256_HEX_LENGTH: Final = 64


class MageStreamPlanningError(ValueError):
    """A stream plan is invalid, overlapping, gapped, or non-deterministic."""


class MageStreamMaterializationError(RuntimeError):
    """An explicitly requested ffmpeg stream-copy operation could not complete."""


@dataclass(frozen=True, slots=True)
class AbsoluteNanosecondInterval:
    """A nonempty half-open absolute interval."""

    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.start_ns, "start_ns")
        _require_nonnegative_int(self.end_ns, "end_ns")
        if self.start_ns >= self.end_ns:
            raise MageStreamPlanningError("absolute interval must be nonempty")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    def contains(self, other: AbsoluteNanosecondInterval) -> bool:
        return self.start_ns <= other.start_ns and other.end_ns <= self.end_ns

    def as_projection(self) -> dict[str, int]:
        return {"start_ns": self.start_ns, "end_ns": self.end_ns}


@dataclass(frozen=True, slots=True)
class MageStreamRecording:
    """Recording identity plus its source-clock interval."""

    recording_key: str
    recording_exact_sha256: str
    interval: AbsoluteNanosecondInterval

    def __post_init__(self) -> None:
        _require_nonempty_text(self.recording_key, "recording_key")
        _require_sha256(self.recording_exact_sha256, "recording_exact_sha256")
        if not isinstance(self.interval, AbsoluteNanosecondInterval):
            raise MageStreamPlanningError("interval must be AbsoluteNanosecondInterval")


class MageStreamSegmentationMode(StrEnum):
    """How immutable storage boundaries are selected before codec stream copy."""

    FIXED_DURATION = "FIXED_DURATION"
    KEYFRAME_ALIGNED = "KEYFRAME_ALIGNED"


@dataclass(frozen=True, slots=True)
class MageStreamPolicy:
    """Independent storage segmentation and causal reasoning policy."""

    scan_segment_duration_ns: int = DEFAULT_SCAN_SEGMENT_DURATION_NS
    reasoning_horizon_duration_ns: int = DEFAULT_REASONING_HORIZON_DURATION_NS
    policy_version: str = MAGE_STREAM_POLICY_VERSION
    segmentation_mode: MageStreamSegmentationMode = MageStreamSegmentationMode.FIXED_DURATION
    keyframe_alignment_tolerance_ns: int = 0

    def __post_init__(self) -> None:
        _require_positive_int(self.scan_segment_duration_ns, "scan_segment_duration_ns")
        _require_positive_int(self.reasoning_horizon_duration_ns, "reasoning_horizon_duration_ns")
        _require_nonempty_text(self.policy_version, "policy_version")
        if not isinstance(self.segmentation_mode, MageStreamSegmentationMode):
            raise MageStreamPlanningError("segmentation_mode must be MageStreamSegmentationMode")
        _require_nonnegative_int(
            self.keyframe_alignment_tolerance_ns, "keyframe_alignment_tolerance_ns"
        )
        if (
            self.segmentation_mode is MageStreamSegmentationMode.FIXED_DURATION
            and self.keyframe_alignment_tolerance_ns != 0
        ):
            raise MageStreamPlanningError(
                "fixed-duration segmentation cannot declare keyframe alignment tolerance"
            )
        if (
            self.segmentation_mode is MageStreamSegmentationMode.KEYFRAME_ALIGNED
            and self.keyframe_alignment_tolerance_ns == 0
        ):
            raise MageStreamPlanningError(
                "keyframe-aligned segmentation requires a positive alignment tolerance"
            )


@dataclass(frozen=True, slots=True)
class MageStorageSegment:
    """One immutable scan/storage segment in the source time domain."""

    ordinal: int
    recording_key: str
    recording_exact_sha256: str
    interval: AbsoluteNanosecondInterval
    segment_policy_version: str
    segment_semantic_sha256: str
    segment_key: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.ordinal, "ordinal")
        _require_nonempty_text(self.recording_key, "recording_key")
        _require_sha256(self.recording_exact_sha256, "recording_exact_sha256")
        if not isinstance(self.interval, AbsoluteNanosecondInterval):
            raise MageStreamPlanningError("segment interval must be AbsoluteNanosecondInterval")
        _require_nonempty_text(self.segment_policy_version, "segment_policy_version")
        _require_sha256(self.segment_semantic_sha256, "segment_semantic_sha256")
        expected_digest = semantic_sha256(self.semantic_projection())
        if self.segment_semantic_sha256 != expected_digest:
            raise MageStreamPlanningError("segment semantic SHA-256 does not match its projection")
        if self.segment_key != f"{MAGE_STREAM_SEGMENT_KEY_NAMESPACE}:{expected_digest}":
            raise MageStreamPlanningError("segment key does not bind its semantic SHA-256")

    def semantic_projection(self) -> dict[str, object]:
        return {
            "segment_policy_version": self.segment_policy_version,
            "recording_key": self.recording_key,
            "recording_exact_sha256": self.recording_exact_sha256,
            "ordinal": self.ordinal,
            "interval": self.interval.as_projection(),
        }


@dataclass(frozen=True, slots=True)
class MageReasoningContext:
    """A causal full-segment materialization around one focus segment."""

    focus_segment_ordinal: int
    reasoning_horizon: AbsoluteNanosecondInterval
    materialized_interval: AbsoluteNanosecondInterval
    ordered_segments: tuple[MageStorageSegment, ...]
    context_policy_version: str
    context_semantic_sha256: str
    context_key: str

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.focus_segment_ordinal, "focus_segment_ordinal")
        if not isinstance(self.reasoning_horizon, AbsoluteNanosecondInterval):
            raise MageStreamPlanningError("reasoning_horizon must be AbsoluteNanosecondInterval")
        if not isinstance(self.materialized_interval, AbsoluteNanosecondInterval):
            raise MageStreamPlanningError(
                "materialized_interval must be AbsoluteNanosecondInterval"
            )
        if not self.ordered_segments:
            raise MageStreamPlanningError("reasoning context requires at least one storage segment")
        _require_nonempty_text(self.context_policy_version, "context_policy_version")
        _require_sha256(self.context_semantic_sha256, "context_semantic_sha256")
        _validate_contiguous_segments(self.ordered_segments)
        focus = self.ordered_segments[-1]
        if focus.ordinal != self.focus_segment_ordinal:
            raise MageStreamPlanningError("focus segment must be the newest context segment")
        expected_materialized = AbsoluteNanosecondInterval(
            self.ordered_segments[0].interval.start_ns,
            focus.interval.end_ns,
        )
        if self.materialized_interval != expected_materialized:
            raise MageStreamPlanningError(
                "context materialization must use complete storage segments"
            )
        if self.reasoning_horizon.end_ns != focus.interval.end_ns:
            raise MageStreamPlanningError("reasoning horizon must end with focused segment")
        if not self.materialized_interval.contains(self.reasoning_horizon):
            raise MageStreamPlanningError("materialized context must contain reasoning horizon")
        expected_digest = semantic_sha256(self.semantic_projection())
        if self.context_semantic_sha256 != expected_digest:
            raise MageStreamPlanningError("context semantic SHA-256 does not match its projection")
        if self.context_key != f"{MAGE_STREAM_CONTEXT_KEY_NAMESPACE}:{expected_digest}":
            raise MageStreamPlanningError("context key does not bind its semantic SHA-256")

    @property
    def context_interval(self) -> AbsoluteNanosecondInterval:
        """The discrete complete-segment interval sent to native codec input."""

        return self.materialized_interval

    def semantic_projection(self) -> dict[str, object]:
        focus = self.ordered_segments[-1]
        return {
            "context_policy_version": self.context_policy_version,
            "recording_key": focus.recording_key,
            "recording_exact_sha256": focus.recording_exact_sha256,
            "focus_segment_ordinal": self.focus_segment_ordinal,
            "reasoning_horizon": self.reasoning_horizon.as_projection(),
            "materialized_interval": self.materialized_interval.as_projection(),
            "ordered_segment_semantic_sha256_values": [
                item.segment_semantic_sha256 for item in self.ordered_segments
            ],
        }


@dataclass(frozen=True, slots=True)
class MageStreamPlan:
    """A deterministic storage partition and one causal context per segment."""

    recording: MageStreamRecording
    policy: MageStreamPolicy
    storage_segments: tuple[MageStorageSegment, ...]
    reasoning_contexts: tuple[MageReasoningContext, ...]
    plan_semantic_sha256: str
    plan_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.recording, MageStreamRecording):
            raise MageStreamPlanningError("recording must be MageStreamRecording")
        if not isinstance(self.policy, MageStreamPolicy):
            raise MageStreamPlanningError("policy must be MageStreamPolicy")
        _validate_storage_segment_partition(
            self.recording, self.storage_segments, policy=self.policy
        )
        if len(self.reasoning_contexts) != len(self.storage_segments):
            raise MageStreamPlanningError("each storage segment must have one reasoning context")
        for segment, context in zip(self.storage_segments, self.reasoning_contexts, strict=True):
            if context.focus_segment_ordinal != segment.ordinal:
                raise MageStreamPlanningError("contexts must be ordered by focus segment ordinal")
            if context.ordered_segments[-1] != segment:
                raise MageStreamPlanningError("context focus must bind the planned storage segment")
            if context.context_policy_version != self.policy.policy_version:
                raise MageStreamPlanningError("context policy must match the stream policy")
        _require_sha256(self.plan_semantic_sha256, "plan_semantic_sha256")
        expected_digest = semantic_sha256(self.semantic_projection())
        if self.plan_semantic_sha256 != expected_digest:
            raise MageStreamPlanningError("plan semantic SHA-256 does not match its projection")
        if self.plan_key != f"{MAGE_STREAM_PLAN_KEY_NAMESPACE}:{expected_digest}":
            raise MageStreamPlanningError("plan key does not bind its semantic SHA-256")

    def semantic_projection(self) -> dict[str, object]:
        return {
            "recording_key": self.recording.recording_key,
            "recording_exact_sha256": self.recording.recording_exact_sha256,
            "recording_interval": self.recording.interval.as_projection(),
            "policy_version": self.policy.policy_version,
            "segmentation_mode": self.policy.segmentation_mode.value,
            "keyframe_alignment_tolerance_ns": (self.policy.keyframe_alignment_tolerance_ns),
            "scan_segment_duration_ns": self.policy.scan_segment_duration_ns,
            "reasoning_horizon_duration_ns": self.policy.reasoning_horizon_duration_ns,
            "storage_segment_semantic_sha256_values": [
                item.segment_semantic_sha256 for item in self.storage_segments
            ],
            "reasoning_context_semantic_sha256_values": [
                item.context_semantic_sha256 for item in self.reasoning_contexts
            ],
        }


def plan_mage_stream(
    *,
    recording: MageStreamRecording,
    policy: MageStreamPolicy | None = None,
) -> MageStreamPlan:
    """Create exact fixed-duration non-overlapping segments plus causal contexts."""

    if not isinstance(recording, MageStreamRecording):
        raise MageStreamPlanningError("recording must be MageStreamRecording")
    effective_policy = policy or MageStreamPolicy()
    if not isinstance(effective_policy, MageStreamPolicy):
        raise MageStreamPlanningError("policy must be MageStreamPolicy or None")
    if effective_policy.segmentation_mode is not MageStreamSegmentationMode.FIXED_DURATION:
        raise MageStreamPlanningError(
            "plan_mage_stream requires FIXED_DURATION; use "
            "plan_keyframe_aligned_mage_stream for native codec execution"
        )
    boundaries = [recording.interval.start_ns]
    cursor = recording.interval.start_ns
    while cursor < recording.interval.end_ns:
        cursor = min(
            cursor + effective_policy.scan_segment_duration_ns,
            recording.interval.end_ns,
        )
        boundaries.append(cursor)
    return _build_mage_stream_plan(
        recording=recording,
        policy=effective_policy,
        boundary_ns_values=tuple(boundaries),
    )


def plan_keyframe_aligned_mage_stream(
    *,
    recording: MageStreamRecording,
    policy: MageStreamPolicy,
    keyframe_offsets_ns: Sequence[int],
) -> MageStreamPlan:
    """Align nominal scan boundaries to exact source keyframe PTS values.

    ``keyframe_offsets_ns`` are relative to the recording start. The first keyframe
    must be exactly zero; the recording end is an EOF boundary and need not be a
    keyframe. Every interior boundary must remain within the declared tolerance of
    its nominal fixed-duration target.
    """

    if not isinstance(recording, MageStreamRecording):
        raise MageStreamPlanningError("recording must be MageStreamRecording")
    if not isinstance(policy, MageStreamPolicy):
        raise MageStreamPlanningError("policy must be MageStreamPolicy")
    if policy.segmentation_mode is not MageStreamSegmentationMode.KEYFRAME_ALIGNED:
        raise MageStreamPlanningError(
            "keyframe-aligned planning requires KEYFRAME_ALIGNED segmentation mode"
        )
    offsets = tuple(keyframe_offsets_ns)
    if not offsets:
        raise MageStreamPlanningError("keyframe-aligned planning requires keyframe offsets")
    for value in offsets:
        _require_nonnegative_int(value, "keyframe offset")
    if offsets != tuple(sorted(set(offsets))):
        raise MageStreamPlanningError("keyframe offsets must be unique and ordered")
    if offsets[0] != 0:
        raise MageStreamPlanningError("native codec recording must begin on keyframe offset zero")

    recording_duration_ns = recording.interval.duration_ns
    boundaries = [recording.interval.start_ns]
    nominal_offset = policy.scan_segment_duration_ns
    previous_offset = 0
    while nominal_offset < recording_duration_ns:
        candidates = tuple(
            value
            for value in offsets
            if previous_offset < value < recording_duration_ns
            and abs(value - nominal_offset) <= policy.keyframe_alignment_tolerance_ns
        )
        if not candidates:
            # A recording may end on a keyframe that is within tolerance of the
            # nominal boundary. That keyframe is the EOF boundary, not an
            # interior split: accept the declared recording end and stop.
            eof_is_aligned = any(
                value == recording_duration_ns
                and previous_offset < value
                and abs(value - nominal_offset) <= policy.keyframe_alignment_tolerance_ns
                for value in offsets
            )
            if eof_is_aligned:
                break
            raise MageStreamPlanningError(
                "no source keyframe is within the declared alignment tolerance of "
                f"nominal boundary {nominal_offset}"
            )
        selected = min(candidates, key=lambda value: (abs(value - nominal_offset), value))
        boundaries.append(recording.interval.start_ns + selected)
        previous_offset = selected
        nominal_offset += policy.scan_segment_duration_ns
    boundaries.append(recording.interval.end_ns)
    return _build_mage_stream_plan(
        recording=recording,
        policy=policy,
        boundary_ns_values=tuple(boundaries),
    )


def _build_mage_stream_plan(
    *,
    recording: MageStreamRecording,
    policy: MageStreamPolicy,
    boundary_ns_values: tuple[int, ...],
) -> MageStreamPlan:
    if len(boundary_ns_values) < 2:
        raise MageStreamPlanningError("stream plan requires at least two boundaries")
    if boundary_ns_values != tuple(sorted(set(boundary_ns_values))):
        raise MageStreamPlanningError("stream plan boundaries must be unique and ordered")
    if (
        boundary_ns_values[0] != recording.interval.start_ns
        or boundary_ns_values[-1] != recording.interval.end_ns
    ):
        raise MageStreamPlanningError("stream plan boundaries must cover the recording")

    segments: list[MageStorageSegment] = []
    for ordinal, (start_ns, end_ns) in enumerate(pairwise(boundary_ns_values)):
        interval = AbsoluteNanosecondInterval(start_ns, end_ns)
        projection = {
            "segment_policy_version": policy.policy_version,
            "recording_key": recording.recording_key,
            "recording_exact_sha256": recording.recording_exact_sha256,
            "ordinal": ordinal,
            "interval": interval.as_projection(),
        }
        digest = semantic_sha256(projection)
        segments.append(
            MageStorageSegment(
                ordinal=ordinal,
                recording_key=recording.recording_key,
                recording_exact_sha256=recording.recording_exact_sha256,
                interval=interval,
                segment_policy_version=policy.policy_version,
                segment_semantic_sha256=digest,
                segment_key=f"{MAGE_STREAM_SEGMENT_KEY_NAMESPACE}:{digest}",
            )
        )

    storage_segments = tuple(segments)
    _validate_storage_segment_partition(recording, storage_segments, policy=policy)
    contexts: list[MageReasoningContext] = []
    for focus_index, focus in enumerate(storage_segments):
        selected: tuple[MageStorageSegment, ...]
        if policy.reasoning_horizon_duration_ns == policy.scan_segment_duration_ns:
            requested_start_ns = focus.interval.start_ns
            selected = (focus,)
        else:
            requested_start_ns = max(
                recording.interval.start_ns,
                focus.interval.end_ns - policy.reasoning_horizon_duration_ns,
            )
            selected = tuple(
                segment
                for segment in storage_segments[: focus_index + 1]
                if segment.interval.end_ns > requested_start_ns
            )
        if not selected:
            raise MageStreamPlanningError("causal context selection produced no segments")
        materialized_interval = AbsoluteNanosecondInterval(
            selected[0].interval.start_ns,
            focus.interval.end_ns,
        )
        reasoning_horizon = AbsoluteNanosecondInterval(requested_start_ns, focus.interval.end_ns)
        projection = {
            "context_policy_version": policy.policy_version,
            "recording_key": recording.recording_key,
            "recording_exact_sha256": recording.recording_exact_sha256,
            "focus_segment_ordinal": focus.ordinal,
            "reasoning_horizon": reasoning_horizon.as_projection(),
            "materialized_interval": materialized_interval.as_projection(),
            "ordered_segment_semantic_sha256_values": [
                item.segment_semantic_sha256 for item in selected
            ],
        }
        digest = semantic_sha256(projection)
        contexts.append(
            MageReasoningContext(
                focus_segment_ordinal=focus.ordinal,
                reasoning_horizon=reasoning_horizon,
                materialized_interval=materialized_interval,
                ordered_segments=selected,
                context_policy_version=policy.policy_version,
                context_semantic_sha256=digest,
                context_key=f"{MAGE_STREAM_CONTEXT_KEY_NAMESPACE}:{digest}",
            )
        )

    context_values = tuple(contexts)
    plan_projection = {
        "recording_key": recording.recording_key,
        "recording_exact_sha256": recording.recording_exact_sha256,
        "recording_interval": recording.interval.as_projection(),
        "policy_version": policy.policy_version,
        "segmentation_mode": policy.segmentation_mode.value,
        "keyframe_alignment_tolerance_ns": policy.keyframe_alignment_tolerance_ns,
        "scan_segment_duration_ns": policy.scan_segment_duration_ns,
        "reasoning_horizon_duration_ns": policy.reasoning_horizon_duration_ns,
        "storage_segment_semantic_sha256_values": [
            item.segment_semantic_sha256 for item in storage_segments
        ],
        "reasoning_context_semantic_sha256_values": [
            item.context_semantic_sha256 for item in context_values
        ],
    }
    plan_digest = semantic_sha256(plan_projection)
    return MageStreamPlan(
        recording=recording,
        policy=policy,
        storage_segments=storage_segments,
        reasoning_contexts=context_values,
        plan_semantic_sha256=plan_digest,
        plan_key=f"{MAGE_STREAM_PLAN_KEY_NAMESPACE}:{plan_digest}",
    )


@dataclass(frozen=True, slots=True)
class FfmpegCommandResult:
    """The minimal result supplied by an injected ffmpeg command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise TypeError("returncode must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("stdout and stderr must be strings")


FfmpegCommandRunner = Callable[[tuple[str, ...]], FfmpegCommandResult]
FfprobeCommandRunner = Callable[[tuple[str, ...]], FfmpegCommandResult]


def probe_video_keyframe_offsets_ns(
    source_path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
    command_runner: FfprobeCommandRunner | None = None,
) -> tuple[int, ...]:
    """Return ordered relative video keyframe PTS values in integer nanoseconds."""

    source = _require_existing_file(source_path, "source_path")
    _require_nonempty_text(ffprobe_binary, "ffprobe_binary")
    runner = command_runner or _subprocess_ffmpeg_runner
    command = (
        ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=pts_time,flags",
        "-of",
        "json",
        str(source),
    )
    try:
        result = runner(command)
    except FileNotFoundError as error:
        raise MageStreamMaterializationError(
            f"ffprobe keyframe inspection is unavailable: {ffprobe_binary!r}"
        ) from error
    except OSError as error:
        raise MageStreamMaterializationError(
            "ffprobe keyframe inspection could not start"
        ) from error
    if not isinstance(result, FfmpegCommandResult):
        raise MageStreamMaterializationError(
            "injected ffprobe command runner must return FfmpegCommandResult"
        )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no ffprobe diagnostic"
        raise MageStreamMaterializationError(
            f"ffprobe keyframe inspection failed with exit code {result.returncode}: {detail}"
        )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise MageStreamMaterializationError(
            "ffprobe keyframe inspection returned invalid JSON"
        ) from error
    packets = document.get("packets") if isinstance(document, dict) else None
    if not isinstance(packets, list):
        raise MageStreamMaterializationError("ffprobe keyframe inspection returned no packets")
    offsets: list[int] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        flags = packet.get("flags")
        pts_time = packet.get("pts_time")
        if not isinstance(flags, str) or "K" not in flags or not isinstance(pts_time, str):
            continue
        try:
            pts_ns = int(
                (Decimal(pts_time) * Decimal(NANOSECONDS_PER_SECOND)).to_integral_value(
                    rounding=ROUND_HALF_UP
                )
            )
        except (ArithmeticError, ValueError) as error:
            raise MageStreamMaterializationError(
                "ffprobe keyframe inspection returned an invalid PTS"
            ) from error
        if pts_ns >= 0:
            offsets.append(pts_ns)
    values = tuple(sorted(set(offsets)))
    if not values:
        raise MageStreamMaterializationError("video contains no nonnegative keyframe PTS")
    return values


@dataclass(frozen=True, slots=True)
class MageMaterializedStorageSegment:
    """Exact bytes emitted by one declared storage stream-copy operation."""

    segment: MageStorageSegment
    source_path: Path
    durable_path: Path
    source_exact_sha256: str
    content_exact_sha256: str
    byte_count: int
    command: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.segment, MageStorageSegment):
            raise TypeError("segment must be MageStorageSegment")
        _require_sha256(self.source_exact_sha256, "source_exact_sha256")
        if self.source_exact_sha256 != self.segment.recording_exact_sha256:
            raise MageStreamMaterializationError(
                "materialized segment source identity does not match segment lineage"
            )
        _require_sha256(self.content_exact_sha256, "content_exact_sha256")
        _require_positive_int(self.byte_count, "byte_count")
        if self.durable_path.is_symlink() or not self.durable_path.is_file():
            raise MageStreamMaterializationError(
                "materialized segment durable_path must be a regular non-symlink file"
            )


@dataclass(frozen=True, slots=True)
class MageMaterializedReasoningContext:
    """One native video input made only by codec-preserving stream copies."""

    context: MageReasoningContext
    camera_id: CameraId
    durable_path: Path
    content_exact_sha256: str
    byte_count: int
    component_segment_exact_sha256_values: tuple[str, ...]
    command: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.context, MageReasoningContext):
            raise TypeError("context must be MageReasoningContext")
        if not isinstance(self.camera_id, CameraId):
            raise TypeError("camera_id must be CameraId")
        _require_sha256(self.content_exact_sha256, "content_exact_sha256")
        _require_positive_int(self.byte_count, "byte_count")
        if not self.component_segment_exact_sha256_values:
            raise MageStreamMaterializationError("materialized context requires component segments")
        for digest in self.component_segment_exact_sha256_values:
            _require_sha256(digest, "component_segment_exact_sha256")
        if self.durable_path.is_symlink() or not self.durable_path.is_file():
            raise MageStreamMaterializationError(
                "materialized context durable_path must be a regular non-symlink file"
            )


class MageStreamMaterializer:
    """Optional ffmpeg-only materializer with an injected command runner."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        command_runner: FfmpegCommandRunner | None = None,
        probe_runner: FfprobeCommandRunner | None = None,
        verify_packet_boundaries: bool = False,
        packet_timestamp_tolerance_ns: int = 1_000_000,
    ) -> None:
        _require_nonempty_text(ffmpeg_binary, "ffmpeg_binary")
        _require_nonempty_text(ffprobe_binary, "ffprobe_binary")
        if not isinstance(verify_packet_boundaries, bool):
            raise TypeError("verify_packet_boundaries must be bool")
        _require_nonnegative_int(packet_timestamp_tolerance_ns, "packet_timestamp_tolerance_ns")
        self._ffmpeg_binary = ffmpeg_binary
        self._ffprobe_binary = ffprobe_binary
        self._command_runner = command_runner or _subprocess_ffmpeg_runner
        self._probe_runner = probe_runner or _subprocess_ffmpeg_runner
        self._verify_packet_boundaries = verify_packet_boundaries
        self._packet_timestamp_tolerance_ns = packet_timestamp_tolerance_ns

    def materialize_storage_segment(
        self,
        *,
        plan: MageStreamPlan,
        source_path: Path,
        segment: MageStorageSegment,
        output_root: Path,
    ) -> MageMaterializedStorageSegment:
        """Stream-copy one declared storage segment for a backpressured consumer.

        This is intentionally separate from context concatenation.  A local
        streaming runner may materialize exactly the focus segment, observe it,
        and only then materialize the next focus segment.  Existing bytes are
        verified and reused, never silently replaced.
        """

        validate_mage_stream_plan(plan)
        if segment not in plan.storage_segments:
            raise MageStreamMaterializationError("segment does not belong to supplied plan")
        source = _require_existing_file(source_path, "source_path")
        root = _prepare_output_root(output_root)
        source_digest, _ = exact_file_sha256(source)
        if source_digest != plan.recording.recording_exact_sha256:
            raise MageStreamMaterializationError(
                "source file SHA-256 does not match the recording identity used for planning"
            )
        extension = source.suffix or ".mp4"
        segment_root = root / "segments"
        segment_root.mkdir(parents=True, exist_ok=True)
        destination = segment_root / (
            f"{segment.ordinal:06d}-{segment.segment_semantic_sha256}{extension}"
        )
        receipt_path = _materialization_receipt_path(destination)
        _reject_symlink_path(destination, "storage segment output")
        _reject_symlink_path(receipt_path, "storage segment receipt")
        command: tuple[str, ...] | None = None
        if destination.exists():
            content_digest, byte_count = _validate_storage_materialization_receipt(
                destination=destination,
                receipt_path=receipt_path,
                segment=segment,
                source_exact_sha256=source_digest,
            )
        else:
            if receipt_path.exists():
                raise MageStreamMaterializationError(
                    "storage segment receipt exists without its materialized output"
                )
            command = self._storage_segment_command(
                source_path=source,
                recording_interval=plan.recording.interval,
                segment=segment,
                destination=destination,
            )
            self._run_checked(command)
            content_digest, byte_count = _require_nonempty_file_hash(
                destination,
                "ffmpeg did not produce a nonempty storage segment",
            )
        if self._verify_packet_boundaries:
            self._verify_materialized_storage_segment(
                plan=plan,
                segment=segment,
                destination=destination,
            )
        if command is not None:
            _write_materialization_receipt(
                receipt_path,
                _storage_materialization_receipt(
                    segment=segment,
                    source_exact_sha256=source_digest,
                    content_exact_sha256=content_digest,
                    byte_count=byte_count,
                ),
            )
        return MageMaterializedStorageSegment(
            segment=segment,
            source_path=source,
            durable_path=destination.resolve(),
            source_exact_sha256=source_digest,
            content_exact_sha256=content_digest,
            byte_count=byte_count,
            command=command,
        )

    def materialize_storage_segments(
        self,
        *,
        plan: MageStreamPlan,
        source_path: Path,
        output_root: Path,
    ) -> tuple[MageMaterializedStorageSegment, ...]:
        """Stream-copy every planned segment and exact-hash the resulting bytes."""

        return tuple(
            self.materialize_storage_segment(
                plan=plan,
                source_path=source_path,
                segment=segment,
                output_root=output_root,
            )
            for segment in plan.storage_segments
        )

    def materialize_reasoning_context(
        self,
        *,
        context: MageReasoningContext,
        camera_id: CameraId,
        storage_segments: Sequence[MageMaterializedStorageSegment],
        output_root: Path,
    ) -> MageMaterializedReasoningContext:
        """Stream-copy concatenate a context, or reuse its one storage segment."""

        if not isinstance(context, MageReasoningContext):
            raise MageStreamMaterializationError("context must be MageReasoningContext")
        if not isinstance(camera_id, CameraId):
            raise MageStreamMaterializationError("camera_id must be CameraId")
        selected = _select_materialized_segments(context, storage_segments)
        if len(selected) == 1:
            item = selected[0]
            return MageMaterializedReasoningContext(
                context=context,
                camera_id=camera_id,
                durable_path=item.durable_path,
                content_exact_sha256=item.content_exact_sha256,
                byte_count=item.byte_count,
                component_segment_exact_sha256_values=(item.content_exact_sha256,),
                command=None,
            )
        root = _prepare_output_root(output_root)
        context_root = root / "contexts"
        context_root.mkdir(parents=True, exist_ok=True)
        extension = selected[0].durable_path.suffix or ".mp4"
        destination = context_root / (
            f"{context.focus_segment_ordinal:06d}-{context.context_semantic_sha256}{extension}"
        )
        component_hashes = tuple(item.content_exact_sha256 for item in selected)
        receipt_path = _materialization_receipt_path(destination)
        _reject_symlink_path(destination, "reasoning context output")
        _reject_symlink_path(receipt_path, "reasoning context receipt")
        command: tuple[str, ...] | None = None
        if destination.exists():
            content_digest, byte_count = _validate_context_materialization_receipt(
                destination=destination,
                receipt_path=receipt_path,
                context=context,
                camera_id=camera_id,
                component_hashes=component_hashes,
            )
        else:
            if receipt_path.exists():
                raise MageStreamMaterializationError(
                    "reasoning context receipt exists without its materialized output"
                )
            concat_manifest = context_root / f".{context.context_semantic_sha256}.ffconcat"
            try:
                concat_manifest.write_text(
                    "".join(_ffconcat_file_line(item.durable_path) for item in selected),
                    encoding="utf-8",
                    newline="\n",
                )
                command = self._concat_context_command(
                    concat_manifest=concat_manifest,
                    destination=destination,
                )
                self._run_checked(command)
            finally:
                try:
                    concat_manifest.unlink(missing_ok=True)
                except OSError as error:
                    raise MageStreamMaterializationError(
                        "could not remove temporary ffmpeg concat manifest"
                    ) from error
            content_digest, byte_count = _require_nonempty_file_hash(
                destination,
                "ffmpeg did not produce a nonempty reasoning context",
            )
            _write_materialization_receipt(
                receipt_path,
                _context_materialization_receipt(
                    context=context,
                    camera_id=camera_id,
                    component_hashes=component_hashes,
                    content_exact_sha256=content_digest,
                    byte_count=byte_count,
                ),
            )
        return MageMaterializedReasoningContext(
            context=context,
            camera_id=camera_id,
            durable_path=destination.resolve(),
            content_exact_sha256=content_digest,
            byte_count=byte_count,
            component_segment_exact_sha256_values=component_hashes,
            command=command,
        )

    def _verify_materialized_storage_segment(
        self,
        *,
        plan: MageStreamPlan,
        segment: MageStorageSegment,
        destination: Path,
    ) -> None:
        if plan.policy.segmentation_mode is not MageStreamSegmentationMode.KEYFRAME_ALIGNED:
            raise MageStreamMaterializationError(
                "packet boundary verification requires a keyframe-aligned stream plan"
            )
        command = (
            self._ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,dts_time,flags",
            "-of",
            "json",
            str(destination),
        )
        try:
            result = self._probe_runner(command)
        except FileNotFoundError as error:
            raise MageStreamMaterializationError(
                f"ffprobe packet verification is unavailable: {self._ffprobe_binary!r}"
            ) from error
        except OSError as error:
            raise MageStreamMaterializationError(
                "ffprobe packet verification could not start"
            ) from error
        if not isinstance(result, FfmpegCommandResult):
            raise MageStreamMaterializationError(
                "injected ffprobe command runner must return FfmpegCommandResult"
            )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no ffprobe diagnostic"
            raise MageStreamMaterializationError(
                f"ffprobe packet verification failed with exit code {result.returncode}: {detail}"
            )
        try:
            document = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise MageStreamMaterializationError(
                "ffprobe packet verification returned invalid JSON"
            ) from error
        raw_packets = document.get("packets") if isinstance(document, dict) else None
        if not isinstance(raw_packets, list) or not raw_packets:
            raise MageStreamMaterializationError(
                "materialized storage segment contains no video packets"
            )
        packets: list[tuple[int, int, str]] = []
        for packet in raw_packets:
            if not isinstance(packet, dict):
                raise MageStreamMaterializationError(
                    "ffprobe packet verification returned an invalid packet"
                )
            pts = packet.get("pts_time")
            dts = packet.get("dts_time")
            flags = packet.get("flags")
            if not isinstance(pts, str) or not isinstance(dts, str) or not isinstance(flags, str):
                raise MageStreamMaterializationError(
                    "ffprobe packet verification omitted PTS, DTS, or flags"
                )
            try:
                pts_ns = int(
                    (Decimal(pts) * Decimal(NANOSECONDS_PER_SECOND)).to_integral_value(
                        rounding=ROUND_HALF_UP
                    )
                )
                dts_ns = int(
                    (Decimal(dts) * Decimal(NANOSECONDS_PER_SECOND)).to_integral_value(
                        rounding=ROUND_HALF_UP
                    )
                )
            except (ArithmeticError, ValueError) as error:
                raise MageStreamMaterializationError(
                    "ffprobe packet verification returned an invalid timestamp"
                ) from error
            packets.append((pts_ns, dts_ns, flags))
        first_pts_ns, first_dts_ns, first_flags = packets[0]
        tolerance_ns = self._packet_timestamp_tolerance_ns
        if "K" not in first_flags:
            raise MageStreamMaterializationError(
                "materialized storage segment does not begin with a video keyframe"
            )
        if first_dts_ns < -tolerance_ns:
            raise MageStreamMaterializationError(
                "materialized storage segment contains negative decoder pre-roll"
            )
        if not (-tolerance_ns <= first_pts_ns <= tolerance_ns):
            raise MageStreamMaterializationError(
                "materialized storage segment first PTS is not normalized to zero"
            )
        if any(pts_ns < -tolerance_ns or dts_ns < -tolerance_ns for pts_ns, dts_ns, _ in packets):
            raise MageStreamMaterializationError(
                "materialized storage segment contains packet timestamps before its boundary"
            )
        if max(pts_ns for pts_ns, _dts_ns, _flags in packets) >= segment.interval.duration_ns:
            raise MageStreamMaterializationError(
                "materialized storage segment contains a packet at or beyond its end boundary"
            )

    def _storage_segment_command(
        self,
        *,
        source_path: Path,
        recording_interval: AbsoluteNanosecondInterval,
        segment: MageStorageSegment,
        destination: Path,
    ) -> tuple[str, ...]:
        relative_start_ns = segment.interval.start_ns - recording_interval.start_ns
        if relative_start_ns < 0:
            raise MageStreamMaterializationError("storage segment begins before recording start")
        bounded_duration_ns = segment.interval.duration_ns - MATERIALIZATION_END_EXCLUSION_NS
        if bounded_duration_ns <= 0:
            raise MageStreamMaterializationError(
                "storage segment is too short for half-open stream-copy materialization"
            )
        return (
            self._ffmpeg_binary,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-ss",
            _ffmpeg_seconds(relative_start_ns),
            "-t",
            _ffmpeg_seconds(bounded_duration_ns),
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "disabled",
            "-n",
            str(destination),
        )

    def _concat_context_command(
        self,
        *,
        concat_manifest: Path,
        destination: Path,
    ) -> tuple[str, ...]:
        return (
            self._ffmpeg_binary,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest),
            "-map",
            "0",
            "-c",
            "copy",
            "-n",
            str(destination),
        )

    def _run_checked(self, command: tuple[str, ...]) -> None:
        try:
            result = self._command_runner(command)
        except FileNotFoundError as error:
            raise MageStreamMaterializationError(
                f"ffmpeg stream-copy materialization is unavailable: {self._ffmpeg_binary!r}"
            ) from error
        except OSError as error:
            raise MageStreamMaterializationError(
                "ffmpeg stream-copy materialization could not start"
            ) from error
        if not isinstance(result, FfmpegCommandResult):
            raise MageStreamMaterializationError(
                "injected ffmpeg command runner must return FfmpegCommandResult"
            )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no ffmpeg diagnostic"
            raise MageStreamMaterializationError(
                f"ffmpeg stream-copy materialization failed with exit code {result.returncode}: "
                f"{detail}"
            )


def build_perception_context_manifest(
    *,
    plan: MageStreamPlan,
    context: MageReasoningContext,
    materialized_context: MageMaterializedReasoningContext,
    codec_policy_version: str,
) -> PerceptionContextManifest:
    """Build the additive perception context for the v1 single-camera input."""

    validate_mage_stream_plan(plan)
    if context not in plan.reasoning_contexts:
        raise MageStreamPlanningError("context does not belong to supplied plan")
    if materialized_context.context != context:
        raise MageStreamPlanningError("materialized context does not match reasoning context")
    _require_nonempty_text(codec_policy_version, "codec_policy_version")
    ordered_segments = tuple(
        StorageSegmentReference(
            segment_ordinal=segment.ordinal,
            segment_key=segment.segment_key,
            segment_semantic_sha256=segment.segment_semantic_sha256,
            interval=NanosecondInterval(
                start_ns=segment.interval.start_ns,
                end_ns=segment.interval.end_ns,
            ),
        )
        for segment in context.ordered_segments
    )
    segment_digests = tuple(item.segment_semantic_sha256 for item in context.ordered_segments)
    bindings: dict[CameraId, CameraContextBinding] = {}
    for camera_id in CAMERA_IDS:
        if camera_id is materialized_context.camera_id:
            bindings[camera_id] = CameraContextBinding(
                camera_id=camera_id,
                available=True,
                selected_for_inference=True,
                codec_stream_exact_sha256=materialized_context.content_exact_sha256,
                segment_semantic_sha256_values=segment_digests,
            )
        else:
            bindings[camera_id] = CameraContextBinding(
                camera_id=camera_id,
                available=False,
                selected_for_inference=False,
                absence_reason=CameraAbsenceReason.UNAVAILABLE,
            )
    return create_perception_context_manifest(
        source_recording_key=plan.recording.recording_key,
        source_recording_exact_sha256=plan.recording.recording_exact_sha256,
        context_interval=NanosecondInterval(
            start_ns=context.materialized_interval.start_ns,
            end_ns=context.materialized_interval.end_ns,
        ),
        ordered_segments=ordered_segments,
        focus_segment_ordinal=context.focus_segment_ordinal,
        cameras=SixCameraMap[CameraContextBinding](bindings),
        codec_policy_version=codec_policy_version,
        context_policy_version=plan.policy.policy_version,
    )


def exact_file_sha256(path: Path) -> tuple[str, int]:
    """Return exact file bytes SHA-256 then byte count."""

    file_path = _require_existing_file(path, "path")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise MageStreamMaterializationError(f"could not hash file: {file_path}") from error
    return digest.hexdigest(), byte_count


def validate_mage_stream_plan(plan: MageStreamPlan) -> None:
    """Revalidate a plan received from storage or another process."""

    if not isinstance(plan, MageStreamPlan):
        raise MageStreamPlanningError("plan must be MageStreamPlan")
    _validate_storage_segment_partition(plan.recording, plan.storage_segments, policy=plan.policy)
    if len(plan.reasoning_contexts) != len(plan.storage_segments):
        raise MageStreamPlanningError("plan must have one context per storage segment")
    for segment, context in zip(plan.storage_segments, plan.reasoning_contexts, strict=True):
        if (
            context.focus_segment_ordinal != segment.ordinal
            or context.ordered_segments[-1] != segment
        ):
            raise MageStreamPlanningError("plan context does not bind its focus storage segment")
    expected_digest = semantic_sha256(plan.semantic_projection())
    if plan.plan_semantic_sha256 != expected_digest:
        raise MageStreamPlanningError("plan semantic SHA-256 does not match its projection")
    if plan.plan_key != f"{MAGE_STREAM_PLAN_KEY_NAMESPACE}:{expected_digest}":
        raise MageStreamPlanningError("plan key does not bind its semantic SHA-256")


def _validate_storage_segment_partition(
    recording: MageStreamRecording,
    segments: Sequence[MageStorageSegment],
    *,
    policy: MageStreamPolicy | None = None,
) -> None:
    if not segments:
        raise MageStreamPlanningError("recording must have at least one storage segment")
    previous: MageStorageSegment | None = None
    for expected_ordinal, segment in enumerate(segments):
        if not isinstance(segment, MageStorageSegment):
            raise MageStreamPlanningError("storage segment must be MageStorageSegment")
        if segment.ordinal != expected_ordinal:
            raise MageStreamPlanningError(
                "storage segment ordinals must be contiguous and zero-based"
            )
        if segment.recording_key != recording.recording_key:
            raise MageStreamPlanningError("storage segment belongs to another recording key")
        if segment.recording_exact_sha256 != recording.recording_exact_sha256:
            raise MageStreamPlanningError("storage segment belongs to another recording digest")
        if policy is not None:
            if segment.segment_policy_version != policy.policy_version:
                raise MageStreamPlanningError("storage segment policy does not match stream policy")
            if policy.segmentation_mode is MageStreamSegmentationMode.FIXED_DURATION:
                if expected_ordinal < len(segments) - 1:
                    if segment.interval.duration_ns != policy.scan_segment_duration_ns:
                        raise MageStreamPlanningError(
                            "non-tail storage segments must use scan duration"
                        )
                elif segment.interval.duration_ns > policy.scan_segment_duration_ns:
                    raise MageStreamPlanningError("tail storage segment exceeds scan duration")
            else:
                if expected_ordinal < len(segments) - 1:
                    nominal_end_ns = (
                        recording.interval.start_ns
                        + (expected_ordinal + 1) * policy.scan_segment_duration_ns
                    )
                    if (
                        abs(segment.interval.end_ns - nominal_end_ns)
                        > policy.keyframe_alignment_tolerance_ns
                    ):
                        raise MageStreamPlanningError(
                            "keyframe-aligned boundary exceeds declared alignment tolerance"
                        )
                maximum_duration_ns = (
                    policy.scan_segment_duration_ns + 2 * policy.keyframe_alignment_tolerance_ns
                )
                if segment.interval.duration_ns > maximum_duration_ns:
                    raise MageStreamPlanningError(
                        "keyframe-aligned storage segment exceeds bounded scan duration"
                    )
        if previous is not None:
            if segment.interval.start_ns < previous.interval.end_ns:
                raise MageStreamPlanningError("storage segments must not overlap")
            if segment.interval.start_ns > previous.interval.end_ns:
                raise MageStreamPlanningError("storage segments must not contain gaps")
        previous = segment
    if segments[0].interval.start_ns != recording.interval.start_ns:
        raise MageStreamPlanningError("storage segments must begin at recording start")
    if segments[-1].interval.end_ns != recording.interval.end_ns:
        raise MageStreamPlanningError("storage segments must end at recording end")


def _validate_contiguous_segments(segments: Sequence[MageStorageSegment]) -> None:
    previous: MageStorageSegment | None = None
    for segment in segments:
        if previous is not None:
            if segment.ordinal != previous.ordinal + 1:
                raise MageStreamPlanningError("context storage ordinals must be consecutive")
            if segment.interval.start_ns < previous.interval.end_ns:
                raise MageStreamPlanningError("context storage segments must not overlap")
            if segment.interval.start_ns > previous.interval.end_ns:
                raise MageStreamPlanningError("context storage segments must not contain gaps")
        previous = segment


def _select_materialized_segments(
    context: MageReasoningContext,
    storage_segments: Sequence[MageMaterializedStorageSegment],
) -> tuple[MageMaterializedStorageSegment, ...]:
    by_digest: dict[str, MageMaterializedStorageSegment] = {}
    for item in storage_segments:
        if not isinstance(item, MageMaterializedStorageSegment):
            raise MageStreamMaterializationError(
                "storage_segments must contain MageMaterializedStorageSegment values"
            )
        digest = item.segment.segment_semantic_sha256
        if digest in by_digest:
            raise MageStreamMaterializationError("duplicate materialized storage segment digest")
        by_digest[digest] = item
    selected: list[MageMaterializedStorageSegment] = []
    for segment in context.ordered_segments:
        selected_item = by_digest.get(segment.segment_semantic_sha256)
        if selected_item is None:
            raise MageStreamMaterializationError(
                "reasoning context storage segment was not materialized"
            )
        if selected_item.segment != segment:
            raise MageStreamMaterializationError(
                "materialized storage segment does not match lineage"
            )
        _validate_materialized_storage_item(selected_item)
        selected.append(selected_item)
    return tuple(selected)


def _subprocess_ffmpeg_runner(command: tuple[str, ...]) -> FfmpegCommandResult:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return FfmpegCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _ffmpeg_seconds(value_ns: int) -> str:
    _require_nonnegative_int(value_ns, "ffmpeg timestamp")
    value = Decimal(value_ns) / Decimal(NANOSECONDS_PER_SECOND)
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _ffconcat_file_line(path: Path) -> str:
    value = str(path.resolve()).replace("'", r"'\''")
    return f"file '{value}'\n"


def _prepare_output_root(path: Path) -> Path:
    if not isinstance(path, Path):
        raise MageStreamMaterializationError("output_root must be pathlib.Path")
    root = path.expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MageStreamMaterializationError(f"could not create output root: {root}") from error
    return root


def _require_existing_file(path: Path, field: str) -> Path:
    if not isinstance(path, Path):
        raise MageStreamMaterializationError(f"{field} must be pathlib.Path")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise MageStreamMaterializationError(f"{field} must be a regular file: {resolved}")
    return resolved


def _reject_symlink_path(path: Path, field: str) -> None:
    if path.is_symlink():
        raise MageStreamMaterializationError(f"{field} must not be a symbolic link: {path}")


def _materialization_receipt_path(destination: Path) -> Path:
    return destination.with_name(destination.name + _MATERIALIZATION_RECEIPT_SUFFIX)


def _require_nonempty_file_hash(path: Path, detail: str) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise MageStreamMaterializationError(detail)
    digest, byte_count = exact_file_sha256(path)
    if byte_count <= 0:
        raise MageStreamMaterializationError(detail)
    return digest, byte_count


def _storage_materialization_receipt(
    *,
    segment: MageStorageSegment,
    source_exact_sha256: str,
    content_exact_sha256: str,
    byte_count: int,
) -> dict[str, object]:
    return {
        "receipt_version": MAGE_STREAM_MATERIALIZATION_RECEIPT_VERSION,
        "kind": "storage_segment",
        "recording_key": segment.recording_key,
        "recording_exact_sha256": segment.recording_exact_sha256,
        "segment_policy_version": segment.segment_policy_version,
        "segment_semantic_sha256": segment.segment_semantic_sha256,
        "segment_key": segment.segment_key,
        "ordinal": segment.ordinal,
        "interval": segment.interval.as_projection(),
        "source_exact_sha256": source_exact_sha256,
        "output_exact_sha256": content_exact_sha256,
        "output_byte_count": byte_count,
    }


def _context_materialization_receipt(
    *,
    context: MageReasoningContext,
    camera_id: CameraId,
    component_hashes: tuple[str, ...],
    content_exact_sha256: str,
    byte_count: int,
) -> dict[str, object]:
    focus = context.ordered_segments[-1]
    return {
        "receipt_version": MAGE_STREAM_MATERIALIZATION_RECEIPT_VERSION,
        "kind": "reasoning_context",
        "recording_key": focus.recording_key,
        "recording_exact_sha256": focus.recording_exact_sha256,
        "context_policy_version": context.context_policy_version,
        "context_semantic_sha256": context.context_semantic_sha256,
        "context_key": context.context_key,
        "focus_segment_ordinal": context.focus_segment_ordinal,
        "reasoning_horizon": context.reasoning_horizon.as_projection(),
        "materialized_interval": context.materialized_interval.as_projection(),
        "camera_id": camera_id.value,
        "component_segment_exact_sha256_values": list(component_hashes),
        "output_exact_sha256": content_exact_sha256,
        "output_byte_count": byte_count,
    }


def _write_materialization_receipt(path: Path, projection: dict[str, object]) -> None:
    if not isinstance(path, Path):
        raise MageStreamMaterializationError("materialization receipt path must be pathlib.Path")
    _reject_symlink_path(path, "materialization receipt")
    try:
        payload = canonical_json_bytes(projection) + b"\n"
    except (TypeError, ValueError) as error:
        raise MageStreamMaterializationError(
            "materialization receipt contains non-canonical values"
        ) from error
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".r-",
            suffix=".t",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except OSError as error:
        raise MageStreamMaterializationError(
            f"could not atomically publish materialization receipt: {path}"
        ) from error
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _strict_receipt_json(path: Path) -> dict[str, object]:
    _reject_symlink_path(path, "materialization receipt")
    if not path.is_file():
        raise MageStreamMaterializationError(f"materialization receipt is missing: {path}")
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_receipt_object_pairs,
            parse_constant=_reject_receipt_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise MageStreamMaterializationError(
            f"materialization receipt is not strict canonical JSON: {path}"
        ) from error
    if not isinstance(document, dict):
        raise MageStreamMaterializationError("materialization receipt root must be an object")
    try:
        expected_bytes = canonical_json_bytes(document) + b"\n"
    except (TypeError, ValueError) as error:
        raise MageStreamMaterializationError(
            "materialization receipt is not canonical JSON"
        ) from error
    if raw != expected_bytes:
        raise MageStreamMaterializationError("materialization receipt bytes are not canonical")
    return document


def _receipt_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate materialization receipt key: {key}")
        result[key] = value
    return result


def _reject_receipt_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _strict_value_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_value_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_value_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _validate_storage_materialization_receipt(
    *,
    destination: Path,
    receipt_path: Path,
    segment: MageStorageSegment,
    source_exact_sha256: str,
) -> tuple[str, int]:
    expected = _storage_materialization_receipt(
        segment=segment,
        source_exact_sha256=source_exact_sha256,
        content_exact_sha256="0" * _SHA256_HEX_LENGTH,
        byte_count=1,
    )
    expected.pop("output_exact_sha256")
    expected.pop("output_byte_count")
    document = _strict_receipt_json(receipt_path)
    if set(document) != set(expected) | {"output_exact_sha256", "output_byte_count"}:
        raise MageStreamMaterializationError(
            "materialization receipt fields do not match storage-segment v1"
        )
    for field, expected_value in expected.items():
        if not _strict_value_equal(document.get(field), expected_value):
            raise MageStreamMaterializationError(
                f"storage segment receipt lineage mismatch for {field}"
            )
    content_digest, byte_count = _require_nonempty_file_hash(
        destination,
        "materialized storage segment is missing or empty",
    )
    if document.get("output_exact_sha256") != content_digest:
        raise MageStreamMaterializationError(
            "materialized storage segment bytes do not match its receipt"
        )
    if (
        type(document.get("output_byte_count")) is not int
        or document["output_byte_count"] != byte_count
    ):
        raise MageStreamMaterializationError(
            "materialized storage segment byte count does not match its receipt"
        )
    return content_digest, byte_count


def _validate_context_materialization_receipt(
    *,
    destination: Path,
    receipt_path: Path,
    context: MageReasoningContext,
    camera_id: CameraId,
    component_hashes: tuple[str, ...],
) -> tuple[str, int]:
    expected = _context_materialization_receipt(
        context=context,
        camera_id=camera_id,
        component_hashes=component_hashes,
        content_exact_sha256="0" * _SHA256_HEX_LENGTH,
        byte_count=1,
    )
    expected.pop("output_exact_sha256")
    expected.pop("output_byte_count")
    document = _strict_receipt_json(receipt_path)
    if set(document) != set(expected) | {"output_exact_sha256", "output_byte_count"}:
        raise MageStreamMaterializationError(
            "materialization receipt fields do not match reasoning-context v1"
        )
    for field, expected_value in expected.items():
        if not _strict_value_equal(document.get(field), expected_value):
            raise MageStreamMaterializationError(
                f"reasoning context receipt lineage mismatch for {field}"
            )
    content_digest, byte_count = _require_nonempty_file_hash(
        destination,
        "materialized reasoning context is missing or empty",
    )
    if document.get("output_exact_sha256") != content_digest:
        raise MageStreamMaterializationError(
            "materialized reasoning context bytes do not match its receipt"
        )
    if (
        type(document.get("output_byte_count")) is not int
        or document["output_byte_count"] != byte_count
    ):
        raise MageStreamMaterializationError(
            "materialized reasoning context byte count does not match its receipt"
        )
    return content_digest, byte_count


def _validate_materialized_storage_item(item: MageMaterializedStorageSegment) -> None:
    if item.source_exact_sha256 != item.segment.recording_exact_sha256:
        raise MageStreamMaterializationError(
            "materialized storage segment source identity does not match segment lineage"
        )
    digest, byte_count = _validate_storage_materialization_receipt(
        destination=item.durable_path,
        receipt_path=_materialization_receipt_path(item.durable_path),
        segment=item.segment,
        source_exact_sha256=item.source_exact_sha256,
    )
    if digest != item.content_exact_sha256 or byte_count != item.byte_count:
        raise MageStreamMaterializationError(
            "materialized storage segment metadata does not match its receipt"
        )


def _require_nonempty_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MageStreamPlanningError(f"{field} must be a nonempty string")


def _require_nonnegative_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MageStreamPlanningError(f"{field} must be a nonnegative integer")


def _require_positive_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MageStreamPlanningError(f"{field} must be a positive integer")


def _require_sha256(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MageStreamPlanningError(f"{field} must be lowercase hexadecimal SHA-256")


__all__ = [
    "DEFAULT_KEYFRAME_ALIGNMENT_TOLERANCE_NS",
    "DEFAULT_REASONING_HORIZON_DURATION_NS",
    "DEFAULT_SCAN_SEGMENT_DURATION_NS",
    "MAGE_STREAM_CONTEXT_KEY_NAMESPACE",
    "MAGE_STREAM_MATERIALIZATION_RECEIPT_VERSION",
    "MAGE_STREAM_PLAN_KEY_NAMESPACE",
    "MAGE_STREAM_POLICY_VERSION",
    "MAGE_STREAM_SEGMENT_KEY_NAMESPACE",
    "AbsoluteNanosecondInterval",
    "FfmpegCommandResult",
    "FfmpegCommandRunner",
    "FfprobeCommandRunner",
    "MageMaterializedReasoningContext",
    "MageMaterializedStorageSegment",
    "MageReasoningContext",
    "MageStorageSegment",
    "MageStreamMaterializationError",
    "MageStreamMaterializer",
    "MageStreamPlan",
    "MageStreamPlanningError",
    "MageStreamPolicy",
    "MageStreamRecording",
    "MageStreamSegmentationMode",
    "build_perception_context_manifest",
    "exact_file_sha256",
    "plan_keyframe_aligned_mage_stream",
    "plan_mage_stream",
    "probe_video_keyframe_offsets_ns",
    "validate_mage_stream_plan",
]
