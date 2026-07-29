"""Deterministic local media observations and supplemental target planning."""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from statistics import median_low
from typing import Any, Final

from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import NanosecondInterval
from robata.contracts.hashing import canonical_json_bytes, semantic_sha256
from robata.contracts.schema_registry import SchemaRef, SchemaRegistry
from robata.ports.decoded_frame import DecodedFrameView
from robata.sampling.grid import SamplingGrid, SamplingRate

LOCAL_MEDIA_QUALITY_POLICY_VERSION: Final = "local-media-quality-observation-v2"
LOCAL_NEIGHBOR_TARGET_POLICY_VERSION: Final = "local-neighbor-target-v1"
LOCAL_MEDIA_QUALITY_REPORT_FORMAT_VERSION: Final = "local-media-quality-report-v1"
LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID: Final = "https://schemas.robata.dev/media-quality-report"
LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION: Final = "1.0.0"
LOCAL_MEDIA_QUALITY_REPORT_WIRE_VERSION: Final = "1.0"


class LocalQualityFlag(StrEnum):
    """Local observations and deliberately limited visual proxies."""

    OBSERVED_BLACK_LUMA = "OBSERVED_BLACK_LUMA"
    OBSERVED_OVEREXPOSED_LUMA = "OBSERVED_OVEREXPOSED_LUMA"
    PROXY_LOW_EDGE_ENERGY = "PROXY_LOW_EDGE_ENERGY"
    PROXY_FROZEN_CONTENT = "PROXY_FROZEN_CONTENT"
    OBSERVED_CADENCE_GAP = "OBSERVED_CADENCE_GAP"
    OBSERVED_SEQUENCE_GAP = "OBSERVED_SEQUENCE_GAP"
    OBSERVED_CROSS_CAMERA_SKEW = "OBSERVED_CROSS_CAMERA_SKEW"


class QualityTriggerSource(StrEnum):
    FRAME = "FRAME"
    CADENCE = "CADENCE"
    SEQUENCE = "SEQUENCE"
    CROSS_CAMERA_SYNC = "CROSS_CAMERA_SYNC"


@dataclass(frozen=True, slots=True)
class LocalMediaQualityPolicy:
    version: str = LOCAL_MEDIA_QUALITY_POLICY_VERSION
    analysis_width: int = 64
    black_luma_max: int = 16
    black_fraction_ppm: int = 950_000
    overexposed_luma_min: int = 240
    overexposed_fraction_ppm: int = 950_000
    low_edge_energy_milli: int = 1_500
    freeze_delta_milli: int = 250
    freeze_min_duration_ns: int = 5_000_000_000
    cadence_ratio_numerator: int = 3
    cadence_ratio_denominator: int = 2
    cadence_min_gap_ns: int = 50_000_000
    sync_rate_numerator: int = 2
    sync_rate_denominator: int = 1
    sync_selection_tolerance_ns: int = 300_000_000
    sync_skew_threshold_ns: int = 20_000_000

    def __post_init__(self) -> None:
        positive = (
            self.analysis_width,
            self.black_fraction_ppm,
            self.overexposed_fraction_ppm,
            self.low_edge_energy_milli,
            self.freeze_min_duration_ns,
            self.cadence_ratio_numerator,
            self.cadence_ratio_denominator,
            self.cadence_min_gap_ns,
            self.sync_rate_numerator,
            self.sync_rate_denominator,
            self.sync_selection_tolerance_ns,
            self.sync_skew_threshold_ns,
        )
        if not self.version:
            raise ValueError("quality policy version must be non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive
        ):
            raise ValueError("quality policy positive integer fields must be positive integers")
        if not 0 <= self.black_luma_max <= 255:
            raise ValueError("black_luma_max must be in [0, 255]")
        if not 0 <= self.overexposed_luma_min <= 255:
            raise ValueError("overexposed_luma_min must be in [0, 255]")
        if self.black_fraction_ppm > 1_000_000 or self.overexposed_fraction_ppm > 1_000_000:
            raise ValueError("luma fractions must be no greater than one million ppm")
        if self.freeze_delta_milli < 0:
            raise ValueError("freeze_delta_milli must be nonnegative")


@dataclass(frozen=True, slots=True)
class NeighborTargetPolicy:
    version: str = LOCAL_NEIGHBOR_TARGET_POLICY_VERSION
    offsets_ns: tuple[int, ...] = (-500_000_000, 500_000_000)
    max_targets_per_camera: int = 64
    max_targets_total: int = 384

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("neighbor target policy version must be non-empty")
        if not self.offsets_ns or any(
            isinstance(value, bool) or not isinstance(value, int) or value == 0
            for value in self.offsets_ns
        ):
            raise ValueError("neighbor offsets must be non-zero integers")
        if tuple(sorted(set(self.offsets_ns))) != self.offsets_ns:
            raise ValueError("neighbor offsets must be unique and strictly increasing")
        if (
            isinstance(self.max_targets_per_camera, bool)
            or not isinstance(self.max_targets_per_camera, int)
            or self.max_targets_per_camera <= 0
            or isinstance(self.max_targets_total, bool)
            or not isinstance(self.max_targets_total, int)
            or self.max_targets_total <= 0
        ):
            raise ValueError("neighbor target budgets must be positive integers")


DEFAULT_MEDIA_QUALITY_POLICY: Final = LocalMediaQualityPolicy()
DEFAULT_NEIGHBOR_TARGET_POLICY: Final = NeighborTargetPolicy()


@dataclass(frozen=True, slots=True)
class FrameTimingEvidence:
    camera_id: CameraId
    packet_index: int
    aligned_timestamp_ns: int
    source_timestamp_ns: int
    source_sequence: int


@dataclass(frozen=True, slots=True)
class FrameQualityObservation:
    camera_id: CameraId
    packet_index: int
    aligned_timestamp_ns: int
    source_timestamp_ns: int
    grayscale_sha256: str
    mean_luma_milli: int
    black_fraction_ppm: int
    overexposed_fraction_ppm: int
    edge_energy_milli: int
    frame_delta_milli: int | None
    flags: tuple[LocalQualityFlag, ...]
    # Supplemental visual facts remain internal until a registered wire schema is added.
    blur_score_milli: int | None = None
    motion_energy_milli: int | None = None
    scene_change_milli: int | None = None


@dataclass(frozen=True, slots=True)
class CadenceGapObservation:
    camera_id: CameraId
    previous_packet_index: int
    packet_index: int
    previous_timestamp_ns: int
    timestamp_ns: int
    delta_ns: int
    expected_delta_ns: int


@dataclass(frozen=True, slots=True)
class SequenceGapObservation:
    camera_id: CameraId
    previous_packet_index: int
    packet_index: int
    previous_sequence: int
    sequence: int
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class CameraMediaQualityLedger:
    camera_id: CameraId
    timing_count: int
    decoded_observations: tuple[FrameQualityObservation, ...]
    cadence_gaps: tuple[CadenceGapObservation, ...]
    sequence_gaps: tuple[SequenceGapObservation, ...]
    flags: tuple[LocalQualityFlag, ...]


@dataclass(frozen=True, slots=True)
class CrossCameraSkewSample:
    target_ns: int
    camera_timestamps_ns: tuple[tuple[CameraId, int], ...]
    skew_ns: int


@dataclass(frozen=True, slots=True)
class CrossCameraSkewReport:
    samples: tuple[CrossCameraSkewSample, ...]
    incomplete_target_count: int
    p50_ns: int | None
    p95_ns: int | None
    max_ns: int | None
    threshold_ns: int
    flags: tuple[LocalQualityFlag, ...]


@dataclass(frozen=True, slots=True)
class QualityTriggerProvenance:
    camera_id: CameraId
    trigger_timestamp_ns: int
    source: QualityTriggerSource
    flag: LocalQualityFlag
    packet_index: int | None = None


@dataclass(frozen=True, slots=True)
class NeighborTargetProvenance:
    trigger: QualityTriggerProvenance
    offset_ns: int
    requested_target_ns: int
    clipped: bool


@dataclass(frozen=True, slots=True)
class SupplementalNeighborTarget:
    camera_id: CameraId
    target_ns: int
    provenance: tuple[NeighborTargetProvenance, ...]


@dataclass(frozen=True, slots=True)
class SupplementalNeighborTargetPlan:
    policy_version: str
    interval: NanosecondInterval
    candidate_count: int
    clipped_count: int
    deduplicated_count: int
    dropped_by_per_camera_budget: int
    dropped_by_total_budget: int
    targets: tuple[SupplementalNeighborTarget, ...]
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class LocalMediaQualityReport:
    policy_version: str
    requested_max_duration_ns: int | None
    recording_duration_ns: int
    requested_interval: NanosecondInterval
    window_limited: bool
    camera_ledgers: tuple[CameraMediaQualityLedger, ...]
    cross_camera_skew: CrossCameraSkewReport
    supplemental_targets: SupplementalNeighborTargetPlan
    semantic_sha256: str


class LocalFrameQualityAnalyzer:
    """Extract deterministic luma observations from normalized decoded frame views."""

    def __init__(
        self,
        camera_id: CameraId,
        policy: LocalMediaQualityPolicy = DEFAULT_MEDIA_QUALITY_POLICY,
    ) -> None:
        self._camera_id = camera_id
        self._policy = policy
        self._previous_gray: bytes | None = None
        self._previous_dimensions: tuple[int, int] | None = None
        self._previous_timestamp_ns: int | None = None
        self._stable_start_ns: int | None = None

    def observe(
        self,
        frame: DecodedFrameView,
        timing: FrameTimingEvidence,
    ) -> FrameQualityObservation:
        """Observe one normalized decoded view and its source-timeline evidence."""

        if not isinstance(frame, DecodedFrameView):
            raise TypeError("frame must be a DecodedFrameView")
        if timing.camera_id is not self._camera_id:
            raise ValueError("frame timing camera differs from analyzer camera")
        if frame.timestamp_ns != timing.aligned_timestamp_ns:
            raise ValueError("decoded frame timestamp differs from aligned timing evidence")
        if (
            self._previous_timestamp_ns is not None
            and timing.aligned_timestamp_ns <= self._previous_timestamp_ns
        ):
            raise ValueError("decoded frame timestamps must be strictly increasing")

        dimensions = (frame.width, frame.height)
        gray = frame.gray_pixels
        pixel_count = frame.pixel_count
        mean_luma_milli = _rounded_ratio(sum(gray) * 1_000, pixel_count)
        black_fraction_ppm = _rounded_ratio(
            sum(value <= self._policy.black_luma_max for value in gray) * 1_000_000,
            pixel_count,
        )
        overexposed_fraction_ppm = _rounded_ratio(
            sum(value >= self._policy.overexposed_luma_min for value in gray) * 1_000_000,
            pixel_count,
        )
        edge_energy_milli = _edge_energy_milli(gray, dimensions)
        if self._previous_dimensions is not None and dimensions != self._previous_dimensions:
            raise ValueError("decoded frame view dimensions must remain stable per analyzer")
        blur_score_milli = _blur_score_milli(gray, dimensions)
        scene_change_milli = (
            None if self._previous_gray is None else _scene_change_milli(gray, self._previous_gray)
        )
        frame_delta_milli = (
            None
            if self._previous_gray is None
            else _rounded_ratio(
                sum(
                    abs(current - previous)
                    for current, previous in zip(gray, self._previous_gray, strict=True)
                )
                * 1_000,
                pixel_count,
            )
        )

        flags: set[LocalQualityFlag] = set()
        if black_fraction_ppm >= self._policy.black_fraction_ppm:
            flags.add(LocalQualityFlag.OBSERVED_BLACK_LUMA)
        if overexposed_fraction_ppm >= self._policy.overexposed_fraction_ppm:
            flags.add(LocalQualityFlag.OBSERVED_OVEREXPOSED_LUMA)
        if edge_energy_milli <= self._policy.low_edge_energy_milli:
            flags.add(LocalQualityFlag.PROXY_LOW_EDGE_ENERGY)

        if frame_delta_milli is not None and frame_delta_milli <= self._policy.freeze_delta_milli:
            if self._stable_start_ns is None:
                assert self._previous_timestamp_ns is not None
                self._stable_start_ns = self._previous_timestamp_ns
            if (
                timing.aligned_timestamp_ns - self._stable_start_ns
                >= self._policy.freeze_min_duration_ns
            ):
                flags.add(LocalQualityFlag.PROXY_FROZEN_CONTENT)
        else:
            self._stable_start_ns = None

        self._previous_gray = gray
        self._previous_dimensions = dimensions
        self._previous_timestamp_ns = timing.aligned_timestamp_ns
        return FrameQualityObservation(
            camera_id=self._camera_id,
            packet_index=timing.packet_index,
            aligned_timestamp_ns=timing.aligned_timestamp_ns,
            source_timestamp_ns=timing.source_timestamp_ns,
            grayscale_sha256=sha256(gray).hexdigest(),
            mean_luma_milli=mean_luma_milli,
            black_fraction_ppm=black_fraction_ppm,
            overexposed_fraction_ppm=overexposed_fraction_ppm,
            edge_energy_milli=edge_energy_milli,
            frame_delta_milli=frame_delta_milli,
            flags=tuple(sorted(flags, key=lambda value: value.value)),
            blur_score_milli=blur_score_milli,
            motion_energy_milli=frame_delta_milli,
            scene_change_milli=scene_change_milli,
        )


def build_local_media_quality_report(
    *,
    requested_max_duration_ns: int | None,
    recording_duration_ns: int,
    requested_interval: NanosecondInterval,
    timings: Mapping[CameraId, Sequence[FrameTimingEvidence]],
    frame_observations: Mapping[CameraId, Sequence[FrameQualityObservation]],
    policy: LocalMediaQualityPolicy = DEFAULT_MEDIA_QUALITY_POLICY,
    neighbor_policy: NeighborTargetPolicy = DEFAULT_NEIGHBOR_TARGET_POLICY,
) -> LocalMediaQualityReport:
    """Reduce decoded observations and exact timing rows into one local report."""

    if set(timings) != set(CAMERA_IDS) or set(frame_observations) != set(CAMERA_IDS):
        raise ValueError("local media quality report requires all six canonical cameras")
    if recording_duration_ns <= 0:
        raise ValueError("recording_duration_ns must be positive")

    ledgers: list[CameraMediaQualityLedger] = []
    triggers: list[QualityTriggerProvenance] = []
    normalized_timings: dict[CameraId, tuple[FrameTimingEvidence, ...]] = {}
    for camera_id in CAMERA_IDS:
        camera_timings = tuple(
            row
            for row in timings[camera_id]
            if requested_interval.start_ns <= row.aligned_timestamp_ns < requested_interval.end_ns
        )
        normalized_timings[camera_id] = camera_timings
        observations = tuple(frame_observations[camera_id])
        if any(observation.camera_id is not camera_id for observation in observations):
            raise ValueError("frame observations must remain in their canonical camera ledger")
        cadence_gaps = _cadence_gaps(camera_id, camera_timings, policy)
        sequence_gaps = _sequence_gaps(camera_id, camera_timings)
        flags = {flag for observation in observations for flag in observation.flags}
        if cadence_gaps:
            flags.add(LocalQualityFlag.OBSERVED_CADENCE_GAP)
        if sequence_gaps:
            flags.add(LocalQualityFlag.OBSERVED_SEQUENCE_GAP)
        ledgers.append(
            CameraMediaQualityLedger(
                camera_id=camera_id,
                timing_count=len(camera_timings),
                decoded_observations=observations,
                cadence_gaps=cadence_gaps,
                sequence_gaps=sequence_gaps,
                flags=tuple(sorted(flags, key=lambda value: value.value)),
            )
        )
        triggers.extend(_camera_triggers(observations, cadence_gaps, sequence_gaps))

    skew_report = _cross_camera_skew(normalized_timings, requested_interval, policy)
    triggers.extend(_skew_triggers(skew_report))
    supplemental = plan_neighbor_targets(
        triggers,
        interval=requested_interval,
        policy=neighbor_policy,
    )
    draft = LocalMediaQualityReport(
        policy_version=policy.version,
        requested_max_duration_ns=requested_max_duration_ns,
        recording_duration_ns=recording_duration_ns,
        requested_interval=requested_interval,
        window_limited=requested_interval.end_ns < recording_duration_ns,
        camera_ledgers=tuple(ledgers),
        cross_camera_skew=skew_report,
        supplemental_targets=supplemental,
        semantic_sha256="0" * 64,
    )
    return replace(draft, semantic_sha256=semantic_sha256(_report_projection(draft)))


def plan_neighbor_targets(
    triggers: Sequence[QualityTriggerProvenance],
    *,
    interval: NanosecondInterval,
    policy: NeighborTargetPolicy = DEFAULT_NEIGHBOR_TARGET_POLICY,
) -> SupplementalNeighborTargetPlan:
    """Create clipped, deduplicated, budgeted targets with complete provenance."""

    candidates: dict[tuple[CameraId, int], list[NeighborTargetProvenance]] = {}
    candidate_count = 0
    clipped_count = 0
    for trigger in sorted(
        triggers,
        key=lambda value: (
            value.trigger_timestamp_ns,
            value.camera_id.value,
            value.source.value,
            value.flag.value,
            -1 if value.packet_index is None else value.packet_index,
        ),
    ):
        for offset_ns in policy.offsets_ns:
            requested_target_ns = trigger.trigger_timestamp_ns + offset_ns
            target_ns = min(max(requested_target_ns, interval.start_ns), interval.end_ns - 1)
            clipped = target_ns != requested_target_ns
            candidate_count += 1
            clipped_count += int(clipped)
            candidates.setdefault((trigger.camera_id, target_ns), []).append(
                NeighborTargetProvenance(
                    trigger=trigger,
                    offset_ns=offset_ns,
                    requested_target_ns=requested_target_ns,
                    clipped=clipped,
                )
            )

    unique_targets = tuple(
        SupplementalNeighborTarget(
            camera_id=camera_id,
            target_ns=target_ns,
            provenance=tuple(provenance),
        )
        for (camera_id, target_ns), provenance in sorted(
            candidates.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        )
    )
    per_camera_admitted: list[SupplementalNeighborTarget] = []
    dropped_by_per_camera_budget = 0
    for camera_id in CAMERA_IDS:
        camera_targets = tuple(target for target in unique_targets if target.camera_id is camera_id)
        per_camera_admitted.extend(camera_targets[: policy.max_targets_per_camera])
        dropped_by_per_camera_budget += max(
            0,
            len(camera_targets) - policy.max_targets_per_camera,
        )

    globally_ordered = tuple(
        sorted(
            per_camera_admitted,
            key=lambda target: (target.target_ns, target.camera_id.value),
        )
    )
    admitted = globally_ordered[: policy.max_targets_total]
    dropped_by_total_budget = len(globally_ordered) - len(admitted)
    draft = SupplementalNeighborTargetPlan(
        policy_version=policy.version,
        interval=interval,
        candidate_count=candidate_count,
        clipped_count=clipped_count,
        deduplicated_count=candidate_count - len(unique_targets),
        dropped_by_per_camera_budget=dropped_by_per_camera_budget,
        dropped_by_total_budget=dropped_by_total_budget,
        targets=admitted,
        semantic_sha256="0" * 64,
    )
    return replace(draft, semantic_sha256=semantic_sha256(_neighbor_projection(draft)))


def pyav_decoded_frame_view(
    frame: Any,
    *,
    timestamp_ns: int,
    analysis_width: int = DEFAULT_MEDIA_QUALITY_POLICY.analysis_width,
) -> DecodedFrameView:
    """Convert one PyAV frame to the compact grayscale view used by local detectors.

    The returned bytes contain no PyAV line padding and are always row-major grayscale
    samples. ``timestamp_ns`` is passed by the source-timeline owner so no float timestamp
    conversion or timing approximation occurs at this boundary.
    """

    dimensions = _gray_dimensions(frame, analysis_width=analysis_width)
    return DecodedFrameView(
        timestamp_ns=timestamp_ns,
        width=dimensions[0],
        height=dimensions[1],
        gray_pixels=_normalized_gray_bytes(frame, dimensions=dimensions),
    )


def decoded_frame_view_from_pyav(
    frame: Any,
    *,
    timestamp_ns: int,
    analysis_width: int = DEFAULT_MEDIA_QUALITY_POLICY.analysis_width,
) -> DecodedFrameView:
    """Alias with source-first naming for :func:`pyav_decoded_frame_view`."""

    return pyav_decoded_frame_view(
        frame,
        timestamp_ns=timestamp_ns,
        analysis_width=analysis_width,
    )


def _gray_dimensions(frame: Any, *, analysis_width: int) -> tuple[int, int]:
    if isinstance(analysis_width, bool) or not isinstance(analysis_width, int):
        raise TypeError("analysis_width must be an integer")
    if analysis_width <= 0:
        raise ValueError("analysis_width must be positive")
    try:
        source_width = int(frame.width)
        source_height = int(frame.height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("frame must expose positive integer width and height") from exc
    if source_width <= 0 or source_height <= 0:
        raise ValueError("frame dimensions must be positive")
    width = min(analysis_width, source_width)
    height = max(1, source_height * width // source_width)
    return width, height


def _normalized_gray_bytes(frame: Any, *, dimensions: tuple[int, int]) -> bytes:
    width, height = dimensions
    converted = frame.reformat(width=width, height=height, format="gray")
    plane = converted.planes[0]
    line_size = int(plane.line_size)
    if line_size < width:
        raise ValueError("reformatted grayscale plane line size is shorter than its width")
    raw = bytes(plane)
    required_length = line_size * height
    if len(raw) < required_length:
        raise ValueError("reformatted grayscale plane is shorter than its declared dimensions")
    return b"".join(raw[row * line_size : row * line_size + width] for row in range(height))


def _rounded_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    return (numerator + denominator // 2) // denominator


def _edge_energy_milli(gray: bytes, dimensions: tuple[int, int]) -> int:
    width, height = dimensions
    horizontal = sum(
        abs(gray[row * width + column] - gray[row * width + column - 1])
        for row in range(height)
        for column in range(1, width)
    )
    vertical = sum(
        abs(gray[row * width + column] - gray[(row - 1) * width + column])
        for row in range(1, height)
        for column in range(width)
    )
    comparisons = height * max(0, width - 1) + width * max(0, height - 1)
    if comparisons == 0:
        return 0
    return _rounded_ratio((horizontal + vertical) * 1_000, comparisons)


def _blur_score_milli(gray: bytes, dimensions: tuple[int, int]) -> int:
    """Return a compact Laplacian-variance proxy for selected sentinel pixels."""

    width, height = dimensions
    if width < 3 or height < 3:
        return 0
    laplacian: list[int] = []
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            index = row * width + column
            center = gray[index]
            laplacian.append(
                4 * center
                - gray[index - 1]
                - gray[index + 1]
                - gray[index - width]
                - gray[index + width]
            )
    if not laplacian:
        return 0
    mean = sum(laplacian) / len(laplacian)
    variance = sum((value - mean) ** 2 for value in laplacian) / len(laplacian)
    return max(0, int(variance * 1_000 + 0.5))


def _scene_change_milli(current: bytes, previous: bytes) -> int:
    """Return a bounded 16-bin histogram distance for a pair of sentinel views."""

    if len(current) != len(previous) or not current:
        return 0
    current_hist = [0] * 16
    previous_hist = [0] * 16
    for value in current:
        current_hist[min(15, value // 16)] += 1
    for value in previous:
        previous_hist[min(15, value // 16)] += 1
    distance = sum(
        abs(left - right) for left, right in zip(current_hist, previous_hist, strict=True)
    )
    return _rounded_ratio(distance * 1_000, len(current))


def _cadence_gaps(
    camera_id: CameraId,
    timings: tuple[FrameTimingEvidence, ...],
    policy: LocalMediaQualityPolicy,
) -> tuple[CadenceGapObservation, ...]:
    if len(timings) < 3:
        return ()
    deltas = tuple(
        current.aligned_timestamp_ns - previous.aligned_timestamp_ns
        for previous, current in pairwise(timings)
    )
    expected_delta_ns = int(median_low(deltas))
    threshold_ns = max(
        policy.cadence_min_gap_ns,
        _rounded_ratio(
            expected_delta_ns * policy.cadence_ratio_numerator,
            policy.cadence_ratio_denominator,
        ),
    )
    return tuple(
        CadenceGapObservation(
            camera_id=camera_id,
            previous_packet_index=previous.packet_index,
            packet_index=current.packet_index,
            previous_timestamp_ns=previous.aligned_timestamp_ns,
            timestamp_ns=current.aligned_timestamp_ns,
            delta_ns=current.aligned_timestamp_ns - previous.aligned_timestamp_ns,
            expected_delta_ns=expected_delta_ns,
        )
        for previous, current in pairwise(timings)
        if current.aligned_timestamp_ns - previous.aligned_timestamp_ns > threshold_ns
    )


def _sequence_gaps(
    camera_id: CameraId,
    timings: tuple[FrameTimingEvidence, ...],
) -> tuple[SequenceGapObservation, ...]:
    return tuple(
        SequenceGapObservation(
            camera_id=camera_id,
            previous_packet_index=previous.packet_index,
            packet_index=current.packet_index,
            previous_sequence=previous.source_sequence,
            sequence=current.source_sequence,
            timestamp_ns=current.aligned_timestamp_ns,
        )
        for previous, current in pairwise(timings)
        if current.source_sequence != previous.source_sequence + 1
    )


def _camera_triggers(
    observations: tuple[FrameQualityObservation, ...],
    cadence_gaps: tuple[CadenceGapObservation, ...],
    sequence_gaps: tuple[SequenceGapObservation, ...],
) -> tuple[QualityTriggerProvenance, ...]:
    triggers = [
        QualityTriggerProvenance(
            camera_id=observation.camera_id,
            trigger_timestamp_ns=observation.aligned_timestamp_ns,
            source=QualityTriggerSource.FRAME,
            flag=flag,
            packet_index=observation.packet_index,
        )
        for observation in observations
        for flag in observation.flags
    ]
    triggers.extend(
        QualityTriggerProvenance(
            camera_id=gap.camera_id,
            trigger_timestamp_ns=gap.timestamp_ns,
            source=QualityTriggerSource.CADENCE,
            flag=LocalQualityFlag.OBSERVED_CADENCE_GAP,
            packet_index=gap.packet_index,
        )
        for gap in cadence_gaps
    )
    triggers.extend(
        QualityTriggerProvenance(
            camera_id=gap.camera_id,
            trigger_timestamp_ns=gap.timestamp_ns,
            source=QualityTriggerSource.SEQUENCE,
            flag=LocalQualityFlag.OBSERVED_SEQUENCE_GAP,
            packet_index=gap.packet_index,
        )
        for gap in sequence_gaps
    )
    return tuple(triggers)


def _cross_camera_skew(
    timings: Mapping[CameraId, tuple[FrameTimingEvidence, ...]],
    interval: NanosecondInterval,
    policy: LocalMediaQualityPolicy,
) -> CrossCameraSkewReport:
    grid = SamplingGrid(
        grid_origin_ns=0,
        rate=SamplingRate(policy.sync_rate_numerator, policy.sync_rate_denominator),
    )
    samples: list[CrossCameraSkewSample] = []
    incomplete_target_count = 0
    by_camera = {
        camera_id: tuple(row.aligned_timestamp_ns for row in timings[camera_id])
        for camera_id in CAMERA_IDS
    }
    for target in grid.iter_unique_targets(interval.start_ns, interval.end_ns):
        selected: list[tuple[CameraId, int]] = []
        for camera_id in CAMERA_IDS:
            timestamp_ns = _nearest_timestamp(by_camera[camera_id], target.target_ns)
            if (
                timestamp_ns is None
                or abs(timestamp_ns - target.target_ns) > policy.sync_selection_tolerance_ns
            ):
                selected = []
                break
            selected.append((camera_id, timestamp_ns))
        if not selected:
            incomplete_target_count += 1
            continue
        values = tuple(timestamp for _, timestamp in selected)
        samples.append(
            CrossCameraSkewSample(
                target_ns=target.target_ns,
                camera_timestamps_ns=tuple(selected),
                skew_ns=max(values) - min(values),
            )
        )

    ordered_skews = tuple(sorted(sample.skew_ns for sample in samples))
    maximum = ordered_skews[-1] if ordered_skews else None
    flags = (
        (LocalQualityFlag.OBSERVED_CROSS_CAMERA_SKEW,)
        if maximum is not None and maximum > policy.sync_skew_threshold_ns
        else ()
    )
    return CrossCameraSkewReport(
        samples=tuple(samples),
        incomplete_target_count=incomplete_target_count,
        p50_ns=_percentile(ordered_skews, 50),
        p95_ns=_percentile(ordered_skews, 95),
        max_ns=maximum,
        threshold_ns=policy.sync_skew_threshold_ns,
        flags=flags,
    )


def _nearest_timestamp(timestamps: tuple[int, ...], target_ns: int) -> int | None:
    if not timestamps:
        return None
    position = bisect_left(timestamps, target_ns)
    candidates = timestamps[max(0, position - 1) : min(len(timestamps), position + 1)]
    return min(candidates, key=lambda value: (abs(value - target_ns), value))


def _percentile(values: tuple[int, ...], percentile: int) -> int | None:
    if not values:
        return None
    return values[percentile * (len(values) - 1) // 100]


def _skew_triggers(
    report: CrossCameraSkewReport,
) -> tuple[QualityTriggerProvenance, ...]:
    return tuple(
        QualityTriggerProvenance(
            camera_id=camera_id,
            trigger_timestamp_ns=sample.target_ns,
            source=QualityTriggerSource.CROSS_CAMERA_SYNC,
            flag=LocalQualityFlag.OBSERVED_CROSS_CAMERA_SKEW,
        )
        for sample in report.samples
        if sample.skew_ns > report.threshold_ns
        for camera_id in CAMERA_IDS
    )


def _neighbor_projection(plan: SupplementalNeighborTargetPlan) -> dict[str, object]:
    return {
        "policy_version": plan.policy_version,
        "interval": {
            "start_ns": str(plan.interval.start_ns),
            "end_ns": str(plan.interval.end_ns),
        },
        "candidate_count": plan.candidate_count,
        "clipped_count": plan.clipped_count,
        "deduplicated_count": plan.deduplicated_count,
        "dropped_by_per_camera_budget": plan.dropped_by_per_camera_budget,
        "dropped_by_total_budget": plan.dropped_by_total_budget,
        "targets": [
            {
                "camera_id": target.camera_id.value,
                "target_ns": str(target.target_ns),
                "provenance": [
                    {
                        "camera_id": item.trigger.camera_id.value,
                        "trigger_timestamp_ns": str(item.trigger.trigger_timestamp_ns),
                        "source": item.trigger.source.value,
                        "flag": item.trigger.flag.value,
                        "packet_index": item.trigger.packet_index,
                        "offset_ns": str(item.offset_ns),
                        "requested_target_ns": str(item.requested_target_ns),
                        "clipped": item.clipped,
                    }
                    for item in target.provenance
                ],
            }
            for target in plan.targets
        ],
    }


def _report_projection(report: LocalMediaQualityReport) -> dict[str, object]:
    return {
        "policy_version": report.policy_version,
        "requested_max_duration_ns": (
            None
            if report.requested_max_duration_ns is None
            else str(report.requested_max_duration_ns)
        ),
        "recording_duration_ns": str(report.recording_duration_ns),
        "requested_interval": {
            "start_ns": str(report.requested_interval.start_ns),
            "end_ns": str(report.requested_interval.end_ns),
        },
        "window_limited": report.window_limited,
        "camera_ledgers": [
            {
                "camera_id": ledger.camera_id.value,
                "timing_count": ledger.timing_count,
                "decoded_observations": [
                    {
                        "packet_index": observation.packet_index,
                        "aligned_timestamp_ns": str(observation.aligned_timestamp_ns),
                        "source_timestamp_ns": str(observation.source_timestamp_ns),
                        "grayscale_sha256": observation.grayscale_sha256,
                        "mean_luma_milli": observation.mean_luma_milli,
                        "black_fraction_ppm": observation.black_fraction_ppm,
                        "overexposed_fraction_ppm": observation.overexposed_fraction_ppm,
                        "edge_energy_milli": observation.edge_energy_milli,
                        "frame_delta_milli": observation.frame_delta_milli,
                        "flags": [flag.value for flag in observation.flags],
                    }
                    for observation in ledger.decoded_observations
                ],
                "cadence_gaps": [
                    {
                        "previous_packet_index": gap.previous_packet_index,
                        "packet_index": gap.packet_index,
                        "previous_timestamp_ns": str(gap.previous_timestamp_ns),
                        "timestamp_ns": str(gap.timestamp_ns),
                        "delta_ns": str(gap.delta_ns),
                        "expected_delta_ns": str(gap.expected_delta_ns),
                    }
                    for gap in ledger.cadence_gaps
                ],
                "sequence_gaps": [
                    {
                        "previous_packet_index": gap.previous_packet_index,
                        "packet_index": gap.packet_index,
                        "previous_sequence": gap.previous_sequence,
                        "sequence": gap.sequence,
                        "timestamp_ns": str(gap.timestamp_ns),
                    }
                    for gap in ledger.sequence_gaps
                ],
                "flags": [flag.value for flag in ledger.flags],
            }
            for ledger in report.camera_ledgers
        ],
        "cross_camera_skew": {
            "samples": [
                {
                    "target_ns": str(sample.target_ns),
                    "camera_timestamps_ns": [
                        [camera_id.value, str(timestamp_ns)]
                        for camera_id, timestamp_ns in sample.camera_timestamps_ns
                    ],
                    "skew_ns": str(sample.skew_ns),
                }
                for sample in report.cross_camera_skew.samples
            ],
            "incomplete_target_count": report.cross_camera_skew.incomplete_target_count,
            "p50_ns": (
                None
                if report.cross_camera_skew.p50_ns is None
                else str(report.cross_camera_skew.p50_ns)
            ),
            "p95_ns": (
                None
                if report.cross_camera_skew.p95_ns is None
                else str(report.cross_camera_skew.p95_ns)
            ),
            "max_ns": (
                None
                if report.cross_camera_skew.max_ns is None
                else str(report.cross_camera_skew.max_ns)
            ),
            "threshold_ns": str(report.cross_camera_skew.threshold_ns),
            "flags": [flag.value for flag in report.cross_camera_skew.flags],
        },
        "supplemental_targets_semantic_sha256": report.supplemental_targets.semantic_sha256,
    }


def local_media_quality_report_document(
    report: LocalMediaQualityReport,
    *,
    schema_ref: SchemaRef,
) -> dict[str, object]:
    """Return an exact-pinned local audit document with its semantic digest."""

    checked_ref = SchemaRef.model_validate(schema_ref.model_dump(mode="python"), strict=True)
    if (
        checked_ref.schema_id != LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID
        or checked_ref.version != LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            "schema_ref must identify "
            f"{LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID}@"
            f"{LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION}"
        )

    projection = _report_projection(report)
    if semantic_sha256(projection) != report.semantic_sha256:
        raise ValueError("media quality report semantic digest is inconsistent")
    neighbor_projection = _neighbor_projection(report.supplemental_targets)
    if semantic_sha256(neighbor_projection) != report.supplemental_targets.semantic_sha256:
        raise ValueError("supplemental neighbor target plan semantic digest is inconsistent")
    return {
        "schema_version": LOCAL_MEDIA_QUALITY_REPORT_WIRE_VERSION,
        "schema_ref": checked_ref.model_dump(mode="json"),
        "format_version": LOCAL_MEDIA_QUALITY_REPORT_FORMAT_VERSION,
        **projection,
        "supplemental_targets": {
            **neighbor_projection,
            "semantic_sha256": report.supplemental_targets.semantic_sha256,
        },
        "semantic_sha256": report.semantic_sha256,
    }


def registered_local_media_quality_report_document(
    report: LocalMediaQualityReport,
    registry: SchemaRegistry,
) -> dict[str, object]:
    """Resolve the catalog pin and validate the complete document before publication."""

    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")
    registered = registry.resolve_version(
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID,
        LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
    )
    document = local_media_quality_report_document(report, schema_ref=registered.ref)
    validate_registered_local_media_quality_report_document(document, registry)
    return document


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON object key: {key}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def load_registered_local_media_quality_report_document(
    path: Path,
    registry: SchemaRegistry,
) -> dict[str, object]:
    """Read one exact canonical report and validate its embedded registry pin."""

    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")
    raw = path.read_bytes()
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid media quality report JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("media quality report root must be an object")
    if canonical_json_bytes(parsed) != raw:
        raise ValueError("media quality report bytes are not exact canonical JSON")
    document: dict[str, object] = parsed
    validate_registered_local_media_quality_report_document(document, registry)
    return document


def validate_registered_local_media_quality_report_document(
    document: Mapping[str, object],
    registry: SchemaRegistry,
) -> Mapping[str, object]:
    """Validate a persisted report against the exact pin embedded in its bytes."""

    if not isinstance(registry, SchemaRegistry):
        raise TypeError("registry must be a SchemaRegistry")
    schema_ref = SchemaRef.model_validate(document.get("schema_ref"), strict=True)
    if (
        schema_ref.schema_id != LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID
        or schema_ref.version != LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION
    ):
        raise ValueError(
            "schema_ref must identify "
            f"{LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID}@"
            f"{LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION}"
        )
    registry.validate_pinned(schema_ref, document)
    supplemental = document["supplemental_targets"]
    if not isinstance(supplemental, Mapping):
        raise ValueError("supplemental_targets must be an object")
    neighbor_projection = dict(supplemental)
    neighbor_digest = neighbor_projection.pop("semantic_sha256")
    if semantic_sha256(neighbor_projection) != neighbor_digest:
        raise ValueError("supplemental neighbor target plan semantic digest is inconsistent")
    if document["supplemental_targets_semantic_sha256"] != neighbor_digest:
        raise ValueError("supplemental neighbor target digest reference is inconsistent")
    report_projection = {
        key: document[key]
        for key in (
            "policy_version",
            "requested_max_duration_ns",
            "recording_duration_ns",
            "requested_interval",
            "window_limited",
            "camera_ledgers",
            "cross_camera_skew",
            "supplemental_targets_semantic_sha256",
        )
    }
    if semantic_sha256(report_projection) != document["semantic_sha256"]:
        raise ValueError("media quality report semantic digest is inconsistent")
    return document


__all__ = [
    "DEFAULT_MEDIA_QUALITY_POLICY",
    "DEFAULT_NEIGHBOR_TARGET_POLICY",
    "LOCAL_MEDIA_QUALITY_POLICY_VERSION",
    "LOCAL_MEDIA_QUALITY_REPORT_FORMAT_VERSION",
    "LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_ID",
    "LOCAL_MEDIA_QUALITY_REPORT_SCHEMA_VERSION",
    "LOCAL_MEDIA_QUALITY_REPORT_WIRE_VERSION",
    "LOCAL_NEIGHBOR_TARGET_POLICY_VERSION",
    "CameraMediaQualityLedger",
    "CrossCameraSkewReport",
    "DecodedFrameView",
    "FrameQualityObservation",
    "FrameTimingEvidence",
    "LocalFrameQualityAnalyzer",
    "LocalMediaQualityPolicy",
    "LocalMediaQualityReport",
    "LocalQualityFlag",
    "NeighborTargetPolicy",
    "QualityTriggerProvenance",
    "QualityTriggerSource",
    "SupplementalNeighborTarget",
    "SupplementalNeighborTargetPlan",
    "build_local_media_quality_report",
    "decoded_frame_view_from_pyav",
    "load_registered_local_media_quality_report_document",
    "local_media_quality_report_document",
    "plan_neighbor_targets",
    "pyav_decoded_frame_view",
    "registered_local_media_quality_report_document",
    "validate_registered_local_media_quality_report_document",
]
