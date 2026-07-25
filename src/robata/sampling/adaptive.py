"""Deterministic adaptive detection, rate reduction, and frozen target resolution.

The runtime sampler reduces registered detector triggers to a bounded window-average
rate. The exact resolver separately consumes frozen policy decisions and produces
identity-bound integer-nanosecond targets; runtime observations never bypass that
publication boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from itertools import islice, pairwise
from math import gcd
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.artifacts import ArtifactId
from robata.contracts.cameras import CAMERA_IDS, CameraId
from robata.contracts.common import (
    NanosecondInterval,
    Nanoseconds,
    SchemaVersion,
    Sha256Digest,
    StrictModel,
)
from robata.contracts.hashing import semantic_sha256 as compute_semantic_sha256
from robata.contracts.schema_registry import SchemaRef
from robata.ports.decoded_frame import DecodedFrameView
from robata.sampling.grid import SamplingGrid, SamplingRate

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeFiniteFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
PositiveFiniteFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]

ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION: Literal["adaptive-target-plan-semantic-v1"] = (
    "adaptive-target-plan-semantic-v1"
)


class AdaptiveSignal(StrEnum):
    """Lightweight per-camera signal types consumed by adaptive sampling."""

    MOTION_ENERGY = "MOTION_ENERGY"
    SCENE_CHANGE = "SCENE_CHANGE"
    BLUR_CHANGE = "BLUR_CHANGE"
    OCCLUSION_CHANGE = "OCCLUSION_CHANGE"
    HAND_PRESENCE = "HAND_PRESENCE"
    HAND_OBJECT_DISTANCE = "HAND_OBJECT_DISTANCE"
    OBJECT_MOTION = "OBJECT_MOTION"


class SignalTrigger(StrictModel):
    """One concrete trigger produced by a :class:`SignalDetector`."""

    signal_type: AdaptiveSignal
    timestamp_ns: Nanoseconds
    strength: NonNegativeFiniteFloat
    confidence: UnitInterval
    camera_id: CameraId | None = None


class AdaptiveSamplingPolicy(StrictModel):
    """Versioned policy that maps signal ranges to min/max FPS and hysteresis."""

    version: SchemaVersion
    min_fps: PositiveFiniteFloat
    max_fps: PositiveFiniteFloat
    triggers: tuple[AdaptiveSignal, ...]
    hysteresis_sec: NonNegativeFiniteFloat

    def model_post_init(self, __context: object) -> None:
        if self.min_fps > self.max_fps:
            raise ValueError("min_fps must not exceed max_fps")


class AdaptiveSamplingResult(StrictModel):
    """Immutable outcome of adaptive sampling for one temporal window."""

    window_id: NonEmptyString
    actual_fps: NonNegativeFiniteFloat
    trigger_count: NonNegativeInt
    trigger_features: tuple[SignalTrigger, ...]


class AdaptiveUpgradeReason(StrEnum):
    """The runtime fact that requires additional evidence around a timestamp."""

    SOURCE_QUALITY_SIGNAL = "SOURCE_QUALITY_SIGNAL"
    COARSE_UNCERTAINTY = "COARSE_UNCERTAINTY"
    CROSS_CAMERA_DISAGREEMENT = "CROSS_CAMERA_DISAGREEMENT"
    EVENT_CANDIDATE = "EVENT_CANDIDATE"
    BOUNDARY_REFINEMENT = "BOUNDARY_REFINEMENT"


class AdaptiveUpgradeTargetRole(StrEnum):
    """The relationship between an upgrade target and its original trigger."""

    TRIGGER = "TRIGGER"
    PRE_CONTEXT = "PRE_CONTEXT"
    POST_CONTEXT = "POST_CONTEXT"


class AdaptiveCoveragePolicy(StrictModel):
    """Versioned bounded coverage and contextual-upgrade policy for one clip."""

    version: SchemaVersion
    base_rate_num: PositiveInt
    base_rate_den: PositiveInt = 1
    base_target_budget_per_camera: PositiveInt
    grid_origin_ns: Nanoseconds = 0
    max_context_offsets: PositiveInt = 16
    context_offsets_ns: tuple[Nanoseconds, ...] = (-500_000_000, 500_000_000)
    max_upgrade_requests: PositiveInt = 1_024
    max_targets_per_camera: PositiveInt
    max_targets_total: PositiveInt

    @model_validator(mode="after")
    def validate_coverage_budget(self) -> Self:
        if gcd(self.base_rate_num, self.base_rate_den) != 1:
            raise ValueError("base sampling rate must be a reduced rational")
        if self.base_target_budget_per_camera > self.max_targets_per_camera:
            raise ValueError("base_target_budget_per_camera must not exceed max_targets_per_camera")
        if self.max_targets_total < len(CAMERA_IDS) * self.base_target_budget_per_camera:
            raise ValueError("max_targets_total cannot reserve the base budget for every camera")
        if not self.context_offsets_ns:
            raise ValueError("context_offsets_ns must include pre and post context")
        if len(self.context_offsets_ns) > self.max_context_offsets:
            raise ValueError("context_offsets_ns exceeds max_context_offsets")
        if tuple(sorted(self.context_offsets_ns)) != self.context_offsets_ns:
            raise ValueError("context_offsets_ns must be strictly increasing")
        if len(set(self.context_offsets_ns)) != len(self.context_offsets_ns):
            raise ValueError("context_offsets_ns must be strictly increasing")
        if 0 in self.context_offsets_ns:
            raise ValueError("context_offsets_ns must not include the trigger offset")
        if not any(offset < 0 for offset in self.context_offsets_ns):
            raise ValueError("context_offsets_ns must include pre-trigger context")
        if not any(offset > 0 for offset in self.context_offsets_ns):
            raise ValueError("context_offsets_ns must include post-trigger context")
        return self


class AdaptiveUpgradeRequest(StrictModel):
    """One exact source observation or semantic fact that merits an upgrade."""

    camera_id: CameraId
    trigger_timestamp_ns: Nanoseconds
    reason: AdaptiveUpgradeReason


class AdaptiveUpgradeProvenance(StrictModel):
    """Auditable provenance retained for every trigger and neighbouring target."""

    reason: AdaptiveUpgradeReason
    trigger_timestamp_ns: Nanoseconds
    role: AdaptiveUpgradeTargetRole
    context_offset_ns: Nanoseconds = 0
    context_clipped: bool = False

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if self.role is AdaptiveUpgradeTargetRole.TRIGGER:
            if self.context_offset_ns != 0 or self.context_clipped:
                raise ValueError("trigger provenance cannot carry contextual clipping")
        elif self.role is AdaptiveUpgradeTargetRole.PRE_CONTEXT:
            if self.context_offset_ns >= 0:
                raise ValueError("pre-context provenance requires a negative offset")
        elif self.context_offset_ns <= 0:
            raise ValueError("post-context provenance requires a positive offset")
        return self


def _upgrade_provenance_sort_key(
    provenance: AdaptiveUpgradeProvenance,
) -> tuple[str, int, int, str, bool]:
    return (
        provenance.reason.value,
        provenance.trigger_timestamp_ns,
        provenance.context_offset_ns,
        provenance.role.value,
        provenance.context_clipped,
    )


class AdaptiveCoverageTarget(StrictModel):
    """One exact target retained for base coverage or an explicit upgrade."""

    camera_id: CameraId
    target_ns: Nanoseconds
    base_coverage: bool
    upgrade_provenance: tuple[AdaptiveUpgradeProvenance, ...] = ()

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        canonical = tuple(sorted(self.upgrade_provenance, key=_upgrade_provenance_sort_key))
        if self.upgrade_provenance != canonical:
            raise ValueError("upgrade_provenance must be in canonical order")
        if any(left == right for left, right in pairwise(self.upgrade_provenance)):
            raise ValueError("upgrade_provenance must not contain duplicates")
        if not self.base_coverage and not self.upgrade_provenance:
            raise ValueError("non-base targets require upgrade provenance")
        return self


class AdaptiveCoveragePlan(StrictModel):
    """Bounded, deterministic per-camera targets with retained upgrade provenance."""

    policy_version: SchemaVersion
    interval: NanosecondInterval
    targets: tuple[AdaptiveCoverageTarget, ...]
    base_target_count: NonNegativeInt
    upgrade_coordinate_count: NonNegativeInt
    upgrade_targets_added: NonNegativeInt
    upgrade_coordinates_deduplicated_into_base: NonNegativeInt
    dropped_by_per_camera_budget: NonNegativeInt
    dropped_by_total_budget: NonNegativeInt

    @model_validator(mode="after")
    def validate_canonical_plan(self) -> Self:
        canonical = tuple(
            sorted(
                self.targets,
                key=lambda target: (target.camera_id.value, target.target_ns),
            )
        )
        if self.targets != canonical:
            raise ValueError("coverage targets must be ordered by camera and timestamp")
        coordinates = tuple((target.camera_id, target.target_ns) for target in self.targets)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("coverage targets must have unique camera/timestamp coordinates")
        if any(not self.interval.contains(target.target_ns) for target in self.targets):
            raise ValueError("coverage targets must lie inside the interval")

        base_targets = tuple(target for target in self.targets if target.base_coverage)
        if self.base_target_count != len(base_targets):
            raise ValueError("base_target_count does not match coverage targets")
        base_cameras = {target.camera_id for target in base_targets}
        if base_cameras != set(CAMERA_IDS):
            raise ValueError("every canonical camera requires base coverage")

        added = sum(
            1 for target in self.targets if not target.base_coverage and target.upgrade_provenance
        )
        deduplicated = sum(
            1 for target in self.targets if target.base_coverage and target.upgrade_provenance
        )
        if self.upgrade_targets_added != added:
            raise ValueError("upgrade_targets_added does not match coverage targets")
        if self.upgrade_coordinates_deduplicated_into_base != deduplicated:
            raise ValueError(
                "upgrade_coordinates_deduplicated_into_base does not match coverage targets"
            )
        expected_coordinates = (
            added + deduplicated + self.dropped_by_per_camera_budget + self.dropped_by_total_budget
        )
        if self.upgrade_coordinate_count != expected_coordinates:
            raise ValueError("upgrade counters do not account for every upgrade coordinate")
        return self


class AdaptiveCoveragePlanner:
    """Produce deterministic base coverage plus bounded, contextual upgrades."""

    def __init__(self, policy: AdaptiveCoveragePolicy) -> None:
        if not isinstance(policy, AdaptiveCoveragePolicy):
            raise TypeError("policy must be an AdaptiveCoveragePolicy")
        self._policy = policy

    @property
    def policy(self) -> AdaptiveCoveragePolicy:
        """The immutable policy used to build coverage plans."""

        return self._policy

    def plan(
        self,
        interval: NanosecondInterval,
        upgrades: Iterable[AdaptiveUpgradeRequest] = (),
    ) -> AdaptiveCoveragePlan:
        """Return bounded targets without dropping base coverage or source observations."""

        if not isinstance(interval, NanosecondInterval):
            raise TypeError("interval must be a NanosecondInterval")

        requests = tuple(islice(upgrades, self._policy.max_upgrade_requests + 1))
        if len(requests) > self._policy.max_upgrade_requests:
            raise ValueError("upgrade requests exceed max_upgrade_requests")
        for request in requests:
            if not isinstance(request, AdaptiveUpgradeRequest):
                raise TypeError("upgrades must contain AdaptiveUpgradeRequest values")
            if not interval.contains(request.trigger_timestamp_ns):
                raise ValueError("upgrade trigger timestamps must lie inside the interval")

        base_by_camera = self._base_targets(interval)
        base_coordinates = {
            (camera_id, target_ns)
            for camera_id, targets in base_by_camera.items()
            for target_ns in targets
        }
        base_target_count = len(base_coordinates)
        if base_target_count > self._policy.max_targets_total:
            raise ValueError("base coverage exceeds max_targets_total")

        provenance_by_coordinate = self._expanded_upgrade_provenance(
            interval=interval,
            requests=requests,
        )
        base_upgrade_coordinates = {
            coordinate for coordinate in provenance_by_coordinate if coordinate in base_coordinates
        }

        candidates_by_camera: dict[CameraId, list[tuple[CameraId, int]]] = {
            camera_id: [] for camera_id in CAMERA_IDS
        }
        for coordinate in provenance_by_coordinate:
            if coordinate not in base_coordinates:
                candidates_by_camera[coordinate[0]].append(coordinate)

        retained_after_camera_budget: list[tuple[CameraId, int]] = []
        dropped_by_per_camera_budget = 0
        for camera_id in CAMERA_IDS:
            candidates = sorted(
                candidates_by_camera[camera_id],
                key=lambda coordinate: _upgrade_coordinate_sort_key(
                    coordinate,
                    provenance_by_coordinate[coordinate],
                ),
            )
            capacity = self._policy.max_targets_per_camera - len(base_by_camera[camera_id])
            trigger_count = sum(
                _coordinate_has_trigger(provenance_by_coordinate[coordinate])
                for coordinate in candidates
            )
            if trigger_count > capacity:
                raise ValueError("per-camera budget cannot preserve every original upgrade trigger")
            retained_after_camera_budget.extend(candidates[:capacity])
            dropped_by_per_camera_budget += len(candidates[capacity:])

        capacity_total = self._policy.max_targets_total - base_target_count
        trigger_count = sum(
            _coordinate_has_trigger(provenance_by_coordinate[coordinate])
            for coordinate in retained_after_camera_budget
        )
        if trigger_count > capacity_total:
            raise ValueError("total budget cannot preserve every original upgrade trigger")
        retained_after_total_budget = tuple(
            sorted(
                retained_after_camera_budget,
                key=lambda coordinate: _upgrade_coordinate_sort_key(
                    coordinate,
                    provenance_by_coordinate[coordinate],
                ),
            )[:capacity_total]
        )
        dropped_by_total_budget = len(retained_after_camera_budget) - len(
            retained_after_total_budget
        )
        retained_coordinates = set(retained_after_total_budget)

        targets: list[AdaptiveCoverageTarget] = []
        for camera_id in CAMERA_IDS:
            for target_ns in base_by_camera[camera_id]:
                coordinate = (camera_id, target_ns)
                targets.append(
                    AdaptiveCoverageTarget(
                        camera_id=camera_id,
                        target_ns=target_ns,
                        base_coverage=True,
                        upgrade_provenance=provenance_by_coordinate.get(coordinate, ()),
                    )
                )
            for coordinate in sorted(
                (coordinate for coordinate in retained_coordinates if coordinate[0] is camera_id),
                key=lambda item: item[1],
            ):
                targets.append(
                    AdaptiveCoverageTarget(
                        camera_id=camera_id,
                        target_ns=coordinate[1],
                        base_coverage=False,
                        upgrade_provenance=provenance_by_coordinate[coordinate],
                    )
                )

        return AdaptiveCoveragePlan(
            policy_version=self._policy.version,
            interval=interval,
            targets=tuple(
                sorted(targets, key=lambda target: (target.camera_id.value, target.target_ns))
            ),
            base_target_count=base_target_count,
            upgrade_coordinate_count=len(provenance_by_coordinate),
            upgrade_targets_added=len(retained_coordinates),
            upgrade_coordinates_deduplicated_into_base=len(base_upgrade_coordinates),
            dropped_by_per_camera_budget=dropped_by_per_camera_budget,
            dropped_by_total_budget=dropped_by_total_budget,
        )

    def _base_targets(
        self,
        interval: NanosecondInterval,
    ) -> dict[CameraId, tuple[int, ...]]:
        grid = SamplingGrid(
            grid_origin_ns=self._policy.grid_origin_ns,
            rate=SamplingRate(self._policy.base_rate_num, self._policy.base_rate_den),
        )
        bounded = _bounded_grid_targets(
            grid=grid,
            interval=interval,
            budget=self._policy.base_target_budget_per_camera,
        )
        return {camera_id: bounded for camera_id in CAMERA_IDS}

    def _expanded_upgrade_provenance(
        self,
        *,
        interval: NanosecondInterval,
        requests: tuple[AdaptiveUpgradeRequest, ...],
    ) -> dict[tuple[CameraId, int], tuple[AdaptiveUpgradeProvenance, ...]]:
        provisional: dict[tuple[CameraId, int], list[AdaptiveUpgradeProvenance]] = {}
        for request in requests:
            trigger = AdaptiveUpgradeProvenance(
                reason=request.reason,
                trigger_timestamp_ns=request.trigger_timestamp_ns,
                role=AdaptiveUpgradeTargetRole.TRIGGER,
            )
            provisional.setdefault(
                (request.camera_id, request.trigger_timestamp_ns),
                [],
            ).append(trigger)
            for offset_ns in self._policy.context_offsets_ns:
                requested_target_ns = request.trigger_timestamp_ns + offset_ns
                target_ns = min(
                    max(requested_target_ns, interval.start_ns),
                    interval.end_ns - 1,
                )
                role = (
                    AdaptiveUpgradeTargetRole.PRE_CONTEXT
                    if offset_ns < 0
                    else AdaptiveUpgradeTargetRole.POST_CONTEXT
                )
                provisional.setdefault((request.camera_id, target_ns), []).append(
                    AdaptiveUpgradeProvenance(
                        reason=request.reason,
                        trigger_timestamp_ns=request.trigger_timestamp_ns,
                        role=role,
                        context_offset_ns=offset_ns,
                        context_clipped=target_ns != requested_target_ns,
                    )
                )
        return {
            coordinate: tuple(
                sorted(
                    set(provenance),
                    key=_upgrade_provenance_sort_key,
                )
            )
            for coordinate, provenance in provisional.items()
        }


def _bounded_grid_targets(
    *,
    grid: SamplingGrid,
    interval: NanosecondInterval,
    budget: int,
) -> tuple[int, ...]:
    """Select evenly spread grid coordinates without enumerating an unbounded clip.

    SamplingGrid already owns the half-even inverse used to locate the first grid index
    inside an interval. Choosing bounded positions in that index range preserves the
    persisted grid phase while preventing an extreme rate or a long clip from expanding
    the planner's memory footprint.
    """

    first_k = grid._first_k_at_or_after(interval.start_ns)
    stop_k = grid._first_k_at_or_after(interval.end_ns)
    if first_k >= stop_k:
        return (interval.start_ns,)

    last_k = stop_k - 1
    if grid.period_num_ns < grid.period_den:
        first_target_ns = grid.target_ns(first_k)
        last_target_ns = grid.target_ns(last_k)
        target_count = min(budget, last_target_ns - first_target_ns + 1)
        if target_count <= 1:
            return ((first_target_ns + last_target_ns) // 2,)
        return tuple(
            first_target_ns
            + (ordinal * (last_target_ns - first_target_ns)) // (target_count - 1)
            for ordinal in range(target_count)
        )

    target_count = min(budget, last_k - first_k + 1)
    if target_count <= 1:
        target_ns = grid.target_ns(first_k + (last_k - first_k) // 2)
        return (target_ns,)
    return tuple(
        grid.target_ns(first_k + (ordinal * (last_k - first_k)) // (target_count - 1))
        for ordinal in range(target_count)
    )


def _coordinate_has_trigger(
    provenance: tuple[AdaptiveUpgradeProvenance, ...],
) -> bool:
    return any(item.role is AdaptiveUpgradeTargetRole.TRIGGER for item in provenance)


def _upgrade_coordinate_sort_key(
    coordinate: tuple[CameraId, int],
    provenance: tuple[AdaptiveUpgradeProvenance, ...],
) -> tuple[int, int, str, int, str]:
    """Keep original observations before context without inventing severity ordering."""

    role_rank = min(
        0 if item.role is AdaptiveUpgradeTargetRole.TRIGGER else 1 for item in provenance
    )
    return (
        role_rank,
        min(item.trigger_timestamp_ns for item in provenance),
        coordinate[0].value,
        coordinate[1],
        min(item.reason.value for item in provenance),
    )


def plan_adaptive_coverage(
    policy: AdaptiveCoveragePolicy,
    interval: NanosecondInterval,
    upgrades: Iterable[AdaptiveUpgradeRequest] = (),
) -> AdaptiveCoveragePlan:
    """Convenience entry point for the pure bounded coverage planner."""

    return AdaptiveCoveragePlanner(policy).plan(interval, upgrades)


class AdaptiveResolutionMode(StrEnum):
    """The frozen coordinate form supplied to the exact resolver."""

    GRID_SEGMENTS = "GRID_SEGMENTS"
    EXPLICIT_TARGETS = "EXPLICIT_TARGETS"


class FrozenAdaptiveTriggerArtifactRef(StrictModel):
    """Exact and semantic identity of one immutable trigger artifact."""

    artifact_id: ArtifactId
    schema_ref: SchemaRef
    exact_bytes_sha256: Sha256Digest
    semantic_sha256: Sha256Digest


class CanonicalAdaptiveGridSegment(StrictModel):
    """One ordered interval with an already-selected exact rational rate."""

    interval: NanosecondInterval
    rate_num: PositiveInt
    rate_den: PositiveInt

    @model_validator(mode="after")
    def validate_reduced_rate(self) -> Self:
        if gcd(self.rate_num, self.rate_den) != 1:
            raise ValueError("adaptive segment rate must be a reduced rational")
        return self


class FrozenAdaptiveResolutionRequest(StrictModel):
    """A frozen, policy-resolved input to exact adaptive target enumeration."""

    trigger_artifact: FrozenAdaptiveTriggerArtifactRef
    camera_id: CameraId
    effective_interval: NanosecondInterval
    grid_origin_ns: Nanoseconds
    resolution_mode: AdaptiveResolutionMode
    segments: tuple[CanonicalAdaptiveGridSegment, ...] = ()
    explicit_targets_ns: tuple[Nanoseconds, ...] = ()
    max_target_count: NonNegativeInt
    resolution_policy_version: SchemaVersion
    semantic_projection_version: Literal["adaptive-target-plan-semantic-v1"] = (
        ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION
    )

    @model_validator(mode="after")
    def validate_frozen_coordinates(self) -> Self:
        _validate_frozen_coordinates(
            effective_interval=self.effective_interval,
            resolution_mode=self.resolution_mode,
            segments=self.segments,
            explicit_targets_ns=self.explicit_targets_ns,
        )
        return self


class ResolvedAdaptiveTarget(StrictModel):
    """One canonical integer-nanosecond adaptive target with grid provenance."""

    ordinal: NonNegativeInt
    target_ns: Nanoseconds
    segment_ordinal: NonNegativeInt | None = None
    grid_k: int | None = None

    @model_validator(mode="after")
    def validate_grid_provenance(self) -> Self:
        if (self.segment_ordinal is None) != (self.grid_k is None):
            raise ValueError("segment_ordinal and grid_k must be supplied together")
        return self


class ResolvedAdaptivePlan(StrictModel):
    """Deterministic target plan derived from one frozen adaptive input."""

    trigger_artifact: FrozenAdaptiveTriggerArtifactRef
    camera_id: CameraId
    effective_interval: NanosecondInterval
    grid_origin_ns: Nanoseconds
    resolution_mode: AdaptiveResolutionMode
    segments: tuple[CanonicalAdaptiveGridSegment, ...] = ()
    explicit_targets_ns: tuple[Nanoseconds, ...] = ()
    targets: tuple[ResolvedAdaptiveTarget, ...]
    max_target_count: NonNegativeInt
    resolution_policy_version: SchemaVersion
    semantic_projection_version: Literal["adaptive-target-plan-semantic-v1"] = (
        ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION
    )
    semantic_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_resolved_plan(self) -> Self:
        _validate_frozen_coordinates(
            effective_interval=self.effective_interval,
            resolution_mode=self.resolution_mode,
            segments=self.segments,
            explicit_targets_ns=self.explicit_targets_ns,
        )

        expected_targets = _resolve_canonical_targets(
            grid_origin_ns=self.grid_origin_ns,
            resolution_mode=self.resolution_mode,
            segments=self.segments,
            explicit_targets_ns=self.explicit_targets_ns,
            max_target_count=self.max_target_count,
        )
        if self.targets != expected_targets:
            raise ValueError("resolved targets do not match the complete canonical target set")

        expected_digest = compute_semantic_sha256(adaptive_target_plan_semantic_projection(self))
        if self.semantic_sha256 != expected_digest:
            raise ValueError("semantic_sha256 does not match the adaptive target plan projection")
        return self


def _validate_frozen_coordinates(
    *,
    effective_interval: NanosecondInterval,
    resolution_mode: AdaptiveResolutionMode,
    segments: tuple[CanonicalAdaptiveGridSegment, ...],
    explicit_targets_ns: tuple[int, ...],
) -> None:
    if resolution_mode is AdaptiveResolutionMode.GRID_SEGMENTS:
        if not segments:
            raise ValueError("GRID_SEGMENTS resolution requires at least one segment")
        if explicit_targets_ns:
            raise ValueError("GRID_SEGMENTS resolution cannot contain explicit targets")

        previous_end_ns: int | None = None
        for segment in segments:
            if segment.interval.start_ns < effective_interval.start_ns:
                raise ValueError("adaptive segments must lie inside effective_interval")
            if segment.interval.end_ns > effective_interval.end_ns:
                raise ValueError("adaptive segments must lie inside effective_interval")
            if previous_end_ns is not None and segment.interval.start_ns < previous_end_ns:
                raise ValueError("adaptive segments must be ordered and non-overlapping")
            previous_end_ns = segment.interval.end_ns
        return

    if segments:
        raise ValueError("EXPLICIT_TARGETS resolution cannot contain grid segments")
    if any(left >= right for left, right in pairwise(explicit_targets_ns)):
        raise ValueError("explicit targets must be strictly increasing")
    if any(not effective_interval.contains(target_ns) for target_ns in explicit_targets_ns):
        raise ValueError("explicit targets must lie inside effective_interval")


def _resolve_canonical_targets(
    *,
    grid_origin_ns: int,
    resolution_mode: AdaptiveResolutionMode,
    segments: tuple[CanonicalAdaptiveGridSegment, ...],
    explicit_targets_ns: tuple[int, ...],
    max_target_count: int,
) -> tuple[ResolvedAdaptiveTarget, ...]:
    if resolution_mode is AdaptiveResolutionMode.EXPLICIT_TARGETS:
        if len(explicit_targets_ns) > max_target_count:
            raise ValueError("adaptive target count exceeds max_target_count")
        return tuple(
            ResolvedAdaptiveTarget(ordinal=ordinal, target_ns=target_ns)
            for ordinal, target_ns in enumerate(explicit_targets_ns)
        )

    targets: list[ResolvedAdaptiveTarget] = []
    seen_target_ns: set[int] = set()
    for segment_ordinal, segment in enumerate(segments):
        grid = SamplingGrid(
            grid_origin_ns=grid_origin_ns,
            rate=SamplingRate(segment.rate_num, segment.rate_den),
        )
        for grid_target in grid.iter_unique_targets(
            segment.interval.start_ns,
            segment.interval.end_ns,
        ):
            if grid_target.target_ns in seen_target_ns:
                raise ValueError("adaptive grid segments resolved to a duplicate target")
            if len(targets) >= max_target_count:
                raise ValueError("adaptive target count exceeds max_target_count")
            seen_target_ns.add(grid_target.target_ns)
            targets.append(
                ResolvedAdaptiveTarget(
                    ordinal=len(targets),
                    target_ns=grid_target.target_ns,
                    segment_ordinal=segment_ordinal,
                    grid_k=grid_target.k,
                )
            )
    return tuple(targets)


def _adaptive_target_plan_projection(
    *,
    trigger_artifact: FrozenAdaptiveTriggerArtifactRef,
    camera_id: CameraId,
    effective_interval: NanosecondInterval,
    grid_origin_ns: int,
    resolution_mode: AdaptiveResolutionMode,
    segments: tuple[CanonicalAdaptiveGridSegment, ...],
    explicit_targets_ns: tuple[int, ...],
    targets: tuple[ResolvedAdaptiveTarget, ...],
    max_target_count: int,
    resolution_policy_version: str,
    semantic_projection_version: Literal["adaptive-target-plan-semantic-v1"],
) -> dict[str, object]:
    return {
        "semantic_projection_version": semantic_projection_version,
        "resolution_policy_version": resolution_policy_version,
        "trigger_artifact": {
            "schema_ref": {
                "schema_id": trigger_artifact.schema_ref.schema_id,
                "version": trigger_artifact.schema_ref.version,
            },
            "semantic_sha256": trigger_artifact.semantic_sha256,
        },
        "camera_id": camera_id.value,
        "effective_interval": {
            "start_ns": str(effective_interval.start_ns),
            "end_ns": str(effective_interval.end_ns),
        },
        "grid_origin_ns": str(grid_origin_ns),
        "resolution_mode": resolution_mode.value,
        "max_target_count": max_target_count,
        "segments": [
            {
                "start_ns": str(segment.interval.start_ns),
                "end_ns": str(segment.interval.end_ns),
                "rate_num": str(segment.rate_num),
                "rate_den": str(segment.rate_den),
            }
            for segment in segments
        ],
        "explicit_targets_ns": [str(target_ns) for target_ns in explicit_targets_ns],
        "resolved_targets": [
            {
                "ordinal": target.ordinal,
                "target_ns": str(target.target_ns),
                "segment_ordinal": target.segment_ordinal,
                "grid_k": None if target.grid_k is None else str(target.grid_k),
            }
            for target in targets
        ],
    }


def adaptive_target_plan_semantic_projection(plan: ResolvedAdaptivePlan) -> dict[str, object]:
    """Return the explicit, versioned semantic projection of a resolved plan."""

    if not isinstance(plan, ResolvedAdaptivePlan):
        raise TypeError("plan must be a ResolvedAdaptivePlan")
    return _adaptive_target_plan_projection(
        trigger_artifact=plan.trigger_artifact,
        camera_id=plan.camera_id,
        effective_interval=plan.effective_interval,
        grid_origin_ns=plan.grid_origin_ns,
        resolution_mode=plan.resolution_mode,
        segments=plan.segments,
        explicit_targets_ns=plan.explicit_targets_ns,
        targets=plan.targets,
        max_target_count=plan.max_target_count,
        resolution_policy_version=plan.resolution_policy_version,
        semantic_projection_version=plan.semantic_projection_version,
    )


def resolve_frozen_adaptive_targets(
    request: FrozenAdaptiveResolutionRequest,
) -> ResolvedAdaptivePlan:
    """Resolve frozen rational segments or exact timestamps without policy inference."""

    if not isinstance(request, FrozenAdaptiveResolutionRequest):
        raise TypeError("request must be a FrozenAdaptiveResolutionRequest")

    resolved_targets = _resolve_canonical_targets(
        grid_origin_ns=request.grid_origin_ns,
        resolution_mode=request.resolution_mode,
        segments=request.segments,
        explicit_targets_ns=request.explicit_targets_ns,
        max_target_count=request.max_target_count,
    )
    projection = _adaptive_target_plan_projection(
        trigger_artifact=request.trigger_artifact,
        camera_id=request.camera_id,
        effective_interval=request.effective_interval,
        grid_origin_ns=request.grid_origin_ns,
        resolution_mode=request.resolution_mode,
        segments=request.segments,
        explicit_targets_ns=request.explicit_targets_ns,
        targets=resolved_targets,
        max_target_count=request.max_target_count,
        resolution_policy_version=request.resolution_policy_version,
        semantic_projection_version=request.semantic_projection_version,
    )
    return ResolvedAdaptivePlan(
        trigger_artifact=request.trigger_artifact,
        camera_id=request.camera_id,
        effective_interval=request.effective_interval,
        grid_origin_ns=request.grid_origin_ns,
        resolution_mode=request.resolution_mode,
        segments=request.segments,
        explicit_targets_ns=request.explicit_targets_ns,
        targets=resolved_targets,
        max_target_count=request.max_target_count,
        resolution_policy_version=request.resolution_policy_version,
        semantic_projection_version=request.semantic_projection_version,
        semantic_sha256=compute_semantic_sha256(projection),
    )


class SignalDetector:
    """Protocol/abstract base class for per-camera signal detection.

    Concrete implementations must override :meth:`detect` and return a sequence
    of :class:`SignalTrigger` values ordered by ``timestamp_ns``.
    """

    def detect(
        self,
        frames: Iterable[DecodedFrameView],
        *,
        camera_id: str,
    ) -> Sequence[SignalTrigger]:
        """Detect signal triggers across the supplied frame sequence.

        Args:
            frames: Ordered iterable of compact decoded grayscale views for one camera.
            camera_id: Canonical camera identifier (e.g. ``cam_01``).

        Returns:
            Ordered sequence of detected triggers.
        """
        raise NotImplementedError("SignalDetector subclasses must implement detect()")


class AdaptiveSampler:
    """Run registered detectors and reduce triggers under a versioned policy."""

    def __init__(
        self,
        policy: AdaptiveSamplingPolicy,
        detectors: Sequence[SignalDetector],
    ) -> None:
        if not detectors:
            raise ValueError("at least one SignalDetector is required")
        self._policy = policy
        self._detectors = tuple(detectors)

    @property
    def policy(self) -> AdaptiveSamplingPolicy:
        """The immutable policy governing this sampler."""
        return self._policy

    @property
    def detectors(self) -> tuple[SignalDetector, ...]:
        """The ordered detectors registered with this sampler."""
        return self._detectors

    def sample(
        self,
        window: object,
        frames: dict[CameraId, Sequence[DecodedFrameView]],
    ) -> AdaptiveSamplingResult:
        """Return a deterministic window-average rate without materializing frames."""

        window_id = getattr(window, "window_id", None)
        interval = getattr(window, "interval", None)
        if not isinstance(window_id, str) or not window_id:
            raise TypeError("window must expose a nonempty window_id")
        if not isinstance(interval, NanosecondInterval):
            raise TypeError("window must expose a NanosecondInterval interval")
        if not isinstance(frames, dict) or set(frames) != set(CameraId):
            raise ValueError("frames must contain every canonical camera exactly once")

        triggers: list[SignalTrigger] = []
        enabled = set(self._policy.triggers)
        for camera_id in CameraId:
            camera_frames = frames[camera_id]
            if not isinstance(camera_frames, Sequence) or any(
                not isinstance(frame, DecodedFrameView) for frame in camera_frames
            ):
                raise TypeError("adaptive frames must be DecodedFrameView values")
            for detector in self._detectors:
                detected = detector.detect(camera_frames, camera_id=camera_id.value)
                for trigger in detected:
                    if not isinstance(trigger, SignalTrigger):
                        raise TypeError("signal detectors must return SignalTrigger values")
                    if trigger.signal_type not in enabled:
                        continue
                    if not interval.contains(trigger.timestamp_ns):
                        continue
                    if trigger.camera_id not in {None, camera_id}:
                        raise ValueError("signal trigger camera_id disagrees with detector input")
                    triggers.append(trigger.model_copy(update={"camera_id": camera_id}))

        canonical = tuple(
            sorted(
                triggers,
                key=lambda item: (
                    item.timestamp_ns,
                    item.camera_id.value if item.camera_id is not None else "",
                    item.signal_type.value,
                    item.strength,
                    item.confidence,
                ),
            )
        )
        actual_fps = _window_average_fps(
            interval=interval,
            triggers=canonical,
            policy=self._policy,
        )
        return AdaptiveSamplingResult(
            window_id=window_id,
            actual_fps=actual_fps,
            trigger_count=len(canonical),
            trigger_features=canonical,
        )


def _window_average_fps(
    *,
    interval: NanosecondInterval,
    triggers: tuple[SignalTrigger, ...],
    policy: AdaptiveSamplingPolicy,
) -> float:
    if not triggers:
        return policy.min_fps
    hysteresis_ns = int(policy.hysteresis_sec * 1_000_000_000)
    if hysteresis_ns <= 0:
        return policy.max_fps

    promoted = sorted(
        (
            trigger.timestamp_ns,
            min(interval.end_ns, trigger.timestamp_ns + hysteresis_ns),
        )
        for trigger in triggers
    )
    covered_ns = 0
    current_start, current_end = promoted[0]
    for start_ns, end_ns in promoted[1:]:
        if start_ns <= current_end:
            current_end = max(current_end, end_ns)
        else:
            covered_ns += current_end - current_start
            current_start, current_end = start_ns, end_ns
    covered_ns += current_end - current_start
    ratio = covered_ns / (interval.end_ns - interval.start_ns)
    return policy.min_fps + (policy.max_fps - policy.min_fps) * ratio


__all__ = [
    "ADAPTIVE_TARGET_PLAN_SEMANTIC_PROJECTION_VERSION",
    "AdaptiveCoveragePlan",
    "AdaptiveCoveragePlanner",
    "AdaptiveCoveragePolicy",
    "AdaptiveCoverageTarget",
    "AdaptiveResolutionMode",
    "AdaptiveSampler",
    "AdaptiveSamplingPolicy",
    "AdaptiveSamplingResult",
    "AdaptiveSignal",
    "AdaptiveUpgradeProvenance",
    "AdaptiveUpgradeReason",
    "AdaptiveUpgradeRequest",
    "AdaptiveUpgradeTargetRole",
    "CanonicalAdaptiveGridSegment",
    "FrozenAdaptiveResolutionRequest",
    "FrozenAdaptiveTriggerArtifactRef",
    "ResolvedAdaptivePlan",
    "ResolvedAdaptiveTarget",
    "SignalDetector",
    "SignalTrigger",
    "adaptive_target_plan_semantic_projection",
    "plan_adaptive_coverage",
    "resolve_frozen_adaptive_targets",
]
