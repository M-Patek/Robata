"""Provider-neutral dense temporal window planning.

The sampling layer owns source/policy frame-budget splitting.  Provider limits
are deliberately absent here; those belong to an inference input plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import gcd
from typing import Annotated, Any, Self

from pydantic import Field, StringConstraints, model_validator

from robata.contracts.cameras import CAMERA_ID_VALUES
from robata.contracts.common import NanosecondInterval, Nanoseconds, SchemaVersion, StrictModel
from robata.contracts.hashing import canonical_json_bytes
from robata.contracts.pipeline import SamplingPurpose, SamplingStrategy
from robata.contracts.sampling_plan import OverflowPolicy, SamplingPlan

NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]

NANOSECONDS_PER_SECOND = 1_000_000_000
CAMERA_COUNT = len(CAMERA_ID_VALUES)
MAX_RATE_DENOMINATOR = 1_000_000
MAX_SPLIT_PARTS = 1_000_000


class DenseSplitPolicy(StrictModel):
    """Versioned overlap policy for provider-neutral dense splitting.

    ``max_frames_*`` are retained as optional compatibility fields for the
    original skeleton.  The authoritative limits come from
    :class:`robata.contracts.sampling_plan.FrameBudget` and, when supplied,
    these values must agree with that contract.
    """

    version: SchemaVersion
    overlap_ns: Nanoseconds = 0
    max_frames_per_camera: PositiveInt | None = None
    max_frames_total: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap_ns < 0:
            raise ValueError("overlap_ns must be nonnegative")
        if (self.max_frames_per_camera is None) != (self.max_frames_total is None):
            raise ValueError("legacy frame-budget fields must be supplied together")
        return self


class TemporalWindow(StrictModel):
    """Immutable dense window or one coordinate of a split window."""

    window_id: NonEmptyString
    requested_interval: NanosecondInterval
    interval: NanosecondInterval
    ordinal: NonNegativeInt = 0
    part_count: PositiveInt = 1
    overlap_before_ns: Nanoseconds = 0
    overlap_after_ns: Nanoseconds = 0
    # Optional lineage fields let callers attach a fully identified source
    # window without making the planner depend on the shared pipeline contract.
    mcap_id: NonEmptyString | None = None
    camera_mapping_run_id: NonEmptyString | None = None
    alignment_id: NonEmptyString | None = None
    parent_window_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.interval.start_ns < self.requested_interval.start_ns:
            raise ValueError("effective interval must be contained by requested_interval")
        if self.interval.end_ns > self.requested_interval.end_ns:
            raise ValueError("effective interval must be contained by requested_interval")
        if self.ordinal >= self.part_count:
            raise ValueError("ordinal must be less than part_count")
        if self.overlap_before_ns < 0 or self.overlap_after_ns < 0:
            raise ValueError("overlap values must be nonnegative")
        if self.overlap_before_ns >= self.interval.duration_ns:
            raise ValueError("overlap_before_ns must be less than the window span")
        if self.overlap_after_ns >= self.interval.duration_ns:
            raise ValueError("overlap_after_ns must be less than the window span")
        if self.part_count == 1 and (self.overlap_before_ns or self.overlap_after_ns):
            raise ValueError("an unsplit window cannot have overlap")
        return self

    # These aliases preserve the attributes used by the original skeleton.
    @property
    def _overlap_before_ns(self) -> int:
        return self.overlap_before_ns

    @property
    def _overlap_after_ns(self) -> int:
        return self.overlap_after_ns

    @property
    def _ordinal(self) -> int:
        return self.ordinal


@dataclass(frozen=True, slots=True)
class IntervalPart:
    """A deterministic coordinate in a provider-neutral split plan."""

    requested_interval: NanosecondInterval
    effective_interval: NanosecondInterval
    ordinal: int
    part_count: int
    overlap_before_ns: int
    overlap_after_ns: int


@dataclass(frozen=True, slots=True)
class _BudgetSpec:
    rates: tuple[tuple[str, int, int], ...]
    max_frames_per_camera: int
    max_frames_total: int
    overflow_policy: OverflowPolicy
    overlap_ns: int


def _rate_fraction(value: Any) -> tuple[int, int]:
    """Convert a finite FPS value to one bounded, reduced rational."""

    if isinstance(value, bool):
        raise ValueError("sampling rates must be finite positive numbers")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sampling rates must be finite positive numbers") from exc
    if number <= 0 or number != number or number in (float("inf"), float("-inf")):
        raise ValueError("sampling rates must be finite positive numbers")
    try:
        fraction = Fraction(str(value)).limit_denominator(MAX_RATE_DENOMINATOR)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("sampling rate cannot be represented as a rational") from exc
    if fraction.numerator <= 0:
        raise ValueError("sampling rates must be finite positive numbers")
    # A nanosecond grid cannot represent two distinct targets closer than one
    # nanosecond.  Rejecting such rates is fail-closed and avoids duplicate
    # rounded targets changing the budget after identity calculation.
    if fraction.numerator > NANOSECONDS_PER_SECOND * fraction.denominator:
        raise ValueError("sampling rate exceeds one distinct target per nanosecond")
    divisor = gcd(fraction.numerator, fraction.denominator)
    return fraction.numerator // divisor, fraction.denominator // divisor


def sampling_plan_projection(
    sampling_plan: SamplingPlan,
    *,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
) -> dict[str, Any]:
    """Return the rationalized projection for one supported sampling purpose.

    The compatibility default remains `ACTION_DENSE` and retains its exact
    historical projection. QA purposes deliberately ignore action-only
    per-camera overrides: each QA pass uses one uniform grid across all six
    cameras, at either the coarse or dense global rate.
    """

    if not isinstance(sampling_plan, SamplingPlan):
        raise TypeError("sampling_plan must be robata.contracts.sampling_plan.SamplingPlan")
    if not isinstance(purpose, SamplingPurpose):
        raise TypeError("purpose must be a SamplingPurpose")
    if purpose is SamplingPurpose.QA_COARSE:
        global_rate = _rate_fraction(sampling_plan.qa_sampling_rate_fps)
        rates = {camera_id: global_rate for camera_id in CAMERA_ID_VALUES}
        rate_key = "qa_rate"
        strategy = SamplingStrategy.UNIFORM
    elif purpose is SamplingPurpose.QA_DENSE:
        global_rate = _rate_fraction(sampling_plan.dense_sampling_rate_fps)
        rates = {camera_id: global_rate for camera_id in CAMERA_ID_VALUES}
        rate_key = "dense_rate"
        strategy = SamplingStrategy.DENSE
    elif purpose in {
        SamplingPurpose.ACTION_DENSE,
        SamplingPurpose.BOUNDARY_REFINEMENT,
    }:
        overrides: dict[str, tuple[int, int]] = {}
        for override in sampling_plan.per_camera:
            camera_id = override.camera_id
            if camera_id not in CAMERA_ID_VALUES:
                raise ValueError(f"unknown camera override: {camera_id!r}")
            if camera_id in overrides:
                raise ValueError(f"duplicate camera override: {camera_id!r}")
            overrides[camera_id] = _rate_fraction(override.dense_sampling_rate_fps)
        global_rate = _rate_fraction(sampling_plan.dense_sampling_rate_fps)
        rates = {camera_id: overrides.get(camera_id, global_rate) for camera_id in CAMERA_ID_VALUES}
        rate_key = "dense_rate"
        strategy = SamplingStrategy.DENSE
    else:
        raise ValueError(
            "provider-neutral planning currently supports only QA_COARSE, "
            "QA_DENSE, ACTION_DENSE, and BOUNDARY_REFINEMENT"
        )
    budget = sampling_plan.frame_budget
    if budget.max_frames_total < CAMERA_COUNT:
        raise ValueError("max_frames_total must allow at least one frame per camera")
    projection: dict[str, Any] = {
        "version": sampling_plan.version,
        rate_key: {"numerator": global_rate[0], "denominator": global_rate[1]},
        "per_camera": {
            camera_id: {"numerator": rates[camera_id][0], "denominator": rates[camera_id][1]}
            for camera_id in CAMERA_ID_VALUES
        },
        "frame_budget": {
            "max_frames_per_camera": budget.max_frames_per_camera,
            "max_frames_total": budget.max_frames_total,
            "overflow_policy": budget.overflow_policy.value,
        },
    }
    if purpose in {
        SamplingPurpose.QA_COARSE,
        SamplingPurpose.QA_DENSE,
        SamplingPurpose.BOUNDARY_REFINEMENT,
    }:
        projection["purpose"] = purpose.value
        projection["strategy"] = strategy.value
    return projection


def _budget_spec(
    sampling_plan: SamplingPlan,
    policy: DenseSplitPolicy,
    *,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
) -> _BudgetSpec:
    projection = sampling_plan_projection(sampling_plan, purpose=purpose)
    budget = sampling_plan.frame_budget
    if policy.max_frames_per_camera is not None:
        if policy.max_frames_per_camera != budget.max_frames_per_camera:
            raise ValueError("DenseSplitPolicy and SamplingPlan per-camera budgets disagree")
        if policy.max_frames_total != budget.max_frames_total:
            raise ValueError("DenseSplitPolicy and SamplingPlan total budgets disagree")
    rates = tuple(
        (
            camera_id,
            projection["per_camera"][camera_id]["numerator"],
            projection["per_camera"][camera_id]["denominator"],
        )
        for camera_id in CAMERA_ID_VALUES
    )
    return _BudgetSpec(
        rates=rates,
        max_frames_per_camera=budget.max_frames_per_camera,
        max_frames_total=budget.max_frames_total,
        overflow_policy=budget.overflow_policy,
        overlap_ns=policy.overlap_ns,
    )


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def frame_counts_for_interval(
    interval: NanosecondInterval,
    sampling_plan: SamplingPlan,
    *,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
) -> tuple[int, ...]:
    """Return conservative integer frame counts for all six camera grids.

    ``ceil(duration * fps)`` is the phase-independent upper bound for a
    half-open periodic grid.  It is deliberately conservative: no interval
    can be admitted when its budget decision depends on floating-point phase.
    """

    projection = sampling_plan_projection(sampling_plan, purpose=purpose)
    duration_ns = interval.duration_ns
    return tuple(
        _ceil_div(
            duration_ns * projection["per_camera"][camera_id]["numerator"],
            NANOSECONDS_PER_SECOND * projection["per_camera"][camera_id]["denominator"],
        )
        for camera_id in CAMERA_ID_VALUES
    )


def _fits_duration(duration_ns: int, spec: _BudgetSpec) -> bool:
    counts = tuple(
        _ceil_div(duration_ns * numerator, NANOSECONDS_PER_SECOND * denominator)
        for _, numerator, denominator in spec.rates
    )
    return (
        all(count <= spec.max_frames_per_camera for count in counts)
        and sum(counts) <= spec.max_frames_total
    )


def _maximum_fitting_duration(duration_ns: int, spec: _BudgetSpec) -> int:
    """Find the largest integer span admitted by both budget constraints."""

    low = 0
    high = duration_ns
    while low < high:
        middle = (low + high + 1) // 2
        if _fits_duration(middle, spec):
            low = middle
        else:
            high = middle - 1
    return low


def plan_interval_parts(
    requested_interval: NanosecondInterval,
    effective_interval: NanosecondInterval,
    sampling_plan: SamplingPlan,
    *,
    overlap_ns: int | None = None,
    split_policy: DenseSplitPolicy | None = None,
    purpose: SamplingPurpose = SamplingPurpose.ACTION_DENSE,
) -> tuple[IntervalPart, ...]:
    """Split one half-open interval under the source frame budget.

    The returned parts are ordered by ordinal and cover the effective interval
    without gaps.  Adjacent parts carry equal overlap values, and every step
    advances strictly even when overlap is nonzero.
    """

    if effective_interval.start_ns < requested_interval.start_ns:
        raise ValueError("effective interval must be contained by requested_interval")
    if effective_interval.end_ns > requested_interval.end_ns:
        raise ValueError("effective interval must be contained by requested_interval")
    resolved_overlap = (
        split_policy.overlap_ns if overlap_ns is None and split_policy is not None else overlap_ns
    )
    if resolved_overlap is None:
        resolved_overlap = 0
    if (
        isinstance(resolved_overlap, bool)
        or not isinstance(resolved_overlap, int)
        or resolved_overlap < 0
    ):
        raise ValueError("overlap_ns must be a nonnegative integer")
    policy = split_policy or DenseSplitPolicy(
        version=sampling_plan.version,
        overlap_ns=resolved_overlap,
    )
    if policy.overlap_ns != resolved_overlap:
        raise ValueError("overlap_ns disagrees with split policy")
    spec = _budget_spec(sampling_plan, policy, purpose=purpose)
    full_duration = effective_interval.duration_ns
    if _fits_duration(full_duration, spec):
        return (
            IntervalPart(
                requested_interval=requested_interval,
                effective_interval=effective_interval,
                ordinal=0,
                part_count=1,
                overlap_before_ns=0,
                overlap_after_ns=0,
            ),
        )

    if spec.overflow_policy is not OverflowPolicy.SPLIT_WINDOW:
        raise ValueError(
            "frame budget exceeded and overflow policy does not permit provider-neutral splitting"
        )

    maximum_span = _maximum_fitting_duration(full_duration, spec)
    if maximum_span <= 0:
        raise ValueError("frame budget cannot admit even a one-nanosecond dense window")
    if spec.overlap_ns >= maximum_span:
        raise ValueError("overlap_ns must be strictly less than the split span")
    stride = maximum_span - spec.overlap_ns
    estimated_parts = (full_duration - maximum_span + stride - 1) // stride + 1
    if estimated_parts > MAX_SPLIT_PARTS:
        raise ValueError("dense split would exceed the maximum supported part count")

    coordinates: list[tuple[int, int]] = []
    current_start = effective_interval.start_ns
    while current_start < effective_interval.end_ns:
        current_end = min(current_start + maximum_span, effective_interval.end_ns)
        span = current_end - current_start
        if span <= 0 or spec.overlap_ns >= span:
            raise ValueError("overlap_ns must be strictly less than every split span")
        coordinates.append((current_start, current_end))
        if current_end == effective_interval.end_ns:
            break
        next_start = current_end - spec.overlap_ns
        if next_start <= current_start:
            raise ValueError("dense split did not make strict progress")
        current_start = next_start

    part_count = len(coordinates)
    parts: list[IntervalPart] = []
    for ordinal, (start_ns, end_ns) in enumerate(coordinates):
        overlap_before = 0
        overlap_after = 0
        if ordinal > 0:
            _, previous_end = coordinates[ordinal - 1]
            overlap_before = previous_end - start_ns
        if ordinal < part_count - 1:
            next_start, _ = coordinates[ordinal + 1]
            overlap_after = end_ns - next_start
        effective = NanosecondInterval(start_ns=start_ns, end_ns=end_ns)
        # Requested child bounds retain any pre/post padding at the outer
        # edges while middle parts use their own effective coordinates.
        requested_start = requested_interval.start_ns if ordinal == 0 else start_ns
        requested_end = requested_interval.end_ns if ordinal == part_count - 1 else end_ns
        requested = NanosecondInterval(start_ns=requested_start, end_ns=requested_end)
        parts.append(
            IntervalPart(
                requested_interval=requested,
                effective_interval=effective,
                ordinal=ordinal,
                part_count=part_count,
                overlap_before_ns=overlap_before,
                overlap_after_ns=overlap_after,
            )
        )
    return tuple(parts)


class DenseSamplingPlanner:
    """Plan deterministic padded and budgeted dense windows."""

    def __init__(
        self,
        policy: DenseSplitPolicy,
        sampling_plan: SamplingPlan | None = None,
    ) -> None:
        self._policy = policy
        self._sampling_plan = sampling_plan

    @property
    def policy(self) -> DenseSplitPolicy:
        """The immutable overlap policy governing this planner."""

        return self._policy

    @property
    def sampling_plan(self) -> SamplingPlan | None:
        """The optional plan bound at construction time."""

        return self._sampling_plan

    def plan_dense_windows(
        self,
        candidate_intervals: Sequence[NanosecondInterval],
        padding_ns: int,
        recording_duration_ns: int,
        sampling_plan: SamplingPlan | None = None,
    ) -> tuple[TemporalWindow, ...]:
        """Produce padded, clipped, and budget-split dense windows.

        Candidate order is not semantic.  Coordinates are sorted before IDs
        are derived; equal candidates retain deterministic occurrence ordinals
        so permuting input cannot change output identities.
        """

        plan = sampling_plan or self._sampling_plan
        if plan is None:
            raise ValueError("a contracts.sampling_plan.SamplingPlan is required")
        if isinstance(padding_ns, bool) or not isinstance(padding_ns, int) or padding_ns < 0:
            raise ValueError("padding_ns must be a nonnegative integer")
        if (
            isinstance(recording_duration_ns, bool)
            or not isinstance(recording_duration_ns, int)
            or recording_duration_ns <= 0
        ):
            raise ValueError("recording_duration_ns must be a positive integer")
        # Validate plan/policy before consuming candidates, including the
        # six-camera total budget invariant.
        _budget_spec(plan, self._policy)

        normalized: list[tuple[int, int, int, int]] = []
        for interval in candidate_intervals:
            if not isinstance(interval, NanosecondInterval):
                raise TypeError("candidate_intervals must contain NanosecondInterval values")
            requested_start = interval.start_ns - padding_ns
            requested_end = interval.end_ns + padding_ns
            if requested_start >= requested_end:
                continue
            if requested_start < -(2**63) or requested_end > 2**63 - 1:
                raise ValueError("padded interval must fit in signed 64-bit nanoseconds")
            effective_start = max(0, requested_start)
            effective_end = min(recording_duration_ns, requested_end)
            if effective_start >= effective_end:
                continue
            normalized.append((requested_start, requested_end, effective_start, effective_end))

        windows: list[TemporalWindow] = []
        for candidate_ordinal, (
            requested_start,
            requested_end,
            effective_start,
            effective_end,
        ) in enumerate(sorted(normalized)):
            requested = NanosecondInterval(start_ns=requested_start, end_ns=requested_end)
            effective = NanosecondInterval(start_ns=effective_start, end_ns=effective_end)
            parts = plan_interval_parts(
                requested,
                effective,
                plan,
                overlap_ns=self._policy.overlap_ns,
                split_policy=self._policy,
            )
            projection = {
                "requested": [str(requested.start_ns), str(requested.end_ns)],
                "effective": [str(effective.start_ns), str(effective.end_ns)],
                "candidate_ordinal": candidate_ordinal,
                "policy": self._policy.version,
                "plan": sampling_plan_projection(plan),
                "parts": [
                    {
                        "ordinal": part.ordinal,
                        "part_count": part.part_count,
                        "start": str(part.effective_interval.start_ns),
                        "end": str(part.effective_interval.end_ns),
                        "overlap_before": str(part.overlap_before_ns),
                        "overlap_after": str(part.overlap_after_ns),
                    }
                    for part in parts
                ],
            }
            plan_digest = sha256(canonical_json_bytes(projection)).hexdigest()
            for part in parts:
                coordinate = {
                    "dense_plan": plan_digest,
                    "candidate_ordinal": candidate_ordinal,
                    "ordinal": part.ordinal,
                    "part_count": part.part_count,
                    "requested_start": str(part.requested_interval.start_ns),
                    "requested_end": str(part.requested_interval.end_ns),
                    "start": str(part.effective_interval.start_ns),
                    "end": str(part.effective_interval.end_ns),
                    "overlap_before": str(part.overlap_before_ns),
                    "overlap_after": str(part.overlap_after_ns),
                }
                window_id = f"dw-{sha256(canonical_json_bytes(coordinate)).hexdigest()[:32]}"
                windows.append(
                    TemporalWindow(
                        window_id=window_id,
                        requested_interval=part.requested_interval,
                        interval=part.effective_interval,
                        ordinal=part.ordinal,
                        part_count=part.part_count,
                        overlap_before_ns=part.overlap_before_ns,
                        overlap_after_ns=part.overlap_after_ns,
                    )
                )
        return tuple(
            sorted(
                windows,
                key=lambda window: (
                    window.requested_interval.start_ns,
                    window.requested_interval.end_ns,
                    window.interval.start_ns,
                    window.interval.end_ns,
                    window.ordinal,
                    window.window_id,
                ),
            )
        )

    def _split_window(
        self,
        interval: NanosecondInterval,
        base_ordinal: int = 0,
        sampling_plan: SamplingPlan | None = None,
    ) -> tuple[TemporalWindow, ...]:
        """Compatibility helper for callers that split an already clipped interval."""

        plan = sampling_plan or self._sampling_plan
        if plan is None:
            raise ValueError("a contracts.sampling_plan.SamplingPlan is required")
        parts = plan_interval_parts(
            interval,
            interval,
            plan,
            overlap_ns=self._policy.overlap_ns,
            split_policy=self._policy,
        )
        result: list[TemporalWindow] = []
        for part in parts:
            coordinate = {
                "policy": self._policy.version,
                "plan": sampling_plan_projection(plan),
                "base_ordinal": base_ordinal,
                "ordinal": part.ordinal,
                "part_count": part.part_count,
                "start": str(part.effective_interval.start_ns),
                "end": str(part.effective_interval.end_ns),
                "overlap_before": str(part.overlap_before_ns),
                "overlap_after": str(part.overlap_after_ns),
            }
            result.append(
                TemporalWindow(
                    window_id=f"dw-{sha256(canonical_json_bytes(coordinate)).hexdigest()[:32]}",
                    requested_interval=part.requested_interval,
                    interval=part.effective_interval,
                    ordinal=part.ordinal,
                    part_count=part.part_count,
                    overlap_before_ns=part.overlap_before_ns,
                    overlap_after_ns=part.overlap_after_ns,
                )
            )
        return tuple(result)


__all__ = [
    "CAMERA_COUNT",
    "DenseSamplingPlanner",
    "DenseSplitPolicy",
    "IntervalPart",
    "TemporalWindow",
    "frame_counts_for_interval",
    "plan_interval_parts",
    "sampling_plan_projection",
]
